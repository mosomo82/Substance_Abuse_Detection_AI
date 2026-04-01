"""
finetuned_classifier.py
=======================
Fine-tunes distilbert-base-uncased on pseudo-labeled posts from the ensemble
output, then runs full-dataset inference to produce a 4th classifier column
for ensemble fusion.

Pseudo-ground-truth:  ensemble_results.csv  →  final_risk_level  (low/med/high)
Text input:           posts_preprocessed.csv →  processed_text

Label map:
    low    → 0
    medium → 1
    high   → 2

Inputs:
    data/processed/ensemble_results.csv      (required)
    data/processed/posts_preprocessed.csv    (required)

Outputs:
    models/finetuned_bert/                   (HuggingFace model checkpoint)
    data/processed/finetuned_results.csv
        Columns: post_id, risk_level, confidence

Run locally (GPU/CPU):
    python src/classifiers/finetuned_classifier.py

Run on Colab TPU:
    See notebooks/colab_finetuned_classifier.ipynb
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional heavy imports (loaded lazily so the file can be imported safely) ──
try:
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "finetuned_bert"

ENSEMBLE_CSV    = PROCESSED / "ensemble_results.csv"
PREPROCESS_CSV  = PROCESSED / "posts_preprocessed.csv"
OUT_CSV         = PROCESSED / "finetuned_results.csv"

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 256          # token limit per post
TRAIN_SPLIT  = 0.80         # 80 % train, 20 % validation
EPOCHS       = 3
BATCH_SIZE   = 16           # per device; TPU multiplies by num_cores
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
SEED         = 42

LABEL2ID = {"low": 0, "medium": 1, "high": 2}
ID2LABEL = {0: "low", 1: "medium", 2: "high"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """
    Join ensemble labels with post text.

    post_id may be NaN in posts_preprocessed (synthetic integer index is used
    in ensemble_results when the source CSV lacked a stable key).  We align
    by row position in that case.
    """
    ens = pd.read_csv(ENSEMBLE_CSV)
    pre = pd.read_csv(PREPROCESS_CSV)

    # Determine text column
    text_col = next(
        (c for c in ("processed_text", "cleaned_text", "original_text")
         if c in pre.columns),
        None,
    )
    if text_col is None:
        raise ValueError(
            f"Cannot find a text column in {PREPROCESS_CSV.name}. "
            f"Available columns: {list(pre.columns)}"
        )

    # Align on post_id when both have valid integer keys; else row-position
    ens_ids_valid = ens["post_id"].notna().all() and pd.api.types.is_integer_dtype(ens["post_id"])
    pre_ids_valid = pre["post_id"].notna().all() and pd.api.types.is_integer_dtype(pre["post_id"])

    if ens_ids_valid and pre_ids_valid:
        merged = ens[["post_id", "final_risk_level"]].merge(
            pre[["post_id", text_col]], on="post_id", how="inner"
        )
    else:
        # Row-position alignment (handles NaN post_id in preprocessed CSV)
        n = min(len(ens), len(pre))
        merged = pd.DataFrame({
            "post_id":          range(n),
            "final_risk_level": ens["final_risk_level"].values[:n],
            text_col:           pre[text_col].values[:n],
        })

    merged = merged.rename(columns={text_col: "text"})
    merged = merged.dropna(subset=["text", "final_risk_level"])
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.len() > 10]   # drop near-empty posts

    # Keep only valid labels
    valid_labels = set(LABEL2ID.keys())
    merged = merged[merged["final_risk_level"].isin(valid_labels)].reset_index(drop=True)
    merged["label"] = merged["final_risk_level"].map(LABEL2ID)

    print(f"  Loaded {len(merged):,} labelled posts (text col: '{text_col}')")
    print(f"  Label distribution:\n{merged['final_risk_level'].value_counts().to_string()}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Tokenisation
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Convert DataFrame → HuggingFace Dataset with tokenised inputs."""
    hf_ds = Dataset.from_pandas(df[["post_id", "text", "label"]])

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,      # DataCollatorWithPadding handles dynamic padding
        )

    return hf_ds.map(_tokenize, batched=True, remove_columns=["text"])


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1_macro":  f1_score(labels, preds, average="macro",  zero_division=0),
        "f1_high":   f1_score(labels, preds, labels=[2], average="micro", zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(df: pd.DataFrame) -> tuple:
    """Fine-tune DistilBERT; return (trainer, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Train / validation split (stratified)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=1 - TRAIN_SPLIT,
        stratify=df["label"], random_state=SEED
    )
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds   = tokenize_dataset(val_df,   tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Detect compute environment
    use_fp16 = torch.cuda.is_available()
    use_bf16 = (
        not use_fp16
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=SEED,
        report_to="none",        # disable wandb / tensorboard by default
        logging_steps=50,
        dataloader_num_workers=0,  # safe default for Windows + Colab
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting fine-tuning …")
    trainer.train()

    # Persist model + tokenizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Save label map alongside model
    with open(MODEL_DIR / "label_map.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    print(f"\nModel saved → {MODEL_DIR}")

    # Final eval metrics
    metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return trainer, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(df: pd.DataFrame, model_dir: str | Path = MODEL_DIR) -> pd.DataFrame:
    """
    Run inference on the full post dataset using the saved fine-tuned model.
    Returns DataFrame with columns: post_id, risk_level, confidence.
    """
    from transformers import pipeline as hf_pipeline

    model_dir = Path(model_dir)
    device = 0 if torch.cuda.is_available() else -1   # 0 = GPU:0, -1 = CPU

    clf = hf_pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=64,
    )

    texts = df["text"].fillna("").astype(str).tolist()

    print(f"\nRunning inference on {len(texts):,} posts …")
    raw = clf(texts)

    label_map = {"LABEL_0": "low", "LABEL_1": "medium", "LABEL_2": "high"}

    results = pd.DataFrame({
        "post_id":    df["post_id"].values,
        "risk_level": [label_map.get(r["label"], r["label"].lower()) for r in raw],
        "confidence": [round(r["score"], 4) for r in raw],
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(results):,} predictions → {OUT_CSV}")

    print("\nFine-tuned risk distribution:")
    print(results["risk_level"].value_counts().to_string())

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not _DEPS_OK:
        raise ImportError(
            "Missing dependencies. Install with:\n"
            "  pip install transformers datasets accelerate scikit-learn torch"
        )

    print("=" * 60)
    print("Fine-Tuned DistilBERT Classifier")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading data …")
    df = load_data()

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training …")
    trainer, tokenizer = train(df)

    # ── 3. Full-dataset inference ─────────────────────────────────────────────
    print("\n[3/3] Running full-dataset inference …")
    run_inference(df, model_dir=MODEL_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
