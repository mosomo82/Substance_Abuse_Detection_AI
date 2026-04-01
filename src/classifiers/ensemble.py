"""
ensemble.py
===========
Ensemble fusion of rule-based, embedding, and LLM classifier outputs.

Combines the three parallel detection methods into a single final risk verdict
using weighted voting. LLM predictions are optional — most posts skip the LLM
layer due to hybrid routing — and weights are re-normalized automatically when
a source is absent.

Inputs:
    data/processed/rule_based_results.csv     (required)
    data/processed/embedding_results.csv      (required)
    data/processed/llm_results.csv            (optional)

Output:
    data/processed/ensemble_results.csv
    Columns: post_id, final_risk_level, ensemble_confidence,
             method_agreement, sources_used,
             rule_risk_level, embedding_risk_level, llm_risk_level

Run:
    python src/ensemble.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent   # project root
PROCESSED = ROOT / "data" / "processed"

RULE_CSV      = PROCESSED / "rule_based_results.csv"
EMBEDDING_CSV = PROCESSED / "embedding_results.csv"
LLM_CSV        = PROCESSED / "llm_results.csv"
HYBRID_CSV     = PROCESSED / "hybrid_results.csv"   # fallback if llm_results absent
FINETUNED_CSV  = PROCESSED / "finetuned_results.csv"  # optional 4th method
OUT_CSV        = PROCESSED / "ensemble_results.csv"

# ── Risk encoding ──────────────────────────────────────────────────────────────
RISK_TO_INT = {"low": 0, "medium": 1, "high": 2}
INT_TO_RISK = {0: "low", 1: "medium", 2: "high"}

# Thresholds for decoding weighted average back to 3-class labels
#   [0.00, 0.50)  → low
#   [0.50, 1.50)  → medium
#   [1.50, 2.00]  → high
LOW_UPPER    = 0.50
MEDIUM_UPPER = 1.50

# Weights when all three classifiers are present (no fine-tuned)
WEIGHTS_FULL   = {"rule": 0.25, "embedding": 0.35, "llm": 0.40}
# Weights when LLM is absent (most posts skip LLM due to hybrid routing)
WEIGHTS_NO_LLM = {"rule": 0.35, "embedding": 0.65}

# Weights when fine-tuned model is also available
WEIGHTS_FULL_FT   = {"rule": 0.20, "embedding": 0.30, "llm": 0.40, "finetuned": 0.10}
WEIGHTS_NO_LLM_FT = {"rule": 0.30, "embedding": 0.60, "finetuned": 0.10}


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_classifier_outputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None
]:
    """
    Load rule-based, embedding, (optionally) LLM, and (optionally) fine-tuned results.
    Returns (rule_df, emb_df, llm_df_or_None, finetuned_df_or_None).
    """
    if not RULE_CSV.exists():
        raise FileNotFoundError(
            f"Rule-based results not found: {RULE_CSV}\n"
            "  Run: python src/rule_based_classifier.py"
        )
    if not EMBEDDING_CSV.exists():
        raise FileNotFoundError(
            f"Embedding results not found: {EMBEDDING_CSV}\n"
            "  Run: python src/embedding_classifier.py"
        )

    rule_df = pd.read_csv(RULE_CSV)
    emb_df  = pd.read_csv(EMBEDDING_CSV)

    # Prefer llm_results.csv; fall back to hybrid_results.csv
    llm_df = None
    for path in (LLM_CSV, HYBRID_CSV):
        if path.exists():
            llm_df = pd.read_csv(path)
            print(f"  LLM results loaded from {path.name} ({len(llm_df):,} rows)")
            break
    if llm_df is None:
        print("  LLM results not found — ensemble will use rule + embedding only.")

    # Optional 4th method: fine-tuned DistilBERT
    finetuned_df = None
    if FINETUNED_CSV.exists():
        finetuned_df = pd.read_csv(FINETUNED_CSV)
        print(f"  Fine-tuned results loaded ({len(finetuned_df):,} rows)")
    else:
        print("  Fine-tuned results not found — running without 4th method.")

    return rule_df, emb_df, llm_df, finetuned_df


# ══════════════════════════════════════════════════════════════════════════════
# Merging
# ══════════════════════════════════════════════════════════════════════════════

def _rename_cols(df: pd.DataFrame, prefix: str,
                 risk_col: str = "risk_level",
                 conf_col: str | None = None) -> pd.DataFrame:
    """
    Rename risk and confidence columns to classifier-prefixed names
    so they stay distinct after the three-way join.
    """
    df = df.copy()
    df = df.rename(columns={risk_col: f"{prefix}_risk_level"})
    if conf_col and conf_col in df.columns:
        df = df.rename(columns={conf_col: f"{prefix}_confidence"})
    return df


def merge_classifiers(rule_df:       pd.DataFrame,
                      emb_df:        pd.DataFrame,
                      llm_df:        pd.DataFrame | None = None,
                      finetuned_df:  pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Join the three classifier DataFrames on post_id.
    Rule + embedding are outer-joined (all posts should have both).
    LLM is left-joined (only ~20–40 % of posts have LLM results).

    When post_id is all-NaN (no stable key), the two frames are assumed to be
    in the same row order and are concatenated positionally instead.
    """
    # Rule-based: confidence proxy is combined_score
    conf_col_rule = "combined_score" if "combined_score" in rule_df.columns else "confidence"
    rule_prep = _rename_cols(rule_df[["post_id", "risk_level", conf_col_rule]
                                     if conf_col_rule in rule_df.columns
                                     else ["post_id", "risk_level"]],
                             prefix="rule",
                             conf_col=conf_col_rule)

    # Embedding
    conf_col_emb = "confidence" if "confidence" in emb_df.columns else None
    emb_prep = _rename_cols(emb_df[["post_id", "risk_level"]
                                    + ([conf_col_emb] if conf_col_emb else [])],
                            prefix="embedding",
                            conf_col=conf_col_emb)

    # Detect all-NaN post_id: merge would create a Cartesian product; use
    # positional concat when both frames have the same length instead.
    _rule_key_valid = rule_prep["post_id"].notna().any()
    _emb_key_valid  = emb_prep["post_id"].notna().any()

    if _rule_key_valid and _emb_key_valid:
        merged = rule_prep.merge(emb_prep, on="post_id", how="outer")
    else:
        # Positional alignment: reset index, drop duplicate post_id column
        print("  post_id is all-NaN — using row-order alignment for merge.")
        rule_reset = rule_prep.reset_index(drop=True).drop(columns=["post_id"],
                                                            errors="ignore")
        emb_reset  = emb_prep.reset_index(drop=True).drop(columns=["post_id"],
                                                           errors="ignore")
        merged = pd.concat([rule_reset, emb_reset], axis=1)
        merged.insert(0, "post_id", range(len(merged)))   # synthetic int key

    # LLM (optional)
    if llm_df is not None and not llm_df.empty and "post_id" in llm_df.columns:
        conf_col_llm = "confidence" if "confidence" in llm_df.columns else None
        llm_prep = _rename_cols(llm_df[["post_id", "risk_level"]
                                        + ([conf_col_llm] if conf_col_llm else [])],
                                prefix="llm",
                                conf_col=conf_col_llm)
        merged = merged.merge(llm_prep, on="post_id", how="left")
    else:
        merged["llm_risk_level"]  = np.nan
        merged["llm_confidence"]  = np.nan

    # Fine-tuned DistilBERT (optional 4th method)
    if finetuned_df is not None and not finetuned_df.empty:
        conf_col_ft = "confidence" if "confidence" in finetuned_df.columns else None
        risk_col_ft = "risk_level" if "risk_level" in finetuned_df.columns else "final_risk_level"
        ft_prep = _rename_cols(
            finetuned_df[["post_id", risk_col_ft]
                         + ([conf_col_ft] if conf_col_ft else [])],
            prefix="finetuned",
            risk_col=risk_col_ft,
            conf_col=conf_col_ft,
        )
        if "post_id" in ft_prep.columns and merged["post_id"].notna().any():
            merged = merged.merge(ft_prep, on="post_id", how="left")
        else:
            # Row-position alignment fallback
            n = min(len(merged), len(ft_prep))
            merged["finetuned_risk_level"] = np.nan
            merged["finetuned_confidence"] = np.nan
            merged.iloc[:n, merged.columns.get_loc("finetuned_risk_level")] = \
                ft_prep["finetuned_risk_level"].values[:n]
            if conf_col_ft:
                merged.iloc[:n, merged.columns.get_loc("finetuned_confidence")] = \
                    ft_prep["finetuned_confidence"].values[:n]
    else:
        merged["finetuned_risk_level"] = np.nan
        merged["finetuned_confidence"] = np.nan

    # Ensure all confidence columns exist with fallback
    for col in ("rule_confidence", "embedding_confidence",
                "llm_confidence", "finetuned_confidence"):
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Voting
# ══════════════════════════════════════════════════════════════════════════════

def compute_ensemble_row(row: "pd.Series") -> dict:
    """
    Compute weighted-vote ensemble for a single post.

    1. Identify which classifiers contributed valid predictions.
    2. Select weight set (WEIGHTS_FULL if LLM present, else WEIGHTS_NO_LLM).
    3. Weighted average of numeric risk scores → threshold → final label.
    4. Average confidence across contributing classifiers.
    5. method_agreement: True if all contributors gave the same label.
    """
    available: dict[str, tuple[int, float, float]] = {}  # name → (risk_int, conf, weight)

    has_llm = (
        pd.notna(row.get("llm_risk_level"))
        and row.get("llm_risk_level") in RISK_TO_INT
    )
    has_finetuned = (
        pd.notna(row.get("finetuned_risk_level"))
        and row.get("finetuned_risk_level") in RISK_TO_INT
    )

    if has_finetuned:
        weights = WEIGHTS_FULL_FT if has_llm else WEIGHTS_NO_LLM_FT
    else:
        weights = WEIGHTS_FULL if has_llm else WEIGHTS_NO_LLM

    for source, risk_col, conf_col in (
        ("rule",       "rule_risk_level",       "rule_confidence"),
        ("embedding",  "embedding_risk_level",  "embedding_confidence"),
        ("llm",        "llm_risk_level",        "llm_confidence"),
        ("finetuned",  "finetuned_risk_level",  "finetuned_confidence"),
    ):
        risk_val = row.get(risk_col)
        if pd.notna(risk_val) and risk_val in RISK_TO_INT:
            conf = float(row.get(conf_col) if pd.notna(row.get(conf_col)) else 0.5)
            available[source] = (RISK_TO_INT[risk_val], conf, weights.get(source, 0.0))

    if not available:
        return {
            "final_risk_level":    "low",
            "ensemble_confidence": 0.0,
            "method_agreement":    False,
            "sources_used":        "",
        }

    # Re-normalize weights to sum to 1.0 (handles partial availability)
    total_w        = sum(v[2] for v in available.values()) or 1.0
    weighted_score = sum(v[0] * v[2] for v in available.values()) / total_w
    avg_confidence = sum(v[1] for v in available.values()) / len(available)

    # Decode weighted score to 3-class label
    if weighted_score < LOW_UPPER:
        final_risk = "low"
    elif weighted_score < MEDIUM_UPPER:
        final_risk = "medium"
    else:
        final_risk = "high"

    risk_labels = {v[0] for v in available.values()}
    agreement   = len(risk_labels) == 1

    return {
        "final_risk_level":    final_risk,
        "ensemble_confidence": round(avg_confidence, 3),
        "method_agreement":    agreement,
        "sources_used":        "+".join(sorted(available.keys())),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble(rule_df:       pd.DataFrame,
                 emb_df:        pd.DataFrame,
                 llm_df:        pd.DataFrame | None = None,
                 finetuned_df:  pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge classifiers, compute ensemble vote, return results DataFrame."""
    merged = merge_classifiers(rule_df, emb_df, llm_df, finetuned_df)
    votes  = merged.apply(compute_ensemble_row, axis=1, result_type="expand")

    keep_cols = ["post_id"]
    for col in ("rule_risk_level", "embedding_risk_level",
                "llm_risk_level", "finetuned_risk_level"):
        if col in merged.columns:
            keep_cols.append(col)

    out = pd.concat([merged[keep_cols], votes], axis=1)
    return out


def main() -> None:
    print("Loading classifier outputs …")
    rule_df, emb_df, llm_df, finetuned_df = load_classifier_outputs()
    print(f"  Rule-based  : {len(rule_df):,} rows")
    print(f"  Embedding   : {len(emb_df):,} rows")
    if finetuned_df is not None:
        print(f"  Fine-tuned  : {len(finetuned_df):,} rows")

    print("\nRunning ensemble fusion …")
    results = run_ensemble(rule_df, emb_df, llm_df, finetuned_df)

    print("\nEnsemble risk distribution:")
    print(results["final_risk_level"].value_counts().to_string())

    agreement_rate = results["method_agreement"].mean()
    llm_rate       = results["sources_used"].str.contains("llm").mean()
    print(f"\nMethod agreement rate : {agreement_rate:.1%}")
    print(f"LLM participated in   : {llm_rate:.1%} of posts")

    print("\nSources used breakdown:")
    print(results["sources_used"].value_counts().to_string())

    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(results):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
