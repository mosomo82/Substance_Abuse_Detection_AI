"""
embedding_classifier.py
=======================
Layer 2 (of 3) in the detection pipeline — embedding-based risk scoring.

Architecture:
    processed_text
        ├── Encode with sentence-transformers (all-MiniLM-L6-v2)
        ├── Compare against seed bank vectors  →  per-class similarity scores
        ├── Compare against topic centroids    →  topic / cluster assignment
        └── Combine                            →  risk_level + confidence + evidence

Role in the full pipeline:
    - Catches indirect language, metaphor, and heavy slang that rule-based misses
    - Provides clustering for Task 2 (behavioral theme discovery)
    - Provides retrieve_similar_posts() for Task 3 RAG explainability layer

Inputs:
    data/processed/posts_preprocessed.csv    (from preprocess_posts.py)
    data/raw/seed_bank.pkl                   (from process_drug_reviews.py)

Outputs:
    data/processed/embedding_results.csv
    data/processed/clustered_posts.csv
    data/processed/cluster_info.json

Run:
    python src/embedding_classifier.py
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent          # project root
DATA      = ROOT / "data"
RAW       = DATA / "raw"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

IN_CSV       = PROCESSED / "posts_preprocessed.csv"
SEED_PKL     = RAW / "seed_bank.pkl"
OUT_CSV      = PROCESSED / "embedding_results.csv"
CLUSTER_CSV  = PROCESSED / "clustered_posts.csv"
CLUSTER_JSON = PROCESSED / "cluster_info.json"

# ── Load sentence-transformers model once ──────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    _model = None


def _require_model():
    if not _ST_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is required.\n"
            "  pip install sentence-transformers"
        )
    return _model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build seed bank from labeled drug reviews
# ══════════════════════════════════════════════════════════════════════════════

def build_seed_bank(labeled_df: pd.DataFrame,
                    text_col:   str = "processed_text",
                    label_col:  str = "risk_level",
                    n_per_class: int = 100) -> dict:
    """
    Build an embedding seed bank from labeled examples.

    labeled_df  : preprocessed drug reviews with a risk_level column
    Returns     : {risk_level: {'texts': [...], 'embeddings': np.array, 'centroid': np.array}}
    """
    model = _require_model()
    seed_bank = {}

    for risk_level in ["high", "medium", "low"]:
        subset = labeled_df[labeled_df[label_col] == risk_level].copy()

        # Weight by usefulCount where available — upvoted posts are more representative
        if "usefulCount" in subset.columns:
            subset = subset.sort_values("usefulCount", ascending=False)

        samples = subset.head(n_per_class)[text_col].tolist()
        if not samples:
            continue

        print(f"  Encoding {len(samples)} {risk_level}-risk seeds …")
        embeddings = model.encode(
            samples,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
        )

        seed_bank[risk_level] = {
            "texts":      samples,
            "embeddings": embeddings,
            "centroid":   embeddings.mean(axis=0),
        }

    return seed_bank


def load_or_build_seed_bank(labeled_df: pd.DataFrame | None = None) -> dict:
    """
    Load existing seed bank from disk; build from labeled_df if missing.
    The seed bank from process_drug_reviews.py uses substance-level keys —
    this function converts to risk-level keys if needed.
    """
    if SEED_PKL.exists() and labeled_df is None:
        with open(SEED_PKL, "rb") as f:
            raw = pickle.load(f)

        # If pkl is substance-keyed ({opioid: [...]}), we can't use it directly
        # for risk-level scoring — need labeled_df to build a proper one.
        if set(raw.keys()) <= {"high", "medium", "low"}:
            print(f"  Loaded seed bank from {SEED_PKL}")
            return raw
        else:
            print("  ⚠  Existing seed_bank.pkl is substance-keyed.")
            print("     Pass labeled_df to build a risk-level seed bank.")
            return {}

    if labeled_df is not None:
        print("  Building seed bank from labeled data …")
        return build_seed_bank(labeled_df)

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Single post scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_post_embedding(post_text: str,
                         seed_bank: dict,
                         top_k: int = 10) -> dict:
    """
    Score one post against all risk-level seed banks.
    Returns risk_level, soft probabilities, confidence margin, and evidence.
    """
    model   = _require_model()
    post_emb = model.encode([post_text], normalize_embeddings=True)

    scores    : dict[str, float]     = {}
    top_texts : dict[str, list[str]] = {}

    for risk_level, bank in seed_bank.items():
        sims         = cosine_similarity(post_emb, bank["embeddings"])[0]
        top_k_idx    = np.argsort(sims)[-top_k:]
        scores[risk_level]    = float(sims[top_k_idx].mean())
        top_texts[risk_level] = [bank["texts"][i] for i in top_k_idx[-3:]]

    total     = sum(scores.values()) or 1.0
    probs     = {k: round(v / total, 3) for k, v in scores.items()}
    predicted = max(probs, key=probs.get)

    sorted_p   = sorted(probs.values(), reverse=True)
    confidence = round(sorted_p[0] - sorted_p[1], 3) if len(sorted_p) > 1 else 1.0

    evidence = [
        f"Similarity to {rl} seeds: {p:.3f}"
        for rl, p in sorted(probs.items(), key=lambda x: -x[1])
    ]
    evidence.append(f"Confidence margin: {confidence:.3f}")

    return {
        "risk_level":        predicted,
        "probabilities":     probs,
        "confidence":        confidence,
        "evidence":          evidence,
        "top_similar_high":  top_texts.get("high", []),
        "method":            "embedding",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Topic clustering (Task 2)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_posts(processed_df: pd.DataFrame,
                  text_col:     str = "processed_text",
                  n_clusters:   int = 8,
                  random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster posts by semantic similarity using K-Means on embeddings.
    Optionally reduces to 2D with UMAP for visualization.
    Returns (clustered_df, embeddings).
    """
    model = _require_model()
    from sklearn.cluster import KMeans

    texts = processed_df[text_col].tolist()
    print(f"  Encoding {len(texts)} posts for clustering …")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"  K-Means clustering (k={n_clusters}) …")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    result_df = processed_df.copy()
    result_df["cluster"] = labels

    # Optional UMAP — skip gracefully if not installed
    try:
        import umap
        print("  UMAP dimensionality reduction …")
        reducer = umap.UMAP(n_components=2, random_state=random_state,
                            metric="cosine")
        coords = reducer.fit_transform(embeddings)
        result_df["umap_x"] = coords[:, 0]
        result_df["umap_y"] = coords[:, 1]
    except ImportError:
        print("  ⚠  umap-learn not installed — 2D coords skipped.")
        print("     pip install umap-learn")

    return result_df, embeddings, kmeans


def label_clusters(clustered_df: pd.DataFrame,
                   embeddings:   np.ndarray,
                   text_col:     str = "processed_text",
                   n_examples:   int = 5) -> dict:
    """
    Auto-label clusters by finding posts closest to each centroid.
    Returns {cluster_id: {size, substance, risk_level, examples, label}}.
    """
    cluster_info = {}

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        mask      = (clustered_df["cluster"] == cluster_id).values
        c_embs    = embeddings[mask]
        c_texts   = clustered_df[mask][text_col].tolist()
        c_rows    = clustered_df[mask]

        centroid  = c_embs.mean(axis=0, keepdims=True)
        sims      = cosine_similarity(centroid, c_embs)[0]
        top_idx   = np.argsort(sims)[-n_examples:][::-1]
        examples  = [c_texts[i] for i in top_idx]

        # Dominant substance
        sub_col = "substances_detected"
        if sub_col in c_rows.columns:
            all_subs = []
            for val in c_rows[sub_col]:
                try:
                    subs = json.loads(val) if isinstance(val, str) else val
                    all_subs.extend(subs if isinstance(subs, list) else [])
                except Exception:
                    pass
            top_sub = Counter(all_subs).most_common(1)
            substance = top_sub[0][0] if top_sub else "mixed"
        else:
            substance = "unknown"

        # Dominant risk level
        top_risk = (
            c_rows["risk_level"].mode()[0]
            if "risk_level" in c_rows.columns
            else "unknown"
        )

        cluster_info[int(cluster_id)] = {
            "size":       int(mask.sum()),
            "substance":  substance,
            "risk_level": top_risk,
            "examples":   examples,
            "label":      f"Cluster {cluster_id}: {substance} / {top_risk} risk",
        }

    return cluster_info


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Batch classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_corpus_embedding(processed_df: pd.DataFrame,
                               seed_bank:    dict,
                               text_col:     str = "processed_text",
                               batch_size:   int = 64,
                               top_k:        int = 10) -> pd.DataFrame:
    """
    Classify all posts using embedding similarity.
    Batch-encodes for efficiency, then vectorised cosine similarity.
    """
    model = _require_model()
    texts = processed_df[text_col].tolist()

    print(f"  Encoding {len(texts)} posts …")
    all_embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Pre-stack seed matrices
    seed_matrices = {
        rl: bank["embeddings"] for rl, bank in seed_bank.items()
    }

    results = []
    for i, post_emb in enumerate(all_embs):
        e2d = post_emb.reshape(1, -1)

        scores = {
            rl: float(np.sort(cosine_similarity(e2d, mat)[0])[-top_k:].mean())
            for rl, mat in seed_matrices.items()
        }

        total     = sum(scores.values()) or 1.0
        probs     = {k: round(v / total, 3) for k, v in scores.items()}
        predicted = max(probs, key=probs.get)
        sorted_p  = sorted(probs.values(), reverse=True)
        confidence = round(sorted_p[0] - sorted_p[1], 3) if len(sorted_p) > 1 else 1.0

        row = processed_df.iloc[i]
        results.append({
            "post_id":      row.get("post_id"),
            "timestamp":    row.get("timestamp"),
            "risk_level":   predicted,
            "confidence":   confidence,
            "prob_high":    probs.get("high",   0),
            "prob_medium":  probs.get("medium", 0),
            "prob_low":     probs.get("low",    0),
            "needs_review": confidence < 0.15,
            "method":       "embedding",
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compare against rule-based results (Task 1)
# ══════════════════════════════════════════════════════════════════════════════

def compare_methods(rule_df:         pd.DataFrame,
                    embedding_df:    pd.DataFrame,
                    ground_truth_col: str = "true_label") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Side-by-side precision/recall/F1 comparison of rule-based vs embedding.
    Prints classification reports and disagreement breakdown.
    """
    from sklearn.metrics import classification_report

    merged = rule_df[["post_id", "risk_level"]].merge(
        embedding_df[["post_id", "risk_level"]],
        on="post_id",
        suffixes=("_rule", "_embedding"),
    )

    has_truth = ground_truth_col in rule_df.columns
    if has_truth:
        merged = merged.merge(
            rule_df[["post_id", ground_truth_col]], on="post_id"
        )
        for name, col in [("Rule-based", "risk_level_rule"),
                          ("Embedding-based", "risk_level_embedding")]:
            print(f"\n=== {name} ===")
            print(classification_report(
                merged[ground_truth_col], merged[col],
                labels=["high", "medium", "low"],
            ))
    else:
        print("No ground truth column found — showing distributions only.")
        print("\nRule-based:   ", rule_df["risk_level"].value_counts().to_dict())
        print("Embedding:    ", embedding_df["risk_level"].value_counts().to_dict())

    agreement   = (merged["risk_level_rule"] == merged["risk_level_embedding"]).mean()
    disagreements = merged[merged["risk_level_rule"] != merged["risk_level_embedding"]]

    print(f"\nMethod agreement rate : {agreement:.1%}")
    print(f"Disagreements         : {len(disagreements):,} posts")
    if not disagreements.empty:
        print("\nDisagreement breakdown (rule → embedding):")
        print(
            disagreements.groupby(["risk_level_rule", "risk_level_embedding"])
            .size().sort_values(ascending=False).to_string()
        )

    return merged, disagreements


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Semantic search for RAG (Task 3)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar_posts(query_text:       str,
                            corpus_df:        pd.DataFrame,
                            corpus_embeddings: np.ndarray,
                            top_k:            int = 5,
                            risk_filter:      str | None = "high") -> list[dict]:
    """
    Retrieve top-k most semantically similar posts to a query.
    Used by the RAG explainability layer to assemble analyst summaries.
    """
    model    = _require_model()
    query_emb = model.encode([query_text], normalize_embeddings=True)

    if risk_filter and "risk_level" in corpus_df.columns:
        mask      = (corpus_df["risk_level"] == risk_filter).values
        filt_df   = corpus_df[mask].reset_index(drop=True)
        filt_embs = corpus_embeddings[mask]
    else:
        filt_df   = corpus_df.reset_index(drop=True)
        filt_embs = corpus_embeddings

    sims    = cosine_similarity(query_emb, filt_embs)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]

    return [
        {
            "text":       filt_df.iloc[i].get("processed_text", ""),
            "similarity": round(float(sims[i]), 3),
            "risk_level": filt_df.iloc[i].get("risk_level", "unknown"),
            "substance":  filt_df.iloc[i].get("substances_detected", "[]"),
        }
        for i in top_idx
    ]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not _ST_AVAILABLE:
        print("⚠  sentence-transformers not installed.")
        print("   pip install sentence-transformers")
        return

    if not IN_CSV.exists():
        print(f"⚠  Input not found: {IN_CSV}")
        print("   Run data/preprocess_posts.py first.")
        return

    print(f"Loading preprocessed posts from {IN_CSV} …")
    df = pd.read_csv(IN_CSV)
    print(f"  {len(df):,} rows loaded\n")

    # ── Seed bank ─────────────────────────────────────────────────────────────
    print("Step 1 — Loading / building seed bank …")
    # Try to use the posts themselves as labeled data if risk_level present
    labeled_df = df if "risk_level" in df.columns else None
    seed_bank  = load_or_build_seed_bank(labeled_df)

    if not seed_bank:
        print("⚠  No seed bank available. "
              "Run data/process_drug_reviews.py to generate one.")
        return

    # ── Batch classification ──────────────────────────────────────────────────
    print("\nStep 2–3 — Classifying posts …")
    results = classify_corpus_embedding(df, seed_bank)

    print("\nRisk level distribution (embedding):")
    print(results["risk_level"].value_counts().to_string())

    print(f"\nLow-confidence posts (needs_review=True): "
          f"{results['needs_review'].sum():,} "
          f"({results['needs_review'].mean():.1%})")

    results.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")

    # ── Compare against rule-based if available ───────────────────────────────
    rule_csv = PROCESSED / "rule_based_results.csv"
    if rule_csv.exists():
        print("\nStep 5 — Method comparison …")
        rule_df = pd.read_csv(rule_csv)
        if "post_id" in rule_df.columns and "post_id" in results.columns:
            compare_methods(rule_df, results)
        else:
            print("  ⚠  post_id column missing — skipping comparison.")

    # ── Topic clustering ──────────────────────────────────────────────────────
    print("\nStep 6 — Clustering posts into topics …")
    clustered_df, embeddings, _ = cluster_posts(df, n_clusters=8)

    cluster_info = label_clusters(clustered_df, embeddings)
    print("\nCluster summary:")
    for cid, info in cluster_info.items():
        print(f"  [{cid}] {info['label']}  (n={info['size']})")

    clustered_df.to_csv(CLUSTER_CSV, index=False)
    with open(CLUSTER_JSON, "w") as f:
        # examples are lists of strings — JSON-serialisable
        json.dump(cluster_info, f, indent=2)
    print(f"Saved → {CLUSTER_CSV}")
    print(f"Saved → {CLUSTER_JSON}")

    # ── RAG retrieval demo ────────────────────────────────────────────────────
    print("\nStep 7 — RAG retrieval demo …")
    sample_query = "i cant stop taking them even though i know its bad"
    similar = retrieve_similar_posts(
        sample_query, df, embeddings, top_k=3, risk_filter=None
    )
    print(f"  Query: '{sample_query}'")
    print("  Top similar posts:")
    for s in similar:
        print(f"    [{s['similarity']:.3f}] {s['text'][:80]} …")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
