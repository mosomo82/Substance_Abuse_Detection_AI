"""
cluster_metrics.py
==================
Computes three cluster quality / embedding metrics for Task 1:

  1. Silhouette Score   — geometric cluster quality using 384-dim embeddings
  2. NDCG               — does cosine similarity to a high-risk centroid
                          surface high-risk posts at the top of the ranking?
  3. Perplexity         — bigram LM perplexity on the domain slang vocabulary
                          (SLANG_LEXICON from preprocess_posts.py)

Inputs:
    data/processed/post_embeddings.parquet   (post_id, embedding, text)
    data/processed/clustered_posts.csv       (post_id, cluster, ...)
    data/processed/ensemble_results.csv      (post_id, final_risk_level)

Output:
    data/processed/cluster_metrics.json

Run:
    python src/eval/cluster_metrics.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

EMBEDDINGS_PQ  = PROCESSED / "post_embeddings.parquet"
CLUSTERED_CSV  = PROCESSED / "clustered_posts.csv"
ENSEMBLE_CSV   = PROCESSED / "ensemble_results.csv"
OUT_JSON       = PROCESSED / "cluster_metrics.json"

# Silhouette: sample size (full 40k × 384 is ~2 min; 8k is ~5 s)
SILHOUETTE_SAMPLE = 8_000
SILHOUETTE_SEED   = 42

# NDCG cutoff values to report
NDCG_K_LIST = [50, 100, 500]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        embeddings  : (N, 384) float32
        cluster_ids : (N,) int
        risk_labels : (N,) str  — 'low' / 'medium' / 'high'
        texts_df    : DataFrame with 'post_id' + 'processed_text' for Perplexity
    """
    print("Loading embeddings ...")
    emb_df = pd.read_parquet(EMBEDDINGS_PQ)

    print("Loading cluster assignments ...")
    clust_df = pd.read_csv(CLUSTERED_CSV, usecols=lambda c: c in (
        "post_id", "cluster", "processed_text", "original_text"
    ))

    print("Loading ensemble labels ...")
    ens_df = pd.read_csv(ENSEMBLE_CSV, usecols=["post_id", "final_risk_level"])

    # ── Align all three on post_id ────────────────────────────────────────────
    # post_id may be NaN/string/int depending on the source file; normalise
    def _norm_id(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    emb_df["post_id"]   = _norm_id(emb_df["post_id"])
    clust_df["post_id"] = _norm_id(clust_df["post_id"])
    ens_df["post_id"]   = _norm_id(ens_df["post_id"])

    # If post_id is all-NaN in clustered_posts, align by row position
    if clust_df["post_id"].isna().all():
        n = min(len(emb_df), len(clust_df), len(ens_df))
        merged = pd.DataFrame({
            "post_id":          range(n),
            "embedding":        emb_df["embedding"].values[:n],
            "cluster":          clust_df["cluster"].values[:n],
            "final_risk_level": ens_df["final_risk_level"].values[:n],
        })
        # Add text column for perplexity
        text_col = "processed_text" if "processed_text" in clust_df.columns else "original_text"
        merged["processed_text"] = clust_df[text_col].values[:n]
    else:
        merged = (
            emb_df[["post_id", "embedding"]]
            .merge(clust_df[["post_id", "cluster"] +
                             (["processed_text"] if "processed_text" in clust_df.columns else [])],
                   on="post_id", how="inner")
            .merge(ens_df, on="post_id", how="inner")
        )

    merged = merged.dropna(subset=["cluster", "final_risk_level"]).reset_index(drop=True)
    merged["cluster"] = merged["cluster"].astype(int)

    # Drop HDBSCAN noise label (-1)
    merged = merged[merged["cluster"] >= 0].reset_index(drop=True)

    embeddings  = np.stack(merged["embedding"].values).astype(np.float32)
    cluster_ids = merged["cluster"].values
    risk_labels = merged["final_risk_level"].values

    print(f"  Posts after alignment & noise-drop: {len(merged):,}")
    print(f"  Unique clusters: {np.unique(cluster_ids).size}")
    print(f"  Risk distribution: {dict(pd.Series(risk_labels).value_counts())}")

    return embeddings, cluster_ids, risk_labels, merged


# ══════════════════════════════════════════════════════════════════════════════
# 1. Silhouette Score
# ══════════════════════════════════════════════════════════════════════════════

def compute_silhouette(embeddings: np.ndarray, cluster_ids: np.ndarray) -> dict:
    """
    Silhouette score on a stratified sample of posts.

    Score interpretation:
        +1  → clusters are dense and well-separated
         0  → overlapping clusters
        -1  → posts are assigned to wrong clusters
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    n = len(embeddings)
    if n > SILHOUETTE_SAMPLE:
        rng = np.random.default_rng(SILHOUETTE_SEED)
        # Stratified sample: preserve cluster proportions
        unique, counts = np.unique(cluster_ids, return_counts=True)
        fracs   = counts / counts.sum()
        budgets = np.maximum(1, (fracs * SILHOUETTE_SAMPLE).astype(int))
        idx = np.concatenate([
            rng.choice(np.where(cluster_ids == c)[0], size=min(b, int((cluster_ids == c).sum())), replace=False)
            for c, b in zip(unique, budgets)
        ])
        X_s  = embeddings[idx]
        lbl_s = cluster_ids[idx]
        print(f"  Silhouette: sampled {len(idx):,} / {n:,} posts (stratified)")
    else:
        X_s, lbl_s = embeddings, cluster_ids

    # Cosine distance (1 - cosine_similarity) is more meaningful for text embeddings
    score = silhouette_score(X_s, lbl_s, metric="cosine")

    # Per-cluster silhouette for reporting
    sample_vals = silhouette_samples(X_s, lbl_s, metric="cosine")
    per_cluster = {
        int(c): round(float(sample_vals[lbl_s == c].mean()), 4)
        for c in np.unique(lbl_s)
    }

    result = {
        "silhouette_score":        round(float(score), 4),
        "sample_size":             int(len(X_s)),
        "n_clusters":              int(np.unique(lbl_s).size),
        "metric":                  "cosine",
        "per_cluster_silhouette":  per_cluster,
        "interpretation": (
            "well-separated" if score > 0.5 else
            "moderate"       if score > 0.2 else
            "overlapping"
        ),
    }
    print(f"  Silhouette Score: {score:.4f} ({result['interpretation']})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. NDCG
# ══════════════════════════════════════════════════════════════════════════════

def compute_ndcg(embeddings: np.ndarray, risk_labels: np.ndarray,
                 cluster_ids: np.ndarray) -> dict:
    """
    Measures whether cosine similarity to a 'high-risk centroid' surfaces
    high-risk posts at the top of the ranking.

    Binary relevance: high-risk = 1, medium/low = 0.

    Also computes cluster-level NDCG: ranks clusters by mean similarity
    to the high-risk centroid; relevance = fraction of high-risk posts
    per cluster (continuous label).

    Uses sklearn.metrics.ndcg_score.
    """
    from sklearn.metrics import ndcg_score

    is_high = (risk_labels == "high").astype(float)

    if is_high.sum() == 0:
        print("  NDCG: no high-risk posts found — skipping.")
        return {"error": "no high-risk posts"}

    # ── High-risk centroid ────────────────────────────────────────────────────
    high_idx    = np.where(is_high == 1)[0]
    centroid    = embeddings[high_idx].mean(axis=0)
    centroid   /= np.linalg.norm(centroid) + 1e-9

    # L2-normalise all embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    E_norm = embeddings / norms

    similarities = E_norm @ centroid          # (N,) cosine sim to high-risk centroid

    # ── Post-level NDCG@k ─────────────────────────────────────────────────────
    # ndcg_score expects shape (1, N)
    true_rel  = is_high.reshape(1, -1)
    pred_score = similarities.reshape(1, -1)

    ndcg_post = {}
    for k in NDCG_K_LIST:
        if k <= len(is_high):
            ndcg_post[f"ndcg_at_{k}"] = round(
                float(ndcg_score(true_rel, pred_score, k=k)), 4
            )

    # ── Cluster-level NDCG ───────────────────────────────────────────────────
    unique_clusters = np.unique(cluster_ids)
    cluster_sim   = np.array([similarities[cluster_ids == c].mean() for c in unique_clusters])
    cluster_relevance = np.array([is_high[cluster_ids == c].mean() for c in unique_clusters])

    ndcg_cluster = round(
        float(ndcg_score(
            cluster_relevance.reshape(1, -1),
            cluster_sim.reshape(1, -1),
        )), 4
    )

    # Top-5 clusters by similarity to high-risk centroid
    top5_idx = np.argsort(cluster_sim)[::-1][:5]
    top5 = [
        {
            "cluster":         int(unique_clusters[i]),
            "mean_sim_to_hr":  round(float(cluster_sim[i]), 4),
            "pct_high_risk":   round(float(cluster_relevance[i] * 100), 1),
            "n_posts":         int((cluster_ids == unique_clusters[i]).sum()),
        }
        for i in top5_idx
    ]

    result = {
        "post_level":    ndcg_post,
        "cluster_level": {"ndcg_all_clusters": ndcg_cluster},
        "top5_clusters_by_risk_similarity": top5,
        "n_high_risk_posts":  int(is_high.sum()),
        "n_total_posts":      int(len(is_high)),
        "pct_high_risk":      round(float(is_high.mean() * 100), 2),
    }
    print(f"  NDCG@100 (post-level): {ndcg_post.get('ndcg_at_100', 'N/A')}")
    print(f"  NDCG cluster-level   : {ndcg_cluster}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Perplexity
# ══════════════════════════════════════════════════════════════════════════════

def _load_slang_lexicon() -> dict[str, str]:
    """Import SLANG_LEXICON from preprocess_posts.py without running the module."""
    import importlib.util, sys

    spec = importlib.util.spec_from_file_location(
        "preprocess_posts",
        ROOT / "src" / "processing" / "preprocess_posts.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Prevent the module's __main__ block from running
    sys.modules.setdefault("preprocess_posts", mod)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass  # partial load is fine — we only need SLANG_LEXICON

    for attr in ("SLANG_LEXICON", "SLANG_MAP", "slang_map", "slang_lexicon"):
        if hasattr(mod, attr):
            return getattr(mod, attr)

    raise AttributeError(
        "Could not find SLANG_LEXICON / SLANG_MAP in preprocess_posts.py"
    )


class BigramLM:
    """
    Additive-smoothed (Laplace) bigram language model.

    Perplexity on a list of tokens W:
        PP = exp( -1/|W| * sum_i log P(w_i | w_{i-1}) )
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.unigram: Counter = Counter()
        self.bigram:  defaultdict = defaultdict(Counter)
        self.vocab_size: int = 0

    def tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"[a-z']+", text.lower())

    def train(self, texts: list[str]) -> None:
        print(f"  Training bigram LM on {len(texts):,} documents ...")
        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                continue
            self.unigram.update(tokens)
            prev = "<s>"
            for tok in tokens:
                self.bigram[prev][tok] += 1
                prev = tok

        self.vocab_size = len(self.unigram)
        print(f"  Vocabulary size: {self.vocab_size:,} tokens")

    def log_prob(self, word: str, context: str = "<s>") -> float:
        """Laplace-smoothed bigram log-probability."""
        bigram_count  = self.bigram[context].get(word, 0) + self.smoothing
        context_total = sum(self.bigram[context].values()) + self.smoothing * self.vocab_size
        return math.log(bigram_count / context_total)

    def perplexity(self, tokens: list[str]) -> float:
        """Sentence-level perplexity for a token sequence."""
        if not tokens:
            return float("inf")
        log_sum = 0.0
        prev = "<s>"
        for tok in tokens:
            log_sum += self.log_prob(tok, prev)
            prev = tok
        return math.exp(-log_sum / len(tokens))


def compute_perplexity(merged_df: pd.DataFrame) -> dict:
    """
    Train a bigram LM on processed_text, then measure perplexity on:
      (a) the slang vocabulary terms (individual words)
      (b) the canonical drug names mapped to by the slang terms
      (c) a random sample of general English words as a baseline

    Lower perplexity on slang terms → model better represents domain language.
    """
    text_col = "processed_text" if "processed_text" in merged_df.columns else "original_text"
    texts = merged_df[text_col].dropna().astype(str).tolist()

    lm = BigramLM(smoothing=1.0)
    lm.train(texts)

    # Load slang lexicon
    try:
        slang_lexicon = _load_slang_lexicon()
    except Exception as e:
        print(f"  Warning: could not load SLANG_LEXICON: {e}")
        slang_lexicon = {
            "oxy": "oxycodone", "h": "heroin", "dope": "heroin",
            "meth": "methamphetamine", "coke": "cocaine", "weed": "cannabis",
            "percs": "percocet", "addys": "adderall", "bars": "xanax",
            "nodding": "opioid intoxication", "fiending": "craving",
        }

    slang_terms     = list(slang_lexicon.keys())
    canonical_terms = list(set(slang_lexicon.values()))

    # Baseline: common English words unlikely to appear in drug posts
    baseline_words = [
        "weather", "garden", "music", "travel", "cooking",
        "technology", "science", "history", "geography", "sports",
    ]

    def _avg_pp(word_list: list[str]) -> float:
        pps = [lm.perplexity(lm.tokenize(w)) for w in word_list]
        pps = [p for p in pps if not math.isinf(p) and not math.isnan(p)]
        return round(float(np.mean(pps)), 2) if pps else float("inf")

    pp_slang     = _avg_pp(slang_terms)
    pp_canonical = _avg_pp(canonical_terms)
    pp_baseline  = _avg_pp(baseline_words)

    # Per-term perplexity for slang
    per_term = {
        w: round(lm.perplexity(lm.tokenize(w)), 2)
        for w in sorted(slang_terms)
    }

    result = {
        "avg_perplexity_slang_terms":    pp_slang,
        "avg_perplexity_canonical_terms": pp_canonical,
        "avg_perplexity_baseline_words":  pp_baseline,
        "domain_coverage_ratio": round(pp_baseline / pp_slang, 2) if pp_slang > 0 else None,
        "n_slang_terms":    len(slang_terms),
        "n_canonical_terms": len(canonical_terms),
        "vocab_size":        lm.vocab_size,
        "smoothing":         lm.smoothing,
        "per_term_perplexity": per_term,
        "interpretation": (
            f"LM perplexity on slang ({pp_slang:.1f}) vs general English ({pp_baseline:.1f}). "
            f"Ratio {round(pp_baseline / pp_slang, 2) if pp_slang > 0 else 'N/A'}x — "
            + ("model strongly captures domain slang." if pp_slang < pp_baseline * 0.5 else
               "model moderately captures domain slang." if pp_slang < pp_baseline else
               "slang terms are rare in corpus.")
        ),
    }
    print(f"  Perplexity slang terms   : {pp_slang}")
    print(f"  Perplexity canonical     : {pp_canonical}")
    print(f"  Perplexity baseline words: {pp_baseline}")
    print(f"  Domain coverage ratio    : {result['domain_coverage_ratio']}x")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Cluster Quality Metrics")
    print("=" * 60)

    embeddings, cluster_ids, risk_labels, merged_df = _load_data()

    # ── 1. Silhouette Score ───────────────────────────────────────────────────
    print("\n[1/3] Silhouette Score ...")
    silhouette_result = compute_silhouette(embeddings, cluster_ids)

    # ── 2. NDCG ──────────────────────────────────────────────────────────────
    print("\n[2/3] NDCG ...")
    ndcg_result = compute_ndcg(embeddings, risk_labels, cluster_ids)

    # ── 3. Perplexity ─────────────────────────────────────────────────────────
    print("\n[3/3] Perplexity ...")
    perplexity_result = compute_perplexity(merged_df)

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "silhouette": silhouette_result,
        "ndcg":       ndcg_result,
        "perplexity": perplexity_result,
    }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=_NpEncoder)

    print(f"\nSaved -> {OUT_JSON}")
    print("\nSummary:")
    print(f"  Silhouette Score : {silhouette_result['silhouette_score']} "
          f"({silhouette_result['interpretation']})")
    ndcg100 = ndcg_result.get("post_level", {}).get("ndcg_at_100", "N/A")
    print(f"  NDCG@100         : {ndcg100}")
    print(f"  Perplexity slang : {perplexity_result['avg_perplexity_slang_terms']}")
    print(f"  Domain ratio     : {perplexity_result['domain_coverage_ratio']}x")


if __name__ == "__main__":
    main()
