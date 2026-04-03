"""
rag_pipeline.py
===============
Task 3 — Advanced RAG with cross-encoder reranking.

Pipeline per detected spike event:
  Step 1  Retrieve top-(retrieve_k=30) candidate posts by cosine similarity
          (bi-encoder: all-MiniLM-L6-v2, already loaded in embedding_classifier)
  Step 2  Build a refined, context-aware query from initial candidates
  Step 3  Rerank with a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
          Gracefully degrades to bi-encoder order if model unavailable.
  Step 4  Keep top-(final_k=10) highest-scored posts
  Step 5  Generate population-level analyst summary via Gemini
          (calls generate_spike_summary from llm_classifier)
  Step 6  Normalise output keys for dashboard compatibility and save JSON

Spike events are sourced from the narrative drift events in
  data/processed/narrative/window_df.parquet  (is_drift_event column)
with a z-score fallback on the ensemble monthly signal.

Run:
    python src/rag_pipeline.py

Outputs:
    data/processed/spike_summaries.json   — analyst reports, one per spike
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent   # project root
DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
RAW       = DATA / "raw"
sys.path.insert(0, str(ROOT / "src" / "classifiers"))
sys.path.insert(0, str(ROOT / "src" / "agents"))
sys.path.insert(0, str(ROOT / "src"))

ENSEMBLE_CSV    = PROCESSED / "ensemble_results.csv"
POSTS_CSV       = PROCESSED / "posts_preprocessed.csv"
EMBED_PARQUET   = PROCESSED / "post_embeddings.parquet"
CDC_CSV         = PROCESSED / "cdc_cleaned.csv"
WINDOW_PARQUET  = PROCESSED / "narrative" / "window_df.parquet"
OUT_SPIKES_JSON = PROCESSED / "spike_summaries.json"

# ── Erowid NMF context loader ─────────────────────────────────────────────────

def _load_erowid_profile_for_substance(
    substance: str,
    profiles_path: Path = PROCESSED / "erowid_substance_profiles.json",
) -> str | None:
    """
    Return a short formatted string describing the Erowid NMF dominant topic
    terms for a given substance, for injection into the Gemini prompt.

    The string is explicitly labeled as aggregate community-report data so
    Gemini treats it as a reference signal, not a verbatim user post.

    Returns None if the profiles file is absent or the substance is not found
    (graceful degradation — the RAG pipeline continues without Erowid context).

    Example output:
        "Erowid self-report corpus (N=842 aggregate experiences, NMF topic
         signals) for heroin: withdrawal, sick, needle, vein, rush, sweat,
         craving, dark, desperate, inject."
    """
    if not profiles_path.exists():
        return None
    try:
        with open(profiles_path, encoding="utf-8") as f:
            data = json.load(f)
        sub_data = data.get("substances", {}).get(substance.lower())
        if not sub_data:
            return None
        terms  = sub_data.get("dominant_topic_terms", [])[:10]
        n_docs = sub_data.get("n_docs", "unknown")
        return (
            f"Erowid self-report corpus (N={n_docs} aggregate experiences, "
            f"NMF topic signals) for {substance}: {', '.join(terms)}."
        )
    except Exception:
        return None


# ── Cross-encoder lazy loader ──────────────────────────────────────────────────
_CE_MODEL      = None
_CE_AVAILABLE  = False
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_cross_encoder():
    """Load the cross-encoder once; silence failure gracefully."""
    global _CE_MODEL, _CE_AVAILABLE
    if _CE_MODEL is not None:
        return _CE_MODEL
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Loading cross-encoder: {_CE_MODEL_NAME} …")
        _CE_MODEL     = CrossEncoder(_CE_MODEL_NAME, max_length=512)
        _CE_AVAILABLE = True
        print("  Cross-encoder ready.")
    except Exception as exc:
        print(f"  !! Cross-encoder unavailable ({exc})")
        print("     Falling back to bi-encoder cosine ranking.")
    return _CE_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# STEP A — Cross-encoder reranking
# ══════════════════════════════════════════════════════════════════════════════

def rerank_posts(query: str,
                 candidates: list[dict],
                 top_k: int = 10) -> list[dict]:
    """
    Re-score `candidates` against `query` using a cross-encoder and return
    the top-k highest-scored results.

    Parameters
    ----------
    query      : search query string (describes the spike / risk concept)
    candidates : list of dicts from retrieve_similar_posts(), each has 'text'
    top_k      : number of posts to keep after reranking

    Returns
    -------
    list[dict] sorted by rerank_score descending, length <= top_k
    Each dict gets an extra 'rerank_score' key (float).

    Fallback
    --------
    If the cross-encoder model cannot be loaded, the function returns the
    candidates sorted by their original cosine 'similarity' score instead,
    so the pipeline never breaks.
    """
    model = _load_cross_encoder()
    if model is None:
        # Fallback: sort by bi-encoder cosine sim (already computed)
        for c in candidates:
            c["rerank_score"] = c.get("similarity", 0.0)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    pairs  = [(query, c["text"][:512]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STEP B — Spike period detection
# ══════════════════════════════════════════════════════════════════════════════

def _detect_spike_periods(window_df: pd.DataFrame,
                           max_spikes: int = 12) -> list[dict]:
    """
    Extract spike periods from the pre-computed narrative drift window table.

    Prioritises rows where:
        is_drift_event == True   (z-score > 2 on drift_score)

    Falls back to the top-N rows by drift_score if no flagged events exist.

    Returns a list of period dicts, sorted by drift_score descending.
    """
    if window_df.empty or "period_start" not in window_df.columns:
        return []

    window_df = window_df.copy()
    window_df["period_start"] = pd.to_datetime(
        window_df["period_start"], errors="coerce"
    )

    if "is_drift_event" in window_df.columns:
        spikes_df = window_df[window_df["is_drift_event"] == True]
    else:
        spikes_df = pd.DataFrame()

    # Fallback: take the top rows by raw drift_score
    if spikes_df.empty and "drift_score" in window_df.columns:
        spikes_df = window_df.nlargest(max_spikes, "drift_score")

    if spikes_df.empty:
        spikes_df = window_df.head(max_spikes)

    # Sort strongest drift first, cap to max_spikes
    if "drift_score" in spikes_df.columns:
        spikes_df = spikes_df.sort_values("drift_score", ascending=False)
    spikes_df = spikes_df.head(max_spikes)

    periods = []
    for _, row in spikes_df.iterrows():
        ps = row["period_start"]
        if pd.isna(ps):
            continue
        # Assume weekly windows (narrative_evolution default freq="W")
        pe = ps + pd.Timedelta(weeks=1)
        periods.append({
            "spike_date":    str(ps)[:10],
            "period_start":  ps,
            "period_end":    pe,
            "pct_high":      float(row["pct_high"])      if "pct_high"      in row.index and pd.notna(row["pct_high"])      else None,
            "drift_score":   float(row["drift_score"])   if "drift_score"   in row.index and pd.notna(row["drift_score"])   else None,
            "post_count":    int(row["post_count"])       if "post_count"    in row.index and pd.notna(row["post_count"])    else None,
        })

    return periods


def _fallback_spike_periods(posts_df: pd.DataFrame,
                             ensemble_df: pd.DataFrame | None,
                             max_spikes: int = 8) -> list[dict]:
    """
    Z-score spike detection on monthly high-risk post volume.
    Used when no window_df is available.
    """
    df = ensemble_df if ensemble_df is not None else posts_df
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        # No temporal signal — return one synthetic 'spike' covering everything
        return [{"spike_date": "all", "period_start": None,
                 "period_end": None, "pct_high": None, "drift_score": None,
                 "post_count": len(df)}]

    tmp = df.copy()
    tmp["ts"]     = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp["period"] = tmp["ts"].dt.to_period("M").dt.to_timestamp()

    if "risk_level" in tmp.columns:
        monthly = (
            tmp[tmp["risk_level"] == "high"]
            .groupby("period")
            .size()
            .rename("n_high")
            .reset_index()
        )
    else:
        monthly = tmp.groupby("period").size().rename("n_high").reset_index()

    if len(monthly) < 3:
        return [{"spike_date": str(monthly.iloc[0]["period"])[:10],
                 "period_start": monthly.iloc[0]["period"],
                 "period_end":   monthly.iloc[0]["period"] + pd.offsets.MonthEnd(1),
                 "pct_high": None, "drift_score": None,
                 "post_count": int(monthly.iloc[0]["n_high"])}]

    mu = monthly["n_high"].mean()
    sd = monthly["n_high"].std()
    if sd > 0:
        monthly["z"] = (monthly["n_high"] - mu) / sd
    else:
        monthly["z"] = 0.0

    spikes = monthly[monthly["z"] > 1.5].nlargest(max_spikes, "z")
    return [
        {
            "spike_date":   str(row["period"])[:10],
            "period_start": row["period"],
            "period_end":   row["period"] + pd.offsets.MonthEnd(1),
            "pct_high":     None,
            "drift_score":  float(row["z"]),
            "post_count":   int(row["n_high"]),
        }
        for _, row in spikes.iterrows()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STEP C — Contextual query construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_retrieval_query(spike: dict, candidates: list[dict]) -> str:
    """
    Build a focused retrieval query for the cross-encoder reranking step.

    Uses:
      - dominant substance from the initial bi-encoder candidates
      - spike date for temporal grounding
      - fixed risk vocabulary to bias toward abusive / distress content
    """
    substances: list[str] = []
    for c in candidates[:20]:
        raw = c.get("substance", "[]")
        try:
            subs = json.loads(raw) if isinstance(raw, str) else raw
            substances.extend(subs if isinstance(subs, list) else [])
        except Exception:
            pass

    top_sub = Counter(substances).most_common(1)
    sub_str = top_sub[0][0] if top_sub else "substance"

    return (
        f"social media post about {sub_str} abuse, dependency, withdrawal, "
        f"craving, relapse, or harm reduction during {spike['spike_date']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP D — Single-spike orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_for_spike(spike:       dict,
                      posts_df:    pd.DataFrame,
                      embeddings:  np.ndarray,
                      cdc_df:      pd.DataFrame | None,
                      retrieve_k:  int = 30,
                      final_k:     int = 10) -> dict | None:
    """
    Full retrieve → rerank → generate pipeline for one spike event.

    Returns a normalised summary dict ready for the dashboard, or None if
    fewer than 3 candidate posts can be found.
    """
    from embedding_classifier import retrieve_similar_posts
    from llm_classifier import generate_spike_summary

    # ── Narrow to the spike's time window if timestamps are available ──────────
    spike_df   = posts_df
    spike_embs = embeddings

    if spike.get("period_start") is not None:
        ts   = pd.to_datetime(posts_df.get("timestamp", pd.Series()), errors="coerce")
        mask = (ts >= spike["period_start"]) & (ts < spike["period_end"])
        if mask.sum() >= 5:
            spike_df   = posts_df[mask].reset_index(drop=True)
            spike_embs = embeddings[mask.values]

    # ── Step 1: Bi-encoder retrieval (top retrieve_k, high-risk first) ─────────
    query_init = f"substance abuse risk dependency withdrawal {spike['spike_date']}"

    candidates = retrieve_similar_posts(
        query_init, spike_df, spike_embs,
        top_k=retrieve_k, risk_filter="high",
    )
    if len(candidates) < 3:
        # Not enough high-risk posts in window — try without risk filter
        candidates = retrieve_similar_posts(
            query_init, spike_df, spike_embs,
            top_k=retrieve_k, risk_filter=None,
        )
    if len(candidates) < 3:
        print(f"  Skipping spike {spike['spike_date']} — only "
              f"{len(candidates)} candidates found.")
        return None

    # ── Step 2: Build context-aware query for cross-encoder ───────────────────
    rerank_query = _build_retrieval_query(spike, candidates)

    # ── Step 3-4: Cross-encoder rerank → keep top final_k ─────────────────────
    top_posts = rerank_posts(rerank_query, candidates, top_k=final_k)

    # ── Step 5: Assemble CDC context for the period ───────────────────────────
    cdc_context: dict | None = None
    if cdc_df is not None and spike.get("period_start") is not None:
        try:
            cdc_tmp = cdc_df.copy()
            cdc_tmp["date"] = pd.to_datetime(cdc_tmp["date"], errors="coerce")
            target_month = spike["period_start"].to_period("M")
            period_cdc   = cdc_tmp[
                cdc_tmp["date"].dt.to_period("M") == target_month
            ]
            if len(period_cdc) > 0:
                top_row     = period_cdc.sort_values("deaths", ascending=False).iloc[0]
                cdc_context = {
                    "deaths":    int(top_row["deaths"]),
                    "substance": str(top_row.get("substance", "N/A")),
                    "state":     str(top_row.get("state", "National")),
                    "pct_change": None,
                }
        except Exception:
            pass

    # ── Step 6: Convert to generate_spike_summary format ─────────────────────
    flagged_posts = [
        {
            "processed_text": p["text"],
            "risk_level":     p.get("risk_level", "high"),
            "rerank_score":   p.get("rerank_score", 0.0),
        }
        for p in top_posts
    ]

    # ── Step 6b: Inject Erowid NMF context for the dominant substance ─────────
    # Identify the dominant substance from the retrieved posts, then prepend an
    # aggregate Erowid reference entry so Gemini has grounded vocabulary about
    # what this substance's user community typically reports.
    # This entry is clearly labeled as aggregate community data, not a real post.
    erowid_context_str: str | None = None
    _all_substances: list[str] = []
    for p in top_posts:
        raw = p.get("substance", "[]")
        try:
            subs = json.loads(raw) if isinstance(raw, str) else raw
            _all_substances.extend(subs if isinstance(subs, list) else [])
        except Exception:
            pass
    _top_sub = Counter(_all_substances).most_common(1)
    if _top_sub:
        erowid_context_str = _load_erowid_profile_for_substance(_top_sub[0][0])
    if erowid_context_str:
        flagged_posts = [{
            "processed_text": erowid_context_str,
            "risk_level":     "context",     # sentinel — not a real risk label
            "rerank_score":   0.0,
            "_erowid_signal": True,
        }] + flagged_posts

    # ── Step 7: Generate via Gemini (or store retrieval-only record) ───────────
    raw_summary = generate_spike_summary(
        flagged_posts=flagged_posts,
        spike_date=spike["spike_date"],
        supporting_cdc_data=cdc_context,
    )

    # ── Normalise keys for dashboard compatibility ─────────────────────────────
    # generate_spike_summary returns: summary, dominant_substances,
    # dominant_signals, severity, recommended_action, confidence,
    # spike_date, post_count, generated_at
    # Dashboard expects: analyst_summary, risk_level, evidence_spans, rationale
    #
    # When Gemini is unavailable raw_summary = {"error": "...", "spike_date": ...}
    # Move "error" -> "llm_error" so the dashboard still renders retrieval results
    # instead of showing the error banner and skipping the whole card.
    normalized = {k: v for k, v in raw_summary.items() if k != "error"}
    if "error" in raw_summary:
        normalized["llm_error"] = raw_summary["error"]

    # ── Fallback values derived from retrieved posts (used when Gemini unavailable)
    _risk_votes  = [p.get("risk_level", "unknown") for p in top_posts
                    if p.get("risk_level", "unknown") != "unknown"]
    _risk_counts = Counter(_risk_votes)
    # Fall back to spike's own pct_high when no risk labels are available
    if _risk_counts:
        _inferred_risk = _risk_counts.most_common(1)[0][0]
    elif spike.get("pct_high") is not None:
        _pct = spike["pct_high"]
        _inferred_risk = "high" if _pct >= 0.10 else ("medium" if _pct >= 0.04 else "low")
    else:
        _inferred_risk = "moderate"

    _rerank_scores = [p.get("rerank_score", p.get("similarity", 0.0)) for p in top_posts]
    _score_range   = max(_rerank_scores) - min(_rerank_scores) if len(_rerank_scores) > 1 else 1.0
    _norm_scores   = [
        (s - min(_rerank_scores)) / _score_range if _score_range > 0 else 1.0
        for s in _rerank_scores
    ]
    _inferred_conf = round(float(np.mean(_norm_scores[:5])), 3) if _norm_scores else 0.0

    # Deduplicate evidence spans while preserving order
    _seen_spans: set[str] = set()
    _deduped_spans: list[str] = []
    for p in top_posts:
        snippet = p["text"][:120].strip()
        if snippet not in _seen_spans:
            _seen_spans.add(snippet)
            _deduped_spans.append(snippet)
        if len(_deduped_spans) >= 5:
            break

    # Always ensure dashboard-required keys are present
    normalized.setdefault("analyst_summary",
                           raw_summary.get("summary", ""))
    normalized.setdefault("risk_level",
                           raw_summary.get("severity", _inferred_risk))
    normalized.setdefault("confidence",
                           raw_summary.get("confidence", _inferred_conf))
    normalized.setdefault("generated_at",
                           raw_summary.get("generated_at",
                                           pd.Timestamp.now().isoformat()))
    normalized.setdefault("evidence_spans", _deduped_spans)
    normalized.setdefault("rationale",
                           raw_summary.get("recommended_action", ""))
    # post_count from spike metadata when LLM path did not supply it
    normalized.setdefault("post_count",
                           spike.get("post_count") or len(top_posts))

    # ── Annotate with retrieval provenance ────────────────────────────────────
    normalized["retrieval_method"]  = "bi-encoder + cross-encoder rerank"
    normalized["retrieve_k"]        = retrieve_k
    normalized["final_k"]           = final_k
    normalized["n_candidates"]      = len(candidates)
    normalized["rerank_query"]      = rerank_query
    normalized["cdc_context"]       = cdc_context
    normalized["pct_high"]          = spike.get("pct_high")
    normalized["drift_score"]       = spike.get("drift_score")
    # Erowid community-reference signal (aggregate NMF, not a real post)
    normalized["erowid_nmf_context"] = erowid_context_str

    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# STEP E — Full pipeline orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(posts_df:    pd.DataFrame,
                     embeddings:  np.ndarray,
                     cdc_df:      pd.DataFrame | None = None,
                     ensemble_df: pd.DataFrame | None = None,
                     max_spikes:  int = 12,
                     retrieve_k:  int = 30,
                     final_k:     int = 10) -> list[dict]:
    """
    Orchestrate the full RAG pipeline across all detected spike events.

    Parameters
    ----------
    posts_df    : preprocessed posts (processed_text, timestamp, risk_level)
    embeddings  : (N, 384) float32 array aligned row-for-row with posts_df
    cdc_df      : optional CDC overdose frame from signal_pipeline
    ensemble_df : optional ensemble results (used for fallback spike detection)
    max_spikes  : cap on number of spike events to process
    retrieve_k  : bi-encoder retrieval pool size
    final_k     : posts kept after cross-encoder rerank

    Returns
    -------
    list[dict] — normalised analyst summaries, one per spike
    """
    # ── Detect spike periods ──────────────────────────────────────────────────
    print("Detecting spike periods …")
    periods: list[dict] = []

    if WINDOW_PARQUET.exists():
        try:
            window_df = pd.read_parquet(str(WINDOW_PARQUET))
            periods   = _detect_spike_periods(window_df, max_spikes)
            print(f"  {len(periods)} spike event(s) from narrative drift window.")
        except Exception as exc:
            print(f"  !! Could not load window_df ({exc}); using z-score fallback.")

    if not periods:
        periods = _fallback_spike_periods(posts_df, ensemble_df, max_spikes)
        print(f"  {len(periods)} spike event(s) from z-score fallback.")

    if not periods:
        print("  No spike periods detected. Exiting.")
        return []

    # ── Process each spike ────────────────────────────────────────────────────
    summaries: list[dict] = []
    print(f"\nProcessing {len(periods)} spike event(s) "
          f"(retrieve_k={retrieve_k}, final_k={final_k}) …\n")

    for i, spike in enumerate(periods):
        print(f"[{i+1}/{len(periods)}] Spike: {spike['spike_date']}  "
              f"drift={spike.get('drift_score', 'n/a')}  "
              f"pct_high={spike.get('pct_high', 'n/a')}")

        result = run_rag_for_spike(
            spike       = spike,
            posts_df    = posts_df,
            embeddings  = embeddings,
            cdc_df      = cdc_df,
            retrieve_k  = retrieve_k,
            final_k     = final_k,
        )

        if result is not None:
            summaries.append(result)
            has_llm = "llm_error" not in result
            status  = "generated" if has_llm else f"retrieval-only ({result.get('llm_error','')})"
            print(f"  -> {status}  |  "
                  f"candidates={result['n_candidates']}  "
                  f"kept={final_k}")
        print()

    return summaries


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load posts ────────────────────────────────────────────────────────────
    if not POSTS_CSV.exists():
        print(f"!! Posts not found: {POSTS_CSV}")
        print("   Run data/preprocess_posts.py first.")
        return

    print(f"Loading posts … ({POSTS_CSV.name})")
    posts_df = pd.read_csv(str(POSTS_CSV))
    if ENSEMBLE_CSV.exists():
        ens = pd.read_csv(str(ENSEMBLE_CSV))
        if "risk_level" in ens.columns and "risk_level" not in posts_df.columns:
            posts_df["risk_level"] = ens["risk_level"].values
        ensemble_df = ens
    else:
        ensemble_df = None
    print(f"  {len(posts_df):,} posts loaded")

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not EMBED_PARQUET.exists():
        print(f"!! Embeddings not found: {EMBED_PARQUET}")
        print("   Run src/embedding_classifier.py first.")
        return

    from embedding_classifier import load_embeddings_parquet
    _, embeddings = load_embeddings_parquet(EMBED_PARQUET)
    print(f"  Embeddings: {embeddings.shape}")

    # ── Align rows ────────────────────────────────────────────────────────────
    if len(embeddings) != len(posts_df):
        n = min(len(embeddings), len(posts_df))
        print(f"  !! Row count mismatch — truncating to {n:,} rows.")
        posts_df   = posts_df.iloc[:n].reset_index(drop=True)
        embeddings = embeddings[:n]

    # ── Load CDC data ─────────────────────────────────────────────────────────
    cdc_df = None
    if CDC_CSV.exists():
        cdc_df = pd.read_csv(str(CDC_CSV), parse_dates=["date"])
        print(f"  CDC data: {len(cdc_df):,} rows")
    else:
        print("  CDC data not available (run signal_pipeline.py to generate).")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    summaries = run_rag_pipeline(
        posts_df    = posts_df,
        embeddings  = embeddings,
        cdc_df      = cdc_df,
        ensemble_df = ensemble_df,
        max_spikes  = 12,
        retrieve_k  = 30,
        final_k     = 10,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(str(OUT_SPIKES_JSON), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)

    n_generated = sum(1 for s in summaries if "llm_error" not in s)
    print(f"Saved {len(summaries)} report(s) to {OUT_SPIKES_JSON}")
    print(f"  LLM-generated : {n_generated}")
    print(f"  Retrieval-only: {len(summaries) - n_generated}")
    print("\nDone. Run 'streamlit run src/dashboard.py' to view the Analyst Reports tab.")


if __name__ == "__main__":
    main()
