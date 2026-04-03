"""
narrative_evolution.py
======================
Temporal narrative tracking for the Substance Abuse Detection pipeline.

Architecture:
    timestamped posts + ensemble risk labels
        ├── Window embeddings       → narrative position per time period
        ├── Drift detection         → when did the narrative shift?
        ├── Topic transition        → which themes are rising/falling?
        ├── Early warning signals   → what precedes an overdose spike?
        └── UMAP trajectory         → 2-D narrative path for the dashboard

Inputs (all from data/processed/):
    posts_preprocessed.csv    — post_id, timestamp, processed_text,
                                 substances_detected
    ensemble_results.csv      — post_id, final_risk_level
    post_embeddings.parquet   — post_id, embedding  (float32 list)
    ../raw/cdc_overdose_data.csv — via signal_pipeline.load_cdc_overdose()

Outputs (data/processed/narrative/):
    window_df.parquet
    topic_df.parquet
    trajectory_df.csv
    warning_report.csv
    rising_events.json
    lag_results.json
    figures/
        dual_axis_drift.html/.png
        topic_prevalence_area.html/.png
        umap_trajectory.html/.png
        rising_alerts_table.html

Run:
    python src/narrative_evolution.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent   # project root
DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
OUT_DIR   = PROCESSED / "narrative"
FIG_DIR   = OUT_DIR / "figures"

POSTS_CSV    = PROCESSED / "posts_preprocessed.csv"
ENSEMBLE_CSV = PROCESSED / "ensemble_results.csv"
EMBED_PARQ   = PROCESSED / "post_embeddings.parquet"

# ── Add src/ to path so we can import signal_pipeline ─────────────────────────
sys.path.insert(0, str(ROOT / "src" / "agents"))
sys.path.insert(0, str(ROOT / "src"))
from signal_pipeline import load_cdc_overdose  # noqa: E402

# ── Sentence-transformers (for ANCHOR_EMBEDDINGS) ─────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _model = _ST("all-MiniLM-L6-v2")
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    _model = None

# ── UMAP ──────────────────────────────────────────────────────────────────────
try:
    import umap as _umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    _umap_lib = None

# ── Plotly ────────────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ── kaleido (static PNG export) ───────────────────────────────────────────────
try:
    import plotly.io as pio  # noqa: F401 — side-effect: registers kaleido
    _KALEIDO_AVAILABLE = True
except Exception:
    _KALEIDO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Topic anchors — encoded once at module load
# ══════════════════════════════════════════════════════════════════════════════

TOPIC_ANCHORS: dict[str, str] = {
    "relapse":        "I relapsed and started using again after being clean",
    "withdrawal":     "I'm going through withdrawals, dope sick, shaking and sweating",
    "craving":        "I can't stop thinking about it, the cravings are overwhelming",
    "harm_reduction": "how to use more safely, reduce risk, naloxone and clean supplies",
    "recovery":       "staying sober, going to meetings, proud of my sobriety",
    "procurement":    "where to get it, finding a dealer, buying drugs",
    "overdose":       "I overdosed, nearly died, revived with narcan",
    "hopelessness":   "I give up, nothing will ever change, no point trying",
}

ANCHOR_EMBEDDINGS: dict[str, np.ndarray] = {}

if _ST_AVAILABLE:
    print("Encoding topic anchors …")
    for _topic, _text in TOPIC_ANCHORS.items():
        ANCHOR_EMBEDDINGS[_topic] = _model.encode(
            [_text], normalize_embeddings=True
        )[0]
    print(f"  {len(ANCHOR_EMBEDDINGS)} anchors ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_analysis_inputs(
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and join posts_preprocessed.csv + ensemble_results.csv, then
    align the post_embeddings.parquet matrix to the same row order.

    Returns
    -------
    posts_df   : DataFrame with columns:
                   post_id, timestamp, processed_text, substances (list),
                   risk_level, ensemble_confidence (optional)
    embeddings : np.ndarray  shape (N, 384), dtype float32, aligned to posts_df
    """
    # ── posts ─────────────────────────────────────────────────────────────────
    if not POSTS_CSV.exists():
        raise FileNotFoundError(
            f"posts_preprocessed.csv not found: {POSTS_CSV}\n"
            "  Run: python data/preprocess_posts.py"
        )
    posts = pd.read_csv(POSTS_CSV)
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], errors="coerce")

    # Parse substances_detected string → Python list
    def _parse_substances(val):
        if pd.isna(val) or val == "":
            return []
        try:
            parsed = ast.literal_eval(str(val))
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (ValueError, SyntaxError):
            return [str(val)]

    posts["substances"] = posts["substances_detected"].apply(_parse_substances)

    # ── ensemble labels ────────────────────────────────────────────────────────
    if not ENSEMBLE_CSV.exists():
        raise FileNotFoundError(
            f"ensemble_results.csv not found: {ENSEMBLE_CSV}\n"
            "  Run: python src/ensemble.py"
        )
    ens_cols_avail = pd.read_csv(ENSEMBLE_CSV, nrows=0).columns.tolist()
    ens_use_cols   = [c for c in ["post_id", "final_risk_level", "ensemble_confidence"]
                      if c in ens_cols_avail]
    ensemble = pd.read_csv(ENSEMBLE_CSV, usecols=ens_use_cols)
    ensemble = ensemble.rename(columns={"final_risk_level": "risk_level"})

    # ── embeddings (load before join so we know row count) ─────────────────────
    if not EMBED_PARQ.exists():
        raise FileNotFoundError(
            f"post_embeddings.parquet not found: {EMBED_PARQ}\n"
            "  Run: python src/embedding_classifier.py"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("pyarrow is required.  pip install pyarrow")

    emb_table = pq.read_table(str(EMBED_PARQ))
    emb_ids   = emb_table.column("post_id").to_pylist()
    emb_vecs  = emb_table.column("embedding").to_pylist()

    # ── join ──────────────────────────────────────────────────────────────────
    # post_id is a valid key → use it; otherwise fall back to row-order alignment.
    _posts_key_valid    = posts["post_id"].notna().any()
    _ensemble_key_valid = ensemble["post_id"].notna().any()
    _emb_key_valid      = any(v not in (None, "nan", "None") for v in emb_ids[:10])

    if _posts_key_valid and _ensemble_key_valid:
        merged = posts.merge(ensemble, on="post_id", how="inner").reset_index(drop=True)
        print(f"  Joined posts: {len(merged):,} rows")

        emb_df = pd.DataFrame({"post_id": emb_ids, "_emb": emb_vecs})
        if _emb_key_valid:
            merged = merged.merge(emb_df, on="post_id", how="inner").reset_index(drop=True)
        else:
            # parquet post_ids are synthetic strings — align positionally
            assert len(emb_vecs) == len(merged), (
                f"Embedding count {len(emb_vecs)} != post count {len(merged)}; "
                "re-run embedding_classifier.py"
            )
            merged["_emb"] = emb_vecs
    else:
        # No stable key — align all three sources positionally
        n = min(len(posts), len(ensemble), len(emb_vecs))
        print(f"  post_id is all-NaN — using row-order alignment ({n:,} rows).")
        merged = posts.iloc[:n].copy().reset_index(drop=True)
        ens_data = ensemble.drop(columns=["post_id"], errors="ignore").iloc[:n].reset_index(drop=True)
        for col in ens_data.columns:
            merged[col] = ens_data[col].values
        merged["_emb"] = emb_vecs[:n]
        merged["post_id"] = range(n)      # synthetic key for downstream use

    print(f"  After embedding alignment: {len(merged):,} rows")
    embeddings = np.array(merged["_emb"].tolist(), dtype="float32")
    merged     = merged.drop(columns=["_emb"])

    return merged, embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Window centroids
# ══════════════════════════════════════════════════════════════════════════════

def compute_window_centroids(posts_df: pd.DataFrame,
                              embeddings: np.ndarray,
                              freq: str = "W",
                              min_posts: int = 10) -> pd.DataFrame:
    """
    Compute the embedding centroid for each time window.

    posts_df must have columns: timestamp, risk_level
    embeddings must be aligned row-for-row with posts_df.

    Returns one row per time window with centroid, risk breakdown, and
    top substances.
    """
    df = posts_df.copy().reset_index(drop=True)
    df["period"] = pd.to_datetime(df["timestamp"]).dt.to_period(freq)

    window_records = []

    for period, group in df.groupby("period"):
        if len(group) < min_posts:
            continue

        # Positional indices (safe regardless of original index values)
        pos_indices   = group.index.tolist()
        window_embs   = embeddings[pos_indices]

        centroid = window_embs.mean(axis=0)

        risk_centroids: dict[str, np.ndarray] = {}
        for risk in ("high", "medium", "low"):
            mask = (group["risk_level"] == risk).values
            if mask.sum() >= 3:
                risk_centroids[risk] = window_embs[mask].mean(axis=0)

        risk_counts = group["risk_level"].value_counts()
        total       = len(group)

        substances_top: dict = {}
        if "substances" in group.columns:
            substances_top = (
                group["substances"]
                .explode()
                .dropna()
                .value_counts()
                .head(3)
                .to_dict()
            )

        window_records.append({
            "period":         period,
            "period_start":   period.start_time,
            "post_count":     total,
            "centroid":       centroid,
            "risk_centroids": risk_centroids,
            "pct_high":       risk_counts.get("high",   0) / total,
            "pct_medium":     risk_counts.get("medium", 0) / total,
            "pct_low":        risk_counts.get("low",    0) / total,
            "substances":     substances_top,
        })

    return pd.DataFrame(window_records).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Narrative drift
# ══════════════════════════════════════════════════════════════════════════════

def compute_narrative_drift(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute period-over-period narrative drift (cosine distance between
    adjacent window centroids).

    Adds columns:
        drift_score      — cosine distance to previous window (0 = identical)
        cumulative_drift — running sum of drift scores
        drift_direction  — 'toward_risk' | 'away_from_risk' | 'unknown'
        is_drift_event   — True when drift z-score > 2.0
    """
    if len(window_df) == 0:
        for col in ("drift_score", "cumulative_drift", "drift_direction", "is_drift_event"):
            window_df[col] = pd.Series(dtype=object)
        return window_df

    df = window_df.copy().sort_values("period_start").reset_index(drop=True)

    drift_scores     = [np.nan]
    cumulative_drift = [0.0]
    drift_direction  = [None]

    for i in range(1, len(df)):
        prev_c = df.loc[i - 1, "centroid"]
        curr_c = df.loc[i,     "centroid"]

        sim   = cosine_similarity(
            prev_c.reshape(1, -1), curr_c.reshape(1, -1)
        )[0][0]
        drift = round(float(1.0 - sim), 4)

        drift_scores.append(drift)
        cumulative_drift.append(round(cumulative_drift[-1] + drift, 4))

        # Direction: is narrative moving toward the high-risk semantic pole?
        prev_rc = df.loc[i - 1, "risk_centroids"]
        if "high" in prev_rc:
            high_c    = prev_rc["high"]
            prev_dist = 1.0 - float(cosine_similarity(
                prev_c.reshape(1, -1), high_c.reshape(1, -1)
            )[0][0])
            curr_dist = 1.0 - float(cosine_similarity(
                curr_c.reshape(1, -1), high_c.reshape(1, -1)
            )[0][0])
            direction = "toward_risk" if curr_dist < prev_dist else "away_from_risk"
        else:
            direction = "unknown"

        drift_direction.append(direction)

    df["drift_score"]      = drift_scores
    df["cumulative_drift"] = cumulative_drift
    df["drift_direction"]  = drift_direction

    # Flag significant drift events (z-score > 2.0 over non-NaN values)
    valid_drift = df["drift_score"].dropna()
    if len(valid_drift) > 1:
        d_mean = valid_drift.mean()
        d_std  = valid_drift.std()
        df["is_drift_event"] = (
            (df["drift_score"] - d_mean) / (d_std if d_std > 0 else 1.0)
        ) > 2.0
    else:
        df["is_drift_event"] = False

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Topic prevalence + rising topic detection
# ══════════════════════════════════════════════════════════════════════════════

def compute_topic_prevalence(posts_df: pd.DataFrame,
                              embeddings: np.ndarray,
                              freq: str = "W",
                              min_posts: int = 10) -> pd.DataFrame:
    """
    For each time window, compute how strongly community language aligns
    with each TOPIC_ANCHOR.

    Prevalence = 75th-percentile cosine similarity across all posts in the
    window (captures the rising edge of a theme better than the mean).

    Returns DataFrame: one row per period, columns:
        period, period_start, post_count,
        <topic>_score, <topic>_pct  (for each anchor)
    """
    if not ANCHOR_EMBEDDINGS:
        raise RuntimeError(
            "ANCHOR_EMBEDDINGS is empty — sentence-transformers not installed.\n"
            "  pip install sentence-transformers"
        )

    df = posts_df.copy().reset_index(drop=True)
    df["period"] = pd.to_datetime(df["timestamp"]).dt.to_period(freq)

    records = []

    for period, group in df.groupby("period"):
        if len(group) < min_posts:
            continue

        pos_indices  = group.index.tolist()
        window_embs  = embeddings[pos_indices]

        row: dict = {
            "period":       period,
            "period_start": period.start_time,
            "post_count":   len(group),
        }

        for topic, anchor_emb in ANCHOR_EMBEDDINGS.items():
            sims = cosine_similarity(
                window_embs, anchor_emb.reshape(1, -1)
            )[:, 0]
            row[f"{topic}_score"] = round(float(np.percentile(sims, 75)), 4)
            row[f"{topic}_pct"]   = round(float((sims > 0.45).mean()), 4)

        records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


def detect_rising_topics(topic_df: pd.DataFrame,
                          window: int = 4,
                          threshold: float = 1.5) -> dict:
    """
    Flag topics whose score is rising faster than their historical baseline.

    Returns
    -------
    dict mapping period_start (Timestamp) → list of rising topic dicts.
    Each dict: topic, z_score, current_score, baseline_mean, pct_increase.
    """
    score_cols = [c for c in topic_df.columns if c.endswith("_score")]
    df         = topic_df.sort_values("period_start").reset_index(drop=True)

    rising_events: dict = {}

    for i in range(window, len(df)):
        period        = df.loc[i, "period_start"]
        rising_topics = []

        for col in score_cols:
            topic_name    = col.replace("_score", "")
            baseline      = df.loc[i - window: i - 1, col]
            current_score = float(df.loc[i, col])

            b_mean = float(baseline.mean())
            b_std  = float(baseline.std())

            if b_std < 0.001:
                continue      # stable topic — skip

            z = (current_score - b_mean) / b_std

            if z >= threshold:
                rising_topics.append({
                    "topic":         topic_name,
                    "z_score":       round(z, 2),
                    "current_score": round(current_score, 4),
                    "baseline_mean": round(b_mean, 4),
                    "pct_increase":  round(
                        (current_score - b_mean) / b_mean * 100, 1
                    ) if b_mean > 0 else 0.0,
                })

        if rising_topics:
            rising_topics.sort(key=lambda x: -x["z_score"])
            rising_events[period] = rising_topics

    return rising_events


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Early warning / lead-lag analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_lead_lag_correlation(topic_df: pd.DataFrame,
                                  cdc_df: pd.DataFrame,
                                  topic_col: str,
                                  cdc_col: str = "deaths",
                                  max_lag: int = 8) -> pd.DataFrame:
    """
    Test whether a rising topic score leads CDC overdose deaths.
    Positive lag = topic leads deaths (early-warning signal).

    Returns DataFrame: lag_periods, pearson_r, p_value, significant, interpretation.
    """
    topic_series = topic_df.set_index("period_start")[topic_col]

    # Aggregate CDC deaths to weekly or monthly totals (sum across states)
    cdc_agg = (
        cdc_df.groupby("date")[cdc_col]
        .sum()
        .rename(cdc_col)
    )

    # Explicit sort=False avoids pandas default-sort deprecation warning.
    combined = pd.concat([topic_series, cdc_agg], axis=1, sort=False).dropna()

    if len(combined) < 5:
        print(f"  ⚠  Not enough overlapping data for {topic_col} ({len(combined)} rows)")
        return pd.DataFrame()

    topic_s = combined[topic_col]
    cdc_s   = combined[cdc_col]

    results = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = cdc_s.shift(lag)
        mask    = shifted.notna()
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(topic_s[mask], shifted[mask])
        results.append({
            "lag_periods":  lag,
            "pearson_r":    round(float(r), 4),
            "p_value":      round(float(p), 4),
            "significant":  bool(p < 0.05),
            "interpretation": (
                f"Topic leads deaths by {lag} period(s)"  if lag > 0  else
                "Simultaneous"                             if lag == 0 else
                f"Deaths lead topic by {abs(lag)} period(s)"
            ),
        })

    lag_df = pd.DataFrame(results)

    # Report best leading signal
    if len(lag_df) > 0:
        leading = lag_df[
            (lag_df["lag_periods"] > 0) & lag_df["significant"]
        ].sort_values("pearson_r", ascending=False)
        if len(leading) > 0:
            best = leading.iloc[0]
            print(f"\n  Best early-warning signal for '{topic_col}':")
            print(f"    Leads overdose deaths by {best['lag_periods']} period(s)")
            print(f"    Pearson r = {best['pearson_r']},  p = {best['p_value']}")

    return lag_df


def build_early_warning_report(rising_events: dict,
                                lag_results_by_topic: dict) -> pd.DataFrame:
    """
    Synthesize rising topics + lead-lag analysis into an analyst-ready
    alert table.

    Columns: period, topic, z_score, pct_increase, lead_lag, alert_level.
    """
    rows = []

    for period, rising_topics in rising_events.items():
        for info in rising_topics:
            topic  = info["topic"]
            lag_df = lag_results_by_topic.get(topic, pd.DataFrame())

            lead_lag_summary = "No CDC correlation data"
            if len(lag_df) > 0:
                best_lead = lag_df[
                    (lag_df["lag_periods"] > 0) & lag_df["significant"]
                ].sort_values("pearson_r", ascending=False)
                if len(best_lead) > 0:
                    best             = best_lead.iloc[0]
                    lead_lag_summary = (
                        f"Leads overdose deaths by "
                        f"{best['lag_periods']} period(s) "
                        f"(r={best['pearson_r']})"
                    )

            rows.append({
                "period":       period,
                "topic":        topic,
                "z_score":      info["z_score"],
                "pct_increase": info["pct_increase"],
                "lead_lag":     lead_lag_summary,
                "alert_level": (
                    "critical" if info["z_score"] >= 3.0 else
                    "elevated" if info["z_score"] >= 2.0 else
                    "watch"
                ),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — UMAP trajectory
# ══════════════════════════════════════════════════════════════════════════════

def compute_narrative_trajectory(window_df: pd.DataFrame,
                                  n_neighbors: int = 5) -> pd.DataFrame:
    """
    Project window centroids into 2-D using UMAP.
    Returns window_df with x, y columns.
    """
    if not _UMAP_AVAILABLE:
        raise RuntimeError(
            "umap-learn is required.\n  pip install umap-learn"
        )

    centroids = np.vstack(window_df["centroid"].values)
    n_pts     = len(centroids)

    reducer = _umap_lib.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, n_pts - 1),
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(centroids)

    traj_cols = [c for c in
                 ["period_start", "post_count", "pct_high",
                  "drift_score", "is_drift_event", "drift_direction"]
                 if c in window_df.columns]
    traj = window_df[traj_cols].copy()
    traj["x"] = coords[:, 0]
    traj["y"] = coords[:, 1]

    return traj.reset_index(drop=True)


def annotate_trajectory(trajectory_df: pd.DataFrame,
                        rising_events: dict) -> pd.DataFrame:
    """
    Add human-readable annotation labels to each trajectory point.
    """
    df          = trajectory_df.copy()
    annotations = []

    for _, row in df.iterrows():
        period         = row.get("period_start")
        period_rising  = rising_events.get(period, [])

        if row.get("is_drift_event"):
            label = f"Narrative shift — {row.get('drift_direction', '')}"
        elif period_rising:
            top   = period_rising[0]
            label = f"Rising: {top['topic']} (+{top['pct_increase']}%)"
        elif row.get("pct_high", 0) > 0.4:
            label = f"High-risk peak: {row['pct_high']:.0%} of posts"
        else:
            label = f"{row.get('post_count', 0)} posts"

        annotations.append(label)

    df["annotation"] = annotations
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_narrative_evolution_analysis(posts_df: pd.DataFrame,
                                     embeddings: np.ndarray,
                                     cdc_df: pd.DataFrame | None = None,
                                     freq: str = "W") -> dict:
    """
    Full narrative evolution pipeline.

    Returns dict with all artifacts needed for the dashboard and presentation:
        window_df      — drift metrics per window
        topic_df       — topic prevalence time series
        rising_events  — flagged early-warning periods
        lag_results    — lead-lag vs CDC deaths
        warning_report — analyst-ready alert table
        trajectory_df  — UMAP coordinates + annotations
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Computing window centroids …")
    window_df = compute_window_centroids(posts_df, embeddings, freq=freq)
    print(f"      {len(window_df)} windows computed.")

    print("[2/6] Computing narrative drift …")
    window_df = compute_narrative_drift(window_df)
    drift_events = int(window_df["is_drift_event"].sum())
    print(f"      {drift_events} significant drift event(s) flagged.")

    print("[3/6] Computing topic prevalence …")
    topic_df = compute_topic_prevalence(posts_df, embeddings, freq=freq)

    print("      Detecting rising topics …")
    rising_events = detect_rising_topics(topic_df)
    print(f"      {len(rising_events)} period(s) with rising topics.")

    lag_results: dict = {}
    if cdc_df is not None:
        print("[4/6] Computing lead-lag correlations vs CDC deaths …")
        for topic in TOPIC_ANCHORS.keys():
            col = f"{topic}_score"
            if col in topic_df.columns:
                lag_results[topic] = compute_lead_lag_correlation(
                    topic_df, cdc_df, topic_col=col
                )
    else:
        print("[4/6] CDC data not provided — skipping lead-lag analysis.")

    print("[5/6] Building early-warning report …")
    warning_report = build_early_warning_report(rising_events, lag_results)
    if len(warning_report) > 0:
        print(f"      {len(warning_report)} alert row(s) generated.")
        crit = (warning_report["alert_level"] == "critical").sum()
        elev = (warning_report["alert_level"] == "elevated").sum()
        print(f"      critical={crit}  elevated={elev}")

    print("[6/6] Computing UMAP trajectory …")
    try:
        trajectory_df = compute_narrative_trajectory(window_df)
        trajectory_df = annotate_trajectory(trajectory_df, rising_events)
    except RuntimeError as exc:
        print(f"      ⚠  UMAP skipped: {exc}")
        trajectory_df = pd.DataFrame()

    # ── Save outputs ─────────────────────────────────────────────────────────
    _save_outputs(window_df, topic_df, rising_events, lag_results,
                  warning_report, trajectory_df)

    return {
        "window_df":      window_df,
        "topic_df":       topic_df,
        "rising_events":  rising_events,
        "lag_results":    lag_results,
        "warning_report": warning_report,
        "trajectory_df":  trajectory_df,
    }


def _save_outputs(window_df, topic_df, rising_events, lag_results,
                  warning_report, trajectory_df) -> None:
    """Persist all pipeline artifacts to data/processed/narrative/."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # window_df — serialize centroid / risk_centroids as bytes before parquet
    win_save = window_df.drop(columns=["centroid", "risk_centroids"],
                               errors="ignore").copy()
    win_save["substances"] = win_save["substances"].apply(json.dumps)
    pq.write_table(pa.Table.from_pandas(win_save), str(OUT_DIR / "window_df.parquet"))

    # topic_df
    pq.write_table(pa.Table.from_pandas(topic_df), str(OUT_DIR / "topic_df.parquet"))

    # trajectory_df
    if len(trajectory_df) > 0:
        trajectory_df.to_csv(OUT_DIR / "trajectory_df.csv", index=False)

    # warning_report
    warning_report.to_csv(OUT_DIR / "warning_report.csv", index=False)

    # rising_events — convert Timestamp keys → ISO strings
    re_serialisable = {
        str(k): v for k, v in rising_events.items()
    }
    with open(OUT_DIR / "rising_events.json", "w") as f:
        json.dump(re_serialisable, f, indent=2)

    # lag_results
    lag_serialisable = {
        topic: df.to_dict(orient="records") if len(df) > 0 else []
        for topic, df in lag_results.items()
    }
    with open(OUT_DIR / "lag_results.json", "w") as f:
        json.dump(lag_serialisable, f, indent=2)

    print(f"\n  Outputs saved → {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def _save_figure(fig, name: str) -> None:
    """Save a Plotly figure as HTML (always) and PNG (when kaleido present)."""
    html_path = FIG_DIR / f"{name}.html"
    fig.write_html(str(html_path))
    print(f"    Saved {html_path.name}")

    try:
        import plotly.io as pio
        # kaleido may be installed but missing its subprocess binary on some systems
        png_path = FIG_DIR / f"{name}.png"
        pio.write_image(fig, str(png_path), width=1200, height=700, scale=2)
        print(f"    Saved {png_path.name}")
    except Exception as exc:
        print(f"    ⚠  PNG export skipped ({exc})")


def visualize(results: dict) -> None:
    """
    Generate and save all four dashboard figures from pipeline results.

    Figures
    -------
    1. dual_axis_drift     — drift score + pct_high over time
    2. topic_prevalence    — stacked area of topic scores
    3. umap_trajectory     — scatter + path, colour = pct_high
    4. rising_alerts_table — HTML table coloured by alert_level
    """
    if not _PLOTLY_AVAILABLE:
        print("⚠  plotly not installed — skipping visualizations.")
        print("   pip install plotly")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    window_df     = results.get("window_df",     pd.DataFrame())
    topic_df      = results.get("topic_df",      pd.DataFrame())
    trajectory_df = results.get("trajectory_df", pd.DataFrame())
    warning_report= results.get("warning_report",pd.DataFrame())

    # ── Figure 1: Dual-axis drift + risk ──────────────────────────────────────
    if len(window_df) > 0 and "drift_score" in window_df.columns:
        print("  Generating dual-axis drift chart …")
        win = window_df.dropna(subset=["drift_score"]).copy()
        win["period_start"] = pd.to_datetime(win["period_start"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=win["period_start"], y=win["drift_score"],
            name="Narrative drift", line=dict(color="#E74C3C", width=2),
            mode="lines+markers",
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=win["period_start"], y=win["pct_high"],
            name="% high-risk posts", line=dict(color="#3498DB", width=2, dash="dot"),
            mode="lines"
        ), secondary_y=True)

        # Mark drift events
        drift_events = win[win["is_drift_event"] == True]  # noqa: E712
        if len(drift_events) > 0:
            fig.add_trace(go.Scatter(
                x=drift_events["period_start"], y=drift_events["drift_score"],
                mode="markers", name="Drift event",
                marker=dict(symbol="star", size=14, color="#F39C12"),
            ), secondary_y=False)

        fig.update_layout(
            title="Narrative Drift Score & High-Risk Post Percentage Over Time",
            xaxis_title="Period",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Cosine drift distance", secondary_y=False)
        fig.update_yaxes(title_text="% high-risk posts", secondary_y=True,
                         tickformat=".0%")
        _save_figure(fig, "dual_axis_drift")

    # ── Figure 2: Stacked area — topic prevalence ─────────────────────────────
    score_cols = [c for c in topic_df.columns if c.endswith("_score")]
    if len(topic_df) > 0 and score_cols:
        print("  Generating topic prevalence area chart …")
        tdf = topic_df.copy()
        tdf["period_start"] = pd.to_datetime(tdf["period_start"])
        tdf = tdf.sort_values("period_start")

        fig2 = go.Figure()
        colours = px.colors.qualitative.Set2
        for idx, col in enumerate(score_cols):
            topic_name = col.replace("_score", "").replace("_", " ").title()
            fig2.add_trace(go.Scatter(
                x=tdf["period_start"], y=tdf[col],
                name=topic_name,
                stackgroup="one",
                line=dict(color=colours[idx % len(colours)]),
                hovertemplate=f"{topic_name}: %{{y:.3f}}<extra></extra>",
            ))

        fig2.update_layout(
            title="Topic Prevalence Over Time (stacked area)",
            xaxis_title="Period", yaxis_title="Topic score (75th pctile cosine sim)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        _save_figure(fig2, "topic_prevalence_area")

    # ── Figure 3: UMAP trajectory ─────────────────────────────────────────────
    if len(trajectory_df) > 0 and "x" in trajectory_df.columns:
        print("  Generating UMAP trajectory scatter …")
        traj = trajectory_df.copy()
        traj["period_str"] = pd.to_datetime(
            traj["period_start"]
        ).dt.strftime("%Y-%m-%d")

        # Path lines
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=traj["x"], y=traj["y"],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # Points coloured by pct_high
        fig3.add_trace(go.Scatter(
            x=traj["x"], y=traj["y"],
            mode="markers+text",
            marker=dict(
                size=traj["post_count"].clip(5, 40)
                     if "post_count" in traj.columns
                     else 10,
                color=traj["pct_high"] if "pct_high" in traj.columns else 0,
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="% High Risk"),
                line=dict(width=1, color="white"),
            ),
            text=traj["period_str"],
            textposition="top center",
            textfont=dict(size=8),
            customdata=traj[["annotation", "post_count"]].values
                       if "annotation" in traj.columns else None,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "%{customdata[0]}<br>"
                "Posts: %{customdata[1]}<extra></extra>"
            ) if "annotation" in traj.columns else None,
            name="Narrative window",
        ))

        fig3.update_layout(
            title="UMAP Narrative Trajectory (weekly windows)",
            xaxis_title="UMAP-1", yaxis_title="UMAP-2",
            template="plotly_white",
        )
        _save_figure(fig3, "umap_trajectory")

    # ── Figure 4: Rising alerts table ─────────────────────────────────────────
    if len(warning_report) > 0:
        print("  Generating rising alerts table …")

        LEVEL_COLORS = {
            "critical": "#FF4136",
            "elevated": "#FF851B",
            "watch":    "#FFDC00",
        }

        wr = warning_report.copy()
        wr["period"] = pd.to_datetime(wr["period"]).dt.strftime("%Y-%m-%d")

        cell_colors = [
            [LEVEL_COLORS.get(lvl, "#FFFFFF") for lvl in wr["alert_level"]]
        ]

        col_order = ["period", "topic", "z_score", "pct_increase",
                     "alert_level", "lead_lag"]
        col_order = [c for c in col_order if c in wr.columns]

        fig4 = go.Figure(data=[go.Table(
            header=dict(
                values=[c.replace("_", " ").title() for c in col_order],
                fill_color="#2C3E50", font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[wr[c].tolist() for c in col_order],
                fill_color=(
                    ["white"] * (len(col_order) - 2)
                    + [cell_colors[0]]       # alert_level column
                    + ["white"]
                ) if "alert_level" in col_order else "white",
                align="left",
                font=dict(size=11),
            ),
        )])

        fig4.update_layout(
            title="Early Warning Alerts — Rising Topics",
            template="plotly_white",
        )
        _save_figure(fig4, "rising_alerts_table")

    print(f"\n  Figures saved → {FIG_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Narrative Evolution Analysis")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\nLoading analysis inputs …")
    posts_df, embeddings = load_analysis_inputs()
    print(f"  posts_df:  {posts_df.shape}")
    print(f"  embeddings:{embeddings.shape}")

    # ── CDC data (optional) ────────────────────────────────────────────────────
    cdc_df = None
    cdc_path = DATA / "raw" / "cdc_overdose_data.csv"
    if cdc_path.exists():
        print("\nLoading CDC overdose data …")
        cdc_df = load_cdc_overdose(cdc_path)
        print(f"  {len(cdc_df):,} CDC rows loaded.")
    else:
        print(f"\n  ⚠  CDC data not found at {cdc_path}")
        print("     Run data/fetch_cdc_data.py to download it.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    results = run_narrative_evolution_analysis(
        posts_df   = posts_df,
        embeddings = embeddings,
        cdc_df     = cdc_df,
        freq       = "W",
    )

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating visualizations …")
    visualize(results)

    print("\nDone ✓")
    print(f"  Outputs: {OUT_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
