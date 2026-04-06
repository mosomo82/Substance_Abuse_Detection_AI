"""
temporal_metrics.py
===================
Member 2 — Daniel Evans | Task 2: Temporal Analysis Metrics

Computes:
  1. MRR (Mean Reciprocal Rank) — for each CDC overdose spike month, checks
     how highly the relevant substance alert is ranked in the system's alert
     list for the corresponding social-signal period.
  2. Detection Lag — for each matched (CDC spike, social alert) pair, computes
     lag = date_alert_issued - date_spike_started in days. Reports mean and
     median detection lag per substance (negative = early warning).

Inputs:
  data/processed/cdc_cleaned.csv
  data/processed/narrative/warning_report.csv
  data/processed/correlations.json

Output:
  data/processed/temporal_metrics.json

Lag convention (from signal_pipeline.py::correlate_signals):
  Negative lag  →  social signal LEADS CDC deaths by |lag| months.
  e.g. opioid lag=-3 (r=0.471) means social spikes precede opioid deaths by 3 months.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent.parent
PROCESSED      = ROOT / "data" / "processed"
NARRATIVE      = PROCESSED / "narrative"

CDC_CSV        = PROCESSED / "cdc_cleaned.csv"
WARNING_CSV    = NARRATIVE / "warning_report.csv"
CORR_JSON      = PROCESSED / "correlations.json"
OUT_JSON       = PROCESSED / "temporal_metrics.json"

# ── Constants ─────────────────────────────────────────────────────────────────
# Map each CDC substance category to the behavioral topics in warning_report
# that are most indicative of that substance class.
SUBSTANCE_TOPIC_MAP: dict[str, list[str]] = {
    "opioid":   ["overdose", "withdrawal", "procurement", "relapse"],
    "cocaine":  ["procurement", "craving", "harm_reduction"],
    "stimulant":["craving", "procurement", "harm_reduction"],
    "alcohol":  ["relapse", "craving", "withdrawal"],
}

# z-score threshold above which a CDC national month is declared a spike
Z_SPIKE_THRESH: float = 1.5

# Minimum prior months required before rolling baseline is stable
MIN_ROLL_PERIODS: int = 6

# Window around each CDC spike to search for a matching social alert
SEARCH_WINDOW_PRE_MONTHS: int = 12   # months before CDC spike
SEARCH_WINDOW_POST_MONTHS: int = 3   # months after CDC spike (to catch lag)


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_cdc() -> pd.DataFrame:
    df = pd.read_csv(CDC_CSV, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    return df


def _load_warning_report() -> pd.DataFrame:
    df = pd.read_csv(WARNING_CSV, parse_dates=["period"])
    df["period"] = df["period"].dt.normalize()
    return df


def _load_correlations() -> dict:
    with open(CORR_JSON, encoding="utf-8") as f:
        return json.load(f)


# ── Step 1: Detect CDC spike months ──────────────────────────────────────────

def detect_cdc_spikes(cdc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CDC deaths nationally (sum across all states) per month and
    substance, then flag months where the rolling z-score exceeds Z_SPIKE_THRESH.

    Rolling baseline: a shifted window of MIN_ROLL_PERIODS prior months so that
    the current month's deaths never contaminate its own baseline.
    """
    national = (
        cdc_df
        .groupby(["date", "substance"], as_index=False)["deaths"]
        .sum()
        .sort_values(["substance", "date"])
    )

    spike_frames: list[pd.DataFrame] = []
    for substance, grp in national.groupby("substance"):
        grp = grp.reset_index(drop=True)
        deaths = grp["deaths"]

        # Shift by 1 so no look-ahead; require at least 2 points
        roll_mean = (
            deaths.shift(1)
            .rolling(window=MIN_ROLL_PERIODS, min_periods=2)
            .mean()
        )
        roll_std = (
            deaths.shift(1)
            .rolling(window=MIN_ROLL_PERIODS, min_periods=2)
            .std()
            .replace(0.0, np.nan)
        )

        z = (deaths - roll_mean) / roll_std
        spikes = grp[z > Z_SPIKE_THRESH].copy()
        spikes["z_score"] = z[z > Z_SPIKE_THRESH].values
        spike_frames.append(spikes)

    if not spike_frames:
        return pd.DataFrame(columns=["date", "substance", "deaths", "z_score"])
    return pd.concat(spike_frames, ignore_index=True)


# ── Step 2: Best social-signal lead (from correlations.json) ─────────────────

def _best_lead_months(corr: dict, substance: str) -> int:
    """
    Return the lag value with the strongest significant correlation where the
    social signal leads CDC deaths (negative lag in correlations convention).
    Returns 0 if no significant negative-lag correlation exists for the substance.
    """
    if substance not in corr:
        return 0

    best_lag, best_abs_r = 0, 0.0
    for lag_str, vals in corr[substance].items():
        lag = int(lag_str)
        r   = float(vals.get("r", 0.0))
        p   = float(vals.get("p", 1.0))
        # Negative lag = social leads CDC; pick the one with highest |r| and p<0.05
        if lag < 0 and p < 0.05 and abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_lag   = lag

    return best_lag   # e.g. -3 (social leads by 3 months); 0 = no lead found


# ── Step 3: Build period-keyed ranked alert list ──────────────────────────────

def _build_alert_index(warn_df: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    """
    For every period in warning_report, produce a list of topics sorted by
    descending z-score (rank 1 = highest z-score alert).
    """
    index: dict[pd.Timestamp, list[str]] = {}
    for period, grp in warn_df.groupby("period"):
        ts = pd.Timestamp(period).normalize()
        index[ts] = (
            grp.sort_values("z_score", ascending=False)["topic"].tolist()
        )
    return index


# ── Step 4: MRR ──────────────────────────────────────────────────────────────

def compute_mrr(
    spike_df: pd.DataFrame,
    alert_index: dict[pd.Timestamp, list[str]],
    corr: dict,
) -> tuple[float, dict[str, float], list[dict]]:
    """
    For each CDC spike event:
      1. Determine the social-signal period that should have preceded it
         (cdc_date + best_lead, where best_lead is negative).
      2. Look up the ranked alert list for that period.
      3. Find the rank of the first topic relevant to the substance.
      4. reciprocal_rank = 1/rank  (0 if no relevant alert exists that period).

    Returns
    -------
    mrr_overall   : float — mean across all events
    mrr_per_sub   : dict  — mean per substance
    details       : list of per-event dicts
    """
    details: list[dict] = []

    for _, row in spike_df.iterrows():
        substance = row["substance"]
        cdc_date  = pd.Timestamp(row["date"]).normalize()
        lead      = _best_lead_months(corr, substance)

        # The social period that (with the measured lead) corresponds to this CDC spike
        signal_ts = (
            cdc_date + pd.DateOffset(months=lead)
        ).to_period("M").to_timestamp()

        relevant_topics = SUBSTANCE_TOPIC_MAP.get(substance, [])
        period_topics   = alert_index.get(signal_ts, [])

        rank: int | None = None
        for i, topic in enumerate(period_topics, start=1):
            if topic in relevant_topics:
                rank = i
                break

        rr = 1.0 / rank if rank is not None else 0.0
        details.append({
            "substance":       substance,
            "cdc_date":        str(cdc_date.date()),
            "signal_period":   str(signal_ts.date()),
            "lead_months":     lead,
            "rank":            rank,
            "reciprocal_rank": round(rr, 4),
        })

    if not details:
        return 0.0, {}, []

    mrr_overall = round(float(np.mean([d["reciprocal_rank"] for d in details])), 4)

    per_sub: dict[str, float] = {}
    for substance in SUBSTANCE_TOPIC_MAP:
        sub_events = [d for d in details if d["substance"] == substance]
        if sub_events:
            per_sub[substance] = round(
                float(np.mean([d["reciprocal_rank"] for d in sub_events])), 4
            )

    return mrr_overall, per_sub, details


# ── Step 5: Detection Lag ─────────────────────────────────────────────────────

def compute_detection_lag(
    spike_df: pd.DataFrame,
    warn_df: pd.DataFrame,
) -> tuple[dict[str, dict], list[dict]]:
    """
    For each CDC spike month, find the *earliest* relevant social-signal alert
    within [cdc_date - SEARCH_WINDOW_PRE_MONTHS, cdc_date + SEARCH_WINDOW_POST_MONTHS].

    lag_days = alert_date - cdc_date
      negative → social alert issued BEFORE CDC spike (early warning — good)
      positive → social alert issued AFTER CDC spike (lagging detection)

    Returns
    -------
    per_substance : dict of aggregated stats per substance
    lag_details   : list of per-event dicts
    """
    per_substance: dict[str, dict] = {}
    lag_details:   list[dict]      = []

    for substance, sub_spikes in spike_df.groupby("substance"):
        relevant_topics = SUBSTANCE_TOPIC_MAP.get(substance, [])
        rel_alerts      = warn_df[warn_df["topic"].isin(relevant_topics)].copy()

        if rel_alerts.empty:
            continue

        lags: list[float] = []
        for _, cdc_row in sub_spikes.iterrows():
            cdc_date   = pd.Timestamp(cdc_row["date"]).normalize()
            window_lo  = cdc_date - pd.DateOffset(months=SEARCH_WINDOW_PRE_MONTHS)
            window_hi  = cdc_date + pd.DateOffset(months=SEARCH_WINDOW_POST_MONTHS)

            in_window = rel_alerts[
                (rel_alerts["period"] >= window_lo) &
                (rel_alerts["period"] <= window_hi)
            ]

            if in_window.empty:
                continue

            # Earliest alert within the window = date_alert_issued
            first = in_window.sort_values("period").iloc[0]
            lag   = (first["period"] - cdc_date).days

            lags.append(lag)
            lag_details.append({
                "substance":   substance,
                "cdc_date":    str(cdc_date.date()),
                "alert_date":  str(first["period"].date()),
                "alert_topic": first["topic"],
                "alert_level": str(first.get("alert_level", "")),
                "lag_days":    int(lag),
            })

        if lags:
            per_substance[substance] = {
                "mean_lag_days":    round(float(np.mean(lags)),   1),
                "median_lag_days":  round(float(np.median(lags)), 1),
                "min_lag_days":     int(np.min(lags)),
                "max_lag_days":     int(np.max(lags)),
                "n_matched_spikes": len(lags),
                "interpretation": (
                    "early warning — system alerts precede CDC spike"
                    if float(np.median(lags)) < 0
                    else "lagging — system alerts follow CDC spike"
                ),
            }

    return per_substance, lag_details


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 60)
    print("Temporal Metrics — Daniel Evans (M2)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    cdc_df   = _load_cdc()
    warn_df  = _load_warning_report()
    corr     = _load_correlations()
    print(f"      CDC rows:            {len(cdc_df):,}")
    print(f"      Warning report rows: {len(warn_df):,}")
    print(f"      Substances in corr:  {list(corr.keys())}")

    print(f"\n[2/5] Detecting CDC spike months (z > {Z_SPIKE_THRESH})...")
    spike_df = detect_cdc_spikes(cdc_df)
    print(f"      Spikes detected: {len(spike_df):,}")
    if not spike_df.empty:
        for sub, ct in spike_df["substance"].value_counts().items():
            print(f"        {sub}: {ct}")

    print("\n[3/5] Building ranked alert index...")
    alert_index = _build_alert_index(warn_df)
    print(f"      Periods with alerts: {len(alert_index):,}")

    print("\n[4/5] Computing MRR...")
    mrr_overall, mrr_per_sub, mrr_details = compute_mrr(spike_df, alert_index, corr)
    print(f"      Overall MRR = {mrr_overall:.4f}")
    for sub, val in mrr_per_sub.items():
        print(f"        {sub}: MRR = {val:.4f}")

    print("\n[5/5] Computing Detection Lag...")
    lag_per_sub, lag_details = compute_detection_lag(spike_df, warn_df)
    for sub, stats in lag_per_sub.items():
        print(
            f"      {sub}: mean={stats['mean_lag_days']:+.0f} d, "
            f"median={stats['median_lag_days']:+.0f} d  "
            f"({stats['interpretation']})"
        )

    output = {
        "mrr": {
            "overall":         mrr_overall,
            "per_substance":   mrr_per_sub,
            "n_cdc_spikes":    len(mrr_details),
            "details":         mrr_details,
        },
        "detection_lag": {
            "per_substance":   lag_per_sub,
            "details":         lag_details,
        },
        "metadata": {
            "z_spike_threshold":        Z_SPIKE_THRESH,
            "n_cdc_spikes_detected":    len(spike_df),
            "spike_counts_by_substance": (
                spike_df["substance"].value_counts().to_dict()
                if not spike_df.empty else {}
            ),
            "search_window_pre_months":  SEARCH_WINDOW_PRE_MONTHS,
            "search_window_post_months": SEARCH_WINDOW_POST_MONTHS,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved → {OUT_JSON}")
    return output


if __name__ == "__main__":
    run()
