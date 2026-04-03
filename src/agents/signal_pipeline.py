"""
signal_pipeline.py
==================
Steps 2-5 of the Substance Abuse Detection pipeline:

  Step 2 — Load & clean CDC data, align to a monthly datetime index
  Step 2b — Load NSDUH population rates + NIDA death rates as denominators
  Step 3 — Compute a social-signal time series from classifier output
            (uses data/raw/posts_classified.csv from the Kaggle drug reviews pipeline)
  Step 4 — Spike detection + cross-correlation (on NSDUH-normalized signal)
  Step 5 — ICD-10 → substance mapping reference (WONDER data helper)

Run:
    python signal_pipeline.py

Outputs:
    data/processed/cdc_cleaned.csv      – cleaned CDC data
    data/processed/social_signal.csv    – monthly post counts per substance
    data/processed/correlations.json    – lag-by-lag Pearson r / p for each substance
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent.parent   # project root
DATA          = ROOT / "data"
RAW_CDC       = DATA / "raw" / "cdc_overdose_data.csv"
RAW_NSDUH     = DATA / "raw" / "nsduh_prevalence.csv"
RAW_NIDA      = DATA / "raw" / "nida_rates.csv"
PROCESSED     = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load & clean CDC data
# ══════════════════════════════════════════════════════════════════════════════

# The CDC CSV stores month as a full name ("January", "February", …).
# pd.to_datetime handles this directly when we combine year + month name.
SUBSTANCE_MAP = {
    # exact indicator strings from the CDC dataset
    "Opioids (T40.1-T40.4,T40.6)":                                         "opioid",
    "Heroin (T40.1)":                                                        "opioid",
    "Synthetic opioids (T40.4)":                                             "opioid",
    "Natural & semi-synthetic opioids (T40.2)":                              "opioid",
    "Methadone (T40.3)":                                                     "opioid",
    "Natural & semi-synthetic opioids, incl. methadone (T40.2, T40.3)":     "opioid",
    "Natural, semi-synthetic, & synthetic opioids, incl. methadone (T40.2-T40.4)": "opioid",
    "Cocaine (T40.5)":                                                       "cocaine",
    "Psychostimulants with abuse potential (T43.6)":                         "stimulant",
    "Stimulants (T43.6)":                                                    "stimulant",
    "Alcohol-induced causes":                                                "alcohol",
}


def load_cdc_overdose(path: Path = RAW_CDC) -> pd.DataFrame:
    """
    Step 2 — load the local CDC CSV, build a proper datetime index,
    map indicator labels to substance categories, and return a clean frame.
    """
    df = pd.read_csv(path, low_memory=False)

    # Build datetime: year (int) + month (full name like "January")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + " " + df["month"].astype(str),
        format="%Y %B",          # %B = full month name
        errors="coerce",
    )

    df["substance"] = df["indicator"].map(SUBSTANCE_MAP)
    df = df.dropna(subset=["substance", "date"])

    # Use data_value first; fall back to predicted_value for provisional rows
    df["deaths"] = pd.to_numeric(df["data_value"], errors="coerce")
    df["deaths"] = df["deaths"].fillna(
        pd.to_numeric(df.get("predicted_value", np.nan), errors="coerce")
    )
    df = df.dropna(subset=["deaths"])

    clean = df[["date", "state", "substance", "deaths"]].copy()
    clean["deaths"] = clean["deaths"].astype(float)
    return clean


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Social signal time series
# ══════════════════════════════════════════════════════════════════════════════

def compute_social_signal(posts_df: pd.DataFrame,
                          freq: str = "M") -> pd.DataFrame:
    """
    Step 3 — convert a posts DataFrame into a time-binned signal.

    posts_df expected columns:
        timestamp   – datetime
        substance   – one of opioid / cocaine / stimulant / alcohol
        risk_level  – low / medium / high

    freq : "M" for monthly (default), "W" for weekly.
        Monthly is better for CDC cross-correlation (CDC data is monthly).
        Weekly provides finer spike detection within the social signal.
    """
    posts_df = posts_df.copy()
    if freq == "W":
        posts_df["date"] = posts_df["timestamp"].dt.to_period("W").dt.to_timestamp()
    else:
        posts_df["date"] = posts_df["timestamp"].dt.to_period("M").dt.to_timestamp()

    signal = (
        posts_df[posts_df["risk_level"].isin(["medium", "high"])]
        .groupby(["date", "substance"])
        .size()
        .reset_index(name="post_count")
    )
    return signal


# Paths to classified posts
POSTS_PATH        = DATA / "raw" / "posts_classified.csv"
EROWID_POSTS_PATH = DATA / "raw" / "erowid_posts.csv"

# Shared columns used when merging sources
_MERGE_COLS = ["post_id", "timestamp", "substance", "risk_level", "review",
               "drugName", "rating", "source"]


def load_erowid_posts(path: Path = EROWID_POSTS_PATH) -> pd.DataFrame | None:
    """
    Load the Erowid classified posts CSV produced by process_erowid.py.
    Returns None (with a warning) if the file does not exist.
    """
    if not path.exists():
        print(f"  ⚠  Erowid posts not found at {path}")
        print("     Run src/processing/process_erowid_lsa.py to generate them.")
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Ensure required columns are present
    for col in ["timestamp", "substance", "risk_level", "review"]:
        if col not in df.columns:
            print(f"  ⚠  Erowid CSV missing column '{col}' — skipping.")
            return None

    if "source" not in df.columns:
        df["source"] = "erowid"

    df = df.dropna(subset=["substance"])
    print(f"  Erowid posts loaded: {len(df):,} rows  "
          f"(with timestamp: {df['timestamp'].notna().sum():,})")
    return df


def merge_post_sources(
    drug_reviews_df: pd.DataFrame | None,
    erowid_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Concatenate drug-review and Erowid posts into a single DataFrame.
    Aligns on shared columns; adds a 'source' tag for attribution.
    At least one source must be non-None.
    """
    frames = []

    if drug_reviews_df is not None and not drug_reviews_df.empty:
        df = drug_reviews_df.copy()
        if "source" not in df.columns:
            df["source"] = "drug_review"
        frames.append(df)

    if erowid_df is not None and not erowid_df.empty:
        frames.append(erowid_df.copy())

    if not frames:
        raise ValueError("No post sources available. Provide at least one dataset.")

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    print(f"  Merged posts: {len(merged):,} total  "
          f"({merged['source'].value_counts().to_dict()})")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — Load NSDUH + NIDA denominators
# ══════════════════════════════════════════════════════════════════════════════

def load_denominators() -> tuple[dict, dict]:
    """
    Load NSDUH disorder prevalence rates and NIDA death rates.

    Returns
    -------
    nsduh_rates : dict  {substance: avg_population_rate_pct (float)}
        Average past-year disorder prevalence % (persons 12+) across available years.
        Used to normalize post_count into population-adjusted units.

    nida_rates : dict  {substance: {year: deaths_per_100k}}
        Secondary reference; attached to output for reporting.
    """
    nsduh, nida = {}, {}

    if RAW_NSDUH.exists():
        df = pd.read_csv(RAW_NSDUH)
        # Use the 3 most recent years to compute a stable average
        recent = df.sort_values("year").groupby("substance").tail(3)
        nsduh = recent.groupby("substance")["population_rate_pct"].mean().round(4).to_dict()
        print(f"  NSDUH loaded: {len(nsduh)} substance(s): {list(nsduh.keys())}")
    else:
        print(f"  ⚠  NSDUH file not found at {RAW_NSDUH}")
        print("     Run fetch_nsduh.py first, or normalization will be skipped.")

    if RAW_NIDA.exists():
        df = pd.read_csv(RAW_NIDA)
        nida = (
            df.groupby(["substance", "year"])["deaths_per_100k"]
            .mean().reset_index()
            .pivot(index="year", columns="substance", values="deaths_per_100k")
            .to_dict()
        )
        print(f"  NIDA loaded: {list(nida.keys())} substances")
    else:
        print(f"  ⚠  NIDA file not found at {RAW_NIDA}")
        print("     Run fetch_nida_summary.py first.")

    return nsduh, nida


def normalize_signal(signal_df: pd.DataFrame,
                     nsduh_rates: dict) -> pd.DataFrame:
    """
    Normalize raw post_count by NSDUH disorder prevalence rate.

    Formula:
        normalized_count = post_count / population_rate_pct * 100

    Interpretation: "posts per 1 % of the population with that disorder".
    This makes the opioid signal (low base rate ~2%) comparable to the
    alcohol signal (high base rate ~10%) on the same scale.

    If a substance has no NSDUH rate, raw post_count is preserved.
    """
    df = signal_df.copy()
    df["population_rate_pct"] = df["substance"].map(nsduh_rates)
    mask = df["population_rate_pct"].notna() & (df["population_rate_pct"] > 0)
    df["normalized_count"] = df["post_count"].copy().astype(float)
    df.loc[mask, "normalized_count"] = (
        df.loc[mask, "post_count"] / df.loc[mask, "population_rate_pct"] * 100
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Spike detection + cross-correlation
# ══════════════════════════════════════════════════════════════════════════════

def detect_spikes(series: pd.Series, window: int = 3,
                  threshold: float = 2.0) -> pd.Series:
    """
    Return a boolean Series marking months where the rolling z-score
    exceeds `threshold` standard deviations.
    """
    rolling_mean = series.rolling(window, center=True, min_periods=1).mean()
    rolling_std  = series.rolling(window, center=True, min_periods=1).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z > threshold


def correlate_signals(social_series: pd.Series, cdc_series: pd.Series,
                      max_lag: int = 3) -> dict:
    """
    Step 4 — compute Pearson cross-correlation between the social signal
    and CDC deaths at lags −max_lag … +max_lag.

    Convention: positive lag means social signal LEADS CDC deaths.
    Returns dict  lag (int)  →  {'r': float, 'p': float}
    """
    combined = pd.concat(
        [social_series.rename("social"), cdc_series.rename("cdc")], axis=1
    ).dropna()

    if len(combined) < 5:
        return {}   # not enough data

    social = combined["social"]
    cdc_   = combined["cdc"]

    results = {}
    for lag in range(-max_lag, max_lag + 1):
        # positive lag → shift CDC forward → social leads
        shifted = cdc_.shift(lag)
        mask = shifted.notna()
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(social[mask], shifted[mask])
        results[lag] = {"r": round(float(r), 3), "p": round(float(p), 4)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — Erowid spillover detection
# ══════════════════════════════════════════════════════════════════════════════
# Uses the NMF substance similarity graph produced by process_erowid_lsa.py to
# detect "spillover" patterns: when one substance spikes, do its semantically
# related peers (per Erowid's experience corpus) also spike within ±35 days?
# This reveals polysubstance and substitution dynamics invisible in CDC data.
# Both functions degrade gracefully when erowid_substance_similarity.json is absent.

EROWID_SIM_PATH = PROCESSED / "erowid_substance_similarity.json"


def load_erowid_similarity_graph(
    sim_path: Path = EROWID_SIM_PATH,
    threshold: float = 0.70,
) -> dict[str, list[str]]:
    """
    Load Erowid NMF substance similarity graph.

    Returns {substance: [related_substance, ...]} for all pairs whose cosine
    similarity in NMF topic space exceeds `threshold`.

    Returns an empty dict if the file is missing (graceful degradation).
    """
    if not sim_path.exists():
        return {}
    try:
        with open(sim_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    graph: dict[str, list[str]] = {}
    for sub, peers in data.get("similarity_matrix", {}).items():
        related = [p for p, score in peers.items() if score >= threshold]
        if related:
            graph[sub] = related
    return graph


def detect_spillover_spikes(
    signal_df: pd.DataFrame,
    similarity_graph: dict[str, list[str]],
    spike_window: int = 3,
    spike_threshold: float = 2.0,
    co_occurrence_days: int = 35,
) -> pd.DataFrame:
    """
    For each detected spike event, check whether NMF-related substances
    (per Erowid similarity graph) show elevated activity within ±co_occurrence_days.

    A "spillover" flag is set when one or more semantically related substances
    also spike around the same date — a pattern consistent with polysubstance
    use or drug-market substitution (e.g., heroin → fentanyl shift).

    Parameters
    ----------
    signal_df          : output of compute_social_signal() (date, substance, post_count)
    similarity_graph   : output of load_erowid_similarity_graph()
    spike_window       : rolling window for z-score spike detection
    spike_threshold    : z-score threshold to declare a spike
    co_occurrence_days : ±day window for spillover co-occurrence check

    Returns
    -------
    DataFrame with columns:
        date, substance, related_substances_spiking (JSON list), spillover_flag,
        erowid_signal (True — marks this as an Erowid-derived finding)

    Returns an empty DataFrame if similarity_graph is empty or no spikes detected.
    """
    if not similarity_graph or signal_df.empty:
        return pd.DataFrame(columns=[
            "date", "substance", "related_substances_spiking",
            "spillover_flag", "erowid_signal",
        ])

    # ── Detect spike dates per substance ─────────────────────────────────────
    spike_dates: dict[str, set] = {}
    for substance in signal_df["substance"].unique():
        series = (
            signal_df[signal_df["substance"] == substance]
            .set_index("date")["post_count"]
            .sort_index()
        )
        spike_mask = detect_spikes(series, window=spike_window,
                                   threshold=spike_threshold)
        spike_set  = set(series.index[spike_mask])
        if spike_set:
            spike_dates[substance] = spike_set

    # ── Check for spillover co-occurrence ─────────────────────────────────────
    rows = []
    delta = pd.Timedelta(days=co_occurrence_days)

    for substance, s_dates in spike_dates.items():
        related = similarity_graph.get(substance, [])
        for date in s_dates:
            related_spiking = [
                r for r in related
                if any(abs(date - d) <= delta for d in spike_dates.get(r, set()))
            ]
            rows.append({
                "date":                       date,
                "substance":                  substance,
                "related_substances_spiking": json.dumps(related_spiking),
                "spillover_flag":             len(related_spiking) > 0,
                "erowid_signal":              True,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "date", "substance", "related_substances_spiking",
        "spillover_flag", "erowid_signal",
    ])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ICD-10 → substance mapping (for WONDER / raw death data)
# ══════════════════════════════════════════════════════════════════════════════

ICD10_MAP = {
    "T40.0": "opioid",    # Opium
    "T40.1": "opioid",    # Heroin
    "T40.2": "opioid",    # Natural/semisynthetic opioids (oxycodone, hydrocodone)
    "T40.3": "opioid",    # Methadone
    "T40.4": "opioid",    # Synthetic opioids (fentanyl, tramadol)
    "T40.5": "cocaine",
    "T40.6": "opioid",    # Other/unspecified narcotics
    "T43.6": "stimulant", # Psychostimulants (meth, Adderall)
    "T51.0": "alcohol",   # Ethanol
    "T51.1": "alcohol",   # Methanol
    "T51.9": "alcohol",   # Unspecified alcohol
}


def map_icd10_to_substance(icd_code: str) -> str | None:
    """Return substance category for an ICD-10 T-code, or None if unknown."""
    return ICD10_MAP.get(icd_code.strip())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Frequency setting: "M" for monthly, "W" for weekly ────────────────────
    # Monthly aligns with CDC data (recommended for cross-correlation).
    # Weekly gives finer spike detection on the social signal itself.
    SIGNAL_FREQ = "M"

    # ── Step 2: load & clean CDC data ─────────────────────────────────────────
    print("Step 2 — Loading CDC data …")
    cdc = load_cdc_overdose()
    print(f"  Rows after cleaning: {len(cdc):,}")
    print(f"  Date range: {cdc['date'].min().date()} → {cdc['date'].max().date()}")
    print(f"  Substances: {sorted(cdc['substance'].unique())}")
    cdc.to_csv(PROCESSED / "cdc_cleaned.csv", index=False)
    print(f"  Saved → {PROCESSED / 'cdc_cleaned.csv'}\n")

    # ── Step 2b: load NSDUH + NIDA denominators ────────────────────────────────
    print("Step 2b — Loading population denominators …")
    nsduh_rates, nida_rates = load_denominators()
    print()

    # ── Step 3: social signal ────────────────────────────────────────────────
    print("Step 3 — Social signal …")

    # Load drug-review posts (Kaggle)
    drug_reviews_df = None
    if POSTS_PATH.exists():
        drug_reviews_df = pd.read_csv(POSTS_PATH, parse_dates=["timestamp"])
        if "source" not in drug_reviews_df.columns:
            drug_reviews_df["source"] = "drug_review"
        print(f"  Drug-review posts loaded: {len(drug_reviews_df):,} rows")
    else:
        print(f"  ⚠  No drug-review posts found at: {POSTS_PATH}")
        print("     Run src/processing/process_drug_reviews.py to generate them.")

    # Load Erowid posts (optional enrichment)
    print("  Loading Erowid posts …")
    erowid_df = load_erowid_posts()

    # Require at least one source
    if drug_reviews_df is None and erowid_df is None:
        print("\nSteps 4–5 skipped — no post sources available. CDC data saved.")
        return

    # Merge both sources
    print("  Merging post sources …")
    posts_df  = merge_post_sources(drug_reviews_df, erowid_df)
    signal_df = compute_social_signal(posts_df, freq=SIGNAL_FREQ)
    print(f"  Signal rows: {len(signal_df):,}  (freq={SIGNAL_FREQ})")
    signal_out = PROCESSED / f"social_signal_{SIGNAL_FREQ.lower()}.csv"
    signal_df.to_csv(signal_out, index=False)
    print(f"  Saved → {signal_out}\n")

    # Normalize by NSDUH population rate (if available)
    if nsduh_rates:
        signal_df = normalize_signal(signal_df, nsduh_rates)
        print("  Signal normalized by NSDUH disorder prevalence rates.")
        signal_val_col = "normalized_count"
    else:
        signal_val_col = "post_count"
        print("  ⚠  No NSDUH rates — using raw post_count for correlation.")
    print()

    # ── Step 4: spike detection + cross-correlation ───────────────────────────
    print("Step 4 — Cross-correlation analysis (social signal vs CDC deaths) …")
    all_correlations = {}

    # Weekly needs a larger rolling window (12 weeks ≈ 3 months)
    spike_window = 12 if SIGNAL_FREQ == "W" else 3
    freq_alias   = "W-SUN" if SIGNAL_FREQ == "W" else "MS"

    for substance in sorted(cdc["substance"].unique()):
        # CDC is always monthly regardless of SIGNAL_FREQ
        cdc_series = (
            cdc[cdc["substance"] == substance]
            .groupby("date")["deaths"].sum()
            .asfreq("MS")
        )

        sig_subset = signal_df[signal_df["substance"] == substance]
        if sig_subset.empty:
            continue

        # Fill sparse periods with 0 (important for weekly — many zero-post weeks)
        social_series = (
            sig_subset.set_index("date")[signal_val_col]
            .asfreq(freq_alias, fill_value=0)
        )

        # Detect spikes on each signal
        cdc_spikes    = detect_spikes(cdc_series.dropna(), window=spike_window)
        social_spikes = detect_spikes(social_series, window=spike_window)
        print(f"\n  [{substance}]")
        print(f"    CDC spike months    : {cdc_spikes.sum()}")
        print(f"    Social spike periods: {social_spikes.sum()}")

        # Cross-correlation: resample weekly signal to monthly first (CDC is monthly)
        if SIGNAL_FREQ == "W":
            social_for_corr = social_series.resample("MS").sum()
        else:
            social_for_corr = social_series

        corr = correlate_signals(social_for_corr, cdc_series, max_lag=3)
        all_correlations[substance] = corr

        best_lag = max(corr, key=lambda k: corr[k]["r"]) if corr else "n/a"
        best_r   = corr[best_lag]["r"] if corr else "n/a"
        print(f"    Best lag (highest r): {best_lag} months  →  r = {best_r}")
        if isinstance(best_lag, int) and best_lag > 0:
            print(f"    ✓ Social signal leads CDC deaths by {best_lag} month(s) "
                  f"— candidate early-warning indicator!")

    # Save full results
    out_path = PROCESSED / "correlations.json"
    with open(out_path, "w") as f:
        json.dump(all_correlations, f, indent=2)
    print(f"\n  Cross-correlation results saved → {out_path}")

    # ── Step 4b: Erowid spillover analysis ───────────────────────────────────
    print("\nStep 4b — Erowid spillover analysis …")
    sim_graph = load_erowid_similarity_graph()
    if not sim_graph:
        print("  Erowid similarity graph not found — skipping spillover analysis.")
        print("  Run 'python scripts/process_erowid_lsa.py' to generate it.")
    else:
        print(f"  Similarity graph loaded: {len(sim_graph)} substance(s) with peers")
        spillover_df = detect_spillover_spikes(signal_df, sim_graph)
        if spillover_df.empty:
            print("  No spillover events detected.")
        else:
            flagged   = spillover_df[spillover_df["spillover_flag"]]
            out_spill = PROCESSED / "erowid_spillover.csv"
            spillover_df.to_csv(out_spill, index=False)
            print(f"  Spillover events detected : {len(flagged):,}")
            print(f"  Total spike rows saved    : {len(spillover_df):,}")
            print(f"  Saved → {out_spill}")

    # ── Step 5: ICD-10 mapping demo ───────────────────────────────────────────
    print("\nStep 5 — ICD-10 mapping reference (sample):")
    for code in ["T40.1", "T40.4", "T40.5", "T43.6", "T51.0"]:
        print(f"  {code} → {map_icd10_to_substance(code)}")

    print("\nPipeline complete ✓")


if __name__ == "__main__":
    main()
