"""
process_erowid_lsa.py
=====================
Parse, transform, and save Erowid experience reports as a classified
posts CSV that is schema-compatible with posts_classified.csv (Kaggle).

Prerequisites:
    Run scripts/fetch_erowid.py first to clone the repo.
    data/raw/erowid-lsa-repo/experiences/  must exist.

Usage:
    python src/processing/process_erowid_lsa.py

Output:
    data/raw/erowid_posts.csv
    Columns: post_id, timestamp, substance, risk_level,
             review, drugName, rating, source

Pipeline:
    1. Walk all substance dirs (minimum MIN_REPORTS reports per substance)
    2. Parse HTML-snippet .txt files → clean text + year extraction
    3. Scrub PII (emails, phones, names via preprocess_posts.scrub_pii)
    4. Map substance dir name → canonical category (opioid/stimulant/etc.)
    5. Assign risk label using phrase matching from process_drug_reviews
    6. Save to data/raw/erowid_posts.csv
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pandas as pd

# ── resolve paths from project root ──────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
RAW         = ROOT / "data" / "raw"
EROWID_DIR  = RAW / "erowid-lsa-repo" / "experiences"
OUT_CSV     = RAW / "erowid_posts.csv"

# Only keep substance dirs that have at least this many reports.
# Set to 5 to accommodate partial scrapes; raise to 30 once a full
# 10,000-ID scrape is available for stronger signal.
MIN_REPORTS = 5

# ── canonical substance mapping ───────────────────────────────────────────────
# Erowid dir name → substance category (same categories as signal_pipeline.py)
EROWID_SUBSTANCE_MAP: dict[str, str] = {
    # Opioids
    "heroin": "opioid",
    "opioids": "opioid",
    "oxycodone": "opioid",
    "hydrocodone": "opioid",
    "oxycontin": "opioid",
    "vicodin": "opioid",
    "buprenorphine": "opioid",
    "suboxone": "opioid",
    "methadone": "opioid",
    "tramadol": "opioid",
    "codeine": "opioid",
    "fentanyl": "opioid",
    "morphine": "opioid",
    "opium": "opioid",
    "poppies": "opioid",
    "poppy": "opioid",
    "hydromorphone": "opioid",
    "tylenol 3": "opioid",
    "percocet": "opioid",
    "demerol": "opioid",
    # Stimulants
    "amphetamine": "stimulant",
    "amphetamines": "stimulant",
    "methamphetamine": "stimulant",
    "methaphetamine": "stimulant",
    "cocaine": "stimulant",
    "crack": "stimulant",
    "crack cocaine": "stimulant",
    "methylphenidate": "stimulant",
    "ritalin": "stimulant",
    "dextroamphetamine": "stimulant",
    "dexedrine": "stimulant",
    "adderall": "stimulant",
    "vyvanse": "stimulant",
    "ephedrine": "stimulant",
    "ephedra": "stimulant",
    "methcathinone": "stimulant",
    "clenbuterol hcl": "stimulant",
    # Alcohol
    "alcohol": "alcohol",
    "absinthe": "alcohol",
    # Benzodiazepines → mapped to "benzo" for local grouping
    "alprazolam": "benzo",
    "xanax": "benzo",
    "diazepam": "benzo",
    "valium": "benzo",
    "clonazepam": "benzo",
    "lorazepam": "benzo",
    "temazepam": "benzo",
    "rohypnol": "benzo",
    "benzodiazepines": "benzo",
    "diazepam (valium)": "benzo",
    "alprazolam (xanax)": "benzo",
    # GHB-class
    "ghb": "other",
    "gbl": "other",
    "1,4-butanediol": "other",
    # Dissociatives
    "ketamine": "other",
    "dxm": "other",
    "pcp": "other",
    # Cannabis
    "cannabis": "other",
    "marijuana": "other",
    "hash": "other",
    "hashish": "other",
}

# ── re-use risk machinery from process_drug_reviews (no circular imports) ────
HIGH_RISK_PHRASES = [
    "addicted", "addiction", "dependent", "dependence",
    "withdrawal", "withdrawals", "relapse", "relapsed",
    "can't stop", "cannot stop", "need more", "running out",
    "doctor shopping", "crushing", "snorting", "shooting up",
    "overdose", "od'd", "blacked out",
]

DISTRESS_PHRASES = [
    "anxiety", "depressed", "suicidal", "hopeless",
    "desperate", "scared", "terrified", "shaking",
    "sweating", "nausea", "vomiting", "craving",
]

# ── year extraction pattern ───────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(19[89]\d|20[012]\d)\b")  # 1980–2029
# PII removal patterns
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\b\d[\d\s\-\(\)]{7,}\d\b")
_ZIP_RE   = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def _parse_experience(raw_text: str) -> tuple[str, int | None]:
    """
    Strip HTML tags from the raw .txt file, extract the earliest
    plausible year, and return (clean_text, year_or_None).
    """
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw_text, "lxml").get_text(separator=" ")
    except Exception:
        # Fallback to simple tag stripping if BS4 unavailable
        text = re.sub(r"<[^>]+>", " ", raw_text)

    text = re.sub(r"\s+", " ", text).strip()

    years = _YEAR_RE.findall(text)
    year = int(years[0]) if years else None

    return text, year


def _scrub_pii(text: str) -> str:
    """Remove emails, phone numbers, and ZIP codes."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ZIP_RE.sub("[ZIP]", text)
    return text


def _map_substance(dir_name: str) -> str:
    """
    Map an Erowid substance directory name to a canonical category.
    For combination dirs (e.g. "cannabis & alcohol"), use the first drug.
    """
    primary = dir_name.split(" & ")[0].strip().lower()
    # direct lookup first
    if primary in EROWID_SUBSTANCE_MAP:
        return EROWID_SUBSTANCE_MAP[primary]
    # partial-match fallback
    for key, cat in EROWID_SUBSTANCE_MAP.items():
        if key in primary:
            return cat
    return "other"


def _assign_risk_label(text: str) -> str:
    """
    Text-only risk labeling (no rating field for Erowid reports).
    Mirrors the phrase-matching branch of process_drug_reviews.assign_risk_label.
    """
    lowered       = text.lower()
    high_hits     = sum(p in lowered for p in HIGH_RISK_PHRASES)
    distress_hits = sum(p in lowered for p in DISTRESS_PHRASES)

    if high_hits >= 1 or distress_hits >= 2:
        return "high"
    elif distress_hits >= 1:
        return "medium"
    else:
        return "low"


def process_erowid(
    experiences_dir: Path = EROWID_DIR,
    out_path: Path = OUT_CSV,
    min_reports: int = MIN_REPORTS,
) -> pd.DataFrame:
    """
    Full ETL for Erowid experience reports.
    Returns the resulting DataFrame and saves it to out_path.
    """
    if not experiences_dir.exists():
        raise FileNotFoundError(
            f"Erowid experiences directory not found: {experiences_dir}\n"
            "Run scripts/fetch_erowid.py first."
        )

    records: list[dict] = []
    substance_dirs = sorted(
        d for d in experiences_dir.iterdir() if d.is_dir()
    )

    print(f"Found {len(substance_dirs):,} substance directories.")
    skipped = 0

    for substance_dir in substance_dirs:
        txt_files = list(substance_dir.glob("*.txt"))
        if len(txt_files) < min_reports:
            skipped += 1
            continue

        dir_name  = substance_dir.name
        substance = _map_substance(dir_name)

        for txt_file in txt_files:
            try:
                raw = txt_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if not raw.strip():
                continue  # skip empty files (failed scrapes)

            clean_text, year = _parse_experience(raw)
            clean_text = _scrub_pii(clean_text)

            if len(clean_text.split()) < 20:
                continue  # skip trivially-short reports

            timestamp = pd.Timestamp(year=year, month=1, day=1) if year else pd.NaT

            records.append({
                "post_id":    str(uuid.uuid4()),
                "timestamp":  timestamp,
                "substance":  substance,
                "risk_level": _assign_risk_label(clean_text),
                "review":     clean_text[:4000],  # cap at 4 K chars
                "drugName":   dir_name,
                "rating":     None,
                "source":     "erowid",
            })

    print(f"Skipped {skipped:,} dirs with fewer than {min_reports} reports.")
    print(f"Parsed {len(records):,} experience reports.")

    if not records:
        print("⚠  No records produced. Check that the experiences dir is populated.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Risk distribution summary
    dist = df["risk_level"].value_counts()
    print(f"\nRisk label distribution:\n{dist.to_string()}")
    sub_dist = df["substance"].value_counts().head(10)
    print(f"\nTop substances:\n{sub_dist.to_string()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    process_erowid()
