"""
process_erowid.py
=================
Transform Erowid experience reports into a posts CSV compatible with
the project schema used by downstream pipeline steps.

Run:
    python src/processing/process_erowid.py

Input:
    data/raw/erowid-lsa-repo/experiences/<substance>/*.txt

Output:
    data/raw/erowid_posts.csv
    Columns:
      post_id, timestamp, substance, risk_level, review,
      drugName, rating, source
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import pandas as pd
from bs4 import BeautifulSoup

from preprocess_posts import scrub_pii
from process_drug_reviews import SUBSTANCE_MAP, assign_risk_label

ROOT = Path(__file__).parent.parent.parent
RAW = ROOT / "data" / "raw"
EXPERIENCES_DIR = RAW / "erowid-lsa-repo" / "experiences"
OUT_CSV = RAW / "erowid_posts.csv"

# Keep parity with the original Erowid-LSA filtering rule.
MIN_REPORTS_PER_SUBSTANCE = 30
YEAR_RE = re.compile(r"Year:\s*(\d{4})", re.IGNORECASE)

# Extra aliases commonly seen in Erowid directory names.
EROWID_ALIASES = {
    "heroin": "opioid",
    "methamphetamine": "stimulant",
}


def walk_experiences(base_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (substance_dir_name, file_path) for every .txt report."""
    if not base_dir.exists():
        return
    for substance_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for file_path in sorted(substance_dir.glob("*.txt")):
            yield substance_dir.name, file_path


def parse_experience(html_text: str) -> tuple[str, int | None]:
    """Strip HTML and extract report year from 'Year: YYYY' when present."""
    soup = BeautifulSoup(html_text, "lxml")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    year_match = YEAR_RE.search(text)
    year = int(year_match.group(1)) if year_match else None
    return text, year


def map_erowid_substance(dir_name: str) -> str:
    """
    Map Erowid directory name to canonical substance category.
    For combination labels, split on '&' and use the first entry.
    """
    primary = dir_name.split("&")[0].strip().lower()
    if primary in EROWID_ALIASES:
        return EROWID_ALIASES[primary]
    for key, category in SUBSTANCE_MAP.items():
        if key in primary:
            return category
    return "other"


def assign_risk_label_text_only(text: str) -> str:
    """
    Reuse process_drug_reviews.assign_risk_label with rating omitted from logic.
    Using NaN for rating makes rating-threshold checks evaluate false.
    """
    row = {"review": text, "rating": float("nan")}
    return assign_risk_label(row)


def _stable_post_id(drug_name: str, file_path: Path) -> str:
    """Deterministic post id based on source path."""
    digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "_", drug_name.lower()).strip("_")
    return f"erowid_{slug}_{digest}"


def process_erowid(
    base_dir: Path = EXPERIENCES_DIR,
    out_csv: Path = OUT_CSV,
    min_reports: int = MIN_REPORTS_PER_SUBSTANCE,
) -> pd.DataFrame:
    """Parse + scrub + map + classify Erowid reports into canonical CSV."""
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Erowid experiences directory not found: {base_dir}\n"
            "Run scripts/fetch_erowid.py first."
        )

    report_counts = {
        d.name: len(list(d.glob("*.txt")))
        for d in base_dir.iterdir()
        if d.is_dir()
    }
    keep_dirs = {name for name, n in report_counts.items() if n >= min_reports}
    skipped_dirs = len(report_counts) - len(keep_dirs)

    records: list[dict] = []
    for dir_name, file_path in walk_experiences(base_dir):
        if dir_name not in keep_dirs:
            continue
        try:
            html_text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if not html_text.strip():
            continue

        clean_text, year = parse_experience(html_text)
        clean_text = scrub_pii(clean_text)
        if not clean_text:
            continue

        timestamp = pd.Timestamp(year=year, month=1, day=1) if year else pd.NaT
        records.append(
            {
                "post_id": _stable_post_id(dir_name, file_path),
                "timestamp": timestamp,
                "substance": map_erowid_substance(dir_name),
                "risk_level": assign_risk_label_text_only(clean_text),
                "review": clean_text,
                "drugName": dir_name,
                "rating": None,
                "source": "erowid",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "post_id",
        "timestamp",
        "substance",
        "risk_level",
        "review",
        "drugName",
        "rating",
        "source",
    ]
    df = pd.DataFrame(records, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.to_csv(out_csv, index=False)

    print(f"Substance dirs found: {len(report_counts):,}")
    print(f"Substance dirs kept (>= {min_reports} reports): {len(keep_dirs):,}")
    print(f"Substance dirs skipped: {skipped_dirs:,}")
    print(f"Saved -> {out_csv} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    process_erowid()
