"""
preprocess_posts.py
===================
6-step preprocessing pipeline for social media / forum post text.

Pipeline order (never swap steps 2 and 3):
    raw text → clean → scrub PII → normalize slang → extract signals → classify

Steps:
  1 — Basic cleaning (HTML, URLs, mentions, whitespace)
  2 — PII scrubbing (regex + optional spaCy NER)
  3 — Slang normalization (substance street names → canonical drug terms)
  4 — Distress signal extraction
  5 — Full pipeline (chains steps 1–4 into a structured record)
  6 — QA report

When run as __main__, processes data/raw/posts_classified.csv and saves:
    data/processed/posts_preprocessed.csv

Usage:
    python preprocess_posts.py

Or import and use in another script:
    from preprocess_posts import preprocess_corpus
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent.parent.parent   # project root
DATA      = ROOT / "data"
IN_CSV    = DATA / "raw" / "posts_classified.csv"
OUT_CSV   = DATA / "processed" / "posts_preprocessed.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Try loading spaCy once at import time ──────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "tok2vec"])
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Basic cleaning
# ══════════════════════════════════════════════════════════════════════════════

def basic_clean(text: str) -> str:
    """Remove HTML entities, URLs, @mentions, # prefix, tags, extra whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"http\S+|www\.\S+", "", text)      # URLs
    text = re.sub(r"@\w+", "", text)                   # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)              # strip # keep word
    text = re.sub(r"<[^>]+>", "", text)                # HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PII scrubbing
# ══════════════════════════════════════════════════════════════════════════════

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_ZIP_RE   = re.compile(r"\b\d{5}(?:-\d{4})?\b")

_NER_LABELS = {"PERSON", "GPE", "LOC", "FAC"}


def scrub_pii(text: str) -> str:
    """
    Remove emails, phone numbers, ZIP codes (regex),
    and names/locations (spaCy NER if available).
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ZIP_RE.sub("[ZIP]", text)

    if _SPACY_AVAILABLE:
        doc = _nlp(text)
        # Iterate reversed to preserve character offsets
        for ent in reversed(doc.ents):
            if ent.label_ in _NER_LABELS:
                text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]

    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Slang normalization
# ══════════════════════════════════════════════════════════════════════════════

SLANG_LEXICON: dict[str, str] = {
    # Opioids
    "oxy":          "oxycodone",
    "oxys":         "oxycodone",
    "percs":        "percocet",
    "perc":         "percocet",
    "vikes":        "vicodin",
    "vics":         "vicodin",
    "subs":         "suboxone",
    "subbies":      "suboxone",
    "done":         "methadone",
    "dones":        "methadone",
    "fent":         "fentanyl",
    "fetty":        "fentanyl",
    "china white":  "fentanyl",
    "h":            "heroin",
    "boy":          "heroin",
    "black tar":    "heroin",
    "dope":         "heroin",
    "smack":        "heroin",
    "junk":         "heroin",
    # Benzodiazepines
    "bars":         "xanax",
    "xannies":      "xanax",
    "benzos":       "benzodiazepine",
    "k-pins":       "klonopin",
    "pins":         "klonopin",
    # Stimulants
    "meth":         "methamphetamine",
    "ice":          "methamphetamine",
    "crystal":      "methamphetamine",
    "tweak":        "methamphetamine",
    "addys":        "adderall",
    "addy":         "adderall",
    "coke":         "cocaine",
    "blow":         "cocaine",
    "crack":        "cocaine",
    "snow":         "cocaine",
    # Alcohol
    "booze":        "alcohol",
    "liquor":       "alcohol",
    "sauce":        "alcohol",
    # Cannabis (lower risk, tracked for context)
    "weed":         "cannabis",
    "pot":          "cannabis",
    "bud":          "cannabis",
    "mary jane":    "cannabis",
    # Behavioral signals
    "nodding":      "opioid intoxication",
    "fiending":     "craving",
    "jonesing":     "withdrawal",
    "dope sick":    "withdrawal",
}

# Pre-compile patterns sorted longest-first (multi-word slang gets priority)
_SLANG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(slang) + r"\b"), canonical)
    for slang, canonical in sorted(SLANG_LEXICON.items(), key=lambda x: -len(x[0]))
]


def normalize_slang(text: str) -> str:
    """Replace street names with canonical drug terms (case-insensitive)."""
    result = text.lower()
    for pattern, canonical in _SLANG_PATTERNS:
        result = pattern.sub(canonical, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Distress signal extraction
# ══════════════════════════════════════════════════════════════════════════════

DISTRESS_SIGNALS: dict[str, list[str]] = {
    "relapse": [
        "relapsed", "relapsing", "fell off", "slipped", "using again",
        "back on", "couldn't stay clean",
    ],
    "craving": [
        "fiending", "jonesing", "craving", "need it", "gotta have",
        "can't stop thinking about",
    ],
    "withdrawal": [
        "dope sick", "withdrawals", "withdrawing", "kicking", "sick",
        "sweating", "shaking", "can't sleep", "vomiting",
    ],
    "desperation": [
        "running out", "out of", "need more", "last one",
        "scraping", "desperate", "please help",
    ],
    "overdose": [
        "od'd", "overdosed", "overdose", "too much", "almost died",
        "narcan", "naloxone", "revived",
    ],
    "hopelessness": [
        "can't quit", "never gonna stop", "hopeless", "gave up",
        "no point", "done trying",
    ],
}


def extract_distress_signals(text: str) -> dict[str, list[str]]:
    """
    Return {signal_type: [matched_phrases]} for all distress signals found.
    Empty dict → no distress signals detected.
    """
    text_lower = text.lower()
    return {
        sig: [p for p in phrases if p in text_lower]
        for sig, phrases in DISTRESS_SIGNALS.items()
        if any(p in text_lower for p in phrases)
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

_CANONICAL_SUBSTANCES = set(SLANG_LEXICON.values())


def preprocess_post(raw_text: str,
                    post_id=None,
                    timestamp=None) -> dict:
    """
    Full preprocessing pipeline for a single post.
    Returns a structured dict ready for the classifier.
    """
    cleaned    = basic_clean(raw_text)          # Step 1
    scrubbed   = scrub_pii(cleaned)             # Step 2 — PII before slang
    normalized = normalize_slang(scrubbed)      # Step 3 — slang after PII
    distress   = extract_distress_signals(normalized)  # Step 4

    substances = list({
        canonical
        for canonical in _CANONICAL_SUBSTANCES
        if re.search(r"\b" + re.escape(canonical) + r"\b", normalized)
    })

    return {
        "post_id":               post_id,
        "timestamp":             timestamp,
        "original_text":         raw_text,      # kept for audit trail
        "processed_text":        normalized,    # fed to classifier
        "substances_detected":   substances,
        "distress_signals":      distress,
        "distress_count":        len(distress),
        "has_substance_mention": len(substances) > 0,
        "has_distress_signal":   len(distress) > 0,
    }


def preprocess_corpus(df: pd.DataFrame,
                      text_col:  str = "review",
                      id_col:    str = "post_id",
                      time_col:  str = "timestamp") -> pd.DataFrame:
    """
    Apply the full pipeline to every row in df.
    Deduplicates on raw text before processing.
    Returns a new DataFrame of structured records.
    """
    # Step 0 — Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Deduplication: {removed:,} duplicate rows removed "
              f"({before:,} -> {len(df):,})")

    records = [
        preprocess_post(
            raw_text  = row[text_col],
            post_id   = row.get(id_col),
            timestamp = row.get(time_col),
        )
        for _, row in df.iterrows()
    ]
    result = pd.DataFrame(records)

    # Serialize list/dict columns so they survive to_csv
    result["substances_detected"] = result["substances_detected"].apply(json.dumps)
    result["distress_signals"]    = result["distress_signals"].apply(json.dumps)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Quality assurance report
# ══════════════════════════════════════════════════════════════════════════════

def preprocessing_qa(processed_df: pd.DataFrame) -> None:
    """Print a QA report after preprocessing."""
    total = len(processed_df)
    has_sub  = processed_df["has_substance_mention"].sum()
    has_dist = processed_df["has_distress_signal"].sum()
    both     = (processed_df["has_substance_mention"] &
                processed_df["has_distress_signal"]).sum()
    empty    = (processed_df["processed_text"].str.len() < 10).sum()

    print(f"\n── Preprocessing QA ─────────────────────────")
    print(f"  Total posts              : {total:,}")
    print(f"  Has substance mention    : {has_sub:,} ({has_sub/total:.1%})")
    print(f"  Has distress signal      : {has_dist:,} ({has_dist/total:.1%})")
    print(f"  Substance + distress     : {both:,} ({both/total:.1%})")
    print(f"  Empty after processing   : {empty:,}")

    # Deserialize for counting if stored as JSON strings
    def _deserialize_list(col):
        if col.dtype == object and col.iloc[0] and isinstance(col.iloc[0], str):
            return col.apply(json.loads)
        return col

    substances_col = _deserialize_list(processed_df["substances_detected"])
    distress_col   = _deserialize_list(processed_df["distress_signals"])

    all_substances = [s for lst in substances_col for s in lst]
    all_distress   = [k for d in distress_col for k in (d.keys() if isinstance(d, dict) else [])]

    print("\n  Top substances detected:")
    for substance, count in Counter(all_substances).most_common(10):
        print(f"    {substance:<28} {count:,}")

    print("\n  Top distress signal types:")
    for signal, count in Counter(all_distress).most_common():
        print(f"    {signal:<28} {count:,}")
    print("─────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not _SPACY_AVAILABLE:
        print("⚠  spaCy not found — PII NER skipped (regex scrubbing still active).")
        print("   To enable full PII scrubbing:")
        print("   pip install spacy && python -m spacy download en_core_web_sm\n")

    if not IN_CSV.exists():
        print(f"⚠  Input not found: {IN_CSV}")
        print("   Run process_drug_reviews.py first.")
        return

    print(f"Loading posts from {IN_CSV} …")
    df = pd.read_csv(IN_CSV)
    print(f"  {len(df):,} rows loaded\n")

    print("Preprocessing …")
    text_col = "review" if "review" in df.columns else "text"
    processed = preprocess_corpus(df, text_col=text_col, time_col="timestamp")

    # Step 6: QA
    preprocessing_qa(processed)

    processed.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(processed):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
