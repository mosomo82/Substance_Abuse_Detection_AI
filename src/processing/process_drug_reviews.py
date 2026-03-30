"""
process_drug_reviews.py
=======================
Processes the KUC Hackathon drug review corpus (Drugs.com) through 5 steps:

  Step 1 — Load + filter to abuse-relevant records
  Step 2 — Map drug names to substance categories
  Step 3 — Assign heuristic risk labels (high / medium / low)
  Step 4 — Build embedding-based seed bank for forum post scoring
  Step 5 — Mine slang / entity lexicon candidates

Primary output:
    data/raw/posts_classified.csv
    Columns: timestamp, substance, risk_level, review, drugName, rating
    → Plugs directly into signal_pipeline.py (Step 3 placeholder)

Secondary output:
    data/raw/seed_bank.pkl
    → Pickled dict {substance: [(text, embedding), …]} for forum post scoring

Prerequisites:
    data/raw/drugsComTrain_raw.csv
    data/raw/drugsComTest_raw.csv
    (Download from Kaggle: KUC Hackathon / UCI ML Drug Review Dataset)

Run:
    python process_drug_reviews.py
"""

from __future__ import annotations

import html
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent.parent   # project root
RAW       = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW / "drugsComTrain_raw.csv"
TEST_CSV  = RAW / "drugsComTest_raw.csv"
OUT_CSV   = RAW / "posts_classified.csv"
SEED_PKL  = RAW / "seed_bank.pkl"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & filter to abuse-relevant records
# ══════════════════════════════════════════════════════════════════════════════

ABUSE_SUBSTANCES = {
    # Opioids
    "oxycodone", "hydrocodone", "oxycontin", "vicodin", "percocet",
    "buprenorphine", "suboxone", "methadone", "tramadol", "codeine",
    "fentanyl", "morphine", "heroin",
    # Benzodiazepines
    "alprazolam", "xanax", "diazepam", "valium", "clonazepam",
    "klonopin", "lorazepam", "ativan",
    # Stimulants
    "amphetamine", "adderall", "methamphetamine", "cocaine",
    "methylphenidate", "ritalin", "vyvanse",
    # Alcohol / dependence treatment
    "naltrexone", "acamprosate", "disulfiram",
    # Commonly abused adjuncts
    "gabapentin", "pregabalin",
}

ABUSE_CONDITIONS = {
    "opiate dependence", "opioid dependence", "alcohol dependence",
    "alcohol use disorder", "drug dependence", "substance abuse",
    "anxiety and stress", "pain", "chronic pain", "adhd", "depression",
}


def load_and_filter(train_path: Path = TRAIN_CSV,
                    test_path:  Path = TEST_CSV) -> pd.DataFrame:
    """Step 1 — load both splits, decode HTML entities, filter abuse-relevant rows."""
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset CSVs not found.\n"
            f"Expected:\n  {train_path}\n  {test_path}\n"
            "Download from Kaggle: 'UCI ML Drug Review Dataset'."
        )

    train = pd.read_csv(train_path, index_col=0)
    test  = pd.read_csv(test_path,  index_col=0)
    df    = pd.concat([train, test], ignore_index=True)

    # Decode HTML entities (&amp; &#039; etc.)
    df["review"] = df["review"].apply(lambda x: html.unescape(str(x)))

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter
    def is_relevant(row) -> bool:
        drug = str(row.get("drugName", "")).lower()
        cond = str(row.get("condition", "")).lower()
        return (
            any(s in drug for s in ABUSE_SUBSTANCES) or
            any(c in cond for c in ABUSE_CONDITIONS)
        )

    abuse_df = df[df.apply(is_relevant, axis=1)].copy()
    print(f"  Abuse-relevant reviews: {len(abuse_df):,} of {len(df):,}")
    return abuse_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Map drug names to substance categories
# ══════════════════════════════════════════════════════════════════════════════

SUBSTANCE_MAP = {
    # Opioids
    "oxycodone": "opioid",  "hydrocodone": "opioid", "oxycontin": "opioid",
    "vicodin":   "opioid",  "percocet":    "opioid",  "buprenorphine": "opioid",
    "suboxone":  "opioid",  "methadone":   "opioid",  "tramadol": "opioid",
    "codeine":   "opioid",  "fentanyl":    "opioid",  "morphine": "opioid",
    "heroin":    "opioid",
    # Benzodiazepines
    "alprazolam": "benzo", "xanax":     "benzo", "diazepam":  "benzo",
    "valium":     "benzo", "clonazepam":"benzo", "klonopin":  "benzo",
    "lorazepam":  "benzo", "ativan":    "benzo",
    # Stimulants
    "amphetamine":   "stimulant", "adderall":      "stimulant",
    "methamphetamine":"stimulant","cocaine":        "stimulant",
    "methylphenidate":"stimulant","ritalin":        "stimulant",
    "vyvanse":       "stimulant",
    # Alcohol treatment (proxy for alcohol-abuse context)
    "naltrexone": "alcohol", "acamprosate": "alcohol", "disulfiram": "alcohol",
    # Gabapentin / pregabalin
    "gabapentin": "gabapentin", "pregabalin": "gabapentin",
}


def map_substance(drug_name: str) -> str:
    drug = str(drug_name).lower()
    for key, category in SUBSTANCE_MAP.items():
        if key in drug:
            return category
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Heuristic risk labeling
# ══════════════════════════════════════════════════════════════════════════════

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


def assign_risk_label(row) -> str:
    """
    Heuristic risk label using phrase matching + rating.
    'high'   — any high-risk phrase, OR ≥2 distress phrases with rating ≤ 3
    'medium' — ≥1 distress phrase, OR rating ≤ 4
    'low'    — otherwise
    """
    text          = str(row["review"]).lower()
    rating        = float(row.get("rating", 5))
    high_hits     = sum(p in text for p in HIGH_RISK_PHRASES)
    distress_hits = sum(p in text for p in DISTRESS_PHRASES)

    if high_hits >= 1 or (distress_hits >= 2 and rating <= 3):
        return "high"
    elif distress_hits >= 1 or rating <= 4:
        return "medium"
    else:
        return "low"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Embedding-based seed bank
# ══════════════════════════════════════════════════════════════════════════════

def build_seed_bank(df: pd.DataFrame,
                    n_per_substance: int = 50) -> dict:
    """
    Encode the top-N high-risk reviews per substance using
    'all-MiniLM-L6-v2' and return a seed bank for cosine-similarity scoring.

    Returns: {substance: [(text, embedding), …]}
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠  sentence-transformers not installed — skipping seed bank.")
        print("     pip install sentence-transformers")
        return {}

    print("  Loading sentence-transformers model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    high_risk = (
        df[df["risk_level"] == "high"]
        .sort_values("usefulCount", ascending=False)
    )

    seed_bank: dict[str, list] = {}
    for substance, group in high_risk.groupby("substance"):
        samples    = group.head(n_per_substance)["review"].tolist()
        embeddings = model.encode(samples, show_progress_bar=False)
        seed_bank[substance] = list(zip(samples, embeddings))
        print(f"    {substance}: {len(samples)} seed examples encoded")

    return seed_bank


def score_forum_post(post_text: str, seed_bank: dict,
                     substance: str | None = None,
                     top_k: int = 5) -> float:
    """
    Score an unlabeled forum post against the seed bank.
    Returns mean cosine similarity to the top-k seeds (0–1).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return 0.0

    post_emb      = model.encode([post_text])
    seeds_to_check = (
        {substance: seed_bank[substance]}
        if substance and substance in seed_bank
        else seed_bank
    )

    best = 0.0
    for _, seeds in seeds_to_check.items():
        if not seeds:
            continue
        _, embs = zip(*seeds)
        sims    = cosine_similarity(post_emb, np.array(embs))[0]
        score   = float(np.sort(sims)[-top_k:].mean())
        best    = max(best, score)

    return round(best, 4)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Slang / entity lexicon mining
# ══════════════════════════════════════════════════════════════════════════════

KNOWN_SLANG: dict[str, str] = {
    "oxy": "opioid",   "oxys": "opioid",   "percs": "opioid",
    "vikes": "opioid", "subs": "opioid",   "subbies": "opioid",
    "done": "opioid",                       # methadone
    "bars": "benzo",   "xannies": "benzo", "benzos": "benzo",
    "addys": "stimulant", "speed": "stimulant",
    "gabas": "gabapentin", "gabbies": "gabapentin",
}

STOPWORDS = {"the", "and", "for", "was", "this", "that", "with", "have",
             "been", "from", "they", "are", "but", "not", "you", "all",
             "when", "can", "just", "like", "has"}


def extract_window_terms(text: str, canonical: str,
                          window: int = 5) -> list[str]:
    """Return words within ±window positions of canonical in text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if canonical not in words:
        return []
    idx       = words.index(canonical)
    neighbors = words[max(0, idx - window): idx + window + 1]
    return [
        w for w in neighbors
        if len(w) >= 3 and w not in STOPWORDS and w != canonical
    ]


def mine_slang_lexicon(df: pd.DataFrame,
                       top_n: int = 30) -> dict[str, Counter]:
    """
    Step 5 — co-occurrence mining around canonical drug names.
    Returns {substance: Counter of candidate slang terms}.
    """
    CANONICALS = {
        "opioid":    ["oxycodone", "hydrocodone", "buprenorphine",
                      "methadone", "fentanyl", "tramadol"],
        "benzo":     ["alprazolam", "clonazepam", "lorazepam", "diazepam"],
        "stimulant": ["amphetamine", "methylphenidate", "vyvanse"],
        "alcohol":   ["naltrexone", "acamprosate"],
        "gabapentin":["gabapentin", "pregabalin"],
    }

    results: dict[str, Counter] = {}
    for substance, drugs in CANONICALS.items():
        subset   = df[df["substance"] == substance]["review"]
        counter  = Counter()
        for review in subset:
            for drug in drugs:
                counter.update(extract_window_terms(review, drug))
        results[substance] = counter

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Step 1: load & filter ────────────────────────────────────────────────
    print("Step 1 — Loading and filtering drug reviews …")
    df = load_and_filter()

    # ── Step 2: substance mapping ─────────────────────────────────────────────
    print("\nStep 2 — Mapping substances …")
    df["substance"] = df["drugName"].apply(map_substance)
    print(df["substance"].value_counts().to_string())

    # ── Step 3: risk labeling ─────────────────────────────────────────────────
    print("\nStep 3 — Assigning risk labels …")
    df["risk_level"] = df.apply(assign_risk_label, axis=1)
    print(df["risk_level"].value_counts().to_string())

    # Save posts_classified.csv (aligned to signal_pipeline.py schema)
    out = df.rename(columns={"date": "timestamp"})[
        ["timestamp", "substance", "risk_level", "review", "drugName",
         "rating", "usefulCount"]
    ].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved {len(out):,} rows → {OUT_CSV}")

    # ── Step 4: seed bank ─────────────────────────────────────────────────────
    print("\nStep 4 — Building embedding seed bank …")
    seed_bank = build_seed_bank(df)
    if seed_bank:
        with open(SEED_PKL, "wb") as f:
            pickle.dump(seed_bank, f)
        print(f"  Saved seed bank → {SEED_PKL}")

    # ── Step 5: slang lexicon ─────────────────────────────────────────────────
    print("\nStep 5 — Mining slang candidates …")
    lexicon = mine_slang_lexicon(df)
    print("\n  Top candidates by substance (review & curate manually):")
    for substance, counter in lexicon.items():
        top = counter.most_common(15)
        if top:
            print(f"\n  [{substance}]")
            for term, count in top:
                flag = " ← known slang" if term in KNOWN_SLANG else ""
                print(f"    {term:20s} {count:4d}{flag}")

    print("\nDone ✓")
    print(f"  posts_classified.csv → {OUT_CSV}")
    if seed_bank:
        print(f"  seed_bank.pkl       → {SEED_PKL}")
    print("\nNext: run  python signal_pipeline.py  — Step 3 will now proceed.")


if __name__ == "__main__":
    main()
