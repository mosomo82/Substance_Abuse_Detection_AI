"""
rule_based_classifier.py
========================
Layer 1 (of 3) in the detection pipeline — rule-based risk scoring.

Architecture:
    processed_text
        ├── Layer 1: Substance mention scorer    → substance_score (0–1)
        ├── Layer 2: Distress signal scorer      → distress_score  (0–1)
        ├── Layer 3: Behavioral pattern scorer   → behavior_score  (0–1)
        └── Weighted combination                 → risk_level (low/medium/high)
                                                + evidence (list of matched rules)

Role in the full pipeline:
    - Handles confident low/high cases cheaply (no API call needed)
    - Routes ambiguous posts (score 0.15–0.70) to LLM classifier
    - Provides the rule_evidence field required for Task 3 explainability

Inputs:
    data/processed/posts_preprocessed.csv  (output of preprocess_posts.py)

Outputs:
    data/processed/rule_based_results.csv

Run:
    python rule_based_classifier.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).parent.parent        # project root (one level above src/)
DATA     = ROOT / "data"
IN_CSV   = DATA / "processed" / "posts_preprocessed.csv"
OUT_CSV  = DATA / "processed" / "rule_based_results.csv"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Substance mention scorer
# ══════════════════════════════════════════════════════════════════════════════

SUBSTANCE_WEIGHTS: dict[str, float] = {
    "fentanyl":           1.00,
    "heroin":             1.00,
    "opioid intoxication":1.00,
    "withdrawal":         0.90,
    "methamphetamine":    0.90,
    "oxycodone":          0.85,
    "percocet":           0.85,
    "hydrocodone":        0.80,
    "cocaine":            0.80,
    "methadone":          0.75,
    "craving":            0.75,
    "xanax":              0.70,
    "benzodiazepine":     0.70,
    "suboxone":           0.60,   # treatment drug — lower weight
    "adderall":           0.50,
    "alcohol":            0.50,
    "cannabis":           0.20,
}

# Patterns that reduce a substance mention's risk weight
_NEGATION_RE = re.compile(
    r"\b(never|not|no longer|quit|stopped|clean from|sober from|"
    r"recovering from|off of|free from)\b.{0,30}",
    re.I,
)
_PAST_RE = re.compile(
    r"\b(used to|back when|years ago|in the past|"
    r"before i got clean|when i was using)\b",
    re.I,
)


def score_substance_mentions(text: str,
                              substances: list[str]) -> tuple[float, list[str]]:
    """
    Returns (substance_score 0–1, evidence list).
    Takes the max across substances — one high-risk mention is enough.
    """
    if not substances:
        return 0.0, []

    scores, evidence = [], []

    for substance in substances:
        weight = SUBSTANCE_WEIGHTS.get(substance, 0.40)

        negated = bool(_NEGATION_RE.search(text + " " + substance))
        past    = bool(_PAST_RE.search(text))

        if negated:
            weight *= 0.20
            ctx = "negated"
        elif past:
            weight *= 0.50
            ctx = "past tense"
        else:
            ctx = "active"

        scores.append(weight)
        evidence.append(f"Substance: {substance} (weight={weight:.2f}, ctx={ctx})")

    return round(max(scores), 3), evidence


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Distress signal scorer
# ══════════════════════════════════════════════════════════════════════════════

DISTRESS_WEIGHTS: dict[str, float] = {
    "overdose":     1.00,
    "relapse":      0.90,
    "withdrawal":   0.85,
    "desperation":  0.80,
    "hopelessness": 0.75,
    "craving":      0.65,
}

_URGENCY_PATTERNS = [
    re.compile(r"\b(help me|i need help|please help|someone help)\b", re.I),
    re.compile(r"\b(going to die|might die|could die|scared to die)\b", re.I),
    re.compile(r"\b(last resort|nowhere to turn|no one to call)\b", re.I),
    re.compile(r"\b(can't take it|can't do this|done with this)\b", re.I),
]

_ESCALATION_PATTERNS = [
    re.compile(r"\b(need more|taking more|upping my dose|more than prescribed)\b", re.I),
    re.compile(r"\b(running out|almost out|last few|scraped together)\b", re.I),
    re.compile(r"\b(doctor shopping|multiple doctors|pharmacy hopping)\b", re.I),
]


def score_distress_signals(text: str,
                            distress: dict[str, list[str]]) -> tuple[float, list[str]]:
    """Returns (distress_score 0–1, evidence list)."""
    scores, evidence = [], []

    for sig_type, phrases in distress.items():
        w = DISTRESS_WEIGHTS.get(sig_type, 0.50)
        scores.append(w)
        evidence.append(f"Distress: {sig_type} (phrases: {', '.join(phrases[:2])})")

    # Co-occurrence bonus
    if len(distress) >= 3:
        scores.append(0.20)
        evidence.append("Co-occurrence bonus: 3+ distress types present")

    # Urgency override
    urgency_hits = sum(1 for p in _URGENCY_PATTERNS if p.search(text))
    if urgency_hits:
        scores.append(0.95)
        evidence.append(f"Urgency language: {urgency_hits} pattern(s) matched")

    # Escalation patterns
    esc_hits = sum(1 for p in _ESCALATION_PATTERNS if p.search(text))
    if esc_hits:
        scores.append(0.70)
        evidence.append(f"Escalation language: {esc_hits} pattern(s) matched")

    score = max(scores) if scores else 0.0
    return round(min(score, 1.0), 3), evidence


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Behavioral pattern scorer
# ══════════════════════════════════════════════════════════════════════════════

BEHAVIORAL_PATTERNS: dict[str, dict] = {
    "route_of_administration": {
        "patterns": [
            re.compile(r"\b(shooting up|shooting it|IV|intravenous|injecting)\b", re.I),
            re.compile(r"\b(snorting|snorted|insufflation|railing|rails)\b", re.I),
            re.compile(r"\b(smoking it|hitting the pipe|freebasing)\b", re.I),
            re.compile(r"\b(plugging|rectal|boofing)\b", re.I),
        ],
        "weight": 0.95,
    },
    "dose_manipulation": {
        "patterns": [
            re.compile(r"\b(crushing|chewing|dissolving).{0,20}(pill|tab|mg)\b", re.I),
            re.compile(r"\b(more than prescribed|double.{0,10}dose|extra.{0,10}dose)\b", re.I),
            re.compile(r"\b(mixing|combining|stacking).{0,30}(with|and)\b", re.I),
        ],
        "weight": 0.85,
    },
    "procurement": {
        "patterns": [
            re.compile(r"\b(where can i (get|find|buy)|how do i (get|find|buy))\b", re.I),
            re.compile(r"\b(connect|plug|dealer|hook me up)\b", re.I),
            re.compile(r"\b(dark web|darknet)\b", re.I),
            re.compile(r"\b(pressed|counterfeit|fake).{0,15}(pill|tab|bar)\b", re.I),
        ],
        "weight": 0.90,
    },
    "concealment": {
        "patterns": [
            re.compile(r"\b(hide|hiding|conceal|secret|without (them|anyone) knowing)\b", re.I),
            re.compile(r"\b(fake|empty).{0,10}(prescription|script|bottle)\b", re.I),
            re.compile(r"\b(beat.{0,10}test|pass.{0,10}drug test|clean.{0,10}piss)\b", re.I),
        ],
        "weight": 0.80,
    },
    "social_disruption": {
        "patterns": [
            re.compile(r"\b(lost my job|fired|can't work|missed work).{0,30}(because|due to)\b", re.I),
            re.compile(r"\b(kicked out|evicted|homeless).{0,30}(because|due to|over)\b", re.I),
            re.compile(r"\b(family.{0,20}(left|gave up|done with me))\b", re.I),
        ],
        "weight": 0.75,
    },
}


def score_behavioral_patterns(text: str) -> tuple[float, list[str]]:
    """Returns (behavior_score 0–1, evidence list)."""
    scores, evidence = [], []

    for name, cfg in BEHAVIORAL_PATTERNS.items():
        hits = sum(1 for p in cfg["patterns"] if p.search(text))
        if hits:
            scores.append(cfg["weight"])
            evidence.append(
                f"Behavior: {name} ({hits} pattern match{'es' if hits > 1 else ''})"
            )

    return round(max(scores) if scores else 0.0, 3), evidence


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Combine into final classifier
# ══════════════════════════════════════════════════════════════════════════════

# Layer weights — behavior highest (most specific to problematic use)
_WEIGHTS = {"substance": 0.30, "distress": 0.35, "behavior": 0.35}

# Score thresholds
_HIGH_THRESHOLD   = 0.55
_MEDIUM_THRESHOLD = 0.30


def classify_post(record: dict) -> dict:
    """
    Full rule-based classification for a single preprocessed record.

    Input:  dict from preprocess_post() / a row of posts_preprocessed.csv
    Output: classification dict with risk_level, scores, and evidence
    """
    text = record.get("processed_text", "")

    # Deserialize JSON-string columns if coming from CSV
    substances = record.get("substances_detected", [])
    if isinstance(substances, str):
        substances = json.loads(substances)

    distress = record.get("distress_signals", {})
    if isinstance(distress, str):
        distress = json.loads(distress)

    s_score, s_ev = score_substance_mentions(text, substances)
    d_score, d_ev = score_distress_signals(text, distress)
    b_score, b_ev = score_behavioral_patterns(text)

    combined = round(
        _WEIGHTS["substance"] * s_score +
        _WEIGHTS["distress"]  * d_score +
        _WEIGHTS["behavior"]  * b_score,
        3,
    )

    # Hard overrides — certain signals force 'high' regardless of combined score
    hard_high = any([
        "overdose" in distress,
        b_score >= 0.90,                           # non-oral ROA or procurement
        s_score >= 0.90 and d_score >= 0.70,
    ])

    if hard_high or combined >= _HIGH_THRESHOLD:
        risk_level = "high"
    elif combined >= _MEDIUM_THRESHOLD:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "post_id":         record.get("post_id"),
        "timestamp":       record.get("timestamp"),
        "risk_level":      risk_level,
        "combined_score":  combined,
        "substance_score": s_score,
        "distress_score":  d_score,
        "behavior_score":  b_score,
        "evidence":        json.dumps(s_ev + d_ev + b_ev),
        "substances":      json.dumps(substances),
        "method":          "rule_based",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Batch classification and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def classify_corpus(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Classify all posts in the preprocessed DataFrame."""
    return pd.DataFrame([classify_post(row.to_dict())
                         for _, row in processed_df.iterrows()])


def evaluate_classifier(results_df: pd.DataFrame,
                         ground_truth_col: str = "risk_level") -> None:
    """Print precision/recall/F1 if ground truth is available."""
    from sklearn.metrics import classification_report

    if ground_truth_col not in results_df.columns:
        print("No ground truth — showing distribution only:")
        print(results_df["risk_level"].value_counts().to_string())
        return

    print(classification_report(
        results_df[ground_truth_col],
        results_df["risk_level"],
        labels=["high", "medium", "low"],
    ))


def inspect_high_risk(results_df: pd.DataFrame, n: int = 5) -> None:
    """Pretty-print evidence for the top-n high-risk posts."""
    high = results_df[results_df["risk_level"] == "high"].head(n)
    for _, row in high.iterrows():
        evidence = json.loads(row["evidence"]) if isinstance(row["evidence"], str) else row["evidence"]
        print(f"\nScore: {row['combined_score']}  |  Substances: {row['substances']}")
        for e in evidence:
            print(f"  • {e}")
        print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LLM routing (cost/latency optimization)
# ══════════════════════════════════════════════════════════════════════════════

def route_to_classifier(result: dict) -> str:
    """
    Decide whether a post needs LLM classification.

    Returns:
        'rule_based_high' — confident high (skip LLM)
        'rule_based_low'  — confident low  (skip LLM)
        'send_to_llm'     — ambiguous      (send to LLM)
    """
    score = result["combined_score"]
    if score >= 0.70:
        return "rule_based_high"
    elif score <= 0.15:
        return "rule_based_low"
    else:
        return "send_to_llm"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not IN_CSV.exists():
        print(f"⚠  Input not found: {IN_CSV}")
        print("   Run preprocess_posts.py first.")
        return

    print(f"Loading preprocessed posts from {IN_CSV} …")
    df = pd.read_csv(IN_CSV)
    print(f"  {len(df):,} rows loaded\n")

    print("Classifying …")
    results = classify_corpus(df)

    # ── Risk level distribution ───────────────────────────────────────────────
    print("\nRisk level distribution:")
    print(results["risk_level"].value_counts().to_string())

    # ── Score statistics ──────────────────────────────────────────────────────
    print("\nScore statistics:")
    print(results[["combined_score", "substance_score",
                    "distress_score", "behavior_score"]].describe().round(3).to_string())

    # ── LLM routing estimate ──────────────────────────────────────────────────
    results["routing"] = results.apply(route_to_classifier, axis=1)
    routing_dist = results["routing"].value_counts()
    llm_pct = routing_dist.get("send_to_llm", 0) / len(results)
    print(f"\nLLM routing:")
    print(routing_dist.to_string())
    print(f"→ {llm_pct:.1%} of posts require LLM classification")

    # ── Sample high-risk evidence ─────────────────────────────────────────────
    print("\nSample high-risk posts (evidence):")
    inspect_high_risk(results, n=3)

    # ── Evaluate against drug review heuristic labels (if available) ──────────
    if "risk_level" in df.columns:
        print("\nEvaluation vs. process_drug_reviews.py heuristic labels:")
        merged = results.copy()
        merged["true_label"] = df["risk_level"].values
        evaluate_classifier(merged, ground_truth_col="true_label")

    results.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(results):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
