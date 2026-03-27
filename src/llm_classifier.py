"""
llm_classifier.py
=================
Layer 3 (of 3) in the detection pipeline — LLM-based risk classification.

Uses Google Gemini Flash for speed and cost efficiency.

Architecture:
    preprocessed post
        ├── Few-shot prompt construction
        ├── Gemini API call (structured JSON output)
        ├── Parse + validate response
        └── risk_level + evidence + analyst_summary

Role in full pipeline:
    - Handles ambiguous posts that rule-based and embedding can't resolve
    - Primary explainability tool (reasoning + evidence spans per post)
    - RAG layer: generates aggregate analyst summaries for detected spikes

Pipeline integration (hybrid routing):
    combined_score ≥ 0.75  → rule_based_high  (no API call)
    combined_score ≤ 0.10  → rule_based_low   (no API call)
    0.10 < score < 0.40    → embedding        (no API call if confident)
    remainder              → LLM              (API call)

Setup:
    pip install google-generativeai
    set GOOGLE_API_KEY=your_key_here   (or add to .env)

Inputs:
    data/processed/posts_preprocessed.csv
    data/processed/rule_based_results.csv
    data/raw/seed_bank.pkl               (for embedding stage of hybrid)

Outputs:
    data/processed/llm_results.csv
    data/processed/hybrid_results.csv
    data/processed/spike_summaries.json
    data/processed/method_comparison.csv

Run:
    python src/llm_classifier.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
RAW       = DATA / "raw"
PROCESSED.mkdir(parents=True, exist_ok=True)

IN_CSV          = PROCESSED / "posts_preprocessed.csv"
RULE_CSV        = PROCESSED / "rule_based_results.csv"
EMBEDDING_CSV   = PROCESSED / "embedding_results.csv"
OUT_LLM_CSV     = PROCESSED / "llm_results.csv"
OUT_HYBRID_CSV  = PROCESSED / "hybrid_results.csv"
OUT_SPIKES_JSON = PROCESSED / "spike_summaries.json"
OUT_COMPARE_CSV = PROCESSED / "method_comparison.csv"

# ── Gemini client ──────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    if _API_KEY:
        genai.configure(api_key=_API_KEY)
    _GEMINI_AVAILABLE = bool(_API_KEY)
except ImportError:
    _GEMINI_AVAILABLE = False

_FLASH_MODEL = "gemini-2.0-flash"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — System prompt
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a public health AI assistant helping researchers
identify substance abuse risk signals in anonymized social media posts.

Your role is to analyze posts for signs of:
- Active substance use or dependence
- Emotional distress, relapse, or withdrawal
- Behavioral patterns indicating problematic use

CRITICAL RULES:
- Never identify, name, or make assumptions about individuals
- Focus on population-level signal detection only
- When uncertain, prefer lower risk levels (bias toward privacy)
- Base assessments only on text evidence present in the post

OUTPUT FORMAT:
Respond with valid JSON only. No preamble, no explanation outside the JSON.

{
  "risk_level": "high" | "medium" | "low",
  "confidence": 0.0-1.0,
  "substances_detected": ["substance1", "substance2"],
  "signal_types": ["relapse" | "withdrawal" | "craving" | "overdose" |
                   "active_use" | "procurement" | "emotional_distress" |
                   "escalation" | "none"],
  "evidence_spans": ["exact short phrase from post that supports assessment"],
  "reasoning": "2-3 sentence explanation of classification for analyst review",
  "analyst_summary": "1 sentence suitable for a public health dashboard",
  "requires_human_review": true | false
}"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Few-shot examples
# ══════════════════════════════════════════════════════════════════════════════

# Gemini uses a flat conversation list: alternating user/model turns
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "role": "user",
        "parts": ['Classify this post:\n"been dope sick for 3 days now cant keep anything '
                  'down shaking so bad just need one more hit to feel normal again '
                  'my dealer isnt answering"'],
    },
    {
        "role": "model",
        "parts": [json.dumps({
            "risk_level": "high",
            "confidence": 0.97,
            "substances_detected": ["heroin", "opioid"],
            "signal_types": ["withdrawal", "craving", "procurement"],
            "evidence_spans": ["dope sick for 3 days", "shaking so bad",
                               "need one more hit to feel normal",
                               "dealer isnt answering"],
            "reasoning": ("Post describes active opioid withdrawal symptoms including "
                          "nausea and tremors. The phrase 'feel normal' indicates physical "
                          "dependence. Attempted procurement from dealer suggests ongoing "
                          "active use."),
            "analyst_summary": ("Active opioid withdrawal with dependence indicators "
                                "and procurement behavior detected."),
            "requires_human_review": False,
        }, indent=2)],
    },
    {
        "role": "user",
        "parts": ['Classify this post:\n"3 months sober today. still think about using '
                  'sometimes especially when im stressed but going to meetings is '
                  'helping a lot"'],
    },
    {
        "role": "model",
        "parts": [json.dumps({
            "risk_level": "medium",
            "confidence": 0.82,
            "substances_detected": [],
            "signal_types": ["craving", "emotional_distress"],
            "evidence_spans": ["still think about using sometimes", "when im stressed"],
            "reasoning": ("Post indicates sustained recovery with active coping strategies. "
                          "Stress-triggered cravings present a medium-level relapse risk. "
                          "Engagement with support meetings is a protective factor."),
            "analyst_summary": ("Early recovery with stress-triggered cravings; "
                                "protective support system in place."),
            "requires_human_review": False,
        }, indent=2)],
    },
    {
        "role": "user",
        "parts": ['Classify this post:\n"my doctor switched me to a lower dose and im '
                  'actually feeling better. been cutting back on alcohol too since '
                  'it was interfering"'],
    },
    {
        "role": "model",
        "parts": [json.dumps({
            "risk_level": "low",
            "confidence": 0.88,
            "substances_detected": ["alcohol"],
            "signal_types": ["none"],
            "evidence_spans": ["cutting back on alcohol", "feeling better"],
            "reasoning": ("Post describes medically supervised dose reduction with positive "
                          "outcomes. Voluntary alcohol reduction suggests health-conscious "
                          "behavior rather than problematic use."),
            "analyst_summary": ("Medically supervised treatment with voluntary substance "
                                "reduction; low population-level risk signal."),
            "requires_human_review": False,
        }, indent=2)],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Single post classification
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_RESULT = {
    "risk_level":            "low",
    "confidence":             0.0,
    "substances_detected":   [],
    "signal_types":          [],
    "evidence_spans":        [],
    "reasoning":             "Classification failed — requires manual review.",
    "analyst_summary":       "Classification unavailable.",
    "requires_human_review": True,
    "method":                "llm",
    "model":                 _FLASH_MODEL,
}


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def classify_post_llm(post_text: str,
                      max_retries: int = 3,
                      retry_delay: float = 1.5) -> dict:
    """
    Classify a single post using Gemini Flash.
    Returns a parsed classification dict (never raises).
    """
    if not _GEMINI_AVAILABLE:
        return {**_FALLBACK_RESULT,
                "reasoning": "GOOGLE_API_KEY not set or google-generativeai not installed."}

    model = genai.GenerativeModel(
        model_name=_FLASH_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # Start conversation from few-shot history
    chat = model.start_chat(history=FEW_SHOT_EXAMPLES)

    for attempt in range(max_retries):
        try:
            t0       = time.perf_counter()
            response = chat.send_message(
                f'Classify this post:\n"{post_text}"'
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            result = _parse_response(response.text)
            result["method"]     = "llm"
            result["model"]      = _FLASH_MODEL
            result["latency_ms"] = latency_ms
            return result

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                # Re-create chat for next attempt (avoid confused state)
                chat = model.start_chat(history=FEW_SHOT_EXAMPLES)
                continue
            return {**_FALLBACK_RESULT, "error": "JSON parse error after retries"}

        except Exception as e:
            err_str = str(e)
            # Rate limit → back off
            if "429" in err_str or "quota" in err_str.lower():
                wait = retry_delay * (2 ** attempt)
                print(f"    Rate limit — waiting {wait:.0f}s …")
                time.sleep(wait)
                chat = model.start_chat(history=FEW_SHOT_EXAMPLES)
                continue
            return {**_FALLBACK_RESULT, "error": err_str}

    return {**_FALLBACK_RESULT, "error": "Max retries exceeded"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Cost-aware hybrid batch classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_corpus_hybrid(processed_df: pd.DataFrame,
                            rule_results_df: pd.DataFrame,
                            seed_bank: dict | None = None,
                            llm_rate_limit_delay: float = 0.3) -> pd.DataFrame:
    """
    Three-stage hybrid pipeline:
      Stage 1 — rule_based_high/low  : confident cases, no API call
      Stage 2 — embedding            : moderate cases (if seed_bank provided)
      Stage 3 — llm                  : ambiguous cases requiring reasoning

    Returns unified DataFrame with routing column.
    """
    from src.embedding_classifier import score_post_embedding

    results: list[dict] = []
    llm_queue: list[dict] = []

    print(f"Routing {len(processed_df):,} posts …")

    for i, (_, row) in enumerate(processed_df.iterrows()):
        rule_row = rule_results_df.iloc[i]
        score    = float(rule_row.get("combined_score", 0))
        base     = rule_row.to_dict()

        # Stage 1a — confident high
        if score >= 0.75:
            results.append({**base, "routing": "rule_high"})
            continue

        # Stage 1b — confident low
        if score <= 0.10:
            results.append({**base, "routing": "rule_low"})
            continue

        # Stage 2 — embedding for moderate-confidence range
        if seed_bank and 0.10 < score < 0.40:
            emb = score_post_embedding(row["processed_text"], seed_bank)
            if emb.get("confidence", 0) >= 0.20:
                results.append({
                    **emb,
                    "post_id":   row.get("post_id"),
                    "timestamp": row.get("timestamp"),
                    "routing":   "embedding",
                })
                continue

        # Stage 3 — LLM for everything else
        llm_queue.append({"idx": i, "row": row})

    print(f"  rule_high : {sum(1 for r in results if r.get('routing') == 'rule_high'):,}")
    print(f"  rule_low  : {sum(1 for r in results if r.get('routing') == 'rule_low'):,}")
    print(f"  embedding : {sum(1 for r in results if r.get('routing') == 'embedding'):,}")
    print(f"  llm queue : {len(llm_queue):,} ({len(llm_queue)/len(processed_df):.1%})")

    for item in llm_queue:
        row    = item["row"]
        result = classify_post_llm(row["processed_text"])
        if result:
            result["post_id"]   = row.get("post_id")
            result["timestamp"] = row.get("timestamp")
            result["routing"]   = "llm"
            results.append(result)
        time.sleep(llm_rate_limit_delay)   # ~3 req/s to stay under free-tier limit

    return pd.DataFrame(results)


def print_routing_summary(results_df: pd.DataFrame) -> None:
    counts = results_df["routing"].value_counts()
    total  = len(results_df)
    print("\n── Routing Summary ────────────────────────")
    for route, count in counts.items():
        print(f"  {route:<22} {count:>6,}  ({count/total:.1%})")
    print(f"  {'TOTAL':<22} {total:>6,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — RAG spike summary (Task 3)
# ══════════════════════════════════════════════════════════════════════════════

RAG_SUMMARY_PROMPT = """You are a public health analyst assistant.
Given anonymized social media posts flagged as high-risk for substance abuse,
write a concise population-level summary.

RULES:
- Never reference individuals or personally identifiable details
- Speak in aggregate terms (e.g. "posts in this cluster", "community signals")
- Highlight dominant substances, signal types, and temporal patterns
- Note any early warning indicators
- Keep summary to 3-4 sentences maximum
- End with a recommended action for public health teams

Respond with JSON only:
{
  "summary": "3-4 sentence population-level summary",
  "dominant_substances": ["substance1"],
  "dominant_signals": ["signal_type1"],
  "severity": "critical" | "elevated" | "moderate",
  "recommended_action": "one sentence action item for public health team",
  "confidence": 0.0-1.0
}"""


def generate_spike_summary(flagged_posts: list[dict],
                            spike_date:   str,
                            supporting_cdc_data: dict | None = None) -> dict:
    """
    Generate a RAG analyst summary for a detected spike event.

    flagged_posts       : list of classified post dicts from the spike period
    spike_date          : human-readable date/period string
    supporting_cdc_data : optional {'deaths', 'substance', 'state', 'pct_change'}
    """
    if not _GEMINI_AVAILABLE:
        return {"error": "Gemini API not available", "spike_date": spike_date}

    post_lines = []
    for i, post in enumerate(flagged_posts[:15]):
        txt = str(post.get("processed_text", ""))[:200]
        post_lines.append(f"Post {i+1} [{post.get('risk_level','?')} risk]: {txt}")

    cdc_ctx = ""
    if supporting_cdc_data:
        cdc_ctx = (
            f"\nCDC Context for {spike_date}:\n"
            f"  Deaths that month : {supporting_cdc_data.get('deaths', 'N/A')}\n"
            f"  Primary substance : {supporting_cdc_data.get('substance', 'N/A')}\n"
            f"  State             : {supporting_cdc_data.get('state', 'N/A')}\n"
            f"  MoM change        : {supporting_cdc_data.get('pct_change', 'N/A')}%\n"
        )

    user_msg = (
        f"Spike detected: {spike_date}\n"
        f"Total flagged: {len(flagged_posts)}\n"
        f"High risk: {sum(1 for p in flagged_posts if p.get('risk_level') == 'high')}\n"
        f"Medium risk: {sum(1 for p in flagged_posts if p.get('risk_level') == 'medium')}\n"
        f"{cdc_ctx}\n"
        "Representative posts:\n" + "\n".join(post_lines) +
        "\n\nGenerate a population-level analyst summary."
    )

    model    = genai.GenerativeModel(_FLASH_MODEL,
                                      system_instruction=RAG_SUMMARY_PROMPT)
    response = model.generate_content(user_msg)

    result = _parse_response(response.text)
    result["spike_date"]   = spike_date
    result["post_count"]   = len(flagged_posts)
    result["generated_at"] = pd.Timestamp.now().isoformat()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Three-method comparison table (Task 1)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_comparison(rule_df:      pd.DataFrame,
                         embedding_df: pd.DataFrame,
                         llm_df:       pd.DataFrame,
                         ground_truth_col: str = "true_label") -> pd.DataFrame:
    """
    Build the Task 1 method comparison table.
    Prints side-by-side metrics and returns a summary DataFrame.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    methods = {
        "Rule-based": (rule_df,      "$0.00",  True),
        "Embedding":  (embedding_df, "~$0.01", False),
        "LLM":        (llm_df,       "~$0.05", True),
    }

    rows = []
    for name, (df, cost, explainable) in methods.items():
        has_truth = ground_truth_col in df.columns
        if has_truth:
            y_true = df[ground_truth_col]
            y_pred = df["risk_level"]
            acc    = round(accuracy_score(y_true, y_pred), 3)
            f1_h   = round(f1_score(y_true, y_pred,
                                    labels=["high"], average="macro",
                                    zero_division=0), 3)
            f1_m   = round(f1_score(y_true, y_pred,
                                    average="macro", zero_division=0), 3)
            print(f"\n{'='*10} {name} {'='*10}")
            print(classification_report(y_true, y_pred,
                                        labels=["high", "medium", "low"]))
        else:
            acc = f1_h = f1_m = float("nan")

        avg_lat = (df["latency_ms"].mean()
                   if "latency_ms" in df.columns else 0.0)

        rows.append({
            "Method":        name,
            "Accuracy":      acc,
            "F1 (high)":     f1_h,
            "F1 (macro)":    f1_m,
            "Avg latency ms": round(avg_lat, 1),
            "Explainable":   explainable,
            "Cost / 1k":     cost,
        })

    cmp_df = pd.DataFrame(rows)
    print("\n── Method Comparison ──────────────────────────────────")
    print(cmp_df.to_string(index=False))
    return cmp_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not _GEMINI_AVAILABLE:
        print("⚠  Gemini API not available.")
        print("   pip install google-generativeai")
        print("   set GOOGLE_API_KEY=your_key")
        return

    if not IN_CSV.exists():
        print(f"⚠  Input not found: {IN_CSV}")
        print("   Run data/preprocess_posts.py first.")
        return

    print(f"Loading posts from {IN_CSV} …")
    df = pd.read_csv(IN_CSV)
    print(f"  {len(df):,} rows loaded\n")

    # ── Option A: LLM-only classification (no hybrid routing) ─────────────────
    if not RULE_CSV.exists():
        print("Rule-based results not found — running LLM on full corpus …")
        print("(This may take a while and consume API quota.)\n")

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 50 == 0:
                print(f"  [{i}/{len(df)}] classifying …")
            result = classify_post_llm(str(row.get("processed_text", "")))
            result["post_id"]   = row.get("post_id")
            result["timestamp"] = row.get("timestamp")
            results.append(result)
            time.sleep(0.2)

        results_df = pd.DataFrame(results)
        results_df.to_csv(OUT_LLM_CSV, index=False)
        print(f"\nSaved {len(results_df):,} rows → {OUT_LLM_CSV}")

    else:
        # ── Option B: Hybrid routing (recommended) ────────────────────────────
        print("Rule-based results found — running hybrid pipeline …\n")
        rule_df = pd.read_csv(RULE_CSV)

        # Try to load seed bank for embedding stage
        seed_bank = None
        if RAW.joinpath("seed_bank.pkl").exists():
            import pickle
            with open(RAW / "seed_bank.pkl", "rb") as f:
                seed_bank = pickle.load(f)
            print(f"  Seed bank loaded: {list(seed_bank.keys())}")

        hybrid_df = classify_corpus_hybrid(df, rule_df, seed_bank)
        print_routing_summary(hybrid_df)

        hybrid_df.to_csv(OUT_HYBRID_CSV, index=False)
        print(f"\nSaved hybrid results → {OUT_HYBRID_CSV}")

        # ── Method comparison ─────────────────────────────────────────────────
        if EMBEDDING_CSV.exists():
            print("\nRunning 3-method comparison …")
            emb_df = pd.read_csv(EMBEDDING_CSV)
            # Add ground truth if risk_level was in original data
            if "risk_level" in df.columns:
                rule_df["true_label"] = df["risk_level"].values
                emb_df["true_label"]  = df["risk_level"].values
                hybrid_df["true_label"] = df["risk_level"].values
            cmp_df = run_full_comparison(rule_df, emb_df, hybrid_df)
            cmp_df.to_csv(OUT_COMPARE_CSV, index=False)
            print(f"Saved → {OUT_COMPARE_CSV}")

    # ── Demo: RAG spike summary ───────────────────────────────────────────────
    print("\nDemo — generating RAG spike summary …")
    sample_posts = df.head(10).to_dict("records")
    for p in sample_posts:
        p["risk_level"] = "high"
        p["processed_text"] = str(p.get("review", p.get("processed_text", "")))[:300]

    summary = generate_spike_summary(
        sample_posts,
        spike_date="2024-03",
        supporting_cdc_data={"deaths": 1247, "substance": "opioid",
                              "state": "National", "pct_change": 8.3},
    )
    print(json.dumps(summary, indent=2))

    with open(OUT_SPIKES_JSON, "w") as f:
        json.dump([summary], f, indent=2)
    print(f"Saved → {OUT_SPIKES_JSON}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
