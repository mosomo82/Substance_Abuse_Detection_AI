"""
Standalone script: generates data/processed/method_comparison.csv
from the three existing classifier result CSVs.
No API key required.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent   # project root
PROCESSED = ROOT / "data" / "processed"

files = {
    "Rule-based": ("rule_based_results.csv",  "risk_level"),
    "Embedding":  ("embedding_results.csv",   "risk_level"),
    "Ensemble":   ("ensemble_results.csv",    "final_risk_level"),
}

rows = []
for name, (fname, col) in files.items():
    p = PROCESSED / fname
    if not p.exists():
        print(f"Missing: {fname}")
        continue
    df = pd.read_csv(str(p))
    if col not in df.columns:
        print(f"No '{col}' in {fname} — columns: {list(df.columns[:5])}")
        continue
    df["risk_level"] = df[col]  # normalise to common name
    n = len(df)
    vc = df["risk_level"].value_counts()
    avg_lat = round(df["latency_ms"].mean(), 1) if "latency_ms" in df.columns else 0.0
    rows.append({
        "Method":         name,
        "Total posts":    n,
        "High risk":      int(vc.get("high", 0)),
        "Medium risk":    int(vc.get("medium", 0)),
        "Low risk":       int(vc.get("low", 0)),
        "High %":         round(vc.get("high", 0) / n * 100, 1),
        "Medium %":       round(vc.get("medium", 0) / n * 100, 1),
        "Low %":          round(vc.get("low", 0) / n * 100, 1),
        "Avg latency ms": avg_lat,
        "Accuracy":       float("nan"),
        "F1 (high)":      float("nan"),
        "F1 (macro)":     float("nan"),
        "Note":           "No ground-truth labels; accuracy metrics require Gemini run",
    })

out = PROCESSED / "method_comparison.csv"
pd.DataFrame(rows).to_csv(str(out), index=False)
print("Saved:", out)
for r in rows:
    print(f"  {r['Method']:<12} High={r['High %']}%  Med={r['Medium %']}%  Low={r['Low %']}%")
