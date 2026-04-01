"""
Standalone script: generates data/processed/method_comparison.csv
from the three existing classifier result CSVs.

Ground-truth labels are derived from posts_classified.csv (heuristic risk labels
based on drug-review phrase matching + rating thresholds). These serve as a
proxy ground truth for computing accuracy / F1 — no API key required.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).parent.parent.parent   # project root
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

# ── Load proxy ground truth from posts_classified.csv ─────────────────────────
_gt_path = RAW / "posts_classified.csv"
ground_truth = None
if _gt_path.exists():
    _gt_df = pd.read_csv(str(_gt_path))
    if "risk_level" in _gt_df.columns:
        ground_truth = _gt_df["risk_level"].str.strip().str.lower().values
        print(f"Ground truth loaded: {len(ground_truth)} labels "
              f"({pd.Series(ground_truth).value_counts().to_dict()})")
    else:
        print("posts_classified.csv has no 'risk_level' column — accuracy metrics skipped.")
else:
    print("posts_classified.csv not found — accuracy metrics skipped.")

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
    df["risk_level"] = df[col].str.strip().str.lower()
    n = len(df)
    vc = df["risk_level"].value_counts()
    avg_lat = round(df["latency_ms"].mean(), 1) if "latency_ms" in df.columns else 0.0

    # Compute accuracy / F1 if ground truth is available and lengths match
    acc = float("nan")
    f1_high = float("nan")
    f1_macro = float("nan")
    note = "Proxy ground truth from posts_classified.csv (heuristic risk labels)"
    if ground_truth is not None:
        # Align lengths — use min(n, len(gt)) rows
        _min_n = min(n, len(ground_truth))
        _pred = df["risk_level"].iloc[:_min_n].values
        _true = ground_truth[:_min_n]
        _labels = ["low", "medium", "high"]
        try:
            acc = round(accuracy_score(_true, _pred), 4)
            f1_high = round(f1_score(_true, _pred, labels=["high"],
                                      average="macro", zero_division=0), 4)
            f1_macro = round(f1_score(_true, _pred, labels=_labels,
                                       average="macro", zero_division=0), 4)
            note = f"Proxy GT ({_min_n} posts aligned); labels: {_labels}"
            print(f"\n{name} vs proxy GT ({_min_n} posts):")
            print(classification_report(_true, _pred, labels=_labels,
                                        zero_division=0))
        except Exception as exc:
            note = f"Metric error: {exc}"

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
        "Accuracy":       acc,
        "F1 (high)":      f1_high,
        "F1 (macro)":     f1_macro,
        "Note":           note,
    })

out = PROCESSED / "method_comparison.csv"
pd.DataFrame(rows).to_csv(str(out), index=False)
print("\nSaved:", out)
for r in rows:
    acc_str = f"  Acc={r['Accuracy']:.3f}  F1={r['F1 (macro)']:.3f}" if not (isinstance(r['Accuracy'], float) and np.isnan(r['Accuracy'])) else ""
    print(f"  {r['Method']:<12} High={r['High %']}%  Med={r['Medium %']}%  Low={r['Low %']}%{acc_str}")
