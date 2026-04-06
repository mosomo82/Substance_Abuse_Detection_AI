"""
evaluation_report.py
====================
Member 2 — Daniel Evans | Task 2: Formal Evaluation Module

Steps:
  1. Build held-out test set (≤2,000 posts where all 3 classifiers agree,
     stratified by risk_level) → data/processed/eval_test_set.csv
  2. Compute per-method: Accuracy, macro-F1, per-class F1, ROC-AUC (one-vs-rest)
  3. Per-substance F1 breakdown (opioid, stimulant, benzo, alcohol)
  4. Build Plotly figures:
       • Multi-class ROC curves  → roc_curves_{method}.html
       • Confusion matrix heatmap → confusion_matrix_{method}.html
       • Per-substance F1 bar chart → per_substance_f1.html
  5. All figures → data/processed/eval_figures/
     All numeric results → data/processed/eval_metrics.json

Ground truth:  data/raw/posts_classified.csv  (column: risk_level)
               Heuristic labels used as pseudo-ground truth (same labels used
               to train / pseudo-label the finetuned model).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
PROCESSED  = ROOT / "data" / "processed"
RAW        = ROOT / "data" / "raw"
FIGURES    = PROCESSED / "eval_figures"

ENSEMBLE_CSV   = PROCESSED / "ensemble_results.csv"
RULE_CSV       = PROCESSED / "rule_based_results.csv"
EMBEDDING_CSV  = PROCESSED / "embedding_results.csv"
FINETUNED_CSV  = PROCESSED / "finetuned_results.csv"
CLASSIFIED_CSV = RAW / "posts_classified.csv"
TEST_SET_CSV   = PROCESSED / "eval_test_set.csv"
METRICS_JSON   = PROCESSED / "eval_metrics.json"

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_CLASSES    = ["low", "medium", "high"]      # ordered for label encoding
LABEL_TO_INT    = {r: i for i, r in enumerate(RISK_CLASSES)}
INT_TO_LABEL    = {i: r for r, i in LABEL_TO_INT.items()}
TARGET_TEST_N   = 2000
SUBSTANCES_EVAL = ["opioid", "stimulant", "benzo", "alcohol"]

METHOD_COLORS = {
    "rule_based": "#2196F3",
    "embedding":  "#4CAF50",
    "finetuned":  "#FF9800",
    "ensemble":   "#9C27B0",
}

# ── Step 1: Load & merge all classifier outputs ───────────────────────────────

def _load_master() -> pd.DataFrame:
    """
    Merge all classifier results with ground-truth labels from posts_classified.csv.
    Alignment is positional: ensemble post_id maps to classified row index.
    """
    ens = pd.read_csv(ENSEMBLE_CSV)
    rb  = pd.read_csv(RULE_CSV)
    emb = pd.read_csv(EMBEDDING_CSV)
    ft  = pd.read_csv(FINETUNED_CSV)
    cls = pd.read_csv(CLASSIFIED_CSV)

    n   = len(ens)
    gt  = cls.iloc[:n][["substance", "risk_level"]].reset_index(drop=True)
    gt  = gt.rename(columns={"risk_level": "gt_risk_level"})

    # rule_based: positional alignment
    rb_cols = ["risk_level", "combined_score"]
    rb_sel  = rb[rb_cols].reset_index(drop=True).rename(
        columns={"risk_level": "rb_risk_level", "combined_score": "rb_score"}
    )

    # embedding: positional alignment
    emb_cols = ["risk_level", "prob_high", "prob_medium", "prob_low"]
    emb_sel  = emb[emb_cols].reset_index(drop=True).rename(
        columns={"risk_level": "emb_risk_level"}
    )

    # finetuned: join on post_id (35 rows may be missing → left join)
    ft_sel = ft[["post_id", "risk_level", "confidence"]].rename(
        columns={"risk_level": "ft_risk_level", "confidence": "ft_confidence"}
    )

    master = (
        ens[["post_id", "rule_risk_level", "embedding_risk_level",
             "finetuned_risk_level", "final_risk_level",
             "ensemble_confidence", "method_agreement"]]
        .reset_index(drop=True)
    )
    master = pd.concat([master, gt, rb_sel, emb_sel], axis=1)
    master = master.merge(ft_sel, on="post_id", how="left")

    return master


# ── Step 2: Build stratified held-out test set ────────────────────────────────

def _build_test_set(master: pd.DataFrame, n: int = TARGET_TEST_N,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Build a 2,000-post stratified held-out test set from the full corpus.

    Rows where all 3 classifiers (rule, embedding, finetuned) agree are flagged
    as `all_classifiers_agree` — these are the highest-confidence pseudo-labeled
    examples within the test set.  Evaluating on the full stratified sample is
    necessary to show meaningful per-method differences (on agreement-only rows
    every method makes the same hard prediction by construction).
    """
    pool = master.dropna(subset=["ft_risk_level"]).copy()

    # Mark agreement rows for reference
    pool["all_classifiers_agree"] = (
        (pool["rb_risk_level"] == pool["emb_risk_level"]) &
        (pool["emb_risk_level"] == pool["ft_risk_level"])
    )
    agree_n = pool["all_classifiers_agree"].sum()
    print(f"      Full pool: {len(pool):,} rows  "
          f"({agree_n:,} with 3-way agreement = "
          f"{agree_n/len(pool):.1%})")

    # Stratified sample across all rows
    strata_counts = pool["gt_risk_level"].value_counts()
    total         = min(n, len(pool))
    rng           = np.random.default_rng(random_state)

    frames = []
    for label, cnt in strata_counts.items():
        proportion = cnt / len(pool)
        k          = max(1, round(proportion * total))
        k          = min(k, cnt)
        idx        = rng.choice(pool[pool["gt_risk_level"] == label].index,
                                size=k, replace=False)
        frames.append(pool.loc[idx])

    test = pd.concat(frames).sample(frac=1, random_state=random_state)
    agree_in_test = test["all_classifiers_agree"].sum()
    print(f"      Test set size: {len(test):,}  "
          f"(strata: {test['gt_risk_level'].value_counts().to_dict()})")
    print(f"      Agreement rows in test set: {agree_in_test:,} "
          f"({agree_in_test/len(test):.1%})")

    FIGURES.mkdir(parents=True, exist_ok=True)
    test.reset_index(drop=True).to_csv(TEST_SET_CSV, index=False)
    print(f"      Saved → {TEST_SET_CSV}")
    return test


# ── Step 3: Soft probabilities ────────────────────────────────────────────────

def _rule_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Derive per-class soft probabilities from combined_score (0–1):
      prob_high   = score
      prob_medium = 1 - |score - 0.35| (peaks at 0.35)
      prob_low    = 1 - score
    Rows normalised to sum to 1.
    """
    s = df["rb_score"].fillna(0.0).values
    ph = s
    pm = 1.0 - np.abs(s - 0.35)
    pl = 1.0 - s
    mat = np.stack([pl, pm, ph], axis=1)          # order: low, medium, high
    mat = np.clip(mat, 0.0, None)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def _emb_probs(df: pd.DataFrame) -> np.ndarray:
    """Embedding probabilities are already per-class."""
    return df[["prob_low", "prob_medium", "prob_high"]].fillna(1/3).values


def _finetuned_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct per-class probabilities from predicted class + confidence.
    Predicted class gets `confidence`; remaining 1-confidence split equally.
    """
    classes = RISK_CLASSES
    n       = len(df)
    mat     = np.full((n, len(classes)), 0.0)
    for i, (rl, conf) in enumerate(
            zip(df["ft_risk_level"].fillna("low"),
                df["ft_confidence"].fillna(0.5))):
        conf = float(conf)
        pred_idx = LABEL_TO_INT.get(rl, 0)
        other    = (1.0 - conf) / (len(classes) - 1)
        for j in range(len(classes)):
            mat[i, j] = conf if j == pred_idx else other
    return mat   # columns: low, medium, high


def _ensemble_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct per-class probabilities from final_risk_level + ensemble_confidence.
    """
    classes = RISK_CLASSES
    n       = len(df)
    mat     = np.full((n, len(classes)), 0.0)
    for i, (rl, conf) in enumerate(
            zip(df["final_risk_level"].fillna("low"),
                df["ensemble_confidence"].fillna(0.5))):
        conf     = float(conf)
        pred_idx = LABEL_TO_INT.get(rl, 0)
        other    = (1.0 - conf) / (len(classes) - 1)
        for j in range(len(classes)):
            mat[i, j] = conf if j == pred_idx else other
    return mat


PROB_BUILDERS = {
    "rule_based": ("rb_risk_level",    _rule_probs),
    "embedding":  ("emb_risk_level",   _emb_probs),
    "finetuned":  ("ft_risk_level",    _finetuned_probs),
    "ensemble":   ("final_risk_level", _ensemble_probs),
}


# ── Step 4: Compute classification metrics ────────────────────────────────────

def _encode(series: pd.Series) -> np.ndarray:
    return series.map(LABEL_TO_INT).fillna(0).astype(int).values


def compute_metrics(
    test: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute Accuracy, macro-F1, per-class F1, and ROC-AUC (one-vs-rest)
    for each method on the held-out test set.
    """
    y_true  = _encode(test["gt_risk_level"])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])   # for OvR ROC

    metrics: dict[str, dict] = {}

    for method, (pred_col, prob_fn) in PROB_BUILDERS.items():
        y_pred     = _encode(test[pred_col])
        y_probs    = prob_fn(test)

        acc        = accuracy_score(y_true, y_pred)
        macro_f1   = f1_score(y_true, y_pred, average="macro",
                              zero_division=0)
        per_class  = f1_score(y_true, y_pred, average=None,
                              labels=[0, 1, 2], zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true_bin, y_probs,
                                    multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = float("nan")

        metrics[method] = {
            "accuracy":      round(float(acc),       4),
            "macro_f1":      round(float(macro_f1),  4),
            "per_class_f1":  {
                c: round(float(per_class[i]), 4)
                for i, c in enumerate(RISK_CLASSES)
            },
            "roc_auc_macro": round(float(roc_auc), 4),
            "y_pred":        y_pred.tolist(),
            "y_probs":       y_probs.tolist(),
        }
        print(
            f"      {method:10s}: acc={acc:.3f}  "
            f"macro-F1={macro_f1:.3f}  AUC={roc_auc:.3f}"
        )

    return metrics


# ── Step 5: Per-substance F1 ──────────────────────────────────────────────────

def compute_substance_f1(
    test: pd.DataFrame,
    metrics: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """
    For each substance in SUBSTANCES_EVAL, compute macro-F1 per method.
    Returns {substance: {method: f1}}.
    """
    y_true = _encode(test["gt_risk_level"])
    results: dict[str, dict[str, float]] = {}

    for sub in SUBSTANCES_EVAL:
        mask = test["substance"] == sub
        if mask.sum() < 5:
            print(f"      ⚠  {sub}: only {mask.sum()} test rows — skipping")
            continue
        yt = y_true[mask]
        sub_f1: dict[str, float] = {}
        for method, (pred_col, _) in PROB_BUILDERS.items():
            yp = _encode(test[pred_col])[mask]
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            sub_f1[method] = round(float(f1), 4)
        results[sub] = sub_f1
        print(f"      {sub}: " + "  ".join(f"{m}={v:.3f}"
              for m, v in sub_f1.items()))

    return results


# ── Step 6: Plotly figures ────────────────────────────────────────────────────

def _roc_curves_figure(
    y_true: np.ndarray,
    metrics: dict[str, dict],
) -> go.Figure:
    """
    Multi-class ROC curves for all methods on one figure (2×2 subplots, one per method).
    Uses one-vs-rest per class.
    """
    from sklearn.metrics import roc_curve

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    method_names = list(PROB_BUILDERS.keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m.replace("_", " ").title() for m in method_names],
        shared_xaxes=False,
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )
    CLASS_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), method in zip(positions, method_names):
        y_probs = np.array(metrics[method]["y_probs"])
        for cls_idx, cls_name in enumerate(RISK_CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_probs[:, cls_idx])
            auc_val = roc_auc_score(y_bin[:, cls_idx], y_probs[:, cls_idx])
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"{cls_name} (AUC={auc_val:.2f})",
                    line=dict(color=CLASS_COLORS[cls_name], width=2),
                    legendgroup=f"{method}_{cls_name}",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row, col=col,
            )
        # Diagonal
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                       line=dict(dash="dash", color="gray", width=1),
                       showlegend=False),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="FPR", row=row, col=col)
        fig.update_yaxes(title_text="TPR", row=row, col=col)

    fig.update_layout(
        title="Multi-Class ROC Curves (One-vs-Rest) — All Methods",
        height=800,
        template="plotly_white",
    )
    return fig


def _confusion_matrix_figure(
    y_true: np.ndarray,
    metrics: dict[str, dict],
    method: str,
) -> go.Figure:
    y_pred = np.array(metrics[method]["y_pred"])
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # Normalise row-wise (true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    labels = [rl.upper() for rl in RISK_CLASSES]
    text   = [[f"{cm_norm[r][c]:.1%}<br>({cm[r][c]})"
               for c in range(3)] for r in range(3)]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title="Rate"),
    ))
    title = method.replace("_", " ").title()
    fig.update_layout(
        title=f"Confusion Matrix — {title}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        template="plotly_white",
        height=450,
    )
    return fig


def _substance_f1_figure(substance_f1: dict[str, dict[str, float]]) -> go.Figure:
    substances = list(substance_f1.keys())
    methods    = list(PROB_BUILDERS.keys())

    fig = go.Figure()
    for method in methods:
        y_vals = [substance_f1[s].get(method, 0.0) for s in substances]
        fig.add_trace(go.Bar(
            name=method.replace("_", " ").title(),
            x=substances,
            y=y_vals,
            marker_color=METHOD_COLORS.get(method, "#607D8B"),
            text=[f"{v:.3f}" for v in y_vals],
            textposition="outside",
        ))

    fig.update_layout(
        title="Per-Substance Macro-F1 by Classifier Method",
        xaxis_title="Substance Class",
        yaxis_title="Macro-F1",
        yaxis=dict(range=[0, 1.05]),
        barmode="group",
        template="plotly_white",
        legend_title="Method",
        height=500,
    )
    return fig


def _overall_metrics_figure(metrics: dict[str, dict]) -> go.Figure:
    """Summary bar chart: Accuracy, Macro-F1, ROC-AUC per method."""
    method_labels = [m.replace("_", " ").title() for m in metrics]
    acc    = [metrics[m]["accuracy"]      for m in metrics]
    mf1    = [metrics[m]["macro_f1"]      for m in metrics]
    auc    = [metrics[m]["roc_auc_macro"] for m in metrics]

    fig = go.Figure()
    for label, vals, color in [
        ("Accuracy",    acc,  "#2196F3"),
        ("Macro-F1",    mf1,  "#4CAF50"),
        ("ROC-AUC",     auc,  "#9C27B0"),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=method_labels,
            y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        title="Classifier Performance Summary",
        xaxis_title="Method",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]),
        barmode="group",
        template="plotly_white",
        height=450,
    )
    return fig


# ── Step 7: Save figures ──────────────────────────────────────────────────────

def _save(fig: go.Figure, name: str) -> None:
    path = FIGURES / name
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"      Saved → {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 60)
    print("Evaluation Report — Daniel Evans (M2)")
    print("=" * 60)

    FIGURES.mkdir(parents=True, exist_ok=True)

    # ── 1. Load master ────────────────────────────────────────────────────────
    print("\n[1/6] Loading classifier outputs + ground truth...")
    master = _load_master()
    print(f"      Master rows: {len(master):,}")

    # ── 2. Build test set ─────────────────────────────────────────────────────
    print(f"\n[2/6] Building stratified held-out test set (target={TARGET_TEST_N})...")
    test = _build_test_set(master)

    y_true = _encode(test["gt_risk_level"])

    # ── 3. Classification metrics ─────────────────────────────────────────────
    print("\n[3/6] Computing per-method classification metrics...")
    metrics = compute_metrics(test)

    # ── 4. Per-substance F1 ───────────────────────────────────────────────────
    print("\n[4/6] Computing per-substance F1...")
    substance_f1 = compute_substance_f1(test, metrics)

    # ── 5. Build & save figures ───────────────────────────────────────────────
    print("\n[5/6] Building Plotly figures...")

    # 5a. ROC curves (all methods, one figure)
    roc_fig = _roc_curves_figure(y_true, metrics)
    _save(roc_fig, "roc_curves_all_methods.html")

    # 5b. Confusion matrices (one per method)
    for method in PROB_BUILDERS:
        cm_fig = _confusion_matrix_figure(y_true, metrics, method)
        _save(cm_fig, f"confusion_matrix_{method}.html")

    # 5c. Per-substance F1 bar chart
    if substance_f1:
        sub_fig = _substance_f1_figure(substance_f1)
        _save(sub_fig, "per_substance_f1.html")

    # 5d. Overall summary bar chart
    summary_fig = _overall_metrics_figure(metrics)
    _save(summary_fig, "classifier_summary.html")

    # ── 6. Save numeric results ───────────────────────────────────────────────
    print("\n[6/6] Saving eval_metrics.json...")
    # Strip internal arrays (y_pred, y_probs) before serialising to JSON
    metrics_clean = {
        m: {k: v for k, v in vals.items() if k not in ("y_pred", "y_probs")}
        for m, vals in metrics.items()
    }

    output = {
        "test_set": {
            "n":                       len(test),
            "strata":                  test["gt_risk_level"].value_counts().to_dict(),
            "all_classifiers_agree_n": int(test["all_classifiers_agree"].sum()),
            "description": (
                "2,000-post stratified sample from full corpus; "
                "agreement rows flagged in all_classifiers_agree column"
            ),
        },
        "per_method_metrics": metrics_clean,
        "per_substance_f1":   substance_f1,
        "figures": [
            "roc_curves_all_methods.html",
            "confusion_matrix_rule_based.html",
            "confusion_matrix_embedding.html",
            "confusion_matrix_finetuned.html",
            "confusion_matrix_ensemble.html",
            "per_substance_f1.html",
            "classifier_summary.html",
        ],
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"      Saved → {METRICS_JSON}")

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Test set: {len(test):,} posts")
    for method, vals in metrics_clean.items():
        print(
            f"  {method:10s}: acc={vals['accuracy']:.3f}  "
            f"macro-F1={vals['macro_f1']:.3f}  "
            f"AUC={vals['roc_auc_macro']:.3f}"
        )
    if substance_f1:
        print("\n  Per-substance Macro-F1 (ensemble):")
        for sub, val in substance_f1.items():
            print(f"    {sub:12s}: {val.get('ensemble', 0):.3f}")

    return output


if __name__ == "__main__":
    run()
