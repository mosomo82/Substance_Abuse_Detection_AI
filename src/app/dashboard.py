"""
dashboard.py
============
Streamlit dashboard for the Substance Abuse Narrative Evolution analysis.

Tabs:
    1. Drift          — dual-axis drift score + high-risk % over time
    2. Topics         — stacked area chart of 8 topic prevalence scores
    3. UMAP Trajectory— 2-D narrative path through semantic space
    4. Alerts         — rising-topic early-warning alert table

Sidebar controls:
    freq             — window frequency: Weekly (W) or Monthly (M)
    min_posts        — minimum posts per window (slider 5–50)
    drift_threshold  — z-score threshold for drift event flagging (1.5–3.0)
    Re-run Analysis  — recompute pipeline with updated parameters

Run:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent   # project root
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "agents"))
sys.path.insert(0, str(ROOT / "src"))

DATA      = ROOT / "data"
PROCESSED = DATA / "processed"
OUT_DIR   = PROCESSED / "narrative"
FIG_DIR   = OUT_DIR / "figures"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Substance Abuse Narrative Evolution",
    page_icon="📊",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Cached data loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading window data …")
def _load_window_df() -> pd.DataFrame:
    path = OUT_DIR / "window_df.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(str(path))


@st.cache_data(show_spinner="Loading topic data …")
def _load_topic_df() -> pd.DataFrame:
    path = OUT_DIR / "topic_df.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(str(path))


@st.cache_data(show_spinner="Loading trajectory …")
def _load_trajectory_df() -> pd.DataFrame:
    path = OUT_DIR / "trajectory_df.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path), parse_dates=["period_start"])


@st.cache_data(show_spinner="Loading alerts …")
def _load_warning_report() -> pd.DataFrame:
    path = OUT_DIR / "warning_report.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path))
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading rising events …")
def _load_rising_events() -> dict:
    path = OUT_DIR / "rising_events.json"
    if not path.exists():
        return {}
    with open(str(path)) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading method comparison …")
def _load_method_comparison() -> pd.DataFrame:
    path = PROCESSED / "method_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path))


@st.cache_data(show_spinner="Loading spike summaries …")
def _load_spike_summaries() -> list:
    path = PROCESSED / "spike_summaries.json"
    if not path.exists():
        return []
    with open(str(path)) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


@st.cache_data(show_spinner="Loading cross-correlations …")
def _load_correlations() -> dict:
    path = PROCESSED / "correlations.json"
    if not path.exists():
        return {}
    with open(str(path)) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading CDC state data …")
def _load_cdc_state() -> pd.DataFrame:
    path = DATA / "raw" / "cdc_overdose_data.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(path))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Re-run helper (calls narrative_evolution pipeline on-demand)
# ══════════════════════════════════════════════════════════════════════════════

def _rerun_analysis(freq: str, min_posts: int, drift_threshold: float) -> None:
    """Reload inputs and re-run the full narrative evolution pipeline."""
    try:
        from utils.narrative_evolution import (
            load_analysis_inputs,
            run_narrative_evolution_analysis,
            detect_rising_topics,
            visualize,
        )
        from signal_pipeline import load_cdc_overdose
    except ImportError as exc:
        st.error(f"Import error: {exc}")
        return

    with st.spinner("Running narrative evolution analysis …"):
        posts_df, embeddings = load_analysis_inputs()

        cdc_df = None
        cdc_path = DATA / "raw" / "cdc_overdose_data.csv"
        if cdc_path.exists():
            cdc_df = load_cdc_overdose(cdc_path)

        results = run_narrative_evolution_analysis(
            posts_df   = posts_df,
            embeddings = embeddings,
            cdc_df     = cdc_df,
            freq       = freq,
        )

        # Re-run drift with custom threshold (override is_drift_event)
        win = results["window_df"]
        if "drift_score" in win.columns:
            valid = win["drift_score"].dropna()
            if len(valid) > 1:
                d_mean = valid.mean()
                d_std  = valid.std()
                win["is_drift_event"] = (
                    (win["drift_score"] - d_mean) / (d_std if d_std > 0 else 1.0)
                ) > drift_threshold
            results["window_df"] = win

        # Re-detect rising topics with custom min_posts already applied above
        visualize(results)

    # Clear Streamlit caches so reloaded data is fresh
    st.cache_data.clear()
    st.success("Analysis complete — refresh to see updated results.")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙ Parameters")
    freq = st.selectbox(
        "Window frequency",
        options=["W", "M"],
        format_func=lambda v: "Weekly" if v == "W" else "Monthly",
        index=0,
    )
    min_posts = st.slider("Min posts per window", min_value=5, max_value=50,
                           value=10, step=5)
    drift_threshold = st.slider("Drift event z-score threshold",
                                 min_value=1.5, max_value=3.0,
                                 value=2.0, step=0.1)

    st.markdown("---")
    if st.button("▶ Re-run Analysis", use_container_width=True):
        _rerun_analysis(freq, min_posts, drift_threshold)

    st.markdown("---")
    st.caption("Outputs saved to `data/processed/narrative/`")
    st.caption("Figures saved to `data/processed/narrative/figures/`")

    st.markdown("---")
    st.markdown("**🔒 Privacy & Ethics**")
    st.caption(
        "All outputs are **population-level aggregates** only. "
        "PII is scrubbed at ingestion (regex + spaCy NER). "
        "No individual is identified or tracked. "
        "Public datasets exclusively — CDC, NSDUH, NIDA, Kaggle."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main content
# ══════════════════════════════════════════════════════════════════════════════

st.title("📊 Substance Abuse Risk Intelligence Dashboard")
st.markdown(
    "**NSF NRT Research-A-Thon 2026 — Challenge 1, Track B** · "
    "End-to-end AI pipeline: social signal detection → temporal drift → "
    "early-warning alerts → explainable analyst reports."
)

# Check if outputs exist; prompt user to run the pipeline if not
_missing = not (OUT_DIR / "window_df.parquet").exists()
if _missing:
    st.warning(
        "No narrative analysis outputs found. "
        "Click **▶ Re-run Analysis** in the sidebar (or run "
        "`python src/narrative_evolution.py`) to generate them first."
    )

tab_drift, tab_topics, tab_umap, tab_alerts, tab_eval, tab_reports, tab_geo, tab_recs = st.tabs([
    "📉 Drift", "📈 Topics", "🔵 UMAP Trajectory", "🚨 Alerts",
    "📊 Model Evaluation", "📝 Analyst Reports", "🗺 Geographic", "💡 Recommendations"
])

# ── Tab 1: Drift ──────────────────────────────────────────────────────────────
with tab_drift:
    st.subheader("Narrative Drift & High-Risk Post Percentage")
    st.markdown(
        "Cosine distance between adjacent weekly embedding centroids "
        "(left axis) vs. percentage of high-risk posts (right axis). "
        "⭐ markers indicate statistically significant drift events."
    )

    html_path = FIG_DIR / "dual_axis_drift.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=False)
    else:
        win_df = _load_window_df()
        if len(win_df) == 0:
            st.info("Run the analysis to generate drift data.")
        else:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                win_df["period_start"] = pd.to_datetime(win_df["period_start"])
                valid = win_df.dropna(subset=["drift_score"])

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(
                    x=valid["period_start"], y=valid["drift_score"],
                    name="Narrative drift",
                    line=dict(color="#E74C3C", width=2),
                    mode="lines+markers",
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=win_df["period_start"], y=win_df["pct_high"],
                    name="% high-risk",
                    line=dict(color="#3498DB", width=2, dash="dot"),
                ), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.error("plotly is required for inline charts. "
                         "pip install plotly")

    # Summary metrics
    win_df = _load_window_df()
    if len(win_df) > 0 and "drift_score" in win_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Windows analysed", len(win_df))
        col2.metric("Drift events",
                    int(win_df["is_drift_event"].sum())
                    if "is_drift_event" in win_df.columns else "—")
        col3.metric("Peak drift",
                    f"{win_df['drift_score'].max():.4f}"
                    if win_df["drift_score"].notna().any() else "—")
        col4.metric("Avg % high-risk",
                    f"{win_df['pct_high'].mean():.0%}"
                    if "pct_high" in win_df.columns else "—")

# ── Tab 2: Topics ─────────────────────────────────────────────────────────────
with tab_topics:
    st.subheader("Topic Prevalence Over Time")
    st.markdown(
        "Stacked area chart showing how community language aligns with "
        "8 semantic topic anchors across each time window. "
        "Score = 75th-percentile cosine similarity to the anchor phrase."
    )

    html_path2 = FIG_DIR / "topic_prevalence_area.html"
    if html_path2.exists():
        with open(html_path2, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=False)
    else:
        topic_df = _load_topic_df()
        if len(topic_df) == 0:
            st.info("Run the analysis to generate topic data.")
        else:
            score_cols = [c for c in topic_df.columns if c.endswith("_score")]
            try:
                import plotly.express as px
                long_df = topic_df.melt(
                    id_vars=["period_start"],
                    value_vars=score_cols,
                    var_name="topic", value_name="score",
                )
                long_df["topic"] = long_df["topic"].str.replace(
                    "_score", "", regex=False
                ).str.replace("_", " ", regex=False).str.title()
                fig2 = px.area(
                    long_df, x="period_start", y="score",
                    color="topic", title="Topic Prevalence",
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.error("plotly is required.  pip install plotly")

    # Rising events sidebar
    rising_events = _load_rising_events()
    if rising_events:
        with st.expander(f"Rising topic events ({len(rising_events)} periods)"):
            for period_str, topics in list(rising_events.items())[:10]:
                st.markdown(f"**{period_str}**")
                for t in topics:
                    badge = "🔴" if t["z_score"] >= 3 else "🟠" if t["z_score"] >= 2 else "🟡"
                    st.markdown(
                        f"  {badge} `{t['topic']}` — z={t['z_score']:.2f}, "
                        f"+{t['pct_increase']:.1f}%"
                    )

# ── Tab 3: UMAP Trajectory ────────────────────────────────────────────────────
with tab_umap:
    st.subheader("UMAP Narrative Trajectory")
    st.markdown(
        "Each point is a weekly window's embedding centroid projected into 2-D. "
        "**Colour** = % high-risk posts. **Size** ∝ post volume. "
        "The path shows how community narrative evolved over time."
    )

    html_path3 = FIG_DIR / "umap_trajectory.html"
    if html_path3.exists():
        with open(html_path3, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=650, scrolling=False)
    else:
        traj_df = _load_trajectory_df()
        if len(traj_df) == 0:
            st.info("Run the analysis to generate the UMAP trajectory.")
        else:
            try:
                import plotly.express as px
                fig3 = px.scatter(
                    traj_df, x="x", y="y",
                    color="pct_high" if "pct_high" in traj_df.columns else None,
                    size="post_count" if "post_count" in traj_df.columns else None,
                    hover_data=["period_start"] + (
                        ["annotation"] if "annotation" in traj_df.columns else []
                    ),
                    color_continuous_scale="Reds",
                    title="UMAP Narrative Trajectory",
                )
                st.plotly_chart(fig3, use_container_width=True)
            except ImportError:
                st.error("plotly is required.  pip install plotly")

    # Data table
    traj_df = _load_trajectory_df()
    if len(traj_df) > 0:
        with st.expander("View trajectory data"):
            show_cols = [c for c in
                         ["period_start", "post_count", "pct_high",
                          "drift_score", "is_drift_event",
                          "drift_direction", "annotation"]
                         if c in traj_df.columns]
            st.dataframe(traj_df[show_cols], use_container_width=True)

# ── Tab 4: Alerts ─────────────────────────────────────────────────────────────
with tab_alerts:
    st.subheader("Early Warning Alerts")
    st.markdown(
        "Periods where a topic's prevalence score rose significantly "
        "above its recent baseline, together with any lead-lag correlation "
        "against CDC overdose deaths."
    )

    html_path4 = FIG_DIR / "rising_alerts_table.html"
    if html_path4.exists():
        with open(html_path4, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=500, scrolling=True)
    else:
        wr = _load_warning_report()
        if len(wr) == 0:
            st.info("No alerts generated yet. Run the analysis to populate.")
        else:
            LEVEL_CSS = {
                "critical": "background-color: #FFCCCC",
                "elevated": "background-color: #FFE5CC",
                "watch":    "background-color: #FFFACC",
            }

            def _highlight_level(row):
                return [LEVEL_CSS.get(row.get("alert_level", ""), "")] * len(row)

            styled = wr.style.apply(_highlight_level, axis=1)
            st.dataframe(styled, use_container_width=True, height=500)

    # Summary KPIs
    wr = _load_warning_report()
    if len(wr) > 0 and "alert_level" in wr.columns:
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total alerts", len(wr))
        k2.metric("Critical 🔴",
                  int((wr["alert_level"] == "critical").sum()))
        k3.metric("Elevated 🟠",
                  int((wr["alert_level"] == "elevated").sum()))
        k4.metric("Watch 🟡",
                  int((wr["alert_level"] == "watch").sum()))

        if "topic" in wr.columns:
            top_topic = wr["topic"].value_counts().idxmax()
            st.info(f"Most-flagged topic: **{top_topic}** "
                    f"({int(wr['topic'].value_counts().max())} alerts)")

    # ── Lead-Lag Correlation Panel ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📡 Social Signal → CDC Deaths: Lead-Lag Correlation")
    st.markdown(
        "Pearson *r* between the monthly social signal rate and CDC provisional "
        "overdose death counts at different time lags. A **negative lag** means "
        "the social signal **leads** (precedes) the CDC death count — indicating "
        "predictive early-warning potential. Shaded bars are statistically "
        "significant at *p* < 0.05."
    )
    _corr_data = _load_correlations()
    if _corr_data:
        try:
            import plotly.graph_objects as go

            _lag_rows = []
            for _substance, _lags in _corr_data.items():
                for _lag_str, _vals in _lags.items():
                    _lag_rows.append({
                        "substance": _substance.title(),
                        "lag_months": int(_lag_str),
                        "r": _vals.get("r", 0),
                        "p": _vals.get("p", 1),
                        "significant": _vals.get("p", 1) < 0.05,
                    })
            _lag_df = pd.DataFrame(_lag_rows)

            _corr_fig = go.Figure()
            _subst_colors = {"Opioid": "#E74C3C", "Stimulant": "#9B59B6",
                             "Cocaine": "#F39C12", "Alcohol": "#3498DB"}
            for _subst in _lag_df["substance"].unique():
                _sd = _lag_df[_lag_df["substance"] == _subst].sort_values("lag_months")
                _color = _subst_colors.get(_subst, "#95A5A6")
                _corr_fig.add_trace(go.Bar(
                    name=_subst,
                    x=_sd["lag_months"].astype(str) + "mo",
                    y=_sd["r"],
                    marker_color=[
                        _color if sig else _color.replace("#", "#") + "80"
                        for sig in _sd["significant"]
                    ],
                    marker_opacity=[1.0 if sig else 0.4 for sig in _sd["significant"]],
                    customdata=_sd[["p", "significant"]].values,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Lag: %{x}<br>r = %{y:.3f}<br>p = %{customdata[0]:.4f}"
                        "<extra></extra>"
                    ),
                ))
            _corr_fig.add_hline(y=0, line_dash="dot", line_color="gray")
            _corr_fig.update_layout(
                barmode="group",
                xaxis_title="Lag (negative = social signal leads CDC deaths)",
                yaxis_title="Pearson r",
                legend_title="Substance",
                height=400,
                annotations=[dict(
                    x=0.01, y=0.97, xref="paper", yref="paper",
                    text="Opaque = p < 0.05 (significant) | Faded = p ≥ 0.05",
                    showarrow=False, font=dict(size=11), bgcolor="rgba(255,255,255,0.7)",
                )],
            )
            st.plotly_chart(_corr_fig, use_container_width=True)

            # Highlight best finding
            _sig = _lag_df[_lag_df["significant"]].sort_values("r", ascending=False)
            if len(_sig) > 0:
                _best = _sig.iloc[0]
                _direction = "leads" if _best["lag_months"] < 0 else "lags"
                _months = abs(_best["lag_months"])
                st.success(
                    f"**Key finding:** {_best['substance']} social signal **{_direction}** "
                    f"CDC overdose deaths by **{_months} month(s)** "
                    f"(r = {_best['r']:.3f}, p = {_best['p']:.4f}). "
                    "This supports early-warning potential from social media monitoring."
                )
        except ImportError:
            st.error("plotly is required — pip install plotly")
    else:
        st.info(
            "No cross-correlation data found. Run `python src/agents/signal_pipeline.py` "
            "to compute CDC alignment and correlations."
        )

# ── Tab 5: Model Evaluation ───────────────────────────────────────────────────
with tab_eval:
    st.subheader("Model Evaluation & Method Comparison")
    st.markdown(
        "Side-by-side comparison of four classifiers: rule-based, "
        "embedding-based (sentence-BERT), fine-tuned DistilBERT, and ensemble fusion. "
        "Metrics are computed against proxy ground-truth labels from `posts_classified.csv`."
    )

    # Metrics table
    cmp_df = _load_method_comparison()
    if len(cmp_df) > 0:
        st.markdown("#### Performance Metrics")
        st.dataframe(cmp_df.style.format(na_rep="—"), use_container_width=True)
    else:
        st.info(
            "No `method_comparison.csv` found. "
            "Run `python src/eval/generate_comparison.py` to generate it."
        )

    # Risk level distribution comparison across classifiers
    st.markdown("#### Risk Level Distribution by Method")
    _clf_files = [
        ("Rule-based", "rule_based_results.csv",  "risk_level"),
        ("Embedding",  "embedding_results.csv",   "risk_level"),
        ("Finetuned",  "finetuned_results.csv",   "risk_level"),
        ("Ensemble",   "ensemble_results.csv",    "final_risk_level"),
    ]
    _clf_dfs = {}
    for _name, _fname, _col in _clf_files:
        _p = PROCESSED / _fname
        if _p.exists():
            _d = pd.read_csv(str(_p))
            if _col in _d.columns:
                _d = _d.rename(columns={_col: "risk_level"})
            if "risk_level" in _d.columns:
                _clf_dfs[_name] = _d

    if _clf_dfs:
        try:
            import plotly.graph_objects as go

            _RISK_COLORS = {"high": "#E74C3C", "medium": "#F39C12", "low": "#2ECC71"}
            _fig_eval = go.Figure()
            for _risk in ["high", "medium", "low"]:
                _fig_eval.add_trace(go.Bar(
                    name=_risk.title(),
                    x=list(_clf_dfs.keys()),
                    y=[
                        round((_clf_dfs[_m]["risk_level"] == _risk).mean() * 100, 1)
                        for _m in _clf_dfs
                    ],
                    marker_color=_RISK_COLORS[_risk],
                    text=[
                        f"{round((_clf_dfs[_m]['risk_level'] == _risk).mean() * 100, 1)}%"
                        for _m in _clf_dfs
                    ],
                    textposition="outside",
                ))
            _fig_eval.update_layout(
                barmode="group",
                yaxis_title="% of posts",
                xaxis_title="Method",
                legend_title="Risk level",
                height=420,
            )
            st.plotly_chart(_fig_eval, use_container_width=True)

            # Count summary table
            _summary_rows = []
            for _mname, _mdf in _clf_dfs.items():
                _vc = _mdf["risk_level"].value_counts()
                _total = len(_mdf)
                _summary_rows.append({
                    "Method":  _mname,
                    "Total":   _total,
                    "High":    int(_vc.get("high", 0)),
                    "Medium":  int(_vc.get("medium", 0)),
                    "Low":     int(_vc.get("low", 0)),
                    "High %":  f"{_vc.get('high', 0) / _total:.1%}",
                    "Medium %": f"{_vc.get('medium', 0) / _total:.1%}",
                })
            st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True)

        except ImportError:
            st.error("plotly is required — pip install plotly")
    else:
        st.info("Run the classifier scripts first to generate result CSVs.")

    # ── Cluster Quality Metrics ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Embedding Cluster Quality Metrics")
    _cm_path = PROCESSED / "cluster_metrics.json"
    if _cm_path.exists():
        with open(str(_cm_path)) as _cmf:
            _cm = json.load(_cmf)

        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _sil = _cm.get("silhouette", {})
        _mc1.metric("Silhouette Score", _sil.get("silhouette_score", "—"),
                    help="Cosine-distance cluster quality. +1 = ideal, 0 = overlapping.")
        _mc2.metric("Interpretation", _sil.get("interpretation", "—"))

        _pp = _cm.get("perplexity", {})
        _mc3.metric("Slang Perplexity", _pp.get("avg_perplexity_slang_terms", "—"),
                    help="Lower = LM better models domain slang vocabulary.")
        _mc4.metric("Domain Coverage Ratio",
                    f"{_pp.get('domain_coverage_ratio', '—')}x",
                    help="Ratio of general-English perplexity / slang perplexity. Higher = better domain fit.")

        _ndcg = _cm.get("ndcg", {})
        _ndcg_post = _ndcg.get("post_level", {})
        if _ndcg_post:
            _nd1, _nd2, _nd3 = st.columns(3)
            _nd1.metric("NDCG@50",  _ndcg_post.get("ndcg_at_50", "—"),
                        help="High-risk post ranking quality at top 50.")
            _nd2.metric("NDCG@100", _ndcg_post.get("ndcg_at_100", "—"),
                        help="High-risk post ranking quality at top 100.")
            _nd3.metric("NDCG@500", _ndcg_post.get("ndcg_at_500", "—"),
                        help="High-risk post ranking quality at top 500.")

        _top5 = _ndcg.get("top5_clusters_by_risk_similarity", [])
        if _top5:
            with st.expander("Top-5 clusters by similarity to high-risk centroid"):
                st.dataframe(pd.DataFrame(_top5), use_container_width=True)
    else:
        st.info(
            "No cluster metrics found. "
            "Run `python src/eval/cluster_metrics.py` to compute Silhouette, NDCG, and Perplexity."
        )

# ── Tab 6: Analyst Reports ────────────────────────────────────────────────────
with tab_reports:
    st.subheader("LLM Analyst Reports")
    st.markdown(
        "RAG-generated population-level summaries for detected high-risk "
        "spike events. Evidence spans are drawn from the most representative "
        "posts in each period. No individual post data is surfaced."
    )

    _summaries = _load_spike_summaries()
    if not _summaries:
        st.info(
            "No analyst reports found.  \n"
            "To generate them:  \n"
            "1. Set the `GOOGLE_API_KEY` environment variable  \n"
            "2. Run `python src/llm_classifier.py`  \n"
            "Reports will be saved to `data/processed/spike_summaries.json`."
        )
    else:
        st.caption(f"{len(_summaries)} report(s) available")
        for _i, _s in enumerate(_summaries):
            _spike_date = _s.get("spike_date", f"Report {_i + 1}")
            _post_count = _s.get("post_count", "?")
            _label = f"🗓 {_spike_date} — {_post_count} flagged posts"
            with st.expander(_label, expanded=(_i == 0)):
                if "llm_error" in _s:
                    st.warning(
                        f"LLM generation unavailable: {_s['llm_error']}  \n"
                        "Retrieval results are shown below."
                    )

                _summary_text = _s.get("analyst_summary", "")
                if _summary_text:
                    st.markdown(f"> {_summary_text}")
                    st.markdown("")

                _ca, _cb, _cc = st.columns(3)
                _ca.metric("Risk assessment", str(_s.get("risk_level", "—")).title())
                _conf = _s.get("confidence")
                _cb.metric(
                    "Confidence",
                    f"{_conf:.0%}" if isinstance(_conf, float) else str(_conf or "—"),
                )
                _gen = _s.get("generated_at", "")
                _cc.metric("Generated", _gen[:10] if _gen else "—")

                _spans = _s.get("evidence_spans", [])
                if _spans:
                    st.markdown("**Supporting evidence (population-level, anonymized):**")
                    for _sp in _spans:
                        st.markdown(f'- *"{_sp}"*')

                _rationale = _s.get("rationale", "")
                if _rationale:
                    with st.expander("Model reasoning"):
                        st.write(_rationale)

# ── Tab 7: Geographic ─────────────────────────────────────────────────────────
with tab_geo:
    st.subheader("Geographic Risk Distribution — CDC Overdose Deaths by State")
    st.markdown(
        "Provisional overdose death counts (CDC VSRR) aggregated by state. "
        "Use the filters below to explore substance-specific and year-specific "
        "patterns across the US."
    )

    _cdc_raw = _load_cdc_state()
    if len(_cdc_raw) == 0:
        st.info(
            "No CDC data found. Run `python scripts/fetch_cdc_data.py` "
            "to download the state-level overdose data."
        )
    else:
        # Filters
        _geo_col1, _geo_col2 = st.columns(2)
        _all_indicators = sorted(_cdc_raw["indicator"].dropna().unique().tolist())
        _default_ind = next(
            (i for i in _all_indicators if "opioid" in i.lower()), _all_indicators[0]
        )
        _sel_indicator = _geo_col1.selectbox(
            "Substance / Indicator", _all_indicators,
            index=_all_indicators.index(_default_ind),
        )
        _all_years = sorted(_cdc_raw["year"].dropna().unique().astype(int).tolist())
        _sel_year = _geo_col2.selectbox(
            "Year", _all_years, index=len(_all_years) - 1
        )

        # Aggregate
        _geo_filtered = _cdc_raw[
            (_cdc_raw["indicator"] == _sel_indicator)
            & (_cdc_raw["year"] == _sel_year)
        ].copy()
        _geo_agg = (
            _geo_filtered.groupby(["state", "state_name"], dropna=True)["data_value"]
            .mean()
            .reset_index()
            .rename(columns={"data_value": "avg_deaths"})
        )

        if len(_geo_agg) == 0:
            st.info("No data for the selected filters.")
        else:
            try:
                import plotly.express as px

                _choropleth = px.choropleth(
                    _geo_agg,
                    locations="state",
                    locationmode="USA-states",
                    color="avg_deaths",
                    hover_name="state_name",
                    hover_data={"avg_deaths": ":.1f", "state": False},
                    color_continuous_scale="Reds",
                    scope="usa",
                    title=f"{_sel_indicator} — Avg Monthly Deaths ({_sel_year})",
                    labels={"avg_deaths": "Avg monthly deaths"},
                )
                _choropleth.update_layout(height=520)
                st.plotly_chart(_choropleth, use_container_width=True)

                # Top-10 states table
                st.markdown("#### Top 10 States by Average Monthly Deaths")
                _top10 = _geo_agg.nlargest(10, "avg_deaths").reset_index(drop=True)
                _top10.index += 1
                _top10.columns = ["State Code", "State", "Avg Monthly Deaths"]
                _top10["Avg Monthly Deaths"] = _top10["Avg Monthly Deaths"].round(1)
                st.dataframe(_top10, use_container_width=True)
            except ImportError:
                st.error("plotly is required — pip install plotly")

# ── Tab 8: Recommendations ────────────────────────────────────────────────────
with tab_recs:
    st.subheader("💡 Intervention Recommendations")
    st.markdown(
        "Actionable public health interventions generated automatically "
        "based on early-warning risk signals and predictive correlations."
    )
    
    _rec_path = PROCESSED / "recommendations.json"
    if not _rec_path.exists():
        st.info("No recommendations found. Run `python src/agents/intervention_engine.py` "
                "to generate rule-based intervention protocols.")
    else:
        with open(_rec_path) as f:
            _recs = json.load(f)
            
        if not _recs:
            st.info("No actionable recommendations at this time.")
        else:
            _rec_df = pd.DataFrame(_recs)
            
            # Substance Filter
            _topics = ["All"] + sorted(_rec_df["topic"].unique().tolist())
            _sel_topic = st.selectbox("Filter by Topic/Substance", _topics)
            
            if _sel_topic != "All":
                _rec_df = _rec_df[_rec_df["topic"] == _sel_topic]
            
            # Display recommendations
            _color_map = {
                "IMMEDIATE": "🔴 IMMEDIATE",
                "MONITOR": "🟠 MONITOR",
                "INFORMATIONAL": "🔵 INFORMATIONAL"
            }
            
            if len(_rec_df) > 0:
                for idx, row in _rec_df.iterrows():
                    _sev_badge = _color_map.get(row.get("severity"), row.get("severity"))
                    with st.expander(f"{row.get('period', '')} | {_sev_badge} : {row.get('topic', '').replace('_', ' ').title()}"):
                        st.markdown(f"**Recommendation:** {row.get('recommendation', '')}")
                        st.markdown(f"*Rationale:* {row.get('rationale', '')}")
            else:
                st.info("No recommendations match the filter.")
