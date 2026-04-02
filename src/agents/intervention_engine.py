import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

def get_top_states(cdc_df, indicator="opioid", top_n=3):
    if "indicator" not in cdc_df.columns or "year" not in cdc_df.columns:
        return ["Unknown"]
    
    sub_df = cdc_df[cdc_df["indicator"].astype(str).str.contains(indicator, case=False, na=False)]
    if len(sub_df) == 0:
        return ["Unknown"]
    latest_year = sub_df["year"].max()
    sub_df = sub_df[sub_df["year"] == latest_year]
    if "state_name" in sub_df.columns:
        state_col = "state_name"
    else:
        state_col = "state"
    agg = sub_df.groupby(state_col)["data_value"].mean().reset_index()
    top = agg.nlargest(top_n, "data_value")[state_col].tolist()
    return top

def main():
    warn_path = PROCESSED / "narrative" / "warning_report.csv"
    if not warn_path.exists():
        print(f"File not found: {warn_path}")
        return
    warnings = pd.read_csv(warn_path)
    
    corrs = {}
    corrs_path = PROCESSED / "correlations.json"
    if corrs_path.exists():
        with open(corrs_path) as f:
            corrs = json.load(f)
            
    cdc_path = DATA / "raw" / "cdc_overdose_data.csv"
    if PROCESSED.joinpath("cdc_cleaned.csv").exists():
        cdc_path = PROCESSED / "cdc_cleaned.csv"
        
    if cdc_path.exists():
        cdc_df = pd.read_csv(cdc_path)
    else:
        cdc_df = pd.DataFrame()

    opioid_corrs = corrs.get("opioid", {})
    best_lag = -3
    best_p = 1.0
    for lag_str, vals in opioid_corrs.items():
        lag = int(lag_str)
        if lag < 0 and vals["p"] < 0.05 and vals["p"] < best_p:
            best_lag = lag
            best_p = vals["p"]
    lead_months = abs(best_lag)

    recommendations = []
    top_opioid_states = get_top_states(cdc_df, indicator="opioid", top_n=3)
    states_str = ", ".join(top_opioid_states)

    for idx, row in warnings.iterrows():
        period = row["period"]
        topic = row["topic"]
        level = row["alert_level"]
        pct_inc = row.get("pct_increase", 0)
        
        if level == "critical":
            recommendations.append({
                "period": period,
                "topic": topic,
                "severity": "IMMEDIATE",
                "recommendation": f"Deploy naloxone in top-3 states by death rate ({states_str})",
                "rationale": f"Critical {topic} alert detected."
            })
            recommendations.append({
                "period": period,
                "topic": topic,
                "severity": "IMMEDIATE",
                "recommendation": f"Activate early warning protocol {lead_months} months before projected spike",
                "rationale": f"Social signal leads CDC deaths by {lead_months} months."
            })
        elif topic == "harm_reduction" and level in ["elevated", "watch"]:
            recommendations.append({
                "period": period,
                "topic": topic,
                "severity": "INFORMATIONAL" if level == "watch" else "MONITOR",
                "recommendation": "Increase harm reduction outreach messaging",
                "rationale": f"Rising harm_reduction topic ({pct_inc}% increase)."
            })
        elif level == "elevated":
            recommendations.append({
                "period": period,
                "topic": topic,
                "severity": "MONITOR",
                "recommendation": "Monitor local community forums and alert local health officials",
                "rationale": f"Elevated {topic} signals ({pct_inc}% increase)."
            })
        elif level == "watch":
            recommendations.append({
                "period": period,
                "topic": topic,
                "severity": "INFORMATIONAL",
                "recommendation": "Track topic trajectory in dashboard",
                "rationale": f"Watch {topic} signals ({pct_inc}% increase)."
            })
             
    out_path = PROCESSED / "recommendations.json"
    with open(out_path, "w") as f:
        json.dump(recommendations, f, indent=4)
        
    print(f"Generated {len(recommendations)} recommendations to {out_path}")

if __name__ == "__main__":
    main()
