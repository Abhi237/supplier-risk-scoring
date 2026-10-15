import plotly.express as px
import streamlit as st
from core import (
    AMBER,
    CYAN,
    MEMO_URL,
    PROJECT_TITLE,
    RED,
    TIER_COLORS,
    TIER_ORDER,
    csv,
    scope_note,
    setup_page,
    strip,
    summary,
    tiles,
    usd,
    vendors,
)

s = summary()
q, t = s["sql_summary"], s["train_summary"]
sc = vendors()
ev = csv("oot_metrics_main.csv")
LATE = ev.target.iloc[0]
e = ev[ev.target == LATE].set_index("score")
C, EB, RAW, VOL, LANE = ("Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Baseline: raw late rate",
                         "Baseline: volume only (smaller = riskier)", "Baseline: lane mix only (country / mode)")
_d = t["auc_deltas"]["eb_late_vs_raw_late_late"]
scn, scn_raw = t["scenario_base_ASSUMED"], t["scenario_raw_baseline_ASSUMED"]
watch = sc[sc.is_scored & (sc.n_lines >= 50) & sc.risk_tier.isin(["High", "Elevated"])]
setup_page(f"Empirical-Bayes shrinkage to score on-time delivery risk across {q['ext_vendors']} USAID / PEPFAR "
           "health-commodity vendors",
           "Executive summary · every number computed by the pipeline · " + PROJECT_TITLE, section="executive summary")
strip([
    ("PROBLEM", f"A US government health-supply programme (USAID / PEPFAR SCMS) bought commodities from "
                f"{q['ext_vendors']} external vendors for PEPFAR countries. {q['ext_late_rate']:.1%} of "
                f"{q['ext_lines']:,} vendor order lines were delivered after the scheduled date, and "
                f"{q['ext_exact_match_share']:.0%} were recorded on exactly the scheduled date. Which vendors carry "
                "recorded delivery risk?"),
    ("METHOD", "DuckDB SQL cleaning and delivery intervals · beta-binomial empirical-Bayes shrinkage with Bayesian-"
               "bootstrap intervals · out-of-time validation (split by scheduled date) against raw-rate, volume and "
               "lane baselines · cold-start tier · k-means archetypes · R logistic GLM, vendor GLMM and negative binomial."),
    ("DECISION", f"Watch-list with the shrunk late rate (out-of-time line AUC {e.loc[EB, 'line_auc']:.3f}), not the "
                 f"composite ({e.loc[C, 'line_auc']:.3f}); the raw rate scores {e.loc[RAW, 'line_auc']:.3f}. "
                 f"{len(watch)} material vendors are High/Elevated. Back-tested reallocation by composite avoids "
                 f"{scn['late_value_reduction_pct']:.1%} of late value (ASSUMED parameters)."),
])
scope_note()

st.subheader("Programme status")
tiles([
    ("Vendor on-time rate", f"{q['ext_on_time_rate']:.1%}", f"{q['ext_lines']:,} external-vendor lines", "green"),
    ("SCMS regional DC on-time", f"{1 - q['rdc_late_rate']:.1%}", f"{q['rdc_lines']:,} lines, not a vendor", "amber"),
    ("Mean days late when late", f"{q['ext_mean_days_late_when_late']:.1f}",
     f"median {q['ext_median_days_late_when_late']:.0f}, p90 {q['ext_p90_days_late_when_late']:.0f} days", "amber"),
    ("Delivered date = scheduled", f"{q['ext_exact_match_share']:.0%}", "of vendor lines: lateness is rare to observe", "info"),
])
st.subheader("Model status")
tiles([
    ("OOT line AUC · EB late rate", f"{e.loc[EB, 'line_auc']:.3f}",
     f"{_d['delta']:+.3f} vs raw rate (95% CI {_d['lo95']:+.3f} to {_d['hi95']:+.3f})", "green"),
    ("OOT line AUC · composite", f"{e.loc[C, 'line_auc']:.3f}",
     f"{e.loc[C, 'line_auc'] - e.loc[RAW, 'line_auc']:+.3f} vs raw rate: baseline wins", "red"),
    ("Scored vs unscored vendors", f"{t['n_vendors_scored']} / {t['n_vendors_unscored']}",
     f"unscored hold {t['unscored_line_share']:.0%} of lines (< {s['config']['scoring']['min_lines_for_score']} lines)", "amber"),
    ("Value at risk (shrunk late rate)", usd(t["exposure_totals"]["value_at_risk"]),
     f"of {usd(t['exposure_totals']['value_usd'])} vendor line value", "red"),
])

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Risk concentration by tier")
    tier = csv("risk_tier_summary.csv").set_index("risk_tier").reindex(TIER_ORDER).dropna(how="all").reset_index()
    m = tier.melt(id_vars="risk_tier", value_vars=["value_share", "late_value_share"], var_name="metric")
    m["metric"] = m.metric.map({"value_share": "share of line value", "late_value_share": "share of late value"})
    fig = px.bar(m, y="risk_tier", x="value", color="metric", barmode="group", orientation="h", text_auto=".0%",
                 color_discrete_sequence=[CYAN, RED], labels={"risk_tier": "", "value": "", "metric": ""})
    fig.update_layout(title="VALUE vs LATE VALUE BY RISK TIER (scored vendors; percentile tiers)", height=360,
                      xaxis_tickformat=".0%", yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader(f"Alerts · {len(watch)} material vendors")
    st.caption("High / Elevated tier, >= 50 lines · ranked by composite risk score")
    rows = []
    for r in watch.sort_values("risk_score", ascending=False).head(7).itertuples():
        sev = "red" if r.risk_tier == "High" else "amber"
        rows.append(f"<div class='alert {sev}'><div class='sev' style='color:{RED if sev == 'red' else AMBER}'>"
                    f"{r.risk_tier.upper()}</div><div><div class='id'>{r.vendor[:34]}</div>"
                    f"<div class='meta'>{r.top_country} · {r.top_product_group} · {int(r.n_lines)} lines · late (EB) "
                    f"{r.late_rate_shrunk:.1%} · {r.mean_days_late_shrunk:.1f} mean days late (EB)</div></div>"
                    f"<div class='score'>{r.risk_score:.0f}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)

st.subheader("Recommended actions")
hi = t["high_tier"]
a = st.columns(4)
a[0].markdown(f"**01 · Use the shrunk late rate to triage.** Out-of-time it ranks next-period lateness at line AUC "
              f"{e.loc[EB, 'line_auc']:.3f} vs {e.loc[RAW, 'line_auc']:.3f} for the raw rate; the composite adds nothing "
              f"here ({e.loc[C, 'line_auc']:.3f}). Volume-only and lane-mix baselines rank the wrong way "
              f"({e.loc[VOL, 'line_auc']:.2f} and {e.loc[LANE, 'line_auc']:.2f}).")
a[1].markdown(f"**02 · Review the {hi['n']} High-tier vendors** - {hi['value_share']:.0%} of vendor line value and "
              f"{hi['late_value_share']:.0%} of late value, late rate {hi['late_rate']:.1%} vs "
              f"{t['overall_ext_late_rate']:.1%} overall.")
a[2].markdown(f"**03 · Do not score thin vendors.** {t['n_vendors_unscored']} vendors have fewer than "
              f"{s['config']['scoring']['min_lines_for_score']} lines ({t['unscored_line_share']:.0%} of lines, "
              f"{t['unscored_value_share']:.1%} of value) and sit in the New / unscored tier at the prior mean.")
a[3].markdown(f"**04 · Hold the reallocation case.** Moving volume away from the top composite group avoids "
              f"{scn['late_value_reduction_pct']:.1%} of next-period late value (ASSUMED); ranking by raw late rate "
              f"would avoid {scn_raw['late_value_reduction_pct']:.1%}. Vendors are often sole-source. "
              f"[Memo (PDF)]({MEMO_URL})")

st.subheader("Score map")
fig = px.scatter(sc[sc.is_scored], x="value_usd", y="risk_score", color="risk_tier", log_x=True,
                 color_discrete_map=TIER_COLORS, category_orders={"risk_tier": TIER_ORDER}, opacity=0.85,
                 size="n_lines", size_max=26,
                 hover_data={"vendor": True, "top_country": True, "n_lines": True, "late_rate_shrunk": ":.1%"},
                 labels={"value_usd": "vendor line value (USD, log)", "risk_score": "composite risk score",
                         "risk_tier": "tier"})
fig.update_layout(height=420, title="RISK SCORE vs LINE VALUE (scored vendors; bubble = lines)")
st.plotly_chart(fig, width="stretch")
