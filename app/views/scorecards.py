import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import GREY, NAVY, RED, TIER_ORDER, callout, csv, scope_note, setup_page, summary, usd, vendors

s = summary()
setup_page("Vendor scorecards", "Search any vendor (names are public in the dataset) · empirical-Bayes shrunk rates with 90% credible "
           "intervals · composite risk score with 90% Bayesian-bootstrap interval", section="vendor scorecards")
scope_note()
sc = vendors()
st.warning("Scores describe **recorded delivery performance in this programme**, not vendor quality overall. Most vendors have few lines; "
           "see the line counts and the New / unscored tier before reading anything into a score.")
with st.container(border=True):
    c = st.columns([2, 1, 1, 1])
    query = c[0].text_input("Vendor name contains", "", placeholder="e.g. cipla")
    tiers = c[1].multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
    countries = c[2].multiselect("Top country", sorted(sc.top_country.dropna().unique()))
    min_lines = c[3].number_input("Min lines", 0, 2000, 0, 5)
f = sc[sc.risk_tier.isin(tiers) & (sc.n_lines >= min_lines)]
if query:
    f = f[f.vendor.str.lower().str.contains(query.strip().lower(), regex=False)]
if countries:
    f = f[f.top_country.isin(countries)]
f = f.sort_values(["risk_score", "n_lines"], ascending=[False, False], na_position="last")
st.caption(f"{len(f):,} vendors match · scored vendors first, sorted by composite risk score (higher = riskier)")
_pri = s["train_summary"]["priors"]
_SPL = f"{s['train_summary']['oot_main']['score_end'][:10]}->{s['train_summary']['oot_main']['validate_end'][:10]}"
_om = s["train_summary"]["oot_main"]
_cs = csv("cold_start.csv").query("split == @_SPL").iloc[0]
_min = s["config"]["scoring"]["min_lines_for_score"]
callout(f"<b>Cold-start policy: New / unscored tier.</b> A vendor with fewer than {_min} lines (or none before the score date) gets the "
        f"empirical-Bayes prior mean for every rate (late {_pri['late'][2]:.1%} on the full period), no composite score and the tier "
        f"<b>New / unscored</b>. Out-of-time ({_cs.split}) such vendors were {int(_cs.n_new_vendors):,} vendors and "
        f"<b>{_cs.new_share_next_lines:.1%} of next-period lines</b>; their late rate was {_cs.new_late:.2%} "
        f"(95% Jeffreys {_cs.new_late_lo95:.2%}-{_cs.new_late_hi95:.2%}) vs a prior of {_cs.prior_mean_late:.2%}. "
        f"{int((~sc.is_scored).sum())} of {len(sc)} vendors are unscored in the full-period table.")
st.dataframe(f[["risk_rank", "vendor", "top_country", "top_product_group", "n_lines", "value_usd", "late_rate_raw", "late_rate_shrunk",
                "mean_days_late_shrunk", "risk_score", "risk_score_lo90", "risk_score_hi90", "risk_tier", "archetype"]].head(200),
             hide_index=True, width="stretch", height=320,
             column_config={"value_usd": st.column_config.NumberColumn("line value (USD)", format="%.0f"),
                            "late_rate_raw": st.column_config.NumberColumn("late (raw)", format="%.3f"),
                            "late_rate_shrunk": st.column_config.NumberColumn("late (EB)", format="%.3f"),
                            "mean_days_late_shrunk": st.column_config.NumberColumn("mean days late (EB)", format="%.2f"),
                            "risk_score": st.column_config.ProgressColumn("risk score", format="%.1f", min_value=0, max_value=100),
                            "risk_score_lo90": st.column_config.NumberColumn("lo90", format="%.1f"),
                            "risk_score_hi90": st.column_config.NumberColumn("hi90", format="%.1f")})

opts = f.vendor.head(200).tolist()
if not opts:
    st.info("No vendor matches the filters.")
    st.stop()
sel = st.selectbox("Vendor detail", opts, index=0)
r = sc.set_index("vendor").loc[sel]
st.subheader(f"Scorecard · {sel[:46]}  ({r.top_country}, {r.top_product_group})")
c = st.columns(5)
if r.is_scored:
    c[0].metric("Risk score", f"{r.risk_score:.1f}", f"90% {r.risk_score_lo90:.0f}-{r.risk_score_hi90:.0f}", delta_color="off")
    c[1].metric("Risk tier", r.risk_tier, f"rank {int(r.risk_rank)} / {int(sc.is_scored.sum())}", delta_color="off")
else:
    c[0].metric("Risk score", "not scored", f"fewer than {_min} lines", delta_color="off")
    c[1].metric("Risk tier", r.risk_tier, "prior-mean defaults", delta_color="off")
c[2].metric("Lines / value", f"{int(r.n_lines):,}", usd(r.value_usd), delta_color="off")
c[3].metric("Late rate (EB)", f"{r.late_rate_shrunk:.1%}", f"raw {r.late_rate_raw:.1%} ({int(r.n_late)} of {int(r.n_lines)})", delta_color="off")
c[4].metric("Value at risk", usd(r.value_at_risk), help="line value x EB-shrunk late rate")
if not r.is_scored:
    st.warning("Thin vendor: the shrunk rates are dominated by the prior - treat as unknown, not as good or bad.")

pri = s["train_summary"]["priors"]
rows = [("Late delivery", r.late_rate_raw, r.late_rate_shrunk, pri["late"][2]),
        ("Severe late (>14 d)", r.severe_late_rate_raw, r.severe_late_rate_shrunk, pri["severe"][2])]
fig = go.Figure()
for i, (nm, raw, shr, prior) in enumerate(rows):
    fig.add_scatter(x=[raw], y=[nm], mode="markers", marker=dict(color=GREY, size=12, symbol="circle-open"), name="raw",
                    showlegend=i == 0)
    fig.add_scatter(x=[shr], y=[nm], mode="markers", marker=dict(color=NAVY, size=13), name="EB-shrunk", showlegend=i == 0)
    fig.add_scatter(x=[prior], y=[nm], mode="markers", marker=dict(color=RED, size=10, symbol="line-ns-open"),
                    name="programme prior mean", showlegend=i == 0)
fig.add_scatter(x=[r.late_rate_lo90, r.late_rate_hi90], y=["Late delivery"] * 2, mode="lines", line=dict(color=NAVY, width=4),
                opacity=0.35, name="90% credible interval")
fig.update_layout(height=300, xaxis_tickformat=".1%", title="Raw vs shrunk rates (shrinkage toward the prior)", xaxis_title="rate")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, width="stretch")
if r.is_scored:
    comp = {"Late (EB)": "late_rate_shrunk", "Severe late (EB)": "severe_late_rate_shrunk", "Mean days late (EB)": "mean_days_late_shrunk",
            "Freight ratio": "freight_ratio", "Lane mix (context)": "lane_mix_risk", "Volume uncertainty": "volume_uncertainty"}
    ref = sc[sc.is_scored]
    pct = {k: float((ref[col] <= r[col]).mean()) for k, col in comp.items()}
    fig = go.Figure(go.Scatterpolar(r=list(pct.values()) + [list(pct.values())[0]], theta=list(pct.keys()) + [list(pct.keys())[0]],
                                    fill="toself", line_color=RED))
    fig.update_layout(height=300, polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
                      title="Component percentiles among scored vendors (1 = riskiest)")
    c2.plotly_chart(fig, width="stretch")
else:
    c2.info("No component percentiles for an unscored vendor.")
yp = csv("vendor_year_panel.csv")
yp = yp[yp.vendor == sel]
if len(yp):
    fig = go.Figure()
    fig.add_bar(x=yp.sched_year, y=yp.n_lines, name="lines (rhs)", marker_color="#24324f", yaxis="y2")
    fig.add_scatter(x=yp.sched_year, y=yp.late_rate, name="late rate", line=dict(color=RED, width=3))
    fig.add_scatter(x=yp.sched_year, y=yp.late_rate_roll3, name="rolling 3-year late rate", line=dict(color=NAVY, width=3, dash="dot"))
    fig.update_layout(height=300, title="Vendor by scheduled year", yaxis=dict(tickformat=".0%"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")
w = s["config"]["scoring"]["weights"]
st.caption(f"Weights (config.yaml): {w}. Lane mix is context (how hard the vendor's country / mode mix is), not vendor-controlled. "
           f"Other cells: {pd.Series({'air share': r.air_share, 'country HHI': r.country_hhi}).round(2).to_dict()}.")
