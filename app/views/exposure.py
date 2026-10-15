import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import NAVY, ORANGE, RED, SEQ, assumed_banner, csv, oot_vendors, scope_note, setup_page, summary, usd, vendors

s = summary()
ex, t = s["config"]["exposure"], s["train_summary"]
setup_page("Exposure & scenario simulator", "Value at risk by vendor, assumed-cost exposure, and a back-tested volume reallocation "
           "scenario", section="exposure & scenarios")
scope_note()
assumed_banner("The dataset has no cost-of-delay data. The cost of a late line, the annual carrying rate, the share of volume moved and how "
               "much of the reliability gap the receiving vendors close are <b>assumptions</b>. Line value, late value and days late are observed.")
st.write("")
sc = vendors()

st.subheader("1 - Exposure by vendor")
with st.container(border=True):
    c = st.columns(2)
    lc = c[0].slider("ASSUMED cost of a late line (% of line value)", 0.0, 0.25, float(ex["late_cost_pct_of_value"]), 0.01, format="%.2f")
    cr = c[1].slider("ASSUMED annual carrying rate on value x days late", 0.0, 0.40, float(ex["carrying_rate_annual"]), 0.01, format="%.2f")
e = sc.assign(late_cost=sc.value_at_risk * lc, carry_cost=sc.value_usd * sc.mean_days_late_shrunk / 365.0 * cr)
e["total"] = e.late_cost + e.carry_cost
sco = e[e.is_scored]
top = sco.nlargest(max(1, int(round(0.1 * len(sco)))), "risk_score")
k = st.columns(5)
k[0].metric("Line value, external vendors", usd(float(e.value_usd.sum())), "observed")
k[1].metric("Value at risk (observed basis)", usd(float(e.value_at_risk.sum())), help="line value x EB-shrunk late rate")
k[2].metric("Capital-years tied up by delays", usd(float(e.wc_excess_delay.sum())).replace("$", "") + " USD-yr",
            help="late value x mean days late / 365 (observed)")
k[3].metric("Total exposure (ASSUMED costs)", usd(float(e.total.sum())), "scenario", delta_color="off")
k[4].metric("Top-decile-risk share of exposure", f"{top.total.sum() / sco.total.sum():.0%}", f"{top.value_usd.sum() / sco.value_usd.sum():.0%} of value",
            delta_color="off")
c1, c2 = st.columns([3, 2])
with c1:
    ee = e.nlargest(15, "total").iloc[::-1]
    fig = go.Figure()
    for col, nm, colr in [("late_cost", "late cost (ASSUMED)", RED), ("carry_cost", "carrying cost (ASSUMED)", ORANGE)]:
        fig.add_bar(y=ee.vendor.str[:30] + " (" + ee.risk_tier + ")", x=ee[col], name=nm, orientation="h", marker_color=colr)
    fig.update_layout(barmode="stack", height=480, title="Top-15 vendors by total exposure", xaxis_title="USD (ASSUMED costs)")
    st.plotly_chart(fig, width="stretch")
with c2:
    v = np.sort(e.value_at_risk.to_numpy())[::-1]
    g = np.sort(e.value_usd.to_numpy())[::-1]
    x = np.arange(1, len(v) + 1) / len(v)
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.cumsum(v) / v.sum(), name="value at risk", line=dict(color=RED))
    fig.add_scatter(x=x, y=np.cumsum(g) / g.sum(), name="line value", line=dict(color=NAVY))
    fig.update_layout(height=480, title="Concentration (Lorenz)", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      xaxis_title="share of vendors (largest first)", yaxis_title="cumulative share")
    st.plotly_chart(fig, width="stretch")
st.caption("Exposure is driven mainly by value: a handful of large vendors hold most line value and value-at-risk. Risk-tier membership and exposure "
           "are different questions - prioritise vendors that are high on both.")
sens = csv("exposure_sensitivity.csv")
st.dataframe(sens.rename(columns={"late_cost_pct_of_value": "late cost % (ASSUMED)", "carrying_rate_annual": "carrying rate (ASSUMED)"}).round(3),
             hide_index=True, width="stretch")

st.subheader("2 - Volume reallocation back-test")
st.markdown("Rank vendors on data **before** the score date, move part of the top group's **next-period** volume to the other vendors, and assume "
            "the moved volume closes part of the observed late-rate gap.")
o = oot_vendors()
with st.container(border=True):
    c = st.columns(4)
    target = c[0].selectbox("Outcome", ["Late value", "Late lines"])
    rank_by = c[1].selectbox("Rank vendors by", ["risk_score", "late_rate_shrunk", "late_rate_raw", "severe_late_rate_shrunk", "lane_mix_risk"])
    share = c[2].slider("Top risk group (share of vendors)", 0.05, 0.40, float(ex["top_risk_share_to_reallocate"]), 0.01)
    frac = c[3].slider("ASSUMED share of their volume moved", 0.0, 1.0, 0.5, 0.05)
    gap = st.slider("ASSUMED share of the late-rate gap realised on moved volume", 0.0, 1.0, float(ex["reallocation_late_rate_improvement"]), 0.05)
if target == "Late value":
    vol, bad = o.next_value, o.next_late_value
else:
    vol, bad = o.next_n_lines.astype(float), o.next_n_late.astype(float)
d = o.assign(vol=vol, bad=bad, s=o[rank_by].fillna(o[rank_by].median()))
d = d[d.vol > 0].sort_values(["s", "vendor"], ascending=[False, True], kind="mergesort")
n_top = max(1, int(round(share * len(d))))
topd, rest = d.head(n_top), d.iloc[n_top:]
tr, rr = topd.bad.sum() / topd.vol.sum(), rest.bad.sum() / rest.vol.sum()
moved = topd.vol.sum() * frac
avoided = moved * max(tr - rr, 0) * gap
unit = usd if target == "Late value" else (lambda x: f"{x:,.0f} lines")
k = st.columns(5)
k[0].metric("Top-group vendors", f"{n_top:,}", f"{topd.vol.sum() / d.vol.sum():.0%} of volume", delta_color="off")
k[1].metric("Top-group next-period rate", f"{tr:.1%}", f"rest {rr:.1%}", delta_color="off")
k[2].metric("Volume moved (ASSUMED)", unit(moved))
k[3].metric("Late avoided (scenario)", unit(avoided), f"{avoided / d.bad.sum():.1%} of total", delta_color="off")
k[4].metric("Late after", unit(d.bad.sum() - avoided), f"before {unit(d.bad.sum())}", delta_color="off")
base = t["scenario_base_ASSUMED"]
raw = t["scenario_raw_baseline_ASSUMED"]
grid = csv("scenario_grid.csv")
gg = grid[grid.improvement_ASSUMED == 0.5]
piv = gg.pivot(index="top_share", columns="realloc_frac", values="late_value_reduction_pct")
piv.index = [f"top {x:.0%}" for x in piv.index]
piv.columns = [f"{x:.0%} moved" for x in piv.columns]
fig = px.imshow(piv, text_auto=".2%", color_continuous_scale=SEQ, aspect="auto",
                labels=dict(x="share moved (ASSUMED)", y="top risk share", color="reduction"))
fig.update_layout(height=300, title="Precomputed grid: late-value reduction (composite ranking, gap realised = 50% ASSUMED)")
st.plotly_chart(fig, width="stretch")
st.caption(f"Main split, ASSUMED parameters (top {base['top_share']:.0%} of vendors, {base['realloc_frac']:.0%} moved, {base['improvement_ASSUMED']:.0%} of gap): ranking by the "
           f"composite avoids {base['late_value_reduction_pct']:.1%} of next-period late value because its top group did not have a higher late "
           f"value rate than the rest ({base['top_late_rate_value']:.1%} vs {base['rest_late_rate_value']:.1%}). Ranking by the raw late rate would avoid "
           f"{raw['late_value_reduction_pct']:.1%} (top group {raw['top_late_rate_value']:.1%} vs {raw['rest_late_rate_value']:.1%}), because it flags the large chronic "
           "late vendors. This is a scenario, not a forecast; moving volume also assumes alternative vendors can supply the same product and country.")
