import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import (
    NAVY,
    ORANGE,
    RED,
    SEQ,
    assumed_banner,
    brl,
    csv,
    oot_sellers,
    proxy_note,
    sellers,
    setup_page,
    summary,
)

s = summary()
ex, t = s["config"]["exposure"], s["train_summary"]
setup_page("Exposure & scenario simulator", "Working-capital and cost exposure by seller, and a back-tested volume "
           "reallocation scenario", section="exposure & scenarios")
proxy_note()
assumed_banner("Olist has no procurement cost data. Late-delivery cost, freight absorption, cancellation cost, the "
               "share of volume moved and how much of the reliability gap the receiving sellers actually close are "
               "<b>assumptions</b>. GMV, late GMV, freight and delivery days are observed.")
st.write("")
sc = sellers()

st.subheader("1 · Exposure by seller")
with st.container(border=True):
    c = st.columns(3)
    lc = c[0].slider("ASSUMED cost of a late delivery (% of item GMV)", 0.0, 0.25, float(ex["late_cost_pct_of_gmv"]),
                     0.01, format="%.2f")
    fa = c[1].slider("ASSUMED share of freight absorbed", 0.0, 1.0, float(ex["freight_absorb_pct"]), 0.05, format="%.2f")
    cc = c[2].slider("ASSUMED cost of a cancellation (% of GMV)", 0.0, 0.5, float(ex["cancel_cost_pct_of_gmv"]), 0.01,
                     format="%.2f")
e = sc.assign(late_cost=sc.gmv_at_risk * lc, freight_cost=sc.freight * fa, cancel_cost=sc.gmv * sc.cancel_rate * cc)
e["total"] = e.late_cost + e.freight_cost + e.cancel_cost
top = e.nlargest(int(round(0.1 * len(e))), "risk_score")
k = st.columns(5)
k[0].metric("GMV at risk (observed basis)", brl(float(e.gmv_at_risk.sum())), help="Σ GMV × EB-shrunk late rate")
k[1].metric("Working capital in transit", brl(float(e.wc_in_transit.sum())),
            help="Σ GMV × median delivery days / 365 - average GMV float tied up between sale and delivery")
k[2].metric("Excess WC from delays", brl(float(e.wc_excess_delay.sum())),
            help="Σ late GMV × mean days late / 365")
k[3].metric("Total exposure (ASSUMED costs)", brl(float(e.total.sum())), "scenario", delta_color="off")
k[4].metric("Top-decile-risk share of exposure", f"{top.total.sum() / e.total.sum():.0%}",
            f"{top.gmv.sum() / e.gmv.sum():.0%} of GMV", delta_color="off")
c1, c2 = st.columns([3, 2])
with c1:
    ee = e.nlargest(15, "total").iloc[::-1]
    fig = go.Figure()
    for col, nm, colr in [("late_cost", "late (ASSUMED)", RED), ("freight_cost", "freight (ASSUMED)", ORANGE),
                          ("cancel_cost", "cancellation (ASSUMED)", NAVY)]:
        fig.add_bar(y=ee.seller_id.str[:10] + "… (" + ee.risk_tier + ")", x=ee[col], name=nm, orientation="h",
                    marker_color=colr)
    fig.update_layout(barmode="stack", height=480, title="Top-15 sellers by total exposure", xaxis_title="BRL")
    st.plotly_chart(fig, width="stretch")
with c2:
    v = np.sort(e.total.to_numpy())[::-1]
    g = np.sort(e.gmv.to_numpy())[::-1]
    x = np.arange(1, len(v) + 1) / len(v)
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.cumsum(v) / v.sum(), name="exposure", line=dict(color=RED))
    fig.add_scatter(x=x, y=np.cumsum(g) / g.sum(), name="GMV", line=dict(color=NAVY))
    fig.update_layout(height=480, title="Concentration (Lorenz)", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      xaxis_title="share of sellers", yaxis_title="cumulative share")
    st.plotly_chart(fig, width="stretch")
st.caption("Exposure is driven mainly by GMV: large reliable sellers carry most absolute freight and GMV-at-risk. "
           "Risk-tier membership and exposure are different questions - prioritise sellers that are high on both.")

st.subheader("2 · Volume reallocation back-test")
st.markdown("Rank sellers on data **before** the score date, move part of the top group's **next-period** volume to "
            "the other sellers, and assume moved volume closes part of the observed late-rate gap.")
o = oot_sellers()
with st.container(border=True):
    c = st.columns(4)
    target = c[0].selectbox("Outcome", ["Late GMV (end-to-end)", "Late hand-offs (seller-controlled)"])
    rank_by = c[1].selectbox("Rank sellers by", ["risk_score", "seller_handoff_late_rate", "late_rate_shrunk",
                                                 "late_rate_raw"])
    share = c[2].slider("Top risk group (share of sellers)", 0.02, 0.30, float(ex["top_risk_share_to_reallocate"]), 0.01)
    frac = c[3].slider("ASSUMED share of their volume moved", 0.0, 1.0, 0.5, 0.05)
    gap = st.slider("ASSUMED share of the late-rate gap realised on moved volume", 0.0, 1.0,
                    float(ex["reallocation_late_rate_improvement"]), 0.05)
if target.startswith("Late GMV"):
    vol, bad = o.next_gmv, o.next_late_gmv
else:
    vol, bad = o.next_n_handoff_obs.astype(float), o.next_n_handoff_late.astype(float)
d = o.assign(vol=vol, bad=bad, s=o[rank_by].fillna(o[rank_by].median()))
d = d[d.vol > 0].sort_values("s", ascending=False, kind="mergesort")
n_top = max(1, int(round(share * len(d))))
topd, rest = d.head(n_top), d.iloc[n_top:]
tr, rr = topd.bad.sum() / topd.vol.sum(), rest.bad.sum() / rest.vol.sum()
moved = topd.vol.sum() * frac
avoided = moved * max(tr - rr, 0) * gap
unit = brl if target.startswith("Late GMV") else (lambda x: f"{x:,.0f} items")
k = st.columns(5)
k[0].metric("Top-group sellers", f"{n_top:,}", f"{topd.vol.sum() / d.vol.sum():.0%} of volume", delta_color="off")
k[1].metric("Top-group next-period rate", f"{tr:.1%}", f"rest {rr:.1%}", delta_color="off")
k[2].metric("Volume moved (ASSUMED)", unit(moved))
k[3].metric("Late avoided (scenario)", unit(avoided), f"{avoided / d.bad.sum():.1%} of total", delta_color="off")
k[4].metric("Late after", unit(d.bad.sum() - avoided), f"before {unit(d.bad.sum())}", delta_color="off")
grid = csv("scenario_grid.csv")
gg = grid[grid.improvement_ASSUMED == 0.5]
piv = gg.pivot(index="top_share", columns="realloc_frac", values="late_gmv_reduction_pct")
piv.index = [f"top {x:.0%}" for x in piv.index]
piv.columns = [f"{x:.0%} moved" for x in piv.columns]
fig = px.imshow(piv,
                text_auto=".2%", color_continuous_scale=SEQ, aspect="auto",
                labels=dict(x="share moved (ASSUMED)", y="top risk share", color="reduction"))
fig.update_layout(height=300, title="Precomputed grid: late-GMV reduction (composite ranking, gap realised = 50%)")
st.plotly_chart(fig, width="stretch")
st.caption("Back-tested on the main split. End-to-end late GMV barely moves because next-period lateness is weakly "
           "persistent at seller level; seller-controlled hand-off lateness responds far more.")
