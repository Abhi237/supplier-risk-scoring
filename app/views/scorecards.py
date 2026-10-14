import plotly.graph_objects as go
import streamlit as st
from core import GREY, NAVY, RED, TIER_ORDER, brl, proxy_note, sellers, setup_page, summary

s = summary()
setup_page("Seller scorecards", "Search any seller (anonymised Olist hash) · empirical-Bayes shrunk rates with 90% "
           "credible intervals · composite risk score with 90% Bayesian-bootstrap interval", section="seller scorecards")
proxy_note()
sc = sellers()
with st.container(border=True):
    c = st.columns([2, 1, 1, 1])
    query = c[0].text_input("Seller ID contains", "", placeholder="e.g. first characters of the seller hash")
    tiers = c[1].multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
    states = c[2].multiselect("Seller state", sorted(sc.seller_state.dropna().unique()))
    min_items = c[3].number_input("Min items", 0, 2000, 20, 5)
f = sc[sc.risk_tier.isin(tiers) & (sc.n_items >= min_items)]
if query:
    f = f[f.seller_id.str.contains(query.strip().lower(), regex=False)]
if states:
    f = f[f.seller_state.isin(states)]
f = f.sort_values("risk_score", ascending=False)
st.caption(f"{len(f):,} sellers match · sorted by composite risk score (higher = riskier)")
st.dataframe(f[["risk_rank", "seller_id", "seller_state", "top_category", "n_items", "gmv", "late_rate_raw",
                "late_rate_shrunk", "seller_handoff_late_rate", "mean_review_score", "risk_score", "risk_score_lo90",
                "risk_score_hi90", "risk_tier", "archetype"]].head(500),
             hide_index=True, width="stretch", height=320,
             column_config={"gmv": st.column_config.NumberColumn("GMV (BRL)", format="%.0f"),
                            "late_rate_raw": st.column_config.NumberColumn("late (raw)", format="%.3f"),
                            "late_rate_shrunk": st.column_config.NumberColumn("late (EB)", format="%.3f"),
                            "seller_handoff_late_rate": st.column_config.NumberColumn("hand-off late (EB)", format="%.3f"),
                            "mean_review_score": st.column_config.NumberColumn("review", format="%.2f"),
                            "risk_score": st.column_config.ProgressColumn("risk score", format="%.1f", min_value=0,
                                                                          max_value=100),
                            "risk_score_lo90": st.column_config.NumberColumn("lo90", format="%.1f"),
                            "risk_score_hi90": st.column_config.NumberColumn("hi90", format="%.1f")})

opts = f.seller_id.head(500).tolist()
if not opts:
    st.info("No seller matches the filters.")
    st.stop()
sel = st.selectbox("Seller detail", opts, index=0)
r = sc.set_index("seller_id").loc[sel]
st.subheader(f"Scorecard · {sel[:12]}…  ({r.seller_state}, {r.top_category})")
c = st.columns(5)
c[0].metric("Risk score", f"{r.risk_score:.1f}", f"90% {r.risk_score_lo90:.0f}-{r.risk_score_hi90:.0f}",
            delta_color="off")
c[1].metric("Risk tier", r.risk_tier, f"rank {int(r.risk_rank):,} / {len(sc):,}", delta_color="off")
c[2].metric("Items / GMV", f"{int(r.n_items):,}", brl(r.gmv), delta_color="off")
c[3].metric("Late rate (EB)", f"{r.late_rate_shrunk:.1%}",
            f"raw {r.late_rate_raw:.1%}" if r.n_delivered else "no deliveries", delta_color="off")
c[4].metric("GMV at risk", brl(r.gmv_at_risk), help="GMV x EB-shrunk late rate")
if r.low_volume_flag:
    st.warning("Low-volume seller: the score is dominated by the prior - treat as *unknown*, not as *bad*.")

pri = s["train_summary"]["priors"]
rows = [("Late delivery", r.late_rate_raw, r.late_rate_shrunk, pri["late"][2], int(r.n_late), int(r.n_delivered)),
        ("Hand-off late", r.handoff_late_rate_raw, r.seller_handoff_late_rate, pri["handoff"][2],
         int(r.n_handoff_late), int(r.n_handoff_obs)),
        ("Cancellation", r.cancel_rate_raw, r.cancel_rate, pri["cancel"][2], None, int(r.n_items))]
fig = go.Figure()
for i, (nm, raw, shr, prior, _k, _n) in enumerate(rows):
    fig.add_scatter(x=[raw], y=[nm], mode="markers", marker=dict(color=GREY, size=12, symbol="circle-open"),
                    name="raw", showlegend=i == 0)
    fig.add_scatter(x=[shr], y=[nm], mode="markers", marker=dict(color=NAVY, size=13), name="EB-shrunk",
                    showlegend=i == 0)
    fig.add_scatter(x=[prior], y=[nm], mode="markers", marker=dict(color=RED, size=10, symbol="line-ns-open"),
                    name="population prior mean", showlegend=i == 0)
fig.add_scatter(x=[r.late_rate_lo90, r.late_rate_hi90], y=["Late delivery"] * 2, mode="lines",
                line=dict(color=NAVY, width=4), opacity=0.35, name="90% credible interval")
fig.update_layout(height=300, xaxis_tickformat=".0%", title="Raw vs shrunk rates (shrinkage toward the prior)",
                  xaxis_title="rate")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, width="stretch")
comp = {"Late (EB)": r.late_rate_shrunk, "Hand-off late (EB)": r.seller_handoff_late_rate,
        "Review risk": r.review_risk, "Cancellation (EB)": r.cancel_rate, "Freight / price": r.freight_ratio,
        "Volume uncertainty": r.volume_uncertainty}
pct = {k: (sc[col] <= v).mean() for (k, v), col in zip(comp.items(), ["late_rate_shrunk", "seller_handoff_late_rate",
                                                                     "review_risk", "cancel_rate", "freight_ratio",
                                                                     "volume_uncertainty"])}
fig = go.Figure(go.Scatterpolar(r=list(pct.values()) + [list(pct.values())[0]],
                                theta=list(pct.keys()) + [list(pct.keys())[0]], fill="toself", line_color=RED))
fig.update_layout(height=300, polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
                  title="Component percentiles (1 = riskiest)")
c2.plotly_chart(fig, width="stretch")
st.caption(f"Weights (config.yaml): {s['config']['scoring']['weights']}")
