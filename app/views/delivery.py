import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import GREEN, NAVY, ORANGE, RED, csv, proxy_note, setup_page, summary

s = summary()
q, r = s["sql_summary"], s.get("r_summary", {})
setup_page("Delivery performance", "Order-to-delivery intervals: approval → seller hand-off (vs shipping limit) → "
           "carrier transit → customer delivery (vs estimate)")
proxy_note()
c = st.columns(4)
c[0].metric("Median purchase → delivery", f"{q['order_median_delivery_days']:.1f} days")
c[1].metric("Mean delay when late", f"{q['order_mean_delay_days_when_late']:.1f} days")
c[2].metric("Review: late vs on time", f"{q['review_late']:.2f} vs {q['review_on_time']:.2f}", "of 5",
            delta_color="off")
c[3].metric("Freight / price", f"{q['freight_to_price']:.1%}", help="Total freight over total item price")

pm = csv("platform_month.csv")
pm["purchase_month"] = pd.to_datetime(pm.purchase_month)
pm = pm[(pm.purchase_month >= "2017-01-01") & (pm.purchase_month < "2018-09-01")]
fig = go.Figure()
fig.add_bar(x=pm.purchase_month, y=pm.n_items, name="items", marker_color="#d6e2ef", yaxis="y2")
fig.add_scatter(x=pm.purchase_month, y=pm.late_rate, name="late vs estimate", line=dict(color=RED, width=3))
fig.add_scatter(x=pm.purchase_month, y=pm.handoff_late_rate, name="seller hand-off late", line=dict(color=NAVY, width=3))
fig.update_layout(title="Monthly lateness by purchase month", height=400, yaxis=dict(tickformat=".0%", title="rate"),
                  yaxis2=dict(overlaying="y", side="right", showgrid=False, title="items"))
st.plotly_chart(fig, width="stretch")
st.caption("End-to-end lateness spikes in Nov-2017 (Black Friday) and Feb-Mar 2018 across the whole platform - a "
           "calendar/carrier shock no seller-level score can anticipate.")

c1, c2 = st.columns(2)
with c1:
    hb = csv("handoff_buckets.csv")
    order = ["<-3d", "-3..-1d", "-1..0d", "0..1d", "1..3d", "3..7d", ">7d"]
    hb = hb[hb.handoff_bucket.isin(order)].set_index("handoff_bucket").reindex(order).reset_index()
    fig = px.bar(hb, x="handoff_bucket", y="late_rate", text_auto=".0%",
                 color=hb.handoff_bucket.isin(order[3:]).map({True: "after limit", False: "before limit"}),
                 color_discrete_map={"after limit": RED, "before limit": GREEN},
                 labels={"handoff_bucket": "hand-off vs shipping limit", "late_rate": "late-delivery rate", "color": ""})
    fig.update_layout(title="Seller hand-off timing drives customer lateness", height=380, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
with c2:
    lh = csv("leadtime_hist.csv")
    lh["lane"] = lh.same_state.map({True: "same-state lane", False: "cross-state lane"})
    lh["share"] = lh.n / lh.groupby("lane").n.transform("sum")
    lh = lh.sort_values("days_bin")
    lh["ecdf"] = lh.groupby("lane").share.cumsum()
    fig = px.line(lh, x="days_bin", y="ecdf", color="lane", color_discrete_sequence=[NAVY, ORANGE],
                  labels={"days_bin": "purchase → delivery (days)", "ecdf": "cumulative share"})
    fig.update_layout(title="Lead-time ECDF by lane type", height=380)
    st.plotly_chart(fig, width="stretch")

st.subheader("By customer state and category")
c1, c2 = st.columns(2)
with c1:
    cs = csv("customer_state.csv").sort_values("late_rate", ascending=False)
    fig = px.bar(cs, x="customer_state", y="late_rate", color="median_days", color_continuous_scale="Reds",
                 labels={"customer_state": "customer state", "late_rate": "late rate", "median_days": "median days"})
    fig.update_layout(height=360, yaxis_tickformat=".0%", title="Late rate by destination state")
    st.plotly_chart(fig, width="stretch")
with c2:
    cat = csv("category.csv").sort_values("late_rate", ascending=False).head(20)
    fig = px.bar(cat, y="category", x="late_rate", orientation="h", color="freight_to_price",
                 color_continuous_scale="Blues", labels={"late_rate": "late rate", "category": ""})
    fig.update_layout(height=360, xaxis_tickformat=".0%", title="Top-20 categories by late rate (n >= 300)")
    st.plotly_chart(fig, width="stretch")

if r:
    st.subheader("R statistical layer (GLM / GLMM / negative binomial)")
    orr = csv("r_glm_odds_ratios.csv")
    orr = orr[~orr.term.str.match(r"^(month|year|\(Intercept\))")]
    fig = go.Figure(go.Scatter(x=orr.estimate, y=orr.term, mode="markers", marker=dict(color=NAVY, size=9),
                               error_x=dict(type="data", symmetric=False, array=orr["conf.high"] - orr.estimate,
                                            arrayminus=orr.estimate - orr["conf.low"], color="#9aa8b8")))
    fig.add_vline(x=1, line_dash="dash", line_color=RED)
    fig.update_layout(xaxis_type="log", height=560, title="Logistic GLM odds ratios for late delivery (95% CI)",
                      xaxis_title="odds ratio (log)")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Hand-off late OR **{r['or_handoff']:.2f}** (95% CI {r['or_handoff_lo']:.2f}-{r['or_handoff_hi']:.2f}); "
                f"same-state OR {r['or_same_state']:.2f}. In a seller random-intercept GLMM the seller share of latent "
                f"variance (ICC) is **{r['glmm_icc_without_handoff']:.1%}**, falling to {r['glmm_icc']:.1%} once "
                f"hand-off lateness is controlled - most end-to-end lateness is not a stable seller trait. "
                f"Full report: `reports/delivery_report.html`.")
