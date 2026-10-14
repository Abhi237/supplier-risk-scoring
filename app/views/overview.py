import plotly.express as px
import streamlit as st
from core import NAVY, RED, TIER_COLORS, TIER_ORDER, brl, callout, csv, proxy_note, sellers, setup_page, summary

s = summary()
q, t = s["sql_summary"], s["train_summary"]
setup_page("Which vendors create the greatest delivery-risk and working-capital exposure?",
           "Executive summary · Olist Brazilian E-Commerce · sellers scored with empirical-Bayes shrinkage and "
           "validated out-of-time · every number computed by the pipeline")
callout("<b>Business question.</b> Procurement wants to know which suppliers put on-time delivery and working "
        "capital at risk, how confident we can be about small suppliers with little history, and whether a "
        "reliability score built today predicts <i>next period's</i> problems better than simple rules.")
proxy_note()

ev = csv("oot_metrics_main.csv").set_index(["target", "score"])
ho = ev.loc["Seller hand-off late"]
c = st.columns(4)
c[0].metric("Order late-delivery rate", f"{q['order_late_rate']:.1%}",
            help="Delivered orders arriving after the estimated date shown at purchase")
c[1].metric("Seller hand-off late", f"{q['item_handoff_late_rate']:.1%}",
            help="Items handed to the carrier after the seller's shipping-limit date (seller-controlled leg)")
c[2].metric("Late rate if hand-off late", f"{q['late_rate_if_handoff_late']:.0%}",
            f"vs {q['late_rate_if_handoff_on_time']:.1%} if on time", delta_color="off")
c[3].metric("Top-10% sellers' GMV share", f"{q['top10pct_sellers_gmv_share']:.0%}",
            help="Supplier concentration")
c = st.columns(4)
c[0].metric("OOT AUC: hand-off (EB score)", f"{ho.loc['Shrunk hand-off late rate (EB) only', 'item_auc']:.3f}",
            f"{ho.loc['Shrunk hand-off late rate (EB) only', 'item_auc'] - ho.loc['Baseline: raw hand-off late rate', 'item_auc']:+.3f} vs raw rate")
c[1].metric("OOT AUC: hand-off (composite)", f"{ho.loc['Composite risk score (shrunk)', 'item_auc']:.3f}",
            f"{ho.loc['Composite risk score (shrunk)', 'item_auc'] - ho.loc['Baseline: volume only (smaller = riskier)', 'item_auc']:+.3f} vs volume-only")
c[2].metric("Top-decile lift (hand-off)", f"{ho.loc['Composite risk score (shrunk)', 'lift_top10']:.2f}x",
            help="Next-period hand-off late rate of the riskiest 10% of sellers vs all scored sellers")
e = t["exposure_totals"]
c[3].metric("GMV at risk (shrunk late rate)", brl(e["gmv_at_risk"]), f"of {brl(e['gmv'])} GMV", delta_color="off")

st.subheader("Where the risk sits")
tier = csv("risk_tier_summary.csv").set_index("risk_tier").reindex(TIER_ORDER).reset_index()
left, right = st.columns([3, 2])
with left:
    m = tier.melt(id_vars="risk_tier", value_vars=["gmv_share", "late_gmv_share"], var_name="metric")
    m["metric"] = m.metric.map({"gmv_share": "share of GMV", "late_gmv_share": "share of late GMV"})
    fig = px.bar(m, x="risk_tier", y="value", color="metric", barmode="group", text_auto=".0%",
                 color_discrete_sequence=[NAVY, RED], labels={"risk_tier": "risk tier", "value": "", "metric": ""})
    fig.update_layout(title="GMV vs late GMV by risk tier (tiers = score percentiles 90+/80-90/50-80/<50)",
                      height=380, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
with right:
    hi = t["high_tier"]
    sc = sellers()
    mat = sc[sc.n_items >= 50]
    hi_mat = mat[mat.risk_tier.isin(["High", "Elevated"])]
    st.markdown("#### Key recommendations")
    st.markdown(f"""
1. **Make seller hand-off the supplier KPI.** A late hand-off raises the late-delivery rate from
   {q['late_rate_if_handoff_on_time']:.1%} to {q['late_rate_if_handoff_late']:.0%}; the shrunk hand-off rate is the
   most predictive out-of-time signal (AUC {ho.loc['Shrunk hand-off late rate (EB) only', 'item_auc']:.3f}).
2. **Review the {len(hi_mat):,} material sellers (>=50 items) in High/Elevated tiers** - they hold
   {hi_mat.gmv.sum() / sc.gmv.sum():.0%} of GMV and {hi_mat.late_gmv.sum() / sc.late_gmv.sum():.0%} of late GMV.
3. **Don't over-react to tiny sellers.** The High tier (n={hi['n']}) has median {hi['median_items']:.0f} items;
   shrinkage and the wide intervals shown on *Seller scorecards* exist to stop one bad order triggering action.
4. **Expect modest reallocation gains.** End-to-end lateness is mostly lane/carrier/calendar driven
   (seller ICC {s['r_summary']['glmm_icc_without_handoff']:.0%}); see the *Exposure & scenario* simulator.
""")

st.subheader("Score tiers")
fig = px.scatter(sellers().query("n_items >= 20"), x="gmv", y="risk_score", color="risk_tier", log_x=True,
                 color_discrete_map=TIER_COLORS, category_orders={"risk_tier": TIER_ORDER}, opacity=0.65,
                 hover_data={"seller_id": True, "seller_state": True, "n_items": True, "late_rate_shrunk": ":.1%"},
                 labels={"gmv": "seller GMV (BRL, log)", "risk_score": "composite risk score", "risk_tier": "tier"})
fig.update_layout(height=420, title="Risk score vs GMV (sellers with >= 20 items)")
st.plotly_chart(fig, width="stretch")
