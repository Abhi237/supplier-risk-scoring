import plotly.express as px
import streamlit as st
from core import (
    AMBER,
    CYAN,
    MEMO_URL,
    RED,
    TIER_COLORS,
    TIER_ORDER,
    brl,
    callout,
    csv,
    proxy_note,
    sellers,
    setup_page,
    summary,
    tiles,
)

s = summary()
q, t = s["sql_summary"], s["train_summary"]
setup_page("Which vendors create the greatest delivery-risk and working-capital exposure?",
           "Executive summary · sellers scored with empirical-Bayes shrinkage and validated out-of-time · every number "
           "computed by the pipeline", section="executive summary")
callout("<b>Mission.</b> Find the suppliers that put on-time delivery and working capital at risk, say how sure we "
        "are about small suppliers with little history, and check whether today's score predicts <i>next period's</i> "
        "problems better than simple rules.")
proxy_note()

ev = csv("oot_metrics_main.csv").set_index(["target", "score"])
ho = ev.loc["Seller hand-off late"]
e = t["exposure_totals"]
EB, RH, C, VO = ("Shrunk hand-off late rate (EB) only", "Baseline: raw hand-off late rate",
                 "Composite risk score (shrunk)", "Baseline: volume only (smaller = riskier)")
sc = sellers()
mat = sc[sc.n_items >= 50]
watch = mat[mat.risk_tier.isin(["High", "Elevated"])]

st.subheader("Network status")
tiles([
    ("Order late-delivery rate", f"{q['order_late_rate']:.1%}", "delivered after the promised date", "amber"),
    ("Seller hand-off late", f"{q['item_handoff_late_rate']:.1%}", "handed to carrier after shipping limit", "amber"),
    ("Late rate if hand-off late", f"{q['late_rate_if_handoff_late']:.0%}",
     f"vs {q['late_rate_if_handoff_on_time']:.1%} when on time", "red"),
    ("Top-10% sellers' GMV share", f"{q['top10pct_sellers_gmv_share']:.0%}", "supplier concentration", "amber"),
])
st.subheader("Model status")
tiles([
    ("OOT AUC · hand-off (EB score)", f"{ho.loc[EB, 'item_auc']:.3f}",
     f"{ho.loc[EB, 'item_auc'] - ho.loc[RH, 'item_auc']:+.3f} vs raw rate", "green"),
    ("OOT AUC · hand-off (composite)", f"{ho.loc[C, 'item_auc']:.3f}",
     f"{ho.loc[C, 'item_auc'] - ho.loc[VO, 'item_auc']:+.3f} vs volume-only", "green"),
    ("Top-decile lift · hand-off", f"{ho.loc[C, 'lift_top10']:.2f}x", "riskiest 10% vs all scored sellers", "info"),
    ("GMV at risk (shrunk late rate)", brl(e["gmv_at_risk"]), f"of {brl(e['gmv'])} GMV", "red"),
])

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Risk concentration by tier")
    tier = csv("risk_tier_summary.csv").set_index("risk_tier").reindex(TIER_ORDER).reset_index()
    m = tier.melt(id_vars="risk_tier", value_vars=["gmv_share", "late_gmv_share"], var_name="metric")
    m["metric"] = m.metric.map({"gmv_share": "share of GMV", "late_gmv_share": "share of late GMV"})
    fig = px.bar(m, y="risk_tier", x="value", color="metric", barmode="group", orientation="h", text_auto=".0%",
                 color_discrete_sequence=[CYAN, RED], labels={"risk_tier": "", "value": "", "metric": ""})
    fig.update_layout(title="GMV vs LATE GMV BY RISK TIER (score pctl 90+ / 80-90 / 50-80 / <50)", height=360,
                      xaxis_tickformat=".0%", yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader(f"Alerts · {len(watch)} material sellers")
    st.caption("High / Elevated tier, >= 50 items · ranked by composite risk score")
    rows = []
    for r in watch.sort_values("risk_score", ascending=False).head(7).itertuples():
        sev = "red" if r.risk_tier == "High" else "amber"
        rows.append(f"<div class='alert {sev}'><div class='sev' style='color:{RED if sev == 'red' else AMBER}'>"
                    f"{r.risk_tier.upper()}</div><div><div class='id'>{r.seller_id[:14]}…</div>"
                    f"<div class='meta'>{r.seller_state} · {r.top_category} · {int(r.n_items)} items · late (EB) "
                    f"{r.late_rate_shrunk:.1%} · hand-off (EB) {r.seller_handoff_late_rate:.1%}</div></div>"
                    f"<div class='score'>{r.risk_score:.0f}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)

st.subheader("Recommended actions")
hi = t["high_tier"]
a = st.columns(4)
a[0].markdown(f"**01 · Hand-off is the supplier KPI.** A late hand-off raises the late-delivery rate from "
              f"{q['late_rate_if_handoff_on_time']:.1%} to {q['late_rate_if_handoff_late']:.0%}; the shrunk hand-off rate "
              f"is the most predictive out-of-time signal (AUC {ho.loc[EB, 'item_auc']:.3f}).")
a[1].markdown(f"**02 · Review {len(watch):,} material sellers** in High/Elevated tiers - "
              f"{watch.gmv.sum() / sc.gmv.sum():.0%} of GMV and {watch.late_gmv.sum() / sc.late_gmv.sum():.0%} of late GMV.")
a[2].markdown(f"**03 · Don't over-react to tiny sellers.** The High tier (n={hi['n']}) has median "
              f"{hi['median_items']:.0f} items; shrinkage and wide intervals stop one bad order triggering action.")
a[3].markdown(f"**04 · Expect modest reallocation gains.** End-to-end lateness is mostly lane/carrier/calendar driven "
              f"(seller ICC {s['r_summary']['glmm_icc_without_handoff']:.0%}). [Memo (PDF)]({MEMO_URL})")

st.subheader("Score map")
fig = px.scatter(sc.query("n_items >= 20"), x="gmv", y="risk_score", color="risk_tier", log_x=True,
                 color_discrete_map=TIER_COLORS, category_orders={"risk_tier": TIER_ORDER}, opacity=0.75,
                 hover_data={"seller_id": True, "seller_state": True, "n_items": True, "late_rate_shrunk": ":.1%"},
                 labels={"gmv": "seller GMV (BRL, log)", "risk_score": "composite risk score", "risk_tier": "tier"})
fig.update_layout(height=420, title="RISK SCORE vs GMV (sellers with >= 20 items)")
st.plotly_chart(fig, width="stretch")
