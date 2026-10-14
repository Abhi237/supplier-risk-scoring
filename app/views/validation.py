import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import GREY, NAVY, ORANGE, RED, TEAL, csv, proxy_note, setup_page, summary

s = summary()
t = s["train_summary"]
m = t["oot_main"]
setup_page("Risk model & validation", f"Out-of-time: score sellers with data before {m['score_end'][:10]}, test on "
           f"items purchased {m['score_end'][:10]} → {m['validate_end'][:10]} · baselines: raw rates, volume-only", section="risk model & validation")
proxy_note()
ev = csv("oot_metrics_main.csv")
target = st.radio("Next-period target", ["Seller hand-off late", "End-to-end late delivery"], horizontal=True)
e = ev[ev.target == target]
c = st.columns(4)
best = e.loc[e.item_auc.idxmax()]
c[0].metric("Sellers evaluated", f"{m['n_sellers_evaluated']:,}",
            f"{m['next_items_covered_share']:.0%} of next-period items", delta_color="off")
c[1].metric("Best item-level AUC", f"{best.item_auc:.3f}", best.score.replace("Baseline: ", ""), delta_color="off")
c[2].metric("Composite top-decile lift", f"{e.set_index('score').loc['Composite risk score (shrunk)', 'lift_top10']:.2f}x")
c[3].metric("Next-period base rate", f"{e.base_late_rate.iloc[0]:.1%}")

cols = [NAVY if not x.startswith("Baseline") else GREY for x in e.score]
fig = go.Figure(go.Bar(x=e.item_auc, y=e.score, orientation="h", marker_color=cols, text=e.item_auc.round(3),
                       textposition="outside"))
fig.add_vline(x=0.5, line_dash="dash", line_color=RED)
fig.update_layout(height=380, xaxis_range=[0.45, max(0.75, e.item_auc.max() + 0.04)], yaxis_autorange="reversed",
                  title=f"Item-level ROC-AUC · target: {target}", xaxis_title="AUC")
st.plotly_chart(fig, width="stretch")
st.dataframe(e.drop(columns="target").round(4), hide_index=True, width="stretch")

d = t["auc_deltas"]
key = "handoff" if target.startswith("Seller") else "late"
c = st.columns(3)
for col, (nm, lab) in zip(c, [(f"eb_handoff_vs_raw_handoff_{key}", "EB hand-off vs raw hand-off"),
                              (f"composite_vs_raw_late_{key}", "Composite vs raw late rate"),
                              (f"composite_vs_volume_{key}", "Composite vs volume-only")]):
    x = d[nm]
    col.metric(f"ΔAUC {lab}", f"{x['delta']:+.3f}", f"95% CI {x['lo95']:+.3f} to {x['hi95']:+.3f}", delta_color="off",
               help="Seller-cluster bootstrap (200 reps)")

c1, c2 = st.columns(2)
with c1:
    dec = csv("oot_deciles.csv")
    fig = go.Figure()
    fig.add_bar(x=dec.decile, y=dec.next_handoff_late_rate, name="hand-off late", marker_color=NAVY)
    fig.add_bar(x=dec.decile, y=dec.next_late_rate, name="late vs estimate", marker_color=RED)
    fig.update_layout(barmode="group", height=360, yaxis_tickformat=".0%", xaxis_title="composite score decile (10 = riskiest)",
                      title="Next-period rates by score decile")
    st.plotly_chart(fig, width="stretch")
with c2:
    bt = csv("oot_backtest_metrics.csv")
    bt = bt[(bt.target == target) & bt.score.isin(["Composite risk score (shrunk)", "Shrunk hand-off late rate (EB) only",
                                                   "Baseline: raw late rate", "Baseline: raw hand-off late rate"])]
    fig = px.line(bt, x="split", y="item_auc", color="score", markers=True,
                  color_discrete_sequence=[NAVY, TEAL, GREY, ORANGE], labels={"split": "score date → validate end"})
    fig.add_hline(y=0.5, line_dash="dash", line_color=RED)
    fig.update_layout(height=360, title="Rolling back-tests (3 splits)", legend=dict(y=-0.45))
    st.plotly_chart(fig, width="stretch")

st.subheader("Shrinkage error reduction for small sellers")
cal = csv("oot_shrinkage_calibration.csv")
fig = px.bar(cal.melt(id_vars="vol_band", value_vars=["mae_raw", "mae_shrunk"]), x="vol_band", y="value",
             color="variable", barmode="group", color_discrete_sequence=[GREY, NAVY], text_auto=".3f",
             labels={"vol_band": "delivered items in score period", "value": "item-weighted MAE vs next-period late rate"})
fig.update_layout(height=340)
st.plotly_chart(fig, width="stretch")
st.markdown("Shrinkage roughly **halves the error for sellers with <5 deliveries** and converges to the raw rate for "
            "large sellers - exactly the intended behaviour. The composite is not a better *lateness* predictor than "
            "the single hand-off rate; it adds review, cancellation and freight signals for triage.")
st.subheader("Weight sensitivity (full-period ranking)")
st.dataframe(csv("weight_sensitivity_ranking.csv").round(3), hide_index=True, width="stretch")
