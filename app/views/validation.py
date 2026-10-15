import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import GREY, NAVY, ORANGE, RED, TEAL, csv, scope_note, setup_page, summary

s = summary()
t = s["train_summary"]
m = t["oot_main"]
setup_page("Risk model & validation", f"Out-of-time: score vendors with outcomes known before {m['score_end'][:10]}, test on lines "
           f"scheduled {m['score_end'][:10]} to {m['validate_end'][:10]} · baselines: raw rate, volume-only, lane-mix-only",
           section="risk model & validation")
scope_note()
ev = csv("oot_metrics_main.csv")
targets = list(ev.target.unique())
target = st.radio("Next-period target", targets, horizontal=True)
e = ev[ev.target == target]
C = "Composite risk score (shrunk)"
c = st.columns(4)
best = e.loc[e.line_auc.idxmax()]
c[0].metric("Vendors evaluated", f"{m['n_vendors_evaluated']:,}", f"{m['next_lines_covered_share']:.0%} of next-period lines", delta_color="off")
c[1].metric("Best line-level AUC", f"{best.line_auc:.3f}", best.score.replace("Baseline: ", ""), delta_color="off")
c[2].metric("Composite top-20% lift", f"{e.set_index('score').loc[C, 'lift_top20']:.2f}x", "riskiest 20% of vendors", delta_color="off")
c[3].metric("Next-period events", f"{int(e.n_next_late_lines.iloc[0]):,}", f"base rate {e.base_late_rate.iloc[0]:.1%}", delta_color="off")

cols = [NAVY if not x.startswith("Baseline") else GREY for x in e.score]
fig = go.Figure(go.Bar(x=e.line_auc, y=e.score, orientation="h", marker_color=cols, text=e.line_auc.round(3), textposition="outside"))
fig.add_vline(x=0.5, line_dash="dash", line_color=RED)
fig.update_layout(height=420, xaxis_range=[0.15, 0.85], yaxis_autorange="reversed", title=f"Line-level ROC-AUC · target: {target}",
                  xaxis_title="AUC (below 0.5 = ranks the wrong way)")
st.plotly_chart(fig, width="stretch")
st.dataframe(e.drop(columns=["target"]).round(4), hide_index=True, width="stretch")

d = t["auc_deltas"]
key = "late" if target == targets[0] else "severe"
c = st.columns(4)
for col, (nm, lab) in zip(c, [(f"eb_late_vs_raw_late_{key}", "EB late rate vs raw late rate"),
                              (f"composite_vs_raw_late_{key}", "Composite vs raw late rate"),
                              (f"composite_vs_volume_{key}", "Composite vs volume-only"),
                              (f"composite_vs_lane_only_{key}", "Composite vs lane-mix-only")], strict=True):
    x = d[nm]
    col.metric(f"AUC diff: {lab}", f"{x['delta']:+.3f}", f"95% CI {x['lo95']:+.3f} to {x['hi95']:+.3f}", delta_color="off",
               help="Vendor-cluster bootstrap (200 reps)")
st.markdown(f"**Honest reading.** The simplest informative baseline wins: the raw late rate scores "
            f"{e.set_index('score').loc['Baseline: raw late rate', 'line_auc']:.3f} and the shrunk late rate "
            f"{e.set_index('score').loc['Shrunk late rate (EB) only', 'line_auc']:.3f}, while the composite scores "
            f"{e.set_index('score').loc[C, 'line_auc']:.3f}. With only {m['n_vendors_evaluated']} evaluable vendors, every interval above "
            "is wide and most differences are not distinguishable from zero. Volume-only and lane-mix-only baselines rank in the "
            "wrong direction. Weights were fixed a priori and not tuned on these results.")

_bt0 = csv("oot_backtest_metrics.csv")
_bt0 = _bt0[_bt0.target == target]
_bt = _bt0.pivot(index="split", columns="score", values="line_auc")
st.caption(f"Across the {len(_bt)} rolling splits the composite beats the raw late rate in {int((_bt[C] > _bt['Baseline: raw late rate']).sum())} "
           f"and loses in {int((_bt[C] < _bt['Baseline: raw late rate']).sum())}; it beats the EB late rate alone in "
           f"{int((_bt[C] > _bt['Shrunk late rate (EB) only']).sum())}. Each split has only {int(_bt0.n_vendors.min())} to {int(_bt0.n_vendors.max())} evaluable vendors.")

c1, c2 = st.columns(2)
with c1:
    q = csv("oot_quintiles.csv")
    fig = go.Figure()
    fig.add_bar(x=q.quintile, y=q.next_late_rate, name="by composite score", marker_color=NAVY)
    fig.add_bar(x=q.quintile, y=q.next_late_rate_raw_baseline, name="by raw late rate", marker_color=GREY)
    fig.update_layout(barmode="group", height=360, yaxis_tickformat=".1%", xaxis_title="vendor quintile (5 = riskiest)",
                      title="Next-period late rate by score quintile")
    st.plotly_chart(fig, width="stretch")
with c2:
    bt = csv("oot_backtest_metrics.csv")
    bt = bt[(bt.target == target) & bt.score.isin([C, "Shrunk late rate (EB) only", "Baseline: raw late rate",
                                                   "Baseline: volume only (smaller = riskier)"])]
    fig = px.line(bt, x="split", y="line_auc", color="score", markers=True, color_discrete_sequence=[NAVY, TEAL, GREY, ORANGE],
                  labels={"split": "score date to validate end"})
    fig.add_hline(y=0.5, line_dash="dash", line_color=RED)
    fig.update_layout(height=360, title="Rolling back-tests (4 splits)", legend=dict(y=-0.5))
    st.plotly_chart(fig, width="stretch")

st.subheader("Shrinkage error reduction for thin vendors")
cal = csv("oot_shrinkage_calibration.csv")
fig = px.bar(cal.melt(id_vars="vol_band", value_vars=["mae_raw", "mae_shrunk"]), x="vol_band", y="value", color="variable",
             barmode="group", color_discrete_sequence=[GREY, NAVY], text_auto=".3f",
             labels={"vol_band": "lines in score period", "value": "line-weighted MAE vs next-period late rate"})
fig.update_layout(height=340)
st.plotly_chart(fig, width="stretch")
cal_i = cal.set_index("vol_band")
st.markdown("Line-weighted absolute error of the late-rate estimate against the next-period rate, by history volume. Shrinkage "
            f"helps in {int((cal.mae_shrunk < cal.mae_raw).sum())} of {len(cal)} volume bands; with {int(cal_i.loc['1-4', 'next_lines'])} "
            "next-period lines in the thinnest band the comparison there is not informative (a raw rate of 0 can look exact).")

st.subheader("Cold-start vendors: prior-mean default checked out-of-time")
cs = csv("cold_start.csv")
_c = cs[cs.split == f"{m['score_end'][:10]}->{m['validate_end'][:10]}"].iloc[0]
st.markdown(
    f"Policy: vendors with fewer than {s['config']['scoring']['min_lines_for_score']} resolved lines at the score date receive the "
    f"empirical-Bayes **prior mean** and the **New / unscored** tier. In the main split they are {int(_c.n_new_vendors):,} vendors holding "
    f"**{_c.new_share_next_lines:.1%} of next-period lines** ({_c.new_share_next_value:.1%} of value). Their pooled late rate was "
    f"**{_c.new_late:.2%}** (95% Jeffreys {_c.new_late_lo95:.2%}-{_c.new_late_hi95:.2%}) vs an assigned prior of "
    f"{_c.prior_mean_late:.2%} and {_c.scored_late:.2%} for scored vendors. The prior is inside the interval in "
    f"{int(cs.prior_inside_ci.sum())} of {len(cs)} splits; where it is outside it is too pessimistic for new vendors. Full-coverage line AUC "
    f"{_c.late_auc_full_coverage:.3f} vs {_c.late_auc_scored_only:.3f} for scored vendors only.")
fig = go.Figure()
fig.add_bar(x=cs.split, y=cs.prior_mean_late, name="prior mean (assigned)", marker_color=GREY)
fig.add_bar(x=cs.split, y=cs.new_late, name="new / thin vendors, observed next period", marker_color=ORANGE,
            error_y=dict(type="data", symmetric=False, array=cs.new_late_hi95 - cs.new_late, arrayminus=cs.new_late - cs.new_late_lo95))
fig.add_bar(x=cs.split, y=cs.scored_late, name="scored vendors, observed next period", marker_color=TEAL)
fig.update_layout(barmode="group", height=320, yaxis_tickformat=".1%", title="Next-period late rate", xaxis_title="score date to validate end")
st.plotly_chart(fig, width="stretch")
st.subheader("Weight sensitivity (full-period ranking)")
st.dataframe(csv("weight_sensitivity_ranking.csv").round(3), hide_index=True, width="stretch")
