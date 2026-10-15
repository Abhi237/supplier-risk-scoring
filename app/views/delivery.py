import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import GREY, NAVY, ORANGE, RED, SEQ_RISK, TEAL, csv, scope_note, setup_page, summary

s = summary()
q, r = s["sql_summary"], s["r_summary"]
setup_page("Delivery performance", "Delivered-to-client date vs scheduled delivery date · external vendors and the SCMS regional "
           "distribution centre · logistics lanes and the R statistical layer", section="delivery performance")
scope_note()
c = st.columns(4)
c[0].metric("Vendor late rate", f"{q['ext_late_rate']:.1%}", f"severe (>14 d) {q['ext_severe_late_rate']:.1%}", delta_color="off")
c[1].metric("Regional DC late rate", f"{q['rdc_late_rate']:.1%}", f"severe {q['rdc_severe_late_rate']:.1%}", delta_color="off")
c[2].metric("Mean days late (when late)", f"{q['ext_mean_days_late_when_late']:.1f}",
            f"p90 {q['ext_p90_days_late_when_late']:.0f} days", delta_color="off")
c[3].metric("Late value share", f"{q['ext_late_value_share']:.1%}", f"{q['ext_late_value_usd'] / 1e6:,.1f}m of {q['ext_value_usd'] / 1e6:,.0f}m USD",
            delta_color="off")

py = csv("programme_year.csv")
fig = go.Figure()
fig.add_bar(x=py.sched_year, y=py.n_lines_vendor, name="vendor lines (rhs)", marker_color="#24324f", yaxis="y2")
fig.add_scatter(x=py.sched_year, y=py.n_late_vendor / py.n_lines_vendor, name="late, external vendors", line=dict(color=RED, width=3))
fig.add_scatter(x=py.sched_year, y=py.n_late_rdc / py.n_lines_rdc.where(py.n_lines_rdc > 0), name="late, SCMS regional DC",
                line=dict(color=NAVY, width=3))
fig.add_scatter(x=py.sched_year, y=py.n_exact_vendor / py.n_lines_vendor, name="delivered = scheduled, vendors",
                line=dict(color=GREY, width=2, dash="dash"))
fig.update_layout(title="Lateness by scheduled year", height=400, yaxis=dict(tickformat=".0%", title="share of lines"),
                  yaxis2=dict(overlaying="y", side="right", showgrid=False, title="vendor lines"),
                  xaxis=dict(dtick=1))
st.plotly_chart(fig, width="stretch")
_ex = py.n_exact_vendor / py.n_lines_vendor
_rl = py.n_late_rdc / py.n_lines_rdc.where(py.n_lines_rdc > 0)
_vl = py.n_late_vendor / py.n_lines_vendor
st.caption(f"The share of vendor lines whose delivered and scheduled dates are identical is {_ex.min():.0%} to {_ex.max():.0%} by year, so "
           f"the observable lateness signal is thin and period-dependent. Vendor late rate peaks in {int(py.sched_year[_vl.idxmax()])} "
           f"({_vl.max():.1%}); the regional DC peaks in {int(py.sched_year[_rl.idxmax()])} ({_rl.max():.1%}).")

c1, c2 = st.columns(2)
with c1:
    oh = csv("offset_hist.csv")
    oh["share"] = oh.n / oh.groupby("channel").n.transform("sum")
    fig = px.bar(oh, x="offset_bin", y="share", color="channel", barmode="group", log_y=True,
                 color_discrete_sequence=[NAVY, ORANGE],
                 labels={"offset_bin": "delivered minus scheduled (days, 5-day bins, clipped -60..120)", "share": "share of lines (log)"})
    fig.update_layout(title="Distribution of days late / early", height=380)
    st.plotly_chart(fig, width="stretch")
with c2:
    tests = csv("r_offset_tests.csv")
    t0 = tests[tests.comparison.str.startswith("External vendors")].iloc[0]
    st.markdown(f"**Vendors vs SCMS regional DC:** late share {t0.share_late_a:.1%} vs {t0.share_late_b:.1%}; KS D = {t0.ks_D:.2f}. "
                f"In the all-lines GLM the regional DC has odds ratio **{r['or_internal_rdc_vs_vendor']:.1f}** "
                f"(95% CI {r['or_internal_rdc_lo']:.1f}-{r['or_internal_rdc_hi']:.1f}) of lateness versus vendor lines after controls.")
    st.dataframe(tests[["comparison", "n_a", "n_b", "share_late_a", "share_late_b", "ks_D", "ks_p", "wilcox_p"]].round(4),
                 hide_index=True, width="stretch", height=250)

st.subheader("By country, mode and product group (external vendors)")
c1, c2 = st.columns([3, 2])
with c1:
    ct = csv("country.csv")
    ct["late_rate"] = ct.n_late / ct.n_lines
    ct = ct[ct.n_lines >= 30].sort_values("late_rate", ascending=False)
    fig = px.bar(ct, x="country", y="late_rate", color="n_lines", color_continuous_scale=SEQ_RISK,
                 labels={"country": "", "late_rate": "late rate", "n_lines": "lines"})
    fig.update_layout(height=380, yaxis_tickformat=".0%", title="Late rate by destination country (>= 30 lines)")
    st.plotly_chart(fig, width="stretch")
with c2:
    md = csv("ship_mode.csv")
    md["late_rate"] = md.n_late / md.n_lines
    fig = px.bar(md, x="ship_mode", y="late_rate", text=md.n_lines.map(lambda n: f"n={n:,}"), color_discrete_sequence=[TEAL],
                 labels={"ship_mode": "shipment mode", "late_rate": "late rate"})
    fig.update_layout(height=380, yaxis_tickformat=".0%", title="Late rate by shipment mode")
    st.plotly_chart(fig, width="stretch")
pg = csv("product_group.csv")
pg["late_rate"] = pg.n_late / pg.n_lines
st.dataframe(pg.round(4), hide_index=True, width="stretch")

st.subheader("R statistical layer (GLM / GLMM / negative binomial)")
orr = csv("r_glm_odds_ratios.csv")
orr = orr[~orr.term.str.match(r"^(\(Intercept\)|country_g)")]
fig = go.Figure(go.Scatter(x=orr.estimate, y=orr.term, mode="markers", marker=dict(color=NAVY, size=9),
                           error_x=dict(type="data", symmetric=False, array=orr["conf.high"] - orr.estimate,
                                        arrayminus=orr.estimate - orr["conf.low"], color="#9fb0c8")))
fig.add_vline(x=1, line_dash="dash", line_color=RED)
fig.update_layout(xaxis_type="log", height=460, title="Logistic GLM odds ratios for late delivery (95% CI; country terms omitted)",
                  xaxis_title="odds ratio (log)")
st.plotly_chart(fig, width="stretch")
st.markdown(f"On {r['n_lines']:,} external-vendor lines ({r['n_late']} late), a vendor random-intercept GLMM on the {r['glmm_n_vendors']} vendors with "
            f">= 5 lines gives a vendor SD of {r['glmm_vendor_sd']:.2f} on the logit scale (latent ICC **{r['glmm_icc']:.1%}**, "
            f"{r['glmm_icc_null']:.1%} with no covariates; shipment-grain sensitivity {r['glmm_ship_icc']:.1%}; median odds ratio "
            f"{r['glmm_mor']:.1f}). A stable vendor-level component is detectable, but it partly reflects each vendor's product, country and period niche. "
            f"A shifted negative binomial for days late among the {r['nb_n_late_lines']} late lines has theta {r['nb_theta']:.2f} "
            f"(Poisson dispersion {r['poisson_dispersion']:.0f}). Full report: `reports/delivery_report.html`.")
