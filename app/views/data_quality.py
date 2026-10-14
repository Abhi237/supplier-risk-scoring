import plotly.express as px
import streamlit as st
from core import NAVY, csv, proxy_note, setup_page, summary

s = summary()
q = s["sql_summary"]
setup_page("Data & quality", "Olist Brazilian E-Commerce public dataset (CC BY-NC-SA 4.0) · integrity-checked "
           "against documented row counts · numbered DuckDB SQL model")
proxy_note()
c = st.columns(4)
c[0].metric("Orders", f"{q['dq']['orders_rows']:,}", "matches documented 99,441", delta_color="off")
c[1].metric("Order items", f"{q['dq']['order_items_rows']:,}", "matches documented 112,650", delta_color="off")
c[2].metric("Sellers", f"{q['dq']['sellers_rows']:,}", "matches documented 3,095", delta_color="off")
c[3].metric("Period (purchases)", f"{q['date_min'][:7]} → {q['date_max'][:7]}")

st.subheader("Pipeline")
st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#1f4e79", fontname=Helvetica];
 raw [label="Olist CSVs\\n(public mirror,\\nrow counts + SHA-256)"]; stg [label="01 staging\\n(typed, reviews dedup)"];
 dq [label="02 data-quality\\nchecks"]; fct [label="03 item fact +\\norder intervals"]; ss [label="04 seller summary"];
 pm [label="05 seller-month panel\\n(window functions)"]; oot [label="06 point-in-time\\nfeatures + outcomes"];
 py [label="Python: EB shrinkage,\\nbootstrap, composite,\\nclusters, OOT, exposure"]; r [label="R: GLM, GLMM,\\nNB, KS/Wilcoxon"];
 app [label="app/data\\n(aggregates only)"];
 raw->stg->dq; stg->fct->ss->py; fct->pm; fct->oot->py; fct->r; py->app; r->app; }
""")

st.subheader("Data-quality checks (sql/02_data_quality.sql)")
dq = csv("dq_report.csv")
dq["share"] = dq.n / dq.total
st.dataframe(dq, hide_index=True, width="stretch",
             column_config={"share": st.column_config.ProgressColumn("share of base", format="%.2f%%",
                                                                     min_value=0, max_value=1)})
issues = dq[~dq.check_name.str.endswith("_rows")].query("n > 0").sort_values("n")
fig = px.bar(issues, x="n", y="check_name", orientation="h", text="n", color_discrete_sequence=[NAVY],
             labels={"n": "rows flagged", "check_name": ""})
fig.update_layout(height=420, title="Flagged rows by check")
st.plotly_chart(fig, width="stretch")
st.markdown("""
**Cleaning / interpretation decisions**
* Lateness is judged only for `delivered` orders that have a customer-delivery timestamp; canceled/unavailable
  orders feed the **cancellation** component instead.
* Late = customer delivery **calendar date** after the estimated date (same-day = on time).
* Hand-off lateness is not judged when the carrier timestamp precedes the purchase (bad clock, 166 orders).
* Orders can have several reviews; the latest per order is kept (`ROW_NUMBER()`).
* Multi-seller orders share one carrier/delivery timestamp; every seller inherits it (flagged,
  {:.1%} of delivered orders).
* Customer data is reduced to **state**; no customer identifiers reach the app.
""".replace("{:.1%}", f"{q['multi_seller_order_share']:.1%}"))
