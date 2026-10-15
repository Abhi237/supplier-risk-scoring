import plotly.express as px
import streamlit as st
from core import NAVY, SOURCE_CITATION, csv, scope_note, setup_page, summary

s = summary()
q, man = s["sql_summary"], s["data_manifest"]
cl = s["config"]["cleaning"]
setup_page("Data & quality", "USAID / PEPFAR SCMS delivery history · integrity-checked against the documented shape "
           "(10,324 rows, 33 columns) · numbered DuckDB SQL model · every cleaning action counted", section="data & quality")
scope_note()
c = st.columns(5)
c[0].metric("Order lines", f"{q['n_lines']:,}", "matches documented 10,324", delta_color="off")
c[1].metric("Shipments (ASN / DN)", f"{q['n_shipments']:,}")
c[2].metric("Vendor strings", f"{q['n_vendor_strings']:,}", f"{q['ext_vendors']} external + SCMS RDC", delta_color="off")
c[3].metric("Destination countries", f"{q['n_countries']:,}")
c[4].metric("Scheduled dates", f"{q['scheduled_min'][:4]} to {q['scheduled_max'][:4]}")

st.subheader("Source and licence")
st.markdown(f"""
* **Citation:** {SOURCE_CITATION}.
* **Obtained from:** `{man['source']}` (HTTP, no credentials). Failed before it: {len(man['failed_sources_before'])} source(s) -
  the data.usaid.gov catalogue export and the usaid.gov original were not reachable from the build machine. The file used is the Internet
  Archive snapshot of the original `usaid.gov/opengov/developer/datasets/SCMS_Delivery_History_Dataset_20150929.csv`. A second Archive snapshot
  (2020) is identical cell for cell; a public GitHub mirror is identical in every column the analysis uses (it differs only in two-digit year
  formatting of two unused procurement date columns). Source SHA-256 `{man['source_sha256'][:16]}...`.
* **Licence is unclear.** This repository and app publish only derived vendor scorecards and aggregates. The raw CSV is git-ignored and
  fetched by `make data`. Vendor names are public in the dataset.
""")

st.subheader("Pipeline")
st.graphviz_chart("""
digraph { rankdir=LR; bgcolor="transparent"; edge [color="#9fb0c8"]; node [shape=box, style="rounded,filled", fillcolor="#16213a", color="#2dd4bf", fontcolor="#e6edf7", fontname=Helvetica];
 raw [label="SCMS CSV\\n(USAID original via\\nInternet Archive)"]; stg [label="01 staging\\n(typed, text values\\nkept as kinds)"];
 dq [label="02 cleaning log\\n+ DQ checks"]; fct [label="03 line fact +\\nshipment freight"]; vs [label="04 vendor summary"];
 pm [label="05 vendor-year panel\\n(window functions)"]; oot [label="06 point-in-time\\nfeatures + outcomes"];
 py [label="Python: EB shrinkage,\\nbootstrap, composite,\\nclusters, OOT, exposure"]; r [label="R: GLM, GLMM,\\nNB, KS/Wilcoxon"];
 app [label="app/data\\n(aggregates only)"];
 raw->stg->dq; stg->fct->vs->py; fct->pm; fct->oot->py; fct->r; py->app; r->app; }
""")

st.subheader("Cleaning log (sql/02_cleaning_and_quality.sql)")
st.caption("One row per cleaning or definition action; counts come from the data. Zero rows affected is reported as zero.")
log = csv("cleaning_log.csv")
log["share_of_base"] = log.n_rows / log.base
st.dataframe(log[["action", "n_rows", "base", "share_of_base", "detail"]], hide_index=True, width="stretch", height=420,
             column_config={"share_of_base": st.column_config.ProgressColumn("share of base", format="%.1f%%", min_value=0, max_value=1)})
issues = log[~log.action.isin(["rows_in_raw_file", "vendor_names_distinct_as_recorded"]) & (log.n_rows > 0)].sort_values("n_rows")
fig = px.bar(issues, x="n_rows", y="action", orientation="h", text="n_rows", color_discrete_sequence=[NAVY],
             labels={"n_rows": "rows affected", "action": ""})
fig.update_layout(height=620, title="Rows affected by each cleaning action")
st.plotly_chart(fig, width="stretch")

st.subheader("Data-quality checks (dq_report)")
dq = csv("dq_report.csv")
dq["share"] = dq.n / dq.total
st.dataframe(dq, hide_index=True, width="stretch",
             column_config={"share": st.column_config.ProgressColumn("share of base", format="%.1f%%", min_value=0, max_value=1)})
st.markdown(f"""
**Definitions and cleaning decisions**
* **On time** = `Delivered to Client Date` <= `Scheduled Delivery Date`. Lateness days = delivered minus scheduled (positive part;
  capped at {cl['days_late_cap']} for mean-lateness metrics only). Severe late = more than {cl['severe_late_days']} days late.
* **No vendor hand-off timestamp exists**, so the hand-off vs customer-delivery split used in the earlier Olist version of this project is
  dropped. Lateness here mixes vendor, freight-forwarder and in-country effects.
* **{q['ext_exact_match_share']:.0%} of external-vendor lines have delivered date = scheduled date exactly** (vs
  {q['rdc_exact_match_share']:.0%} for the regional DC). If scheduled dates were back-filled with actual dates in some records, the on-time rate is
  overstated; there is no way to verify this from the file.
* No 'Date Not Captured' strings occur in the three delivery dates (0 unparseable); they occur in the procurement dates (PO sent:
  {q['cleaning']['po_sent_date_not_captured']:,} lines), which are not used.
* Freight text is typed, not coerced: 'Freight Included in Commodity Cost' ({q['cleaning']['freight_included_in_commodity_cost']:,}),
  'Invoiced Separately' ({q['cleaning']['freight_invoiced_separately']:,}) and 'See ASN/DN-... (ID#:n)' ({q['cleaning']['freight_see_reference']:,}).
  Reference lines are resolved at shipment grain through the first line, and the freight ratio uses shipments with numeric freight
  only. Freight above {cl['freight_ratio_cap']:.0f}x shipment value is capped ({q['cleaning']['freight_ratio_capped']:,} shipments).
* Weights are cleaned and logged but not modelled ({q['cleaning']['weight_see_reference'] + q['cleaning']['weight_captured_separately']:,} lines have no usable weight).
* The vendor **'{cl['internal_channel_vendor']}'** is SCMS's own regional distribution centre ({q['rdc_line_share']:.0%} of lines), not an external
  vendor. It is kept as a reference channel and excluded from vendor scoring.
* Only delivered lines are in the file; undelivered or cancelled lines are absent, so late rates are lower bounds.
* Vendor spellings are used as recorded ({q['n_vendor_strings']} strings); possible variants (for example Abbott / AbbVie entities) are not merged.
""")
