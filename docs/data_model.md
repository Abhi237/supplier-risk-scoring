# Data model

One raw table (the USAID / PEPFAR SCMS Delivery History, one row per order line) and the DuckDB model built by `sql/01..06`.

```mermaid
erDiagram
    RAW_SCMS ||--o{ STG_LINES : "typed by 01_stage"
    STG_LINES ||--o{ FCT_LINES : "03_line_intervals"
    STG_LINES ||--o{ FCT_SHIPMENTS : "grouped by ASN/DN"
    FCT_SHIPMENTS ||--o{ FCT_LINES : "freight, weight, value per shipment"
    FCT_LINES ||--o{ VENDOR_SUMMARY : "04"
    FCT_LINES ||--o{ VENDOR_LANE : "04"
    FCT_LINES ||--o{ VENDOR_YEAR_PANEL : "05 window functions"
    FCT_LINES ||--o{ OOT_FEATURES : "06 point-in-time, known_date < score_end"
    FCT_LINES ||--o{ OOT_OUTCOMES : "06 scheduled in [score_end, validate_end)"
    STG_LINES ||--o{ CLEANING_LOG : "02 counted actions"
    STG_LINES ||--o{ DQ_REPORT : "02 checks"

    FCT_LINES { int line_id PK  varchar shipment_id FK  varchar vendor  bool is_internal_rdc  varchar country  varchar ship_mode  date scheduled_date  date delivered_date  date known_date  bool is_late  int days_late  int days_late_capped  bool is_severe_late  double line_value  double freight_obs_usd }
    FCT_SHIPMENTS { varchar shipment_id PK  int n_lines  double ship_value  double ship_freight_usd  bool has_numeric_freight }
    VENDOR_SUMMARY { varchar vendor PK  bool is_internal_rdc  int n_lines  int n_late  double value_usd  double country_hhi }
    OOT_FEATURES { varchar vendor PK  int n_lines  int n_late  date max_known_date_used }
    OOT_OUTCOMES { varchar vendor PK  int next_n_lines  int next_n_late  double next_late_value }
```

Grain checks (unique keys, row preservation) are enforced in `tests/test_sql_contracts.py`.
The internal regional distribution centre (`SCMS from RDC`) is kept in `fct_lines` and `vendor_summary` with `is_internal_rdc = TRUE`
and excluded from `vendor_lane`, the point-in-time tables and the scored population.
