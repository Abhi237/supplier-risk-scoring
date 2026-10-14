# Data model

Olist source tables (left) and the DuckDB model built by `sql/01..06` (right). **Sellers are proxies for suppliers.**

```mermaid
erDiagram
    ORDERS ||--o{ ORDER_ITEMS : contains
    SELLERS ||--o{ ORDER_ITEMS : fulfils
    PRODUCTS ||--o{ ORDER_ITEMS : "is sold as"
    CUSTOMERS ||--o{ ORDERS : places
    ORDERS ||--o| ORDER_REVIEWS : "reviewed by (latest kept)"
    CATEGORY_TRANSLATION ||--o{ PRODUCTS : names

    ORDERS { varchar order_id PK  varchar customer_id FK  varchar order_status  timestamp purchase_ts  timestamp approved_ts  timestamp carrier_ts  timestamp delivered_ts  timestamp estimated_ts }
    ORDER_ITEMS { varchar order_id PK  int order_item_id PK  varchar seller_id FK  varchar product_id FK  timestamp shipping_limit_ts  decimal price  decimal freight_value }
    SELLERS { varchar seller_id PK  varchar seller_zip_prefix  varchar seller_city  varchar seller_state }
    CUSTOMERS { varchar customer_id PK  varchar customer_state }
    PRODUCTS { varchar product_id PK  varchar category  double weight_g }
    ORDER_REVIEWS { varchar order_id PK  int review_score  timestamp review_created_ts }
```

```mermaid
erDiagram
    FCT_ITEMS }o--|| SELLER_SUMMARY : "aggregates to"
    FCT_ITEMS }o--|| SELLER_MONTH_PANEL : "aggregates to"
    FCT_ITEMS }o--|| FCT_ORDERS_DELIVERED : "rolls up to"
    FCT_ITEMS }o--|| OOT_FEATURES : "point-in-time (< score_end)"
    FCT_ITEMS }o--|| OOT_OUTCOMES : "[score_end, validate_end)"
    PARAMS ||--|| OOT_FEATURES : parameterises

    FCT_ITEMS { varchar order_id PK  int order_item_id PK  varchar seller_id  bool is_late  bool is_handoff_late  bool is_canceled  double handoff_delay_days  double delivery_days  int delay_days  double freight_to_price }
    SELLER_SUMMARY { varchar seller_id PK  int n_items  int n_late  int n_delivered  int n_handoff_late  double gmv  double late_gmv  double mean_review_score }
    SELLER_MONTH_PANEL { varchar seller_id PK  date purchase_month PK  double late_rate  double late_rate_roll3  int cum_items  int gmv_rank_in_month }
    FCT_ORDERS_DELIVERED { varchar order_id PK  bool is_late  bool is_handoff_late  bool same_state  double delivery_days }
    OOT_FEATURES { varchar seller_id PK  int n_late  int n_delivered  timestamp max_delivered_ts_used }
    OOT_OUTCOMES { varchar seller_id PK  int next_n_late  int next_n_delivered  int next_n_handoff_late }
    PARAMS { timestamp score_end  timestamp validate_end }
```

| Table | Grain | Built in |
|---|---|---|
| `stg_*` | as source (reviews: one per order) | `01_stage.sql` |
| `dq_report` | one row per check | `02_data_quality.sql` |
| `fct_items` | order x item | `03_order_intervals.sql` |
| `fct_orders_delivered` | delivered order | `03_order_intervals.sql` |
| `seller_summary` | seller | `04_seller_summary.sql` |
| `seller_month_panel`, `platform_month` | seller x month, month | `05_seller_month_panel.sql` |
| `oot_features`, `oot_outcomes` | seller (per split) | `06_oot_features.sql` |
