# Data dictionary

**Sellers are proxies for suppliers.** Monetary values are BRL. Columns marked ASSUMED depend on scenario parameters.

## `fct_items` (order x item)
| Column | Type | Definition |
|---|---|---|
| order_id, order_item_id | varchar, int | composite key |
| seller_id | varchar | anonymised seller hash (supplier proxy) |
| seller_state / customer_state | varchar | Brazilian UF |
| category | varchar | English product category (`unknown` if missing) |
| purchase_ts, approved_ts, carrier_ts, delivered_ts, estimated_ts | timestamp | order lifecycle timestamps |
| shipping_limit_ts | timestamp | seller's deadline to hand the item to the carrier |
| price, freight_value | double | item price and allocated freight |
| freight_to_price | double | freight_value / price |
| same_state | bool | seller and customer in the same state |
| is_multi_seller_order | bool | order has >1 seller (timestamps shared) |
| approval_hours | double | purchase -> approval |
| handoff_days | double | purchase -> carrier hand-off |
| handoff_delay_days | double | carrier hand-off - shipping limit (>0 = late) |
| carrier_transit_days | double | carrier -> customer |
| delivery_days | double | purchase -> customer delivery |
| delay_days | int | delivered date - estimated date (calendar days) |
| is_delivered | bool | status delivered and delivery timestamp present |
| is_late | bool | delivered date > estimated date; NULL if not delivered |
| is_handoff_late | bool | carrier_ts > shipping_limit_ts; NULL if no/invalid carrier timestamp |
| is_canceled | bool | status canceled or unavailable |
| review_score | int | latest review for the order (1-5) |

## `seller_summary` / `seller_scores.parquet` (seller)
| Column | Definition |
|---|---|
| n_items, n_orders, n_delivered | volume counts |
| n_late, n_handoff_late, n_handoff_obs, n_canceled | event counts |
| gmv, freight, late_gmv, canceled_gmv | sums of item price / freight |
| late_rate_raw | n_late / n_delivered |
| late_rate_shrunk | EB posterior mean (a + n_late) / (a + b + n_delivered) |
| late_rate_lo90 / hi90 | 90% Beta posterior credible interval |
| seller_handoff_late_rate | EB-shrunk hand-off late rate |
| cancel_rate | EB-shrunk cancellation rate |
| review_score_shrunk | (Σ scores + 5 x population mean) / (n_reviews + 5) |
| review_risk | 1 - review_score_shrunk / 5 |
| freight_ratio | freight_to_price capped at 2 |
| volume_uncertainty | 1 / (1 + log(1 + n_items)) |
| risk_score | 100 x Σ w_c x percentile(component c); weights in `config.yaml` |
| risk_score_lo90 / hi90 | 5th / 95th percentile over 200 Bayesian-bootstrap draws |
| risk_tier | High (top 10% of score), Elevated (80-90%), Moderate (50-80%), Low (<50%) |
| risk_score_{late_heavy, handoff_heavy, balanced_alt} | sensitivity weightings |
| archetype, cluster, pc1, pc2 | k-means archetype (sellers >= 20 items) and PCA coordinates |
| gmv_at_risk | gmv x late_rate_shrunk |
| wc_in_transit | gmv x median_delivery_days / 365 (working-capital proxy) |
| wc_excess_delay | late_gmv x mean_delay_days_when_late / 365 |
| late_cost_ASSUMED | gmv_at_risk x late_cost_pct_of_gmv |
| freight_exposure_ASSUMED | freight x freight_absorb_pct |
| cancel_cost_ASSUMED | gmv x cancel_rate x cancel_cost_pct_of_gmv |
| total_exposure_ASSUMED | sum of the three ASSUMED costs |

## `seller_month_panel` (seller x month)
late_rate, late_rate_roll3 (3-month `RANGE` window), gmv_roll3, cum_items (running total), prev_month_gmv (`LAG`),
month_index (`ROW_NUMBER`), gmv_rank_in_month (`RANK`), gmv_share_in_month.

## `oot_features` / `oot_outcomes` (seller, per split)
Features as in `seller_summary` but restricted to purchases before `score_end` and outcomes observed before
`score_end`; `max_delivered_ts_used` is asserted < `score_end` in tests. Outcomes: `next_n_items`, `next_n_delivered`,
`next_n_late`, `next_late_gmv`, `next_gmv`, `next_n_handoff_obs`, `next_n_handoff_late` for purchases in
[`score_end`, `validate_end`).
