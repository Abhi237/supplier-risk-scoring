-- 04 Seller (supplier-proxy) summary over the full observation window.  Grain: one row per seller_id.
-- Money sums use DECIMAL so results are bit-for-bit reproducible regardless of parallel aggregation order.
CREATE OR REPLACE TABLE seller_summary AS
WITH cat_rank AS (   -- deterministic modal category (ties broken alphabetically)
  SELECT seller_id, category,
         ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY COUNT(*) DESC, category) AS rn
  FROM fct_items GROUP BY seller_id, category
)
SELECT f.seller_id,
       ANY_VALUE(seller_state)                                        AS seller_state,
       COUNT(*)                                                       AS n_items,
       COUNT(DISTINCT order_id)                                       AS n_orders,
       CAST(SUM(price::DECIMAL(18, 2)) AS DOUBLE)                     AS gmv,
       CAST(SUM(freight_value::DECIMAL(18, 2)) AS DOUBLE)             AS freight,
       CAST(SUM(freight_value::DECIMAL(18, 2)) / NULLIF(SUM(price::DECIMAL(18, 2)), 0) AS DOUBLE) AS freight_to_price,
       SUM(CASE WHEN is_delivered THEN 1 ELSE 0 END)                  AS n_delivered,
       SUM(CASE WHEN is_late THEN 1 ELSE 0 END)                       AS n_late,
       CAST(SUM(CASE WHEN is_late THEN price::DECIMAL(18, 2) ELSE 0 END) AS DOUBLE) AS late_gmv,
       SUM(CASE WHEN is_handoff_late IS NOT NULL THEN 1 ELSE 0 END)   AS n_handoff_obs,
       SUM(CASE WHEN is_handoff_late THEN 1 ELSE 0 END)               AS n_handoff_late,
       SUM(CASE WHEN is_canceled THEN 1 ELSE 0 END)                   AS n_canceled,
       CAST(SUM(CASE WHEN is_canceled THEN price::DECIMAL(18, 2) ELSE 0 END) AS DOUBLE) AS canceled_gmv,
       AVG(review_score)                                              AS mean_review_score,
       SUM(CASE WHEN review_score IS NOT NULL THEN 1 ELSE 0 END)      AS n_reviews,
       AVG(CASE WHEN review_score <= 2 THEN 1.0 WHEN review_score IS NOT NULL THEN 0.0 END) AS low_review_share,
       MEDIAN(delivery_days)                                          AS median_delivery_days,
       MEDIAN(handoff_days)                                           AS median_handoff_days,
       AVG(CASE WHEN is_late THEN delay_days END)                     AS mean_delay_days_when_late,
       COUNT(DISTINCT f.category)                                       AS n_categories,
       ANY_VALUE(c.category)                                          AS top_category,
       AVG(CASE WHEN same_state THEN 1.0 ELSE 0.0 END)                AS same_state_share,
       MIN(purchase_ts)                                               AS first_sale_ts,
       MAX(purchase_ts)                                               AS last_sale_ts
FROM fct_items f
LEFT JOIN cat_rank c ON c.seller_id = f.seller_id AND c.rn = 1
GROUP BY f.seller_id;
