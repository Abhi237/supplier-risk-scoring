-- 05 Seller-month panel with window functions.  Grain: one row per (seller_id, purchase_month).
CREATE OR REPLACE TABLE seller_month_panel AS
WITH m AS (
  SELECT seller_id, purchase_month,
         COUNT(*) AS n_items, CAST(SUM(price) AS DOUBLE) AS gmv, CAST(SUM(freight_value) AS DOUBLE) AS freight,
         SUM(CASE WHEN is_delivered THEN 1 ELSE 0 END) AS n_delivered,
         SUM(CASE WHEN is_late THEN 1 ELSE 0 END) AS n_late,
         SUM(CASE WHEN is_handoff_late IS NOT NULL THEN 1 ELSE 0 END) AS n_handoff_obs,
         SUM(CASE WHEN is_handoff_late THEN 1 ELSE 0 END) AS n_handoff_late,
         SUM(CASE WHEN is_canceled THEN 1 ELSE 0 END) AS n_canceled,
         AVG(review_score) AS mean_review_score
  FROM fct_items GROUP BY seller_id, purchase_month
)
SELECT m.*,
       n_late::DOUBLE / NULLIF(n_delivered, 0) AS late_rate,
       -- rolling 3-month (current + 2 prior calendar months) late rate
       SUM(n_late) OVER w3::DOUBLE / NULLIF(SUM(n_delivered) OVER w3, 0) AS late_rate_roll3,
       SUM(gmv) OVER w3                                               AS gmv_roll3,
       -- cumulative to date
       SUM(n_items) OVER (PARTITION BY seller_id ORDER BY purchase_month
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_items,
       LAG(gmv) OVER (PARTITION BY seller_id ORDER BY purchase_month)         AS prev_month_gmv,
       ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY purchase_month)     AS month_index,
       RANK() OVER (PARTITION BY purchase_month ORDER BY gmv DESC)            AS gmv_rank_in_month,
       gmv / SUM(gmv) OVER (PARTITION BY purchase_month)                      AS gmv_share_in_month
FROM m
WINDOW w3 AS (PARTITION BY seller_id ORDER BY purchase_month
              RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW);

-- platform-month KPIs
CREATE OR REPLACE TABLE platform_month AS
SELECT purchase_month, COUNT(*) AS n_items, COUNT(DISTINCT seller_id) AS n_active_sellers, CAST(SUM(price) AS DOUBLE) AS gmv,
       AVG(CASE WHEN is_late IS NOT NULL THEN is_late::INT END) AS late_rate,
       AVG(CASE WHEN is_handoff_late IS NOT NULL THEN is_handoff_late::INT END) AS handoff_late_rate,
       AVG(is_canceled::INT) AS cancel_rate, AVG(review_score) AS mean_review_score,
       MEDIAN(delivery_days) AS median_delivery_days, CAST(SUM(freight_value) / SUM(price) AS DOUBLE) AS freight_to_price
FROM fct_items GROUP BY purchase_month ORDER BY purchase_month;
