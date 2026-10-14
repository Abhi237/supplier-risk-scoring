-- 06 Point-in-time (out-of-time) seller features and next-period outcomes.
-- params(score_end, validate_end).  Features use ONLY information observable before score_end:
--   * items purchased before score_end
--   * lateness counted only if the customer delivery happened before score_end (otherwise unresolved)
--   * hand-off counted only if the carrier hand-off happened before score_end
--   * reviews counted only if created before score_end
-- Outcomes for the seller's items purchased in [score_end, validate_end):
--   end-to-end lateness (delivered vs estimated) and seller-controlled hand-off lateness (carrier vs limit).
CREATE OR REPLACE TABLE oot_features AS
SELECT f.seller_id,
       ANY_VALUE(f.seller_state)                                                     AS seller_state,
       COUNT(*)                                                                      AS n_items,
       CAST(SUM(f.price::DECIMAL(18, 2)) AS DOUBLE)                                  AS gmv,
       CAST(SUM(f.freight_value::DECIMAL(18, 2)) / NULLIF(SUM(f.price::DECIMAL(18, 2)), 0) AS DOUBLE) AS freight_to_price,
       SUM(CASE WHEN f.is_delivered AND f.delivered_ts < p.score_end THEN 1 ELSE 0 END)       AS n_delivered,
       SUM(CASE WHEN f.is_late AND f.delivered_ts < p.score_end THEN 1 ELSE 0 END)            AS n_late,
       SUM(CASE WHEN f.is_handoff_late IS NOT NULL AND f.carrier_ts < p.score_end THEN 1 ELSE 0 END) AS n_handoff_obs,
       SUM(CASE WHEN f.is_handoff_late AND f.carrier_ts < p.score_end THEN 1 ELSE 0 END)      AS n_handoff_late,
       SUM(CASE WHEN f.is_canceled THEN 1 ELSE 0 END)                                AS n_canceled,
       AVG(CASE WHEN f.review_created_ts < p.score_end THEN f.review_score END)      AS mean_review_score,
       SUM(CASE WHEN f.review_created_ts < p.score_end THEN 1 ELSE 0 END)            AS n_reviews,
       MEDIAN(CASE WHEN f.delivered_ts < p.score_end THEN f.delivery_days END)       AS median_delivery_days,
       MAX(CASE WHEN f.delivered_ts < p.score_end THEN f.delivered_ts END)           AS max_delivered_ts_used,
       MAX(f.purchase_ts)                                                            AS max_purchase_ts_used
FROM fct_items f CROSS JOIN params p
WHERE f.purchase_ts < p.score_end
GROUP BY f.seller_id;

CREATE OR REPLACE TABLE oot_outcomes AS
SELECT f.seller_id,
       COUNT(*)                                     AS next_n_items,
       SUM(CASE WHEN f.is_delivered THEN 1 ELSE 0 END) AS next_n_delivered,
       SUM(CASE WHEN f.is_late THEN 1 ELSE 0 END)   AS next_n_late,
       CAST(SUM(CASE WHEN f.is_late THEN f.price::DECIMAL(18, 2) ELSE 0 END) AS DOUBLE) AS next_late_gmv,
       SUM(CASE WHEN f.is_handoff_late IS NOT NULL THEN 1 ELSE 0 END) AS next_n_handoff_obs,
       SUM(CASE WHEN f.is_handoff_late THEN 1 ELSE 0 END) AS next_n_handoff_late,
       CAST(SUM(f.price::DECIMAL(18, 2)) AS DOUBLE) AS next_gmv
FROM fct_items f CROSS JOIN params p
WHERE f.purchase_ts >= p.score_end AND f.purchase_ts < p.validate_end
GROUP BY f.seller_id;
