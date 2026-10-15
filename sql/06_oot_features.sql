-- 06 Point-in-time (out-of-time) vendor features and next-period outcomes.  params(score_end, validate_end).
-- Features use ONLY information observable before score_end:
--   * lines scheduled before score_end AND whose outcome was already known (known_date = GREATEST(delivered, recorded)
--     before score_end); lines still unresolved at score_end are excluded
-- Outcomes are the vendor's lines SCHEDULED in [score_end, validate_end).  External vendors only (RDC excluded).
CREATE OR REPLACE TABLE oot_features AS
SELECT f.vendor,
       COUNT(*)                                                                    AS n_lines,
       COUNT(DISTINCT f.shipment_id)                                               AS n_shipments,
       COUNT(DISTINCT f.country)                                                   AS n_countries,
       CAST(SUM(f.line_value) AS DOUBLE)                                           AS value_usd,
       SUM(f.is_late::INT)                                                         AS n_late,
       SUM(f.is_severe_late::INT)                                                  AS n_severe_late,
       CAST(SUM(CASE WHEN f.is_late THEN f.line_value ELSE 0 END) AS DOUBLE)       AS late_value_usd,
       CAST(SUM(f.days_late_capped) AS DOUBLE)                                     AS sum_days_late_capped,
       CAST(AVG(CASE WHEN f.is_late THEN f.days_late_capped END) AS DOUBLE)        AS mean_days_when_late,
       CAST(SUM(f.freight_obs_usd) AS DOUBLE)                                      AS freight_usd,
       CAST(SUM(f.freight_obs_value) AS DOUBLE)                                    AS freight_base_value_usd,
       CAST(AVG(f.is_air::INT) AS DOUBLE)                                          AS air_share,
       MAX(f.known_date)                                                           AS max_known_date_used,
       MAX(f.scheduled_date)                                                       AS max_scheduled_used
FROM fct_lines f CROSS JOIN params p
WHERE NOT f.is_internal_rdc AND f.scheduled_date < p.score_end AND f.known_date < p.score_end
GROUP BY f.vendor;

CREATE OR REPLACE TABLE oot_vendor_lane AS
SELECT f.vendor, f.country, f.ship_mode, COUNT(*) AS n_lines, SUM(f.is_late::INT) AS n_late
FROM fct_lines f CROSS JOIN params p
WHERE NOT f.is_internal_rdc AND f.scheduled_date < p.score_end AND f.known_date < p.score_end
GROUP BY f.vendor, f.country, f.ship_mode;

CREATE OR REPLACE TABLE oot_outcomes AS
SELECT f.vendor,
       COUNT(*)                                                               AS next_n_lines,
       SUM(f.is_late::INT)                                                    AS next_n_late,
       SUM(f.is_severe_late::INT)                                             AS next_n_severe,
       CAST(SUM(f.line_value) AS DOUBLE)                                      AS next_value,
       CAST(SUM(CASE WHEN f.is_late THEN f.line_value ELSE 0 END) AS DOUBLE)  AS next_late_value,
       CAST(SUM(f.days_late_capped) AS DOUBLE)                                AS next_sum_days_late
FROM fct_lines f CROSS JOIN params p
WHERE NOT f.is_internal_rdc AND f.scheduled_date >= p.score_end AND f.scheduled_date < p.validate_end
GROUP BY f.vendor;
