-- 04 Vendor summary (full period).  Grain: one row per vendor string.  The internal SCMS regional distribution centre
-- (is_internal_rdc) is kept as a reference row but is NOT part of the scored external-vendor population.
CREATE OR REPLACE TABLE vendor_lane AS
SELECT vendor, country, ship_mode, COUNT(*) AS n_lines, SUM(is_late::INT) AS n_late
FROM fct_lines WHERE NOT is_internal_rdc
GROUP BY vendor, country, ship_mode;

CREATE OR REPLACE TABLE vendor_summary AS
WITH hhi AS (
  SELECT vendor, SUM(POWER(n::DOUBLE / tot, 2)) AS country_hhi
  FROM (SELECT vendor, country, COUNT(*) AS n, SUM(COUNT(*)) OVER (PARTITION BY vendor) AS tot
        FROM fct_lines GROUP BY vendor, country) GROUP BY vendor),
top_c AS (
  SELECT vendor, country AS top_country FROM (
    SELECT vendor, country, ROW_NUMBER() OVER (PARTITION BY vendor ORDER BY COUNT(*) DESC, country) AS rn
    FROM fct_lines GROUP BY vendor, country) WHERE rn = 1),
top_g AS (
  SELECT vendor, product_group AS top_product_group FROM (
    SELECT vendor, product_group, ROW_NUMBER() OVER (PARTITION BY vendor ORDER BY COUNT(*) DESC, product_group) AS rn
    FROM fct_lines GROUP BY vendor, product_group) WHERE rn = 1)
SELECT f.vendor,
       BOOL_OR(f.is_internal_rdc)                                                   AS is_internal_rdc,
       COUNT(*)                                                                     AS n_lines,
       COUNT(DISTINCT f.shipment_id)                                                AS n_shipments,
       COUNT(DISTINCT f.country)                                                    AS n_countries,
       CAST(SUM(f.line_value) AS DOUBLE)                                            AS value_usd,
       SUM(f.is_late::INT)                                                          AS n_late,
       SUM(f.is_severe_late::INT)                                                   AS n_severe_late,
       SUM(f.is_exact_match::INT)                                                   AS n_exact_match,
       CAST(SUM(CASE WHEN f.is_late THEN f.line_value ELSE 0 END) AS DOUBLE)        AS late_value_usd,
       CAST(SUM(f.days_late_capped) AS DOUBLE)                                      AS sum_days_late_capped,
       CAST(AVG(CASE WHEN f.is_late THEN f.days_late_capped END) AS DOUBLE)         AS mean_days_when_late,
       CAST(MEDIAN(f.offset_days) AS DOUBLE)                                        AS median_offset_days,
       CAST(SUM(f.freight_obs_usd) AS DOUBLE)                                       AS freight_usd,
       CAST(SUM(f.freight_obs_value) AS DOUBLE)                                     AS freight_base_value_usd,
       COUNT(f.freight_obs_usd)                                                     AS n_shipments_freight_obs,
       SUM(f.is_air::INT)                                                           AS n_air_lines,
       CAST(AVG(f.is_air::INT) AS DOUBLE)                                           AS air_share,
       MIN(f.scheduled_date)                                                        AS first_scheduled,
       MAX(f.scheduled_date)                                                        AS last_scheduled,
       ANY_VALUE(c.top_country)                                                     AS top_country,
       ANY_VALUE(g.top_product_group)                                               AS top_product_group,
       CAST(ANY_VALUE(h.country_hhi) AS DOUBLE)                                     AS country_hhi
FROM fct_lines f
JOIN hhi h USING (vendor) JOIN top_c c USING (vendor) JOIN top_g g USING (vendor)
GROUP BY f.vendor;
