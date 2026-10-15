-- 05 Vendor x scheduled-year panel with window functions (rolling 3-year late rate, LAG, running totals, rank),
-- plus the programme-level year table (all lines / external vendors / SCMS regional distribution centre).
CREATE OR REPLACE TABLE vendor_year_panel AS
WITH base AS (
  SELECT vendor, sched_year, COUNT(*) AS n_lines, SUM(is_late::INT) AS n_late, SUM(is_severe_late::INT) AS n_severe,
         CAST(SUM(line_value) AS DOUBLE) AS value_usd
  FROM fct_lines WHERE NOT is_internal_rdc GROUP BY vendor, sched_year)
SELECT vendor, sched_year, n_lines, n_late, n_severe, value_usd,
       n_late::DOUBLE / n_lines                                                         AS late_rate,
       SUM(n_late) OVER w3::DOUBLE / NULLIF(SUM(n_lines) OVER w3, 0)                    AS late_rate_roll3,
       LAG(n_late::DOUBLE / n_lines) OVER (PARTITION BY vendor ORDER BY sched_year)    AS late_rate_prev_year,
       SUM(n_lines) OVER (PARTITION BY vendor ORDER BY sched_year)                      AS cum_lines,
       RANK() OVER (PARTITION BY sched_year ORDER BY n_lines DESC, vendor)              AS volume_rank_in_year,
       ROW_NUMBER() OVER (PARTITION BY vendor ORDER BY sched_year) - 1                  AS year_index
FROM base
WINDOW w3 AS (PARTITION BY vendor ORDER BY sched_year RANGE BETWEEN 2 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE TABLE programme_year AS
SELECT sched_year,
       COUNT(*) AS n_lines, SUM(is_late::INT) AS n_late,
       SUM(is_exact_match::INT) AS n_exact_match,
       SUM(CASE WHEN NOT is_internal_rdc THEN 1 ELSE 0 END) AS n_lines_vendor,
       SUM(CASE WHEN NOT is_internal_rdc AND is_late THEN 1 ELSE 0 END) AS n_late_vendor,
       SUM(CASE WHEN NOT is_internal_rdc AND is_exact_match THEN 1 ELSE 0 END) AS n_exact_vendor,
       SUM(CASE WHEN is_internal_rdc THEN 1 ELSE 0 END) AS n_lines_rdc,
       SUM(CASE WHEN is_internal_rdc AND is_late THEN 1 ELSE 0 END) AS n_late_rdc,
       CAST(SUM(line_value) AS DOUBLE) AS value_usd
FROM fct_lines GROUP BY sched_year;
