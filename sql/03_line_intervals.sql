-- 03 Line-grain fact with delivery intervals.  Grain: one row per order line (line_id).
-- ON-TIME DEFINITION: a line is on time when delivered_to_client_date <= scheduled_delivery_date; lateness days are
-- delivered - scheduled (positive part).  The dataset has NO separate vendor hand-off timestamp, so the vendor-controlled
-- vs end-to-end split is not observable: lateness here mixes vendor, freight forwarder and in-country effects.
-- known_date = when the outcome was observable in the system: GREATEST(delivered date, delivery recorded date).
CREATE OR REPLACE TABLE fct_shipments AS
SELECT l.shipment_id,
       ANY_VALUE(l.vendor)                                           AS vendor,
       ANY_VALUE(l.country)                                          AS country,
       ANY_VALUE(l.ship_mode)                                        AS ship_mode,
       COUNT(*)                                                      AS n_lines,
       MIN(l.line_id)                                                AS min_line_id,
       CAST(SUM(l.line_value_raw) AS DOUBLE)                         AS ship_value,
       MAX(CASE WHEN l.is_first_line THEN l.line_id END)             AS first_line_id,
       MAX(CASE WHEN l.is_first_line THEN l.freight_kind END)        AS first_line_freight_kind,
       MAX(CASE WHEN l.is_first_line AND l.freight_kind = 'numeric' AND l.freight_num_raw >= 0 THEN l.freight_num_raw END) AS ship_freight_usd,
       MAX(CASE WHEN l.is_first_line AND l.weight_kind = 'numeric' AND l.weight_num_raw > 0 THEN l.weight_num_raw END)    AS ship_weight_kg,
       BOOL_OR(l.freight_kind = 'numeric' AND l.freight_num_raw >= 0 AND l.is_first_line) AS has_numeric_freight
FROM stg_lines l
GROUP BY l.shipment_id;

CREATE OR REPLACE TABLE fct_lines AS
SELECT l.line_id, l.shipment_id, l.vendor, (l.vendor = p.internal_vendor) AS is_internal_rdc,
       l.country, l.ship_mode, l.product_group, l.fulfill_via, l.inco_term, l.managed_by, l.is_first_line,
       l.scheduled_date, l.delivered_date, l.recorded_date,
       GREATEST(l.delivered_date, l.recorded_date)                                   AS known_date,
       EXTRACT(year FROM l.scheduled_date)::INTEGER                                  AS sched_year,
       l.line_qty,
       CASE WHEN l.line_value_raw >= 0 THEN l.line_value_raw END                     AS line_value,
       l.freight_kind,
       s.n_lines                                                                     AS ship_n_lines,
       s.ship_value,
       CASE WHEN s.has_numeric_freight THEN s.ship_freight_usd END                   AS ship_freight_usd,
       CASE WHEN s.has_numeric_freight AND s.ship_value > 0 THEN LEAST(s.ship_freight_usd / s.ship_value, p.freight_ratio_cap) END AS ship_freight_ratio,
       -- freight observation carried once per shipment (on its first line) for exact ratio aggregation
       CASE WHEN l.is_first_line AND s.has_numeric_freight AND s.ship_value > 0
            THEN LEAST(s.ship_freight_usd, p.freight_ratio_cap * s.ship_value) END                          AS freight_obs_usd,
       CASE WHEN l.is_first_line AND s.has_numeric_freight AND s.ship_value > 0 THEN s.ship_value END       AS freight_obs_value,
       s.ship_weight_kg,
       DATE_DIFF('day', l.scheduled_date, l.delivered_date)                          AS offset_days,
       (l.delivered_date > l.scheduled_date)                                         AS is_late,
       GREATEST(DATE_DIFF('day', l.scheduled_date, l.delivered_date), 0)             AS days_late,
       LEAST(GREATEST(DATE_DIFF('day', l.scheduled_date, l.delivered_date), 0), p.cap_days) AS days_late_capped,
       (DATE_DIFF('day', l.scheduled_date, l.delivered_date) > p.severe_days)        AS is_severe_late,
       (l.delivered_date = l.scheduled_date)                                         AS is_exact_match,
       (DATE_DIFF('day', l.scheduled_date, l.delivered_date) < -p.early_days)        AS is_extreme_early,
       GREATEST(DATE_DIFF('day', l.delivered_date, l.recorded_date), 0)              AS record_lag_days,
       (l.ship_mode IN ('Air', 'Air Charter'))                                       AS is_air
FROM stg_lines l
JOIN fct_shipments s USING (shipment_id)
CROSS JOIN params p
WHERE l.scheduled_date IS NOT NULL AND l.delivered_date IS NOT NULL;
