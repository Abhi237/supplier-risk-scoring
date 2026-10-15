-- 02 Cleaning log + data-quality report.  cleaning_log: one row per cleaning/definition action with the number of rows
-- affected (counts come from the data, nothing is typed by hand).  dq_report: one row per check (n, total, detail).
CREATE OR REPLACE TABLE cleaning_log AS
WITH s AS (SELECT * FROM stg_lines)
SELECT 1.0 AS ord, 'rows_in_raw_file' AS action, COUNT(*) AS n_rows, COUNT(*) AS base, 'raw rows read' AS detail FROM s
UNION ALL SELECT 2, 'scheduled_date_unparseable', COUNT(*), (SELECT COUNT(*) FROM s),
       'scheduled delivery date not a date (e.g. Date Not Captured): would be excluded from on-time' FROM s WHERE scheduled_date IS NULL
UNION ALL SELECT 3, 'delivered_date_unparseable', COUNT(*), (SELECT COUNT(*) FROM s),
       'delivered-to-client date not a date: would be excluded from on-time' FROM s WHERE delivered_date IS NULL
UNION ALL SELECT 4, 'recorded_date_unparseable', COUNT(*), (SELECT COUNT(*) FROM s),
       'delivery recorded date not a date' FROM s WHERE recorded_date IS NULL
UNION ALL SELECT 5, 'po_sent_date_not_captured', COUNT(*), (SELECT COUNT(*) FROM s),
       'procurement date text "Date Not Captured" (kept as status; not used for on-time)' FROM s WHERE po_sent_raw = 'Date Not Captured'
UNION ALL SELECT 6, 'po_sent_date_na_from_rdc', COUNT(*), (SELECT COUNT(*) FROM s),
       'PO date "N/A - From RDC": line filled from SCMS regional stock, no PO to a vendor' FROM s WHERE po_sent_raw LIKE 'N/A%'
UNION ALL SELECT 7, 'pq_date_pre_pq_process', COUNT(*), (SELECT COUNT(*) FROM s),
       'PQ date text "Pre-PQ Process" or "Date Not Captured" (not used)' FROM s WHERE pq_sent_raw IN ('Pre-PQ Process', 'Date Not Captured')
UNION ALL SELECT 8, 'ship_mode_na_to_unknown', COUNT(*), (SELECT COUNT(*) FROM s),
       'shipment mode "N/A" relabelled Unknown' FROM s WHERE ship_mode = 'Unknown'
UNION ALL SELECT 9, 'freight_numeric', COUNT(*), (SELECT COUNT(*) FROM s), 'freight is a number' FROM s WHERE freight_kind = 'numeric'
UNION ALL SELECT 10, 'freight_included_in_commodity_cost', COUNT(*), (SELECT COUNT(*) FROM s),
       'text: Freight Included in Commodity Cost -> freight unknown (NULL), excluded from freight ratio' FROM s WHERE freight_kind = 'included_in_commodity_cost'
UNION ALL SELECT 11, 'freight_invoiced_separately', COUNT(*), (SELECT COUNT(*) FROM s),
       'text: Invoiced Separately -> freight unknown (NULL)' FROM s WHERE freight_kind = 'invoiced_separately'
UNION ALL SELECT 12, 'freight_see_reference', COUNT(*), (SELECT COUNT(*) FROM s),
       'text: See ASN/DN-... (ID#:n) -> freight is carried by the shipment first line; resolved at shipment grain' FROM s WHERE freight_kind = 'see_reference'
UNION ALL SELECT 13, 'freight_reference_resolves_to_numeric_first_line', COUNT(*), (SELECT COUNT(*) FROM s WHERE freight_kind = 'see_reference'),
       'see-reference lines whose referenced line is in the same shipment and carries numeric freight'
       FROM s JOIN s r ON r.line_id = s.freight_ref_line_id
       WHERE s.freight_kind = 'see_reference' AND r.shipment_id = s.shipment_id AND r.freight_kind = 'numeric'
UNION ALL SELECT 14, 'freight_negative_set_null', COUNT(*), (SELECT COUNT(*) FROM s WHERE freight_kind = 'numeric'),
       'negative numeric freight set to NULL' FROM s WHERE freight_kind = 'numeric' AND freight_num_raw < 0
UNION ALL SELECT 15, 'freight_zero_numeric_kept', COUNT(*), (SELECT COUNT(*) FROM s WHERE freight_kind = 'numeric'),
       'numeric freight exactly 0 (kept; ratio 0)' FROM s WHERE freight_kind = 'numeric' AND freight_num_raw = 0
UNION ALL SELECT 16, 'weight_numeric', COUNT(*), (SELECT COUNT(*) FROM s), 'weight is a number' FROM s WHERE weight_kind = 'numeric'
UNION ALL SELECT 17, 'weight_captured_separately', COUNT(*), (SELECT COUNT(*) FROM s),
       'text: Weight Captured Separately -> NULL' FROM s WHERE weight_kind = 'captured_separately'
UNION ALL SELECT 18, 'weight_see_reference', COUNT(*), (SELECT COUNT(*) FROM s),
       'text: See ASN/DN-... -> NULL at line level (carried by shipment first line)' FROM s WHERE weight_kind = 'see_reference'
UNION ALL SELECT 19, 'weight_negative_set_null', COUNT(*), (SELECT COUNT(*) FROM s WHERE weight_kind = 'numeric'),
       'negative weight set to NULL' FROM s WHERE weight_kind = 'numeric' AND weight_num_raw < 0
UNION ALL SELECT 20, 'weight_zero_set_null', COUNT(*), (SELECT COUNT(*) FROM s WHERE weight_kind = 'numeric'),
       'zero kilograms is not physical -> NULL' FROM s WHERE weight_kind = 'numeric' AND weight_num_raw = 0
UNION ALL SELECT 21, 'line_value_negative', COUNT(*), (SELECT COUNT(*) FROM s), 'negative line item value' FROM s WHERE line_value_raw < 0
UNION ALL SELECT 22, 'line_value_zero_kept', COUNT(*), (SELECT COUNT(*) FROM s),
       'zero line value kept in counts, excluded from freight / value ratios' FROM s WHERE line_value_raw = 0
UNION ALL SELECT 23, 'duplicate_rows_ignoring_id', (SELECT COUNT(*) FROM (SELECT 1 FROM (SELECT * EXCLUDE ("ID") FROM raw_scms) GROUP BY ALL HAVING COUNT(*) > 1)),
       (SELECT COUNT(*) FROM s), 'groups of rows identical in every column except ID (kept: distinct IDs, plausibly distinct lines)'
UNION ALL SELECT 24, 'recorded_before_delivered', COUNT(*), (SELECT COUNT(*) FROM s),
       'delivery recorded date earlier than delivered date (kept; recorded lag floored at 0 for point-in-time availability)'
       FROM s WHERE recorded_date < delivered_date
UNION ALL SELECT 25, 'delivered_equals_scheduled', COUNT(*), (SELECT COUNT(*) FROM s),
       'delivered date equals scheduled date exactly (on time by definition; a large share suggests scheduled dates were back-filled)'
       FROM s WHERE delivered_date = scheduled_date
UNION ALL SELECT 26, 'delivered_more_than_early_threshold_before_schedule', COUNT(*), (SELECT COUNT(*) FROM s),
       'delivered more than extreme_early_days before the scheduled date (flag only; on time either way)'
       FROM s, params p WHERE DATE_DIFF('day', scheduled_date, delivered_date) < -p.early_days
UNION ALL SELECT 27, 'days_late_capped', COUNT(*), (SELECT COUNT(*) FROM s),
       'days late above the cap, capped for mean-lateness metrics only (is_late flag unchanged)'
       FROM s, params p WHERE DATE_DIFF('day', scheduled_date, delivered_date) > p.cap_days
UNION ALL SELECT 27.5, 'freight_ratio_capped', COUNT(*), (SELECT COUNT(*) FROM s WHERE is_first_line AND freight_kind = 'numeric'),
       'shipments whose numeric freight exceeds freight_ratio_cap x shipment line value; freight capped at that multiple'
       FROM (SELECT sh.shipment_id FROM (SELECT shipment_id, SUM(line_value_raw) v, MAX(CASE WHEN is_first_line AND freight_kind = 'numeric' THEN freight_num_raw END) f
                                         FROM s GROUP BY shipment_id) sh, params p WHERE sh.f > p.freight_ratio_cap * sh.v AND sh.v > 0)
UNION ALL SELECT 28, 'shipments_with_two_scheduled_dates', (SELECT COUNT(*) FROM (SELECT shipment_id FROM s GROUP BY 1 HAVING COUNT(DISTINCT scheduled_date) > 1)),
       (SELECT COUNT(DISTINCT shipment_id) FROM s), 'shipments whose lines carry different scheduled dates (line dates used as given)'
UNION ALL SELECT 29, 'vendor_names_distinct_as_recorded', COUNT(DISTINCT vendor), COUNT(DISTINCT vendor),
       'distinct vendor strings; spelling variants (e.g. Abbott / AbbVie entities) are NOT merged' FROM s;

CREATE OR REPLACE TABLE dq_report AS
SELECT 'lines_rows' AS check_name, COUNT(*) AS n, COUNT(*) AS total, 'raw order-line rows' AS detail FROM stg_lines
UNION ALL SELECT 'duplicate_line_id', COUNT(*) - COUNT(DISTINCT line_id), COUNT(*), 'duplicate ID keys' FROM stg_lines
UNION ALL SELECT 'null_vendor', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'vendor blank' FROM stg_lines WHERE vendor IS NULL OR vendor = ''
UNION ALL SELECT 'null_country', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'country blank' FROM stg_lines WHERE country IS NULL OR country = ''
UNION ALL SELECT 'missing_delivery_dates', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'scheduled or delivered date unusable' FROM stg_lines WHERE scheduled_date IS NULL OR delivered_date IS NULL
UNION ALL SELECT 'recorded_before_delivered', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'recorded date precedes delivered date' FROM stg_lines WHERE recorded_date < delivered_date
UNION ALL SELECT 'delivered_after_extract_date', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'delivered date after 2015-09-29 (file date)' FROM stg_lines WHERE delivered_date > DATE '2015-09-29'
UNION ALL SELECT 'scheduled_after_extract_date', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'scheduled date after 2015-09-29 yet delivered (early)' FROM stg_lines WHERE scheduled_date > DATE '2015-09-29'
UNION ALL SELECT 'nonpositive_line_value', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'line value <= 0' FROM stg_lines WHERE line_value_raw <= 0
UNION ALL SELECT 'freight_text_not_numeric', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'freight is text (included / separate / reference)' FROM stg_lines WHERE freight_kind <> 'numeric'
UNION ALL SELECT 'weight_text_not_numeric', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'weight is text' FROM stg_lines WHERE weight_kind <> 'numeric'
UNION ALL SELECT 'shipments_without_first_line', (SELECT COUNT(*) FROM (SELECT shipment_id FROM stg_lines GROUP BY 1 HAVING SUM(is_first_line::INT) = 0)),
       (SELECT COUNT(DISTINCT shipment_id) FROM stg_lines), 'shipments with no first-line designation'
UNION ALL SELECT 'shipments_multiple_first_lines', (SELECT COUNT(*) FROM (SELECT shipment_id FROM stg_lines GROUP BY 1 HAVING SUM(is_first_line::INT) > 1)),
       (SELECT COUNT(DISTINCT shipment_id) FROM stg_lines), 'shipments with more than one first line'
UNION ALL SELECT 'delivered_equals_scheduled', COUNT(*), (SELECT COUNT(*) FROM stg_lines), 'delivered date = scheduled date exactly' FROM stg_lines WHERE delivered_date = scheduled_date;
