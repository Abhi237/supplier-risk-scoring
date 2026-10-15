-- 01 Staging: typed, renamed copy of the raw SCMS Delivery History table (view raw_scms; every column is text).
-- Dataset: USAID / PEPFAR Supply Chain Management System (SCMS) delivery history, ~10.3k order-line rows.
-- Buyer = a US government health-supply programme; deliveries go to PEPFAR-supported countries.
-- Cleaning rules applied here (every action is counted in 02_cleaning_log.sql):
--   * delivery dates (d-Mon-yy) parsed with TRY_STRPTIME; unparseable strings become NULL (never guessed)
--   * 'Date Not Captured' / 'Pre-PQ Process' / 'N/A - From RDC' in the procurement dates are kept as a status
--   * freight and weight text values are typed into numeric + kind + reference columns; text is never coerced to 0
--   * negative or zero physical quantities become NULL; zero line values are kept but excluded from ratios
CREATE OR REPLACE TABLE stg_lines AS
SELECT
  CAST("ID" AS INTEGER)                                         AS line_id,
  "Project Code"                                                AS project_code,
  "ASN/DN #"                                                    AS shipment_id,
  TRIM("Country")                                               AS country,
  "Managed By"                                                  AS managed_by,
  "Fulfill Via"                                                 AS fulfill_via,
  "Vendor INCO Term"                                            AS inco_term,
  CASE WHEN TRIM("Shipment Mode") IN ('', 'N/A') THEN 'Unknown' ELSE TRIM("Shipment Mode") END AS ship_mode,
  "Shipment Mode"                                               AS ship_mode_raw,
  "Product Group"                                               AS product_group,
  TRIM("Vendor")                                                AS vendor,
  ("First Line Designation" = 'Yes')                            AS is_first_line,
  COALESCE(TRY_STRPTIME("Scheduled Delivery Date", '%d-%b-%y'), TRY_STRPTIME("Scheduled Delivery Date", '%d-%b-%Y'))::DATE
                                                                AS scheduled_date,
  COALESCE(TRY_STRPTIME("Delivered to Client Date", '%d-%b-%y'), TRY_STRPTIME("Delivered to Client Date", '%d-%b-%Y'))::DATE
                                                                AS delivered_date,
  COALESCE(TRY_STRPTIME("Delivery Recorded Date", '%d-%b-%y'), TRY_STRPTIME("Delivery Recorded Date", '%d-%b-%Y'))::DATE
                                                                AS recorded_date,
  "Scheduled Delivery Date"                                     AS scheduled_raw,
  "Delivered to Client Date"                                    AS delivered_raw,
  "Delivery Recorded Date"                                      AS recorded_raw,
  COALESCE(TRY_STRPTIME("PO Sent to Vendor Date", '%m/%d/%Y'), TRY_STRPTIME("PO Sent to Vendor Date", '%m/%d/%y'))::DATE
                                                                AS po_sent_date,
  "PO Sent to Vendor Date"                                      AS po_sent_raw,
  "PQ First Sent to Client Date"                                AS pq_sent_raw,
  TRY_CAST("Line Item Quantity" AS DOUBLE)                      AS line_qty,
  TRY_CAST("Line Item Value" AS DOUBLE)                         AS line_value_raw,
  TRY_CAST("Unit Price" AS DOUBLE)                              AS unit_price,
  TRY_CAST("Line Item Insurance (USD)" AS DOUBLE)               AS insurance_usd,
  -- freight: numeric / included in commodity cost / invoiced separately / reference to another line / other
  CASE WHEN regexp_matches("Freight Cost (USD)", '^-?[0-9]+(\.[0-9]+)?$') THEN 'numeric'
       WHEN "Freight Cost (USD)" = 'Freight Included in Commodity Cost' THEN 'included_in_commodity_cost'
       WHEN "Freight Cost (USD)" = 'Invoiced Separately' THEN 'invoiced_separately'
       WHEN "Freight Cost (USD)" LIKE 'See %' THEN 'see_reference'
       WHEN TRIM("Freight Cost (USD)") = '' THEN 'blank'
       ELSE 'other' END                                         AS freight_kind,
  TRY_CAST("Freight Cost (USD)" AS DOUBLE)                      AS freight_num_raw,
  TRY_CAST(NULLIF(regexp_extract("Freight Cost (USD)", 'ID#:([0-9]+)', 1), '') AS INTEGER) AS freight_ref_line_id,
  CASE WHEN regexp_matches("Weight (Kilograms)", '^-?[0-9]+(\.[0-9]+)?$') THEN 'numeric'
       WHEN "Weight (Kilograms)" = 'Weight Captured Separately' THEN 'captured_separately'
       WHEN "Weight (Kilograms)" LIKE 'See %' THEN 'see_reference'
       WHEN TRIM("Weight (Kilograms)") = '' THEN 'blank'
       ELSE 'other' END                                         AS weight_kind,
  TRY_CAST("Weight (Kilograms)" AS DOUBLE)                      AS weight_num_raw
FROM raw_scms;
