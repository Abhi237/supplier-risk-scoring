# Data dictionary

Source: USAID / PEPFAR SCMS Delivery History (file date 2015-09-29), 10,324 order-line rows, 33 columns. See
[../data/README.md](../data/README.md) for provenance and the licence caveat. The buyer is a US government health-supply
programme; deliveries go to PEPFAR-supported countries. Nothing here is a US-domestic delivery.

## Raw columns used

| Raw column | Staged name | Meaning and handling |
|---|---|---|
| `ID` | `line_id` | Unique line id (checked unique). |
| `ASN/DN #` | `shipment_id` | Advance shipping notice / delivery note. Lines of one shipment share it. `DN-` = from the regional distribution centre, `ASN-` = vendor shipment. |
| `Country` | `country` | Destination country (43). |
| `Fulfill Via` | `fulfill_via` | `Direct Drop` or `From RDC`. |
| `Vendor INCO Term` | `inco_term` | Incoterm (EXW, DDP, ...). |
| `Shipment Mode` | `ship_mode` | Air, Truck, Ocean, Air Charter; `N/A` becomes `Unknown` (logged). |
| `Product Group` | `product_group` | ARV, HRDT, ANTM, ACT, MRDT. |
| `Vendor` | `vendor` | Supplier name as recorded (73 strings; not merged). `SCMS from RDC` is the programme's own regional distribution centre, not an external vendor. |
| `Scheduled Delivery Date` | `scheduled_date` | `d-Mon-yy`, parsed with `TRY_STRPTIME`; never guessed. |
| `Delivered to Client Date` | `delivered_date` | Same format. |
| `Delivery Recorded Date` | `recorded_date` | Date the delivery was recorded; used for point-in-time availability. |
| `Line Item Value` | `line_value` | USD, nominal. Zero kept in counts, excluded from ratios. |
| `First Line Designation` | `is_first_line` | The first line of a shipment carries shipment-level freight and weight. |
| `Freight Cost (USD)` | `freight_kind`, `freight_num_raw` | Typed: numeric, `Freight Included in Commodity Cost`, `Invoiced Separately`, `See ASN/DN-... (ID#:n)`. Text is never coerced to 0. |
| `Weight (Kilograms)` | `weight_kind`, `weight_num_raw` | Same typing. Cleaned and logged but not modelled. |

Not used: PQ / PO fields, item descriptions, dosage, unit price, pack price, manufacturing site, insurance. The PO and
PQ date columns contain status strings (`Date Not Captured`, `Pre-PQ Process`, `N/A - From RDC`), counted in the cleaning log.

## Definitions

* **On time:** `delivered_date <= scheduled_date`. **Late:** delivered after scheduled. **Days late:** delivered minus
  scheduled (positive part), capped at 90 for mean-lateness metrics only (the flag is never capped). **Severe late:** more
  than 14 days late. **There is no vendor hand-off timestamp**, so lateness mixes vendor, freight-forwarder and in-country effects.
* **Known date:** `GREATEST(delivered_date, recorded_date)`: when an outcome was observable in the system.
* **Freight ratio:** shipment freight (first line, numeric only) / shipment line value, capped at 2x, value-weighted per vendor.

## Model tables (DuckDB, `sql/`)

| Table | Grain | Key columns |
|---|---|---|
| `stg_lines` | line | typed staging of every raw column used |
| `cleaning_log` | action | `ord, action, n_rows, base, detail` |
| `dq_report` | check | `check_name, n, total, detail` |
| `fct_shipments` | shipment | `n_lines, ship_value, ship_freight_usd, ship_weight_kg, has_numeric_freight` |
| `fct_lines` | line | `is_internal_rdc, offset_days, is_late, days_late, days_late_capped, is_severe_late, is_exact_match, is_extreme_early, known_date, freight_obs_usd, freight_obs_value, ship_n_lines, is_air` |
| `vendor_lane` / `vendor_summary` | vendor x country x mode / vendor | counts, value, late value, days late, freight, air share, country HHI, first/last scheduled |
| `vendor_year_panel` / `programme_year` | vendor x year / year | window functions: rolling 3-year late rate, lag, running totals, rank by volume |
| `oot_features` / `oot_vendor_lane` / `oot_outcomes` | vendor (x lane) | point-in-time features before `score_end`; outcomes for lines scheduled in `[score_end, validate_end)` |

## Published vendor scorecard (`app/data/vendors.parquet`, one row per external vendor)

`vendor, top_country, top_product_group, n_lines, n_shipments, value_usd, n_late, n_severe_late, n_exact_match,
late_rate_raw, late_rate_shrunk (+ lo90/hi90), severe_late_rate_*, mean_days_late_raw/shrunk, freight_ratio,
air_share, country_hhi, lane_mix_risk, volume_uncertainty, is_scored, risk_score (+ lo90/hi90), risk_tier, risk_rank,
archetype, pc1, pc2, value_at_risk, late_cost_ASSUMED, carrying_cost_ASSUMED, total_exposure_ASSUMED`.
Vendors with fewer than 10 lines have `is_scored = False`, no composite and the tier `New / unscored`.
Columns suffixed `_ASSUMED` depend on assumed parameters in `config/config.yaml`.

Other files in `app/data/` are aggregates (programme-year, country, mode, product group, validation tables, R outputs). No line-level rows are published.
