# Data

**Dataset:** USAID / PEPFAR Supply Chain Management System (SCMS) *Delivery History* (a.k.a. *Supply Chain Shipment Pricing
Data*), file date 2015-09-29. 10,324 order-line rows x 33 columns, 7,030 shipments, 73 vendor strings, 43 destination countries,
scheduled delivery dates 2006-05 to 2015-12.

**Citation:** U.S. Agency for International Development (USAID), PEPFAR Supply Chain Management System (SCMS), *Delivery History
Dataset* (published on data.usaid.gov / usaid.gov open data; a copy is on Kaggle as
`divyeshardeshana/supply-chain-shipment-pricing-data`).

**Licence: unclear.** USAID published it as open data but the copies available to this project carry no explicit licence text,
and the Kaggle page is a third-party re-upload. For that reason the raw CSV is **never committed**: `data/raw/` is git-ignored and
only derived vendor scorecards and aggregates are published in `app/data/` and `artifacts/`. Vendor names are as recorded in the public dataset.

## Fetch (no credentials needed)

```bash
make data          # = python scripts/download_data.py
python scripts/download_data.py --source file --file /path/to/SCMS_Delivery_History_Dataset.csv   # a copy you downloaded by hand
```

`scripts/download_data.py` tries, in the order of `config/config.yaml: data.sources`, and accepts the first copy that downloads
**and** parses to the documented shape (10,324 rows, 33 columns, required columns present, unique `ID`):

| # | Source | Result on the build that produced every number here |
|---|---|---|
| 1 | `data.usaid.gov` catalogue export (`a3rc-nmtj`) | failed: SSL error (`UNEXPECTED_EOF_WHILE_READING`) |
| 2 | `usaid.gov/opengov/developer/datasets/SCMS_Delivery_History_Dataset_20150929.csv` | failed: HTTP 403 |
| 3 | Internet Archive snapshot of source 2 (`web.archive.org/web/20180906063754id_/...`) | **used**, SHA-256 `9cffdb4a...3578a`, 3,819,612 bytes, Latin-1 |
| 4 | public GitHub mirror (`MSadriAghdam/Supply-Chain-Prediction_Neural-Network-ML`) | not needed; checked: identical in every column the analysis uses, two unused procurement date columns use two-digit years |
| - | Kaggle copy | only with `KAGGLE_USERNAME` / `KAGGLE_KEY` and `--source kaggle`; not used |

A second Internet Archive snapshot (2020) of source 2 is identical cell for cell. The script writes the canonical UTF-8 CSV
(`data/raw/SCMS_Delivery_History_Dataset.csv`, SHA-256 `eef7bb75...64837a44`, all values untouched), `data/raw/manifest.json`, and
`data/interim/scms_raw.parquet` (every column kept as text; typing and cleaning happen in SQL, `sql/01_stage.sql`). If no source
yields the documented shape, the script stops and writes nothing.

## What is and is not published

| Path | Content | Committed |
|---|---|---|
| `data/raw/`, `data/interim/`, `data/processed/` | raw CSV, text parquet, DuckDB model, line-level parquet and R export | no (git-ignored) |
| `artifacts/*.csv`, `*.json` | cleaning log, DQ report, validation tables, vendor scorecard extracts | yes (aggregates and vendor-level only) |
| `app/data/` | the same plus vendor scorecards and year / country / mode aggregates (about 0.25 MB) | yes |
| `tests/fixtures/scms_edge_cases.csv` | 14 handcrafted rows with invented vendor names to exercise the cleaning rules | yes |

## Known data issues (counted in `artifacts/cleaning_log.csv`)

Delivery dates are complete and parse (0 unparseable). 6,324 of 10,324 lines (61%) have delivered date = scheduled date exactly
(86.5% for external vendors), which may reflect back-filled scheduled dates; freight and weight are partly text values
(included in cost, invoiced separately, or a reference to another line); 40 lines are more than 90 days late (capped for mean
lateness only); 209 shipments have freight above 2x their value (ratio capped); 17 lines have zero value; 10 shipments carry two
scheduled dates; one group of rows is identical except for `ID`. Only delivered lines are present, so late rates are lower bounds.
