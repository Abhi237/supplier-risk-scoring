# Data

**Source:** Olist / André Sionek (2018). *Brazilian E-Commerce Public Dataset by Olist* [Dataset]. Kaggle.
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

**License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Attribute Olist, use **non-commercially**, and share derivatives under the same license. Derived artifacts in this
repository (aggregated tables in `app/data/`, figures, memo) inherit these terms; the MIT license covers the code only.

## Fetch

Kaggle requires a login, and no Kaggle API credentials are used here, so `scripts/download_data.py` pulls the original
CSVs from a public GitHub mirror (`config.yaml: data.mirror_base`, currently
`https://raw.githubusercontent.com/Kaaykun/OlistAnalysis/master/data/csv`) and then **verifies integrity**:

```bash
make data          # = python scripts/download_data.py
```

| Check | Expected (documented) | Observed |
|---|---|---|
| `olist_orders_dataset.csv` rows | 99,441 | 99,441 |
| `olist_order_items_dataset.csv` rows | 112,650 | 112,650 |
| `olist_sellers_dataset.csv` rows | 3,095 | 3,095 |

The script aborts if a count differs, records row counts and SHA-256 digests in `data/raw/manifest.json` and
`artifacts/data_manifest.json`, and converts each CSV to `data/interim/*.parquet`. If the mirror disappears, download
the zip from Kaggle manually into `data/raw/` and re-run (the script skips files that exist; the checks still apply).
`olist_geolocation_dataset.csv` (1M rows) and `olist_order_payments_dataset.csv` are not needed for seller scoring;
payments are downloaded for completeness only.

| Folder | Contents | Committed? |
|---|---|---|
| `data/raw/` | CSVs as downloaded + manifest | **No** |
| `data/interim/` | parquet conversion | **No** |
| `data/processed/` | DuckDB database, model tables, order-grain CSV for R | **No** |

## Tables used

| File | Grain | Used for |
|---|---|---|
| orders | order | status, purchase / approval / carrier / delivery / estimated timestamps |
| order_items | order x item | seller, product, `shipping_limit_date`, price, freight |
| sellers | seller | state, city, zip prefix |
| customers | customer (per order) | **state only** (lane analysis) |
| order_reviews | review | score, creation date (latest per order kept) |
| products + category translation | product | English category, weight |

## Privacy guidance

* Olist IDs are already anonymised hashes and company/partner names were replaced. Do **not** attempt to
  re-identify sellers or customers or join to external data.
* Customer data is reduced to `customer_state` in staging; customer IDs never leave `data/processed/` and are not in
  any committed artifact. The app contains seller-level aggregates only.
* **Sellers are proxies for suppliers**: treat scores as analytics on marketplace merchants, not as findings about
  any real vendor relationship. If you apply this to real supplier data, scores about identifiable vendors are
  commercially sensitive; restrict access and document the methodology before using them in sourcing decisions.
