# Data

**Source:** Olist / André Sionek (2018). *Brazilian E-Commerce Public Dataset by Olist* [Dataset]. Kaggle.
<https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

**License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Attribute Olist, use **non-commercially**, and share derivatives under the same license. Derived artifacts in this
repository (aggregated tables in `app/data/`, figures, memo) inherit these terms; the MIT license covers the code only.

## Fetch (primary: Kaggle; fallback: verified mirror)

The **primary, documented source is Kaggle** (`olistbr/brazilian-ecommerce`). `scripts/download_data.py` uses it
whenever Kaggle API credentials are present in the environment, and otherwise falls back to a public GitHub mirror of
the same CSVs (`config.yaml: data.mirror_base`, currently
`https://raw.githubusercontent.com/Kaaykun/OlistAnalysis/master/data/csv`), which is what the committed results
were built from.

```bash
export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>   # optional; from kaggle.com > Settings > API
make data                                        # = python scripts/download_data.py  (--source auto)
python scripts/download_data.py --source kaggle  # force Kaggle (fails if no credentials)
python scripts/download_data.py --source mirror  # force the mirror
```

With credentials the script calls the `kaggle` CLI if installed, else the Kaggle REST endpoint
`/api/v1/datasets/download/olistbr/brazilian-ecommerce` (HTTP basic auth). A Kaggle failure under `--source auto`
falls back to the mirror. You can also download the zip from Kaggle by hand, unzip into `data/raw/`, and run
`make data`: existing files are verified, not re-downloaded.

### Verification (recorded on the build that produced every number in this repo)

Both paths are checked against the values recorded in `config.yaml: data.recorded`. **Row counts are a hard check on
either path.** SHA-256 must match exactly on the mirror path; on the Kaggle path a digest mismatch is reported as a
warning (re-packaging can change bytes such as line endings without changing content) provided the row counts match.
The observed values, and which source was used, are written to `data/raw/manifest.json` and
`artifacts/data_manifest.json`. To verify by hand: `sha256sum data/raw/*.csv`.

| File | Rows | SHA-256 |
|---|---:|---|
| `olist_orders_dataset.csv` | 99,441 | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |
| `olist_order_items_dataset.csv` | 112,650 | `0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279` |
| `olist_sellers_dataset.csv` | 3,095 | `1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e` |
| `olist_customers_dataset.csv` | 99,441 | `983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2` |
| `olist_order_reviews_dataset.csv` | 99,224 | `012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7` |
| `olist_products_dataset.csv` | 32,951 | `3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc` |
| `olist_order_payments_dataset.csv` | 103,886 | `4f713964f2815dbbaa40b9488268c55aac3627bfce5aa96cf58d1f3616de3cc0` |
| `product_category_name_translation.csv` | 71 | `a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278` |

The documented dataset sizes (99,441 orders, 112,650 items, 3,095 sellers) match. `olist_geolocation_dataset.csv`
(1M rows) is not needed; payments are downloaded for completeness only.

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
