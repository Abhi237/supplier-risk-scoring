# Architecture

```mermaid
flowchart LR
    A[Olist CSVs<br/>public GitHub mirror] -->|download_data.py<br/>row counts + SHA-256| B[(data/interim<br/>parquet)]
    B --> C[sql/01-06<br/>DuckDB]
    C --> D[(olist.duckdb<br/>fct_items, seller_summary,<br/>seller_month_panel, oot_*)]
    D --> E[Python src/supplier_risk<br/>EB shrinkage, Bayesian bootstrap,<br/>composite score, k-means,<br/>OOT validation, exposure, scenarios]
    D -->|orders_delivered_for_r.csv| F[R/delivery_analysis.R<br/>GLM, GLMM, NB, KS/Wilcoxon]
    E --> G[artifacts/ + docs/figures]
    F --> G
    G --> H[reports/delivery_report.html<br/>Quarto]
    G --> I[docs/procurement_memo.md/.pdf<br/>build_memo.py]
    G --> J[notebooks/ executed]
    G -->|export_app_data.py<br/>aggregates only| K[app/data ~1 MB]
    K --> L[Streamlit multipage app]
```

| Step | Command | Main outputs |
|---|---|---|
| data | `make data` | `data/raw/*.csv`, `data/interim/*.parquet`, manifest |
| sql | `make sql` | `data/processed/olist.duckdb`, parquet exports, `artifacts/sql_summary.json` |
| train | `make train` | `artifacts/seller_scores.parquet`, OOT metrics, clusters, scenarios, `docs/figures/*` |
| report | `make report` | `artifacts/r_glm/*`, `reports/delivery_report.html`, memo, executed notebooks |
| app-data | `make app-data` | `app/data/*` |
| test | `make test` | ruff + pytest (fixtures; real-data regression tests run when artifacts exist) |

**Determinism:** all random steps are seeded (`scoring.random_state`, `clustering.random_state`, R `set.seed(42)`);
k-means labels are re-ordered by cluster size. Re-running `make sql train` reproduces identical artifacts.
