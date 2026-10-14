"""DuckDB helpers: register raw sources, run the numbered SQL model, export tables."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import ROOT, load_config, resolve

SQL_DIR = ROOT / "sql"
SOURCES = {
    "raw_orders": "olist_orders_dataset",
    "raw_items": "olist_order_items_dataset",
    "raw_sellers": "olist_sellers_dataset",
    "raw_customers": "olist_customers_dataset",
    "raw_reviews": "olist_order_reviews_dataset",
    "raw_products": "olist_products_dataset",
    "raw_category_translation": "product_category_name_translation",
}


def connect(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    path = resolve(cfg["paths"]["duckdb"]) if db_path is None else Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def register_inputs(con: duckdb.DuckDBPyConnection, sources: dict[str, pd.DataFrame] | None, cfg: dict,
                    score_end: str | None = None, validate_end: str | None = None) -> None:
    """Register raw tables (DataFrames for tests, interim parquet otherwise) and the params table."""
    interim = resolve(cfg["paths"]["interim_dir"])
    for view, stem in SOURCES.items():
        if sources is not None:
            con.register(f"{view}_df", sources[view])
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {view}_df")
        else:
            p = (interim / f"{stem}.parquet").as_posix()
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{p}')")
    sc = cfg["scoring"]
    con.execute("CREATE OR REPLACE TABLE params AS SELECT ?::TIMESTAMP AS score_end, ?::TIMESTAMP AS validate_end",
                [score_end or sc["score_period_end"], validate_end or sc["validate_period_end"]])


def run_sql_model(con: duckdb.DuckDBPyConnection, sources: dict[str, pd.DataFrame] | None = None,
                  cfg: dict | None = None, verbose: bool = True, **params) -> list[str]:
    cfg = cfg or load_config()
    register_inputs(con, sources, cfg, **params)
    ran = []
    for script in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        con.execute(script.read_text())
        ran.append(script.name)
        if verbose:
            print(f"  ran {script.name}")
    return ran


def set_oot_window(con: duckdb.DuckDBPyConnection, score_end: str, validate_end: str) -> None:
    """Re-point params and rebuild only the out-of-time tables (06) for another split."""
    con.execute("CREATE OR REPLACE TABLE params AS SELECT ?::TIMESTAMP AS score_end, ?::TIMESTAMP AS validate_end",
                [score_end, validate_end])
    con.execute((SQL_DIR / "06_oot_features.sql").read_text())
