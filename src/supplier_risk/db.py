"""DuckDB helpers: register the raw SCMS table, run the numbered SQL model, re-point the out-of-time window."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import ROOT, load_config, resolve

SQL_DIR = ROOT / "sql"


def connect(db_path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = load_config()
    path = resolve(cfg["paths"]["duckdb"]) if db_path is None else Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    con.execute("SET threads = 1")      # parallel float SUMs are not bit-reproducible; one thread makes every run identical
    return con


def make_params(con: duckdb.DuckDBPyConnection, cfg: dict, score_end: str | None = None,
                validate_end: str | None = None) -> None:
    sc, cl = cfg["scoring"], cfg["cleaning"]
    con.execute(
        "CREATE OR REPLACE TABLE params AS SELECT ?::DATE AS score_end, ?::DATE AS validate_end, "
        "?::VARCHAR AS internal_vendor, ?::INTEGER AS severe_days, ?::INTEGER AS cap_days, "
        "?::INTEGER AS early_days, ?::DOUBLE AS freight_ratio_cap",
        [score_end or sc["score_period_end"], validate_end or sc["validate_period_end"], cl["internal_channel_vendor"],
         cl["severe_late_days"], cl["days_late_cap"], cl["extreme_early_days"], cl["freight_ratio_cap"]])


def register_inputs(con: duckdb.DuckDBPyConnection, raw: pd.DataFrame | None, cfg: dict, **params) -> None:
    """Register raw_scms (a DataFrame for tests, the interim parquet otherwise) and the params table."""
    con.execute("SET threads = 1")      # deterministic float aggregation (see connect)
    if raw is not None:
        con.register("raw_scms_df", raw)
        con.execute("CREATE OR REPLACE VIEW raw_scms AS SELECT * FROM raw_scms_df")
    else:
        p = (resolve(cfg["paths"]["interim_dir"]) / "scms_raw.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW raw_scms AS SELECT * FROM read_parquet('{p}')")
    make_params(con, cfg, **params)


def run_sql_model(con: duckdb.DuckDBPyConnection, raw: pd.DataFrame | None = None, cfg: dict | None = None,
                  verbose: bool = True, **params) -> list[str]:
    cfg = cfg or load_config()
    register_inputs(con, raw, cfg, **params)
    ran = []
    for script in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        con.execute(script.read_text())
        ran.append(script.name)
        if verbose:
            print(f"  ran {script.name}")
    return ran


def set_oot_window(con: duckdb.DuckDBPyConnection, score_end: str, validate_end: str,
                   cfg: dict | None = None) -> None:
    """Re-point params and rebuild only the out-of-time tables (06) for another split."""
    make_params(con, cfg or load_config(), score_end, validate_end)
    con.execute((SQL_DIR / "06_oot_features.sql").read_text())
