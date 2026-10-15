"""Shared fixtures: a deterministic synthetic SCMS-format extract and the committed handcrafted edge cases, run through the SQL model."""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import FIX, synthetic_raw  # noqa: E402

from supplier_risk.config import load_config  # noqa: E402
from supplier_risk.db import run_sql_model  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw():
    return synthetic_raw()


@pytest.fixture(scope="session")
def edge_raw_df():
    return pd.read_csv(FIX / "scms_edge_cases.csv", dtype=str, keep_default_na=False).drop(columns="case")


@pytest.fixture(scope="session")
def con(raw, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, raw=raw, cfg=cfg, verbose=False, score_end="2012-01-01", validate_end="2016-01-01")
    yield c
    c.close()


@pytest.fixture(scope="session")
def edge_con(edge_raw_df, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, raw=edge_raw_df, cfg=cfg, verbose=False)
    yield c
    c.close()
