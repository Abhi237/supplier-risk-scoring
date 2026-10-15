"""Two runs of the SQL model and scoring give bit-identical outputs (single-threaded DuckDB, explicit ordering)."""
import duckdb
import pandas as pd

from supplier_risk import pipeline
from supplier_risk.db import run_sql_model


def _run(raw, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, raw=raw, cfg=cfg, verbose=False, score_end="2012-01-01", validate_end="2016-01-01")
    vs = c.execute("SELECT * FROM vendor_summary ORDER BY vendor").df()
    py = c.execute("SELECT * FROM programme_year ORDER BY 1").df()
    ev, meta = pipeline.oot_split(c, cfg, "2012-01-01", "2016-01-01")
    feat = vs[~vs.is_internal_rdc]
    lane = c.execute("SELECT * FROM vendor_lane ORDER BY vendor, country, ship_mode").df()
    comp, _, _ = pipeline.score_vendors(feat, lane, cfg)
    c.close()
    return vs, py, ev, meta, comp


def test_two_runs_are_identical(raw, cfg):
    a, b = _run(raw, cfg), _run(raw, cfg)
    for x, y in zip(a, b, strict=True):
        if isinstance(x, pd.DataFrame):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        else:
            assert x == y


def test_row_order_of_input_does_not_change_vendor_scores(raw, cfg):
    shuffled = raw.sample(frac=1.0, random_state=5).reset_index(drop=True)
    a, b = _run(raw, cfg), _run(shuffled, cfg)
    pd.testing.assert_frame_equal(a[4].drop(columns="risk_rank", errors="ignore"), b[4].drop(columns="risk_rank", errors="ignore"),
                                  check_exact=False, rtol=1e-9)
