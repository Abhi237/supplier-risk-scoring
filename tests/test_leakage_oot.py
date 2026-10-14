"""Out-of-time / leakage tests: score-period features must not see anything at or after score_end."""
import duckdb
import pandas as pd

from supplier_risk.db import run_sql_model, set_oot_window

SCORE_END, VALID_END = "2018-01-01", "2018-09-01"


def feats(sources, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, sources=sources, cfg=cfg, verbose=False, score_end=SCORE_END, validate_end=VALID_END)
    f = c.execute("SELECT * FROM oot_features ORDER BY seller_id").df()
    c.close()
    return f


def test_features_use_only_pre_cutoff_information(con):
    set_oot_window(con, SCORE_END, VALID_END)
    f = con.execute("SELECT * FROM oot_features").df()
    assert pd.to_datetime(f.max_delivered_ts_used).max() < pd.Timestamp(SCORE_END)
    assert pd.to_datetime(f.max_purchase_ts_used).max() < pd.Timestamp(SCORE_END)


def test_future_rows_do_not_change_features(sources, cfg):
    base = feats(sources, cfg)
    s2 = {k: v.copy() for k, v in sources.items()}
    o = s2["raw_orders"]
    fut = o.order_purchase_timestamp >= SCORE_END
    o.loc[fut, "order_delivered_customer_date"] = o.loc[fut, "order_estimated_delivery_date"] + pd.Timedelta(days=30)
    # also: deliveries of PRE-cutoff orders that land after the cutoff are unresolved at scoring time
    straddle = (o.order_purchase_timestamp < SCORE_END) & (o.order_delivered_customer_date >= SCORE_END)
    o.loc[straddle, "order_delivered_customer_date"] += pd.Timedelta(days=40)
    s2["raw_reviews"].loc[pd.to_datetime(s2["raw_reviews"].review_creation_date) >= SCORE_END, "review_score"] = 1
    pd.testing.assert_frame_equal(base.drop(columns=["max_delivered_ts_used"]),
                                  feats(s2, cfg).drop(columns=["max_delivered_ts_used"]))


def test_outcomes_strictly_in_next_window(con):
    set_oot_window(con, SCORE_END, VALID_END)
    n_out = con.execute("SELECT SUM(next_n_items) FROM oot_outcomes").fetchone()[0]
    n_win = con.execute(f"SELECT COUNT(*) FROM fct_items WHERE purchase_ts >= '{SCORE_END}' "
                        f"AND purchase_ts < '{VALID_END}'").fetchone()[0]
    assert n_out == n_win
