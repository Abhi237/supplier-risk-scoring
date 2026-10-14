"""Data contracts for the SQL model: schema, grain, nulls, interval logic and data-quality checks."""
import pytest

FCT_COLS = {"order_id", "order_item_id", "seller_id", "seller_state", "customer_state", "category", "purchase_ts",
            "carrier_ts", "delivered_ts", "estimated_ts", "shipping_limit_ts", "price", "freight_value",
            "freight_to_price", "handoff_delay_days", "delivery_days", "delay_days", "is_late", "is_handoff_late",
            "is_canceled", "is_delivered", "review_score", "is_multi_seller_order"}


def cols(con, t):
    return {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}


def test_schema(con):
    assert FCT_COLS <= cols(con, "fct_items")
    assert {"seller_id", "n_items", "n_late", "n_delivered", "n_handoff_late", "n_handoff_obs", "gmv",
            "late_gmv", "mean_review_score"} <= cols(con, "seller_summary")
    assert {"late_rate_roll3", "cum_items", "gmv_rank_in_month", "month_index"} <= cols(con, "seller_month_panel")


@pytest.mark.parametrize("table,key", [("fct_items", "order_id, order_item_id"), ("seller_summary", "seller_id"),
                                       ("seller_month_panel", "seller_id, purchase_month"),
                                       ("fct_orders_delivered", "order_id"), ("stg_reviews", "order_id"),
                                       ("oot_features", "seller_id"), ("oot_outcomes", "seller_id")])
def test_grain_unique(con, table, key):
    n, d = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT ({key})) FROM {table}").fetchone()
    assert n == d and n > 0


def test_item_count_preserved(con, sources):
    assert con.execute("SELECT COUNT(*) FROM fct_items").fetchone()[0] == len(sources["raw_items"])


def test_no_null_keys(con):
    assert con.execute("SELECT COUNT(*) FROM fct_items WHERE seller_id IS NULL OR order_id IS NULL "
                       "OR purchase_ts IS NULL").fetchone()[0] == 0


def test_late_flag_only_for_delivered(con):
    q = "SELECT COUNT(*) FROM fct_items WHERE is_late IS NOT NULL AND NOT is_delivered"
    assert con.execute(q).fetchone()[0] == 0


def test_seller_summary_reconciles(con):
    a = con.execute("SELECT SUM(n_items), SUM(n_late), SUM(gmv) FROM seller_summary").fetchone()
    b = con.execute("SELECT COUNT(*), SUM(is_late::INT), SUM(price)::DOUBLE FROM fct_items").fetchone()
    assert a[:2] == b[:2]
    assert a[2] == pytest.approx(b[2], abs=0.01)


def test_rolling_window_bounds(con):
    bad = con.execute("SELECT COUNT(*) FROM seller_month_panel WHERE late_rate_roll3 < 0 OR late_rate_roll3 > 1"
                      ).fetchone()[0]
    assert bad == 0


def flag(edge_con, oid, col, item=1):
    return edge_con.execute(f"SELECT {col} FROM fct_items WHERE order_id = ? AND order_item_id = ?",
                            [oid, item]).fetchone()[0]


def test_edge_late_and_handoff_logic(edge_con):
    assert flag(edge_con, "E1", "is_late") is False and flag(edge_con, "E1", "is_handoff_late") is False
    assert flag(edge_con, "E2", "is_late") is True and flag(edge_con, "E2", "is_handoff_late") is True
    assert flag(edge_con, "E4", "is_late") is False          # same calendar day as estimate is on time
    assert flag(edge_con, "E5", "is_handoff_late") is None   # bad clock -> not judged
    assert flag(edge_con, "E3", "is_canceled") is True and flag(edge_con, "E3", "is_late") is None
    assert flag(edge_con, "E6", "is_delivered") is False     # delivered status but no timestamp
    assert flag(edge_con, "E7", "is_multi_seller_order") is True
    assert flag(edge_con, "E2", "delay_days") == 5


def test_edge_review_dedup_keeps_latest(edge_con):
    assert edge_con.execute("SELECT review_score FROM stg_reviews WHERE order_id='E1'").fetchone()[0] == 2


def test_edge_staging_normalises(edge_con):
    z = dict(edge_con.execute("SELECT seller_id, seller_zip_prefix FROM stg_sellers").fetchall())
    assert z["S1"] == "00123"
    assert edge_con.execute("SELECT seller_state FROM stg_sellers WHERE seller_id='S1'").fetchone()[0] == "SP"


def test_edge_dq_report(edge_con):
    dq = dict(edge_con.execute("SELECT check_name, n FROM dq_report").fetchall())
    assert dq["orders_without_items"] == 1
    assert dq["delivered_missing_delivery_ts"] == 1
    assert dq["carrier_before_purchase"] == 1
    assert dq["missing_approved_ts"] == 1
    assert dq["zero_freight"] == 1
    assert dq["orders_canceled_or_unavailable"] == 1
    assert dq["duplicate_item_key"] == 0
