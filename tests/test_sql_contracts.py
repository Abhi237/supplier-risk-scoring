"""SQL model contracts on the handcrafted edge cases (committed fixture) and the synthetic extract."""
import pytest


def cols(con, t):
    return {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}


def line(edge_con, line_id):
    return edge_con.execute("SELECT * FROM fct_lines WHERE line_id = ?", [line_id]).df().iloc[0]


def log(edge_con, action):
    return int(edge_con.execute("SELECT n_rows FROM cleaning_log WHERE action = ?", [action]).fetchone()[0])


def test_schema(con):
    assert {"line_id", "shipment_id", "vendor", "is_internal_rdc", "scheduled_date", "delivered_date", "known_date", "is_late",
            "days_late", "days_late_capped", "is_severe_late", "is_exact_match", "offset_days"} <= cols(con, "fct_lines")
    assert {"vendor", "n_lines", "n_late", "n_severe_late", "value_usd"} <= cols(con, "vendor_summary")
    assert {"late_rate_roll3", "cum_lines", "volume_rank_in_year"} <= cols(con, "vendor_year_panel")


@pytest.mark.parametrize("table,key", [("fct_lines", "line_id"), ("fct_shipments", "shipment_id"), ("vendor_summary", "vendor"),
                                       ("vendor_year_panel", "vendor, sched_year"), ("oot_features", "vendor"),
                                       ("oot_outcomes", "vendor")])
def test_grain_unique(con, table, key):
    n, d = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT ({key})) FROM {table}").fetchone()
    assert n == d and n > 0


def test_line_count_preserved(con, raw):
    n = con.execute("SELECT COUNT(*) FROM fct_lines").fetchone()[0]
    assert n == len(raw)


def test_late_is_strictly_after_scheduled(edge_con):
    assert bool(line(edge_con, 1).is_late) and line(edge_con, 1).days_late == 5
    assert not bool(line(edge_con, 3).is_late) and bool(line(edge_con, 3).is_exact_match)   # delivered == scheduled is on time
    assert bool(line(edge_con, 14).is_late) and line(edge_con, 14).days_late == 1           # one day late counts as late


def test_early_delivery_is_on_time_with_zero_days_late(edge_con):
    r = line(edge_con, 4)
    assert not bool(r.is_late) and r.days_late == 0 and r.offset_days == -120 and bool(r.is_extreme_early)


def test_severe_late_threshold_and_cap(edge_con):
    assert bool(line(edge_con, 5).is_severe_late) and line(edge_con, 5).days_late_capped == 20
    r = line(edge_con, 6)
    assert r.days_late == 200 and r.days_late_capped == 90 and bool(r.is_late)   # cap is for means only, flag unchanged
    assert not bool(line(edge_con, 14).is_severe_late)


def test_unparseable_delivered_date_is_excluded_not_guessed(edge_con):
    ids = set(edge_con.execute("SELECT line_id FROM fct_lines").df().line_id)
    assert 10 not in ids and len(ids) == 13
    assert log(edge_con, "delivered_date_unparseable") == 1


def test_internal_rdc_flagged_and_excluded_from_scoring_inputs(edge_con):
    assert bool(line(edge_con, 7).is_internal_rdc)
    vs = edge_con.execute("SELECT vendor, is_internal_rdc FROM vendor_summary").df().set_index("vendor").is_internal_rdc
    assert vs["SCMS from RDC"] and not vs["V_A"] and not vs["V_B"]
    assert "SCMS from RDC" not in set(edge_con.execute("SELECT vendor FROM vendor_lane").df().vendor)
    assert "SCMS from RDC" not in set(edge_con.execute("SELECT vendor FROM oot_features").df().vendor)
    assert log(edge_con, "po_sent_date_na_from_rdc") == 1


def test_freight_text_is_typed_not_coerced_to_zero(edge_con):
    kinds = edge_con.execute("SELECT freight_kind, COUNT(*) n FROM stg_lines GROUP BY 1").df().set_index("freight_kind").n
    assert kinds["included_in_commodity_cost"] == 1 and kinds["invoiced_separately"] == 1 and kinds["see_reference"] == 1
    assert edge_con.execute("SELECT freight_num_raw FROM stg_lines WHERE line_id = 3").fetchone()[0] is None


def test_see_reference_line_inherits_shipment_freight(edge_con):
    a, b = line(edge_con, 1), line(edge_con, 2)
    assert a.ship_freight_usd == b.ship_freight_usd == 100
    assert a.freight_obs_usd == 100 and b.freight_obs_usd != b.freight_obs_usd    # carried once, on the first line


def test_freight_ratio_capped_at_config_multiple(edge_con, cfg):
    r = line(edge_con, 13)
    assert r.ship_freight_ratio == cfg["cleaning"]["freight_ratio_cap"] and r.freight_obs_usd == 200
    assert log(edge_con, "freight_ratio_capped") >= 1


def test_negative_freight_becomes_null_not_capped_ratio(edge_con):
    r = line(edge_con, 9)
    assert r.ship_freight_usd != r.ship_freight_usd and r.ship_freight_ratio != r.ship_freight_ratio
    assert log(edge_con, "freight_negative_set_null") == 1


def test_zero_value_kept_and_excluded_from_ratio(edge_con):
    r = line(edge_con, 8)
    assert r.line_value == 0 and r.ship_freight_ratio != r.ship_freight_ratio
    assert log(edge_con, "line_value_zero_kept") == 1
    assert log(edge_con, "weight_zero_set_null") == 1


def test_mode_na_becomes_unknown(edge_con):
    assert line(edge_con, 12).ship_mode == "Unknown" and log(edge_con, "ship_mode_na_to_unknown") == 1


def test_recorded_before_delivered_floors_lag(edge_con):
    r = line(edge_con, 11)
    assert r.record_lag_days == 0 and str(r.known_date)[:10] == "2012-03-04"
    assert log(edge_con, "recorded_before_delivered") == 1


def test_cleaning_log_is_complete_and_consistent(edge_con):
    cl = edge_con.execute("SELECT * FROM cleaning_log").df()
    assert cl.action.is_unique and (cl.n_rows >= 0).all() and (cl.n_rows <= cl.base).all()
    assert cl.loc[cl.action == "rows_in_raw_file", "n_rows"].iloc[0] == 14


def test_dq_report_counts_edge_cases(edge_con):
    dq = edge_con.execute("SELECT * FROM dq_report").df().set_index("check_name")
    assert dq.loc["delivered_equals_scheduled", "n"] == 5
    assert dq.loc["duplicate_line_id", "n"] == 0
