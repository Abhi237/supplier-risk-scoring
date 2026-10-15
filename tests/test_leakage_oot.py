"""Point-in-time tests: score-period features must not see anything at or after score_end."""
import duckdb

from supplier_risk.db import run_sql_model, set_oot_window

SCORE_END, VALID_END = "2012-01-01", "2016-01-01"


def test_features_use_only_pre_cutoff_information(con):
    set_oot_window(con, SCORE_END, VALID_END)
    f = con.execute("SELECT * FROM oot_features").df()
    assert len(f) > 5
    assert (f.max_known_date_used < __import__("pandas").Timestamp(SCORE_END)).all()
    assert (f.max_scheduled_used < __import__("pandas").Timestamp(SCORE_END)).all()


def test_features_match_manual_recount_with_known_date_rule(con):
    set_oot_window(con, SCORE_END, VALID_END)
    manual = con.execute(f"""SELECT vendor, COUNT(*) n, SUM(is_late::INT) k FROM fct_lines
        WHERE NOT is_internal_rdc AND scheduled_date < DATE '{SCORE_END}' AND known_date < DATE '{SCORE_END}' GROUP BY vendor ORDER BY vendor""").df()
    f = con.execute("SELECT vendor, n_lines n, n_late k FROM oot_features ORDER BY vendor").df()
    assert manual.equals(f)


def test_lines_unresolved_at_score_date_are_excluded(con):
    set_oot_window(con, SCORE_END, VALID_END)
    straddle = con.execute(f"""SELECT COUNT(*) FROM fct_lines WHERE NOT is_internal_rdc AND scheduled_date < DATE '{SCORE_END}'
        AND known_date >= DATE '{SCORE_END}'""").fetchone()[0]
    all_before = con.execute(f"SELECT COUNT(*) FROM fct_lines WHERE NOT is_internal_rdc AND scheduled_date < DATE '{SCORE_END}'").fetchone()[0]
    used = con.execute("SELECT SUM(n_lines) FROM oot_features").fetchone()[0]
    assert straddle > 0 and used == all_before - straddle


def test_outcomes_are_only_in_validation_window(con):
    set_oot_window(con, SCORE_END, VALID_END)
    n = con.execute("SELECT SUM(next_n_lines) FROM oot_outcomes").fetchone()[0]
    ref = con.execute(f"""SELECT COUNT(*) FROM fct_lines WHERE NOT is_internal_rdc AND scheduled_date >= DATE '{SCORE_END}'
        AND scheduled_date < DATE '{VALID_END}'""").fetchone()[0]
    assert n == ref


def test_future_lateness_does_not_change_features(raw, cfg):
    base = _features(raw, cfg)
    r2 = raw.copy()
    r2["Delivered to Client Date"] = r2["Delivered to Client Date"].where(r2["Scheduled Delivery Date"].map(_yr) < 2012, "1-Jan-15")
    r2["Delivery Recorded Date"] = r2["Delivery Recorded Date"].where(r2["Scheduled Delivery Date"].map(_yr) < 2012, "1-Jan-15")
    assert base.equals(_features(r2, cfg))


def _yr(s):
    return 2000 + int(s.split("-")[-1])


def _features(raw, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, raw=raw, cfg=cfg, verbose=False, score_end=SCORE_END, validate_end=VALID_END)
    out = c.execute("SELECT vendor, n_lines, n_late FROM oot_features ORDER BY vendor").df()
    c.close()
    return out
