"""Regression tests on the committed, aggregate-only artifacts in app/data (no raw data needed).

Expected values are the figures of the committed run; if the pipeline is re-run and a number legitimately changes,
update the test AND the memo / README (they are generated from the same files)."""
import json
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "app" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "summary.json").exists(), reason="app/data not exported")
C, EB, RAW = "Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Baseline: raw late rate"


def summ():
    return json.loads((DATA / "summary.json").read_text())


def test_data_is_the_documented_scms_file():
    q = summ()["sql_summary"]
    assert (q["n_lines"], q["n_shipments"], q["n_vendor_strings"], q["n_countries"]) == (10324, 7030, 73, 43)
    assert summ()["data_manifest"]["rows"] == 10324 and summ()["data_manifest"]["columns"] == 33
    assert q["dq"]["lines_rows"] == 10324 and q["dq"]["duplicate_line_id"] == 0 and q["dq"]["missing_delivery_dates"] == 0


def test_cleaning_counts_are_logged():
    cl = pd.read_csv(DATA / "cleaning_log.csv").set_index("action").n_rows
    assert cl["rows_in_raw_file"] == 10324 and cl["delivered_date_unparseable"] == 0
    assert cl["delivered_equals_scheduled"] == 6324 and cl["freight_see_reference"] == 2445 and cl["days_late_capped"] == 40


def test_vendor_population_and_scoring_split():
    t = summ()["train_summary"]
    assert t["n_vendors_external"] == 72 and t["n_vendors_scored"] == 29 and t["n_vendors_unscored"] == 43
    v = pd.read_parquet(DATA / "vendors.parquet")
    assert len(v) == 72 and int(v.is_scored.sum()) == 29 and "SCMS from RDC" not in set(v.vendor)
    assert v.loc[~v.is_scored, "risk_tier"].eq("New / unscored").all() and v.loc[~v.is_scored, "risk_score"].isna().all()


def test_headline_rates():
    q = summ()["sql_summary"]
    assert q["ext_late_rate"] == pytest.approx(0.0526, abs=5e-4)
    assert q["rdc_late_rate"] == pytest.approx(0.1715, abs=5e-4)
    assert q["ext_exact_match_share"] == pytest.approx(0.865, abs=2e-3)


def test_shrinkage_pulls_thin_vendors_more():
    s = summ()["train_summary"]["shrinkage"]
    assert s["median_abs_shift_late_rate_pp_lt30_lines"] > 10 * s["median_abs_shift_late_rate_pp_ge100_lines"]
    assert s["n_vendors_raw_rate_100pct"] == 0 and s["min_shrunk_late_rate_among_zero_late"] > 0


def test_oot_metrics_and_baseline_ordering():
    ev = pd.read_csv(DATA / "oot_metrics_main.csv")
    late = ev[ev.target.str.startswith("Late")].set_index("score").line_auc
    assert late[C] == pytest.approx(0.704, abs=0.005) and late[EB] == pytest.approx(0.7375, abs=0.005)
    assert late[EB] >= late[C] and late[RAW] >= late[C]                    # the composite does not beat the simple baselines
    assert late["Baseline: volume only (smaller = riskier)"] < 0.5 and late["Baseline: lane mix only (country / mode)"] < 0.5


def test_auc_difference_intervals_reported_and_not_overclaimed():
    d = summ()["train_summary"]["auc_deltas"]
    x = d["composite_vs_raw_late_late"]
    assert x["lo95"] < 0 < x["hi95"] and x["n_valid_resamples"] >= 190


def test_oot_main_split_is_point_in_time():
    m = summ()["train_summary"]["oot_main"]
    assert m["max_known_date_used"][:10] < m["score_end"] and m["max_scheduled_used"][:10] < m["score_end"]
    assert (m["n_vendors_evaluated"], m["next_lines"], m["next_late_lines"]) == (21, 2211, 108)


def test_four_backtest_splits_cover_both_targets():
    bt = pd.read_csv(DATA / "oot_backtest_metrics.csv")
    assert bt.split.nunique() == 4 and bt.target.nunique() == 2


def test_cold_start_tier_is_documented_and_prior_is_not_always_valid():
    cs = pd.read_csv(DATA / "cold_start.csv")
    assert len(cs) == 4 and (cs.new_share_next_lines > 0).all()
    assert cs.new_late.notna().all() and cs.prior_mean_late.notna().all()


def test_clusters_are_indicative():
    c = summ()["train_summary"]["clusters"]
    assert c["k"] in (2, 3, 4) and c["n_clustered"] == 24 and 0 < c["silhouette"] < 1
    assert len(set(c["names"])) == c["k"]


def test_scenario_is_labelled_assumed_and_composite_avoids_nothing():
    t = summ()["train_summary"]
    assert t["scenario_base_ASSUMED"]["late_value_avoided"] == 0
    assert t["scenario_raw_baseline_ASSUMED"]["late_value_reduction_pct"] > 0.1
    assert all(k.endswith("ASSUMED") or "_ASSUMED" in k for k in t if k.startswith("scenario"))
    g = pd.read_csv(DATA / "scenario_grid.csv")
    assert "improvement_ASSUMED" in g.columns and len(g) == 27


def test_r_layer_findings():
    r = summ()["r_summary"]
    assert r["n_lines"] == 4920 and r["n_late"] == 259
    assert 0.2 < r["glmm_icc"] < 0.4 and r["glmm_p_vendor_boundary"] < 1e-6 and r["glmm_singular"] is False
    assert r["or_internal_rdc_lo"] > 1 and r["or_truck"] < 1


def test_published_data_has_no_line_level_rows():
    files = {p.name for p in DATA.iterdir()}
    assert "fct_lines.parquet" not in files and not any("lines_for_r" in f for f in files)
    assert max(len(pd.read_csv(p)) for p in DATA.glob("*.csv")) < 2000
    assert len(pd.read_parquet(DATA / "vendors.parquet")) == 72
