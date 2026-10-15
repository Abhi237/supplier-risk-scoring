"""Metric correctness for OOT validation, clustering helpers and the exposure / scenario logic."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from supplier_risk import clustering, exposure, pipeline, validation


def test_line_auc_matches_expanded():
    rng = np.random.default_rng(0)
    s = rng.random(50)
    n = rng.integers(1, 20, 50)
    k = rng.binomial(n, 0.2 + 0.5 * s)
    y = np.concatenate([np.r_[np.ones(ki), np.zeros(ni - ki)] for ki, ni in zip(k, n)])
    x = np.concatenate([np.full(ni, si) for si, ni in zip(s, n)])
    assert validation.line_level_auc(s, k, n) == pytest.approx(roc_auc_score(y, x))


def test_line_auc_single_class_is_nan():
    assert np.isnan(validation.line_level_auc([0.1, 0.2], [0, 0], [5, 5]))


def test_lift_and_capture():
    s = np.arange(10)
    k = np.r_[np.zeros(9), 10]
    n = np.full(10, 10)
    r = validation.top_share_lift(s, k, n, 0.1)
    assert r["lift"] == pytest.approx(10.0) and r["capture_late_lines"] == pytest.approx(1.0)


def test_perfect_random_and_inverted_scores():
    n = np.full(200, 10)
    k = np.r_[np.zeros(100), np.full(100, 10)]
    good = np.r_[np.zeros(100), np.ones(100)]
    assert validation.line_level_auc(good, k, n) == pytest.approx(1.0)
    assert validation.line_level_auc(np.ones(200), k, n) == pytest.approx(0.5)
    assert validation.line_level_auc(-good, k, n) == pytest.approx(0.0)


def test_bootstrap_delta_sign_and_nan_guard():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    n = np.full(300, 20)
    k = rng.binomial(n, p)
    r = validation.bootstrap_auc_delta(p, rng.random(300), k, n, n_boot=50, seed=0)
    assert r["delta"] > 0 and r["lo95"] > 0 and r["n_valid_resamples"] == 50
    r2 = validation.bootstrap_auc_delta(p[:6], p[:6][::-1], np.r_[np.zeros(5), 3], np.full(6, 5), n_boot=30, seed=0)
    assert r2["n_valid_resamples"] <= 30


def test_reallocation_zero_when_no_gap():
    d = pd.DataFrame({"s": np.arange(10.0), "val": 100.0, "lval": 10.0, "k": 1, "n": 10})
    r = exposure.reallocation_backtest(d.s, d.val, d.lval, d.k, d.n, 0.1, 1.0, 1.0)
    assert r["late_value_avoided"] == 0


def test_reallocation_full_gap_and_assumption_scaling():
    d = pd.DataFrame({"s": np.arange(10.0), "val": 100.0, "lval": [0.0] * 9 + [50.0], "k": 1, "n": 10})
    r = exposure.reallocation_backtest(d.s, d.val, d.lval, d.k, d.n, 0.1, 1.0, 1.0)
    assert r["late_value_avoided"] == pytest.approx(50.0)
    assert exposure.reallocation_backtest(d.s, d.val, d.lval, d.k, d.n, 0.1, 0.5, 0.5)["late_value_avoided"] == pytest.approx(12.5)


def test_reallocation_never_negative_when_top_is_better():
    d = pd.DataFrame({"s": np.arange(10.0), "val": 100.0, "lval": [50.0] + [0.0] * 8 + [0.0], "k": 1, "n": 10})
    assert exposure.reallocation_backtest(d.s, d.val, d.lval, d.k, d.n, 0.1, 1.0, 1.0)["late_value_avoided"] == 0


def test_scenario_grid_shape_and_monotonic_in_improvement():
    d = pd.DataFrame({"next_value": 100.0, "next_late_value": [0.0] * 8 + [40.0, 60.0], "next_n_late": 1, "next_n_lines": 10})
    g = exposure.scenario_grid(np.arange(10.0), d, [0.1, 0.2], [0.5, 1.0], [0.25, 1.0])
    assert len(g) == 8
    a = g[(g.top_share == 0.2) & (g.realloc_frac == 1.0)].sort_values("improvement_ASSUMED")
    assert a.late_value_avoided.is_monotonic_increasing


def test_vendor_exposure_columns_use_assumed_params():
    df = pd.DataFrame({"value_usd": [1000.0], "late_rate_shrunk": [0.1], "late_value_usd": [100.0], "mean_days_when_late": [73.0],
                       "mean_days_late_shrunk": [3.65]})
    p = {"late_cost_pct_of_value": 0.05, "carrying_rate_annual": 0.10}
    e = exposure.vendor_exposure(df, p).iloc[0]
    assert e.value_at_risk == pytest.approx(100)
    assert e.wc_excess_delay == pytest.approx(100 * 73 / 365)
    assert e.late_cost_ASSUMED == pytest.approx(5) and e.carrying_cost_ASSUMED == pytest.approx(1000 * 3.65 / 365 * 0.1)
    assert e.total_exposure_ASSUMED == pytest.approx(5 + 0.1 * 10)


def test_concentration_is_lorenz_like():
    c = exposure.concentration([1, 1, 1, 97])
    assert c.cum_share.iloc[0] == pytest.approx(0.97) and c.cum_share.iloc[-1] == pytest.approx(1.0)


def test_cluster_select_k_and_names():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.3, (60, 2)) for m in ([0, 0], [4, 4], [0, 4])])
    k, tab = clustering.select_k(X, [2, 3, 4])
    assert k == 3 and list(tab.k) == [2, 3, 4]
    z = pd.DataFrame({"late_rate_shrunk": [2.0, -0.5], "mean_days_late_shrunk": [1.0, -0.3], "freight_ratio": [0, 0],
                      "log_value": [0.0, 0.0], "country_hhi": [0, 0], "air_share": [0, 1.0]})
    names = clustering.name_clusters(z)
    assert len(set(names.values())) == 2


def test_archetypes_deterministic(con, cfg):
    feat = con.execute("SELECT * FROM vendor_summary WHERE NOT is_internal_rdc AND n_lines >= 5 ORDER BY vendor").df()
    feat["late_rate_shrunk"] = feat.n_late / feat.n_lines
    feat["mean_days_late_shrunk"] = feat.sum_days_late_capped / feat.n_lines
    feat["freight_ratio"] = (feat.freight_usd / feat.freight_base_value_usd.replace(0, np.nan)).fillna(0)
    feat["log_value"] = np.log1p(feat.value_usd)
    f = cfg["clustering"]["features"]
    a = clustering.fit_archetypes(feat, f, cfg["clustering"]["k_range"])
    b = clustering.fit_archetypes(feat.sample(frac=1, random_state=3), f, cfg["clustering"]["k_range"])
    assert (a[0] == b[0]).all() and a[1] == b[1]


def test_oot_split_returns_all_scores_and_baselines(con, cfg):
    ev, meta = pipeline.oot_split(con, cfg, "2012-01-01", "2016-01-01")
    assert {"Composite risk score (shrunk)", "Baseline: raw late rate", "Baseline: volume only (smaller = riskier)",
            "Baseline: lane mix only (country / mode)"} <= set(ev.score)
    assert ev.line_auc.dropna().between(0, 1).all()
    assert meta["n_vendors_evaluated"] <= meta["n_vendors_scored"] and 0 < meta["next_lines_covered_share"] <= 1
    ev2, meta2 = pipeline.oot_split(con, cfg, "2012-01-01", "2016-01-01")
    pd.testing.assert_frame_equal(ev, ev2)
