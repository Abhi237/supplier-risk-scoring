"""Metric correctness for OOT validation, clustering helpers and the exposure/scenario logic."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from supplier_risk import clustering, exposure, validation


def test_item_auc_matches_expanded():
    rng = np.random.default_rng(0)
    s = rng.random(50)
    n = rng.integers(1, 20, 50)
    k = rng.binomial(n, 0.2 + 0.5 * s)
    y = np.concatenate([np.r_[np.ones(ki), np.zeros(ni - ki)] for ki, ni in zip(k, n)])
    x = np.concatenate([np.full(ni, si) for si, ni in zip(s, n)])
    assert validation.item_level_auc(s, k, n) == pytest.approx(roc_auc_score(y, x))


def test_lift_and_capture():
    s = np.arange(10)
    k = np.r_[np.zeros(9), 10]
    n = np.full(10, 10)
    r = validation.top_share_lift(s, k, n, 0.1)
    assert r["lift"] == pytest.approx(10.0)
    assert r["capture_late_items"] == pytest.approx(1.0)


def test_perfect_and_random_scores():
    n = np.full(200, 10)
    k = np.r_[np.zeros(100), np.full(100, 10)]
    assert validation.item_level_auc(np.r_[np.zeros(100), np.ones(100)], k, n) == pytest.approx(1.0)
    assert validation.item_level_auc(np.ones(200), k, n) == pytest.approx(0.5)


def test_bootstrap_delta_sign():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    n = np.full(300, 20)
    k = rng.binomial(n, p)
    r = validation.bootstrap_auc_delta(p, rng.random(300), k, n, n_boot=50, seed=0)
    assert r["delta"] > 0 and r["lo95"] > 0


def test_reallocation_zero_when_no_gap():
    d = pd.DataFrame({"s": np.arange(10.0), "gmv": 100.0, "lgmv": 10.0, "k": 1, "n": 10})
    r = exposure.reallocation_backtest(d.s, d.gmv, d.lgmv, d.k, d.n, 0.1, 1.0, 1.0)
    assert r["late_gmv_avoided"] == 0


def test_reallocation_full_gap():
    d = pd.DataFrame({"s": np.arange(10.0), "gmv": 100.0, "lgmv": [0.0] * 9 + [50.0], "k": 1, "n": 10})
    r = exposure.reallocation_backtest(d.s, d.gmv, d.lgmv, d.k, d.n, 0.1, 1.0, 1.0)
    assert r["late_gmv_avoided"] == pytest.approx(50.0)
    assert exposure.reallocation_backtest(d.s, d.gmv, d.lgmv, d.k, d.n, 0.1, 0.5, 0.5)["late_gmv_avoided"] == \
        pytest.approx(12.5)


def test_seller_exposure_columns():
    df = pd.DataFrame({"gmv": [1000.0], "late_rate_shrunk": [0.1], "median_delivery_days": [10.0],
                       "late_gmv": [100.0], "mean_delay_days_when_late": [7.3], "freight": [150.0],
                       "cancel_rate": [0.01]})
    p = {"late_cost_pct_of_gmv": 0.05, "freight_absorb_pct": 0.3, "cancel_cost_pct_of_gmv": 0.1}
    e = exposure.seller_exposure(df, p).iloc[0]
    assert e.gmv_at_risk == pytest.approx(100)
    assert e.wc_in_transit == pytest.approx(1000 * 10 / 365)
    assert e.total_exposure_ASSUMED == pytest.approx(5 + 45 + 1)


def test_cluster_names_unique_and_select_k():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.3, (60, 2)) for m in ([0, 0], [4, 4], [0, 4])])
    k, tab = clustering.select_k(X, [2, 3, 4])
    assert k == 3 and len(tab) == 3
    z = pd.DataFrame({"late_rate_shrunk": [1.0, -1.0, 0.1], "log_gmv": [0.1, 2.0, 1.5]})
    names = clustering.name_clusters(z)
    assert len(set(names.values())) == 3
