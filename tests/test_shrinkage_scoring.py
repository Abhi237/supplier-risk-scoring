"""Unit tests: beta-binomial empirical Bayes, composite score, bootstrap intervals."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from supplier_risk import scoring
from supplier_risk.shrinkage import BetaPrior, fit_beta_prior, posterior_interval, posterior_mean, shrink_mean


def test_posterior_mean_between_raw_and_prior():
    p = BetaPrior(2, 18)                         # mean 0.10
    m = posterior_mean([5], [10], p)[0]          # raw 0.5
    assert p.mean < m < 0.5
    assert m == pytest.approx(7 / 30)


def test_zero_trials_returns_prior_and_large_n_returns_raw():
    p = BetaPrior(2, 18)
    assert posterior_mean([0], [0], p)[0] == pytest.approx(0.1)
    assert posterior_mean([30000], [100000], p)[0] == pytest.approx(0.3, abs=1e-3)


def test_interval_contains_mean_and_narrows_with_n():
    p = BetaPrior(2, 18)
    lo1, hi1 = posterior_interval([1], [10], p)
    lo2, hi2 = posterior_interval([100], [1000], p)
    assert lo1[0] < posterior_mean([1], [10], p)[0] < hi1[0]
    assert (hi2 - lo2)[0] < (hi1 - lo1)[0]


def test_fit_beta_prior_recovers_truth():
    rng = np.random.default_rng(1)
    a, b = 2.0, 18.0
    n = rng.integers(20, 200, 4000)
    k = rng.binomial(n, rng.beta(a, b, len(n)))
    fit = fit_beta_prior(k, n)
    assert fit.mean == pytest.approx(a / (a + b), abs=0.01)
    assert 10 < fit.strength < 40


def test_fit_beta_prior_fallback_on_degenerate():
    fb = BetaPrior(1, 9)
    assert fit_beta_prior([0, 0, 0], [5, 5, 5], fb) == fb


def test_shrink_mean():
    assert shrink_mean([0], [0], 4.0, 5)[0] == pytest.approx(4.0)
    assert shrink_mean([5], [1], 4.0, 5)[0] == pytest.approx(25 / 6)


def _toy(n=60, seed=3):
    rng = np.random.default_rng(seed)
    ni = rng.integers(1, 300, n)
    nd = ni
    df = pd.DataFrame({"seller_id": [f"s{i}" for i in range(n)], "n_items": ni, "n_delivered": nd,
                       "n_late": rng.binomial(nd, 0.08), "n_handoff_obs": nd, "n_handoff_late": rng.binomial(nd, 0.1),
                       "n_canceled": rng.binomial(ni, 0.005), "n_reviews": nd,
                       "mean_review_score": rng.uniform(3, 5, n), "freight_to_price": rng.uniform(0.05, 0.6, n)})
    rc = pd.DataFrame({s: np.r_[rng.multinomial(int(r), [0.1, 0.05, 0.1, 0.25, 0.5])]
                       for s, r in zip(df.seller_id, df.n_reviews)}, index=[1, 2, 3, 4, 5]).T
    return df, rc


def test_composite_bounds_and_monotonic(cfg):
    df, _ = _toy()
    pri = scoring.fit_priors(df, cfg)
    c = scoring.score_components(df, pri)
    s = scoring.composite(c, cfg["scoring"]["weights"])
    assert s.min() >= 0 and s.max() <= 100
    worse = c.copy()
    worse.loc[0, "late_rate_shrunk"] = 0.99
    assert scoring.composite(worse, cfg["scoring"]["weights"], scoring.percentile_ref(c))[0] >= s[0]


def test_weights_sum_to_one(cfg):
    assert sum(cfg["scoring"]["weights"].values()) == pytest.approx(1.0)
    for w in cfg["scoring"]["weight_sensitivity"].values():
        assert sum(w.values()) == pytest.approx(1.0)
        assert set(w) == set(scoring.COMPONENTS)


def test_bootstrap_intervals_wider_for_small_sellers(cfg):
    df, rc = _toy()
    pri = {**scoring.fit_priors(df, cfg), "late": BetaPrior(2, 18)}   # weak prior so data dominate for big n
    c = scoring.score_components(df, pri)
    ref = scoring.percentile_ref(c)
    c["risk_score"] = scoring.composite(c, cfg["scoring"]["weights"], ref)
    b = scoring.bootstrap_scores(c, rc, pri, cfg["scoring"]["weights"], ref, 4.0, n_boot=300, seed=1)
    w = b.late_rate_boot_hi90 - b.late_rate_boot_lo90
    small, large = w[c.n_items < 30].median(), w[c.n_items > 150].median()
    assert small > large
    assert (b.late_rate_boot_lo90 <= c.late_rate_shrunk + 1e-9).mean() > 0.95
    # determinism
    b2 = scoring.bootstrap_scores(c, rc, pri, cfg["scoring"]["weights"], ref, 4.0, n_boot=300, seed=1)
    pd.testing.assert_frame_equal(b, b2)


def test_to_pct_midrank():
    ref = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    assert scoring.to_pct(np.array([2.5]), ref)[0] == pytest.approx(0.5)
    assert scoring.to_pct(np.array([0.0]), ref)[0] == 0.0
    assert stats.rankdata([1, 2])[0] == 1
