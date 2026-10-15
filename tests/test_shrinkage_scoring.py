"""Unit tests: beta-binomial empirical Bayes, composite score, Bayesian-bootstrap intervals."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from supplier_risk import pipeline, scoring
from supplier_risk.shrinkage import BetaPrior, fit_beta_prior, posterior_interval, posterior_mean, shrink_mean


def test_posterior_mean_between_raw_and_prior():
    p = BetaPrior(2, 18)
    m = posterior_mean([5], [10], p)[0]
    assert p.mean < m < 0.5 and m == pytest.approx(7 / 30)


def test_zero_trials_returns_prior_and_large_n_returns_raw():
    p = BetaPrior(2, 18)
    assert posterior_mean([0], [0], p)[0] == pytest.approx(0.1)
    assert posterior_mean([30000], [100000], p)[0] == pytest.approx(0.3, abs=1e-3)


def test_zero_late_vendor_is_not_shrunk_to_exactly_zero():
    assert posterior_mean([0], [20], BetaPrior(0.7, 21))[0] > 0


def test_interval_contains_mean_and_narrows_with_n():
    p = BetaPrior(2, 18)
    lo1, hi1 = posterior_interval([1], [10], p)
    lo2, hi2 = posterior_interval([100], [1000], p)
    assert lo1[0] < posterior_mean([1], [10], p)[0] < hi1[0]
    assert (hi2 - lo2)[0] < (hi1 - lo1)[0]


def test_fit_beta_prior_recovers_truth():
    rng = np.random.default_rng(1)
    n = rng.integers(20, 200, 4000)
    k = rng.binomial(n, rng.beta(2.0, 18.0, len(n)))
    fit = fit_beta_prior(k, n)
    assert fit.mean == pytest.approx(0.1, abs=0.01) and 10 < fit.strength < 40


def test_fit_beta_prior_fallback_on_degenerate():
    fb = BetaPrior(1, 9)
    assert fit_beta_prior([0, 0, 0], [5, 5, 5], fb) == fb


def test_shrink_mean():
    assert shrink_mean([0], [0], 4.0, 5)[0] == pytest.approx(4.0)
    assert shrink_mean([5], [1], 4.0, 5)[0] == pytest.approx(25 / 6)


@pytest.fixture(scope="module")
def scored(con, cfg):
    feat = con.execute("SELECT * FROM vendor_summary WHERE NOT is_internal_rdc ORDER BY vendor").df()
    lane = con.execute("SELECT * FROM vendor_lane ORDER BY vendor, country, ship_mode").df()
    comp, priors, ref = pipeline.score_vendors(feat, lane, cfg)
    return comp, priors, ref, feat


def test_composite_bounds(scored):
    comp = scored[0]
    s = comp[comp.is_scored].risk_score
    assert len(s) > 5 and s.min() >= 0 and s.max() <= 100


def test_composite_monotonic_in_late_rate(scored, cfg):
    comp, _, ref, _ = scored
    sc = comp[comp.is_scored].reset_index(drop=True)
    worse = sc.copy()
    worse.loc[0, "late_rate_shrunk"] = 0.99
    assert scoring.composite(worse, cfg["scoring"]["weights"], ref)[0] >= scoring.composite(sc, cfg["scoring"]["weights"], ref)[0]


def test_weights_sum_to_one_and_cover_components(cfg):
    assert sum(cfg["scoring"]["weights"].values()) == pytest.approx(1.0)
    for w in cfg["scoring"]["weight_sensitivity"].values():
        assert sum(w.values()) == pytest.approx(1.0) and set(w) == set(scoring.COMPONENTS)


def test_shrunk_rate_sits_between_raw_and_prior(scored):
    comp, priors, _, _ = scored
    c = comp[(comp.n_lines > 0) & comp.late_rate_raw.notna()]
    lo = np.minimum(c.late_rate_raw, priors["late"].mean) - 1e-12
    hi = np.maximum(c.late_rate_raw, priors["late"].mean) + 1e-12
    assert ((c.late_rate_shrunk >= lo) & (c.late_rate_shrunk <= hi)).all()


def test_scoring_population_excludes_thin_vendors(scored, cfg):
    comp = scored[0]
    assert (comp.loc[comp.is_scored, "n_lines"] >= cfg["scoring"]["min_lines_for_score"]).all()
    assert (comp.loc[~comp.is_scored, "n_lines"] < cfg["scoring"]["min_lines_for_score"]).all()


def test_lane_mix_leaves_own_vendor_out():
    v = pd.DataFrame({"vendor": ["a", "b"], "country": ["X", "X"], "ship_mode": ["Air", "Air"], "n_lines": [100, 100], "n_late": [100, 0]})
    r = scoring.lane_mix_risk(v, pd.Series(["a", "b"]), prior_n=0.0)
    assert r["a"] == pytest.approx(0.0) and r["b"] == pytest.approx(1.0)    # a is judged by b's lateness and vice versa


def test_bootstrap_intervals_wider_for_thin_vendors(scored, cfg):
    comp, priors, ref, _ = scored
    sc = comp[comp.is_scored].reset_index(drop=True)
    days = {v: np.zeros(int(n)) for v, n in zip(sc.vendor, sc.n_lines)}
    b = scoring.bootstrap_scores(sc, days, priors, cfg["scoring"]["weights"], ref, comp.attrs["pop_days_late"],
                                 cfg["scoring"]["mean_days_prior_n"], n_boot=200, seed=1)
    w = (b.late_rate_boot_hi90 - b.late_rate_boot_lo90).to_numpy()
    order = np.argsort(sc.n_lines.to_numpy())
    assert w[order[:5]].mean() > w[order[-5:]].mean()
    b2 = scoring.bootstrap_scores(sc, days, priors, cfg["scoring"]["weights"], ref, comp.attrs["pop_days_late"],
                                  cfg["scoring"]["mean_days_prior_n"], n_boot=200, seed=1)
    pd.testing.assert_frame_equal(b, b2)


def test_to_pct_midrank():
    ref = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
    assert scoring.to_pct(np.array([2.5]), ref)[0] == pytest.approx(0.5)
    assert scoring.to_pct(np.array([0.0]), ref)[0] == 0.0
    assert stats.rankdata([1, 2])[0] == 1


def test_risk_tier_percentiles():
    t = scoring.risk_tier(pd.Series(np.arange(100.0)))
    assert t.iloc[-1] == "High" and t.iloc[0] == "Low" and set(t) <= {"High", "Elevated", "Moderate", "Low"}
