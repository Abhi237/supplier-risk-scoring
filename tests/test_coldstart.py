"""Cold-start policy: thin vendors get the EB prior mean, a 'New / unscored' tier and an honest OOT check."""
import numpy as np
import pandas as pd
import pytest

from supplier_risk import coldstart, pipeline
from supplier_risk.shrinkage import BetaPrior, posterior_mean

PRI = {"late": BetaPrior(2, 28), "severe": BetaPrior(1, 29)}


def test_cold_start_assigns_prior_mean_and_tier():
    cs = coldstart.cold_start_scores(["a", "b"], PRI, 1.3)
    assert list(cs.late_rate_shrunk) == pytest.approx([2 / 30] * 2)
    assert list(cs.mean_days_late_shrunk) == pytest.approx([1.3] * 2)
    assert (cs.risk_tier == coldstart.NEW_TIER).all() and cs.risk_score.isna().all()
    assert posterior_mean([0], [0], PRI["late"])[0] == pytest.approx(cs.late_rate_shrunk[0])


def test_jeffreys_interval_contains_rate():
    lo, hi = coldstart.jeffreys(10, 100)
    assert lo < 0.1 < hi and 0 < lo and hi < 1
    assert coldstart.jeffreys(0, 50)[0] == 0.0 and coldstart.jeffreys(50, 50)[1] == 1.0


def test_evaluate_cold_start_separates_new_and_scored():
    scored = pd.DataFrame({"vendor": ["s1", "s2"], "late_rate_shrunk": [0.05, 0.3]})
    hist = pd.DataFrame({"vendor": ["s1", "s2", "n2"]})
    out = pd.DataFrame({"vendor": ["s1", "s2", "n1", "n2"], "next_n_lines": [100, 100, 50, 50], "next_n_late": [5, 30, 5, 5],
                        "next_n_severe": [1, 5, 1, 1], "next_value": [1000.0, 1000, 500, 500]})
    r = coldstart.evaluate_cold_start(out, scored, hist, PRI, 1.0)
    assert r["n_new_vendors"] == 2 and r["n_new_no_history"] == 1 and r["n_scored_vendors"] == 2
    assert r["new_share_next_lines"] == pytest.approx(100 / 300)
    assert r["late"]["new_rate"] == pytest.approx(0.1) and r["late"]["scored_rate"] == pytest.approx(35 / 200)
    assert r["late"]["prior_inside_ci"] is True
    assert 0 <= r["full_coverage_late_auc"] <= 1


def test_new_vendors_are_exactly_those_below_threshold_or_unseen(con, cfg):
    _, meta = pipeline.oot_split(con, cfg, "2012-01-01", "2016-01-01")
    feat = con.execute("SELECT vendor, n_lines FROM oot_features").df()
    scored = set(feat[feat.n_lines >= cfg["scoring"]["min_lines_for_score"]].vendor)
    post = con.execute("SELECT vendor FROM oot_outcomes WHERE next_n_lines > 0").df().vendor
    assert meta["cold_start"]["n_new_vendors"] == int(np.sum(~post.isin(scored)))
    assert meta["cold_start"]["n_scored_vendors"] == int(np.sum(post.isin(scored)))
