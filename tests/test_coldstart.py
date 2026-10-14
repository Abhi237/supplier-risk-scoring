"""Cold-start policy: new sellers get the EB prior mean, a 'New / unscored' tier, and an honest OOT check."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from supplier_risk import coldstart, pipeline
from supplier_risk.db import run_sql_model
from supplier_risk.shrinkage import BetaPrior, posterior_mean

PRI = {"late": BetaPrior(2, 28), "handoff": BetaPrior(1, 9), "cancel": BetaPrior(0.1, 9.9)}


def test_cold_start_assigns_prior_mean_and_tier():
    cs = coldstart.cold_start_scores(["a", "b"], PRI, 4.1)
    assert list(cs.late_rate_shrunk) == pytest.approx([2 / 30] * 2)
    assert list(cs.seller_handoff_late_rate) == pytest.approx([0.1] * 2)
    assert (cs.risk_tier == coldstart.NEW_TIER).all() and cs.risk_score.isna().all()
    # identical to the EB posterior with zero observations
    assert posterior_mean([0], [0], PRI["handoff"])[0] == pytest.approx(cs.seller_handoff_late_rate[0])


def test_jeffreys_interval_contains_rate():
    lo, hi = coldstart.jeffreys(10, 100)
    assert lo < 0.1 < hi and 0 < lo and hi < 1
    assert coldstart.jeffreys(0, 50)[0] == 0.0


def test_evaluate_cold_start_separates_new_and_scored():
    scored = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_handoff_late_rate": [0.05, 0.3]})
    out = pd.DataFrame({"seller_id": ["s1", "s2", "n1", "n2"], "next_n_delivered": [100, 100, 50, 50],
                        "next_n_late": [5, 30, 5, 5], "next_n_handoff_obs": [100, 100, 50, 50],
                        "next_n_handoff_late": [5, 30, 10, 10], "next_gmv": [1000.0, 1000, 500, 500]})
    r = coldstart.evaluate_cold_start(out, scored, PRI)
    assert r["n_new_sellers"] == 2 and r["n_scored_sellers"] == 2
    assert r["new_share_next_items"] == pytest.approx(100 / 300)
    assert r["handoff"]["new_rate"] == pytest.approx(0.2)
    assert r["handoff"]["scored_rate"] == pytest.approx(35 / 200)
    assert r["handoff"]["prior_inside_ci"] is False   # 0.10 vs observed 0.20 on 100 items
    assert 0 <= r["full_coverage_handoff_auc"] <= 1


def test_new_sellers_are_exactly_those_without_pre_cutoff_sales(sources, cfg):
    c = duckdb.connect(":memory:")
    run_sql_model(c, sources=sources, cfg=cfg, verbose=False, score_end="2018-01-01", validate_end="2018-09-01")
    _, meta = pipeline.oot_split(c, cfg, "2018-01-01", "2018-09-01")
    pre = set(c.execute("SELECT DISTINCT seller_id FROM fct_items WHERE purchase_ts < '2018-01-01'").df().seller_id)
    post = c.execute("SELECT seller_id FROM oot_outcomes WHERE next_n_delivered > 0").df().seller_id
    assert meta["cold_start"]["n_new_sellers"] == int(np.sum(~post.isin(pre)))
    c.close()
