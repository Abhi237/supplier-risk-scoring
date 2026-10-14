"""Cold-start policy for new (unscored) sellers.

A seller with no sales before the score date has no evidence of its own. Policy:
* assign every rate component its empirical-Bayes PRIOR MEAN (the network average implied by the fitted Beta prior),
  i.e. the posterior with n = 0;
* do NOT compute a composite percentile score (it would be an artefact of the prior); label the tier
  "New / unscored" and review after the first `min_items_for_score` items.
Out-of-time, `evaluate_cold_start` checks whether the prior is a fair default by comparing new sellers'
next-period outcomes with the prior mean (Jeffreys 95% interval on the pooled new-seller rate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .shrinkage import BetaPrior
from .validation import item_level_auc

NEW_TIER = "New / unscored"


def cold_start_scores(seller_ids, priors: dict[str, BetaPrior], review_mu: float) -> pd.DataFrame:
    ids = list(seller_ids)
    return pd.DataFrame({
        "seller_id": ids,
        "n_items": 0, "n_delivered": 0,
        "late_rate_shrunk": priors["late"].mean, "seller_handoff_late_rate": priors["handoff"].mean,
        "cancel_rate": priors["cancel"].mean, "review_score_shrunk": review_mu,
        "risk_score": np.nan, "risk_tier": NEW_TIER,
    })


def jeffreys(k: float, n: float, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return (float(stats.beta.ppf(a, k + 0.5, n - k + 0.5)) if k > 0 else 0.0,
            float(stats.beta.ppf(1 - a, k + 0.5, n - k + 0.5)) if k < n else 1.0)


def evaluate_cold_start(outcomes: pd.DataFrame, scored: pd.DataFrame, priors: dict[str, BetaPrior],
                        review_mu: float = 4.0) -> dict:
    """outcomes: oot_outcomes (next period); scored: score-period components incl. seller_handoff_late_rate."""
    o = outcomes[outcomes.next_n_delivered > 0]
    new = o[~o.seller_id.isin(scored.seller_id)]
    old = o[o.seller_id.isin(scored.seller_id)]
    out = {"n_new_sellers": int(len(new)), "n_scored_sellers": int(len(old)),
           "new_share_next_items": float(new.next_n_delivered.sum() / o.next_n_delivered.sum()),
           "new_share_next_gmv": float(new.next_gmv.sum() / o.next_gmv.sum())}
    for key, k, n, pr in [("handoff", "next_n_handoff_late", "next_n_handoff_obs", priors["handoff"]),
                          ("late", "next_n_late", "next_n_delivered", priors["late"])]:
        kn, nn = float(new[k].sum()), float(new[n].sum())
        ko, no = float(old[k].sum()), float(old[n].sum())
        lo, hi = jeffreys(kn, nn)
        out[key] = {"prior_mean": pr.mean, "new_rate": kn / nn if nn else float("nan"), "new_lo95": lo,
                    "new_hi95": hi, "new_items": nn, "scored_rate": ko / no if no else float("nan"),
                    "prior_inside_ci": bool(lo <= pr.mean <= hi)}
    # full-coverage evaluation: new sellers get the prior mean as their hand-off score
    cs = cold_start_scores(new.seller_id, priors, review_mu)
    sc = pd.concat([scored[["seller_id", "seller_handoff_late_rate"]], cs[["seller_id", "seller_handoff_late_rate"]]])
    m = o.merge(sc, on="seller_id", how="inner")
    out["full_coverage_handoff_auc"] = item_level_auc(m.seller_handoff_late_rate, m.next_n_handoff_late,
                                                      m.next_n_handoff_obs)
    out["scored_only_handoff_auc"] = item_level_auc(
        m[m.seller_id.isin(scored.seller_id)].seller_handoff_late_rate,
        m[m.seller_id.isin(scored.seller_id)].next_n_handoff_late,
        m[m.seller_id.isin(scored.seller_id)].next_n_handoff_obs)
    return out
