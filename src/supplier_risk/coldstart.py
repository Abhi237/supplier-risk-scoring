"""Cold-start policy for new / thin-history vendors.

A vendor with fewer than `min_lines_for_score` resolved lines at the score date has too little evidence of its own.
Policy:
* assign every rate component its empirical-Bayes PRIOR MEAN (the programme-wide average implied by the fitted Beta
  prior), i.e. the posterior with n = 0;
* do NOT publish a composite percentile score (it would be an artefact of the prior); label the tier
  "New / unscored" and review once the vendor reaches `min_lines_for_score` lines.
Out-of-time, `evaluate_cold_start` checks whether the prior is a fair default by comparing the next-period outcomes of
vendors that were unscored at the score date with the prior mean (Jeffreys 95% interval on the pooled rate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .shrinkage import BetaPrior
from .validation import line_level_auc

NEW_TIER = "New / unscored"


def cold_start_scores(vendors, priors: dict[str, BetaPrior], pop_days: float) -> pd.DataFrame:
    ids = list(vendors)
    return pd.DataFrame({"vendor": ids, "n_lines": 0, "late_rate_shrunk": priors["late"].mean,
                         "severe_late_rate_shrunk": priors["severe"].mean, "mean_days_late_shrunk": pop_days,
                         "risk_score": np.nan, "risk_tier": NEW_TIER})


def jeffreys(k: float, n: float, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return (float(stats.beta.ppf(a, k + 0.5, n - k + 0.5)) if k > 0 else 0.0,
            float(stats.beta.ppf(1 - a, k + 0.5, n - k + 0.5)) if k < n else 1.0)


def evaluate_cold_start(outcomes: pd.DataFrame, scored: pd.DataFrame, history: pd.DataFrame,
                        priors: dict[str, BetaPrior], pop_days: float) -> dict:
    """outcomes: oot_outcomes (next period); scored: vendors with a composite score at the score date;
    history: all vendors with any line before the score date (n_lines > 0)."""
    o = outcomes[outcomes.next_n_lines > 0]
    new = o[~o.vendor.isin(scored.vendor)]
    old = o[o.vendor.isin(scored.vendor)]
    no_hist = new[~new.vendor.isin(history.vendor)]
    out = {"n_new_vendors": int(len(new)), "n_new_no_history": int(len(no_hist)), "n_scored_vendors": int(len(old)),
           "new_share_next_lines": float(new.next_n_lines.sum() / o.next_n_lines.sum()),
           "new_share_next_value": float(new.next_value.sum() / o.next_value.sum())}
    for key, k, pr in [("late", "next_n_late", priors["late"]), ("severe", "next_n_severe", priors["severe"])]:
        kn, nn = float(new[k].sum()), float(new.next_n_lines.sum())
        ko, no = float(old[k].sum()), float(old.next_n_lines.sum())
        lo, hi = jeffreys(kn, nn) if nn else (float("nan"), float("nan"))
        out[key] = {"prior_mean": pr.mean, "new_rate": kn / nn if nn else float("nan"), "new_lo95": lo, "new_hi95": hi,
                    "new_lines": nn, "new_events": kn, "scored_rate": ko / no if no else float("nan"),
                    "prior_inside_ci": bool(lo <= pr.mean <= hi) if nn else False}
    cs = cold_start_scores(new.vendor, priors, pop_days)
    sc = pd.concat([scored[["vendor", "late_rate_shrunk"]], cs[["vendor", "late_rate_shrunk"]]])
    m = o.merge(sc, on="vendor", how="inner")
    out["full_coverage_late_auc"] = line_level_auc(m.late_rate_shrunk, m.next_n_late, m.next_n_lines)
    s_only = m[m.vendor.isin(scored.vendor)]
    out["scored_only_late_auc"] = line_level_auc(s_only.late_rate_shrunk, s_only.next_n_late, s_only.next_n_lines)
    return out
