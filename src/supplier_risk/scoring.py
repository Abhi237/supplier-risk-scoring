"""Seller reliability scoring: shrunk components, composite risk score, bootstrap intervals.

Olist SELLERS are used as PROXIES for suppliers. Higher score = higher procurement risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shrinkage import BetaPrior, fit_beta_prior, posterior_interval, posterior_mean, shrink_mean

COMPONENTS = ["late_rate_shrunk", "seller_handoff_late_rate", "review_risk", "cancel_rate", "freight_ratio",
              "volume_uncertainty"]
REVIEW_PRIOR_N = 5.0
FREIGHT_CAP = 2.0


def fit_priors(df: pd.DataFrame, cfg: dict) -> dict[str, BetaPrior]:
    fb = BetaPrior(cfg["scoring"]["beta_prior_a"], cfg["scoring"]["beta_prior_b"])
    return {
        "late": fit_beta_prior(df.n_late, df.n_delivered, fb),
        "handoff": fit_beta_prior(df.n_handoff_late, df.n_handoff_obs, fb),
        "cancel": fit_beta_prior(df.n_canceled, df.n_items, BetaPrior(1.0, 99.0)),
    }


def score_components(df: pd.DataFrame, priors: dict[str, BetaPrior], review_mu: float | None = None) -> pd.DataFrame:
    """Add raw rates, shrunk rates (+90% credible interval for late rate) and the six risk components."""
    out = df.copy()
    for c in ["n_late", "n_delivered", "n_handoff_late", "n_handoff_obs", "n_canceled", "n_items", "n_reviews"]:
        out[c] = out[c].fillna(0).astype(float)
    out["late_rate_raw"] = out.n_late / out.n_delivered.replace(0, np.nan)
    out["late_rate_shrunk"] = posterior_mean(out.n_late, out.n_delivered, priors["late"])
    out["late_rate_lo90"], out["late_rate_hi90"] = posterior_interval(out.n_late, out.n_delivered, priors["late"])
    out["handoff_late_rate_raw"] = out.n_handoff_late / out.n_handoff_obs.replace(0, np.nan)
    out["seller_handoff_late_rate"] = posterior_mean(out.n_handoff_late, out.n_handoff_obs, priors["handoff"])
    out["cancel_rate_raw"] = out.n_canceled / out.n_items.replace(0, np.nan)
    out["cancel_rate"] = posterior_mean(out.n_canceled, out.n_items, priors["cancel"])
    if review_mu is None:
        tot = (out.mean_review_score.fillna(0) * out.n_reviews).sum()
        review_mu = float(tot / max(out.n_reviews.sum(), 1))
    rev_sum = out.mean_review_score.fillna(0) * out.n_reviews
    out["review_score_shrunk"] = shrink_mean(rev_sum, out.n_reviews, review_mu, REVIEW_PRIOR_N)
    out["review_risk"] = 1 - out.review_score_shrunk / 5.0
    out["freight_ratio"] = out.freight_to_price.fillna(out.freight_to_price.median()).clip(0, FREIGHT_CAP)
    out["volume_uncertainty"] = 1.0 / (1.0 + np.log1p(out.n_items))
    return out


def percentile_ref(ref: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted reference distributions used to map component values to [0, 1] percentiles."""
    return {c: np.sort(ref[c].to_numpy(float)) for c in COMPONENTS}


def to_pct(values: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    """Empirical CDF (mid-rank) of values against a sorted reference; works on any-shaped arrays."""
    lo = np.searchsorted(sorted_ref, values, side="left")
    hi = np.searchsorted(sorted_ref, values, side="right")
    return (lo + hi) / (2.0 * len(sorted_ref))


def composite(df: pd.DataFrame, weights: dict[str, float], ref: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Weighted sum of component percentiles, scaled 0-100 (higher = riskier)."""
    ref = ref or percentile_ref(df)
    w = np.array([weights[c] for c in COMPONENTS], float)
    w = w / w.sum()
    p = np.column_stack([to_pct(df[c].to_numpy(float), ref[c]) for c in COMPONENTS])
    return 100.0 * p @ w


def bootstrap_scores(df: pd.DataFrame, review_counts: pd.DataFrame, priors: dict[str, BetaPrior],
                     weights: dict[str, float], ref: dict[str, np.ndarray], review_mu: float,
                     n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Bayesian bootstrap (Rubin, 1981) of each seller's outcomes, with the empirical-Bayes prior as pseudo-data.

    Dirichlet-weighting a seller's n binary outcomes plus the prior pseudo-counts gives a resampled rate
    ~ Beta(a + k, b + n - k); review-score shares are resampled ~ Dirichlet(counts + prior pseudo-counts).
    Unlike the classical bootstrap this does not collapse to zero width for sellers with 0 late items, so
    small sellers correctly get WIDER intervals. Freight ratio and volume are held fixed; the percentile
    reference is fixed so the interval reflects the seller's own uncertainty. Returns 5th/95th percentiles.
    """
    rng = np.random.default_rng(seed)
    S, B = len(df), n_boot

    def rate_draws(k, n, prior: BetaPrior):
        k, n = np.asarray(k, float), np.asarray(n, float)
        return rng.beta((prior.a + k)[:, None], (prior.b + n - k)[:, None], size=(S, B))

    late_b = rate_draws(df.n_late, df.n_delivered, priors["late"])
    ho_b = rate_draws(df.n_handoff_late, df.n_handoff_obs, priors["handoff"])
    ca_b = rate_draws(df.n_canceled, df.n_items, priors["cancel"])
    rc = review_counts.reindex(df.seller_id).fillna(0)[[1, 2, 3, 4, 5]].to_numpy(float)
    pool = review_counts[[1, 2, 3, 4, 5]].to_numpy(float).sum(0)
    alpha = rc + REVIEW_PRIOR_N * pool / pool.sum()
    g = rng.gamma(alpha[:, None, :], 1.0, size=(S, B, 5))
    shares = g / g.sum(-1, keepdims=True)
    rev_b = 1 - (shares @ np.arange(1, 6)) / 5.0
    comp = {"late_rate_shrunk": late_b, "seller_handoff_late_rate": ho_b, "review_risk": rev_b,
            "cancel_rate": ca_b,
            "freight_ratio": np.repeat(df.freight_ratio.to_numpy(float)[:, None], B, 1),
            "volume_uncertainty": np.repeat(df.volume_uncertainty.to_numpy(float)[:, None], B, 1)}
    w = np.array([weights[c] for c in COMPONENTS], float)
    w = w / w.sum()
    score_b = sum(100.0 * wi * to_pct(comp[c], ref[c]) for wi, c in zip(w, COMPONENTS))
    return pd.DataFrame({"seller_id": df.seller_id.to_numpy(),
                         "risk_score_lo90": np.percentile(score_b, 5, axis=1),
                         "risk_score_hi90": np.percentile(score_b, 95, axis=1),
                         "late_rate_boot_lo90": np.percentile(late_b, 5, axis=1),
                         "late_rate_boot_hi90": np.percentile(late_b, 95, axis=1)})


def risk_tier(score: pd.Series) -> pd.Series:
    q = score.rank(pct=True)
    return pd.cut(q, [0, 0.5, 0.8, 0.9, 1.0], labels=["Low", "Moderate", "Elevated", "High"],
                  include_lowest=True).astype(str)
