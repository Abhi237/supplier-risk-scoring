"""Empirical-Bayes shrinkage for small-volume sellers.

* Rates (late, hand-off late, cancel): beta-binomial model. The Beta(a, b) prior is fitted by maximum
  marginal likelihood across sellers, then each seller's posterior is Beta(a + k, b + n - k).
* Means (review score): normal-normal shrinkage toward the population mean with a pseudo-count prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class BetaPrior:
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def strength(self) -> float:
        """Prior pseudo-count a + b: a seller with n observations gets weight n / (n + a + b) on its own data."""
        return self.a + self.b


def betabinom_neg_loglik(log_ab: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    a, b = np.exp(log_ab)
    ll = (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
          + special.betaln(k + a, n - k + b) - special.betaln(a, b))
    return -float(np.sum(ll))


def fit_beta_prior(k, n, fallback: BetaPrior | None = None, min_n: int = 1) -> BetaPrior:
    """Maximum marginal-likelihood Beta prior from per-seller successes k out of n trials."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    m = n >= min_n
    k, n = k[m], n[m]
    if len(k) < 5 or k.sum() == 0 or k.sum() == n.sum():
        return fallback or BetaPrior(1.0, 1.0)
    p0 = np.clip(k.sum() / n.sum(), 1e-3, 1 - 1e-3)
    x0 = np.log([p0 * 10, (1 - p0) * 10])
    res = optimize.minimize(betabinom_neg_loglik, x0, args=(k, n), method="L-BFGS-B",
                            bounds=[(-6, 8), (-6, 10)])
    a, b = np.exp(res.x)
    return BetaPrior(float(a), float(b))


def posterior_mean(k, n, prior: BetaPrior) -> np.ndarray:
    k, n = np.asarray(k, float), np.asarray(n, float)
    return (prior.a + k) / (prior.a + prior.b + n)


def posterior_interval(k, n, prior: BetaPrior, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    lo = (1 - level) / 2
    a_post, b_post = prior.a + k, prior.b + n - k
    return stats.beta.ppf(lo, a_post, b_post), stats.beta.ppf(1 - lo, a_post, b_post)


def shrink_mean(sum_x, n, prior_mean: float, prior_n: float) -> np.ndarray:
    """(sum_x + prior_n * prior_mean) / (n + prior_n); returns prior_mean when n == 0."""
    sum_x, n = np.asarray(sum_x, float), np.asarray(n, float)
    return (sum_x + prior_n * prior_mean) / (n + prior_n)
