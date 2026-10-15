"""Vendor delivery-risk scoring: shrunk components, composite risk score, Bayesian-bootstrap intervals.

Vendors are the real suppliers named in the USAID / PEPFAR SCMS Delivery History. Higher score = higher delivery risk.
On-time = delivered-to-client date <= scheduled delivery date. There is no vendor hand-off timestamp in the data, so a
late line may reflect the vendor, the freight forwarder or in-country handling; scores describe RECORDED DELIVERY
PERFORMANCE for this programme only, not vendor quality overall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shrinkage import BetaPrior, fit_beta_prior, posterior_interval, posterior_mean, shrink_mean

COMPONENTS = ["late_rate_shrunk", "severe_late_rate_shrunk", "mean_days_late_shrunk", "freight_ratio",
              "lane_mix_risk", "volume_uncertainty"]
LANE_PRIOR_N = 20.0      # pseudo-lines used when pooling country / mode late rates (smoothing constant)


def fit_priors(df: pd.DataFrame, cfg: dict) -> dict[str, BetaPrior]:
    fb = BetaPrior(cfg["scoring"]["beta_prior_a"], cfg["scoring"]["beta_prior_b"])
    return {"late": fit_beta_prior(df.n_late, df.n_lines, fb),
            "severe": fit_beta_prior(df.n_severe_late, df.n_lines, fb)}


def lane_mix_risk(vlane: pd.DataFrame, vendors: pd.Series, prior_n: float = LANE_PRIOR_N) -> pd.Series:
    """Context component: how hard is the vendor's lane mix?  For each vendor, the line-weighted average of the pooled
    late rate of its destination countries and of its shipment modes, with the vendor's OWN lines left out of the
    pooled rate (leave-one-vendor-out) so the component is not mechanically correlated with its own lateness.
    Each pooled rate is smoothed toward the overall rate with `prior_n` pseudo-lines.  Averages the country and mode
    views.  Vendors with no lines get the overall rate."""
    p0 = float(vlane.n_late.sum() / max(vlane.n_lines.sum(), 1))
    parts = []
    for dim in ("country", "ship_mode"):
        tot = vlane.groupby(dim)[["n_lines", "n_late"]].sum()
        vg = vlane.groupby(["vendor", dim])[["n_lines", "n_late"]].sum().reset_index().join(tot, on=dim, rsuffix="_all")
        loo_n, loo_k = vg.n_lines_all - vg.n_lines, vg.n_late_all - vg.n_late
        vg["rate"] = (loo_k + prior_n * p0) / (loo_n + prior_n)
        parts.append(vg.groupby("vendor").apply(lambda g: float((g.rate * g.n_lines).sum() / g.n_lines.sum()),
                                                include_groups=False))
    out = (parts[0] + parts[1]) / 2
    return out.reindex(vendors).fillna(p0)


def score_components(df: pd.DataFrame, priors: dict[str, BetaPrior], vlane: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add raw rates, shrunk rates (+90% credible interval for the late rate) and the six risk components."""
    out = df.copy().sort_values("vendor", kind="mergesort").reset_index(drop=True)
    for c in ["n_lines", "n_late", "n_severe_late", "sum_days_late_capped", "freight_usd", "freight_base_value_usd",
              "value_usd"]:
        out[c] = out[c].fillna(0).astype(float)
    out["late_rate_raw"] = out.n_late / out.n_lines.replace(0, np.nan)
    out["late_rate_shrunk"] = posterior_mean(out.n_late, out.n_lines, priors["late"])
    out["late_rate_lo90"], out["late_rate_hi90"] = posterior_interval(out.n_late, out.n_lines, priors["late"])
    out["severe_late_rate_raw"] = out.n_severe_late / out.n_lines.replace(0, np.nan)
    out["severe_late_rate_shrunk"] = posterior_mean(out.n_severe_late, out.n_lines, priors["severe"])
    pop_days = float(out.sum_days_late_capped.sum() / max(out.n_lines.sum(), 1))
    out["mean_days_late_raw"] = out.sum_days_late_capped / out.n_lines.replace(0, np.nan)
    out["mean_days_late_shrunk"] = shrink_mean(out.sum_days_late_capped, out.n_lines, pop_days,
                                               cfg["scoring"]["mean_days_prior_n"])
    pooled_ratio = float(out.freight_usd.sum() / max(out.freight_base_value_usd.sum(), 1))
    base = out.freight_base_value_usd[out.freight_base_value_usd > 0]
    prior_value = float(base.median()) if len(base) else 1.0
    out["freight_ratio_raw"] = out.freight_usd / out.freight_base_value_usd.replace(0, np.nan)
    out["freight_ratio"] = shrink_mean(out.freight_usd, out.freight_base_value_usd, pooled_ratio, prior_value)
    out["lane_mix_risk"] = lane_mix_risk(vlane, out.vendor).to_numpy()
    out["volume_uncertainty"] = 1.0 / (1.0 + np.log1p(out.n_lines))
    out.attrs.update({"pop_days_late": pop_days, "pooled_freight_ratio": pooled_ratio, "freight_prior_value": prior_value})
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


def bootstrap_scores(df: pd.DataFrame, days_by_vendor: dict[str, np.ndarray], priors: dict[str, BetaPrior],
                     weights: dict[str, float], ref: dict[str, np.ndarray], pop_days: float, prior_n: float,
                     n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    """Bayesian bootstrap (Rubin, 1981) of each vendor's outcomes with the empirical-Bayes prior as pseudo-data.

    Dirichlet-weighting a vendor's n binary outcomes plus the prior pseudo-counts gives a resampled rate
    ~ Beta(a + k, b + n - k); the mean days late is resampled by Dirichlet weights over the vendor's own lines plus one
    prior pseudo-line at the population mean carrying `prior_n` weight.  Unlike the classical bootstrap this does not
    collapse to zero width for vendors with 0 late lines, so thin vendors correctly get WIDER intervals.  Freight ratio,
    lane mix and volume are held fixed and the percentile reference is fixed, so the interval reflects the vendor's own
    outcome uncertainty.  Returns 5th / 95th percentiles.  `df` must be the vendors to bootstrap, in a fixed order.
    """
    rng = np.random.default_rng(seed)
    S, B = len(df), n_boot

    def rate_draws(k, n, prior: BetaPrior):
        k, n = np.asarray(k, float), np.asarray(n, float)
        return rng.beta((prior.a + k)[:, None], (prior.b + n - k)[:, None], size=(S, B))

    late_b = rate_draws(df.n_late, df.n_lines, priors["late"])
    sev_b = rate_draws(df.n_severe_late, df.n_lines, priors["severe"])
    days_b = np.empty((S, B))
    for i, v in enumerate(df.vendor):
        x = np.asarray(days_by_vendor.get(v, np.array([])), float)
        g = rng.gamma(np.r_[np.ones(len(x)), prior_n], 1.0, size=(B, len(x) + 1))
        w = g / g.sum(1, keepdims=True)
        days_b[i] = w[:, :-1] @ x + w[:, -1] * pop_days
    comp = {"late_rate_shrunk": late_b, "severe_late_rate_shrunk": sev_b, "mean_days_late_shrunk": days_b,
            "freight_ratio": np.repeat(df.freight_ratio.to_numpy(float)[:, None], B, 1),
            "lane_mix_risk": np.repeat(df.lane_mix_risk.to_numpy(float)[:, None], B, 1),
            "volume_uncertainty": np.repeat(df.volume_uncertainty.to_numpy(float)[:, None], B, 1)}
    w = np.array([weights[c] for c in COMPONENTS], float)
    w = w / w.sum()
    score_b = sum(100.0 * wi * to_pct(comp[c], ref[c]) for wi, c in zip(w, COMPONENTS))
    return pd.DataFrame({"vendor": df.vendor.to_numpy(),
                         "risk_score_lo90": np.percentile(score_b, 5, axis=1),
                         "risk_score_hi90": np.percentile(score_b, 95, axis=1),
                         "late_rate_boot_lo90": np.percentile(late_b, 5, axis=1),
                         "late_rate_boot_hi90": np.percentile(late_b, 95, axis=1)})


def risk_tier(score: pd.Series) -> pd.Series:
    """Percentile tiers among SCORED vendors: top 10% High, 80-90 Elevated, 50-80 Moderate, below median Low."""
    q = score.rank(pct=True)
    return pd.cut(q, [0, 0.5, 0.8, 0.9, 1.0], labels=["Low", "Moderate", "Elevated", "High"],
                  include_lowest=True).astype(str)
