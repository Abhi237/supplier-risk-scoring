"""Working-capital / exposure proxies and the volume-reallocation scenario.

Observed quantities (GMV, late GMV, freight, delivery days) come from the data. Monetary costs use ASSUMED
parameters from config.yaml (exposure.*) and are labelled ASSUMED everywhere they are reported.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def seller_exposure(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    # observed / computed
    out["gmv_at_risk"] = out.gmv * out.late_rate_shrunk                       # expected late GMV (shrunk rate)
    out["wc_in_transit"] = out.gmv * out.median_delivery_days.fillna(0) / 365.0   # avg GMV float in transit
    out["wc_excess_delay"] = out.late_gmv * out.mean_delay_days_when_late.fillna(0) / 365.0
    # ASSUMED cost parameters
    out["late_cost_ASSUMED"] = out.gmv_at_risk * p["late_cost_pct_of_gmv"]
    out["freight_exposure_ASSUMED"] = out.freight * p["freight_absorb_pct"]
    out["cancel_cost_ASSUMED"] = out.gmv * out.cancel_rate * p["cancel_cost_pct_of_gmv"]
    out["total_exposure_ASSUMED"] = out[["late_cost_ASSUMED", "freight_exposure_ASSUMED",
                                         "cancel_cost_ASSUMED"]].sum(axis=1)
    return out


def reallocation_backtest(score, next_gmv, next_late_gmv, next_n_late, next_n_delivered,
                          top_share: float, realloc_frac: float, improvement: float) -> dict[str, float]:
    """Counterfactual on the NEXT period: move `realloc_frac` of the volume of the top `top_share` sellers
    (ranked by score built on the PREVIOUS period) to the remaining sellers.

    ASSUMPTION (`improvement`): the share of the late-rate gap (top-group GMV late rate minus remaining
    sellers' GMV late rate) that is actually realised on the moved volume. 1.0 = moved volume performs exactly
    like the remaining sellers did; 0.5 = only half the gap is closed (capacity strain, lane mismatch).
    """
    d = pd.DataFrame({"s": score, "gmv": next_gmv, "lgmv": next_late_gmv, "k": next_n_late,
                      "n": next_n_delivered}).query("n > 0").sort_values("s", ascending=False, kind="mergesort")
    n_top = max(1, int(round(top_share * len(d))))
    top, rest = d.head(n_top), d.iloc[n_top:]
    rest_rate_gmv = rest.lgmv.sum() / rest.gmv.sum()
    top_rate_gmv = top.lgmv.sum() / top.gmv.sum()
    moved_gmv = top.gmv.sum() * realloc_frac
    late_gmv_before = d.lgmv.sum()
    avoided = moved_gmv * max(top_rate_gmv - rest_rate_gmv, 0) * improvement
    return {"top_share": top_share, "realloc_frac": realloc_frac, "improvement_ASSUMED": improvement,
            "n_top_sellers": int(n_top), "top_gmv_share": float(top.gmv.sum() / d.gmv.sum()),
            "top_late_rate_gmv": float(top_rate_gmv), "rest_late_rate_gmv": float(rest_rate_gmv),
            "moved_gmv": float(moved_gmv), "late_gmv_before": float(late_gmv_before),
            "late_gmv_avoided": float(avoided), "late_gmv_after": float(late_gmv_before - avoided),
            "late_gmv_reduction_pct": float(avoided / late_gmv_before)}


def scenario_grid(score, next_df: pd.DataFrame, shares, fracs, improvements) -> pd.DataFrame:
    rows = [reallocation_backtest(score, next_df.next_gmv, next_df.next_late_gmv, next_df.next_n_late,
                                  next_df.next_n_delivered, s, f, i)
            for s, f, i in itertools.product(shares, fracs, improvements)]
    return pd.DataFrame(rows)


def exposure_sensitivity(df: pd.DataFrame, base: dict, grid: dict) -> pd.DataFrame:
    rows = []
    for lc, fa in itertools.product(grid["late_cost_pct_of_gmv"], grid["freight_absorb_pct"]):
        p = {**base, "late_cost_pct_of_gmv": lc, "freight_absorb_pct": fa}
        e = seller_exposure(df, p)
        top = e.nlargest(max(1, int(round(0.1 * len(e)))), "risk_score")
        rows.append({"late_cost_pct_of_gmv": lc, "freight_absorb_pct": fa,
                     "total_exposure_ASSUMED": float(e.total_exposure_ASSUMED.sum()),
                     "top_decile_risk_exposure_ASSUMED": float(top.total_exposure_ASSUMED.sum()),
                     "top_decile_share": float(top.total_exposure_ASSUMED.sum() / e.total_exposure_ASSUMED.sum())})
    return pd.DataFrame(rows)


def concentration(values) -> pd.DataFrame:
    v = np.sort(np.asarray(values, float))[::-1]
    c = np.cumsum(v) / v.sum()
    return pd.DataFrame({"seller_share": np.arange(1, len(v) + 1) / len(v), "cum_share": c})
