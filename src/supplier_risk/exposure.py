"""Value exposure proxies and the volume-reallocation scenario.

Observed quantities (line value, late value, days late, freight) come from the data.  Monetary cost layers use ASSUMED
parameters from config.yaml (exposure.*) and are labelled ASSUMED everywhere they are reported.  Values are USD line
item values as recorded in the dataset (nominal, not inflation-adjusted, 2006-2015).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def vendor_exposure(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    # observed / computed
    out["value_at_risk"] = out.value_usd * out.late_rate_shrunk                      # expected late value (shrunk rate)
    out["wc_excess_delay"] = out.late_value_usd * out.mean_days_when_late.fillna(0) / 365.0   # USD-years of capital
    # ASSUMED cost layers
    out["late_cost_ASSUMED"] = out.value_at_risk * p["late_cost_pct_of_value"]
    out["carrying_cost_ASSUMED"] = out.value_usd * out.mean_days_late_shrunk / 365.0 * p["carrying_rate_annual"]
    out["total_exposure_ASSUMED"] = out[["late_cost_ASSUMED", "carrying_cost_ASSUMED"]].sum(axis=1)
    return out


def reallocation_backtest(score, next_value, next_late_value, next_n_late, next_n_total,
                          top_share: float, realloc_frac: float, improvement: float) -> dict[str, float]:
    """Counterfactual on the NEXT period: move `realloc_frac` of the volume of the top `top_share` vendors
    (ranked by a score built on the PREVIOUS period) to the remaining vendors.

    ASSUMPTION (`improvement`): the share of the late-rate gap (top-group value late rate minus remaining vendors'
    value late rate) that is actually realised on the moved volume.  1.0 = moved volume performs exactly like the
    remaining vendors did; 0.5 = only half the gap is closed.  The scenario also ASSUMES the other vendors can supply the
    same products and countries, which is often not true for sole-source health commodities.
    """
    d = pd.DataFrame({"s": score, "val": next_value, "lval": next_late_value, "k": next_n_late,
                      "n": next_n_total}).query("n > 0").sort_values("s", ascending=False, kind="mergesort")
    n_top = max(1, int(round(top_share * len(d))))
    top, rest = d.head(n_top), d.iloc[n_top:]
    rest_rate = rest.lval.sum() / rest.val.sum()
    top_rate = top.lval.sum() / top.val.sum()
    moved = top.val.sum() * realloc_frac
    before = d.lval.sum()
    avoided = moved * max(top_rate - rest_rate, 0) * improvement
    return {"top_share": top_share, "realloc_frac": realloc_frac, "improvement_ASSUMED": improvement,
            "n_top_vendors": int(n_top), "top_value_share": float(top.val.sum() / d.val.sum()),
            "top_late_rate_value": float(top_rate), "rest_late_rate_value": float(rest_rate),
            "moved_value": float(moved), "late_value_before": float(before), "late_value_avoided": float(avoided),
            "late_value_after": float(before - avoided), "late_value_reduction_pct": float(avoided / before)}


def scenario_grid(score, next_df: pd.DataFrame, shares, fracs, improvements) -> pd.DataFrame:
    rows = [reallocation_backtest(score, next_df.next_value, next_df.next_late_value, next_df.next_n_late,
                                  next_df.next_n_lines, s, f, i)
            for s, f, i in itertools.product(shares, fracs, improvements)]
    return pd.DataFrame(rows)


def exposure_sensitivity(df: pd.DataFrame, base: dict, grid: dict) -> pd.DataFrame:
    rows = []
    sc = df[df.is_scored]
    for lc, cr in itertools.product(grid["late_cost_pct_of_value"], grid["carrying_rate_annual"]):
        p = {**base, "late_cost_pct_of_value": lc, "carrying_rate_annual": cr}
        e = vendor_exposure(sc, p)
        top = e.nlargest(max(1, int(round(0.1 * len(e)))), "risk_score")
        rows.append({"late_cost_pct_of_value": lc, "carrying_rate_annual": cr,
                     "total_exposure_ASSUMED": float(e.total_exposure_ASSUMED.sum()),
                     "top_decile_risk_exposure_ASSUMED": float(top.total_exposure_ASSUMED.sum()),
                     "top_decile_share": float(top.total_exposure_ASSUMED.sum() / e.total_exposure_ASSUMED.sum())})
    return pd.DataFrame(rows)


def concentration(values) -> pd.DataFrame:
    v = np.sort(np.asarray(values, float))[::-1]
    c = np.cumsum(v) / v.sum()
    return pd.DataFrame({"vendor_share": np.arange(1, len(v) + 1) / len(v), "cum_share": c})
