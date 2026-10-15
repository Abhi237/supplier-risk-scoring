"""Out-of-time validation metrics: does a vendor score built on period t predict lateness in period t+1?

Metrics are line-level (every next-period line inherits its vendor's score) and vendor-level; with few vendors the
vendor-level numbers are very noisy and the bootstrap intervals are wide - they are reported as they come."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def line_level_auc(score, n_late, n_total) -> float:
    """AUC of a vendor-level score for next-period LINE lateness (each line inherits its vendor's score).

    Implemented with sample weights: one positive row weighted by late lines and one negative row weighted
    by on-time lines per vendor, which is identical to expanding to line grain.
    """
    score, k, n = (np.asarray(x, float) for x in (score, n_late, n_total))
    m = n > 0
    s, k, n = score[m], k[m], n[m]
    y = np.r_[np.ones_like(s), np.zeros_like(s)]
    w = np.r_[k, n - k]
    keep = w > 0
    if len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], np.r_[s, s][keep], sample_weight=w[keep]))


def top_share_lift(score, n_late, n_total, share: float = 0.20) -> dict[str, float]:
    """Late-rate lift and late-line capture for the top `share` of vendors ranked by score."""
    df = pd.DataFrame({"s": score, "k": n_late, "n": n_total}).query("n > 0")
    df = df.sort_values("s", ascending=False, kind="mergesort")
    top = df.head(max(1, int(round(share * len(df)))))
    base = df.k.sum() / df.n.sum()
    rate = top.k.sum() / top.n.sum()
    return {"lift": float(rate / base), "late_rate_top": float(rate), "base_late_rate": float(base),
            "capture_late_lines": float(top.k.sum() / df.k.sum()), "share_lines": float(top.n.sum() / df.n.sum())}


def vendor_rank_corr(score, n_late, n_total, min_n: int = 10) -> float:
    s, k, n = (np.asarray(x, float) for x in (score, n_late, n_total))
    m = n >= min_n
    return float(stats.spearmanr(s[m], k[m] / n[m]).statistic)


def vendor_level_auc(score, n_late, n_total, min_n: int = 10) -> float:
    """AUC for flagging vendors whose next-period late rate exceeds the programme rate (vendors with >= min_n)."""
    s, k, n = (np.asarray(x, float) for x in (score, n_late, n_total))
    m = n >= min_n
    base = k[m].sum() / n[m].sum()
    y = (k[m] / n[m]) > base
    return float(roc_auc_score(y, s[m])) if 0 < y.sum() < len(y) else float("nan")


def evaluate(scores: dict[str, np.ndarray], n_late, n_total) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        lt = top_share_lift(s, n_late, n_total, 0.20)
        rows.append({"score": name, "line_auc": line_level_auc(s, n_late, n_total),
                     "vendor_auc": vendor_level_auc(s, n_late, n_total),
                     "spearman": vendor_rank_corr(s, n_late, n_total),
                     "lift_top20": lt["lift"], "capture_top20": lt["capture_late_lines"],
                     "late_rate_top20": lt["late_rate_top"], "base_late_rate": lt["base_late_rate"]})
    return pd.DataFrame(rows)


def bootstrap_auc_delta(s_a, s_b, n_late, n_total, n_boot: int = 200, seed: int = 42) -> dict[str, float]:
    """Vendor-cluster bootstrap of the line-level AUC difference (a - b)."""
    rng = np.random.default_rng(seed)
    s_a, s_b, k, n = (np.asarray(x, float) for x in (s_a, s_b, n_late, n_total))
    idx = np.arange(len(k))
    d = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx), replace=True)
        d.append(line_level_auc(s_a[i], k[i], n[i]) - line_level_auc(s_b[i], k[i], n[i]))
    d = np.array(d)
    d = d[~np.isnan(d)]          # resamples that contain a single outcome class have no AUC
    return {"delta": float(line_level_auc(s_a, k, n) - line_level_auc(s_b, k, n)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_le_0": float((d <= 0).mean()), "n_valid_resamples": int(len(d))}
