"""Out-of-time validation metrics: does a score built on period t predict lateness in period t+1?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def item_level_auc(score, n_late, n_delivered) -> float:
    """AUC of a seller-level score for next-period ITEM lateness (each item inherits its seller's score).

    Implemented with sample weights: one positive row weighted by late items and one negative row weighted
    by on-time items per seller, which is identical to expanding to item grain.
    """
    score, k, n = (np.asarray(x, float) for x in (score, n_late, n_delivered))
    m = n > 0
    s, k, n = score[m], k[m], n[m]
    y = np.r_[np.ones_like(s), np.zeros_like(s)]
    w = np.r_[k, n - k]
    keep = w > 0
    return float(roc_auc_score(y[keep], np.r_[s, s][keep], sample_weight=w[keep]))


def top_share_lift(score, n_late, n_delivered, share: float = 0.10) -> dict[str, float]:
    """Late-rate lift and late-item capture for the top `share` of sellers ranked by score."""
    df = pd.DataFrame({"s": score, "k": n_late, "n": n_delivered}).query("n > 0")
    df = df.sort_values("s", ascending=False, kind="mergesort")
    top = df.head(max(1, int(round(share * len(df)))))
    base = df.k.sum() / df.n.sum()
    rate = top.k.sum() / top.n.sum()
    return {"lift": float(rate / base), "late_rate_top": float(rate), "base_late_rate": float(base),
            "capture_late_items": float(top.k.sum() / df.k.sum()), "share_items": float(top.n.sum() / df.n.sum())}


def seller_rank_corr(score, n_late, n_delivered, min_n: int = 10) -> float:
    s, k, n = (np.asarray(x, float) for x in (score, n_late, n_delivered))
    m = n >= min_n
    return float(stats.spearmanr(s[m], k[m] / n[m]).statistic)


def seller_level_auc(score, n_late, n_delivered, min_n: int = 10) -> float:
    """AUC for flagging sellers whose next-period late rate exceeds the platform rate (sellers with >= min_n)."""
    s, k, n = (np.asarray(x, float) for x in (score, n_late, n_delivered))
    m = n >= min_n
    base = k[m].sum() / n[m].sum()
    y = (k[m] / n[m]) > base
    return float(roc_auc_score(y, s[m])) if 0 < y.sum() < len(y) else float("nan")


def evaluate(scores: dict[str, np.ndarray], n_late, n_delivered) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        lt = top_share_lift(s, n_late, n_delivered, 0.10)
        rows.append({"score": name, "item_auc": item_level_auc(s, n_late, n_delivered),
                     "seller_auc": seller_level_auc(s, n_late, n_delivered),
                     "spearman": seller_rank_corr(s, n_late, n_delivered),
                     "lift_top10": lt["lift"], "capture_top10": lt["capture_late_items"],
                     "late_rate_top10": lt["late_rate_top"], "base_late_rate": lt["base_late_rate"]})
    return pd.DataFrame(rows)


def bootstrap_auc_delta(s_a, s_b, n_late, n_delivered, n_boot: int = 200, seed: int = 42) -> dict[str, float]:
    """Seller-cluster bootstrap of the item-level AUC difference (a - b)."""
    rng = np.random.default_rng(seed)
    s_a, s_b, k, n = (np.asarray(x, float) for x in (s_a, s_b, n_late, n_delivered))
    idx = np.arange(len(k))
    d = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx), replace=True)
        d.append(item_level_auc(s_a[i], k[i], n[i]) - item_level_auc(s_b[i], k[i], n[i]))
    d = np.array(d)
    return {"delta": float(item_level_auc(s_a, k, n) - item_level_auc(s_b, k, n)),
            "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "p_le_0": float((d <= 0).mean())}
