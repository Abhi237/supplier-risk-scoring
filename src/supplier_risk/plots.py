"""Static figures (matplotlib) for docs/figures. Reads only artifacts/ and data/processed outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import resolve  # noqa: E402

NAVY, RED, GREEN, PURPLE, ORANGE, TEAL, GREY = ("#1f4e79", "#c0504d", "#5b9a3c", "#8064a2", "#e8892b",
                                                "#2e9bb5", "#7f8c99")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10, "axes.titleweight": "bold"})
PROXY = "Olist sellers used as supplier proxies"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def platform_monthly(fig_dir: Path):
    pm = pd.read_parquet(resolve("data/processed/platform_month.parquet"))
    pm["purchase_month"] = pd.to_datetime(pm.purchase_month)
    pm = pm[(pm.purchase_month >= "2017-01-01") & (pm.purchase_month < "2018-09-01")]
    x = pm.purchase_month
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x, pm.n_items, width=20, color="#d6e2ef", label="items sold (rhs scale)")
    ax2 = ax.twinx()
    ax2.plot(x, pm.late_rate * 100, "-o", color=RED, label="late vs estimate (%)")
    ax2.plot(x, pm.handoff_late_rate * 100, "-s", color=NAVY, label="seller hand-off late vs limit (%)")
    ax.set_ylabel("items")
    ax2.set_ylabel("% of items")
    ax2.grid(False)
    ax.set_title("Monthly late-delivery and seller hand-off lateness (by purchase month)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    _save(fig, fig_dir / "platform_monthly_lateness.png")


def shrinkage(sc: pd.DataFrame, fig_dir: Path, prior_mean: float):
    d = sc[sc.n_delivered > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(d.n_delivered, d.late_rate_raw, s=8, alpha=0.35, color=GREY, label="raw late rate")
    ax[0].scatter(d.n_delivered, d.late_rate_shrunk, s=8, alpha=0.5, color=NAVY, label="EB-shrunk late rate")
    ax[0].axhline(prior_mean, color=RED, ls="--", lw=1, label=f"fitted prior mean {prior_mean:.1%}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("delivered items (log)")
    ax[0].set_ylabel("late rate")
    ax[0].set_title("Beta-binomial shrinkage pulls small sellers to the prior")
    ax[0].legend(fontsize=8)
    w = d.risk_score_hi90 - d.risk_score_lo90
    ax[1].scatter(d.n_items, w, s=8, alpha=0.4, color=PURPLE)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("items sold (log)")
    ax[1].set_ylabel("90% interval width (score points)")
    ax[1].set_title("Bootstrap interval of the risk score narrows with volume")
    _save(fig, fig_dir / "shrinkage_and_intervals.png")


def top_risk(sc: pd.DataFrame, fig_dir: Path):
    m = sc[sc.n_items >= 50].nlargest(20, "risk_score").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(m))
    ax.errorbar(m.risk_score, y, xerr=[m.risk_score - m.risk_score_lo90, m.risk_score_hi90 - m.risk_score],
                fmt="o", color=RED, ecolor="#e3a3a1", capsize=3)
    ax.set_yticks(y, [f"{s[:8]}… ({st}, {int(n)} items)" for s, st, n in zip(m.seller_id, m.seller_state, m.n_items)],
                  fontsize=8)
    ax.set_xlabel("composite risk score (0-100) with 90% bootstrap interval")
    ax.set_title(f"Top-20 risk among material sellers (>=50 items)\n{PROXY}", fontsize=10)
    _save(fig, fig_dir / "top_risk_material_sellers.png")


def oot(art: Path, fig_dir: Path):
    ev = pd.read_csv(art / "oot_metrics_main.csv")
    keep = ["Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Shrunk hand-off late rate (EB) only",
            "Baseline: raw late rate", "Baseline: raw hand-off late rate", "Baseline: volume only (smaller = riskier)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, tgt in zip(axes, ["End-to-end late delivery", "Seller hand-off late"]):
        e = ev[(ev.target == tgt) & ev.score.isin(keep)].set_index("score").reindex(keep)
        cols = [NAVY if not s.startswith("Baseline") else GREY for s in e.index]
        ax.barh(range(len(e)), e.item_auc, color=cols)
        for i, v in enumerate(e.item_auc):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color=RED, ls="--", lw=1)
        ax.set_xlim(0.45, max(0.75, e.item_auc.max() + 0.03))
        ax.set_yticks(range(len(e)), e.index, fontsize=8)
        ax.set_title(f"Next-period target: {tgt}")
        ax.set_xlabel("item-level ROC-AUC (out-of-time)")
    axes[0].invert_yaxis()
    _save(fig, fig_dir / "oot_validation_auc.png")
    dec = pd.read_csv(art / "oot_deciles.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dec.decile - 0.2, dec.next_handoff_late_rate * 100, width=0.4, color=NAVY, label="hand-off late (%)")
    ax.bar(dec.decile + 0.2, dec.next_late_rate * 100, width=0.4, color=RED, label="late vs estimate (%)")
    ax.set_xticks(dec.decile)
    ax.set_xlabel("risk-score decile at 2018-01-01 (10 = riskiest)")
    ax.set_ylabel("next-period rate (%)")
    ax.set_title("Out-of-time: next-period lateness by risk-score decile")
    ax.legend()
    _save(fig, fig_dir / "oot_deciles.png")
    bt = pd.read_csv(art / "oot_backtest_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, tgt in zip(axes, ["End-to-end late delivery", "Seller hand-off late"]):
        for nm, c in [("Composite risk score (shrunk)", NAVY), ("Shrunk hand-off late rate (EB) only", TEAL),
                      ("Baseline: raw late rate", GREY), ("Baseline: raw hand-off late rate", ORANGE)]:
            b = bt[(bt.target == tgt) & (bt.score == nm)]
            ax.plot(b.split.str[:10], b.item_auc, "-o", color=c, label=nm)
        ax.axhline(0.5, color=RED, ls="--", lw=1)
        ax.set_title(tgt)
        ax.set_xlabel("score date (next-period window follows)")
    axes[0].set_ylabel("item-level ROC-AUC")
    axes[1].legend(fontsize=7)
    _save(fig, fig_dir / "oot_backtest_splits.png")


def archetypes(sc: pd.DataFrame, art: Path, fig_dir: Path):
    d = sc[sc.pc1.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    cols = [NAVY, RED, GREEN, ORANGE, PURPLE, TEAL, GREY]
    for i, (nm, g) in enumerate(d.groupby("archetype")):
        ax[0].scatter(g.pc1, g.pc2, s=10, alpha=0.6, color=cols[i % 7], label=f"{nm} (n={len(g)})")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("Indicative seller archetypes (k-means; soft segments)")
    ax[0].legend(fontsize=7)
    z = pd.read_csv(art / "archetype_centroids_z.csv", index_col=0)
    im = ax[1].imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax[1].set_xticks(range(z.shape[1]), z.columns, rotation=35, ha="right", fontsize=8)
    ax[1].set_yticks(range(z.shape[0]), z.index, fontsize=8)
    for (i, j), v in np.ndenumerate(z.to_numpy()):
        ax[1].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], label="centroid z-score")
    ax[1].set_title("Archetype centroids (z-scores)")
    _save(fig, fig_dir / "seller_archetypes.png")


def exposure_figs(sc: pd.DataFrame, art: Path, fig_dir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, c, lab in [("gmv", NAVY, "GMV"), ("late_gmv", RED, "observed late GMV"),
                        ("gmv_at_risk", ORANGE, "GMV at risk (shrunk)")]:
        v = np.sort(sc[col].to_numpy())[::-1]
        ax[0].plot(np.arange(1, len(v) + 1) / len(v) * 100, np.cumsum(v) / v.sum() * 100, color=c, label=lab)
    ax[0].set_xlabel("% of sellers (ranked by the metric)")
    ax[0].set_ylabel("cumulative %")
    ax[0].set_title("Concentration of GMV and late GMV")
    ax[0].legend()
    m = sc[sc.n_items >= 20]
    sca = ax[1].scatter(m.gmv / 1e3, m.late_rate_shrunk * 100, c=m.risk_score, cmap="RdYlGn_r", s=12, alpha=0.7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("seller GMV, BRL thousands (log)")
    ax[1].set_ylabel("EB-shrunk late rate (%)")
    ax[1].set_title("Exposure map: volume x lateness (sellers >= 20 items)")
    fig.colorbar(sca, ax=ax[1], label="risk score")
    _save(fig, fig_dir / "exposure_concentration.png")
    g = pd.read_csv(art / "scenario_grid.csv")
    g = g[g.improvement_ASSUMED == 0.5].pivot(index="top_share", columns="realloc_frac",
                                               values="late_gmv_reduction_pct") * 100
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(g.to_numpy(), cmap="Greens", aspect="auto")
    ax.set_xticks(range(g.shape[1]), [f"{c:.0%}" for c in g.columns])
    ax.set_yticks(range(g.shape[0]), [f"top {r:.0%}" for r in g.index])
    for (i, j), v in np.ndenumerate(g.to_numpy()):
        ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=9)
    ax.set_xlabel("share of their volume reallocated (ASSUMED)")
    ax.set_ylabel("risk group")
    ax.set_title("Back-tested late-GMV reduction, gap realised = 50% (ASSUMED)", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    _save(fig, fig_dir / "scenario_reallocation_grid.png")


def handoff_link(fig_dir: Path):
    o = pd.read_parquet(resolve("data/processed/fct_orders_delivered.parquet"))
    o = o[o.handoff_delay_days.notna()]
    o["bucket"] = pd.cut(o.handoff_delay_days, [-60, -3, -1, 0, 1, 3, 7, 400],
                         labels=["<-3d", "-3..-1d", "-1..0d", "0..1d", "1..3d", "3..7d", ">7d"])
    g = o.groupby("bucket", observed=True).agg(late=("is_late", "mean"), n=("is_late", "size"),
                                                rev=("review_score", "mean"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(g.index.astype(str), g.late * 100, color=[GREEN] * 3 + [RED] * 4)
    for i, (v, n) in enumerate(zip(g.late, g.n)):
        ax.text(i, v * 100 + 0.8, f"{v:.0%}\nn={n:,}", ha="center", fontsize=7)
    ax.set_xlabel("seller hand-off to carrier relative to shipping limit (days; >0 = late)")
    ax.set_ylabel("% delivered after estimate")
    ax.set_ylim(0, g.late.max() * 100 * 1.18)
    ax.set_title("Late hand-off by the seller drives late delivery to the customer")
    _save(fig, fig_dir / "handoff_delay_vs_late.png")


def make_all(art: Path, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(art / "seller_scores.parquet")
    s = json.loads((art / "train_summary.json").read_text())
    platform_monthly(fig_dir)
    shrinkage(sc, fig_dir, s["priors"]["late"][2])
    top_risk(sc, fig_dir)
    oot(art, fig_dir)
    archetypes(sc, art, fig_dir)
    exposure_figs(sc, art, fig_dir)
    handoff_link(fig_dir)
