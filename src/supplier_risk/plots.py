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
SCOPE = "USAID / PEPFAR SCMS delivery history, 2006-2015"
LATE_T, SEV_T = "Late delivery (delivered after scheduled date)", "Severe late (more than 14 days late)"


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def programme_yearly(fig_dir: Path):
    py = pd.read_parquet(resolve("data/processed/programme_year.parquet"))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(py.sched_year, py.n_lines_vendor, color="#d6e2ef", label="external-vendor lines (lhs)")
    ax.bar(py.sched_year, py.n_lines_rdc, bottom=py.n_lines_vendor, color="#eceff3", label="SCMS regional DC lines (lhs)")
    ax.set_ylabel("order lines")
    ax2 = ax.twinx()
    ax2.plot(py.sched_year, py.n_late_vendor / py.n_lines_vendor * 100, "-o", color=RED, label="late, external vendors (%)")
    ax2.plot(py.sched_year, py.n_late_rdc / py.n_lines_rdc.replace(0, np.nan) * 100, "-s", color=NAVY,
             label="late, SCMS regional DC (%)")
    ax2.plot(py.sched_year, py.n_exact_vendor / py.n_lines_vendor * 100, "--^", color=GREY,
             label="delivered date = scheduled date, external vendors (%)")
    ax2.set_ylabel("% of lines")
    ax2.grid(False)
    ax.set_xticks(py.sched_year)
    ax.set_title("Lateness by scheduled year, and how often the two dates are identical")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7, frameon=False)
    _save(fig, fig_dir / "programme_yearly_lateness.png")


def offset_hist(fig_dir: Path):
    f = pd.read_parquet(resolve("data/processed/fct_lines.parquet"), columns=["offset_days", "is_internal_rdc"])
    f = f[~f.is_internal_rdc]
    bins = np.arange(-60, 121, 5)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(f.offset_days.clip(-60, 120), bins=bins, color=NAVY)
    ax.set_yscale("log")
    ax.axvline(0.5, color=RED, ls="--", lw=1)
    ax.set_xlabel("delivered date minus scheduled date (days; > 0 = late; clipped to -60..120)")
    ax.set_ylabel("order lines (log scale)")
    n0 = int((f.offset_days == 0).sum())
    ax.set_title(f"External-vendor lines: {n0:,} of {len(f):,} have delivered date = scheduled date")
    _save(fig, fig_dir / "delivery_offset_distribution.png")


def shrinkage(sc: pd.DataFrame, fig_dir: Path, prior_mean: float):
    d = sc[sc.n_lines > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(d.n_lines, d.late_rate_raw, s=22, alpha=0.5, color=GREY, label="raw late rate")
    ax[0].scatter(d.n_lines, d.late_rate_shrunk, s=22, alpha=0.7, color=NAVY, label="EB-shrunk late rate")
    ax[0].axhline(prior_mean, color=RED, ls="--", lw=1, label=f"fitted prior mean {prior_mean:.1%}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("order lines (log)")
    ax[0].set_ylabel("late rate")
    ax[0].set_title("Shrinkage pulls thin vendors to the prior")
    ax[0].legend(fontsize=8)
    s = d[d.is_scored]
    ax[1].scatter(s.n_lines, s.risk_score_hi90 - s.risk_score_lo90, s=26, alpha=0.7, color=PURPLE)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("order lines (log), scored vendors")
    ax[1].set_ylabel("90% interval width (score points)")
    ax[1].set_title("Bootstrap interval of the risk score vs volume")
    _save(fig, fig_dir / "shrinkage_and_intervals.png")


def top_risk(sc: pd.DataFrame, fig_dir: Path):
    m = sc[sc.is_scored].nlargest(15, "risk_score").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(m))
    ax.errorbar(m.risk_score, y, xerr=[m.risk_score - m.risk_score_lo90, m.risk_score_hi90 - m.risk_score],
                fmt="o", color=RED, ecolor="#e3a3a1", capsize=3)
    ax.set_yticks(y, [f"{v[:34]} ({int(n)} lines)" for v, n in zip(m.vendor, m.n_lines, strict=True)], fontsize=8)
    ax.set_xlabel("composite risk score (0-100) with 90% Bayesian-bootstrap interval")
    ax.set_title(f"Top-15 scored vendors by recorded delivery risk\n{SCOPE}", fontsize=10)
    _save(fig, fig_dir / "top_risk_vendors.png")


def oot(art: Path, fig_dir: Path):
    ev = pd.read_csv(art / "oot_metrics_main.csv")
    keep = ["Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Shrunk severe-late rate (EB) only",
            "Baseline: raw late rate", "Baseline: raw severe-late rate", "Baseline: volume only (smaller = riskier)",
            "Baseline: lane mix only (country / mode)"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, tgt in zip(axes, [LATE_T, SEV_T], strict=True):
        e = ev[(ev.target == tgt) & ev.score.isin(keep)].set_index("score").reindex(keep)
        cols = [NAVY if not s.startswith("Baseline") else GREY for s in e.index]
        ax.barh(range(len(e)), e.line_auc, color=cols)
        for i, v in enumerate(e.line_auc):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
        ax.axvline(0.5, color=RED, ls="--", lw=1)
        ax.set_xlim(0.2, 0.85)
        ax.set_yticks(range(len(e)), e.index, fontsize=8)
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("line-level ROC-AUC (out-of-time)")
    axes[0].invert_yaxis()
    _save(fig, fig_dir / "oot_validation_auc.png")
    q = pd.read_csv(art / "oot_quintiles.csv")
    s_end = json.loads((art / "train_summary.json").read_text())["oot_main"]["score_end"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(q.quintile - 0.2, q.next_late_rate * 100, width=0.4, color=NAVY, label="by composite score")
    ax.bar(q.quintile + 0.2, q.next_late_rate_raw_baseline * 100, width=0.4, color=GREY, label="by raw late rate")
    ax.set_xticks(q.quintile)
    ax.set_xlabel(f"vendor quintile at {s_end} (5 = riskiest)")
    ax.set_ylabel("next-period late rate (%)")
    ax.set_title("Out-of-time: next-period late rate by score quintile (about 4 vendors each)")
    ax.legend()
    _save(fig, fig_dir / "oot_quintiles.png")
    bt = pd.read_csv(art / "oot_backtest_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
    for ax, tgt in zip(axes, [LATE_T, SEV_T], strict=True):
        for nm, c in [("Composite risk score (shrunk)", NAVY), ("Shrunk late rate (EB) only", TEAL),
                      ("Baseline: raw late rate", GREY), ("Baseline: volume only (smaller = riskier)", ORANGE)]:
            b = bt[(bt.target == tgt) & (bt.score == nm)]
            ax.plot(b.split.str[:10], b.line_auc, "-o", color=c, label=nm)
        ax.axhline(0.5, color=RED, ls="--", lw=1)
        ax.set_title(tgt, fontsize=9)
        ax.set_xlabel("score date (next-period window follows)")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("line-level ROC-AUC")
    axes[1].legend(fontsize=7)
    _save(fig, fig_dir / "oot_backtest_splits.png")


def archetypes(sc: pd.DataFrame, art: Path, fig_dir: Path):
    d = sc[sc.pc1.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    cols = [NAVY, RED, GREEN, ORANGE, PURPLE, TEAL, GREY]
    for i, (nm, g) in enumerate(d.groupby("archetype")):
        ax[0].scatter(g.pc1, g.pc2, s=40, alpha=0.7, color=cols[i % 7], label=f"{nm} (n={len(g)})")
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("Indicative vendor archetypes (k-means; soft segments)")
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
    _save(fig, fig_dir / "vendor_archetypes.png")


def exposure_figs(sc: pd.DataFrame, art: Path, fig_dir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, c, lab in [("value_usd", NAVY, "line value"), ("late_value_usd", RED, "observed late value"),
                        ("value_at_risk", ORANGE, "value at risk (shrunk)")]:
        v = np.sort(sc[col].to_numpy())[::-1]
        ax[0].plot(np.arange(1, len(v) + 1) / len(v) * 100, np.cumsum(v) / v.sum() * 100, color=c, label=lab)
    ax[0].set_xlabel("% of external vendors (ranked by the metric)")
    ax[0].set_ylabel("cumulative %")
    ax[0].set_title("Concentration of value and late value")
    ax[0].legend()
    m = sc[sc.is_scored]
    sca = ax[1].scatter(m.value_usd / 1e6, m.late_rate_shrunk * 100, c=m.risk_score, cmap="RdYlGn_r", s=34, alpha=0.85)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("vendor line value, USD millions (log)")
    ax[1].set_ylabel("EB-shrunk late rate (%)")
    ax[1].set_title("Exposure map: value x lateness (scored vendors)")
    fig.colorbar(sca, ax=ax[1], label="risk score")
    _save(fig, fig_dir / "exposure_concentration.png")
    g = pd.read_csv(art / "scenario_grid.csv")
    g = g[g.improvement_ASSUMED == 0.5].pivot(index="top_share", columns="realloc_frac",
                                               values="late_value_reduction_pct") * 100
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(g.to_numpy(), cmap="Greens", aspect="auto", vmin=0)
    ax.set_xticks(range(g.shape[1]), [f"{c:.0%}" for c in g.columns])
    ax.set_yticks(range(g.shape[0]), [f"top {r:.0%}" for r in g.index])
    for (i, j), v in np.ndenumerate(g.to_numpy()):
        ax.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=9)
    ax.set_xlabel("share of their volume reallocated (ASSUMED)")
    ax.set_ylabel("risk group (composite ranking)")
    ax.set_title("Back-tested late-value reduction, gap realised = 50% (ASSUMED)", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    _save(fig, fig_dir / "scenario_reallocation_grid.png")


def make_all(art: Path, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(art / "vendor_scores.parquet")
    s = json.loads((art / "train_summary.json").read_text())
    programme_yearly(fig_dir)
    offset_hist(fig_dir)
    shrinkage(sc, fig_dir, s["priors"]["late"][2])
    top_risk(sc, fig_dir)
    oot(art, fig_dir)
    archetypes(sc, art, fig_dir)
    exposure_figs(sc, art, fig_dir)
