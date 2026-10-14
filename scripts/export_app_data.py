"""Export compact, aggregated artifacts for the Streamlit app into app/data/ (committed, <25 MB).

Only seller-level aggregates (anonymised Olist seller hashes) and summary tables are exported.
No customer identifiers, no order-level rows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART, OUT, PROC = ROOT / "artifacts", ROOT / "app" / "data", ROOT / "data" / "processed"

SELLER_COLS = ["seller_id", "seller_state", "top_category", "n_items", "n_orders", "gmv", "freight", "freight_to_price",
               "n_delivered", "n_late", "late_gmv", "late_rate_raw", "late_rate_shrunk", "late_rate_lo90",
               "late_rate_hi90", "n_handoff_obs", "n_handoff_late", "handoff_late_rate_raw",
               "seller_handoff_late_rate", "cancel_rate_raw", "cancel_rate", "mean_review_score",
               "review_score_shrunk", "n_reviews", "median_delivery_days", "median_handoff_days",
               "mean_delay_days_when_late", "same_state_share", "risk_score", "risk_score_lo90", "risk_score_hi90",
               "risk_tier", "risk_rank", "low_volume_flag", "risk_score_late_heavy", "risk_score_handoff_heavy",
               "risk_score_balanced_alt", "archetype", "pc1", "pc2", "gmv_at_risk", "wc_in_transit",
               "wc_excess_delay", "late_cost_ASSUMED", "freight_exposure_ASSUMED", "cancel_cost_ASSUMED",
               "total_exposure_ASSUMED", "review_risk", "volume_uncertainty", "freight_ratio"]
OOT_COLS = ["seller_id", "risk_score", "late_rate_shrunk", "seller_handoff_late_rate", "late_rate_raw",
            "handoff_late_rate_raw", "n_items", "n_delivered", "next_gmv", "next_late_gmv", "next_n_late",
            "next_n_delivered", "next_n_handoff_late", "next_n_handoff_obs"]
CSVS = ["cold_start.csv",
        "oot_metrics_main.csv", "oot_backtest_metrics.csv", "oot_deciles.csv", "oot_shrinkage_calibration.csv",
        "archetype_profiles.csv", "archetype_centroids_z.csv", "cluster_selection.csv",
        "weight_sensitivity_ranking.csv", "scenario_grid.csv", "exposure_sensitivity.csv", "risk_tier_summary.csv",
        "dq_report.csv"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(ART / "seller_scores.parquet")[SELLER_COLS]
    sc.to_parquet(OUT / "sellers.parquet", index=False)
    pd.read_parquet(ART / "oot_main.parquet")[OOT_COLS].to_parquet(OUT / "oot_sellers.parquet", index=False)
    for c in CSVS:
        shutil.copy(ART / c, OUT / c)
    for f in (ROOT / "artifacts/r_glm").glob("*.csv"):
        if f.name != "glmm_seller_effects.csv":
            shutil.copy(f, OUT / f"r_{f.name}")
    pm = pd.read_parquet(PROC / "platform_month.parquet")
    pm.to_csv(OUT / "platform_month.csv", index=False)
    o = pd.read_parquet(PROC / "fct_orders_delivered.parquet")
    # aggregated lane / category tables (no order rows)
    o["handoff_bucket"] = pd.cut(o.handoff_delay_days, [-60, -3, -1, 0, 1, 3, 7, 400],
                                 labels=["<-3d", "-3..-1d", "-1..0d", "0..1d", "1..3d", "3..7d", ">7d"]).astype(str)
    (o.groupby("handoff_bucket").agg(n=("is_late", "size"), late_rate=("is_late", "mean"),
                                     review=("review_score", "mean")).reset_index()
     .to_csv(OUT / "handoff_buckets.csv", index=False))
    (o.groupby("customer_state").agg(n=("is_late", "size"), late_rate=("is_late", "mean"),
                                     median_days=("delivery_days", "median"), gmv=("order_gmv", "sum"))
     .reset_index().to_csv(OUT / "customer_state.csv", index=False))
    (o.groupby("category").agg(n=("is_late", "size"), late_rate=("is_late", "mean"),
                               median_days=("delivery_days", "median"), freight_to_price=("freight_to_price", "median"))
     .reset_index().query("n >= 300").to_csv(OUT / "category.csv", index=False))
    # lead-time histogram (binned days) by lane type
    o["days_bin"] = o.delivery_days.clip(0, 60).round()
    (o.groupby(["same_state", "days_bin"]).size().rename("n").reset_index().to_csv(OUT / "leadtime_hist.csv",
                                                                                     index=False))
    summary = {n.replace(".json", ""): json.loads((ART / n).read_text()) for n in ["sql_summary.json",
                                                                                 "train_summary.json"]}
    rs = ROOT / "artifacts/r_glm/r_summary.json"
    if rs.exists():
        summary["r_summary"] = json.loads(rs.read_text())
    summary["config"] = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    tot = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"app/data: {len(list(OUT.iterdir()))} files, {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
