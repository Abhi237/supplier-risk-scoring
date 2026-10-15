"""Export compact, aggregated artifacts for the Streamlit app into app/data/ (committed).

Only DERIVED vendor scorecards and aggregates are exported (vendor names are public in the source dataset).
No order-line rows, no shipment identifiers, no raw CSV. The source licence is unclear, so the raw file stays out of git.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART, OUT, PROC = ROOT / "artifacts", ROOT / "app" / "data", ROOT / "data" / "processed"

VENDOR_COLS = ["vendor", "vendor_code", "top_country", "top_product_group", "n_lines", "n_shipments", "n_countries",
               "value_usd", "n_late", "n_severe_late", "n_exact_match", "late_value_usd", "late_rate_raw",
               "late_rate_shrunk", "late_rate_lo90", "late_rate_hi90", "severe_late_rate_raw",
               "severe_late_rate_shrunk", "mean_days_late_raw", "mean_days_late_shrunk", "mean_days_when_late",
               "freight_ratio_raw", "freight_ratio", "air_share", "country_hhi", "lane_mix_risk", "volume_uncertainty",
               "first_scheduled", "last_scheduled", "is_scored", "risk_score", "risk_score_lo90", "risk_score_hi90",
               "risk_tier", "risk_rank", "risk_score_late_heavy", "risk_score_tail_heavy", "risk_score_no_lane_context",
               "archetype", "pc1", "pc2", "value_at_risk", "wc_excess_delay", "late_cost_ASSUMED",
               "carrying_cost_ASSUMED", "total_exposure_ASSUMED"]
OOT_COLS = ["vendor", "risk_score", "late_rate_shrunk", "late_rate_raw", "severe_late_rate_shrunk", "lane_mix_risk",
            "n_lines", "next_value", "next_late_value", "next_n_late", "next_n_lines", "next_n_severe"]
CSVS = ["cold_start.csv", "oot_metrics_main.csv", "oot_backtest_metrics.csv", "oot_quintiles.csv",
        "oot_shrinkage_calibration.csv", "archetype_profiles.csv", "archetype_centroids_z.csv", "cluster_selection.csv",
        "weight_sensitivity_ranking.csv", "scenario_grid.csv", "exposure_sensitivity.csv", "risk_tier_summary.csv",
        "dq_report.csv", "cleaning_log.csv"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(ART / "vendor_scores.parquet")[VENDOR_COLS]
    sc.to_parquet(OUT / "vendors.parquet", index=False)
    pd.read_parquet(ART / "oot_main.parquet")[OOT_COLS].to_parquet(OUT / "oot_vendors.parquet", index=False)
    for c in CSVS:
        shutil.copy(ART / c, OUT / c)
    for f in sorted((ROOT / "artifacts/r_glm").glob("*.csv")):
        if f.name != "glmm_vendor_effects.csv":
            shutil.copy(f, OUT / f"r_{f.name}")
    shutil.copy(ROOT / "artifacts/r_glm/glmm_vendor_effects.csv", OUT / "r_glmm_vendor_effects.csv")
    pd.read_parquet(PROC / "programme_year.parquet").to_csv(OUT / "programme_year.csv", index=False)
    pd.read_parquet(PROC / "vendor_year_panel.parquet").to_csv(OUT / "vendor_year_panel.csv", index=False)
    f = pd.read_parquet(PROC / "fct_lines.parquet")
    f["channel"] = np.where(f.is_internal_rdc, "SCMS regional DC", "External vendors")
    # aggregated tables (no line rows)
    agg = dict(n_lines=("is_late", "size"), n_late=("is_late", "sum"), n_severe=("is_severe_late", "sum"),
               value_usd=("line_value", "sum"))
    ext = f[~f.is_internal_rdc]
    (ext.groupby("country").agg(**agg, n_vendors=("vendor", "nunique"), n_exact=("is_exact_match", "sum"))
     .reset_index().sort_values(["n_lines", "country"], ascending=[False, True]).to_csv(OUT / "country.csv", index=False))
    (ext.groupby("ship_mode").agg(**agg).reset_index().sort_values(["n_lines", "ship_mode"], ascending=[False, True])
     .to_csv(OUT / "ship_mode.csv", index=False))
    (ext.groupby("product_group").agg(**agg).reset_index().sort_values(["n_lines", "product_group"],
                                                                        ascending=[False, True])
     .to_csv(OUT / "product_group.csv", index=False))
    f["offset_bin"] = f.offset_days.clip(-60, 120).floordiv(5) * 5
    (f.groupby(["channel", "offset_bin"]).size().rename("n").reset_index().to_csv(OUT / "offset_hist.csv", index=False))
    summary = {n.replace(".json", ""): json.loads((ART / n).read_text())
               for n in ["sql_summary.json", "train_summary.json", "data_manifest.json"]}
    summary["r_summary"] = json.loads((ART / "r_glm/r_summary.json").read_text())
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    summary["config"] = cfg
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    tot = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"app/data: {len(list(OUT.iterdir()))} files, {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
