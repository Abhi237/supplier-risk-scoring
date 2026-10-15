"""End-to-end pipeline steps used by the CLI: sql -> train (score, validate, cluster, exposure, scenarios)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clustering, coldstart, exposure, plots, scoring, validation
from .config import load_config, resolve
from .db import connect, run_sql_model, set_oot_window

EXPORT = {"dq_report": "check_name", "cleaning_log": "ord", "fct_lines": "line_id", "fct_shipments": "shipment_id",
          "vendor_summary": "vendor", "vendor_lane": "vendor, country, ship_mode",
          "vendor_year_panel": "vendor, sched_year", "programme_year": "sched_year"}

LATE_TGT, SEVERE_TGT = "Late delivery (delivered after scheduled date)", "Severe late (more than 14 days late)"


def _art():
    p = resolve(load_config()["paths"]["artifacts"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump(obj, name):
    (_art() / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def sql_summary(con) -> dict:
    q = lambda s: con.execute(s).fetchone()  # noqa: E731
    cl = dict(con.execute("SELECT action, n_rows FROM cleaning_log").fetchall())
    dq = dict(con.execute("SELECT check_name, n FROM dq_report").fetchall())
    n, n_ship, n_vend, n_ctry, n_late, n_sev, n_exact, val = q(
        "SELECT COUNT(*), COUNT(DISTINCT shipment_id), COUNT(DISTINCT vendor), COUNT(DISTINCT country), "
        "SUM(is_late::INT), SUM(is_severe_late::INT), SUM(is_exact_match::INT), SUM(line_value)::DOUBLE FROM fct_lines")
    ext = q("SELECT COUNT(*), COUNT(DISTINCT vendor), COUNT(DISTINCT shipment_id), COUNT(DISTINCT country), "
            "SUM(is_late::INT), SUM(is_severe_late::INT), SUM(is_exact_match::INT), SUM(line_value)::DOUBLE, "
            "SUM(CASE WHEN is_late THEN line_value END)::DOUBLE, AVG(CASE WHEN is_late THEN days_late END), "
            "AVG(is_air::INT) FROM fct_lines WHERE NOT is_internal_rdc")
    rdc = q("SELECT COUNT(*), SUM(is_late::INT), SUM(is_severe_late::INT), SUM(is_exact_match::INT), "
            "SUM(line_value)::DOUBLE FROM fct_lines WHERE is_internal_rdc")
    smin, smax, dmin, dmax = q("SELECT MIN(scheduled_date), MAX(scheduled_date), MIN(delivered_date), "
                               "MAX(delivered_date) FROM fct_lines")
    rec = q("SELECT AVG(record_lag_days), MEDIAN(record_lag_days), MAX(record_lag_days), "
            "AVG((record_lag_days > 30)::INT) FROM fct_lines")
    late_pos = q("SELECT AVG(days_late), MEDIAN(days_late), QUANTILE_CONT(days_late, 0.9) FROM fct_lines "
                 "WHERE NOT is_internal_rdc AND is_late")
    conc = con.execute("SELECT value_usd FROM vendor_summary WHERE NOT is_internal_rdc "
                       "ORDER BY value_usd DESC, vendor").df().value_usd
    mode = con.execute("SELECT ship_mode, COUNT(*) n FROM fct_lines WHERE NOT is_internal_rdc GROUP BY 1 "
                       "ORDER BY 2 DESC, 1").df()
    pg = con.execute("SELECT product_group, COUNT(*) n FROM fct_lines WHERE NOT is_internal_rdc GROUP BY 1 "
                     "ORDER BY 2 DESC, 1").df()
    vs = con.execute("SELECT n_lines FROM vendor_summary WHERE NOT is_internal_rdc").df().n_lines
    return {
        "cleaning": cl, "dq": dq, "n_lines": n, "n_shipments": n_ship, "n_vendor_strings": n_vend, "n_countries": n_ctry,
        "value_usd": float(val), "scheduled_min": str(smin), "scheduled_max": str(smax), "delivered_min": str(dmin),
        "delivered_max": str(dmax),
        "all_late_rate": n_late / n, "all_on_time_rate": 1 - n_late / n, "all_exact_match_share": n_exact / n,
        "all_severe_late_rate": n_sev / n,
        "ext_lines": ext[0], "ext_vendors": ext[1], "ext_shipments": ext[2], "ext_countries": ext[3],
        "ext_late_rate": ext[4] / ext[0], "ext_on_time_rate": 1 - ext[4] / ext[0],
        "ext_severe_late_rate": ext[5] / ext[0], "ext_exact_match_share": ext[6] / ext[0], "ext_value_usd": float(ext[7]),
        "ext_late_value_usd": float(ext[8]), "ext_late_value_share": float(ext[8] / ext[7]),
        "ext_mean_days_late_when_late": float(ext[9]), "ext_air_share": float(ext[10]),
        "ext_median_days_late_when_late": float(late_pos[1]), "ext_p90_days_late_when_late": float(late_pos[2]),
        "rdc_lines": rdc[0], "rdc_late_rate": rdc[1] / rdc[0], "rdc_severe_late_rate": rdc[2] / rdc[0],
        "rdc_exact_match_share": rdc[3] / rdc[0], "rdc_value_usd": float(rdc[4]), "rdc_line_share": rdc[0] / n,
        "record_lag_mean_days": float(rec[0]), "record_lag_median_days": float(rec[1]), "record_lag_max_days": float(rec[2]),
        "record_lag_gt30_share": float(rec[3]),
        "top10pct_vendors_value_share": float(conc.head(int(round(0.1 * len(conc)))).sum() / conc.sum()),
        "ext_vendors_ge10_lines": int((vs >= 10).sum()), "ext_vendors_ge20_lines": int((vs >= 20).sum()),
        "ext_vendors_ge50_lines": int((vs >= 50).sum()), "ext_vendors_lt10_lines": int((vs < 10).sum()),
        "ext_mode_lines": dict(zip(mode["ship_mode"], mode["n"], strict=True)),
        "ext_product_group_lines": dict(zip(pg["product_group"], pg["n"], strict=True)),
    }


def run_sql() -> dict:
    cfg = load_config()
    con = connect()
    run_sql_model(con, cfg=cfg)
    out = resolve("data/processed")
    for t, order in EXPORT.items():
        con.execute(f"SELECT * FROM {t} ORDER BY {order}").df().to_parquet(out / f"{t}.parquet", index=False)
    con.execute("SELECT * FROM dq_report ORDER BY check_name").df().to_csv(_art() / "dq_report.csv", index=False)
    con.execute("SELECT ord, action, n_rows, base, detail FROM cleaning_log ORDER BY ord").df().to_csv(
        _art() / "cleaning_log.csv", index=False)
    # line-grain table for R: explicit row order, no free-text columns
    con.execute("SELECT line_id, shipment_id, vendor, is_internal_rdc, country, ship_mode, product_group, fulfill_via, "
                "inco_term, is_first_line, ship_n_lines, sched_year, line_value, line_qty, ship_freight_ratio, offset_days, "
                "days_late, is_late, is_severe_late, is_air FROM fct_lines ORDER BY line_id").df().to_csv(
        out / "lines_for_r.csv", index=False)
    s = sql_summary(con)
    _dump(s, "sql_summary.json")
    con.close()
    return s


def score_vendors(feat: pd.DataFrame, vlane: pd.DataFrame, cfg: dict, weights: dict | None = None):
    """Components + composite for every vendor with data; the percentile reference uses SCORED vendors only
    (n_lines >= min_lines_for_score).  Unscored vendors keep a numeric composite internally (is_scored = False) but
    it is never published or evaluated."""
    priors = scoring.fit_priors(feat, cfg)
    comp = scoring.score_components(feat, priors, vlane, cfg)
    comp["is_scored"] = comp.n_lines >= cfg["scoring"]["min_lines_for_score"]
    ref = scoring.percentile_ref(comp[comp.is_scored])
    comp["risk_score"] = scoring.composite(comp, weights or cfg["scoring"]["weights"], ref)
    return comp, priors, ref


def oot_split(con, cfg, score_end, validate_end, with_scores: bool = False):
    set_oot_window(con, score_end, validate_end, cfg)
    feat = con.execute("SELECT * FROM oot_features ORDER BY vendor").df()
    lane = con.execute("SELECT * FROM oot_vendor_lane ORDER BY vendor, country, ship_mode").df()
    outc = con.execute("SELECT * FROM oot_outcomes ORDER BY vendor").df()
    comp, priors, ref = score_vendors(feat, lane, cfg)
    scored = comp[comp.is_scored]
    pooled = comp.n_late.sum() / max(comp.n_lines.sum(), 1)
    pooled_sev = comp.n_severe_late.sum() / max(comp.n_lines.sum(), 1)
    d_all = comp.merge(outc, on="vendor", how="inner")
    d_all = d_all[d_all.next_n_lines > 0].reset_index(drop=True)
    d = d_all[d_all.is_scored].reset_index(drop=True)
    lane_only = d.lane_mix_risk.to_numpy()
    scores = {
        "Composite risk score (shrunk)": d.risk_score.to_numpy(),
        "Shrunk late rate (EB) only": d.late_rate_shrunk.to_numpy(),
        "Shrunk severe-late rate (EB) only": d.severe_late_rate_shrunk.to_numpy(),
        "Baseline: raw late rate": d.late_rate_raw.fillna(pooled).to_numpy(),
        "Baseline: raw severe-late rate": d.severe_late_rate_raw.fillna(pooled_sev).to_numpy(),
        "Baseline: volume only (smaller = riskier)": -np.log1p(d.n_lines.to_numpy()),
        "Baseline: lane mix only (country / mode)": lane_only,
    }
    for nm, w in cfg["scoring"]["weight_sensitivity"].items():
        scores[f"Sensitivity: {nm} weights"] = scoring.composite(d, w, ref)
    ev_late = validation.evaluate(scores, d.next_n_late, d.next_n_lines)
    ev_late.insert(0, "target", LATE_TGT)
    ev_sev = validation.evaluate(scores, d.next_n_severe, d.next_n_lines)
    ev_sev.insert(0, "target", SEVERE_TGT)
    ev = pd.concat([ev_late, ev_sev], ignore_index=True)
    ev["n_vendors"] = len(d)
    ev["n_next_late_lines"] = np.where(ev.target == LATE_TGT, d.next_n_late.sum(), d.next_n_severe.sum())
    nxt_all = outc[outc.next_n_lines > 0]
    meta = {"score_end": score_end, "validate_end": validate_end, "n_vendors_with_history": int(len(comp)),
            "n_vendors_scored": int(len(scored)), "n_vendors_evaluated": int(len(d)),
            "n_vendors_next_period": int(len(nxt_all)),
            "next_lines_covered_share": float(d.next_n_lines.sum() / nxt_all.next_n_lines.sum()),
            "next_lines": int(d.next_n_lines.sum()), "next_late_lines": int(d.next_n_late.sum()),
            "next_severe_lines": int(d.next_n_severe.sum()),
            "next_late_rate": float(d.next_n_late.sum() / d.next_n_lines.sum()),
            "next_severe_rate": float(d.next_n_severe.sum() / d.next_n_lines.sum()),
            "feature_lines": int(comp.n_lines.sum()),
            "max_known_date_used": str(feat.max_known_date_used.max()),
            "max_scheduled_used": str(feat.max_scheduled_used.max()),
            "prior_late": [priors["late"].a, priors["late"].b],
            "cold_start": coldstart.evaluate_cold_start(outc, scored, comp, priors, comp.attrs["pop_days_late"])}
    return (ev, meta, d, scores, d_all) if with_scores else (ev, meta)


def train() -> dict:
    cfg = load_config()
    sc, art = cfg["scoring"], _art()
    con = connect()
    out_dir = resolve("data/processed")
    # ---------- 1. full-period scorecard ----------
    vs = pd.read_parquet(out_dir / "vendor_summary.parquet")
    full = vs[~vs.is_internal_rdc].sort_values("vendor", kind="mergesort").reset_index(drop=True)
    vlane = pd.read_parquet(out_dir / "vendor_lane.parquet")
    comp, priors, ref = score_vendors(full, vlane, cfg)
    pop_days = comp.attrs["pop_days_late"]
    attrs = dict(comp.attrs)
    lines = pd.read_parquet(out_dir / "fct_lines.parquet", columns=["vendor", "is_internal_rdc", "days_late_capped",
                                                                    "line_id"])
    lines = lines[~lines.is_internal_rdc].sort_values("line_id")
    days_by_vendor = {v: g.days_late_capped.to_numpy(float) for v, g in lines.groupby("vendor")}
    sdf = comp[comp.is_scored].reset_index(drop=True)
    boot = scoring.bootstrap_scores(sdf, days_by_vendor, priors, sc["weights"], ref, pop_days, sc["mean_days_prior_n"],
                                    sc["n_bootstrap"], sc["random_state"])
    comp = comp.merge(boot, on="vendor", how="left")
    comp["risk_tier"] = coldstart.NEW_TIER
    comp.loc[comp.is_scored, "risk_tier"] = scoring.risk_tier(comp.loc[comp.is_scored, "risk_score"]).to_numpy()
    comp["risk_rank"] = comp.risk_score.where(comp.is_scored).rank(ascending=False, method="first")
    comp.loc[~comp.is_scored, ["risk_score", "risk_score_lo90", "risk_score_hi90"]] = np.nan
    for nm, w in sc["weight_sensitivity"].items():
        comp[f"risk_score_{nm}"] = scoring.composite(comp, w, ref)
        comp.loc[~comp.is_scored, f"risk_score_{nm}"] = np.nan
    comp["vendor_code"] = [f"V{i + 1:02d}" for i in range(len(comp))]
    # ---------- 2. exposure ----------
    comp = exposure.vendor_exposure(comp, cfg["exposure"])
    # ---------- 3. clustering ----------
    cl = cfg["clustering"]
    comp["log_value"] = np.log1p(comp.value_usd)
    elig = comp[comp.is_scored & (comp.n_lines >= cl["min_lines"])].dropna(subset=cl["features"]).copy()
    labels, names, z, ktab, X, order_v = clustering.fit_archetypes(elig, cl["features"], cl["k_range"],
                                                                   cl["random_state"])
    elig = elig.set_index("vendor").loc[order_v].reset_index()
    elig["cluster"] = labels
    elig["archetype"] = elig.cluster.map(names)
    comp = comp.merge(elig[["vendor", "cluster", "archetype"]], on="vendor", how="left")
    comp["archetype"] = comp.archetype.fillna(f"Unclustered (<{cl['min_lines']} lines)")
    prof = (comp[comp.cluster.notna()].groupby("archetype")
            .agg(n_vendors=("vendor", "size"), value_usd=("value_usd", "sum"), median_lines=("n_lines", "median"),
                 late_rate_shrunk=("late_rate_shrunk", "median"), mean_days_late=("mean_days_late_shrunk", "median"),
                 freight_ratio=("freight_ratio", "median"), air_share=("air_share", "median"),
                 country_hhi=("country_hhi", "median"), risk_score=("risk_score", "median"),
                 value_at_risk=("value_at_risk", "sum"))
            .reset_index().sort_values(["risk_score", "archetype"], ascending=[False, True]))
    prof["value_share"] = prof.value_usd / comp.value_usd.sum()
    z.index = [names[i] for i in z.index]
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    elig["pc1"], elig["pc2"] = xy[:, 0], xy[:, 1]
    comp = comp.merge(elig[["vendor", "pc1", "pc2"]], on="vendor", how="left")
    # ---------- 4. weight sensitivity on the full-period ranking (scored vendors) ----------
    scored_c = comp[comp.is_scored]
    base_top = set(scored_c.nsmallest(5, "risk_rank").vendor)
    sens_rows = []
    for nm in sc["weight_sensitivity"]:
        alt_top = set(scored_c.nlargest(5, f"risk_score_{nm}").vendor)
        sens_rows.append({"weights": nm, "spearman_vs_base": scored_c.risk_score.corr(scored_c[f"risk_score_{nm}"],
                                                                                        method="spearman"),
                          "top5_overlap": len(base_top & alt_top) / 5})
    # ---------- 5. out-of-time validation ----------
    ev_all, metas = [], []
    for s_end, v_end in sc["backtest_splits"]:
        ev, meta = oot_split(con, cfg, s_end, v_end)
        ev.insert(0, "split", f"{s_end}->{v_end}")
        ev_all.append(ev)
        metas.append(meta)
    ev_all = pd.concat(ev_all, ignore_index=True)
    s_end, v_end = sc["score_period_end"], sc["validate_period_end"]
    ev_main, meta_main, d_main, scores_main, d_all_main = oot_split(con, cfg, s_end, v_end, with_scores=True)
    cs_rows = []
    for mt in metas:
        c = mt["cold_start"]
        cs_rows.append({"split": f"{mt['score_end']}->{mt['validate_end']}", "n_new_vendors": c["n_new_vendors"],
                        "n_new_no_history": c["n_new_no_history"], "new_share_next_lines": c["new_share_next_lines"],
                        "new_share_next_value": c["new_share_next_value"], "new_lines": c["late"]["new_lines"],
                        "prior_mean_late": c["late"]["prior_mean"], "new_late": c["late"]["new_rate"],
                        "new_late_lo95": c["late"]["new_lo95"], "new_late_hi95": c["late"]["new_hi95"],
                        "scored_late": c["late"]["scored_rate"], "prior_inside_ci": c["late"]["prior_inside_ci"],
                        "late_auc_scored_only": c["scored_only_late_auc"],
                        "late_auc_full_coverage": c["full_coverage_late_auc"]})
    pd.DataFrame(cs_rows).round(4).to_csv(art / "cold_start.csv", index=False)
    deltas = {}
    for tgt, k in [("late", d_main.next_n_late), ("severe", d_main.next_n_severe)]:
        n = d_main.next_n_lines
        for nm, a, b in [("composite_vs_raw_late", "Composite risk score (shrunk)", "Baseline: raw late rate"),
                         ("eb_late_vs_raw_late", "Shrunk late rate (EB) only", "Baseline: raw late rate"),
                         ("composite_vs_volume", "Composite risk score (shrunk)",
                          "Baseline: volume only (smaller = riskier)"),
                         ("composite_vs_lane_only", "Composite risk score (shrunk)",
                          "Baseline: lane mix only (country / mode)")]:
            deltas[f"{nm}_{tgt}"] = validation.bootstrap_auc_delta(scores_main[a], scores_main[b], k, n, 200,
                                                                   sc["random_state"])
    # quintile table for the main split (few vendors: quintiles, not deciles)
    dd = d_main.assign(quintile=pd.qcut(d_main.risk_score.rank(method="first"), 5, labels=False) + 1)
    dec = (dd.groupby("quintile").agg(n_vendors=("vendor", "size"), score_min=("risk_score", "min"),
                                     next_late=("next_n_late", "sum"), next_lines=("next_n_lines", "sum"),
                                     next_value=("next_value", "sum"), next_late_value=("next_late_value", "sum"),
                                     next_severe=("next_n_severe", "sum")).reset_index())
    dec["next_late_rate"] = dec.next_late / dec.next_lines
    dec["next_severe_rate"] = dec.next_severe / dec.next_lines
    dd["raw_quintile"] = pd.qcut(pd.Series(scores_main["Baseline: raw late rate"]).rank(method="first"), 5,
                                 labels=False) + 1
    dec["next_late_rate_raw_baseline"] = dd.groupby("raw_quintile").apply(
        lambda g: g.next_n_late.sum() / g.next_n_lines.sum(), include_groups=False).to_numpy()
    # shrinkage calibration: raw vs shrunk late rate error against the next-period rate, by history volume band
    da = d_all_main.copy()
    da["vol_band"] = pd.cut(da.n_lines, [0, 4, 9, 29, 99, 1e9], labels=["1-4", "5-9", "10-29", "30-99", "100+"])
    nl = da.next_n_late / da.next_n_lines
    calib = (da.assign(err_raw=(da.late_rate_raw.fillna(meta_main["next_late_rate"]) - nl).abs() * da.next_n_lines,
                       err_eb=(da.late_rate_shrunk - nl).abs() * da.next_n_lines)
             .groupby("vol_band", observed=True)
             .agg(n_vendors=("vendor", "size"), next_lines=("next_n_lines", "sum"), err_raw=("err_raw", "sum"),
                  err_eb=("err_eb", "sum")).reset_index())
    calib["mae_raw"] = calib.err_raw / calib.next_lines
    calib["mae_shrunk"] = calib.err_eb / calib.next_lines
    calib = calib.drop(columns=["err_raw", "err_eb"])
    # ---------- 6. scenario back-test (ASSUMED) ----------
    ex = cfg["exposure"]
    grid = exposure.scenario_grid(d_main.risk_score, d_main, [0.10, 0.20, 0.30], [0.25, 0.5, 1.0], [0.25, 0.5, 1.0])

    def scn(score):
        return exposure.reallocation_backtest(score, d_main.next_value, d_main.next_late_value, d_main.next_n_late,
                                              d_main.next_n_lines, ex["top_risk_share_to_reallocate"], 0.5,
                                              ex["reallocation_late_rate_improvement"])
    base_scn = scn(d_main.risk_score)
    raw_scn = scn(scores_main["Baseline: raw late rate"])
    eb_scn = scn(scores_main["Shrunk late rate (EB) only"])
    exp_sens = exposure.exposure_sensitivity(comp, ex, ex["sensitivity"])
    sc_only = comp[comp.is_scored]
    conc_val = exposure.concentration(comp.value_usd)
    conc_risk = exposure.concentration(comp.value_at_risk)
    # ---------- write artifacts ----------
    comp.to_parquet(art / "vendor_scores.parquet", index=False)
    d_main.to_parquet(art / "oot_main.parquet", index=False)
    ev_all.to_csv(art / "oot_backtest_metrics.csv", index=False)
    ev_main.to_csv(art / "oot_metrics_main.csv", index=False)
    dec.to_csv(art / "oot_quintiles.csv", index=False)
    calib.to_csv(art / "oot_shrinkage_calibration.csv", index=False)
    prof.to_csv(art / "archetype_profiles.csv", index=False)
    z.to_csv(art / "archetype_centroids_z.csv")
    ktab.to_csv(art / "cluster_selection.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(art / "weight_sensitivity_ranking.csv", index=False)
    grid.to_csv(art / "scenario_grid.csv", index=False)
    exp_sens.to_csv(art / "exposure_sensitivity.csv", index=False)
    top = comp[comp.is_scored].nsmallest(25, "risk_rank")[
        ["risk_rank", "vendor", "top_country", "top_product_group", "n_lines", "value_usd", "late_rate_raw",
         "late_rate_shrunk", "late_rate_lo90", "late_rate_hi90", "mean_days_late_shrunk", "freight_ratio",
         "risk_score", "risk_score_lo90", "risk_score_hi90", "archetype", "value_at_risk"]]
    top.to_csv(art / "top_risk_vendors.csv", index=False)
    tier = (comp.groupby("risk_tier").agg(n_vendors=("vendor", "size"), value_usd=("value_usd", "sum"),
                                          late_value_usd=("late_value_usd", "sum"),
                                          value_at_risk=("value_at_risk", "sum"), n_lines=("n_lines", "sum"),
                                          n_late=("n_late", "sum"), exposure_ASSUMED=("total_exposure_ASSUMED", "sum"))
            .reset_index())
    tier["late_rate"] = tier.n_late / tier.n_lines
    tier["value_share"] = tier.value_usd / tier.value_usd.sum()
    tier["late_value_share"] = tier.late_value_usd / tier.late_value_usd.sum()
    tier.to_csv(art / "risk_tier_summary.csv", index=False)
    hi = comp[comp.risk_tier == "High"]
    mat = comp[comp.is_scored & (comp.n_lines >= 50)]
    unsc = comp[~comp.is_scored]
    summary = {
        "n_vendors_external": int(len(comp)), "n_vendors_scored": int(comp.is_scored.sum()),
        "n_vendors_unscored": int(len(unsc)), "unscored_line_share": float(unsc.n_lines.sum() / comp.n_lines.sum()),
        "unscored_value_share": float(unsc.value_usd.sum() / comp.value_usd.sum()),
        "priors": {k: [v.a, v.b, v.mean, v.strength] for k, v in priors.items()},
        "pop_mean_days_late": pop_days, "pooled_freight_ratio": attrs["pooled_freight_ratio"],
        "freight_prior_value": attrs["freight_prior_value"],
        "overall_ext_late_rate": float(comp.n_late.sum() / comp.n_lines.sum()),
        "shrinkage": {"median_abs_shift_late_rate_pp_lt30_lines": float(
            (comp.late_rate_shrunk - comp.late_rate_raw).abs()[comp.n_lines < 30].median() * 100),
            "median_abs_shift_late_rate_pp_ge100_lines": float(
                (comp.late_rate_shrunk - comp.late_rate_raw).abs()[comp.n_lines >= 100].median() * 100),
            "vendors_with_zero_late_raw": int((comp.n_late == 0).sum()),
            "min_shrunk_late_rate_among_zero_late": float(comp.late_rate_shrunk[comp.n_late == 0].min()),
            "max_raw_late_rate": float(comp.late_rate_raw.max()), "n_vendors_raw_rate_100pct": int(
                (comp.late_rate_raw == 1).sum())},
        "high_tier": {"n": int(len(hi)), "value_share": float(hi.value_usd.sum() / comp.value_usd.sum()),
                      "late_value_share": float(hi.late_value_usd.sum() / comp.late_value_usd.sum()),
                      "late_rate": float(hi.n_late.sum() / hi.n_lines.sum()), "median_lines": float(hi.n_lines.median())},
        "material_vendors_50plus": {"n": int(len(mat)), "value_share": float(mat.value_usd.sum() / comp.value_usd.sum())},
        "top_vendors": top.head(5)[["vendor", "n_lines", "late_rate_shrunk", "risk_score"]].to_dict("records"),
        "interval_width_median": {
            "small_lt30": float((comp.risk_score_hi90 - comp.risk_score_lo90)[comp.n_lines < 30].median()),
            "large_ge100": float((comp.risk_score_hi90 - comp.risk_score_lo90)[comp.n_lines >= 100].median())},
        "clusters": {"k": int(ktab.loc[ktab.silhouette.idxmax(), "k"]), "silhouette": float(ktab.silhouette.max()),
                     "n_clustered": int(len(elig)), "names": list(names.values())},
        "oot_main": meta_main, "oot_backtests": metas, "auc_deltas": deltas,
        "scenario_base_ASSUMED": base_scn, "scenario_raw_baseline_ASSUMED": raw_scn, "scenario_eb_ASSUMED": eb_scn,
        "exposure_totals": {"value_usd": float(comp.value_usd.sum()), "late_value_usd": float(comp.late_value_usd.sum()),
                            "value_at_risk": float(comp.value_at_risk.sum()),
                            "wc_excess_delay": float(comp.wc_excess_delay.sum()),
                            "total_exposure_ASSUMED": float(comp.total_exposure_ASSUMED.sum()),
                            "scored_value_share": float(sc_only.value_usd.sum() / comp.value_usd.sum()),
                            "top10pct_value_at_risk_share": float(conc_risk.cum_share.iloc[int(0.1 * len(comp)) - 1]),
                            "top10pct_value_share": float(conc_val.cum_share.iloc[int(0.1 * len(comp)) - 1])},
        "exposure_params_ASSUMED": {k: v for k, v in ex.items() if k != "sensitivity"},
    }
    _dump(summary, "train_summary.json")
    plots.make_all(art, resolve(cfg["paths"]["figures"]))
    con.close()
    return {k: summary[k] for k in ["n_vendors_scored", "high_tier", "clusters", "auc_deltas",
                                    "scenario_base_ASSUMED"]} | {"oot_main": ev_main.round(4).to_dict("records")}
