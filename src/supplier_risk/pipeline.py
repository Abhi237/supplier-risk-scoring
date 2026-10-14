"""End-to-end pipeline steps used by the CLI: sql -> train (score, validate, cluster, exposure, scenarios)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clustering, exposure, plots, scoring, validation
from .config import load_config, resolve
from .db import connect, run_sql_model, set_oot_window

EXPORT = ["dq_report", "fct_items", "fct_orders_delivered", "seller_summary", "seller_month_panel",
          "platform_month"]


def _art():
    p = resolve(load_config()["paths"]["artifacts"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dump(obj, name):
    (_art() / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def sql_summary(con) -> dict:
    q = lambda s: con.execute(s).fetchone()  # noqa: E731
    dq = dict(con.execute("SELECT check_name, n FROM dq_report").fetchall())
    n_items, n_orders, n_sellers, gmv, freight = q(
        "SELECT COUNT(*), COUNT(DISTINCT order_id), COUNT(DISTINCT seller_id), SUM(price)::DOUBLE, "
        "SUM(freight_value)::DOUBLE "
        "FROM fct_items")
    late, ho, canc, rev = q("SELECT AVG(is_late::INT), AVG(is_handoff_late::INT), AVG(is_canceled::INT), "
                            "AVG(review_score) FROM fct_items")
    o_late, o_n, o_med, o_mean_delay = q(
        "SELECT AVG(is_late::INT), COUNT(*), MEDIAN(delivery_days), AVG(CASE WHEN is_late THEN delay_days END) "
        "FROM fct_orders_delivered")
    rev_late, rev_ontime = q("SELECT AVG(CASE WHEN is_late THEN review_score END), "
                             "AVG(CASE WHEN NOT is_late THEN review_score END) FROM fct_orders_delivered")
    late_given_ho = q("SELECT AVG(CASE WHEN is_handoff_late THEN is_late::INT END), "
                      "AVG(CASE WHEN NOT is_handoff_late THEN is_late::INT END) FROM fct_orders_delivered")
    dmin, dmax = q("SELECT MIN(purchase_ts)::DATE, MAX(purchase_ts)::DATE FROM fct_items")
    conc = con.execute("SELECT gmv FROM seller_summary ORDER BY gmv DESC, seller_id").df().gmv
    multi = q("SELECT AVG(is_multi_seller_order::INT) FROM fct_orders_delivered")[0]
    return {
        "dq": dq, "n_items": n_items, "n_orders_with_items": n_orders, "n_sellers_active": n_sellers,
        "gmv_brl": float(gmv), "freight_brl": float(freight), "freight_to_price": float(freight / gmv),
        "date_min": str(dmin), "date_max": str(dmax),
        "item_late_rate": float(late), "item_handoff_late_rate": float(ho), "item_cancel_rate": float(canc),
        "item_mean_review": float(rev), "n_delivered_orders": int(o_n), "order_late_rate": float(o_late),
        "order_median_delivery_days": float(o_med), "order_mean_delay_days_when_late": float(o_mean_delay),
        "review_late": float(rev_late), "review_on_time": float(rev_ontime),
        "late_rate_if_handoff_late": float(late_given_ho[0]), "late_rate_if_handoff_on_time": float(late_given_ho[1]),
        "top10pct_sellers_gmv_share": float(conc.head(int(round(0.1 * len(conc)))).sum() / conc.sum()),
        "multi_seller_order_share": float(multi),
    }


def run_sql() -> dict:
    cfg = load_config()
    con = connect()
    run_sql_model(con, cfg=cfg)
    out = resolve("data/processed")
    order = {"fct_items": "order_id, order_item_id", "fct_orders_delivered": "order_id", "seller_summary": "seller_id",
             "seller_month_panel": "seller_id, purchase_month", "platform_month": "purchase_month",
             "dq_report": "check_name"}
    for t in EXPORT:
        con.execute(f"SELECT * FROM {t} ORDER BY {order[t]}").df().to_parquet(out / f"{t}.parquet", index=False)
    con.execute("SELECT * FROM dq_report ORDER BY check_name").df().to_csv(_art() / "dq_report.csv", index=False)
    # order-grain table for R (no customer identifiers)
    con.execute("SELECT * EXCLUDE (order_id) FROM fct_orders_delivered ORDER BY order_id").df().to_csv(
        out / "orders_delivered_for_r.csv", index=False)
    s = sql_summary(con)
    _dump(s, "sql_summary.json")
    con.close()
    return s


def review_counts(con, where: str = "TRUE") -> pd.DataFrame:
    """Per-seller counts of review scores 1..5 at ITEM grain (matches seller_summary.n_reviews)."""
    df = con.execute(f"SELECT seller_id, review_score, COUNT(*) AS n FROM fct_items CROSS JOIN params "
                     f"WHERE review_score IS NOT NULL AND {where} GROUP BY 1, 2 ORDER BY 1, 2").df()
    return df.pivot(index="seller_id", columns="review_score", values="n").reindex(columns=[1, 2, 3, 4, 5]).fillna(0)


def score_sellers(df: pd.DataFrame, cfg: dict, weights: dict | None = None):
    priors = scoring.fit_priors(df, cfg)
    comp = scoring.score_components(df, priors)
    ref = scoring.percentile_ref(comp)
    comp["risk_score"] = scoring.composite(comp, weights or cfg["scoring"]["weights"], ref)
    return comp, priors, ref


def oot_split(con, cfg, score_end, validate_end, with_scores: bool = False):
    set_oot_window(con, score_end, validate_end)
    feat = con.execute("SELECT * FROM oot_features ORDER BY seller_id").df()
    outc = con.execute("SELECT * FROM oot_outcomes ORDER BY seller_id").df()
    comp, priors, ref = score_sellers(feat, cfg)
    pooled = comp.n_late.sum() / max(comp.n_delivered.sum(), 1)
    pooled_ho = comp.n_handoff_late.sum() / max(comp.n_handoff_obs.sum(), 1)
    d = comp.merge(outc, on="seller_id", how="inner")
    d = d[d.next_n_delivered > 0].reset_index(drop=True)
    scores = {
        "Composite risk score (shrunk)": d.risk_score.to_numpy(),
        "Shrunk late rate (EB) only": d.late_rate_shrunk.to_numpy(),
        "Shrunk hand-off late rate (EB) only": d.seller_handoff_late_rate.to_numpy(),
        "Baseline: raw late rate": d.late_rate_raw.fillna(pooled).to_numpy(),
        "Baseline: raw hand-off late rate": d.handoff_late_rate_raw.fillna(pooled_ho).to_numpy(),
        "Baseline: volume only (smaller = riskier)": -np.log1p(d.n_items.to_numpy()),
    }
    for nm, w in cfg["scoring"]["weight_sensitivity"].items():
        scores[f"Sensitivity: {nm} weights"] = scoring.composite(d, w, ref)
    ev_late = validation.evaluate(scores, d.next_n_late, d.next_n_delivered)
    ev_late.insert(0, "target", "End-to-end late delivery")
    ev_ho = validation.evaluate(scores, d.next_n_handoff_late, d.next_n_handoff_obs)
    ev_ho.insert(0, "target", "Seller hand-off late")
    ev = pd.concat([ev_late, ev_ho], ignore_index=True)
    nxt_all = outc[outc.next_n_delivered > 0]
    meta = {"score_end": score_end, "validate_end": validate_end, "n_sellers_scored": int(len(comp)),
            "n_sellers_evaluated": int(len(d)),
            "next_items_covered_share": float(d.next_n_delivered.sum() / nxt_all.next_n_delivered.sum()),
            "next_late_rate": float(d.next_n_late.sum() / d.next_n_delivered.sum()),
            "next_handoff_late_rate": float(d.next_n_handoff_late.sum() / d.next_n_handoff_obs.sum()),
            "max_delivered_ts_used": str(feat.max_delivered_ts_used.max()),
            "prior_late": [priors["late"].a, priors["late"].b]}
    return (ev, meta, d, scores) if with_scores else (ev, meta)


def train() -> dict:
    cfg = load_config()
    sc, art = cfg["scoring"], _art()
    con = connect()
    # ---------- 1. full-period scorecard ----------
    full = con.execute("SELECT * FROM seller_summary ORDER BY seller_id").df()
    comp, priors, ref = score_sellers(full, cfg)
    rc = review_counts(con)
    review_mu = float((full.mean_review_score.fillna(0) * full.n_reviews).sum() / full.n_reviews.sum())
    boot = scoring.bootstrap_scores(comp, rc, priors, sc["weights"], ref, review_mu, sc["n_bootstrap"],
                                    sc["random_state"])
    comp = comp.merge(boot, on="seller_id")
    comp["risk_tier"] = scoring.risk_tier(comp.risk_score)
    comp["risk_rank"] = comp.risk_score.rank(ascending=False, method="first").astype(int)
    comp["low_volume_flag"] = comp.n_items < sc["min_items_for_score"]
    for nm, w in sc["weight_sensitivity"].items():
        comp[f"risk_score_{nm}"] = scoring.composite(comp, w, ref)
    # ---------- 2. exposure ----------
    comp = exposure.seller_exposure(comp, cfg["exposure"])
    # ---------- 3. clustering ----------
    cl = cfg["clustering"]
    comp["log_gmv"] = np.log1p(comp.gmv)
    elig = comp[(comp.n_items >= cl["min_items"])].dropna(subset=cl["features"]).copy()
    labels, names, z, ktab, X = clustering.fit_archetypes(elig, cl["features"], cl["k_range"], cl["random_state"])
    elig["cluster"] = labels
    elig["archetype"] = elig.cluster.map(names)
    comp = comp.merge(elig[["seller_id", "cluster", "archetype"]], on="seller_id", how="left")
    comp["archetype"] = comp.archetype.fillna(f"Unclustered (<{cl['min_items']} items)")
    prof = (comp[comp.cluster.notna()].groupby("archetype")
            .agg(n_sellers=("seller_id", "size"), gmv=("gmv", "sum"), median_items=("n_items", "median"),
                 late_rate_shrunk=("late_rate_shrunk", "median"),
                 handoff_late=("seller_handoff_late_rate", "median"), review=("mean_review_score", "median"),
                 freight_to_price=("freight_to_price", "median"), cancel_rate=("cancel_rate", "median"),
                 median_delivery_days=("median_delivery_days", "median"), risk_score=("risk_score", "median"),
                 gmv_at_risk=("gmv_at_risk", "sum"))
            .reset_index().sort_values("risk_score", ascending=False))
    prof["gmv_share"] = prof.gmv / comp.gmv.sum()
    z.index = [names[i] for i in z.index]
    # PCA 2-D coordinates for plotting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0).fit(X)
    xy = pca.transform(X)
    elig["pc1"], elig["pc2"] = xy[:, 0], xy[:, 1]
    comp = comp.merge(elig[["seller_id", "pc1", "pc2"]], on="seller_id", how="left")
    # ---------- 4. weight sensitivity on full-period ranking ----------
    base_top = set(comp.nsmallest(100, "risk_rank").seller_id)
    sens_rows = []
    for nm in sc["weight_sensitivity"]:
        alt_top = set(comp.nlargest(100, f"risk_score_{nm}").seller_id)
        sens_rows.append({"weights": nm, "spearman_vs_base": comp.risk_score.corr(comp[f"risk_score_{nm}"],
                                                                                    method="spearman"),
                          "top100_overlap": len(base_top & alt_top) / 100})
    # ---------- 5. out-of-time validation ----------
    ev_all, metas = [], []
    for s_end, v_end in sc["backtest_splits"]:
        ev, meta = oot_split(con, cfg, s_end, v_end)
        ev.insert(0, "split", f"{s_end}->{v_end}")
        ev_all.append(ev)
        metas.append(meta)
    ev_all = pd.concat(ev_all, ignore_index=True)
    s_end, v_end = sc["score_period_end"], sc["validate_period_end"]
    ev_main, meta_main, d_main, scores_main = oot_split(con, cfg, s_end, v_end, with_scores=True)
    deltas = {}
    for tgt, k, n in [("late", d_main.next_n_late, d_main.next_n_delivered),
                      ("handoff", d_main.next_n_handoff_late, d_main.next_n_handoff_obs)]:
        deltas[f"composite_vs_raw_late_{tgt}"] = validation.bootstrap_auc_delta(
            d_main.risk_score, scores_main["Baseline: raw late rate"], k, n, 200, sc["random_state"])
        deltas[f"eb_handoff_vs_raw_handoff_{tgt}"] = validation.bootstrap_auc_delta(
            scores_main["Shrunk hand-off late rate (EB) only"], scores_main["Baseline: raw hand-off late rate"],
            k, n, 200, sc["random_state"])
        deltas[f"composite_vs_volume_{tgt}"] = validation.bootstrap_auc_delta(
            d_main.risk_score, scores_main["Baseline: volume only (smaller = riskier)"], k, n, 200,
            sc["random_state"])
    # decile table for main split
    dd = d_main.assign(decile=pd.qcut(d_main.risk_score.rank(method="first"), 10, labels=False) + 1)
    dec = (dd.groupby("decile").agg(n_sellers=("seller_id", "size"), score_min=("risk_score", "min"),
                                    next_late=("next_n_late", "sum"), next_delivered=("next_n_delivered", "sum"),
                                    next_gmv=("next_gmv", "sum"), next_late_gmv=("next_late_gmv", "sum"),
                                    next_ho_late=("next_n_handoff_late", "sum"),
                                    next_ho_obs=("next_n_handoff_obs", "sum"))
           .reset_index())
    dec["next_late_rate"] = dec.next_late / dec.next_delivered
    dec["next_handoff_late_rate"] = dec.next_ho_late / dec.next_ho_obs
    # raw-rate decile for comparison
    dd["raw_decile"] = pd.qcut(pd.Series(scores_main["Baseline: raw late rate"]).rank(method="first"), 10,
                               labels=False) + 1
    dec_raw = dd.groupby("raw_decile").apply(lambda g: g.next_n_late.sum() / g.next_n_delivered.sum(),
                                             include_groups=False)
    dec["next_late_rate_raw_baseline"] = dec_raw.values
    # small vs large seller calibration: shrunk vs raw error
    dd["vol_band"] = pd.cut(dd.n_delivered, [-1, 4, 19, 99, 1e9], labels=["<5", "5-19", "20-99", "100+"])
    nl = dd.next_n_late / dd.next_n_delivered
    calib = (dd.assign(err_raw=(dd.late_rate_raw.fillna(meta_main["next_late_rate"]) - nl).abs() * dd.next_n_delivered,
                       err_eb=(dd.late_rate_shrunk - nl).abs() * dd.next_n_delivered)
             .groupby("vol_band", observed=True)
             .agg(n_sellers=("seller_id", "size"), next_items=("next_n_delivered", "sum"),
                  err_raw=("err_raw", "sum"), err_eb=("err_eb", "sum")).reset_index())
    calib["mae_raw"] = calib.err_raw / calib.next_items
    calib["mae_shrunk"] = calib.err_eb / calib.next_items
    calib = calib.drop(columns=["err_raw", "err_eb"])
    # ---------- 6. scenario back-test (ASSUMED) ----------
    ex = cfg["exposure"]
    grid = exposure.scenario_grid(d_main.risk_score, d_main, [0.05, 0.10, 0.20], [0.25, 0.5, 1.0],
                                  [0.25, 0.5, 1.0])
    base_scn = exposure.reallocation_backtest(d_main.risk_score, d_main.next_gmv, d_main.next_late_gmv,
                                              d_main.next_n_late, d_main.next_n_delivered,
                                              ex["top_risk_share_to_reallocate"], 0.5,
                                              ex["reallocation_late_rate_improvement"])
    raw_scn = exposure.reallocation_backtest(scores_main["Baseline: raw late rate"], d_main.next_gmv,
                                             d_main.next_late_gmv, d_main.next_n_late, d_main.next_n_delivered,
                                             ex["top_risk_share_to_reallocate"], 0.5,
                                             ex["reallocation_late_rate_improvement"])
    ho_scn = {nm: exposure.reallocation_backtest(scores_main[nm], d_main.next_n_handoff_obs,
                                                 d_main.next_n_handoff_late, d_main.next_n_handoff_late,
                                                 d_main.next_n_handoff_obs, ex["top_risk_share_to_reallocate"],
                                                 0.5, ex["reallocation_late_rate_improvement"])
              for nm in ["Composite risk score (shrunk)", "Shrunk hand-off late rate (EB) only",
                         "Baseline: raw hand-off late rate"]}
    ho_scn = {nm: {k.replace("_gmv", "_items").replace("moved_items", "moved_items"): v for k, v in r.items()}
              for nm, r in ho_scn.items()}
    exp_sens = exposure.exposure_sensitivity(comp, ex, ex["sensitivity"])
    conc_gmv = exposure.concentration(comp.gmv)
    conc_risk = exposure.concentration(comp.gmv_at_risk)
    # ---------- write artifacts ----------
    comp.to_parquet(art / "seller_scores.parquet", index=False)
    d_main.to_parquet(art / "oot_main.parquet", index=False)
    ev_all.to_csv(art / "oot_backtest_metrics.csv", index=False)
    ev_main.to_csv(art / "oot_metrics_main.csv", index=False)
    dec.to_csv(art / "oot_deciles.csv", index=False)
    calib.to_csv(art / "oot_shrinkage_calibration.csv", index=False)
    prof.to_csv(art / "archetype_profiles.csv", index=False)
    z.to_csv(art / "archetype_centroids_z.csv")
    ktab.to_csv(art / "cluster_selection.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(art / "weight_sensitivity_ranking.csv", index=False)
    grid.to_csv(art / "scenario_grid.csv", index=False)
    exp_sens.to_csv(art / "exposure_sensitivity.csv", index=False)
    top = comp.nsmallest(25, "risk_rank")[["risk_rank", "seller_id", "seller_state", "top_category", "n_items", "gmv",
                                           "late_rate_raw", "late_rate_shrunk", "late_rate_lo90", "late_rate_hi90",
                                           "seller_handoff_late_rate", "mean_review_score", "risk_score",
                                           "risk_score_lo90", "risk_score_hi90", "archetype", "gmv_at_risk"]]
    top.to_csv(art / "top25_risk_sellers.csv", index=False)
    hi = comp[comp.risk_tier == "High"]
    tier = (comp.groupby("risk_tier").agg(n_sellers=("seller_id", "size"), gmv=("gmv", "sum"),
                                          late_gmv=("late_gmv", "sum"), gmv_at_risk=("gmv_at_risk", "sum"),
                                          n_items=("n_items", "sum"), n_late=("n_late", "sum"),
                                          n_delivered=("n_delivered", "sum"),
                                          exposure_ASSUMED=("total_exposure_ASSUMED", "sum")).reset_index())
    tier["late_rate"] = tier.n_late / tier.n_delivered
    tier["gmv_share"] = tier.gmv / tier.gmv.sum()
    tier["late_gmv_share"] = tier.late_gmv / tier.late_gmv.sum()
    tier.to_csv(art / "risk_tier_summary.csv", index=False)
    # material sellers: enough volume that the score is not driven by noise
    mat = comp[comp.n_items >= 50]
    summary = {
        "n_sellers_scored": int(len(comp)), "priors": {k: [v.a, v.b, v.mean, v.strength] for k, v in priors.items()},
        "review_mu": review_mu, "low_volume_sellers": int(comp.low_volume_flag.sum()),
        "high_tier": {"n": int(len(hi)), "gmv_share": float(hi.gmv.sum() / comp.gmv.sum()),
                      "late_gmv_share": float(hi.late_gmv.sum() / comp.late_gmv.sum()),
                      "late_rate": float(hi.n_late.sum() / hi.n_delivered.sum()),
                      "median_items": float(hi.n_items.median())},
        "material_sellers_50plus": {"n": int(len(mat)), "gmv_share": float(mat.gmv.sum() / comp.gmv.sum())},
        "top_material_risk": mat.nsmallest(10, "risk_rank")[["seller_id", "seller_state", "n_items", "gmv",
                                                            "late_rate_shrunk", "risk_score"]].to_dict("records"),
        "interval_width_median": {"small_lt20": float((comp.risk_score_hi90 - comp.risk_score_lo90)[comp.n_items < 20]
                                                     .median()),
                                  "large_ge100": float((comp.risk_score_hi90 - comp.risk_score_lo90)[
                                      comp.n_items >= 100].median())},
        "clusters": {"k": int(ktab.loc[ktab.silhouette.idxmax(), "k"]),
                     "silhouette": float(ktab.silhouette.max()), "n_clustered": int(len(elig)),
                     "names": list(names.values())},
        "oot_main": meta_main, "oot_backtests": metas,
        "auc_deltas": deltas,
        "scenario_base_ASSUMED": base_scn, "scenario_raw_baseline_ASSUMED": raw_scn,
        "scenario_handoff_items_ASSUMED": ho_scn,
        "exposure_totals": {"gmv": float(comp.gmv.sum()), "late_gmv": float(comp.late_gmv.sum()),
                            "gmv_at_risk": float(comp.gmv_at_risk.sum()), "freight": float(comp.freight.sum()),
                            "wc_in_transit": float(comp.wc_in_transit.sum()),
                            "wc_excess_delay": float(comp.wc_excess_delay.sum()),
                            "total_exposure_ASSUMED": float(comp.total_exposure_ASSUMED.sum()),
                            "top10pct_gmv_at_risk_share": float(conc_risk.cum_share.iloc[int(0.1 * len(comp)) - 1]),
                            "top10pct_gmv_share": float(conc_gmv.cum_share.iloc[int(0.1 * len(comp)) - 1])},
        "exposure_params_ASSUMED": {k: v for k, v in ex.items() if k != "sensitivity"},
    }
    _dump(summary, "train_summary.json")
    plots.make_all(art, resolve(cfg["paths"]["figures"]))
    con.close()
    return {k: summary[k] for k in ["n_sellers_scored", "high_tier", "clusters", "auc_deltas",
                                    "scenario_base_ASSUMED"]} | {"oot_main": ev_main.round(4).to_dict("records")}
