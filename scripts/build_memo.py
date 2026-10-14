"""Generate the 2-page procurement recommendation memo (markdown + PDF) from computed artifacts only.

Every number in the memo is read from artifacts/ produced by the pipeline; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import markdown
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS = ROOT / "artifacts", ROOT / "docs"


def brl(x):
    return f"BRL {x / 1e6:,.2f}m" if abs(x) >= 1e6 else f"BRL {x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"BRL {x:,.0f}"


def build() -> str:
    q = json.loads((ART / "sql_summary.json").read_text())
    t = json.loads((ART / "train_summary.json").read_text())
    r = json.loads((ART / "r_glm/r_summary.json").read_text())
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    ev = pd.read_csv(ART / "oot_metrics_main.csv").set_index(["target", "score"])
    tier = pd.read_csv(ART / "risk_tier_summary.csv").set_index("risk_tier")
    prof = pd.read_csv(ART / "archetype_profiles.csv").set_index("archetype")
    cal = pd.read_csv(ART / "oot_shrinkage_calibration.csv").set_index("vol_band")
    sc = pd.read_parquet(ART / "seller_scores.parquet")
    dec = pd.read_csv(ART / "oot_deciles.csv").set_index("decile")
    grid = pd.read_csv(ART / "scenario_grid.csv")
    m, e, ex = t["oot_main"], t["exposure_totals"], cfg["exposure"]
    ho, lt = ev.loc["Seller hand-off late"], ev.loc["End-to-end late delivery"]
    C, EB, RH, RL, VO = ("Composite risk score (shrunk)", "Shrunk hand-off late rate (EB) only",
                         "Baseline: raw hand-off late rate", "Baseline: raw late rate",
                         "Baseline: volume only (smaller = riskier)")
    d = t["auc_deltas"]
    mat = sc[sc.n_items >= 50]
    watch = mat[mat.risk_tier.isin(["High", "Elevated"])]
    sd = prof.loc["Slow dispatchers"]
    scn = t["scenario_base_ASSUMED"]
    hs = t["scenario_handoff_items_ASSUMED"][C]
    g_max = grid[(grid.top_share == 0.2) & (grid.realloc_frac == 1.0) & (grid.improvement_ASSUMED == 1.0)].iloc[0]
    hi = tier.loc["High"]

    md = f"""# Procurement Recommendation: Supplier Delivery-Risk & Exposure

**To:** Head of Procurement / Marketplace Ops | **From:** Supplier Analytics | **Data:** Olist Brazilian E-Commerce,
purchases {q['date_min']} to {q['date_max']} ({q['n_items']:,} items, {q['n_sellers_active']:,} sellers). Every figure
is computed by the pipeline in this repository. **Olist sellers are used as proxies for suppliers.** Items marked
**ASSUMED** are scenario inputs, not observed results.

## 1. The question
Which vendors create the greatest delivery-risk and working-capital exposure - and can a score built today tell us
who will cause problems next period?

## 2. What the data says (observed)
* **{q['order_late_rate']:.1%} of {q['n_delivered_orders']:,} delivered orders arrived after the promised date**
  (mean {q['order_mean_delay_days_when_late']:.1f} days late). Late orders average a {q['review_late']:.2f}/5 review vs
  {q['review_on_time']:.2f} on time.
* **The seller-controlled leg matters most.** {q['item_handoff_late_rate']:.1%} of items were handed to the carrier after
  the shipping limit; those orders were late {q['late_rate_if_handoff_late']:.0%} of the time vs
  {q['late_rate_if_handoff_on_time']:.1%} otherwise. Controlling for lane, region, category, freight and season, a late
  hand-off multiplies the odds of a late delivery by **{r['or_handoff']:.2f}** (95% CI {r['or_handoff_lo']:.2f}-
  {r['or_handoff_hi']:.2f}; logistic GLM, R).
* **But end-to-end lateness is mostly not a seller trait.** In a seller random-intercept model only
  {r['glmm_icc_without_handoff']:.1%} of latent variance sits between sellers ({r['glmm_icc']:.1%} after hand-off is
  controlled). Platform-wide spikes (Nov-2017, Feb-Mar 2018) dominate.
* **Spend is concentrated:** the top 10% of sellers hold {q['top10pct_sellers_gmv_share']:.0%} of GMV and
  {e['top10pct_gmv_at_risk_share']:.0%} of GMV-at-risk.

![](figures/handoff_delay_vs_late.png) ![](figures/oot_validation_auc.png)

## 3. Does the score predict next period? (out-of-time, scored at {m['score_end'][:10]})
{m['n_sellers_evaluated']:,} sellers scored on pre-{m['score_end'][:4]} data, evaluated on items bought
{m['score_end'][:10]} to {m['validate_end'][:10]} ({m['next_items_covered_share']:.0%} of next-period items).

| Score | AUC: next hand-off late | Top-10% lift | AUC: next late delivery |
|---|---|---|---|
| EB-shrunk hand-off rate | **{ho.loc[EB, 'item_auc']:.3f}** | {ho.loc[EB, 'lift_top10']:.2f}x | {lt.loc[EB, 'item_auc']:.3f} |
| Composite risk score | {ho.loc[C, 'item_auc']:.3f} | {ho.loc[C, 'lift_top10']:.2f}x | {lt.loc[C, 'item_auc']:.3f} |
| Raw hand-off rate (baseline) | {ho.loc[RH, 'item_auc']:.3f} | {ho.loc[RH, 'lift_top10']:.2f}x | {lt.loc[RH, 'item_auc']:.3f} |
| Raw late rate (baseline) | {ho.loc[RL, 'item_auc']:.3f} | {ho.loc[RL, 'lift_top10']:.2f}x | {lt.loc[RL, 'item_auc']:.3f} |
| Volume only (baseline) | {ho.loc[VO, 'item_auc']:.3f} | {ho.loc[VO, 'lift_top10']:.2f}x | {lt.loc[VO, 'item_auc']:.3f} |

* Shrinkage beats the raw rate on hand-off lateness (dAUC {d['eb_handoff_vs_raw_handoff_handoff']['delta']:+.3f},
  95% CI {d['eb_handoff_vs_raw_handoff_handoff']['lo95']:+.3f} to {d['eb_handoff_vs_raw_handoff_handoff']['hi95']:+.3f})
  and halves the error for sellers with <5 deliveries (MAE {cal.loc['<5', 'mae_raw']:.3f} -> {cal.loc['<5', 'mae_shrunk']:.3f}).
* The riskiest score decile had a next-period hand-off late rate of {dec.loc[10, 'next_handoff_late_rate']:.1%} vs
  {dec.loc[1, 'next_handoff_late_rate']:.1%} in the safest. **No score predicts end-to-end lateness well** (AUC
  <= {lt.item_auc.max():.3f}); it is carrier/lane/calendar driven.

## 4. Who to act on
* **{len(watch):,} material sellers (>=50 items) are in the High/Elevated tiers**, holding
  {watch.gmv.sum() / sc.gmv.sum():.1%} of GMV and {watch.late_gmv.sum() / sc.late_gmv.sum():.1%} of late GMV. The
  full High tier ({int(hi.n_sellers)} sellers) carries {hi.gmv_share:.1%} of GMV but {hi.late_gmv_share:.1%} of late GMV,
  yet its median seller has only {t['high_tier']['median_items']:.0f} items - treat these as *unknown*, not *bad*.
* **"Slow dispatchers" archetype:** {int(sd.n_sellers)} sellers, {sd.gmv_share:.1%} of GMV, median hand-off late rate
  {sd.handoff_late:.1%} vs {prof.loc['High-volume core suppliers', 'handoff_late']:.1%} for core suppliers - the priority
  group for dispatch-SLA conversations. (Clusters are soft: silhouette {t['clusters']['silhouette']:.2f}.)

![](figures/top_risk_material_sellers.png) ![](figures/seller_archetypes.png)

## 5. Exposure and the reallocation scenario
* Observed: GMV {brl(e['gmv'])}, late GMV {brl(e['late_gmv'])}, GMV-at-risk {brl(e['gmv_at_risk'])}, freight
  {brl(e['freight'])}. Working-capital proxy: {brl(e['wc_in_transit'])} of GMV in transit on average; delays add
  {brl(e['wc_excess_delay'])}.
* **ASSUMED** costs ({ex['late_cost_pct_of_gmv']:.0%} of GMV per late item, {ex['freight_absorb_pct']:.0%} freight
  absorbed, {ex['cancel_cost_pct_of_gmv']:.0%} per cancellation) give total exposure of
  {brl(e['total_exposure_ASSUMED'])} - a scenario figure, not a P&L.
* **Back-tested reallocation (ASSUMED):** moving {scn['realloc_frac']:.0%} of the top-{scn['top_share']:.0%} sellers'
  next-period volume, closing {scn['improvement_ASSUMED']:.0%} of the gap, would have avoided only
  {scn['late_gmv_reduction_pct']:.2%} of late GMV ({brl(scn['late_gmv_avoided'])}); even the most aggressive grid cell
  (top 20%, all volume, full gap) gives {g_max.late_gmv_reduction_pct:.1%}. On the seller-controlled leg the same
  move cuts late hand-offs by {hs['late_items_reduction_pct']:.1%}.

## 6. Recommendations & trade-offs
1. **Adopt the EB-shrunk hand-off late rate as the supplier SLA KPI**, reviewed monthly with intervals; it is the
   most predictive, least noisy signal and is what sellers control.
2. **Run dispatch-SLA reviews with the {len(watch):,} material High/Elevated sellers**, starting with "Slow dispatchers";
   use the composite only for triage (it adds review/cancellation/freight context but predicts no better).
3. **Do not expect volume reallocation to fix customer lateness**; carrier and lane performance (North/Northeast
   lanes, peak months) need a logistics workstream.
4. **Onboarding watch-list, not penalties, for low-volume sellers** ({t['low_volume_sellers']:,} have <5 items).

*Limitations: sellers are marketplace merchants, not contracted suppliers; costs are ASSUMED; scores cover
{m['next_items_covered_share']:.0%} of next-period items (new sellers unscored); multi-seller orders share timestamps;
GLM effects are associations. Data: Olist, CC BY-NC-SA 4.0.*
"""
    return md


CSS = """
@page { size: A4; margin: 12mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.1pt; line-height: 1.28; color: #1a1a1a; }
h1 { font-size: 13pt; color: #1f4e79; margin: 0 0 4px 0; border-bottom: 2px solid #1f4e79; }
h2 { font-size: 9.6pt; color: #1f4e79; margin: 6px 0 2px 0; }
ul, ol { margin: 2px 0 2px 16px; padding: 0; } li { margin-bottom: 2px; } p { margin: 3px 0; }
table { border-collapse: collapse; margin: 4px 0; } th, td { border: 1px solid #bbb; padding: 1px 6px; }
th { background: #e8eef5; }
img { width: 49%; } p:has(img) { text-align: center; }
"""


def main():
    md = build()
    (DOCS / "procurement_memo.md").write_text(md)
    html = (f"<html><head><style>{CSS}</style></head><body>"
            f"{markdown.markdown(md, extensions=['tables', 'attr_list'])}</body></html>")
    from weasyprint import HTML

    doc = HTML(string=html, base_url=str(DOCS)).render()
    doc.write_pdf(DOCS / "procurement_memo.pdf")
    print(f"memo written ({len(doc.pages)} pages): {DOCS / 'procurement_memo.pdf'}")


if __name__ == "__main__":
    main()
