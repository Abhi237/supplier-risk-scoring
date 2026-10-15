"""Generate the procurement recommendation memo (markdown + PDF) from computed artifacts only.

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
C, EB, RAW = "Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Baseline: raw late rate"
VOL, LANE = "Baseline: volume only (smaller = riskier)", "Baseline: lane mix only (country / mode)"


def usd(x):
    return f"USD {x / 1e6:,.1f}m" if abs(x) >= 1e6 else f"USD {x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"USD {x:,.0f}"


def build() -> str:
    q = json.loads((ART / "sql_summary.json").read_text())
    t = json.loads((ART / "train_summary.json").read_text())
    r = json.loads((ART / "r_glm/r_summary.json").read_text())
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    ev = pd.read_csv(ART / "oot_metrics_main.csv")
    late = ev[ev.target.str.startswith("Late")].set_index("score")
    cs_all = pd.read_csv(ART / "cold_start.csv")
    cs = cs_all[cs_all.split == f"{t['oot_main']['score_end'][:10]}->{t['oot_main']['validate_end'][:10]}"].iloc[0]
    n_inside = int(cs_all.prior_inside_ci.sum())
    bt = pd.read_csv(ART / "oot_backtest_metrics.csv")
    btl = bt[bt.target.str.startswith("Late")].pivot(index="split", columns="score", values="line_auc")
    sc = pd.read_parquet(ART / "vendor_scores.parquet")
    top = sc[(sc.n_lines >= 50) & sc.risk_tier.isin(["High", "Elevated"])].sort_values("risk_rank")
    prof = pd.read_csv(ART / "archetype_profiles.csv")
    m, e, ex, d = t["oot_main"], t["exposure_totals"], cfg["exposure"], t["auc_deltas"]
    scn, raw = t["scenario_base_ASSUMED"], t["scenario_raw_baseline_ASSUMED"]
    sh, hi = t["shrinkage"], t["high_tier"]
    dc = d["composite_vs_raw_late_late"]
    top_rows = "\n".join(
        f"| {v.vendor} | {v.top_country} | {int(v.n_lines)} | {v.late_rate_raw:.1%} | {v.late_rate_shrunk:.1%} "
        f"[{v.late_rate_lo90:.1%}-{v.late_rate_hi90:.1%}] | {v.risk_score:.0f} [{v.risk_score_lo90:.0f}-{v.risk_score_hi90:.0f}] |"
        for v in top.itertuples())

    md = f"""# Procurement memo: delivery-risk scorecard for {t['n_vendors_external']} health-commodity vendors

**To:** supply-chain / vendor-management lead  **Basis:** USAID / PEPFAR SCMS Delivery History (file date 2015-09-29),
{q['n_lines']:,} order lines, scheduled {q['scheduled_min'][:4]} to {q['scheduled_max'][:4]}. Every figure below is read from the pipeline
outputs. **The buyer is a US government health-supply programme and deliveries go to PEPFAR-supported countries; none is a
US-domestic delivery.** On time means delivered-to-client date <= scheduled date. **There is no vendor hand-off timestamp**, so
lateness mixes vendor, freight-forwarder and in-country effects. Scores describe **recorded delivery performance in this programme,
not overall vendor quality**, and rest on limited samples.

## 1. Situation
{q['ext_late_rate']:.1%} of {q['ext_lines']:,} external-vendor lines were delivered after the scheduled date ({q['ext_severe_late_rate']:.1%}
more than {cfg['cleaning']['severe_late_days']} days late; {q['ext_late_value_share']:.1%} of line value, {usd(q['ext_late_value_usd'])} of
{usd(q['ext_value_usd'])}). The programme's own regional distribution centre (not a vendor, {q['rdc_line_share']:.0%} of lines) was late
{q['rdc_late_rate']:.1%} of the time. {q['ext_exact_match_share']:.1%} of vendor lines have delivered date equal to scheduled date exactly,
which may reflect back-filled scheduled dates, so the true on-time rate may be lower. Only delivered lines are in the file, so late rates are lower bounds.

## 2. Method
Empirical-Bayes beta-binomial shrinkage of each vendor's late and severe-late rates (prior late rate {t['priors']['late'][2]:.1%}, worth
{t['priors']['late'][3]:.0f} pseudo-lines), a six-component composite (weights fixed a priori, not tuned), Bayesian-bootstrap intervals,
a New / unscored cold-start tier for vendors with fewer than {cfg['scoring']['min_lines_for_score']} lines ({t['n_vendors_unscored']} of
{t['n_vendors_external']} vendors, {t['unscored_line_share']:.1%} of lines), out-of-time validation (features known before {m['score_end'][:10]},
outcomes {m['score_end'][:10]} to {m['validate_end'][:10]}), and R GLM / GLMM / negative-binomial models.

## 3. Findings
* **Concentration.** {t['n_vendors_scored']} vendors are scored; the {t['material_vendors_50plus']['n']} with 50+ lines hold
  {t['material_vendors_50plus']['value_share']:.0%} of line value. The High tier ({hi['n']} vendors) holds {hi['value_share']:.0%} of value but
  {hi['late_value_share']:.0%} of late value (late rate {hi['late_rate']:.1%}).
* **Shrinkage matters for thin vendors.** Median shift in late rate {sh['median_abs_shift_late_rate_pp_lt30_lines']:.1f} points for vendors
  under 30 lines vs {sh['median_abs_shift_late_rate_pp_ge100_lines']:.1f} for 100+; {sh['vendors_with_zero_late_raw']} vendors have zero late lines raw
  but none is scored as exactly zero risk.
* **Out-of-time validity is modest and fragile.** On {m['n_vendors_evaluated']} evaluable vendors ({m['next_late_lines']} late lines of
  {m['next_lines']:,}), line-level AUC is {late.loc[C, 'line_auc']:.3f} for the composite, {late.loc[EB, 'line_auc']:.3f} for the EB late rate alone
  and {late.loc[RAW, 'line_auc']:.3f} for the raw late rate; volume-only ({late.loc[VOL, 'line_auc']:.3f}) and lane-mix-only
  ({late.loc[LANE, 'line_auc']:.3f}) rank the wrong way. **A simple baseline wins:** composite minus raw rate is {dc['delta']:+.3f}
  (95% CI {dc['lo95']:+.3f} to {dc['hi95']:+.3f}). Across four rolling splits the composite beats the raw late rate in
  {int((btl[C] > btl[RAW]).sum())} and loses in {int((btl[C] < btl[RAW]).sum())}.
* **Cold start.** New / thin vendors were {cs.new_share_next_lines:.1%} of next-period lines (main split); their late rate was
  {cs.new_late:.2%} (95% Jeffreys {cs.new_late_lo95:.2%}-{cs.new_late_hi95:.2%}) against an assigned prior of {cs.prior_mean_late:.2%}, so the
  prior is too pessimistic there; the assigned prior was inside the interval in {n_inside} of {len(cs_all)} splits.
* **Archetypes are indicative** ({t['clusters']['n_clustered']} vendors, silhouette {t['clusters']['silhouette']:.2f}):
  {', '.join(f'{a.archetype} ({int(a.n_vendors)} vendors, {a.value_share:.0%} of value)' for a in prof.itertuples())}.
* **R models.** A vendor random intercept explains a latent {r['glmm_icc']:.0%} of late-delivery variance ({r['glmm_icc_null']:.0%} with no
  covariates; median odds ratio {r['glmm_mor']:.1f}); lines from the regional DC have odds {r['or_internal_rdc_vs_vendor']:.1f}x those of vendor lines
  (95% CI {r['or_internal_rdc_lo']:.1f}-{r['or_internal_rdc_hi']:.1f}). These are associations; part of the vendor effect is product and country niche.

| High / Elevated tier vendors with 50+ lines | Top country | Lines | Late (raw) | Late (EB) [90%] | Score [90%] |
|---|---|---|---|---|---|
{top_rows}

![Out-of-time AUC](figures/oot_validation_auc.png) ![Top-risk vendors](figures/top_risk_vendors.png)

## 4. Scenario (ASSUMED parameters, not observed)
ASSUMED: moving {scn['realloc_frac']:.0%} of the volume of the top {scn['top_share']:.0%} of vendors to the rest and closing
{scn['improvement_ASSUMED']:.0%} of the late-rate gap; late cost {ex['late_cost_pct_of_value']:.0%} of line value and carrying rate
{ex['carrying_rate_annual']:.0%} per year. Value at risk (observed late rate x value, EB-shrunk) is {usd(e['value_at_risk'])}; ASSUMED-cost exposure
is {usd(e['total_exposure_ASSUMED'])}, with the top 10% of scored vendors holding {e['top10pct_value_at_risk_share']:.0%} of value at risk. Back-tested on
the next period, ranking by the composite would have avoided {scn['late_value_reduction_pct']:.1%} of late value (its top group's value late rate
was {scn['top_late_rate_value']:.1%} vs {scn['rest_late_rate_value']:.1%} for the rest), whereas ranking by the raw late rate would have avoided
{raw['late_value_reduction_pct']:.1%}. This also assumes alternative vendors can supply the same product and country, which is often false for sole-source health commodities.

## 5. Recommendations and trade-offs
1. **Use the EB-shrunk late rate with its interval as the primary monitoring KPI**; use the composite for triage and context only, because it did not beat it.
2. **Open delivery-performance reviews with the {len(top)} High / Elevated-tier vendors that have 50+ lines** (table above) and ask for root causes
   by lane: with no hand-off timestamp the data cannot say whether the vendor or the freight leg is responsible.
3. **Treat the {t['n_vendors_unscored']} New / unscored vendors as unknown, not as good**; review each when it reaches {cfg['scoring']['min_lines_for_score']} lines.
4. **Do not rely on reallocation until delivery dates are audited** (back-filled dates, undelivered lines absent) and substitutability is confirmed.

*Limitations: one programme and one date definition; {t['n_vendors_unscored']} of {t['n_vendors_external']} vendors have under
{cfg['scoring']['min_lines_for_score']} lines; {m['next_late_lines']} late lines drive the main-split metrics; no hand-off split; costs and the reallocation effect are ASSUMED;
GLM effects are associations. Data: USAID / PEPFAR SCMS Delivery History (licence unclear; only derived aggregates are published).*
"""
    return md


CSS = """
@page { size: A4; margin: 12mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 7.9pt; line-height: 1.27; color: #1a1a1a; }
h1 { font-size: 13pt; color: #0f3d4a; margin: 0 0 4px 0; border-bottom: 2px solid #0f3d4a; }
h2 { font-size: 9.6pt; color: #0f3d4a; margin: 6px 0 2px 0; }
ul, ol { margin: 2px 0 2px 16px; padding: 0; } li { margin-bottom: 2px; } p { margin: 3px 0; }
table { border-collapse: collapse; margin: 4px 0; } th, td { border: 1px solid #bbb; padding: 1px 6px; }
th { background: #e3eef0; }
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
