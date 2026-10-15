"""Render README.md from computed artifacts so every number in it comes from the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
TITLE = ("Health Supply Vendor Risk - Using empirical-Bayes shrinkage and multilevel models to score on-time delivery risk "
         "on USAID / PEPFAR health-commodity order lines")
C, EB, RAW = "Composite risk score (shrunk)", "Shrunk late rate (EB) only", "Baseline: raw late rate"
VOL, LANE = "Baseline: volume only (smaller = riskier)", "Baseline: lane mix only (country / mode)"


def usd(x):
    return f"USD {x / 1e6:,.1f}m" if abs(x) >= 1e6 else f"USD {x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"USD {x:,.0f}"


def main():
    q = json.loads((ART / "sql_summary.json").read_text())
    t = json.loads((ART / "train_summary.json").read_text())
    r = json.loads((ART / "r_glm/r_summary.json").read_text())
    man = json.loads((ART / "data_manifest.json").read_text())
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    ev = pd.read_csv(ART / "oot_metrics_main.csv")
    late, sev = (ev[ev.target.str.startswith(k)].set_index("score") for k in ("Late", "Severe"))
    bt = pd.read_csv(ART / "oot_backtest_metrics.csv")
    btl = bt[bt.target.str.startswith("Late")].pivot(index="split", columns="score", values="line_auc")
    cs = pd.read_csv(ART / "cold_start.csv")
    prof = pd.read_csv(ART / "archetype_profiles.csv")
    tier = pd.read_csv(ART / "risk_tier_summary.csv").set_index("risk_tier")
    cal = pd.read_csv(ART / "oot_shrinkage_calibration.csv")
    sc = pd.read_parquet(ART / "vendor_scores.parquet")
    cl = pd.read_csv(ART / "cleaning_log.csv")
    grid = pd.read_csv(ART / "scenario_grid.csv")
    es = pd.read_csv(ART / "exposure_sensitivity.csv")
    sens = pd.read_csv(ART / "weight_sensitivity_ranking.csv")
    m, e, ex, d, pri = t["oot_main"], t["exposure_totals"], cfg["exposure"], t["auc_deltas"], t["priors"]
    scn, raw = t["scenario_base_ASSUMED"], t["scenario_raw_baseline_ASSUMED"]
    dc, de = d["composite_vs_raw_late_late"], d["eb_late_vs_raw_late_late"]
    w = cfg["scoring"]["weights"]
    watch = sc[(sc.n_lines >= 50) & sc.risk_tier.isin(["High", "Elevated"])].sort_values("risk_rank")
    main_split = f"{m['score_end'][:10]}->{m['validate_end'][:10]}"
    csm = cs[cs.split == main_split].iloc[0]

    def log(a):
        return int(cl.loc[cl.action == a, "n_rows"].iloc[0])

    cs_rows = "\n".join(
        f"| {x.split} | {int(x.n_new_vendors)} | {x.new_share_next_lines:.1%} | {x.prior_mean_late:.2%} | "
        f"{x.new_late:.2%} [{x.new_late_lo95:.2%}-{x.new_late_hi95:.2%}] | {x.scored_late:.2%} | {x.late_auc_scored_only:.3f} / "
        f"{x.late_auc_full_coverage:.3f} |" for x in cs.itertuples())
    bt_rows = "\n".join(f"| {s} | {row[C]:.3f} | {row[EB]:.3f} | {row[RAW]:.3f} |" for s, row in btl.iterrows())
    tier_rows = "\n".join(
        f"| {k} | {int(tier.loc[k, 'n_vendors'])} | {tier.loc[k, 'value_share']:.1%} | {tier.loc[k, 'late_value_share']:.1%} | "
        f"{tier.loc[k, 'late_rate']:.1%} |" for k in ["High", "Elevated", "Moderate", "Low", "New / unscored"])
    top_rows = "\n".join(
        f"| {v.vendor} | {v.top_country} | {v.top_product_group} | {int(v.n_lines)} | {v.late_rate_raw:.1%} | "
        f"{v.late_rate_shrunk:.1%} [{v.late_rate_lo90:.1%}-{v.late_rate_hi90:.1%}] | {v.risk_score:.1f} "
        f"[{v.risk_score_lo90:.0f}-{v.risk_score_hi90:.0f}] | {v.risk_tier} | {v.archetype} |" for v in watch.itertuples())
    arch_rows = "\n".join(
        f"| {a.archetype} | {int(a.n_vendors)} | {a.value_share:.1%} | {a.median_lines:.0f} | {a.late_rate_shrunk:.1%} | "
        f"{a.mean_days_late:.2f} | {a.freight_ratio:.3f} | {a.air_share:.0%} | {a.risk_score:.0f} |" for a in prof.itertuples())
    cal_rows = "\n".join(f"| {c.vol_band} | {int(c.n_vendors)} | {int(c.next_lines)} | {c.mae_raw:.3f} | {c.mae_shrunk:.3f} |"
                         for c in cal.itertuples())
    ev_rows = "\n".join(
        f"| {nm} | {late.loc[s, 'line_auc']:.3f} | {late.loc[s, 'vendor_auc']:.3f} | {late.loc[s, 'lift_top20']:.2f}x | "
        f"{sev.loc[s, 'line_auc']:.3f} |"
        for nm, s in [("Composite risk score", C), ("EB-shrunk late rate only", EB), ("Raw late rate (baseline)", RAW),
                      ("Volume only (baseline)", VOL), ("Lane mix only (baseline)", LANE)])
    g_max = grid.loc[grid.late_value_reduction_pct.idxmax()]
    sens_rows = "\n".join(f"| {s.weights} | {s.spearman_vs_base:.3f} | {s.top5_overlap:.0%} |" for s in sens.itertuples())

    md = f"""# {TITLE}

> **Outcome:** an empirical-Bayes delivery-risk scorecard for {t['n_vendors_external']} vendors of the USAID / PEPFAR Supply Chain
> Management System (SCMS), built on {q['n_lines']:,} order lines. The scorecard is honest about what it can and cannot
> do: out-of-time line-level AUC is {late.loc[C, 'line_auc']:.3f} for the composite, {late.loc[EB, 'line_auc']:.3f} for the shrunk late rate alone
> and {late.loc[RAW, 'line_auc']:.3f} for the raw late rate on only {m['n_vendors_evaluated']} evaluable vendors and {m['next_late_lines']} late lines, so a simple
> baseline matches or beats the composite and every interval is wide.

[![CI](https://github.com/Abhi237/supplier-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhi237/supplier-risk-scoring/actions/workflows/ci.yml)
· SQL (DuckDB) · Python (empirical Bayes, scikit-learn) · R (GLM, lme4, MASS, Quarto) · Streamlit

| | |
|---|---|
| ![Programme lateness](docs/figures/programme_yearly_lateness.png) | ![OOT validation](docs/figures/oot_validation_auc.png) |
| ![Shrinkage](docs/figures/shrinkage_and_intervals.png) | ![Archetypes](docs/figures/vendor_archetypes.png) |

**Deliverables:** [procurement memo (PDF)](docs/procurement_memo.pdf) · [memo (md)](docs/procurement_memo.md) ·
[R delivery report (HTML)](reports/delivery_report.html) · [EDA notebook](notebooks/01_eda.ipynb) ·
[model notebook](notebooks/02_model_walkthrough.ipynb) · [data dictionary](docs/data_dictionary.md) ·
[data model](docs/data_model.md) · [architecture](docs/architecture.md)

> **Scope and framing.** The buyer is a **US government health-supply programme** (USAID / PEPFAR SCMS). Commodities
> (antiretrovirals, HIV test kits and a few others) are shipped to PEPFAR-supported countries ({q['n_countries']} destinations, mostly in
> sub-Saharan Africa). **Nothing here is a US-domestic delivery.** Vendors are the real suppliers named in the dataset.
> On time means `Delivered to Client Date <= Scheduled Delivery Date`. **There is no vendor hand-off timestamp**, so there is no
> vendor-controlled vs end-to-end split: lateness mixes vendor, freight-forwarder and in-country effects. Scores describe
> **recorded delivery performance in this programme, not vendor quality overall**, and rest on limited samples.

## Interactive app
A multipage dark "control tower" Streamlit app (`app/streamlit_app.py`, deployed from `main`) reads **only** derived vendor
scorecards and aggregates committed in `app/data/` (about 0.25 MB; no line-level rows, no raw data).

| Page | What it shows |
|---|---|
| Executive summary | question, KPI cards, risk tiers by value and late value, recommendations |
| Data & quality | provenance, pipeline graph, the cleaning log, DQ checks and definitions |
| Delivery performance | lateness by year, offset distribution, country / mode, R GLM odds ratios and GLMM |
| Vendor scorecards | searchable vendor lookup: raw vs shrunk rates, credible and bootstrap intervals, component radar, cold-start tier |
| Risk model & validation | OOT AUC / lift vs baselines for two targets, AUC-difference intervals, quintiles, back-tests, shrinkage error, cold start |
| Vendor archetypes | k-means archetypes in PCA space, centroid heatmap, silhouette / BIC |
| Exposure & scenario simulator | **ASSUMED** cost sliders, value-at-risk ranking, concentration, back-tested reallocation |
| Methodology & limitations | definitions, scoring maths, validation design, caveats |

```bash
make app-data        # refresh app/data/ from pipeline outputs
make app             # http://localhost:8501
```
Streamlit Community Cloud: main file `app/streamlit_app.py`, Python 3.11, dependencies from `app/requirements.txt`.

| | |
|---|---|
| ![Overview](docs/app_screenshots/01_overview.png) | ![Scorecards](docs/app_screenshots/04_scorecards.png) |
| ![Validation](docs/app_screenshots/05_validation.png) | ![Simulator](docs/app_screenshots/07_exposure.png) |

All eight page screenshots (captured headlessly with Playwright): [`docs/app_screenshots/`](docs/app_screenshots/).

---

## 1. Situation and approach
| | |
|---|---|
| **Question** | Which vendors carry the most delivery risk, how sure can we be, and does a score built on the past predict the future? |
| **Data** | {q['n_lines']:,} order lines, {q['n_shipments']:,} shipments, {q['n_vendor_strings']} vendor strings ({q['ext_vendors']} external vendors plus the programme's own regional distribution centre, "SCMS from RDC", {q['rdc_line_share']:.0%} of lines, scored as a reference channel only), scheduled {q['scheduled_min'][:4]} to {q['scheduled_max'][:4]} |
| **Method** | DuckDB SQL cleaning with a counted log; beta-binomial empirical-Bayes shrinkage; Bayesian-bootstrap intervals; composite score; k-means archetypes; point-in-time out-of-time validation against baselines; New / unscored cold-start tier; R GLM / GLMM / negative binomial; exposure and ASSUMED reallocation scenarios |
| **Decision** | Monitor the EB-shrunk late rate with its interval; review the {len(watch)} High / Elevated vendors with 50+ lines; treat the {t['n_vendors_unscored']} New / unscored vendors as unknown; do not rely on reallocation until delivery dates are audited |

## 2. Data
| | |
|---|---|
| Source | USAID / PEPFAR SCMS Delivery History (file date 2015-09-29), obtained **without credentials** from the Internet Archive snapshot of the original usaid.gov file (`{man['source']}`); the data.usaid.gov export (SSL error) and the usaid.gov original (HTTP 403) failed. Citation, order tried, SHA-256 values and the licence caveat: [data/README.md](data/README.md) |
| Licence | **unclear**; the raw CSV is git-ignored and fetched by `make data`; only derived vendor scorecards and aggregates are published |
| Integrity | {man['rows']:,} rows x {man['columns']} columns, unique `ID`, required columns present, delivery dates 0 unparseable |
| Size | line value {usd(q['value_usd'])} nominal USD; {q['n_countries']} countries; external vendors {q['ext_lines']:,} lines ({usd(q['ext_value_usd'])}) |

**Cleaning log** (`sql/02_cleaning_and_quality.sql`, counts from the data): {log('delivered_equals_scheduled'):,} lines have delivered = scheduled
exactly ({q['ext_exact_match_share']:.1%} for external vendors; possible back-filling); {log('freight_see_reference'):,} freight values are references to another line
(resolved at shipment grain through the first line), {log('freight_included_in_commodity_cost'):,} are "included in commodity cost" and
{log('freight_invoiced_separately'):,} "invoiced separately" (typed, never coerced to 0); {log('days_late_capped')} lines are capped at
{cfg['cleaning']['days_late_cap']} days late for mean-lateness metrics only; {log('freight_ratio_capped')} shipments have freight above
{cfg['cleaning']['freight_ratio_cap']:.0f}x value (ratio capped); {log('line_value_zero_kept')} zero-value lines are kept in counts and excluded from ratios;
{log('recorded_before_delivered')} lines are recorded before delivery; {log('shipments_with_two_scheduled_dates')} shipments carry two scheduled dates;
{log('ship_mode_na_to_unknown')} missing shipment modes are labelled Unknown. Only delivered lines are present, so late rates are lower bounds.

## 3. SQL model (DuckDB)
| Script | Output |
|---|---|
| `01_stage.sql` | typed staging of every raw column used (text kept as typed kinds) |
| `02_cleaning_and_quality.sql` | `cleaning_log` (counted actions) and `dq_report` |
| `03_line_intervals.sql` | `fct_shipments`, `fct_lines`: offset, late, days late, severe, exact-match, known date, shipment freight |
| `04_vendor_summary.sql` | `vendor_lane`, `vendor_summary` |
| `05_vendor_year_panel.sql` | vendor-year panel with rolling 3-year windows, `LAG`, rank, running totals; `programme_year` |
| `06_oot_features.sql` | point-in-time vendor features before `score_end` (outcome known before it) and next-period outcomes |

DuckDB runs single-threaded because parallel floating-point `SUM` is not bit-reproducible.

## 4. Method
* **Shrinkage.** Beta prior fitted by marginal likelihood across vendors: late Beta({pri['late'][0]:.2f}, {pri['late'][1]:.2f}) (mean {pri['late'][2]:.1%},
  {pri['late'][3]:.0f} pseudo-lines). Median shift in late rate {t['shrinkage']['median_abs_shift_late_rate_pp_lt30_lines']:.1f} points for vendors under 30 lines vs
  {t['shrinkage']['median_abs_shift_late_rate_pp_ge100_lines']:.1f} for 100+; {t['shrinkage']['vendors_with_zero_late_raw']} vendors have zero late lines raw and none is scored as exactly zero.
* **Composite** = percentile-weighted components, scored vendors only (>= {cfg['scoring']['min_lines_for_score']} lines): {', '.join(f'{k} {v:g}' for k, v in w.items())}. Weights are a priori, not tuned. Lane mix is context
  (leave-one-vendor-out pooled country / mode late rates), not vendor-controlled.
* **Intervals:** {cfg['scoring']['n_bootstrap']} Bayesian-bootstrap draws per vendor (median 90% late-rate interval width {t['interval_width_median']['small_lt30']:.1f} points under 30 lines vs {t['interval_width_median']['large_ge100']:.1f} at 100+).
* **Weight sensitivity** (full-period ranking):

| Weights | Spearman vs base | Top-5 overlap |
|---|---|---|
{sens_rows}

## 5. Out-of-time validation
Features at {m['score_end'][:10]} use only lines scheduled before it **and** with delivered and recorded dates before it; outcomes are lines scheduled
{m['score_end'][:10]} to {m['validate_end'][:10]}. The main split was fixed from line volume (about half of vendor lines before it) before any metric was computed. {m['n_vendors_evaluated']} of
{m['n_vendors_scored']} scored vendors have next-period lines ({m['next_lines']:,} lines, {m['next_late_lines']} late, base rate {m['next_late_rate']:.1%}).

| Score | Line AUC (late) | Vendor AUC | Top-20% lift | Line AUC (severe, >{cfg['cleaning']['severe_late_days']} d) |
|---|---|---|---|---|
{ev_rows}

* **A simple baseline wins.** Composite minus raw-rate AUC is {dc['delta']:+.3f} (95% vendor-cluster bootstrap CI {dc['lo95']:+.3f} to {dc['hi95']:+.3f});
  EB rate minus raw rate is {de['delta']:+.3f} (CI {de['lo95']:+.3f} to {de['hi95']:+.3f}). Volume-only and lane-mix-only baselines rank the wrong way (AUC below 0.5).
* Rolling back-tests (line AUC, late target; each split has {int(bt.n_vendors.min())} to {int(bt.n_vendors.max())} evaluable vendors):

| Split | Composite | EB late rate | Raw late rate |
|---|---|---|---|
{bt_rows}

  The composite beats the raw rate in {int((btl[C] > btl[RAW]).sum())} of {len(btl)} splits.
* Shrinkage error by history volume (line-weighted MAE of late-rate estimate against next-period rate):

| Lines in score period | Vendors | Next lines | MAE raw | MAE shrunk |
|---|---|---|---|---|
{cal_rows}

  The 1-4 band has only {int(cal.iloc[0].next_lines)} next-period lines, so a raw rate of 0 can look exact there; the comparison is not informative.

**Cold start (New / unscored).** A vendor below {cfg['scoring']['min_lines_for_score']} lines at the score date gets the prior mean for every rate, no composite and the tier
New / unscored (`src/supplier_risk/coldstart.py`). In the main split such vendors were {int(csm.n_new_vendors)} vendors, {csm.new_share_next_lines:.1%} of next-period lines
and {csm.new_share_next_value:.1%} of value. Out-of-time check of the default:

| Split | New vendors | Share of next lines | Prior late | New vendors' next late [95% Jeffreys] | Scored vendors' next late | AUC scored only / full coverage |
|---|---|---|---|---|---|---|
{cs_rows}

The prior is inside the interval in {int(cs.prior_inside_ci.sum())} of {len(cs)} splits and too pessimistic in the main split.

## 6. Results
* **Lateness:** {q['ext_late_rate']:.1%} of external-vendor lines were late ({q['ext_severe_late_rate']:.1%} more than {cfg['cleaning']['severe_late_days']} days; median {q['ext_median_days_late_when_late']:.0f} days late when late,
  p90 {q['ext_p90_days_late_when_late']:.0f}); late value {q['ext_late_value_share']:.1%} of value. The regional DC was late {q['rdc_late_rate']:.1%} of the time.
* **R layer:** on {r['n_lines']:,} external-vendor lines ({r['n_late']} late) the vendor random intercept has SD {r['glmm_vendor_sd']:.2f} (latent ICC **{r['glmm_icc']:.1%}**, {r['glmm_icc_null']:.1%} without covariates,
  shipment-grain sensitivity {r['glmm_ship_icc']:.1%}; median odds ratio {r['glmm_mor']:.1f}). The all-lines GLM gives the regional DC odds {r['or_internal_rdc_vs_vendor']:.1f}x
  vendor lines (95% CI {r['or_internal_rdc_lo']:.1f}-{r['or_internal_rdc_hi']:.1f}); truck vs air OR {r['or_truck']:.2f}; negative-binomial theta {r['nb_theta']:.2f}
  (Poisson dispersion {r['poisson_dispersion']:.0f}). GLM AUC {r['glm_auc']:.3f} (in-sample). Effects are associations; the Incoterm effect differs in sign between the marginal GLM and the vendor-conditional GLMM, so it is fragile.
* **Risk tiers** (full period, percentile tiers among {t['n_vendors_scored']} scored vendors):

| Tier | Vendors | Value share | Late-value share | Late rate |
|---|---|---|---|---|
{tier_rows}

  The {t['material_vendors_50plus']['n']} vendors with 50+ lines hold {t['material_vendors_50plus']['value_share']:.0%} of line value.
* **High / Elevated vendors with 50+ lines:**

| Vendor | Top country | Product | Lines | Late raw | Late EB [90%] | Score [90% boot] | Tier | Archetype |
|---|---|---|---|---|---|---|---|---|
{top_rows}

* **Indicative archetypes** (k = {t['clusters']['k']}, silhouette {t['clusters']['silhouette']:.2f}, {t['clusters']['n_clustered']} vendors with 20+ lines):

| Archetype | Vendors | Value share | Median lines | Late (EB) | Mean days late | Freight ratio | Air share | Risk score |
|---|---|---|---|---|---|---|---|---|
{arch_rows}

* **Exposure (observed):** line value {usd(e['value_usd'])}, late value {usd(e['late_value_usd'])}, value at risk (EB late rate x value) {usd(e['value_at_risk'])},
  top 10% of scored vendors hold {e['top10pct_value_at_risk_share']:.0%} of it. **ASSUMED** late cost {ex['late_cost_pct_of_value']:.0%} of value and carrying rate
  {ex['carrying_rate_annual']:.0%} per year give total exposure {usd(e['total_exposure_ASSUMED'])} (range {usd(es.total_exposure_ASSUMED.min())} to {usd(es.total_exposure_ASSUMED.max())} over the sensitivity grid).
* **Reallocation back-test (ASSUMED):** moving {scn['realloc_frac']:.0%} of the volume of the top {scn['top_share']:.0%} of vendors (by composite) to the rest, closing {scn['improvement_ASSUMED']:.0%}
  of the late-rate gap, avoids {scn['late_value_reduction_pct']:.1%} of next-period late value: the top group's value late rate was {scn['top_late_rate_value']:.1%} vs {scn['rest_late_rate_value']:.1%} for the rest.
  Ranking by the raw late rate would have avoided {raw['late_value_reduction_pct']:.1%}. The best composite grid cell (top {g_max.top_share:.0%}, {g_max.realloc_frac:.0%} moved, {g_max.improvement_ASSUMED:.0%} of gap closed) gives {g_max.late_value_reduction_pct:.2%}.

![Exposure](docs/figures/exposure_concentration.png)

## 7. Actions, trade-offs, limitations
**Actions:** (1) monitor the EB-shrunk late rate with its interval, using the composite for triage and context only; (2) open delivery reviews with the {len(watch)} High / Elevated
vendors that have 50+ lines and ask for root causes by lane; (3) treat the {t['n_vendors_unscored']} New / unscored vendors as unknown and review each at {cfg['scoring']['min_lines_for_score']} lines;
(4) audit the scheduled-date field before using any reallocation result.

**Trade-offs:** stronger shrinkage protects thin vendors from noise but is slower to flag a genuinely bad newcomer; the volume-uncertainty weight pushes unknown vendors up the list at the cost of some false alarms;
moving volume concentrates spend in fewer vendors.

**Limitations:**
* One programme, one US government buyer, one date definition (delivered vs scheduled); nominal USD 2006-2015.
* No hand-off timestamp: no vendor-controlled vs end-to-end split. {q['ext_exact_match_share']:.1%} of vendor lines have delivered = scheduled exactly, so the on-time rate may be overstated.
* Small samples: {q['ext_vendors_lt10_lines']} of {q['ext_vendors']} vendors have under {cfg['scoring']['min_lines_for_score']} lines and {r['n_late']} late lines exist in total; only {m['n_vendors_evaluated']} vendors can be validated out-of-time.
* Predictive validity is modest and a simple baseline wins; the composite weights are a priori and were not tuned.
* Only delivered lines are in the file; vendor names are used as recorded (spelling variants not merged); clusters are soft; costs and the reallocation effect are ASSUMED.
* GLM / GLMM effects are associations; the vendor effect partly reflects product, country and period niche.
* Licence of the source data is unclear; scores are not a statement about any vendor's quality.

## 8. How to run
```bash
make setup          # Python 3.11 venv + pinned deps (requirements.txt) + package
make data           # download + verify the SCMS CSV (no credentials); writes data/raw (git-ignored) and data/interim parquet
make sql            # DuckDB model (sql/01-06), cleaning log
make train          # scoring, bootstrap, clusters, OOT validation, cold start, exposure, scenarios, figures
make report         # R GLM/GLMM/NB + Quarto HTML, memo PDF, README, executed notebooks
make app-data       # export app/data (derived scorecards and aggregates only)
make test           # ruff + pytest
make all            # everything above in order
make app            # Streamlit on :8501
```
Requirements: Python 3.11 (pinned in `requirements.txt` / `requirements-ci.txt`), R >= 4.3 with the packages in [`renv.lock`](renv.lock) (see [`R/README.md`](R/README.md)),
Quarto >= 1.4 (optional), WeasyPrint for the memo PDF. All parameters live in [`config/config.yaml`](config/config.yaml). Two consecutive runs of sql, train, R and export give byte-identical outputs.

**Tests** (`tests/`, no raw data needed): SQL edge cases from a committed handcrafted fixture (date parsing, freight and weight text, see-references, zero / negative values,
internal-channel flag, exact match, caps), grain and key checks, point-in-time leakage tests, shrinkage and bootstrap unit tests, cold start, metric correctness, scenario logic,
determinism, data-download verification, repository hygiene, regression on the committed aggregates, and Streamlit **AppTest** for all eight pages. CI runs ruff + pytest on synthetic fixtures.

## Licence
Code: MIT ([LICENSE](LICENSE)). Data: USAID / PEPFAR SCMS Delivery History, licence unclear; see [data/README.md](data/README.md). Only derived aggregates and vendor scorecards are published.
"""
    (ROOT / "README.md").write_text(md)
    print(f"README.md written ({len(md):,} chars)")


if __name__ == "__main__":
    main()
