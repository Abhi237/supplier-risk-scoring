"""Render README.md from computed artifacts so every number in it comes from the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def brl(x):
    return f"BRL {x / 1e6:,.2f}m" if abs(x) >= 1e6 else f"BRL {x / 1e3:,.0f}k"


def main():
    q = json.loads((ART / "sql_summary.json").read_text())
    t = json.loads((ART / "train_summary.json").read_text())
    r = json.loads((ART / "r_glm/r_summary.json").read_text())
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    ev = pd.read_csv(ART / "oot_metrics_main.csv").set_index(["target", "score"])
    bt = pd.read_csv(ART / "oot_backtest_metrics.csv").set_index(["split", "target", "score"])
    prof = pd.read_csv(ART / "archetype_profiles.csv")
    cal = pd.read_csv(ART / "oot_shrinkage_calibration.csv").set_index("vol_band")
    sens = pd.read_csv(ART / "weight_sensitivity_ranking.csv")
    tier = pd.read_csv(ART / "risk_tier_summary.csv").set_index("risk_tier")
    sc = pd.read_parquet(ART / "seller_scores.parquet")
    dec = pd.read_csv(ART / "oot_deciles.csv").set_index("decile")
    grid = pd.read_csv(ART / "scenario_grid.csv")
    es = pd.read_csv(ART / "exposure_sensitivity.csv")
    dq = q["dq"]
    m, e, ex, d = t["oot_main"], t["exposure_totals"], cfg["exposure"], t["auc_deltas"]
    C, EB, EBL, RH, RL, VO = ("Composite risk score (shrunk)", "Shrunk hand-off late rate (EB) only",
                              "Shrunk late rate (EB) only", "Baseline: raw hand-off late rate",
                              "Baseline: raw late rate", "Baseline: volume only (smaller = riskier)")
    ho, lt = ev.loc["Seller hand-off late"], ev.loc["End-to-end late delivery"]
    mat = sc[sc.n_items >= 50]
    watch = mat[mat.risk_tier.isin(["High", "Elevated"])]
    scn, hs = t["scenario_base_ASSUMED"], t["scenario_handoff_items_ASSUMED"][C]
    g_max = grid[(grid.top_share == 0.2) & (grid.realloc_frac == 1.0) & (grid.improvement_ASSUMED == 1.0)].iloc[0]
    pri = t["priors"]
    w = cfg["scoring"]["weights"]

    def row(nm, label):
        return (f"| {label} | {ho.loc[nm, 'item_auc']:.3f} | {ho.loc[nm, 'seller_auc']:.3f} | "
                f"{ho.loc[nm, 'spearman']:.3f} | {ho.loc[nm, 'lift_top10']:.2f}x | {lt.loc[nm, 'item_auc']:.3f} | "
                f"{lt.loc[nm, 'lift_top10']:.2f}x |")

    splits = [f"{a}->{b}" for a, b in cfg["scoring"]["backtest_splits"]]
    bt_rows = "\n".join(
        f"| {s} | {bt.loc[(s, 'Seller hand-off late', EB), 'item_auc']:.3f} | "
        f"{bt.loc[(s, 'Seller hand-off late', RH), 'item_auc']:.3f} | "
        f"{bt.loc[(s, 'Seller hand-off late', C), 'item_auc']:.3f} | "
        f"{bt.loc[(s, 'End-to-end late delivery', C), 'item_auc']:.3f} | "
        f"{bt.loc[(s, 'End-to-end late delivery', RL), 'item_auc']:.3f} |" for s in splits)
    arch_rows = "\n".join(
        f"| {a.archetype} | {int(a.n_sellers)} | {a.gmv_share:.1%} | {int(a.median_items)} | {a.late_rate_shrunk:.1%} | "
        f"{a.handoff_late:.1%} | {a.review:.2f} | {a.freight_to_price:.2f} | {a.cancel_rate:.2%} | {a.risk_score:.0f} |"
        for a in prof.itertuples())
    tier_rows = "\n".join(
        f"| {k} | {int(tier.loc[k, 'n_sellers']):,} | {tier.loc[k, 'gmv_share']:.1%} | {tier.loc[k, 'late_gmv_share']:.1%} | "
        f"{tier.loc[k, 'late_rate']:.1%} |" for k in ["High", "Elevated", "Moderate", "Low"])
    top = mat.nsmallest(8, "risk_rank")
    top_rows = "\n".join(
        f"| `{s.seller_id[:10]}…` | {s.seller_state} | {s.top_category} | {int(s.n_items)} | {s.late_rate_raw:.1%} | "
        f"{s.late_rate_shrunk:.1%} [{s.late_rate_lo90:.1%}-{s.late_rate_hi90:.1%}] | {s.seller_handoff_late_rate:.1%} | "
        f"{s.risk_score:.1f} [{s.risk_score_lo90:.0f}-{s.risk_score_hi90:.0f}] | {s.archetype} |" for s in top.itertuples())
    sens_rows = "\n".join(f"| {s.weights} | {s.spearman_vs_base:.3f} | {s.top100_overlap:.0%} | "
                          f"{ho.loc[f'Sensitivity: {s.weights} weights', 'item_auc']:.3f} | "
                          f"{lt.loc[f'Sensitivity: {s.weights} weights', 'item_auc']:.3f} |" for s in sens.itertuples())
    dq_issue = {k: v for k, v in dq.items() if not k.endswith("_rows")}

    md = f"""# Supplier Reliability & Procurement Risk Scoring

> **Outcome:** an empirical-Bayes reliability scorecard for {q['n_sellers_active']:,} Olist sellers (used as
> **supplier proxies**) that predicts next-period seller-controlled dispatch lateness out-of-time
> (AUC {ho.loc[EB, 'item_auc']:.3f} vs {ho.loc[RH, 'item_auc']:.3f} for the raw rate and {ho.loc[VO, 'item_auc']:.3f} for
> volume-only; top score decile {dec.loc[10, 'next_handoff_late_rate']:.1%} late hand-offs vs
> {dec.loc[1, 'next_handoff_late_rate']:.1%} in the bottom decile) - and shows, honestly, that end-to-end customer
> lateness is mostly *not* a stable supplier trait, so volume reallocation alone would barely move it.

[![CI](https://github.com/Abhi237/supplier-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhi237/supplier-risk-scoring/actions/workflows/ci.yml)
· SQL (DuckDB) · Python (empirical Bayes, scikit-learn) · R (GLM, lme4, MASS, Quarto) · Streamlit

| | |
|---|---|
| ![Hand-off vs late](docs/figures/handoff_delay_vs_late.png) | ![OOT validation](docs/figures/oot_validation_auc.png) |
| ![Shrinkage](docs/figures/shrinkage_and_intervals.png) | ![Archetypes](docs/figures/seller_archetypes.png) |

**Deliverables:** [procurement memo (PDF)](docs/procurement_memo.pdf) · [memo (md)](docs/procurement_memo.md) ·
[R delivery report (HTML)](reports/delivery_report.html) · [EDA notebook](notebooks/01_eda.ipynb) ·
[model notebook](notebooks/02_model_walkthrough.ipynb) · [data dictionary](docs/data_dictionary.md) ·
[data model](docs/data_model.md) · [architecture](docs/architecture.md)

> **Methodology note - sellers as supplier proxies.** Olist sellers are independent marketplace merchants who ship
> through Olist's carriers. They stand in for suppliers: they have a dispatch deadline (`shipping_limit_date`), a
> customer promise date, freight cost, cancellations and quality feedback. They are **not** a buyer's contracted
> vendors (no POs, invoices or payment terms), so the *method* transfers to procurement data while vendor-level facts
> do not.

## Interactive app
A multipage Streamlit app (`app/streamlit_app.py`) presents the project for stakeholders. It reads **only** compact
aggregated artifacts committed in `app/data/` (~1 MB: summaries, validation tables, aggregated lane/category tables
and the {len(sc):,} seller scorecards). No customer-level data, no order rows, no ML libraries at deploy time.

| Page | What it shows |
|---|---|
| Executive summary | business question, KPI cards, GMV vs late GMV by risk tier, recommendations |
| Data & quality | integrity counts, pipeline graph, all data-quality checks, cleaning decisions |
| Delivery performance | monthly lateness, hand-off timing vs lateness, lead-time ECDFs, states/categories, GLM odds ratios |
| Seller scorecards | searchable seller lookup: raw vs shrunk rates, credible + bootstrap intervals, component radar |
| Risk model & validation | out-of-time AUC/lift vs baselines for two targets, ΔAUC CIs, deciles, back-tests, shrinkage error |
| Seller archetypes | k-means archetypes in PCA space, centroid heatmap, silhouette/BIC selection |
| Exposure & scenario simulator | **ASSUMED** cost sliders, exposure ranking and concentration, back-tested reallocation |
| Methodology & limitations | definitions, scoring maths, validation design, caveats |

```bash
make app-data        # refresh app/data/ from pipeline outputs
make app             # http://localhost:8501
```
**Deploy on Streamlit Community Cloud:** main file `app/streamlit_app.py`, Python 3.11; dependencies from
`app/requirements.txt` (streamlit, pandas, numpy, pyarrow, plotly). Theme: `.streamlit/config.toml`.

| | |
|---|---|
| ![Overview](docs/app_screenshots/01_overview.png) | ![Scorecards](docs/app_screenshots/04_scorecards.png) |
| ![Validation](docs/app_screenshots/05_validation.png) | ![Simulator](docs/app_screenshots/07_exposure.png) |

All eight page screenshots: [`docs/app_screenshots/`](docs/app_screenshots/).

---

## 1. Situation & question
A marketplace operator (standing in for a procurement team) depends on {q['n_sellers_active']:,} independent sellers
to dispatch on time. Late deliveries hurt reviews ({q['review_late']:.2f}/5 when late vs {q['review_on_time']:.2f} on
time) and tie up working capital. The question:

> **Which vendors create the greatest delivery-risk and working-capital exposure?**

Answered with four lenses: **shrunk reliability scores** with uncertainty (who is unreliable, and how sure are we),
**out-of-time validation** (does a score built today predict next period), **seller archetypes** (what kind of
problem), and **exposure + scenario analysis** (how much is at stake, and what would reallocating volume achieve).

## 2. Data
| | |
|---|---|
| Source | [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0), fetched from a public GitHub mirror - see [data/README.md](data/README.md) |
| Integrity | orders {dq['orders_rows']:,} · items {dq['order_items_rows']:,} · sellers {dq['sellers_rows']:,} - all equal to the documented counts; SHA-256 recorded |
| Period | purchases {q['date_min']} to {q['date_max']} |
| Grain | `fct_items`: one row per order x item ({q['n_items']:,}); `fct_orders_delivered`: {q['n_delivered_orders']:,} delivered orders; `seller_summary`: {q['n_sellers_active']:,} sellers; `seller_month_panel` |
| Size | GMV {brl(q['gmv_brl'])}, freight {brl(q['freight_brl'])} ({q['freight_to_price']:.1%} of price) |
| Concentration | top 10% of sellers = {q['top10pct_sellers_gmv_share']:.1%} of GMV; {t['low_volume_sellers']:,} sellers have <5 items |

**Data-quality checks** (`sql/02_data_quality.sql`, counts from `dq_report`): {', '.join(f'{k} = {v:,}' for k, v in dq_issue.items())}.

**Cleaning / definition decisions:** lateness only for delivered orders with a delivery timestamp; late = delivery
*calendar date* after the estimate; hand-off lateness not judged when the carrier timestamp precedes purchase
({dq['carrier_before_purchase']} orders); latest review per order; customer data reduced to state; multi-seller
orders ({q['multi_seller_order_share']:.1%} of delivered orders) share one delivery timestamp and every seller in them
inherits it.

**Caveats:** sellers are supplier proxies; Olist's estimated date already embeds regional slack; orders purchased near
the data end that were not yet delivered cannot be judged (right-censoring); no cost data exists, so all costs are ASSUMED.

## 3. SQL model (DuckDB, ANSI/Postgres-friendly)
| Script | Output |
|---|---|
| `01_stage.sql` | typed staging; review de-dup (`ROW_NUMBER`) |
| `02_data_quality.sql` | `dq_report` (one row per check) |
| `03_order_intervals.sql` | `fct_items` with approval / hand-off / transit / delivery intervals, late & hand-off flags; `fct_orders_delivered` |
| `04_seller_summary.sql` | `seller_summary` (counts, GMV, late GMV, freight, reviews, medians) |
| `05_seller_month_panel.sql` | seller-month panel with rolling 3-month windows, `LAG`, `RANK`, running totals; `platform_month` |
| `06_oot_features.sql` | point-in-time seller features before `score_end` and next-period outcomes |

Seller-controlled lateness vs the shipping limit:
```sql
DATE_DIFF('second', i.shipping_limit_ts, o.carrier_ts) / 86400.0                    AS handoff_delay_days,
CASE WHEN o.carrier_ts IS NOT NULL AND o.carrier_ts >= o.purchase_ts
     THEN o.carrier_ts > i.shipping_limit_ts END                                   AS is_handoff_late,
CASE WHEN o.order_status = 'delivered' AND o.delivered_ts IS NOT NULL
     THEN CAST(o.delivered_ts AS DATE) > CAST(o.estimated_ts AS DATE) END          AS is_late
```
Rolling seller-month window:
```sql
SUM(n_late) OVER w3::DOUBLE / NULLIF(SUM(n_delivered) OVER w3, 0) AS late_rate_roll3
WINDOW w3 AS (PARTITION BY seller_id ORDER BY purchase_month
              RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW)
```
Point-in-time features (no information from on/after the score date):
```sql
SUM(CASE WHEN f.is_late AND f.delivered_ts < p.score_end THEN 1 ELSE 0 END) AS n_late,
AVG(CASE WHEN f.review_created_ts < p.score_end THEN f.review_score END)    AS mean_review_score
FROM fct_items f CROSS JOIN params p WHERE f.purchase_ts < p.score_end
```

## 4. Python + R
**Python (`src/supplier_risk`, CLI `supplier-risk sql|train`):**
* **Empirical-Bayes beta-binomial shrinkage** (`shrinkage.py`): Beta prior fitted by maximum marginal likelihood -
  late Beta({pri['late'][0]:.2f}, {pri['late'][1]:.2f}) (mean {pri['late'][2]:.1%}, {pri['late'][3]:.0f} pseudo-items),
  hand-off Beta({pri['handoff'][0]:.2f}, {pri['handoff'][1]:.2f}) (mean {pri['handoff'][2]:.1%}), cancellation
  Beta({pri['cancel'][0]:.2f}, {pri['cancel'][1]:.2f}); posterior mean (a+k)/(a+b+n) with 90% credible intervals.
* **Composite risk score** = 100 x Σ w x component percentile, weights documented a priori: {', '.join(f'{k} {v}' for k, v in w.items())}.
* **Uncertainty:** {cfg['scoring']['n_bootstrap']}-draw Bayesian bootstrap per seller (Dirichlet-weighted outcomes with
  the EB prior as pseudo-data). Median 90% interval width: {t['interval_width_median']['small_lt20']:.0f} score points for
  sellers with <20 items vs {t['interval_width_median']['large_ge100']:.0f} for >=100 items. (A classical bootstrap gives
  zero width for sellers with 0 late items - wrong direction.)
* **Archetypes:** k-means on 7 standardised features for {t['clusters']['n_clustered']:,} sellers with >=20 items; k in
  {cfg['clustering']['k_range']} chosen by silhouette (GMM BIC reported alongside).
* **Exposure:** GMV at risk, working capital in transit, excess WC from delays (observed) and ASSUMED cost layers.
* **Out-of-time validation** and **reallocation back-test** (below).

**R (`R/delivery_analysis.R` -> `reports/delivery_report.html`):** logistic GLM for lateness (hand-off, customer &
seller region, lane, freight, GMV, category, month, year), seller random-intercept GLMM (lme4), negative-binomial
lead-time model (MASS), KS / Wilcoxon / Kruskal-Wallis lead-time comparisons with ECDFs, calibration, Hosmer-Lemeshow
and GVIF diagnostics.

## 5. Validation, baselines, metrics, sensitivity
Score sellers using only data observable before **{m['score_end'][:10]}**; evaluate on items purchased
{m['score_end'][:10]} -> {m['validate_end'][:10]}. {m['n_sellers_evaluated']:,} sellers evaluated, covering
{m['next_items_covered_share']:.0%} of next-period items (new sellers can't be scored). Item-level AUC = each
next-period item inherits its seller's score. Next-period base rates: hand-off late {m['next_handoff_late_rate']:.1%},
late delivery {m['next_late_rate']:.1%}.

| Score | Hand-off: item AUC | seller AUC | Spearman | top-10% lift | Late delivery: item AUC | top-10% lift |
|---|---|---|---|---|---|---|
{row(EB, '**EB-shrunk hand-off rate**')}
{row(C, '**Composite risk score**')}
{row(EBL, 'EB-shrunk late rate')}
{row(RH, 'Baseline: raw hand-off rate')}
{row(RL, 'Baseline: raw late rate')}
{row(VO, 'Baseline: volume only')}

* **Shrinkage vs raw** (hand-off target): ΔAUC {d['eb_handoff_vs_raw_handoff_handoff']['delta']:+.3f} (seller-cluster
  bootstrap 95% CI {d['eb_handoff_vs_raw_handoff_handoff']['lo95']:+.3f} to {d['eb_handoff_vs_raw_handoff_handoff']['hi95']:+.3f}).
  Item-weighted MAE vs next-period late rate for sellers with <5 deliveries: raw {cal.loc['<5', 'mae_raw']:.3f} ->
  shrunk {cal.loc['<5', 'mae_shrunk']:.3f}; for 100+ deliveries {cal.loc['100+', 'mae_raw']:.3f} -> {cal.loc['100+', 'mae_shrunk']:.3f}.
  The raw hand-off rate has a higher top-10% lift ({ho.loc[RH, 'lift_top10']:.2f}x vs {ho.loc[EB, 'lift_top10']:.2f}x)
  but its top decile holds less volume, so it captures fewer late items ({ho.loc[RH, 'capture_top10']:.1%} vs
  {ho.loc[EB, 'capture_top10']:.1%}).
* **Composite vs volume-only** (hand-off): ΔAUC {d['composite_vs_volume_handoff']['delta']:+.3f}
  (95% CI {d['composite_vs_volume_handoff']['lo95']:+.3f} to {d['composite_vs_volume_handoff']['hi95']:+.3f}).
  **Composite vs raw late rate** on end-to-end lateness: ΔAUC {d['composite_vs_raw_late_late']['delta']:+.3f}
  (95% CI {d['composite_vs_raw_late_late']['lo95']:+.3f} to {d['composite_vs_raw_late_late']['hi95']:+.3f}) - no
  score predicts end-to-end lateness well.
* **Rolling back-tests** (item AUC):

| Split | Hand-off: EB | Hand-off: raw | Hand-off: composite | Late: composite | Late: raw late rate |
|---|---|---|---|---|---|
{bt_rows}

* **Weight sensitivity:**

| Alternative weights | Spearman vs base ranking | top-100 overlap | OOT hand-off AUC | OOT late AUC |
|---|---|---|---|---|
{sens_rows}

* **R cross-check:** GLM AUC {r['glm_auc']:.3f} in-sample ({r['glm_auc_without_handoff']:.3f} without the hand-off term;
  LRT p < 0.001); Hosmer-Lemeshow rejects at n = {r['n_orders']:,} (stat {r['hosmer_lemeshow_stat']:.0f}) though decile
  gaps are small; max adjusted GVIF {r['max_gvif_adj']:.2f}; Poisson dispersion {r['poisson_dispersion']:.2f} -> negative
  binomial (theta {r['nb_theta']:.2f}).

![OOT deciles](docs/figures/oot_deciles.png)

## 6. Results
* **Lateness:** {q['order_late_rate']:.1%} of delivered orders were late (mean {q['order_mean_delay_days_when_late']:.1f}
  days late); median lead time {q['order_median_delivery_days']:.1f} days. {q['item_handoff_late_rate']:.1%} of items were
  handed off after the shipping limit; those orders were late {q['late_rate_if_handoff_late']:.1%} of the time vs
  {q['late_rate_if_handoff_on_time']:.1%}.
* **Seller effect (R):** late hand-off OR **{r['or_handoff']:.2f}** (95% CI {r['or_handoff_lo']:.2f}-{r['or_handoff_hi']:.2f});
  same-state lane OR {r['or_same_state']:.2f}; hand-off late lengthens lead time by {r['irr_handoff'] - 1:.0%} (NB rate
  ratio). Seller random-intercept ICC **{r['glmm_icc_without_handoff']:.1%}** ({r['glmm_icc']:.1%} once hand-off is
  controlled): most end-to-end lateness is lane/carrier/calendar driven.
* **Risk tiers** (full period):

| Tier | Sellers | GMV share | Late-GMV share | Late rate |
|---|---|---|---|---|
{tier_rows}

  The High tier's median seller has only {t['high_tier']['median_items']:.0f} items. Among **material sellers
  (>=50 items: {t['material_sellers_50plus']['n']} sellers, {t['material_sellers_50plus']['gmv_share']:.0%} of GMV)**,
  {len(watch)} are High/Elevated, holding {watch.gmv.sum() / sc.gmv.sum():.1%} of GMV and
  {watch.late_gmv.sum() / sc.late_gmv.sum():.1%} of late GMV. Top material sellers by risk:

| Seller | State | Top category | Items | Late raw | Late EB [90% CrI] | Hand-off EB | Risk [90% boot] | Archetype |
|---|---|---|---|---|---|---|---|---|
{top_rows}

* **Archetypes** (k = {t['clusters']['k']}, silhouette {t['clusters']['silhouette']:.3f} - soft segments):

| Archetype | Sellers | GMV share | Median items | Late (EB) | Hand-off late | Review | Freight/price | Cancel | Risk score |
|---|---|---|---|---|---|---|---|---|---|
{arch_rows}

* **Exposure:** GMV {brl(e['gmv'])}; observed late GMV {brl(e['late_gmv'])}; GMV at risk {brl(e['gmv_at_risk'])} (top
  10% of sellers by this metric = {e['top10pct_gmv_at_risk_share']:.0%}); working capital in transit
  {brl(e['wc_in_transit'])}; excess WC from delays {brl(e['wc_excess_delay'])}. With **ASSUMED** {ex['late_cost_pct_of_gmv']:.0%}
  late cost, {ex['freight_absorb_pct']:.0%} freight absorbed and {ex['cancel_cost_pct_of_gmv']:.0%} cancellation cost,
  total exposure = {brl(e['total_exposure_ASSUMED'])} (range {brl(es.total_exposure_ASSUMED.min())}-
  {brl(es.total_exposure_ASSUMED.max())} over the sensitivity grid); the top risk decile holds only
  {es.top_decile_share.min():.0%}-{es.top_decile_share.max():.0%} of it - **exposure follows GMV, not risk tier**.
* **Reallocation back-test (ASSUMED):** moving {scn['realloc_frac']:.0%} of the top-{scn['top_share']:.0%} (composite)
  sellers' next-period volume and closing {scn['improvement_ASSUMED']:.0%} of the late-rate gap
  ({scn['top_late_rate_gmv']:.1%} vs {scn['rest_late_rate_gmv']:.1%} GMV-weighted) avoids
  {brl(scn['late_gmv_avoided']) if scn['late_gmv_avoided'] >= 1e4 else f"BRL {scn['late_gmv_avoided']:,.0f}"}
  = **{scn['late_gmv_reduction_pct']:.2%} of late GMV**; the most aggressive grid cell (top 20%, all volume, full gap)
  gives {g_max.late_gmv_reduction_pct:.1%}. On the seller-controlled leg the base move cuts late hand-offs by
  {hs['late_items_reduction_pct']:.1%}.

![Exposure](docs/figures/exposure_concentration.png)

## 7. Actions, trade-offs, limitations
**Actions**
1. Make the **EB-shrunk hand-off late rate** the supplier SLA KPI (most predictive, least noisy, seller-controlled),
   reported with its credible interval.
2. Run dispatch-SLA reviews with the **{len(watch)} material High/Elevated sellers**, starting with the
   "Slow dispatchers" archetype; use the composite score for triage context (reviews, cancellations, freight).
3. Treat low-volume High-tier sellers as an **onboarding watch-list**, not as confirmed bad suppliers.
4. Address end-to-end lateness through **carrier/lane** management (North/Northeast destinations, peak months),
   not supplier switching.

**Trade-offs:** stronger shrinkage protects small sellers from noise but is slower to flag a genuinely bad newcomer;
volume-uncertainty weight pushes unknown sellers up the list (cautious) at the cost of some false alarms; reallocating
volume concentrates spend in fewer sellers (supply-concentration risk is already {q['top10pct_sellers_gmv_share']:.0%}
of GMV in the top 10%).

**Limitations:** sellers are proxies for suppliers; costs and reallocation effects are ASSUMED; scores cover
{m['next_items_covered_share']:.0%} of next-period items; clusters are weakly separated; multi-seller orders share
timestamps; GLM/GLMM effects are associations, not causal; data are non-commercial licensed (CC BY-NC-SA 4.0).

## 8. How to run
```bash
make setup          # Python 3.11 venv + pinned deps (requirements.txt) + package
make data           # download from public mirror, verify row counts, parquet
make sql            # DuckDB model (sql/01-06)
make train          # scoring, bootstrap, clusters, OOT validation, exposure, scenarios, figures
make report         # R GLM/GLMM/NB + Quarto HTML, memo PDF, README, executed notebooks
make app-data       # export app/data
make test           # ruff + pytest
make all            # everything above in order
make app            # Streamlit on :8501
```
Requirements: Python 3.11 (pinned in `requirements.txt` / `requirements-ci.txt`), R >= 4.3 with the packages in
[`R/README.md`](R/README.md), Quarto >= 1.4 (optional), WeasyPrint for the memo PDF. All parameters live in
[`config/config.yaml`](config/config.yaml); results are deterministic (seeded).

**Tests** (`tests/`): schema, grain uniqueness, null keys, SQL edge cases (same-day delivery, bad clocks, missing
timestamps, review de-dup, multi-seller orders), **leakage / out-of-time** (mutating post-cutoff data must not change
features), shrinkage and bootstrap unit tests, metric correctness, scenario logic, real-data metric regression (skipped
without artifacts) and Streamlit **AppTest** for all eight pages. CI (GitHub Actions) runs ruff + pytest on synthetic
fixtures.

## License
Code: MIT ([LICENSE](LICENSE)). Data and derived data: CC BY-NC-SA 4.0 (Olist).
"""
    (ROOT / "README.md").write_text(md)
    print("README written")


if __name__ == "__main__":
    main()
