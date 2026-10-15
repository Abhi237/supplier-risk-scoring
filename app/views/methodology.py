import streamlit as st
from core import MEMO_URL, REPO_URL, SOURCE_CITATION, callout, csv, setup_page, summary

s = summary()
_b = csv("oot_backtest_metrics.csv")
_b = _b[_b.target.str.startswith("Late")].pivot(index="split", columns="score", values="line_auc")
_comp_wins = int((_b["Composite risk score (shrunk)"] > _b["Baseline: raw late rate"]).sum())
t, c, q = s["train_summary"], s["config"], s["sql_summary"]
setup_page("Methodology & limitations", "How the scores are built, how they were validated, and what they can't tell you",
           section="methodology")
callout("<b>Buyer and scope.</b> The buyer is a US government health-supply programme (USAID / PEPFAR SCMS). Commodities (antiretrovirals, "
        "HIV test kits and a few others) are shipped to PEPFAR-supported countries, mostly in sub-Saharan Africa. These are <b>not</b> US-domestic "
        "deliveries. Vendors are the real suppliers named in the dataset. Scores describe <b>recorded delivery performance for this programme</b>, "
        "not vendor quality overall, on limited samples.")
w = c["scoring"]["weights"]
pr = t["priors"]
st.markdown(f"""
### Definitions
| Metric | Definition |
|---|---|
| On time | `Delivered to Client Date` <= `Scheduled Delivery Date` (calendar days) |
| Late / days late | delivered after scheduled; days late = delivered - scheduled (positive part, capped at {c['cleaning']['days_late_cap']} for means) |
| Severe late | more than {c['cleaning']['severe_late_days']} days late |
| Freight ratio | shipment freight (USD, numeric first line only) / shipment line value, capped at {c['cleaning']['freight_ratio_cap']:.0f}x, value-weighted per vendor |
| Lane mix risk | pooled late rate of the vendor's destination countries and shipment modes (leave-one-vendor-out), line-weighted; context, not vendor-controlled |
| Value at risk | line value x EB-shrunk late rate |
| Known date | GREATEST(delivered date, delivery recorded date): when an outcome could be known in the system |

**There is no vendor hand-off timestamp in the dataset**, so the vendor-controlled vs end-to-end split used in the earlier Olist version of this
project is dropped; lateness mixes vendor, freight-forwarder and in-country effects.

### Scoring
* **Population.** {q['ext_vendors']} external vendors. The SCMS regional distribution centre ('{c['cleaning']['internal_channel_vendor']}', {q['rdc_line_share']:.0%} of lines)
  is the programme's own logistics and is excluded from scoring.
* **Empirical-Bayes beta-binomial shrinkage.** A Beta(a, b) prior is fitted by maximum marginal likelihood across vendors: late Beta({pr['late'][0]:.2f}, {pr['late'][1]:.2f})
  (mean {pr['late'][2]:.1%}, strength {pr['late'][3]:.0f} pseudo-lines); severe late Beta({pr['severe'][0]:.2f}, {pr['severe'][1]:.2f}). Posterior mean = (a + k) / (a + b + n).
  Mean days late and freight ratio are shrunk with pseudo-count priors ({c['scoring']['mean_days_prior_n']} lines, ASSUMED smoothing constant; median vendor shipment value).
* **Composite** = weighted sum of component percentiles x 100 among scored vendors (>= {c['scoring']['min_lines_for_score']} lines). Weights documented a priori, not tuned:
  `{w}`. Three alternative weight vectors are a sensitivity check.
* **Uncertainty:** {c['scoring']['n_bootstrap']} Bayesian-bootstrap draws per scored vendor (Dirichlet-weighted outcomes with the EB prior as pseudo-data; mean days late resampled over
  the vendor's own lines) give a 5th-95th percentile score interval. A classical bootstrap collapses to zero width for vendors with no late lines.
* **Cold start:** vendors below {c['scoring']['min_lines_for_score']} lines are New / unscored: prior-mean rates, no composite.

### Validation (out-of-time)
* Features at score date T use lines scheduled before T whose outcome was known before T; lines still unresolved at T are excluded. Outcomes: lines scheduled in [T, T_end).
  Main split T = {c['scoring']['score_period_end']} to {c['scoring']['validate_period_end']}, fixed from line volume (about half of vendor lines before it) before any metric was run; rolling splits
  {c['scoring']['backtest_splits']}.
* Metrics: line-level ROC-AUC (each next-period line inherits its vendor's score), vendor-level AUC, Spearman, top-20% lift and capture. Baselines: raw rate, volume-only, lane-mix-only.
  AUC differences use a vendor-cluster bootstrap.

### Limitations
* **One programme, one definition.** Scheduled 2006-2015, nominal USD, a single US government buyer. {q['ext_exact_match_share']:.0%} of vendor lines have delivered date = scheduled date exactly, so on-time may be overstated
  if scheduled dates were back-filled.
* **Small samples.** {q['ext_vendors_lt10_lines']} of {q['ext_vendors']} vendors have fewer than 10 lines; only {t['oot_main']['n_vendors_evaluated']} vendors can be evaluated out-of-time and {int(t['oot_main']['next_late_lines'])} late lines
  drive the main-split metrics. Intervals are wide and most method differences are not distinguishable from zero.
* **A simple baseline wins.** In the main split the raw and shrunk late rates rank next-period lateness better than the composite, and across the {len(_b)} rolling splits the composite beats the raw late rate in {_comp_wins}. Volume-only and lane-mix-only baselines rank the wrong way.
* **Only delivered lines** are in the file; undelivered or cancelled lines are absent, so late rates are lower bounds.
* **Clusters are soft** (silhouette {t['clusters']['silhouette']:.2f}, {t['clusters']['n_clustered']} vendors).
* **Costs are ASSUMED.** Exposure beyond observed value and days late is a scenario. The reallocation scenario also assumes other vendors can supply the same product and country, which is often false for sole-source health commodities.
* **Observational.** GLM odds ratios are associations. The vendor GLMM effect partly reflects product, country and period niche. Lines within a shipment share dates (shipment-grain sensitivity reported).
* **Vendor names** are used as recorded; spelling variants are not merged. Scores are not a statement about any vendor's quality or ethics.
* **Licence is unclear;** only derived scorecards and aggregates are published. Source: {SOURCE_CITATION}.
""")

st.markdown(f"[GitHub repository]({REPO_URL}) · [Procurement memo (PDF)]({MEMO_URL}) · Stack: DuckDB SQL · empirical Bayes · scikit-learn · "
            "R (GLM, lme4, MASS) · Quarto · Streamlit")
