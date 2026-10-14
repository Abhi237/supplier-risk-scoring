import streamlit as st
from core import callout, setup_page, summary

s = summary()
t, c = s["train_summary"], s["config"]
setup_page("Methodology & limitations", "How the scores are built, how they were validated, and what they can't tell you")
callout("<b>Olist sellers are proxies for suppliers.</b> They are independent marketplace merchants shipping through "
        "Olist's carrier network, not a buyer's contracted vendors with POs, lead-time contracts or invoices. The "
        "<i>method</i> (shrunk reliability rates, point-in-time validation, exposure proxies) transfers to real "
        "procurement data; the vendor-level facts do not.")
st.markdown(f"""
### Definitions
| Metric | Definition | Leg |
|---|---|---|
| Late delivery | customer delivery date > estimated delivery date (calendar days) | end-to-end |
| Hand-off late | carrier hand-off timestamp > item `shipping_limit_date` | **seller-controlled** |
| Cancellation | order status `canceled` / `unavailable` | seller / platform |
| Freight / price | freight value / item price | cost friction |
| GMV at risk | GMV × EB-shrunk late rate | exposure |
| WC in transit | GMV × median delivery days / 365 | working-capital proxy |

### Scoring
* **Empirical-Bayes beta-binomial shrinkage.** For each rate, a Beta(a, b) prior is fitted by maximum marginal
  likelihood across sellers: late Beta({t['priors']['late'][0]:.2f}, {t['priors']['late'][1]:.2f}) (mean
  {t['priors']['late'][2]:.1%}, strength {t['priors']['late'][3]:.0f} pseudo-items); hand-off Beta(
  {t['priors']['handoff'][0]:.2f}, {t['priors']['handoff'][1]:.2f}). Posterior mean = (a + k) / (a + b + n).
* **Review score** is shrunk toward the population mean with 5 pseudo-reviews.
* **Composite** = weighted sum of component percentiles × 100. Weights (documented a priori, not tuned):
  `{c['scoring']['weights']}`. Three alternative weight vectors are reported as a sensitivity check.
* **Uncertainty:** {c['scoring']['n_bootstrap']} Bayesian-bootstrap draws per seller (Dirichlet-weighted outcomes with
  the EB prior as pseudo-data) → 5th-95th percentile interval of the score. A classical bootstrap collapses to zero
  width for sellers with no late items, which is why the Bayesian variant is used.

### Validation (out-of-time)
* Features at score date T use only items purchased before T **and** outcomes observed before T (deliveries, hand-offs,
  reviews created before T). Unit tests mutate post-T data and assert features are unchanged.
* Outcomes: items purchased in [T, T+). Main split T = {c['scoring']['score_period_end']}; rolling splits
  {c['scoring']['backtest_splits']}.
* Metrics: item-level ROC-AUC (each next-period item inherits its seller's score), seller-level AUC, Spearman
  correlation (sellers with >= 10 next-period items), top-decile lift and capture. Baselines: raw (unshrunk) rates and
  volume-only. ΔAUC confidence intervals from a seller-cluster bootstrap.

### Limitations
* **Weak persistence of end-to-end lateness.** Seller scores predict next-period hand-off lateness (AUC ≈ 0.67-0.70)
  much better than end-to-end lateness (≈ 0.54). Carrier, lane and calendar shocks (e.g. Feb-Mar 2018) dominate.
* **Clusters are soft** (silhouette {t['clusters']['silhouette']:.2f}).
* **Costs are ASSUMED.** Exposure in BRL beyond GMV / freight is a scenario, not a P&L.
* **Survivorship / censoring.** Orders purchased near the data end that were not yet delivered are not judged; sellers
  new in the next period cannot be scored (coverage {t['oot_main']['next_items_covered_share']:.0%} of next-period items).
* **Multi-seller orders** share one delivery timestamp ({s['sql_summary']['multi_seller_order_share']:.1%} of orders).
* **Observational.** GLM odds ratios are associations; a late hand-off may proxy for unobserved capacity issues.
* **License:** Olist data is CC BY-NC-SA 4.0 (non-commercial, share-alike); derived outputs inherit those terms.
""")
