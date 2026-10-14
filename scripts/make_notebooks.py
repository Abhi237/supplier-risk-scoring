"""Create the two walkthrough notebooks (executed afterwards with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import os, json, duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)
con = duckdb.connect('data/processed/olist.duckdb', read_only=True)"""

eda = nbf.v4.new_notebook()
eda.cells = [
    md("# 01 - Exploratory data analysis: Olist sellers as supplier proxies\nAll tables come from the DuckDB model built by "
       "`make sql` (`sql/01..06`). Raw data is not committed - run `make data` first.\n\n"
       "**Methodology note:** Olist marketplace sellers are used as *proxies* for suppliers."),
    code(SETUP),
    md("## Integrity and data-quality profile"),
    code("json.load(open('artifacts/data_manifest.json'))"),
    code("con.execute('SELECT * FROM dq_report').df()"),
    md("## Grain of the model"),
    code("""con.execute('''SELECT 'fct_items' t, COUNT(*) n FROM fct_items UNION ALL
SELECT 'fct_orders_delivered', COUNT(*) FROM fct_orders_delivered UNION ALL
SELECT 'seller_summary', COUNT(*) FROM seller_summary UNION ALL SELECT 'seller_month_panel', COUNT(*) FROM seller_month_panel''').df()"""),
    md("## Monthly lateness: platform-wide shocks"),
    code("""pm = con.execute("SELECT * FROM platform_month WHERE purchase_month BETWEEN '2017-01-01' AND '2018-08-01'").df()
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(pm.purchase_month, pm.late_rate, 'o-', label='late vs estimate'); ax.plot(pm.purchase_month, pm.handoff_late_rate, 's-', label='hand-off late')
ax.legend(); ax.set_title('Monthly late rates'); plt.tight_layout(); pm[['purchase_month','n_items','late_rate','handoff_late_rate','median_delivery_days']].tail(10)"""),
    md("The Nov-2017 and Feb-Mar 2018 spikes hit all sellers at once - they are not seller-specific."),
    md("## Interval decomposition"),
    code("""con.execute('''SELECT ROUND(MEDIAN(approval_hours),1) approval_h, ROUND(MEDIAN(handoff_days),2) purchase_to_carrier_d,
ROUND(MEDIAN(carrier_transit_days),2) carrier_transit_d, ROUND(MEDIAN(delivery_days),2) purchase_to_customer_d,
ROUND(AVG(is_late::INT),4) late_rate, ROUND(AVG(is_handoff_late::INT),4) handoff_late_rate FROM fct_orders_delivered''').df()"""),
    code("""con.execute('''SELECT is_handoff_late, COUNT(*) n, AVG(is_late::INT) late_rate, AVG(review_score) review
FROM fct_orders_delivered WHERE is_handoff_late IS NOT NULL GROUP BY 1''').df()"""),
    md("## Seller volume distribution (why shrinkage is needed)"),
    code("""s = con.execute('SELECT n_items, gmv FROM seller_summary').df()
print(s.n_items.describe(percentiles=[.25,.5,.75,.9,.99]).round(1))
print(f"sellers with <5 items: {(s.n_items<5).mean():.1%}; top 10% GMV share: {s.gmv.sort_values(ascending=False).head(int(len(s)*.1)).sum()/s.gmv.sum():.1%}")"""),
    md("## Window functions on the seller-month panel"),
    code("""con.execute('''SELECT seller_id, purchase_month, n_items, late_rate, late_rate_roll3, cum_items, gmv_rank_in_month
FROM seller_month_panel WHERE seller_id = (SELECT seller_id FROM seller_summary ORDER BY gmv DESC LIMIT 1)
ORDER BY purchase_month''').df().tail(8)"""),
    md("## Freight and lanes"),
    code("""con.execute('''SELECT same_state, COUNT(*) n, MEDIAN(delivery_days) med_days, AVG(is_late::INT) late_rate,
MEDIAN(freight_to_price) med_freight_ratio FROM fct_orders_delivered GROUP BY 1''').df()"""),
    code("from IPython.display import Image; Image('docs/figures/handoff_delay_vs_late.png', width=700)"),
]

mw = nbf.v4.new_notebook()
mw.cells = [
    md("# 02 - Model walkthrough: shrinkage, composite score, out-of-time validation, archetypes, exposure\n"
       "Reads `artifacts/` produced by `make train`. Sellers are supplier proxies."),
    code(SETUP + "\nfrom supplier_risk.shrinkage import BetaPrior, posterior_mean\nt = json.load(open('artifacts/train_summary.json'))\nsc = pd.read_parquet('artifacts/seller_scores.parquet')"),
    md("## 1. Empirical-Bayes priors (maximum marginal likelihood)"),
    code("pd.DataFrame(t['priors'], index=['a','b','mean','strength']).T"),
    code("""p = BetaPrior(*t['priors']['late'][:2])
pd.DataFrame({'n_delivered':[1,5,20,100,1000], 'n_late':[1,2,4,20,200]}).assign(
    raw=lambda d: d.n_late/d.n_delivered, shrunk=lambda d: posterior_mean(d.n_late, d.n_delivered, p))"""),
    md("A seller with 1 late out of 1 is pulled almost entirely to the prior; at 1,000 deliveries the data dominate."),
    code("from IPython.display import Image; Image('docs/figures/shrinkage_and_intervals.png', width=850)"),
    md("## 2. Composite score and tiers"),
    code("pd.read_csv('artifacts/risk_tier_summary.csv')"),
    code("pd.read_csv('artifacts/top25_risk_sellers.csv').head(10)"),
    code("pd.read_csv('artifacts/weight_sensitivity_ranking.csv')"),
    md("## 3. Out-of-time validation vs baselines"),
    code("ev = pd.read_csv('artifacts/oot_metrics_main.csv'); ev.round(3)"),
    code("pd.DataFrame(t['auc_deltas']).T.round(4)"),
    code("pd.read_csv('artifacts/oot_backtest_metrics.csv').query(\"score in ['Composite risk score (shrunk)','Shrunk hand-off late rate (EB) only','Baseline: raw hand-off late rate']\")[['split','target','score','item_auc','lift_top10']].round(3)"),
    code("pd.read_csv('artifacts/oot_shrinkage_calibration.csv').round(4)"),
    code("Image('docs/figures/oot_validation_auc.png', width=900)"),
    md("**Reading:** seller scores predict next-period *hand-off* lateness (seller-controlled) far better than "
       "*end-to-end* lateness, which is dominated by lane/carrier/calendar effects (see R GLMM ICC)."),
    md("## 4. Archetypes"),
    code("pd.read_csv('artifacts/cluster_selection.csv').round(3)"),
    code("pd.read_csv('artifacts/archetype_profiles.csv').round(3)"),
    code("Image('docs/figures/seller_archetypes.png', width=900)"),
    md("## 5. Exposure and scenario (ASSUMED parameters)"),
    code("{k: round(v) if v > 10 else round(v, 3) for k, v in t['exposure_totals'].items()}"),
    code("pd.DataFrame([t['scenario_base_ASSUMED'], t['scenario_raw_baseline_ASSUMED']], index=['composite','raw late rate']).T"),
    code("pd.read_csv('artifacts/exposure_sensitivity.csv').round(3)"),
    md("## 6. R layer summary"),
    code("json.load(open('artifacts/r_glm/r_summary.json'))"),
]
nbf.write(eda, "notebooks/01_eda.ipynb")
nbf.write(mw, "notebooks/02_model_walkthrough.ipynb")
print("notebooks written")
