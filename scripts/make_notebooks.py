"""Create the two walkthrough notebooks (executed afterwards with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import os, json, duckdb, pandas as pd, numpy as np, matplotlib.pyplot as plt
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)
con = duckdb.connect('data/processed/scms.duckdb', read_only=True)"""
SCOPE = ("**Scope:** USAID / PEPFAR SCMS delivery history, one programme, scheduled 2006-2015. The buyer is a US government "
         "health-supply programme and deliveries go to PEPFAR countries (not US-domestic). On time = delivered-to-client date <= "
         "scheduled date. There is **no vendor hand-off timestamp**. Scores describe recorded delivery performance for this "
         "programme, not vendor quality overall, on limited samples.")

eda = nbf.v4.new_notebook()
eda.cells = [
    md("# 01 - Exploratory data analysis: SCMS delivery history\nAll tables come from the DuckDB model built by `make sql` "
       "(`sql/01..06`). Raw data is not committed - run `make data` first.\n\n" + SCOPE),
    code(SETUP),
    md("## Provenance and integrity"),
    code("json.load(open('artifacts/data_manifest.json'))"),
    code("con.execute('SELECT * FROM cleaning_log ORDER BY ord').df()"),
    code("con.execute('SELECT * FROM dq_report').df()"),
    md("## Grain of the model"),
    code("""con.execute('''SELECT 'stg_lines' t, COUNT(*) n FROM stg_lines UNION ALL SELECT 'fct_lines', COUNT(*) FROM fct_lines UNION ALL
SELECT 'fct_shipments', COUNT(*) FROM fct_shipments UNION ALL SELECT 'vendor_summary', COUNT(*) FROM vendor_summary UNION ALL
SELECT 'vendor_year_panel', COUNT(*) FROM vendor_year_panel''').df()"""),
    md("## Who delivers what: external vendors vs the SCMS regional distribution centre"),
    code("""con.execute('''SELECT is_internal_rdc, COUNT(*) n_lines, ROUND(AVG(is_late::INT),4) late_rate, ROUND(AVG(is_severe_late::INT),4) severe_rate,
ROUND(AVG(is_exact_match::INT),3) delivered_eq_scheduled, ROUND(SUM(line_value)/1e6,1) value_usd_m FROM fct_lines GROUP BY 1''').df()"""),
    md("The exact-match share is high in both channels. If scheduled dates were back-filled with actual dates in some records, "
       "the on-time rate is overstated; the file cannot tell us."),
    md("## Lateness by scheduled year"),
    code("""py = con.execute('SELECT * FROM programme_year ORDER BY sched_year').df()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(py.sched_year, py.n_late_vendor / py.n_lines_vendor, 'o-', label='external vendors')
ax.plot(py.sched_year, py.n_late_rdc / py.n_lines_rdc.where(py.n_lines_rdc > 0), 's-', label='SCMS regional DC')
ax.legend(); ax.set_title('Late share by scheduled year'); plt.tight_layout(); py[['sched_year','n_lines_vendor','n_late_vendor','n_lines_rdc','n_late_rdc']]"""),
    md("## Delivery offset distribution (delivered minus scheduled)"),
    code("""o = con.execute('SELECT offset_days FROM fct_lines WHERE NOT is_internal_rdc').df().offset_days
print(o.describe(percentiles=[.01, .05, .5, .9, .95, .99]).round(1)); print('late:', (o > 0).mean().round(4), ' severe (>14):', (o > 14).mean().round(4))"""),
    md("## Vendor volume distribution (why shrinkage is needed)"),
    code("""s = con.execute('SELECT n_lines, value_usd FROM vendor_summary WHERE NOT is_internal_rdc').df()
print(s.n_lines.describe(percentiles=[.25, .5, .75, .9]).round(1))
print(f"vendors with <10 lines: {(s.n_lines < 10).sum()} of {len(s)}; top 10% by value hold {s.value_usd.sort_values(ascending=False).head(int(len(s)*.1)).sum()/s.value_usd.sum():.1%} of value")"""),
    md("## Window functions on the vendor-year panel"),
    code("""con.execute('''SELECT vendor, sched_year, n_lines, late_rate, late_rate_roll3, cum_lines, volume_rank_in_year
FROM vendor_year_panel WHERE vendor = (SELECT vendor FROM vendor_summary WHERE NOT is_internal_rdc ORDER BY n_lines DESC LIMIT 1)
ORDER BY sched_year''').df()"""),
    md("## Lanes"),
    code("""con.execute('''SELECT ship_mode, COUNT(*) n, ROUND(AVG(is_late::INT),4) late_rate FROM fct_lines WHERE NOT is_internal_rdc GROUP BY 1 ORDER BY n DESC''').df()"""),
    code("from IPython.display import Image; Image('docs/figures/programme_yearly_lateness.png', width=800)"),
]

mw = nbf.v4.new_notebook()
mw.cells = [
    md("# 02 - Model walkthrough: shrinkage, composite score, out-of-time validation, archetypes, exposure\n"
       "Reads `artifacts/` produced by `make train`.\n\n" + SCOPE),
    code(SETUP + "\nfrom supplier_risk.shrinkage import BetaPrior, posterior_mean\nt = json.load(open('artifacts/train_summary.json'))\nsc = pd.read_parquet('artifacts/vendor_scores.parquet')"),
    md("## 1. Empirical-Bayes priors (maximum marginal likelihood)"),
    code("pd.DataFrame(t['priors'], index=['a','b','mean','strength']).T"),
    code("""p = BetaPrior(*t['priors']['late'][:2])
pd.DataFrame({'n_lines':[1,5,20,100,1000], 'n_late':[1,2,4,20,200]}).assign(
    raw=lambda d: d.n_late/d.n_lines, shrunk=lambda d: posterior_mean(d.n_late, d.n_lines, p))"""),
    md("A vendor with 1 late line out of 1 is pulled almost entirely to the prior; at 1,000 lines the data dominate."),
    code("from IPython.display import Image; Image('docs/figures/shrinkage_and_intervals.png', width=850)"),
    md("## 2. Composite score and tiers (scored vendors have >= 10 lines)"),
    code("pd.read_csv('artifacts/risk_tier_summary.csv')"),
    code("pd.read_csv('artifacts/top_risk_vendors.csv').head(10)"),
    code("pd.read_csv('artifacts/weight_sensitivity_ranking.csv')"),
    md("## 3. Out-of-time validation vs baselines"),
    code("ev = pd.read_csv('artifacts/oot_metrics_main.csv'); ev.round(3)"),
    code("pd.DataFrame(t['auc_deltas']).T.round(4)"),
    code("pd.read_csv('artifacts/oot_backtest_metrics.csv').query(\"score in ['Composite risk score (shrunk)','Shrunk late rate (EB) only','Baseline: raw late rate']\")[['split','target','score','line_auc','n_vendors']].round(3)"),
    code("pd.read_csv('artifacts/oot_shrinkage_calibration.csv').round(4)"),
    code("Image('docs/figures/oot_validation_auc.png', width=900)"),
    md("**Reading:** with 21 evaluable vendors and 108 late lines the intervals are wide. The raw and EB-shrunk late rates rank "
       "next-period lateness at least as well as the composite in the main split; volume-only and lane-mix-only baselines rank the wrong way."),
    md("## 4. Cold start: New / unscored tier"),
    code("pd.read_csv('artifacts/cold_start.csv').round(4)"),
    md("## 5. Archetypes (indicative)"),
    code("pd.read_csv('artifacts/cluster_selection.csv').round(3)"),
    code("pd.read_csv('artifacts/archetype_profiles.csv').round(3)"),
    code("Image('docs/figures/vendor_archetypes.png', width=900)"),
    md("## 6. Exposure and scenario (ASSUMED parameters)"),
    code("{k: round(v) if v > 10 else round(v, 3) for k, v in t['exposure_totals'].items()}"),
    code("pd.DataFrame([t['scenario_base_ASSUMED'], t['scenario_raw_baseline_ASSUMED']], index=['composite ranking','raw late-rate ranking']).T"),
    code("pd.read_csv('artifacts/exposure_sensitivity.csv').round(3)"),
    md("## 7. R layer summary"),
    code("json.load(open('artifacts/r_glm/r_summary.json'))"),
]
nbf.write(eda, "notebooks/01_eda.ipynb")
nbf.write(mw, "notebooks/02_model_walkthrough.ipynb")
print("notebooks written")
