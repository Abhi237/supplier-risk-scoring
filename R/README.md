# R layer

`delivery_analysis.R` reads `data/processed/lines_for_r.csv` (all 10,324 lines, written by `make sql` with an explicit `ORDER BY line_id` so results are
reproducible; the models use the 4,920 external-vendor lines, and one all-lines GLM estimates the regional-DC contrast) and fits:

* a logistic GLM for late delivery (mode, product group, country group, Incoterm, period, log value, log freight ratio plus a freight-missing flag, shipment size),
  with VIF, a likelihood-ratio test for period, and calibration deciles;
* a vendor random-intercept GLMM (`lme4`, bobyqa) on vendors with at least 5 lines, a null model for the latent ICC and
  median odds ratio, and a shipment-grain sensitivity GLMM;
* a shifted negative binomial (`MASS::glm.nb`) for days late among late lines (days_late - 1);
* KS, Wilcoxon and Kruskal-Wallis comparisons of signed delivery offsets (air vs truck, ARV vs HRDT, DDP vs EXW, vendors vs the
  SCMS regional distribution centre, 2013-15 vs 2010-12).

Outputs go to `artifacts/r_glm/` (CSV, `r_summary.json`, `sessionInfo.txt`) and `docs/figures/r_*.png`;
`reports/delivery_report.qmd` renders them to `reports/delivery_report.html`.

```bash
Rscript R/delivery_analysis.R            # about a minute on 4,920 lines
cd reports && quarto render delivery_report.qmd --to html
```

## Reproducible R environment

`renv.lock` at the repository root pins R 4.5.0 and the packages used (readr, dplyr, tidyr, ggplot2, MASS, lme4, broom, pROC, car,
jsonlite, knitr, rmarkdown plus dependencies). The package set of this analysis is the same as the one the lockfile was written
from, and the versions that produced the committed results match it (for example MASS 7.3-65, lme4 1.1-37, ggplot2 3.5.1,
dplyr 1.1.4, broom 1.0.7, pROC 1.18.5, car 3.1-3). The lockfile was therefore not re-snapshotted for this branch; the exact session
is recorded in `artifacts/r_glm/sessionInfo.txt`. Quarto renders the report and is not an R package, so it is not in the lockfile.

```bash
Rscript -e 'renv::restore(lockfile = "renv.lock", prompt = FALSE)'   # run from the repo root
```
