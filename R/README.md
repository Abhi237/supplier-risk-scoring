# R layer

`delivery_analysis.R` fits the logistic GLM, the seller random-intercept GLMM (lme4), the negative-binomial
lead-time model (MASS) and the KS / Wilcoxon / Kruskal-Wallis comparisons, writing `artifacts/r_glm/` and
`docs/figures/r_*.png`. `reports/delivery_report.qmd` renders those outputs to HTML.

```bash
Rscript R/delivery_analysis.R                       # ~8 min (GLMM + NB on 96k orders)
cd reports && quarto render delivery_report.qmd --to html
```

**Versions used for the committed results** (no `renv.lock`; the environment is a system R library):
R 4.5.0, Quarto 1.10.19, MASS 7.3.65, lme4 1.1.37, ggplot2 3.5.1, dplyr 1.1.4, readr 2.1.5, tidyr 1.3.1,
broom 1.0.7, pROC 1.18.5, car 3.1.3, jsonlite 1.9.1, knitr 1.50. To pin with renv:
`install.packages("renv"); renv::init(bare = TRUE); renv::install(c("MASS@7.3-65", "lme4@1.1-37", ...)); renv::snapshot()`.
