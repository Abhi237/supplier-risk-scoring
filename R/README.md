# R layer

`delivery_analysis.R` fits the logistic GLM, the seller random-intercept GLMM (lme4), the negative-binomial
lead-time model (MASS) and the KS / Wilcoxon / Kruskal-Wallis comparisons, writing `artifacts/r_glm/` and
`docs/figures/r_*.png`. `reports/delivery_report.qmd` renders those outputs to HTML.

```bash
Rscript R/delivery_analysis.R                       # ~8 min (GLMM + NB on 96k orders)
cd reports && quarto render delivery_report.qmd --to html
```

## Reproducible R environment (`renv.lock`)

`renv.lock` at the repository root pins R 4.5.0 and the 96 packages (direct + recursive dependencies) used by
`R/delivery_analysis.R` and `reports/delivery_report.qmd`: readr, dplyr, tidyr, ggplot2, MASS, lme4, broom, pROC,
car, jsonlite, knitr, rmarkdown. It was written with `renv::snapshot(packages = ..., force = TRUE)` from the library
that produced the committed results. Key versions: MASS 7.3-65, lme4 1.1-37, ggplot2 3.5.1, dplyr 1.1.4,
readr 2.1.5, tidyr 1.3.1, broom 1.0.7, pROC 1.18.5, car 3.1-3, jsonlite 1.9.1, knitr 1.50. Quarto 1.10.19 renders
the report (Quarto is not an R package, so it is not in the lockfile).

Restore into a project-local library:

```bash
Rscript -e 'install.packages("renv", repos = "https://cloud.r-project.org")'
Rscript -e 'renv::restore(lockfile = "renv.lock", prompt = FALSE)'   # run from the repo root
Rscript R/delivery_analysis.R
```

To refresh the lockfile after upgrading packages:
`Rscript -e 'renv::snapshot(packages = c("readr","dplyr","ggplot2","MASS","lme4","broom","pROC","tidyr","car","jsonlite","knitr","rmarkdown"), force = TRUE, prompt = FALSE)'`.
