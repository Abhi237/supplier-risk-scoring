# Delivery-lateness statistical layer (R) for the USAID / PEPFAR SCMS Delivery History:
#   logistic GLM for late delivery, vendor random-intercept GLMM, negative-binomial count model for days late,
#   KS / Wilcoxon comparisons of delivered-minus-scheduled days, diagnostics.
# Input : data/processed/lines_for_r.csv  (line grain, explicit ORDER BY line_id, no free text)
# Output: artifacts/r_glm/*.csv, docs/figures/r_*.png
# On-time = delivered-to-client date <= scheduled delivery date.  There is no vendor hand-off timestamp in the data, so
# "late" mixes vendor, forwarder and in-country effects.  The modelled population is the EXTERNAL-vendor lines (the SCMS
# regional distribution centre is not a vendor); one all-lines GLM compares the two channels.
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(MASS); library(lme4); library(broom)
  library(pROC); library(tidyr); library(car); library(jsonlite)
})
set.seed(42)
root <- Sys.getenv("SUPPLIER_RISK_ROOT", unset = normalizePath("."))
out <- file.path(root, "artifacts", "r_glm"); dir.create(out, showWarnings = FALSE, recursive = TRUE)
figs <- file.path(root, "docs", "figures"); dir.create(figs, showWarnings = FALSE, recursive = TRUE)
min_country_n <- 100

raw <- read_csv(file.path(root, "data/processed/lines_for_r.csv"), show_col_types = FALSE) %>% arrange(line_id)
prep <- function(x) {
  top_c <- x %>% filter(!is_internal_rdc) %>% count(country, sort = TRUE) %>% filter(n >= min_country_n) %>% pull(country)
  x %>% mutate(
    late = as.integer(is_late), severe = as.integer(is_severe_late),
    mode = factor(case_when(ship_mode == "Air" ~ "Air", ship_mode == "Truck" ~ "Truck", ship_mode == "Ocean" ~ "Ocean",
                            TRUE ~ "Air charter / unknown"), levels = c("Air", "Truck", "Ocean", "Air charter / unknown")),
    # ANTM / ACT / MRDT (46 external lines, none late) are pooled with HRDT to avoid complete separation
    pgroup = relevel(factor(ifelse(product_group == "ARV", "ARV", "HRDT / other")), "ARV"),
    country_g = relevel(factor(ifelse(country %in% top_c, country, "Other countries")), "South Africa"),
    inco = relevel(factor(ifelse(inco_term %in% c("EXW", "DDP"), inco_term, "Other terms")), "EXW"),
    year_g = factor(case_when(sched_year <= 2008 ~ "2006-08", sched_year <= 2009 ~ "2009", sched_year <= 2012 ~ "2010-12",
                              TRUE ~ "2013-15"), levels = c("2010-12", "2006-08", "2009", "2013-15")),
    log_value = as.numeric(scale(log1p(pmax(coalesce(line_value, 0), 0)))),   # standardised: one SD of log value
    freight_missing = as.integer(is.na(ship_freight_ratio)),
    log_freight_ratio = ifelse(is.na(ship_freight_ratio), 0, log1p(pmin(ship_freight_ratio, 2))),
    log_ship_lines = log(ship_n_lines), vendor = factor(vendor))
}
dall <- prep(raw)
d <- dall %>% filter(!is_internal_rdc) %>% droplevels()
cat(nrow(d), "external-vendor lines;", sum(d$late), "late;", nlevels(d$vendor), "vendors\n")

# ---------------- 1. Logistic GLM (external-vendor lines) ----------------
f_glm <- late ~ mode + pgroup + country_g + inco + year_g + log_value + log_freight_ratio + freight_missing + log_ship_lines
m_glm <- glm(f_glm, data = d, family = binomial())
or <- tidy(m_glm, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ signif(.x, 5)))
write_csv(or, file.path(out, "glm_odds_ratios.csv"))
d$p_glm <- fitted(m_glm)
auc_glm <- as.numeric(auc(roc(d$late, d$p_glm, quiet = TRUE)))
m_year0 <- update(m_glm, . ~ . - year_g)
lrt_year <- anova(m_year0, m_glm, test = "LRT")
vif_tab <- as.data.frame(car::vif(m_glm)); vif_tab$term <- rownames(vif_tab)
write_csv(vif_tab, file.path(out, "glm_vif.csv"))
calib <- d %>% mutate(bin = ntile(p_glm, 10)) %>% group_by(bin) %>%
  summarise(mean_pred = mean(p_glm), observed = mean(late), n = n(), .groups = "drop")
write_csv(calib, file.path(out, "glm_calibration_deciles.csv"))
hl_stat <- sum((calib$observed - calib$mean_pred)^2 * calib$n / (calib$mean_pred * (1 - calib$mean_pred)))
hl_p <- pchisq(hl_stat, df = 8, lower.tail = FALSE)

# all lines: is the SCMS regional distribution centre later than external vendors, after controls?
m_all <- glm(late ~ is_internal_rdc + mode + pgroup + year_g + log_value, data = dall %>% mutate(is_internal_rdc = as.integer(is_internal_rdc)),
             family = binomial())
or_all <- tidy(m_all, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) %>% filter(term == "is_internal_rdc")

# ---------------- 2. Vendor random-intercept GLMM ----------------
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
vn <- d %>% count(vendor) %>% filter(n >= 5)
dm <- d %>% filter(vendor %in% vn$vendor) %>% droplevels()
f_fix <- ~ mode + pgroup + inco + year_g + log_value + log_freight_ratio + freight_missing + log_ship_lines
m_re0 <- glmer(late ~ 1 + (1 | vendor), data = dm, family = binomial(), control = ctrl)
m_re  <- glmer(update(f_fix, late ~ . + (1 | vendor)), data = dm, family = binomial(), control = ctrl)
m_fx  <- glm(update(f_fix, late ~ .), data = dm, family = binomial())
icc_fun <- function(m) { v <- as.data.frame(VarCorr(m))$vcov[1]; v / (v + pi^2 / 3) }
mor_fun <- function(m) { v <- as.data.frame(VarCorr(m))$vcov[1]; exp(sqrt(2 * v) * qnorm(0.75)) }
lr_vendor <- 2 * (as.numeric(logLik(m_re)) - as.numeric(logLik(m_fx)))   # boundary: p-value is conservative (chi2 on 1 df / 2)
p_vendor <- 0.5 * pchisq(lr_vendor, df = 1, lower.tail = FALSE)
re <- ranef(m_re, condVar = TRUE)$vendor
vend_eff <- tibble(vendor = rownames(re), vendor_logit_effect = re[, 1],
                   se = sqrt(as.numeric(attr(re, "postVar")))) %>%
  left_join(dm %>% count(vendor = as.character(vendor), name = "n_lines"), by = "vendor") %>% arrange(vendor)
write_csv(vend_eff, file.path(out, "glmm_vendor_effects.csv"))
fe <- summary(m_re)$coefficients
or_re <- tibble(term = rownames(fe), estimate = exp(fe[, 1]), conf.low = exp(fe[, 1] - 1.96 * fe[, 2]),
                conf.high = exp(fe[, 1] + 1.96 * fe[, 2]), p.value = fe[, 4]) %>%
  mutate(across(where(is.numeric), ~ signif(.x, 5)))
write_csv(or_re, file.path(out, "glmm_odds_ratios.csv"))
# sensitivity: one row per shipment (late if any line late) to remove within-shipment clustering
ds <- dm %>% group_by(shipment_id, vendor) %>% summarise(late = max(late), mode = first(mode), pgroup = first(pgroup),
        inco = first(inco), year_g = first(year_g), log_value = mean(log_value),
        log_freight_ratio = first(log_freight_ratio), freight_missing = first(freight_missing), log_ship_lines = first(log_ship_lines), .groups = "drop") %>%
  arrange(shipment_id)
m_ship <- glmer(update(f_fix, late ~ . + (1 | vendor)), data = ds, family = binomial(), control = ctrl)

# ---------------- 3. Negative-binomial count model: days late among late lines ----------------
# Days late is a count that is only defined when a line is late (second stage of a hurdle model); we model
# (days_late - 1) >= 0 with a negative binomial.  This ignores zero-truncation beyond the shift: an approximation.
dl <- d %>% filter(late == 1) %>% mutate(days_m1 = pmin(days_late, 365) - 1)
f_nb <- days_m1 ~ mode + pgroup + inco + year_g + log_value
m_nb <- glm.nb(f_nb, data = dl, control = glm.control(maxit = 100))
irr <- tidy(m_nb, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ signif(.x, 5)))
write_csv(irr, file.path(out, "nb_rate_ratios.csv"))
m_pois <- glm(f_nb, data = dl, family = poisson())
disp_pois <- sum(residuals(m_pois, type = "pearson")^2) / df.residual(m_pois)

# ---------------- 4. Delivered-minus-scheduled distribution comparisons ----------------
tests <- list()
add_test <- function(name, x, y) {
  ks <- suppressWarnings(ks.test(x, y)); wx <- suppressWarnings(wilcox.test(x, y, conf.int = FALSE))
  tests[[length(tests) + 1]] <<- data.frame(comparison = name, n_a = length(x), n_b = length(y),
    median_a = median(x), median_b = median(y), share_late_a = mean(x > 0), share_late_b = mean(y > 0),
    ks_D = unname(ks$statistic), ks_p = ks$p.value, wilcox_W = unname(wx$statistic), wilcox_p = wx$p.value)
}
add_test("Air vs Truck (external vendors)", d$offset_days[d$mode == "Air"], d$offset_days[d$mode == "Truck"])
add_test("ARV vs HRDT / other (external vendors)", d$offset_days[d$pgroup == "ARV"], d$offset_days[d$pgroup == "HRDT / other"])
add_test("DDP vs EXW terms (external vendors)", d$offset_days[d$inco == "DDP"], d$offset_days[d$inco == "EXW"])
add_test("External vendors vs SCMS regional DC", d$offset_days, dall$offset_days[dall$is_internal_rdc])
add_test("Scheduled 2013-15 vs 2010-12 (external)", d$offset_days[d$year_g == "2013-15"], d$offset_days[d$year_g == "2010-12"])
tests <- bind_rows(tests); write_csv(tests, file.path(out, "offset_tests.csv"))
kw <- kruskal.test(offset_days ~ country_g, data = d)

theme_p <- theme_minimal(base_size = 11) + theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
pal <- c("#1f4e79", "#c0504d", "#5b9a3c", "#8064a2", "#e8892b", "#2e9bb5", "#7f8c99")
g1 <- ggplot(d %>% filter(offset_days >= -30, offset_days <= 60), aes(offset_days, colour = mode)) + stat_ecdf(linewidth = 0.8) +
  scale_colour_manual(values = pal) + geom_vline(xintercept = 0.5, linetype = 2, colour = "#c0504d") +
  labs(title = "ECDF of delivered minus scheduled days, by shipment mode (external vendors)",
       x = "days (> 0 = late; clipped to -30..60)", y = "cumulative share", colour = NULL) + theme_p
ggsave(file.path(figs, "r_ecdf_offset_mode.png"), g1, width = 8, height = 4.5, dpi = 140)
g2 <- ggplot(dall %>% filter(offset_days >= -30, offset_days <= 60) %>%
               mutate(channel = ifelse(is_internal_rdc, "SCMS regional DC", "External vendors")),
             aes(offset_days, colour = channel)) + stat_ecdf(linewidth = 0.8) + scale_colour_manual(values = pal) +
  geom_vline(xintercept = 0.5, linetype = 2, colour = "#c0504d") +
  labs(title = "ECDF of delivered minus scheduled days: vendors vs SCMS regional DC",
       x = "days (> 0 = late; clipped to -30..60)", y = "cumulative share", colour = NULL) + theme_p
ggsave(file.path(figs, "r_ecdf_offset_channel.png"), g2, width = 8, height = 4.5, dpi = 140)
ft <- or %>% filter(term != "(Intercept)")
g3 <- ggplot(ft, aes(estimate, reorder(term, estimate))) + geom_vline(xintercept = 1, linetype = 2, colour = "#c0504d") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, colour = "#7f8c99") +
  geom_point(colour = "#1f4e79", size = 2) + scale_x_log10() +
  labs(title = "Logistic GLM: odds ratios for late delivery (95% Wald CI)", x = "odds ratio (log scale)", y = NULL) + theme_p
ggsave(file.path(figs, "r_glm_odds_ratios.png"), g3, width = 8, height = 6.5, dpi = 140)
g4 <- ggplot(calib, aes(mean_pred, observed)) + geom_abline(linetype = 2, colour = "#7f8c99") +
  geom_point(colour = "#1f4e79", size = 2.5) + geom_line(colour = "#1f4e79") +
  labs(title = "GLM calibration by predicted-risk decile", x = "mean predicted P(late)", y = "observed late rate") + theme_p
ggsave(file.path(figs, "r_glm_calibration.png"), g4, width = 5.5, height = 4.5, dpi = 140)
g5 <- ggplot(vend_eff %>% mutate(lo = vendor_logit_effect - 1.96 * se, hi = vendor_logit_effect + 1.96 * se),
             aes(vendor_logit_effect, reorder(substr(vendor, 1, 30), vendor_logit_effect))) +
  geom_vline(xintercept = 0, linetype = 2, colour = "#c0504d") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.2, colour = "#7f8c99") + geom_point(colour = "#1f4e79", size = 1.8) +
  labs(title = "GLMM vendor random effects (logit scale, +/-1.96 conditional SD)", x = "vendor effect on log-odds of late", y = NULL) +
  theme_p + theme(axis.text.y = element_text(size = 6))
ggsave(file.path(figs, "r_glmm_vendor_effects.png"), g5, width = 8, height = 6, dpi = 140)

orv <- function(t) or$estimate[or$term == t]
summ <- list(n_lines = nrow(d), n_late = sum(d$late), late_rate = mean(d$late), n_vendors = nlevels(d$vendor),
             glm_auc = auc_glm, lrt_year_dev = lrt_year$Deviance[2], lrt_year_p = lrt_year$`Pr(>Chi)`[2],
             hosmer_lemeshow_stat = hl_stat, hosmer_lemeshow_p = hl_p, max_gvif_adj = max(vif_tab[, 3]),
             or_year_2013_15 = orv("year_g2013-15"), or_year_2006_08 = orv("year_g2006-08"),
             or_truck = orv("modeTruck"), or_hrdt = orv("pgroupHRDT / other"), or_ddp = orv("incoDDP"),
             or_internal_rdc_vs_vendor = or_all$estimate, or_internal_rdc_lo = or_all$conf.low, or_internal_rdc_hi = or_all$conf.high,
             glmm_n_lines = nrow(dm), glmm_n_vendors = nlevels(dm$vendor),
             glmm_vendor_sd_null = sqrt(as.data.frame(VarCorr(m_re0))$vcov[1]), glmm_icc_null = icc_fun(m_re0),
             glmm_vendor_sd = sqrt(as.data.frame(VarCorr(m_re))$vcov[1]), glmm_icc = icc_fun(m_re), glmm_mor = mor_fun(m_re),
             glmm_lr_vendor = lr_vendor, glmm_p_vendor_boundary = p_vendor,
             glmm_singular = isSingular(m_re), glmm_ship_n = nrow(ds), glmm_ship_icc = icc_fun(m_ship),
             glmm_ship_vendor_sd = sqrt(as.data.frame(VarCorr(m_ship))$vcov[1]),
             nb_n_late_lines = nrow(dl), nb_theta = m_nb$theta, poisson_dispersion = disp_pois,
             irr_truck = irr$estimate[irr$term == "modeTruck"], irr_hrdt = irr$estimate[irr$term == "pgroupHRDT / other"],
             irr_2013_15 = irr$estimate[irr$term == "year_g2013-15"],
             kruskal_country_H = unname(kw$statistic), kruskal_country_p = kw$p.value,
             r_version = R.version.string,
             pkg_versions = lapply(setNames(nm = c("MASS", "lme4", "ggplot2", "dplyr", "broom", "pROC", "car")),
                                   function(p) as.character(packageVersion(p))))
write_json(summ, file.path(out, "r_summary.json"), auto_unbox = TRUE, digits = 8, pretty = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
cat("R analysis done\n")
