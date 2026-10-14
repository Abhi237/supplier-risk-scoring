# Delivery-risk statistical layer (R): logistic GLM for lateness, seller random-intercept GLMM,
# negative-binomial model for lead time, KS / Wilcoxon lead-time comparisons and ECDFs, diagnostics.
# Input : data/processed/orders_delivered_for_r.csv  (order grain, delivered orders, no customer IDs)
# Output: artifacts/r_glm/*.csv, docs/figures/r_*.png
# Olist SELLERS are used as PROXIES for suppliers.
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(ggplot2); library(MASS); library(lme4); library(broom)
  library(pROC); library(tidyr); library(car); library(jsonlite)
})
set.seed(42)
root <- Sys.getenv("SUPPLIER_RISK_ROOT", unset = normalizePath("."))
out <- file.path(root, "artifacts", "r_glm"); dir.create(out, showWarnings = FALSE, recursive = TRUE)
figs <- file.path(root, "docs", "figures")

region_of <- function(uf) dplyr::case_when(
  uf %in% c("SP", "RJ", "MG", "ES") ~ "Southeast",
  uf %in% c("PR", "SC", "RS") ~ "South",
  uf %in% c("DF", "GO", "MT", "MS") ~ "Center-West",
  uf %in% c("BA", "SE", "AL", "PE", "PB", "RN", "CE", "PI", "MA") ~ "Northeast",
  uf %in% c("AM", "RR", "AP", "PA", "TO", "RO", "AC") ~ "North",
  TRUE ~ "Unknown")

raw <- read_csv(file.path(root, "data/processed/orders_delivered_for_r.csv"), show_col_types = FALSE)
d <- raw %>%
  filter(!is.na(is_late), !is.na(delivery_days), delivery_days >= 0, !is.na(seller_state),
         !is.na(customer_state), order_gmv > 0) %>%
  mutate(late = as.integer(is_late),
         handoff_late = as.integer(coalesce(is_handoff_late, FALSE)),
         cust_region = relevel(factor(region_of(customer_state)), "Southeast"),
         seller_region = relevel(factor(region_of(seller_state)), "Southeast"),
         same_state = as.integer(same_state),
         log_gmv = log(order_gmv), log_freight_ratio = log1p(pmin(freight_to_price, 3)),
         month = factor(format(as.Date(purchase_month), "%m")),
         year = factor(format(as.Date(purchase_month), "%Y")),
         lead_days = round(delivery_days))
top_cats <- d %>% count(category, sort = TRUE) %>% filter(n >= 2000) %>% pull(category)
d <- d %>% mutate(cat = relevel(factor(ifelse(category %in% top_cats, category, "other")), "other"))
cat(nrow(d), "delivered orders;", mean(d$late), "late\n")

# ---------------- 1. Logistic GLM ----------------
f_glm <- late ~ handoff_late + cust_region + seller_region + same_state + log_freight_ratio + log_gmv + cat +
  month + year
m_glm <- glm(f_glm, data = d, family = binomial())
or <- tidy(m_glm, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ signif(.x, 5)))
write_csv(or, file.path(out, "glm_odds_ratios.csv"))
d$p_glm <- fitted(m_glm)
auc_glm <- as.numeric(auc(roc(d$late, d$p_glm, quiet = TRUE)))
# without the seller hand-off term: how much does the seller-controlled leg add?
m_glm0 <- update(m_glm, . ~ . - handoff_late)
auc_glm0 <- as.numeric(auc(roc(d$late, fitted(m_glm0), quiet = TRUE)))
lrt <- anova(m_glm0, m_glm, test = "LRT")
vif_tab <- as.data.frame(car::vif(m_glm)); vif_tab$term <- rownames(vif_tab)
write_csv(vif_tab, file.path(out, "glm_vif.csv"))
calib <- d %>% mutate(bin = ntile(p_glm, 10)) %>% group_by(bin) %>%
  summarise(mean_pred = mean(p_glm), observed = mean(late), n = n(), .groups = "drop")
write_csv(calib, file.path(out, "glm_calibration_deciles.csv"))
hl_stat <- sum((calib$observed - calib$mean_pred)^2 * calib$n / (calib$mean_pred * (1 - calib$mean_pred)))
hl_p <- pchisq(hl_stat, df = 8, lower.tail = FALSE)

# ---------------- 2. Seller random-intercept GLMM (variance attributable to sellers) ----------------
sel_n <- d %>% count(seller_id) %>% filter(n >= 5)
dm <- d %>% filter(seller_id %in% sel_n$seller_id)
m_mm <- glmer(late ~ handoff_late + cust_region + seller_region + same_state + log_freight_ratio + log_gmv +
                month + year + (1 | seller_id), data = dm, family = binomial(),
              control = glmerControl(optimizer = "bobyqa"), nAGQ = 0)
sd_seller <- as.data.frame(VarCorr(m_mm))$sdcor[1]
icc <- sd_seller^2 / (sd_seller^2 + pi^2 / 3)
m_mm0 <- glmer(late ~ cust_region + seller_region + same_state + log_freight_ratio + log_gmv + month + year +
                 (1 | seller_id), data = dm, family = binomial(), control = glmerControl(optimizer = "bobyqa"),
               nAGQ = 0)
sd_seller0 <- as.data.frame(VarCorr(m_mm0))$sdcor[1]
icc0 <- sd_seller0^2 / (sd_seller0^2 + pi^2 / 3)
re <- ranef(m_mm)$seller_id; re$seller_id <- rownames(re); names(re)[1] <- "seller_logit_effect"
write_csv(re, file.path(out, "glmm_seller_effects.csv"))

# ---------------- 3. Negative binomial for lead time (days) ----------------
m_nb <- glm.nb(lead_days ~ handoff_late + cust_region + seller_region + same_state + log_freight_ratio + log_gmv +
                 cat + month + year, data = d)
irr <- tidy(m_nb, conf.int = TRUE, conf.method = "Wald", exponentiate = TRUE) %>%
  mutate(across(where(is.numeric), ~ signif(.x, 5)))
write_csv(irr, file.path(out, "nb_rate_ratios.csv"))
m_pois <- glm(formula(m_nb), data = d, family = poisson())
disp_pois <- sum(residuals(m_pois, type = "pearson")^2) / df.residual(m_pois)

# ---------------- 4. Lead-time distribution comparisons ----------------
tests <- list()
add_test <- function(name, x, y) {
  ks <- suppressWarnings(ks.test(x, y)); wx <- wilcox.test(x, y, conf.int = FALSE)
  tests[[length(tests) + 1]] <<- data.frame(comparison = name, n_a = length(x), n_b = length(y),
    median_a = median(x), median_b = median(y), ks_D = unname(ks$statistic), ks_p = ks$p.value,
    wilcox_W = unname(wx$statistic), wilcox_p = wx$p.value)
}
add_test("same-state vs cross-state lane", d$delivery_days[d$same_state == 1], d$delivery_days[d$same_state == 0])
add_test("hand-off late vs on time", d$delivery_days[d$handoff_late == 1], d$delivery_days[d$handoff_late == 0])
for (r in setdiff(levels(d$cust_region), "Southeast"))
  add_test(paste0("customer ", r, " vs Southeast"), d$delivery_days[d$cust_region == r],
           d$delivery_days[d$cust_region == "Southeast"])
tests <- bind_rows(tests); write_csv(tests, file.path(out, "leadtime_tests.csv"))
kw <- kruskal.test(delivery_days ~ cat, data = d)

theme_p <- theme_minimal(base_size = 11) + theme(plot.title = element_text(face = "bold"), legend.position = "bottom")
pal <- c("#1f4e79", "#c0504d", "#5b9a3c", "#8064a2", "#e8892b", "#2e9bb5", "#7f8c99")
g1 <- ggplot(d %>% filter(delivery_days <= 60), aes(delivery_days, colour = cust_region)) + stat_ecdf(linewidth = 0.8) +
  scale_colour_manual(values = pal) + labs(title = "Lead time ECDF by customer region",
  x = "purchase -> customer delivery (days)", y = "cumulative share", colour = NULL) + theme_p
ggsave(file.path(figs, "r_ecdf_leadtime_region.png"), g1, width = 8, height = 4.5, dpi = 140)
g2 <- ggplot(d %>% filter(delivery_days <= 60, cat != "other"), aes(delivery_days, colour = cat)) +
  stat_ecdf(linewidth = 0.7) + labs(title = "Lead time ECDF by top product category",
  x = "purchase -> customer delivery (days)", y = "cumulative share", colour = NULL) + theme_p +
  guides(colour = guide_legend(ncol = 3))
ggsave(file.path(figs, "r_ecdf_leadtime_category.png"), g2, width = 8, height = 5, dpi = 140)
ft <- or %>% filter(term != "(Intercept)", !grepl("^month|^year", term))
g3 <- ggplot(ft, aes(estimate, reorder(term, estimate))) + geom_vline(xintercept = 1, linetype = 2, colour = "#c0504d") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, colour = "#7f8c99") +
  geom_point(colour = "#1f4e79", size = 2) + scale_x_log10() +
  labs(title = "Logistic GLM: odds ratios for late delivery (95% CI)", x = "odds ratio (log scale)", y = NULL) + theme_p
ggsave(file.path(figs, "r_glm_odds_ratios.png"), g3, width = 8, height = 6, dpi = 140)
g4 <- ggplot(calib, aes(mean_pred, observed)) + geom_abline(linetype = 2, colour = "#7f8c99") +
  geom_point(colour = "#1f4e79", size = 2.5) + geom_line(colour = "#1f4e79") +
  labs(title = "GLM calibration by predicted-risk decile", x = "mean predicted P(late)", y = "observed late rate") + theme_p
ggsave(file.path(figs, "r_glm_calibration.png"), g4, width = 5.5, height = 4.5, dpi = 140)

summ <- list(n_orders = nrow(d), late_rate = mean(d$late), glm_auc = auc_glm, glm_auc_without_handoff = auc_glm0,
             lrt_handoff_dev = lrt$Deviance[2], lrt_handoff_p = lrt$`Pr(>Chi)`[2],
             or_handoff = or$estimate[or$term == "handoff_late"],
             or_handoff_lo = or$conf.low[or$term == "handoff_late"], or_handoff_hi = or$conf.high[or$term == "handoff_late"],
             or_same_state = or$estimate[or$term == "same_state"],
             hosmer_lemeshow_stat = hl_stat, hosmer_lemeshow_p = hl_p, max_gvif_adj = max(vif_tab[, 3]),
             glmm_n_orders = nrow(dm), glmm_n_sellers = nrow(sel_n), glmm_seller_sd = sd_seller, glmm_icc = icc,
             glmm_seller_sd_without_handoff = sd_seller0, glmm_icc_without_handoff = icc0,
             nb_theta = m_nb$theta, poisson_dispersion = disp_pois,
             irr_handoff = irr$estimate[irr$term == "handoff_late"], irr_same_state = irr$estimate[irr$term == "same_state"],
             kruskal_category_H = unname(kw$statistic), kruskal_category_p = kw$p.value,
             r_version = R.version.string,
             pkg_versions = lapply(setNames(nm = c("MASS", "lme4", "ggplot2", "dplyr", "broom", "pROC", "car")),
                                   function(p) as.character(packageVersion(p))))
write_json(summ, file.path(out, "r_summary.json"), auto_unbox = TRUE, digits = 8, pretty = TRUE)
cat("R analysis done\n")
