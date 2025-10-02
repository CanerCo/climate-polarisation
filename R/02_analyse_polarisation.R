# Analyse polarisation from stance, sarcasm, toxicity
# Run with: source("R/02_analyse_polarisation.R")

suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(nnet)       # multinom
  library(broom)
  library(ggplot2)
  library(patchwork)
  # optional: library(DER)  # formal Esteban–Ray/DER index
})

# ---- Paths ----
in_csv <- "data/tweeteval_climate_with_scores.csv"
fig_dir <- "fig"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

stopifnot(file.exists(in_csv))
df <- readr::read_csv(in_csv, show_col_types = FALSE)

# ---- Map and factors ----
stance_levels <- c("none", "against", "favor")
df <- df %>%
  mutate(
    stance  = factor(label_name, levels = stance_levels),
    extreme = as.integer(is_extreme),
    sarcasm = sarcasm_prob,
    tox     = toxicity
  )

# ---- Q1: Is sarcasm more prevalent in against vs favour tweets? ----
q1 <- df %>%
  filter(stance %in% c("against", "favor")) %>%
  group_by(stance) %>%
  summarise(
    n = n(),
    mean_sarcasm = mean(sarcasm, na.rm = TRUE),
    sd = sd(sarcasm, na.rm = TRUE),
    .groups = "drop"
  )

# Effect size (Cohen's d)
cohens_d <- function(x, y) {
  nx <- length(x); ny <- length(y)
  sx <- sd(x, na.rm = TRUE); sy <- sd(y, na.rm = TRUE)
  sp <- sqrt(((nx - 1) * sx^2 + (ny - 1) * sy^2) / (nx + ny - 2))
  (mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE)) / sp
}

d_against_favor <- with(
  df %>% filter(stance %in% c("against", "favor")),
  cohens_d(sarcasm[stance == "against"], sarcasm[stance == "favor"])
)

# Plot group means
p_q1 <- df %>%
  filter(stance %in% c("against", "favor")) %>%
  ggplot(aes(stance, sarcasm)) +
  stat_summary(fun = mean, geom = "bar", width = 0.6) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2) +
  labs(
    title   = "Mean sarcasm probability by stance",
    y       = "P(sarcasm)", x = NULL,
    caption = sprintf("Cohen's d (against - favor) = %.2f", d_against_favor)
  )

ggplot2::ggsave(file.path(fig_dir, "q1_sarcasm_by_stance.png"),
                p_q1, width = 6, height = 4, dpi = 300)

# ---- Q2: Does toxicity predict stance extremity? ----
# Logistic regression: extreme (1) vs neutral (0)
logit <- glm(extreme ~ tox + sarcasm, data = df, family = binomial())
logit_tidy <- broom::tidy(logit, exponentiate = TRUE, conf.int = TRUE)

# Visualise marginal effect of toxicity
newd <- tibble(
  tox     = seq(quantile(df$tox, .01, na.rm = TRUE),
                quantile(df$tox, .99, na.rm = TRUE), length.out = 100),
  sarcasm = mean(df$sarcasm, na.rm = TRUE)
)
newd$pred_extreme <- predict(logit, newdata = newd, type = "response")

p_q2 <- ggplot(newd, aes(tox, pred_extreme)) +
  geom_line(size = 1) +
  labs(
    title    = "Predicted probability of extreme stance",
    subtitle = "Logit: extreme ~ toxicity + sarcasm (controls for sarcasm at mean)",
    x        = "Toxicity score", y = "P(extreme)"
  )

ggplot2::ggsave(file.path(fig_dir, "q2_toxicity_effect.png"),
                p_q2, width = 6, height = 4, dpi = 300)

# ---- Multinomial logistic: stance ~ toxicity + sarcasm ----
mult <- nnet::multinom(stance ~ tox + sarcasm, data = df, trace = FALSE)
mult_tidy <- broom::tidy(mult, exponentiate = TRUE, conf.int = TRUE)
readr::write_csv(mult_tidy, file.path(fig_dir, "multinomial_or.csv"))

# ---- Q3: How polarised is the stance distribution? ----
# ER-style polarisation proxy for discrete stances
er_proxy <- function(p, y = c(-1, 0, 1), alpha = 1) {
  p <- as.numeric(p)
  stopifnot(length(p) == length(y))
  stopifnot(abs(sum(p) - 1) < 1e-8)
  M <- abs(outer(y, y, "-"))              # 3x3 distances
  P <- tcrossprod(p^(1 + alpha), p)       # 3x3 outer product: p_i^(1+a) * p_j
  raw <- sum(P * M)
  pmax <- c(0.5, 0.0, 0.5)
  Pmax <- tcrossprod(pmax^(1 + alpha), pmax)
  raw_max <- sum(Pmax * M)
  as.numeric(raw / raw_max)
}

# Proportions (robust to missing classes)
p_hat <- prop.table(table(df$stance))
p_hat <- p_hat[stance_levels]
p_hat[is.na(p_hat)] <- 0
p_hat <- as.numeric(p_hat)
p_hat <- p_hat / sum(p_hat)  # ensure sums to 1

p_er  <- er_proxy(p_hat, alpha = 1)

# Bootstrap CI for ER proxy (handle missing classes per draw)
set.seed(42)
B <- 1000
boot_vals <- replicate(B, {
  idx <- sample.int(nrow(df), replace = TRUE)
  pb <- prop.table(table(df$stance[idx]))
  pb <- pb[stance_levels]
  pb[is.na(pb)] <- 0
  pb <- as.numeric(pb)
  pb <- pb / sum(pb)
  er_proxy(pb, alpha = 1)
})
ci <- quantile(boot_vals, c(.025, .975), na.rm = TRUE)

# Save summary table
summary_tbl <- tibble(
  metric = c("ER_proxy", "ER_proxy_CI_low", "ER_proxy_CI_high"),
  value  = c(p_er, ci[[1]], ci[[2]])
)
readr::write_csv(summary_tbl, file.path(fig_dir, "polarisation_summary.csv"))

# Plot stance distribution
p_q3 <- df %>%
  count(stance) %>%
  mutate(p = n / sum(n)) %>%
  ggplot(aes(stance, p)) +
  geom_col() +
  labs(
    title = sprintf("Stance distribution (ER proxy = %.2f)", p_er),
    x = NULL, y = "Share"
  )

ggplot2::ggsave(file.path(fig_dir, "q3_stance_distribution.png"),
                p_q3, width = 6, height = 4, dpi = 300)

# ---- Save model summaries ----
readr::write_csv(q1, file.path(fig_dir, "q1_sarcasm_group_means.csv"))
readr::write_csv(logit_tidy, file.path(fig_dir, "q2_logit_odds_ratios.csv"))

cat("\nAnalysis complete. Files written to fig/ and CSV outputs saved.\n")
