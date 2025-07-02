library(tidyverse)
library(arrow)
library(sandwich)
library(lmtest)
library(scales)
library(broom)

# io ----------------------------------------------------------------------

MASS_APPOINTMENT <- ymd('2023-12-13')
IA_RESIGN <- ymd_hms('2025-05-24T18:56:00-0500')
BALLO_START <- ymd_hms('2025-05-27T04:25:00-0500')

# dates are in UTC/GMT
min_delays <- read_parquet('secy_decisions+delays.parquet') %>%
  mutate(
    delay = as.numeric(delay, units = 'hours'),
    hour = hour(PROPOSAL_DATE),
    post_resign = DECISION_DATE > IA_RESIGN,
    transition = IA_RESIGN <= DECISION_DATE &
      DECISION_DATE <= BALLO_START,
    DECISION=tolower(DECISION)
  )

# plot a histogram of the delay values
delay_hist <- ggplot(min_delays, aes(x = delay, fill = CHAMBER)) +
  geom_histogram(position = 'identity', alpha = 0.7) +
  scale_x_log10(breaks = c(1, 10, 25, 50, 100), labels = label_comma()) +
  theme_minimal() +
  labs(title = 'log minimum delay', y = 'Count', x = 'Minimum delay (hours)')
ggsave('secretariat_delays_hist ggplot.jpg', delay_hist, dpi = 128)
delay_hist

# regressions -------------------------------------------------------------

library(plm)
est <- plm(
  log(delay) ~ CHAMBER + CHAMBER:post_resign + CHAMBER:transition + 1,
  data = min_delays,
  index = c('hour'),
  model = 'within'
)
est_wi <- within_intercept(est, return.model = TRUE)
est_sm <- summary(
  est_wi,
  vcov = function(x)
    vcovHC(x, type = 'HC3', cluster = 'time')
)
results_summary <- capture.output(est_sm)

# summarise and convert to hours ------------------------------------------

est_coefs <- as_tibble(est_sm$coefficients, rownames='term')
colnames(est_coefs) <- c('term', 'estimate', 'std.error', 't.value', 'p.value')
est_coefs$avg_effect <- vector(mode='numeric', length=nrow(est_coefs))
est_coefs$avg_effect[2:nrow(est_coefs)] <-
  exp(est_coefs$estimate[1] + est_coefs$estimate[2:nrow(est_coefs)]) -
  exp(est_coefs$estimate[1])
est_coefs$avg_effect[1] <- exp(est_coefs$estimate[1])
est_coefs <- est_coefs %>%
  mutate(
    stars = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      p.value < 0.1   ~ ".",
      TRUE            ~ ""
    )
  )
write_csv(est_coefs, 'secy_delays__plm_results.csv')

# fixed effect plot -------------------------------------------------------

fe_estimates <- fixef(est) - est_wi$coefficients['(Intercept)']
fe_estimates <- tibble(hours_utc = as.integer(names(fe_estimates)), fe = as.numeric(fe_estimates)) %>%
  mutate(hours_est = (hours_utc - 5) %% 24)
fe_plot <- ggplot(fe_estimates, aes(x = reorder(hours_est, fe), y = fe)) +
  geom_col() +
  coord_flip() +
  labs(title = "Hourly fixed effects", x = 'Hours (EST)', y = "Percentage effect (eg 25% longer)") +
  theme_minimal()
ggsave('secretariat_delays fe_hours ggplot.jpg', fe_plot, dpi = 128)
fe_plot
