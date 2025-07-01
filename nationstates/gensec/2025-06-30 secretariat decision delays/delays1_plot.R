library(tidyverse)
library(zoo)
library(arrow)

decs <- read_csv('secy_decisions.csv.xz')
MASS_APPOINTMENT <- ymd('2023-12-13')
IA_RESIGN <- ymd_hms('2025-05-24T18:56:00-0500')
BALLO_START <- ymd_hms('2025-05-27T04:25:00-0500')

# find the minimum delay between proposal and decision
proposal_mins <- decs %>%
  mutate(delay = DECISION_DATE - PROPOSAL_DATE) %>%
  group_by(PROPOSAL_ID) %>%
  slice_min(delay, with_ties = FALSE)

weekly_average <- proposal_mins %>%
  filter(CHAMBER == 'GA', PROPOSAL_DATE > MASS_APPOINTMENT) %>%
  mutate(delay = as.numeric(delay, units = 'hours')) %>%
  group_by(week = floor_date(PROPOSAL_DATE, unit = "week", week_start = 1)) %>%
  summarise(
    mean_delay = mean(delay, na.rm = TRUE),
    median_delay = median(delay, na.rm = TRUE)
  ) %>%
  mutate(
    rolling_mean = rollmean(
      mean_delay,
      k = 3,
      fill = NA,
      align = 'right'
    ),
    rolling_median = rollmean(
      median_delay,
      k = 3,
      fill = NA,
      align = 'right'
    ),
  )

delay_plot <- ggplot(weekly_average, aes(week)) +
  geom_line(aes(y = mean_delay), linetype = 'dotted') +
  # geom_line(aes(y = median_delay, color = 'Raw median'), linetype = 'dotted') +
  geom_line(aes(y = rolling_mean), ) +
  # geom_line(aes(y = rolling_median, color = 'Median')) +
  geom_vline(xintercept = IA_RESIGN, color = "maroon") +
  labs(
    color = 'Raw values',
    title = 'Average decision delay by week',
    y = 'Delay in hours',
    x = 'Week'
  ) +
  theme_minimal()

ggsave('secretariat_delays ggplot.jpg', delay_plot, dpi = 128)
write_parquet(proposal_mins, 'secy_decisions+delays.parquet', compression = 'zstd')
