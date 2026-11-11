# Matched-pairs analysis in R
# Main window: 1970-2021 (the longest observed overlap of the two WDI indicators)
# Run this script from the root folder of the resource-rents-gdp-growth repository.

required_packages <- c("readr", "dplyr", "tidyr", "stringr", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Install these packages first: ",
    paste(missing_packages, collapse = ", "),
    "\nRun: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
})

START_YEAR <- 1970L
END_YEAR <- 2021L
RENT_SERIES <- "NY.GDP.TOTL.RT.ZS"
GROWTH_SERIES <- "NY.GDP.PCAP.KD.ZG"

raw_file <- file.path("data", "raw", "wdi_growth_rents_1960_2025.csv")
whitelist_file <- file.path("data", "whitelist", "country_list_to_review.csv")
results_dir <- file.path("results")
figures_dir <- file.path(results_dir, "figures")

if (!file.exists(raw_file) || !file.exists(whitelist_file)) {
  stop(
    "Set the R working directory to the repository root. Expected to find:\n",
    raw_file, "\n", whitelist_file
  )
}

dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Read all fields as text first so WDI codes and year labels stay unchanged.
wdi <- read_csv(
  raw_file,
  na = c("..", ""),
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
country_list <- read_csv(
  whitelist_file,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)

country_codes <- country_list %>%
  filter(!is.na(`Country Code`)) %>%
  pull(`Country Code`) %>%
  trimws() %>%
  unique()

year_columns <- names(wdi)[
  str_detect(names(wdi), "^\\d{4} \\[YR\\d{4}\\]$")
]
year_numbers <- as.integer(str_extract(year_columns, "^\\d{4}"))
year_columns <- year_columns[
  year_numbers >= START_YEAR & year_numbers <= END_YEAR
]
if (length(year_columns) == 0) {
  stop("No WDI year columns found for 1970-2021.")
}

selected <- wdi %>%
  filter(
    trimws(`Country Code`) %in% country_codes,
    `Series Code` %in% c(RENT_SERIES, GROWTH_SERIES)
  ) %>%
  select(`Country Name`, `Country Code`, `Series Code`, all_of(year_columns))

duplicates <- selected %>%
  count(`Country Code`, `Series Code`) %>%
  filter(n > 1)
if (nrow(duplicates) > 0) {
  stop("Duplicate Country Code × Series Code rows found. Check the source CSV.")
}

long_data <- selected %>%
  pivot_longer(
    cols = all_of(year_columns),
    names_to = "year_label",
    values_to = "value"
  ) %>%
  mutate(
    Year = as.integer(str_extract(year_label, "^\\d{4}")),
    value = as.numeric(value)
  ) %>%
  filter(Year >= START_YEAR, Year <= END_YEAR)

panel <- long_data %>%
  select(`Country Name`, `Country Code`, Year, `Series Code`, value) %>%
  pivot_wider(
    names_from = `Series Code`,
    values_from = value
  ) %>%
  rename(
    resource_rents_pct_gdp = `NY.GDP.TOTL.RT.ZS`,
    gdp_pc_growth_pct = `NY.GDP.PCAP.KD.ZG`
  ) %>%
  filter(
    is.finite(resource_rents_pct_gdp),
    is.finite(gdp_pc_growth_pct)
  ) %>%
  arrange(`Country Code`, Year)

if (anyDuplicated(panel[c("Country Code", "Year")]) > 0) {
  stop("Duplicate Country Code × Year rows found after combining indicators.")
}

# Create one low-rent and high-rent mean for each country/economy.
pair_rows <- list()
excluded_fewer_than_6_years <- 0L
excluded_equal_cutoffs <- 0L
excluded_too_few_in_group <- 0L

for (code in unique(panel$`Country Code`)) {
  g <- panel %>%
    filter(`Country Code` == code) %>%
    arrange(Year)

  if (nrow(g) < 6L) {
    excluded_fewer_than_6_years <- excluded_fewer_than_6_years + 1L
    next
  }

  cutoffs <- as.numeric(quantile(
    g$resource_rents_pct_gdp,
    probs = c(1 / 3, 2 / 3),
    type = 7,       # linear interpolation; matches pandas' default linear quantile
    names = FALSE
  ))
  low_cutoff <- cutoffs[1]
  high_cutoff <- cutoffs[2]

  if (low_cutoff >= high_cutoff) {
    excluded_equal_cutoffs <- excluded_equal_cutoffs + 1L
    next
  }

  low_group <- g %>% filter(resource_rents_pct_gdp <= low_cutoff)
  high_group <- g %>% filter(resource_rents_pct_gdp >= high_cutoff)

  if (nrow(low_group) < 3L || nrow(high_group) < 3L) {
    excluded_too_few_in_group <- excluded_too_few_in_group + 1L
    next
  }

  pair_rows[[length(pair_rows) + 1L]] <- tibble(
    analysis_window = "1970-2021 main",
    `Country Code` = code,
    `Country Name` = g$`Country Name`[[1]],
    observed_first_joint_year = min(g$Year),
    observed_last_joint_year = max(g$Year),
    complete_country_years = nrow(g),
    low_rent_cutoff_pct_gdp = low_cutoff,
    high_rent_cutoff_pct_gdp = high_cutoff,
    low_rent_years = nrow(low_group),
    high_rent_years = nrow(high_group),
    mean_growth_low_rents_pct = mean(low_group$gdp_pc_growth_pct),
    mean_growth_high_rents_pct = mean(high_group$gdp_pc_growth_pct)
  )
}

country_pairs <- bind_rows(pair_rows) %>%
  mutate(
    difference_high_minus_low_pp =
      mean_growth_high_rents_pct - mean_growth_low_rents_pct
  ) %>%
  arrange(`Country Code`)

if (nrow(country_pairs) == 0) {
  stop("No eligible country/economy pairs were created.")
}

differences <- country_pairs$difference_high_minus_low_pp
n_pairs <- length(differences)

# Paired t-test is equivalent to a one-sample t-test on the paired differences.
paired_t <- t.test(
  country_pairs$mean_growth_high_rents_pct,
  country_pairs$mean_growth_low_rents_pct,
  paired = TRUE,
  alternative = "two.sided"
)

# Wilcoxon signed-rank. R calls its positive-rank statistic V; we also calculate
# the two-sided W = min(W+, W-) for comparison with common software output.
wilcoxon <- wilcox.test(
  country_pairs$mean_growth_high_rents_pct,
  country_pairs$mean_growth_low_rents_pct,
  paired = TRUE,
  alternative = "two.sided",
  exact = FALSE,
  correct = FALSE
)

nonzero_differences <- differences[differences != 0]
difference_ranks <- rank(abs(nonzero_differences), ties.method = "average")
rank_sum_positive <- sum(difference_ranks[nonzero_differences > 0])
rank_sum_negative <- sum(difference_ranks[nonzero_differences < 0])
wilcoxon_w_two_sided <- min(rank_sum_positive, rank_sum_negative)

mean_difference <- mean(differences)
sd_difference <- sd(differences)
se_difference <- sd_difference / sqrt(n_pairs)
positive_pairs <- sum(differences > 0)
negative_pairs <- sum(differences < 0)
tied_pairs <- sum(differences == 0)

summary_row <- tibble(
  analysis_window = "1970-2021 main",
  complete_country_years = nrow(panel),
  whitelisted_country_economies = length(country_codes),
  units_with_joint_data = n_distinct(panel$`Country Code`),
  units_without_joint_data =
    length(setdiff(country_codes, unique(panel$`Country Code`))),
  country_pairs = n_pairs,
  excluded_fewer_than_6_joint_years = excluded_fewer_than_6_years,
  excluded_identical_percentile_cutoffs = excluded_equal_cutoffs,
  excluded_too_few_years_in_low_or_high_group = excluded_too_few_in_group,
  mean_growth_low_rents_pct = mean(country_pairs$mean_growth_low_rents_pct),
  mean_growth_high_rents_pct = mean(country_pairs$mean_growth_high_rents_pct),
  mean_difference_high_minus_low_pp = mean_difference,
  sd_paired_difference_pp = sd_difference,
  se_paired_difference_pp = se_difference,
  ci_95_low_pp = paired_t$conf.int[1],
  ci_95_high_pp = paired_t$conf.int[2],
  paired_t_statistic = unname(paired_t$statistic),
  degrees_of_freedom = unname(paired_t$parameter),
  paired_t_p_value_two_sided = paired_t$p.value,
  wilcoxon_statistic_two_sided = wilcoxon_w_two_sided,
  wilcoxon_V_as_reported_by_R = unname(wilcoxon$statistic),
  wilcoxon_p_value_two_sided = wilcoxon$p.value,
  positive_country_differences = positive_pairs,
  negative_country_differences = negative_pairs,
  tied_country_differences = tied_pairs
)

write_csv(summary_row, file.path(results_dir, "r_analysis_summary.csv"))
write_csv(country_pairs, file.path(results_dir, "r_country_pairs.csv"))

plot_data <- country_pairs %>%
  arrange(difference_high_minus_low_pp) %>%
  mutate(pair_order = row_number())

paired_plot <- ggplot(
  plot_data,
  aes(x = pair_order, y = difference_high_minus_low_pp)
) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#374151") +
  geom_hline(yintercept = mean_difference, color = "#d97706", linewidth = 0.9) +
  geom_point(color = "#2f80b7", alpha = 0.68, size = 2) +
  labs(
    title = "GDP per capita growth difference by economy, 1970-2021",
    subtitle = paste0(
      "High-rent mean minus low-rent mean; one paired difference per economy (n = ",
      n_pairs, ")"
    ),
    x = "Countries/economies, ordered by paired difference",
    y = "Difference in mean annual growth (percentage points)"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(figures_dir, "r_paired_differences.png"),
  plot = paired_plot,
  width = 10,
  height = 6,
  dpi = 180
)

cat("\nMatched-pairs main analysis: 1970-2021\n")
cat("Complete country-years:", format(nrow(panel), big.mark = ","), "\n")
cat("Units with both indicators:", n_distinct(panel$`Country Code`), "\n")
cat("Paired units analyzed:", n_pairs, "\n")
cat("Mean GDP per capita growth, low rents:",
    sprintf("%.4f%%", summary_row$mean_growth_low_rents_pct), "\n")
cat("Mean GDP per capita growth, high rents:",
    sprintf("%.4f%%", summary_row$mean_growth_high_rents_pct), "\n")
cat("Mean paired difference (high - low):",
    sprintf("%.4f percentage points", mean_difference), "\n")
cat("95% confidence interval:",
    sprintf("[%.4f, %.4f]", paired_t$conf.int[1], paired_t$conf.int[2]), "\n")
cat("Paired t-test:",
    sprintf("t(%d) = %.4f, p = %.6g",
            as.integer(paired_t$parameter), paired_t$statistic, paired_t$p.value), "\n")
cat("Wilcoxon signed-rank:",
    sprintf("two-sided W = %.0f; R's V = %.0f; p = %.6g",
            wilcoxon_w_two_sided, wilcoxon$statistic, wilcoxon$p.value), "\n")
cat("Positive / negative / tied pairs:",
    positive_pairs, "/", negative_pairs, "/", tied_pairs, "\n")
cat("\nSaved tables in results/ and graph in results/figures/r_paired_differences.png\n")
