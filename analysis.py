"""Run the main matched-pairs analysis over the full observed joint-data window.

Both selected WDI indicators overlap in 1970-2021. This script uses that full
window to maximize available country-year data for the within-economy means.
The inferential sample remains one paired difference per eligible economy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "wdi_growth_rents_1960_2025.csv"
COUNTRY_FILE = ROOT / "data" / "whitelist" / "country_list_to_review.csv"
OUTPUT_DIR = ROOT / "results"

COUNTRY = "Country Code"
COUNTRY_NAME = "Country Name"
SERIES = "Series Code"
YEAR = "Year"
RENTS_CODE = "NY.GDP.TOTL.RT.ZS"
GROWTH_CODE = "NY.GDP.PCAP.KD.ZG"
MIN_YEARS_PER_GROUP = 3
LOW_QUANTILE = 1 / 3
HIGH_QUANTILE = 2 / 3
ALPHA = 0.05


def load_panel() -> tuple[pd.DataFrame, int]:
    """Read the two selected WDI series and the project's economy list."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล WDI: {RAW_FILE}")
    if not COUNTRY_FILE.exists():
        raise FileNotFoundError(f"ไม่พบรายชื่อประเทศ/เขตเศรษฐกิจ: {COUNTRY_FILE}")

    countries = pd.read_csv(COUNTRY_FILE, dtype={COUNTRY: "string"})
    country_codes = set(countries[COUNTRY].dropna().astype(str).str.strip())
    raw = pd.read_csv(
        RAW_FILE,
        dtype={COUNTRY: "string"},
        na_values=["..", ""],
        encoding="utf-8-sig",
    )

    required_columns = {COUNTRY, COUNTRY_NAME, SERIES}
    missing = required_columns.difference(raw.columns)
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็นในไฟล์ WDI: {sorted(missing)}")

    year_columns = [
        column
        for column in raw.columns
        if re.fullmatch(r"\d{4} \[YR\d{4}\]", str(column))
    ]
    if not year_columns:
        raise ValueError("หา column ปีในรูปแบบ YYYY [YRYYYY] ไม่พบ")

    selected = raw.loc[
        raw[COUNTRY].astype("string").str.strip().isin(country_codes)
        & raw[SERIES].isin([RENTS_CODE, GROWTH_CODE]),
        [COUNTRY, COUNTRY_NAME, SERIES, *year_columns],
    ].copy()
    if selected.empty:
        raise ValueError("ไม่พบสอง Series Code ที่ใช้ในการวิเคราะห์")

    duplicate_series = selected.duplicated([COUNTRY, SERIES], keep=False)
    if duplicate_series.any():
        details = selected.loc[duplicate_series, [COUNTRY, COUNTRY_NAME, SERIES]]
        raise ValueError(
            "พบ Series ซ้ำภายใน Country Code; หยุดเพื่อป้องกันการนับซ้ำ:\n"
            + details.head(10).to_string(index=False)
        )

    long = selected.melt(
        id_vars=[COUNTRY, COUNTRY_NAME, SERIES],
        value_vars=year_columns,
        var_name="year_label",
        value_name="value",
    )
    long[YEAR] = long["year_label"].str.extract(r"^(\d{4})")[0].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if long.duplicated([COUNTRY, COUNTRY_NAME, YEAR, SERIES]).any():
        raise ValueError("พบ Country-Year-Series ซ้ำหลังแปลงข้อมูล")

    panel = long.pivot(
        index=[COUNTRY, COUNTRY_NAME, YEAR], columns=SERIES, values="value"
    ).reset_index()
    panel.columns.name = None
    panel = panel.rename(
        columns={
            RENTS_CODE: "resource_rents_pct_gdp",
            GROWTH_CODE: "gdp_pc_growth_pct",
        }
    )
    panel = panel.dropna(
        subset=["resource_rents_pct_gdp", "gdp_pc_growth_pct"]
    )
    panel = panel.loc[
        np.isfinite(panel["resource_rents_pct_gdp"])
        & np.isfinite(panel["gdp_pc_growth_pct"])
    ].copy()
    if panel.duplicated([COUNTRY, YEAR]).any():
        raise ValueError("พบ Country-Year ซ้ำใน panel ที่รวมแล้ว")

    return panel, len(country_codes)


def analyze_window(
    panel: pd.DataFrame,
    label: str,
    start_year: int,
    end_year: int,
    whitelist_size: int,
) -> tuple[dict, pd.DataFrame]:
    window = panel.loc[panel[YEAR].between(start_year, end_year)].copy()
    pairs = []
    exclusions = {
        "fewer_than_2x_minimum_years": 0,
        "identical_low_high_cutoffs": 0,
        "too_few_years_in_either_group": 0,
    }

    for (code, name), group in window.groupby([COUNTRY, COUNTRY_NAME], sort=True):
        group = group.sort_values(YEAR)
        if len(group) < 2 * MIN_YEARS_PER_GROUP:
            exclusions["fewer_than_2x_minimum_years"] += 1
            continue

        low_cutoff = group["resource_rents_pct_gdp"].quantile(
            LOW_QUANTILE, interpolation="linear"
        )
        high_cutoff = group["resource_rents_pct_gdp"].quantile(
            HIGH_QUANTILE, interpolation="linear"
        )
        if not low_cutoff < high_cutoff:
            exclusions["identical_low_high_cutoffs"] += 1
            continue

        low = group.loc[group["resource_rents_pct_gdp"] <= low_cutoff]
        high = group.loc[group["resource_rents_pct_gdp"] >= high_cutoff]
        if len(low) < MIN_YEARS_PER_GROUP or len(high) < MIN_YEARS_PER_GROUP:
            exclusions["too_few_years_in_either_group"] += 1
            continue

        low_mean = float(low["gdp_pc_growth_pct"].mean())
        high_mean = float(high["gdp_pc_growth_pct"].mean())
        pairs.append(
            {
                "analysis_window": label,
                COUNTRY: code,
                COUNTRY_NAME: name,
                "requested_start_year": start_year,
                "requested_end_year": end_year,
                "observed_first_joint_year": int(group[YEAR].min()),
                "observed_last_joint_year": int(group[YEAR].max()),
                "complete_country_years": len(group),
                "low_rent_cutoff_pct_gdp": float(low_cutoff),
                "high_rent_cutoff_pct_gdp": float(high_cutoff),
                "low_rent_years": len(low),
                "high_rent_years": len(high),
                "mean_growth_low_rents_pct": low_mean,
                "mean_growth_high_rents_pct": high_mean,
                "difference_high_minus_low_pp": high_mean - low_mean,
            }
        )

    paired = pd.DataFrame(pairs)
    if len(paired) < 2:
        raise ValueError(f"มีคู่ประเทศไม่พอสำหรับการทดสอบในช่วง {label}")

    differences = paired["difference_high_minus_low_pp"].to_numpy(dtype=float)
    n_pairs = len(differences)
    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    se_diff = sd_diff / np.sqrt(n_pairs)
    df = n_pairs - 1
    critical_t = float(stats.t.ppf(1 - ALPHA / 2, df))
    ci_low = mean_diff - critical_t * se_diff
    ci_high = mean_diff + critical_t * se_diff
    t_result = stats.ttest_1samp(
        differences, popmean=0.0, alternative="two-sided"
    )
    if np.any(~np.isclose(differences, 0.0)):
        wilcoxon_result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    summary = {
        "analysis_window": label,
        "requested_start_year": start_year,
        "requested_end_year": end_year,
        "actual_first_complete_year": int(window[YEAR].min()),
        "actual_last_complete_year": int(window[YEAR].max()),
        "whitelisted_country_economies": whitelist_size,
        "units_with_joint_data": int(window[COUNTRY].nunique()),
        "units_without_joint_data": whitelist_size - int(window[COUNTRY].nunique()),
        "complete_country_years": int(len(window)),
        "country_pairs": n_pairs,
        "excluded_fewer_than_6_joint_years": exclusions[
            "fewer_than_2x_minimum_years"
        ],
        "excluded_identical_percentile_cutoffs": exclusions[
            "identical_low_high_cutoffs"
        ],
        "excluded_too_few_years_in_low_or_high_group": exclusions[
            "too_few_years_in_either_group"
        ],
        "median_joint_years_per_analyzed_unit": float(
            paired["complete_country_years"].median()
        ),
        "mean_growth_low_rents_pct": float(
            paired["mean_growth_low_rents_pct"].mean()
        ),
        "mean_growth_high_rents_pct": float(
            paired["mean_growth_high_rents_pct"].mean()
        ),
        "mean_difference_high_minus_low_pp": mean_diff,
        "sd_paired_difference_pp": sd_diff,
        "se_paired_difference_pp": se_diff,
        "ci_95_low_pp": ci_low,
        "ci_95_high_pp": ci_high,
        "paired_t_statistic": float(t_result.statistic),
        "degrees_of_freedom": df,
        "paired_t_p_value_two_sided": float(t_result.pvalue),
        "wilcoxon_statistic_two_sided": wilcoxon_stat,
        "wilcoxon_p_value_two_sided": wilcoxon_p,
        "positive_country_differences": int((differences > 0).sum()),
        "negative_country_differences": int((differences < 0).sum()),
        "tied_country_differences": int(np.isclose(differences, 0.0).sum()),
    }
    return summary, paired


def main() -> None:
    panel, whitelist_size = load_panel()
    periods = [
        ("1970-2021 main", 1970, 2021),
    ]
    summaries = []
    country_results = []
    for label, start_year, end_year in periods:
        summary, paired = analyze_window(
            panel, label, start_year, end_year, whitelist_size
        )
        summaries.append(summary)
        country_results.append(paired)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    country_df = pd.concat(country_results, ignore_index=True)
    summary_path = OUTPUT_DIR / "analysis_summary.csv"
    country_path = OUTPUT_DIR / "country_pairs.csv"
    report_path = OUTPUT_DIR / "analysis_report.txt"
    summary_df.to_csv(summary_path, index=False, float_format="%.10g")
    country_df.to_csv(country_path, index=False, float_format="%.10g")

    main_label = periods[0][0]
    main_pairs = country_df.loc[country_df["analysis_window"].eq(main_label)].sort_values(
        "difference_high_minus_low_pp"
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(1, len(main_pairs) + 1)
    ax.scatter(x, main_pairs["difference_high_minus_low_pp"], s=20, alpha=0.72)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="No difference")
    ax.axhline(
        summary_df.iloc[0]["mean_difference_high_minus_low_pp"],
        color="#d95f02",
        linewidth=1.8,
        label="Mean paired difference",
    )
    ax.set_title("Within-country GDP per capita growth difference")
    ax.set_xlabel("Countries/economies, ordered by paired difference")
    ax.set_ylabel("High-rent mean minus low-rent mean (percentage points)")
    ax.legend(frameon=False)
    fig.tight_layout()
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "paired_differences.svg", format="svg")
    plt.close(fig)

    lines = [
        "MATCHED-PAIRS MAIN ANALYSIS",
        "",
        "Input: WDI export has year columns 1960-2025; rents and GDP per capita growth overlap in 1970-2021.",
        "Variables: Total natural resources rents (% GDP) and GDP per capita growth (annual %).",
        f"Country/economy whitelist size: {whitelist_size}.",
        f"Complete country-years used: {int(summary_df.iloc[0]['complete_country_years']):,}.",
        "",
        "Method: country-specific linear P33/P67 cutoffs; middle observations omitted;",
        "at least 3 complete years in each rent group; one high-minus-low mean difference per unit.",
        "Each unit is equally weighted. The analysis uses the longest overlap with data for both indicators.",
        "",
    ]
    for row in summaries:
        lines.extend(
            [
                row["analysis_window"],
                f"- Units with joint data: {row['units_with_joint_data']}",
                f"- Paired units analyzed: {row['country_pairs']}",
                f"- Units without joint data: {row['units_without_joint_data']}",
                f"- Excluded: fewer than 6 years={row['excluded_fewer_than_6_joint_years']}; identical cutoffs={row['excluded_identical_percentile_cutoffs']}; fewer than 3 years in a group={row['excluded_too_few_years_in_low_or_high_group']}",
                f"- Mean growth, low rents: {row['mean_growth_low_rents_pct']:.4f}%",
                f"- Mean growth, high rents: {row['mean_growth_high_rents_pct']:.4f}%",
                f"- Mean difference (high minus low): {row['mean_difference_high_minus_low_pp']:.4f} percentage points",
                f"- 95% CI: [{row['ci_95_low_pp']:.4f}, {row['ci_95_high_pp']:.4f}]",
                f"- Paired t: t({row['degrees_of_freedom']}) = {row['paired_t_statistic']:.4f}, p = {row['paired_t_p_value_two_sided']:.6g}",
                f"- Wilcoxon signed-rank: W = {row['wilcoxon_statistic_two_sided']:.0f}, p = {row['wilcoxon_p_value_two_sided']:.6g}",
                "",
            ]
        )
    lines.extend(
        [
            "How to read N",
            "The file contains country-year observations used to classify rent periods and calculate means, but the paired tests use one difference per eligible country/economy. The inferential sample size is the number of pairs, not the number of country-year rows.",
            "",
            "Missing values were not imputed. The two rent-group averages generally come from different calendar years; the tests do not adjust for year-specific shocks or other time-varying factors. Results describe associations, not causal effects.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for row in summaries:
        print(
            f"{row['analysis_window']}: {row['complete_country_years']:,} country-years; "
            f"{row['country_pairs']} pairs; mean difference="
            f"{row['mean_difference_high_minus_low_pp']:.4f} pp; "
            f"paired-t p={row['paired_t_p_value_two_sided']:.6g}; "
            f"Wilcoxon p={row['wilcoxon_p_value_two_sided']:.6g}"
        )
    print(f"Saved: {summary_path}")
    print(f"Saved: {country_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

