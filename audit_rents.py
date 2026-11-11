"""Audit rent outliers, consecutive-year movements, and component consistency.

Run from the repository root with: python audit_rents.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import (
    COUNTRY, COUNTRY_NAME, SERIES, YEAR, RENTS_CODE,
    RAW_FILE, COUNTRY_FILE, OUTPUT_DIR, load_panel, analyze_window,
)

ROOT = Path(__file__).resolve().parent
COMPONENT_FILE = ROOT / "data" / "validation" / "rent_components_2000_2021.csv"
MAIN_START, MAIN_END = 2000, 2021
MIN_YEARS_PER_GROUP = 3


def read_rents_only() -> pd.DataFrame:
    countries = pd.read_csv(COUNTRY_FILE, dtype={COUNTRY: "string"})
    codes = set(countries[COUNTRY].dropna().astype(str).str.strip())
    raw = pd.read_csv(RAW_FILE, dtype={COUNTRY: "string}", na_values=["..", ""], encoding="utf-8-sig")
    year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4} \[YR\d{4}\]", str(c))]
    rents = raw.loc[
        raw[COUNTRY].astype("string").str.strip().isin(codes) & raw[SERIES].eq(RENTS_CODE),
        [COUNTRY, COUNTRY_NAME, *year_cols],
    ].copy()
    if rents.duplicated(COUNTRY).any():
        raise ValueError("พบแถว rents ซ้ำใน Country Code")
    long = rents.melt(id_vars=[COUNTRY, COUNTRY_NAME], var_name="year_label", value_name="rents_pct_gdp")
    long[YEAR] = long["year_label"].str.extract(r"^(\d{4})")[0].astype(int)
    long["rents_pct_gdp"] = pd.to_numeric(long["rents_pct_gdp"], errors="coerce")
    long = long.loc[long[YEAR].between(MAIN_START, MAIN_END)].dropna(subset=["rents_pct_gdp"])
    return long.sort_values([COUNTRY, YEAR]).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rents = read_rents_only()
    values = rents["rents_pct_gdp"]
    q1, q3 = values.quantile([0.25, 0.75])
    global_fence = q3 + 1.5 * (q3 - q1)

    tukey_rows = []
    for (code, name), group in rents.groupby([COUNTRY, COUNTRY_NAME], sort=True):
        if len(group) < 8:
            continue
        c_q1, c_q3 = group["rents_pct_gdp"].quantile([0.25, 0.75])
        fence = c_q3 + 1.5 * (c_q3 - c_q1)
        for row in group.loc[group["rents_pct_gdp"] > fence].itertuples(index=False):
            tukey_rows.append({
                "country_code": code, "country_name": name, "year": int(getattr(row, YEAR)),
                "rents_pct_gdp": float(row.rents_pct_gdp), "country_q1": float(c_q1),
                "country_q3": float(c_q3), "country_upper_fence": float(fence),
                "rule": "rents > country Q3 + 1.5*IQR; at least 8 observations",
            })
    tukey = pd.DataFrame(tukey_rows)
    if tukey.empty:
        tukey = pd.DataFrame(columns=[
            "country_code", "country_name", "year", "rents_pct_gdp", "country_q1",
            "country_q3", "country_upper_fence", "rule",
        ])
    tukey.to_csv(OUTPUT_DIR / "within_country_tukey_candidates.csv", index=False)

    rents.nlargest(20, "rents_pct_gdp").to_csv(OUTPUT_DIR / "top_rent_values.csv", index=False)

    panel, whitelist_size = load_panel()
    panel = panel.loc[panel[YEAR].between(MAIN_START, MAIN_END)].copy()
    _, pairs = analyze_window(panel, "2000-2021 main", MAIN_START, MAIN_END, whitelist_size)
    pair_lookup = pairs.set_index(COUNTRY)
    matched = panel.loc[panel[COUNTRY].isin(pair_lookup.index)].copy()
    transition_counts = {(a, b): 0 for a in "LMH" for b in "LMH"}
    changes = []
    for code, group in matched.groupby(COUNTRY, sort=True):
        pair = pair_lookup.loc[code]
        low_cut, high_cut = pair["low_rent_cutoff_pct_gdp"], pair["high_rent_cutoff_pct_gdp"]
        group = group.sort_values(YEAR).copy()
        group["rent_group"] = np.where(
            group["resource_rents_pct_gdp"] <= low_cut, "L",
            np.where(group["resource_rents_pct_gdp"] >= high_cut, "H", "M"),
        )
        rows = group[[COUNTRY_NAME, YEAR, "resource_rents_pct_gdp", "rent_group"]].to_dict("records")
        for previous, current in zip(rows, rows[1:]):
            if int(current[YEAR]) - int(previous[YEAR]) != 1:
                continue
            a, b = previous["rent_group"], current["rent_group"]
            transition_counts[(a, b)] += 1
            before, after = previous["resource_rents_pct_gdp"], current["resource_rents_pct_gdp"]
            changes.append({
                "country_code": code, "country_name": current[COUNTRY_NAME],
                "from_year": int(previous[YEAR]), "to_year": int(current[YEAR]),
                "rents_from_pct_gdp": float(before), "rents_to_pct_gdp": float(after),
                "absolute_change_pp": abs(float(after - before)), "change_pp": float(after - before),
                "group_from": a, "group_to": b,
            })

    transition = pd.DataFrame([
        {"from_group": a, "to_group": b, "count": transition_counts[(a, b)]}
        for a in "LMH" for b in "LMH"
    ])
    transition.to_csv(OUTPUT_DIR / "adjacent_group_transition_matrix.csv", index=False)
    changes_df = pd.DataFrame(changes).sort_values("absolute_change_pp", ascending=False)
    changes_df.head(20).to_csv(OUTPUT_DIR / "largest_adjacent_rent_changes.csv", index=False)

    total = int(len(changes_df))
    outer = changes_df.loc[
        changes_df["group_from"].isin(["L", "H"]) & changes_df["group_to"].isin(["L", "H"])
    ]
    same_any = int((changes_df["group_from"] == changes_df["group_to"]).sum())
    same_outer = int((outer["group_from"] == outer["group_to"]).sum())
    switched_outer = int(len(outer) - same_outer)

    component_result = {
        "rows_with_total_and_all_components": np.nan,
        "max_absolute_total_minus_component_sum": np.nan,
    }
    if COMPONENT_FILE.exists():
        components = pd.read_csv(COMPONENT_FILE)
        component_cols = [
            "oil_rents_pct_gdp", "natural_gas_rents_pct_gdp", "coal_rents_pct_gdp",
            "mineral_rents_pct_gdp", "forest_rents_pct_gdp",
        ]
        valid = components.dropna(subset=["total_resource_rents_pct_gdp", *component_cols]).copy()
        residual = valid["total_resource_rents_pct_gdp"] - valid[component_cols].sum(axis=1)
        component_result = {
            "rows_with_total_and_all_components": int(len(valid)),
            "max_absolute_total_minus_component_sum": float(residual.abs().max()),
        }

    summary_rows = [
        ("Rents observations 2000-2021", len(rents), "country-years"),
        ("Economies with rents data", rents[COUNTRY].nunique(), "economies"),
        ("Minimum rents", values.min(), "% of GDP"),
        ("Maximum rents", values.max(), "% of GDP"),
        ("Negative rents observations", int((values < 0).sum()), "country-years"),
        ("Rents above 100", int((values > 100).sum()), "country-years"),
        ("Rents above 50", int((values > 50).sum()), "country-years"),
        ("Global Tukey Q1", q1, "% of GDP"),
        ("Global Tukey Q3", q3, "% of GDP"),
        ("Global Tukey upper fence", global_fence, "% of GDP"),
        ("Values above global Tukey fence", int((values > global_fence).sum()), "country-years"),
        ("Within-economy Tukey candidates", len(tukey), "country-years"),
        ("Economies with within-economy Tukey candidates", tukey["country_code"].nunique(), "economies"),
        ("Adjacent complete-year changes in 189-pair sample", total, "transitions"),
        ("Adjacent observations staying in same L/M/H group", same_any, "transitions"),
        ("Same group share of adjacent transitions", same_any / total, "share"),
        ("Adjacent observations both in low/high tails", len(outer), "transitions"),
        ("Low/high tail transitions staying in same tail", same_outer, "transitions"),
        ("Low/high tail transitions switching tails", switched_outer, "transitions"),
        ("Same-tail share among low/high tail transitions", same_outer / len(outer), "share"),
        ("Median absolute adjacent-year rent change", changes_df["absolute_change_pp"].median(), "percentage points"),
        ("95th percentile absolute adjacent-year rent change", changes_df["absolute_change_pp"].quantile(.95), "percentage points"),
        ("Adjacent changes greater than 20 percentage points", int((changes_df["absolute_change_pp"] > 20).sum()), "transitions"),
        ("Adjacent changes greater than 40 percentage points", int((changes_df["absolute_change_pp"] > 40).sum()), "transitions"),
        ("Rows with total and all five components", component_result["rows_with_total_and_all_components"], "country-years"),
        ("Max |total - sum of components|", component_result["max_absolute_total_minus_component_sum"], "percentage points"),
    ]
    pd.DataFrame(summary_rows, columns=["diagnostic", "value", "unit"]).to_csv(
        OUTPUT_DIR / "rent_diagnostics_summary.csv", index=False
    )

    if len(outer):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(
            ["Stayed in same
low/high group", "Switched
low ↔ high"],
            [same_outer / len(outer) * 100, switched_outer / len(outer) * 100],
            color=["#2b6cb0", "#ed8936"],
        )
        ax.set_ylabel("Share of adjacent-year low/high comparisons (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Rent-group persistence, 2000-2021")
        for i, v in enumerate([same_outer / len(outer) * 100, switched_outer / len(outer) * 100]):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(figures / "rent_group_transitions.svg", format="svg")
        plt.close(fig)

    print(f"Rents observations: {len(rents):,}; within-country Tukey candidates: {len(tukey)}")
    print(f"Adjacent complete-year transitions: {total}; outer-tail persistence: {same_outer}/{len(outer)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
