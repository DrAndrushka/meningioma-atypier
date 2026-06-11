"""Step 01 — load the spreadsheet, optionally cut to ANALYSIS_YEARS.

load_raw handles csv or xlsx. filter_cohort: None = all years, [2024, 2025] = subset.
Empty list is an error — use None instead.
"""
from __future__ import annotations

import pandas as pd


def load_raw(data_path: str) -> pd.DataFrame:
    if str(data_path).endswith(".csv"):
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_excel(data_path)
    print(f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    return df_raw


def _fmt_year_list(years) -> str:
    return "[" + ", ".join(str(int(y)) for y in years) + "]"


def filter_cohort(
    df_raw: pd.DataFrame,
    year_column: str,
    analysis_years: list[int] | None,
) -> pd.DataFrame:
    if analysis_years is not None and len(analysis_years) == 0:
        raise ValueError("ANALYSIS_YEARS is []; use None for all years or e.g. [2025]")

    if analysis_years is not None:
        n_before = len(df_raw)
        filtered = df_raw.loc[
            pd.to_numeric(df_raw[year_column], errors="coerce").isin(analysis_years)
        ].copy()
        if filtered.empty:
            raise ValueError(f"No rows with {year_column} in {analysis_years!r}")
        years_label = _fmt_year_list(analysis_years)
        dropped = n_before - len(filtered)
        print(
            f"📅 Cohort filter · {year_column} ∈ {years_label}\n"
            f"{len(filtered)} rows kept · {dropped} dropped"
        )
        return filtered

    years = sorted(
        pd.to_numeric(df_raw[year_column], errors="coerce").dropna().astype(int).unique()
    )
    years_label = _fmt_year_list(years)
    print(
        f"📅 Cohort · all years in {year_column}\n"
        f"{years_label}\n"
        f"{len(df_raw)} rows"
    )
    return df_raw
