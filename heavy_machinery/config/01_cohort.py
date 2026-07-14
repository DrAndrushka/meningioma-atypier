"""Step 01 — load raw cohort export and optional year filter.

``load_raw`` supports CSV or Excel (single path or list of paths). ``filter_cohort``: ``None`` = all years;
``[2024, 2025]`` = subset. Empty list raises — use ``None`` instead.
"""
from __future__ import annotations

import pandas as pd


def _read_export(data_path: str) -> pd.DataFrame:
    if str(data_path).endswith(".csv"):
        return pd.read_csv(data_path)
    return pd.read_excel(data_path)


def load_raw(data_path: str | list[str]) -> pd.DataFrame:
    if isinstance(data_path, str):
        df_raw = _read_export(data_path)
        print(f"Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        return df_raw

    paths = list(data_path)
    if not paths:
        raise ValueError("data_path is empty")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _read_export(path)
        print(f"Loaded {path}: {df.shape[0]} rows × {df.shape[1]} columns")
        frames.append(df)

    df_raw = pd.concat(frames, ignore_index=True)
    print(
        f"Combined: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns "
        f"({len(frames)} files)"
    )
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
