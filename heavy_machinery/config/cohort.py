"""Load raw cohort export; optional year filter for the row-filters section.

``load_raw`` supports CSV or Excel (single path or list of paths).
``apply_cohort_year_filter`` / ``filter_cohort``: ``None`` = all years; ``[2024, 2025]`` = subset.
Empty list raises — use ``None`` instead. The logged variant returns a row-filters-style log
for ``finalize_row_drops`` / drop_log.
"""
from __future__ import annotations

import pandas as pd


def _read_export(data_path: str) -> pd.DataFrame:
    if str(data_path).endswith(".csv"):
        return pd.read_csv(data_path)
    return pd.read_excel(data_path)


def load_raw(data_path: str | list[str]) -> pd.DataFrame:
    """Read the raw export(s), remembering which files they came from.

    The names ride along in ``df.attrs`` so the cleaning summary can record
    them: without that, nothing downstream of this call knows what the whole
    pipeline was built from, and the report has no way to say.
    """
    if isinstance(data_path, str):
        df_raw = _read_export(data_path)
        df_raw.attrs["source_files"] = [str(data_path)]
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
    df_raw.attrs["source_files"] = [str(p) for p in paths]
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
    filtered, _log = apply_cohort_year_filter(df_raw, year_column, analysis_years)
    return filtered


def apply_cohort_year_filter(
    df: pd.DataFrame,
    year_column: str,
    analysis_years: list[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Year subset filter plus a one-row log compatible with ``row_filters`` / drop_log."""
    if analysis_years is not None and len(analysis_years) == 0:
        raise ValueError("ANALYSIS_YEARS is []; use None for all years or e.g. [2025]")

    n_before = len(df)

    if analysis_years is not None:
        filtered = df.loc[
            pd.to_numeric(df[year_column], errors="coerce").isin(analysis_years)
        ].copy()
        if filtered.empty:
            raise ValueError(f"No rows with {year_column} in {analysis_years!r}")
        years_label = _fmt_year_list(analysis_years)
        note = f"{year_column} ∈ {years_label}"
        print(
            f"📅 Cohort filter · {note}\n"
            f"{len(filtered)} rows kept · {n_before - len(filtered)} dropped"
        )
    else:
        years = sorted(
            pd.to_numeric(df[year_column], errors="coerce").dropna().astype(int).unique()
        )
        years_label = _fmt_year_list(years)
        note = f"all years in {year_column}: {years_label}"
        print(
            f"📅 Cohort · all years in {year_column}\n"
            f"{years_label}\n"
            f"{n_before} rows"
        )
        filtered = df

    log = pd.DataFrame([{
        "name": "cohort year",
        "active": True,
        "rows_before": n_before,
        "rows_after": len(filtered),
        "rows_removed": n_before - len(filtered),
        "note": note,
    }])
    return filtered, log
