"""§06 — declarative row filters (keep logic) and cleaning artifact export."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from IPython.display import display

from cleaning import export_cleaning_artifacts

SPINAL_MENINGIOMA_LABEL = "Mugurkaula meningioma"


@dataclass
class RowFilter:
    """Keep rows where ``keep(df)`` is True. Set ``active=False`` to skip."""

    name: str
    keep: Callable[[pd.DataFrame], pd.Series]
    note: str  # human-readable rule text for logs and report
    active: bool = True


def keep_brain_meningioma(df: pd.DataFrame) -> pd.Series:
    """Keep intracranial cases; drop rows flagged as spinal meningioma."""
    side = df["side"].astype("string")
    mri = df["mri_date"].astype("string")
    flagged = (side == SPINAL_MENINGIOMA_LABEL) | (mri == SPINAL_MENINGIOMA_LABEL)
    return ~flagged.fillna(False)


def brain_meningioma_row_filter(*, active: bool = True) -> RowFilter:
    """Pre-schema inclusion filter — run before ``apply_schema``."""
    return RowFilter(
        name="Meningioma location - brain",
        keep=keep_brain_meningioma,
        note="inclusion criteria - brain meningioma",
        active=active,
    )


def apply_row_filter(
    df: pd.DataFrame,
    log: list[dict],
    row_filter: RowFilter,
) -> pd.DataFrame:
    """Apply one filter; append a log entry to ``log``."""
    rows_before = len(df)

    if not row_filter.active:
        log.append({
            "name": row_filter.name,
            "active": False,
            "rows_before": rows_before,
            "rows_after": rows_before,
            "rows_removed": 0,
            "note": row_filter.note,
        })
        return df

    mask = row_filter.keep(df)
    if len(mask) != rows_before:
        raise ValueError(
            f"RowFilter {row_filter.name!r}: keep mask length {len(mask)} "
            f"!= len(df) {rows_before}"
        )

    out = df.loc[mask].copy()
    rows_after = len(out)
    log.append({
        "name": row_filter.name,
        "active": True,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "note": row_filter.note,
    })
    return out


def apply_row_filters(
    df: pd.DataFrame,
    filters: list[RowFilter],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply filters in order; return filtered df and a log DataFrame."""
    log: list[dict] = []
    for row_filter in filters:
        df = apply_row_filter(df, log, row_filter)
    return df, pd.DataFrame(log)


def combine_row_filter_logs(*logs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate pre- and post-schema filter logs in run order."""
    frames = [log for log in logs if log is not None and not log.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _drop_log_for_export(row_filter_log: pd.DataFrame) -> list[dict]:
    """Map row_filter_log → drop_log format for cleaning_summary.csv / report."""
    if row_filter_log.empty:
        return []
    entries: list[dict] = []
    for _, row in row_filter_log.iterrows():
        entries.append({
            "reason": row["name"],
            "criterion": row["note"],
            "n_before": int(row["rows_before"]),
            "n_dropped": int(row["rows_removed"]),
            "n_remaining": int(row["rows_after"]),
        })
    return entries


def finalize_row_drops(
    df: pd.DataFrame,
    row_filter_log: pd.DataFrame,
    *,
    output_root: Path,
    df_raw: pd.DataFrame,
    n_rows_pre_schema: int | None = None,
    n_rows_after_schema: int,
    schema,
    dupes,
    schema_log,
) -> pd.DataFrame:
    if row_filter_log.empty:
        print(f"Row filters applied — {len(df)} rows remaining")
    else:
        display(row_filter_log)

    export_cleaning_artifacts(
        output_root,
        df=df,
        n_rows_raw=len(df_raw),
        n_rows_pre_schema=n_rows_pre_schema,
        n_rows_after_schema=n_rows_after_schema,
        n_rows_final=len(df),
        schema=schema,
        drop_log=_drop_log_for_export(row_filter_log),
        dupes=dupes,
        schema_log=schema_log,
    )
    return df
