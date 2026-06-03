"""§06 — declarative derivations (notebook-driven)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pandas as pd
from IPython.display import display

from cleaning import write_cleaned_csv
from schema_infer import ColSpec

_LOG_COLUMNS = [
    "derivation",
    "type",
    "active",
    "source",
    "created",
    "kind",
    "rows_total",
    "rows_nonmissing",
    "rows_missing",
    "matched_n",
    "schema_action",
    "warning",
    "reason",
]


@dataclass
class BinNumeric:
    """Cut a numeric source column into ordered bins."""

    name: str
    source: str
    bins: list
    labels: list
    kind: str = "ordinal"
    active: bool = True
    overwrite: bool = False
    reason: str = ""
    right: bool = False
    ordered_levels: list | None = None


@dataclass
class IsIn:
    """Boolean flag: True when source value is in ``values``, else False."""

    name: str
    source: str
    values: list
    kind: str = "binary"
    active: bool = True
    overwrite: bool = False
    reason: str = ""


@dataclass
class Apply:
    """Custom derivation from a source column."""

    name: str
    source: str
    fn: Callable[[pd.Series], pd.Series]
    kind: str = "continuous"
    active: bool = True
    overwrite: bool = False
    reason: str = ""
    ordered_levels: list | None = None


def _copy_schema(schema: dict[str, ColSpec]) -> dict[str, ColSpec]:
    return {k: replace(v) for k, v in schema.items()}


def _log_entry(**kwargs) -> dict:
    row = {col: "" for col in _LOG_COLUMNS}
    for key, value in kwargs.items():
        if key in row:
            row[key] = value
    return row


def _row_stats(series: pd.Series) -> tuple[int, int, int]:
    rows_total = len(series)
    rows_nonmissing = int(series.notna().sum())
    rows_missing = rows_total - rows_nonmissing
    return rows_total, rows_nonmissing, rows_missing


def _set_schema_col(
    schema: dict[str, ColSpec],
    name: str,
    kind: str,
    *,
    ordered_levels: list | None = None,
    reason: str = "",
) -> str:
    existed = name in schema
    schema[name] = ColSpec(
        name=name,
        kind=kind,
        ordered_levels=ordered_levels,
        note=reason,
    )
    verb = "updated" if existed else "added"
    return f"{verb} ColSpec ({kind}) for {name}"


def _base_log(spec, type_name: str) -> dict:
    return {
        "derivation": spec.name,
        "type": type_name,
        "active": spec.active,
        "source": spec.source,
        "created": spec.name,
        "kind": spec.kind,
        "reason": spec.reason,
    }


def _should_apply(
    spec,
    df: pd.DataFrame,
    log: list[dict],
    type_name: str,
) -> bool:
    base = _base_log(spec, type_name)

    if not spec.active:
        log.append(_log_entry(
            **base,
            schema_action="skipped (inactive)",
        ))
        return False

    if spec.source not in df.columns:
        log.append(_log_entry(
            **base,
            schema_action="skipped (source missing)",
            warning=f"Source column {spec.source!r} not in df.",
        ))
        return False

    if spec.name in df.columns and not spec.overwrite:
        log.append(_log_entry(
            **base,
            schema_action="skipped (already exists)",
            warning=f"Column {spec.name!r} exists; set overwrite=True to replace.",
        ))
        return False

    return True


def _apply_bin_numeric(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    spec: BinNumeric,
    log: list[dict],
) -> pd.DataFrame:
    type_name = "BinNumeric"
    if not _should_apply(spec, df, log, type_name):
        return df

    if len(spec.bins) - 1 != len(spec.labels):
        log.append(_log_entry(
            **_base_log(spec, type_name),
            schema_action="skipped (invalid bins)",
            warning="len(bins) - 1 must equal len(labels).",
        ))
        return df

    levels = spec.ordered_levels if spec.ordered_levels is not None else list(spec.labels)
    cut = pd.cut(
        df[spec.source],
        bins=list(spec.bins),
        labels=spec.labels,
        right=spec.right,
        include_lowest=True,
    )
    df[spec.name] = pd.Categorical(cut, categories=levels, ordered=True)

    rows_total, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        ordered_levels=levels,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **_base_log(spec, type_name),
        rows_total=rows_total,
        rows_nonmissing=rows_nonmissing,
        rows_missing=rows_missing,
        schema_action=schema_action,
    ))
    return df


def _apply_is_in(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    spec: IsIn,
    log: list[dict],
) -> pd.DataFrame:
    type_name = "IsIn"
    if not _should_apply(spec, df, log, type_name):
        return df

    matched = df[spec.source].isin(spec.values)
    matched_n = int(matched.sum())
    df[spec.name] = matched.astype("boolean")

    warning = ""
    if matched_n == 0:
        warning = "No rows matched values; check dtype or values."

    rows_total, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **_base_log(spec, type_name),
        rows_total=rows_total,
        rows_nonmissing=rows_nonmissing,
        rows_missing=rows_missing,
        matched_n=matched_n,
        schema_action=schema_action,
        warning=warning,
    ))
    return df


def _apply_apply(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    spec: Apply,
    log: list[dict],
) -> pd.DataFrame:
    type_name = "Apply"
    if not _should_apply(spec, df, log, type_name):
        return df

    result = spec.fn(df[spec.source])
    if spec.kind == "ordinal" and spec.ordered_levels is not None:
        result = pd.Categorical(result, categories=spec.ordered_levels, ordered=True)

    df[spec.name] = result

    rows_total, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        ordered_levels=spec.ordered_levels if spec.kind == "ordinal" else None,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **_base_log(spec, type_name),
        rows_total=rows_total,
        rows_nonmissing=rows_nonmissing,
        rows_missing=rows_missing,
        schema_action=schema_action,
    ))
    return df


def apply_derivations(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    derivations: list,
    *,
    output_root: Path | str | None = None,
    write_csv: bool = False,
    preview: bool = True,
) -> tuple[pd.DataFrame, dict[str, ColSpec], pd.DataFrame]:
    """Apply notebook-declared derivations; return updated df, schema, and audit log."""
    out = df.copy()
    out_schema = _copy_schema(schema)
    log: list[dict] = []
    old_cols = set(out.columns)

    for spec in derivations:
        if isinstance(spec, BinNumeric):
            out = _apply_bin_numeric(out, out_schema, spec, log)
        elif isinstance(spec, IsIn):
            out = _apply_is_in(out, out_schema, spec, log)
        elif isinstance(spec, Apply):
            out = _apply_apply(out, out_schema, spec, log)
        else:
            raise TypeError(f"Unknown derivation type: {type(spec)!r}")

    new_cols = [c for c in out.columns if c not in old_cols]
    if preview and new_cols:
        display(out[new_cols].head())

    if write_csv and output_root is not None:
        write_cleaned_csv(output_root, out, out_schema)

    derivation_log = (
        pd.DataFrame(log, columns=_LOG_COLUMNS)
        if log
        else pd.DataFrame(columns=_LOG_COLUMNS)
    )
    return out, out_schema, derivation_log
