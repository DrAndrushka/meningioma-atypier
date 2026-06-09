"""§06 — declarative derivations (notebook-driven)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    "kind",
    "rows_nonmissing",
    "rows_missing",
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

@dataclass
class Compute:
    """Custom derivation that needs several columns: ``fn`` receives the whole frame."""

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    sources: list[str] = field(default_factory=list)
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

    _, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        ordered_levels=levels,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **_base_log(spec, type_name),
        rows_nonmissing=rows_nonmissing,
        rows_missing=rows_missing,
        schema_action=schema_action,
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

    _, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        ordered_levels=spec.ordered_levels if spec.kind == "ordinal" else None,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **_base_log(spec, type_name),
        rows_nonmissing=rows_nonmissing,
        rows_missing=rows_missing,
        schema_action=schema_action,
    ))
    return df


def _apply_compute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    spec: Compute,
    log: list[dict],
    ) -> pd.DataFrame:
    type_name = "Compute"
    base = {
        "derivation": spec.name,
        "type": type_name,
        "active": spec.active,
        "source": ", ".join(spec.sources),
        "kind": spec.kind,
        "reason": spec.reason,
    }

    if not spec.active:
        log.append(_log_entry(**base, schema_action="skipped (inactive)"))
        return df

    missing = [c for c in spec.sources if c not in df.columns]
    if missing:
        log.append(_log_entry(
            **base,
            schema_action="skipped (source missing)",
            warning=f"Source column(s) {missing!r} not in df.",
        ))
        return df

    if spec.name in df.columns and not spec.overwrite:
        log.append(_log_entry(
            **base,
            schema_action="skipped (already exists)",
            warning=f"Column {spec.name!r} exists; set overwrite=True to replace.",
        ))
        return df

    result = spec.fn(df)
    if spec.kind == "ordinal" and spec.ordered_levels is not None:
        result = pd.Categorical(result, categories=spec.ordered_levels, ordered=True)

    df[spec.name] = result

    _, rows_nonmissing, rows_missing = _row_stats(df[spec.name])
    schema_action = _set_schema_col(
        schema,
        spec.name,
        spec.kind,
        ordered_levels=spec.ordered_levels if spec.kind == "ordinal" else None,
        reason=spec.reason,
    )
    log.append(_log_entry(
        **base,
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
        elif isinstance(spec, Compute):
            out = _apply_compute(out, out_schema, spec, log)
        elif isinstance(spec, Apply):
            out = _apply_apply(out, out_schema, spec, log)
        else:
            raise TypeError(f"Unknown derivation type: {type(spec)!r}")

    new_cols = [c for c in out.columns if c not in old_cols]
    if preview and new_cols:
        display(out[new_cols].head())

    derivation_log = (
        pd.DataFrame(log, columns=_LOG_COLUMNS)
        if log
        else pd.DataFrame(columns=_LOG_COLUMNS)
    )

    if write_csv and output_root is not None:
        write_cleaned_csv(output_root, out, out_schema)
        out_dir = Path(output_root) / "cleaning"
        out_dir.mkdir(parents=True, exist_ok=True)
        derivation_log.to_csv(out_dir / "derivation_log.csv", index=False)
        _update_cleaning_summary_derived(output_root, new_cols, len(out))
        # Append derived columns to the schema summary so they appear in the
        # report's Schema story. Append-only: we must NOT re-export the whole
        # post-cleaning schema, because cleaning rewrites binned datetime
        # columns to kind='ordinal' in place — that would clobber the
        # originally-declared 'datetime' kind written at schema-definition time.
        _append_derived_to_schema_summary(output_root, out_schema, new_cols)

    return out, out_schema, derivation_log


def _append_derived_to_schema_summary(
    output_root: Path | str,
    out_schema: dict[str, ColSpec],
    created_cols: list[str],
) -> None:
    """Add derived columns to schema_summary.csv without touching existing rows.

    Idempotent: existing rows for the derived columns are replaced.
    """
    if not created_cols:
        return
    from schema_infer import export_schema_summary, schema_summary

    summary_path = Path(output_root) / "schema" / "schema_summary.csv"
    if not summary_path.exists():
        export_schema_summary(out_schema, output_root)
        return

    from cleaning import format_table_for_csv

    existing = pd.read_csv(summary_path)
    derived_schema = {c: out_schema[c] for c in created_cols if c in out_schema}
    derived_rows = format_table_for_csv(schema_summary(derived_schema))
    derived_rows = derived_rows.reindex(columns=existing.columns)
    if "column" in existing.columns:
        existing = existing[~existing["column"].isin(created_cols)]
    combined = pd.concat([existing, derived_rows], ignore_index=True)
    combined.to_csv(summary_path, index=False)


def _update_cleaning_summary_derived(
    output_root: Path | str,
    created_cols: list[str],
    n_rows: int,
) -> None:
    """Insert a ``derived`` row into cleaning_summary.csv naming the new columns.

    Idempotent: any existing ``derived`` row is replaced, so re-running the
    derivations cell does not stack duplicate rows.
    """
    summary_path = Path(output_root) / "cleaning" / "cleaning_summary.csv"
    if not summary_path.exists() or not created_cols:
        return

    summary = pd.read_csv(summary_path)
    if "step" not in summary.columns:
        return

    summary = summary[summary["step"] != "derived"].copy()
    row = {c: "" for c in summary.columns}
    row["step"] = "derived"
    if "detail" in row:
        row["detail"] = (
            f"added {len(created_cols)} column(s): " + ", ".join(created_cols))
    if "n_rows" in row:
        row["n_rows"] = n_rows
    if "n_dropped" in row:
        row["n_dropped"] = 0

    new_df = pd.DataFrame([row], columns=summary.columns)
    if (summary["step"] == "final").any():
        pos = summary.index.get_loc(summary.index[summary["step"] == "final"][0])
        summary = pd.concat(
            [summary.iloc[:pos], new_df, summary.iloc[pos:]], ignore_index=True)
    else:
        summary = pd.concat([summary, new_df], ignore_index=True)

    summary.to_csv(summary_path, index=False)
