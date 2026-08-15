"""Derived columns (bins, flags, custom functions).

``BinNumeric``, ``Apply`` (single column), ``Compute`` (whole frame). The derivation
list lives in the cleaning notebook; ``apply_derivations`` runs it and updates schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

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
    "rule",
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
    rule: str = ""      # how the column is computed (shown in the report)
    reason: str = ""    # why — citation for literature-derived cutoffs
    right: bool = False
    ordered_levels: list | None = None
    dda_in_derived: bool = False  # True → DDA report "Derived" table
    eda_in_derived: bool | None = False  # True→Derived, False→Native, None→omit from EDA
    hide_parent: bool = False  # True → parent stays in df, omitted from EDA/DDA/modelling show
    positive_class: Any = None  # index / present class copied onto ColSpec


@dataclass
class Apply:
    """Custom derivation from a source column."""

    name: str
    source: str
    fn: Callable[[pd.Series], pd.Series]
    kind: str = "continuous"
    active: bool = True
    rule: str = ""
    reason: str = ""
    ordered_levels: list | None = None
    dda_in_derived: bool = False  # True → DDA report "Derived" table
    eda_in_derived: bool | None = False  # True→Derived, False→Native, None→omit from EDA
    hide_parent: bool = False  # True → parent stays in df, omitted from EDA/DDA/modelling show
    positive_class: Any = None  # index / present class copied onto ColSpec


@dataclass
class Compute:
    """Custom derivation that needs several columns: ``fn`` receives the whole frame."""

    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    sources: list[str] = field(default_factory=list)
    kind: str = "continuous"
    active: bool = True
    rule: str = ""
    reason: str = ""
    ordered_levels: list | None = None
    dda_in_derived: bool = False  # True → DDA report "Derived" table
    eda_in_derived: bool | None = False  # True→Derived, False→Native, None→omit from EDA
    hide_parent: bool = False  # True → parent stays in df, omitted from EDA/DDA/modelling show
    positive_class: Any = None  # index / present class copied onto ColSpec


def derived_dependencies_from(derivations: Sequence) -> dict[str, list[str]]:
    """Build MICE parent→child map from active ``Apply`` / ``Compute`` / ``BinNumeric``.

    Skips inactive specs. Skips in-place adjustments where ``name`` is also listed
    as a source (e.g. structural-zero rewrite of ``edema_volume_cm3``) — those
    stay in the imputation matrix instead of drop/recreate.
    """
    out: dict[str, list[str]] = {}
    for spec in derivations:
        if not getattr(spec, "active", True):
            continue
        if isinstance(spec, (Apply, BinNumeric)):
            sources = [spec.source]
        elif isinstance(spec, Compute):
            sources = list(spec.sources)
        else:
            continue
        if spec.name in sources:
            continue
        out[spec.name] = sources
    return out


def dda_in_derived_columns(derivations: Sequence) -> frozenset[str]:
    """Column names flagged ``dda_in_derived=True`` on derivation specs."""
    return frozenset(
        spec.name
        for spec in derivations
        if getattr(spec, "dda_in_derived", False) is True
    )


def eda_in_derived_columns(derivations: Sequence) -> frozenset[str]:
    """Column names with ``eda_in_derived=True`` (EDA Derived table)."""
    return frozenset(
        spec.name
        for spec in derivations
        if getattr(spec, "eda_in_derived", False) is True
    )


def eda_excluded_columns(derivations: Sequence) -> frozenset[str]:
    """Column names with ``eda_in_derived=None`` (omit from EDA entirely)."""
    return frozenset(
        spec.name
        for spec in derivations
        if getattr(spec, "eda_in_derived", False) is None
    )


def hidden_parent_columns(derivations: Sequence) -> frozenset[str]:
    """Parent column names flagged ``hide_parent=True`` on active specs.

    Parents stay in the dataframe (MICE / re-derive) but are omitted from
    EDA, DDA, and modelling predictor assembly via ``hidden_parent_columns.csv``.

    In-place rewrites (``name`` also listed as a source) never hide ``name``.
    """
    hidden: set[str] = set()
    for spec in derivations:
        if not getattr(spec, "active", True):
            continue
        if not getattr(spec, "hide_parent", False):
            continue
        if isinstance(spec, (Apply, BinNumeric)):
            if spec.source != spec.name:
                hidden.add(spec.source)
        elif isinstance(spec, Compute):
            for src in spec.sources:
                if src != spec.name:
                    hidden.add(src)
    return frozenset(hidden)


def write_hidden_parent_columns(
    output_root: Path | str,
    derivations: Sequence,
) -> Path:
    """Persist hidden parents for EDA/DDA/modelling (``cleaning/hidden_parent_columns.csv``)."""
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"column": sorted(hidden_parent_columns(derivations))}).to_csv(
        path, index=False,
    )
    return path


def load_hidden_parent_columns(output_root: Path | str | None) -> frozenset[str]:
    """Load ``cleaning/hidden_parent_columns.csv``; empty if missing."""
    if output_root is None:
        return frozenset()
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    if not path.exists():
        return frozenset()
    try:
        tbl = pd.read_csv(path)
    except Exception:
        return frozenset()
    if "column" not in tbl.columns:
        return frozenset()
    return frozenset(str(c) for c in tbl["column"].dropna().tolist())


def write_dda_in_derived_columns(
    output_root: Path | str,
    derivations: Sequence,
) -> Path:
    """Persist DDA-derived names for the HTML report (``cleaning/dda_derived_columns.csv``)."""
    path = Path(output_root) / "cleaning" / "dda_derived_columns.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"column": sorted(dda_in_derived_columns(derivations))}).to_csv(
        path, index=False,
    )
    return path


def write_eda_in_derived_columns(
    output_root: Path | str,
    derivations: Sequence,
) -> tuple[Path, Path]:
    """Persist EDA Derived / excluded column lists for the HTML report.

    * ``cleaning/eda_derived_columns.csv`` — ``eda_in_derived=True``
    * ``cleaning/eda_excluded_columns.csv`` — ``eda_in_derived=None`` (hidden)
    """
    root = Path(output_root) / "cleaning"
    root.mkdir(parents=True, exist_ok=True)
    derived_path = root / "eda_derived_columns.csv"
    excluded_path = root / "eda_excluded_columns.csv"
    pd.DataFrame({"column": sorted(eda_in_derived_columns(derivations))}).to_csv(
        derived_path, index=False,
    )
    pd.DataFrame({"column": sorted(eda_excluded_columns(derivations))}).to_csv(
        excluded_path, index=False,
    )
    return derived_path, excluded_path


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
    positive_class: Any = None,
) -> str:
    existed = name in schema
    schema[name] = ColSpec(
        name=name,
        kind=kind,
        ordered_levels=ordered_levels,
        note=reason,
        positive_class=positive_class,
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
        "rule": spec.rule,
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
        positive_class=spec.positive_class,
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
        positive_class=spec.positive_class,
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
        "rule": spec.rule,
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
        positive_class=spec.positive_class,
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
        write_dda_in_derived_columns(output_root, derivations)
        write_eda_in_derived_columns(output_root, derivations)
        write_hidden_parent_columns(output_root, derivations)
        _update_cleaning_summary_derived(
            output_root,
            new_cols,
            n_rows=len(out),
            n_columns=out.shape[1],
            derivation_log=derivation_log,
        )
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
    n_columns: int,
    derivation_log: pd.DataFrame | None = None,
) -> None:
    """Insert one ``derived`` row per new column into cleaning_summary.csv.

    ``detail`` names the column (and source when known); ``criterion`` carries
    the derivation ``reason`` so the cleaning story shows why each column exists.

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

    meta: dict[str, dict[str, str]] = {}
    if derivation_log is not None and not derivation_log.empty:
        for _, entry in derivation_log.iterrows():
            name = str(entry.get("derivation", "")).strip()
            if not name:
                continue
            meta[name] = {
                "source": str(entry.get("source", "") or "").strip(),
                "criterion": (str(entry.get("rule", "") or "").strip()
                              or str(entry.get("reason", "") or "").strip()),
            }

    base_cols = n_columns - len(created_cols)
    rows: list[dict] = []
    for i, col in enumerate(created_cols):
        row = {c: "" for c in summary.columns}
        row["step"] = "derived"
        info = meta.get(col, {})
        source = info.get("source", "")
        if "detail" in row:
            row["detail"] = (
                f"added {col} ← {source}" if source else f"added {col}"
            )
        if "n_rows" in row:
            row["n_rows"] = n_rows
        if "n_columns" in row:
            row["n_columns"] = base_cols + i + 1
        if "criterion" in row:
            row["criterion"] = info.get("criterion", "")
        rows.append(row)

    new_df = pd.DataFrame(rows, columns=summary.columns)
    if (summary["step"] == "final").any():
        pos = summary.index.get_loc(summary.index[summary["step"] == "final"][0])
        if "n_columns" in summary.columns:
            summary.loc[summary["step"] == "final", "n_columns"] = n_columns
        summary = pd.concat(
            [summary.iloc[:pos], new_df, summary.iloc[pos:]], ignore_index=True)
    else:
        summary = pd.concat([summary, new_df], ignore_index=True)

    summary.to_csv(summary_path, index=False)


def apply_post_mice_derivations(
    frame: pd.DataFrame,
    schema: dict[str, ColSpec],
    derivations: list,
    ) -> pd.DataFrame:
    """Recreate derived columns from imputed sources after R MICE."""
    out, _schema, _log = apply_derivations(
        frame, schema, derivations, preview=False,
    )
    return out
