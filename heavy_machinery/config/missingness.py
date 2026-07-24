"""Missingness policy before MICE.

``StructuralGroup``: shared measurement slots (blank = not measured).
``MnarColumn``: missing may be informative → optional ``*_missing`` flags.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from schema_infer import ColSpec

_LOG_COLUMNS = [
    "policy",
    "name",
    "requested_cols",
    "available_cols",
    "missing_cols",
    "created_cols",
    "schema_action",
    "reason",
    "status",
]


@dataclass
class StructuralGroup:
    """Slot-style columns where NaN means the slot does not exist."""

    name: str
    cols: list[str]
    derive_count_col: str | None = None
    derive_max_col: str | None = None
    skip_raw: bool = True
    reason: str = ""


@dataclass
class MnarColumn:
    """Column whose missingness itself may be informative."""

    col: str
    flag_col: str | None = None
    reason: str = ""


def _copy_schema(schema: dict[str, ColSpec]) -> dict[str, ColSpec]:
    return {k: replace(v) for k, v in schema.items()}


def _log_entry(**kwargs) -> dict:
    return {col: kwargs.get(col, "") for col in _LOG_COLUMNS}


def _apply_structural_group(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    group: StructuralGroup,
    log: list[dict],
) -> pd.DataFrame:
    available = [c for c in group.cols if c in df.columns]
    missing = [c for c in group.cols if c not in df.columns]
    created: list[str] = []
    schema_actions: list[str] = []

    if not available:
        log.append(_log_entry(
            policy="structural_group",
            name=group.name,
            requested_cols=", ".join(group.cols),
            missing_cols=", ".join(missing),
            reason=group.reason,
            status="skipped_no_available_cols",
        ))
        return df

    if group.derive_count_col:
        df[group.derive_count_col] = df[available].notna().sum(axis=1).astype("int8")
        created.append(group.derive_count_col)
        levels = sorted(df[group.derive_count_col].dropna().unique().tolist())
        schema[group.derive_count_col] = ColSpec(
            name=group.derive_count_col,
            kind="ordinal",
            ordered_levels=levels,
            note=f"structural count over {available}",
        )
        schema_actions.append(f"added ordinal ColSpec for {group.derive_count_col}")

    if group.derive_max_col:
        numeric_view = df[available].apply(pd.to_numeric, errors="coerce")
        df[group.derive_max_col] = numeric_view.max(axis=1, skipna=True)
        created.append(group.derive_max_col)
        levels = sorted(df[group.derive_max_col].dropna().unique().tolist())
        schema[group.derive_max_col] = ColSpec(
            name=group.derive_max_col,
            kind="ordinal",
            ordered_levels=levels,
            note=f"structural max over {available}",
        )
        schema_actions.append(f"added ordinal ColSpec for {group.derive_max_col}")

    if group.skip_raw:
        skipped: list[str] = []
        for col in available:
            if col not in schema:
                continue
            note = schema[col].note or ""
            schema[col] = replace(
                schema[col],
                kind="skip",
                note=f"{note} [structural-missing, derived above]".strip(),
            )
            skipped.append(col)
        if skipped:
            schema_actions.append(f"marked skip: {', '.join(skipped)}")

    log.append(_log_entry(
        policy="structural_group",
        name=group.name,
        requested_cols=", ".join(group.cols),
        available_cols=", ".join(available),
        missing_cols=", ".join(missing),
        created_cols=", ".join(created),
        schema_action="; ".join(schema_actions) if schema_actions else "none",
        reason=group.reason,
        status="applied",
    ))
    return df


def _apply_mnar_column(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    mnar: MnarColumn,
    log: list[dict],
) -> pd.DataFrame:
    flag_col = mnar.flag_col or f"{mnar.col}_missing"

    if mnar.col not in df.columns:
        log.append(_log_entry(
            policy="mnar_column",
            name=mnar.col,
            requested_cols=mnar.col,
            missing_cols=mnar.col,
            reason=mnar.reason,
            status="skipped_col_missing",
        ))
        return df

    df[flag_col] = df[mnar.col].isna().astype("int8")
    schema[flag_col] = ColSpec(
        name=flag_col,
        kind="binary",
        note=f"MNAR flag for {mnar.col}",
    )

    log.append(_log_entry(
        policy="mnar_column",
        name=mnar.col,
        requested_cols=mnar.col,
        available_cols=mnar.col,
        created_cols=flag_col,
        schema_action=f"added binary ColSpec for {flag_col}",
        reason=mnar.reason,
        status="applied",
    ))
    return df


def apply_missingness_policy(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    structural_groups: list[StructuralGroup],
    mnar_columns: list[MnarColumn],
) -> tuple[pd.DataFrame, dict[str, ColSpec], pd.DataFrame]:
    """Apply structural and MNAR decisions; return updated df, schema, and audit log."""
    out = df.copy()
    out_schema = _copy_schema(schema)
    log: list[dict] = []

    for group in structural_groups:
        out = _apply_structural_group(out, out_schema, group, log)

    for mnar in mnar_columns:
        out = _apply_mnar_column(out, out_schema, mnar, log)

    missingness_log = (
        pd.DataFrame(log, columns=_LOG_COLUMNS)
        if log
        else pd.DataFrame(columns=_LOG_COLUMNS)
    )
    return out, out_schema, missingness_log
