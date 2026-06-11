"""Guess what each column is, print a dict you edit in the notebook.

infer_schema → print_schema_template → tweak → cleaning.apply_schema.
ColSpec is the object everything downstream actually reads.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional
import re

import numpy as np
import pandas as pd


Kind = Literal[
    "id", "binary", "ordinal", "nominal",
    "continuous", "count", "datetime", "text", "skip",
]

DatetimeBin = Literal["year", "month", "day", "hour", "full"]


@dataclass
class ColSpec:
    """One column — kind, levels, null markers, keep/skip."""
    name: str
    kind: Kind
    # Optional declared ordering for ordinal kinds (low -> high).
    ordered_levels: Optional[list[Any]] = None
    # Values to treat as null (in addition to NaN/None).
    nulls: tuple[Any, ...] = ()
    # Value remapping applied before type coercion.
    replace: dict[Any, Any] = field(default_factory=dict)
    # Keep in analysis output (False = computed but hidden, e.g. raw IDs).
    keep: bool = True
    # For kind='datetime': bin in place. ``full`` = h/m/s when present, else day-only.
    datetime_bin: Optional[DatetimeBin] = None
    # Free-text note for the schema printout.
    note: str = ""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_BINARY_TRUE = {"true", "t", "yes", "y", "1", "1.0", "positive", "pos"}
_BINARY_FALSE = {"false", "f", "no", "n", "0", "0.0", "negative", "neg"}


def _looks_binary(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    if len(vals) != 2:
        return False
    as_str = {str(v).strip().lower() for v in vals}
    if as_str <= (_BINARY_TRUE | _BINARY_FALSE):
        return True
    return set(vals) <= {0, 1} or set(vals) <= {True, False}


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _looks_id(s: pd.Series, n_rows: int) -> bool:
    nu = s.nunique(dropna=True)
    if nu < 0.95 * n_rows:
        return False
    name = (s.name or "").lower()
    return bool(re.search(r"(^|_)(id|pk|uuid|guid|code|nr|num)(_|$)", name))


def _infer_one(s: pd.Series, n_rows: int, ordinal_max_levels: int) -> Kind:
    nn = s.dropna()
    if nn.empty:
        return "skip"
    if _looks_id(s, n_rows):
        return "id"
    if _looks_datetime(s):
        return "datetime"
    if _looks_binary(s):
        return "binary"

    nu = nn.nunique()

    # Categorical dtype with declared order -> ordinal.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "ordinal" if s.dtype.ordered else "nominal"

    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or (nn % 1 == 0).all()
        if is_int and nu <= ordinal_max_levels and nn.min() >= 0:
            # Small non-negative integer set -> likely ordinal/count.
            return "ordinal" if nu <= 10 else "count"
        return "continuous"

    # Object / string
    if nu <= ordinal_max_levels:
        return "nominal"
    return "text"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_schema(
    df: pd.DataFrame,
    *,
    ordinal_max_levels: int = 15,
    overrides: dict[str, Kind] | None = None,
) -> dict[str, ColSpec]:
    """Return {col_name: ColSpec} with inferred kind for every column."""
    overrides = overrides or {}
    out: dict[str, ColSpec] = {}
    n_rows = len(df)
    for col in df.columns:
        kind = overrides.get(col) or _infer_one(df[col], n_rows, ordinal_max_levels)
        spec = ColSpec(name=col, kind=kind)
        if kind == "ordinal":
            try:
                spec.ordered_levels = sorted(df[col].dropna().unique().tolist())
            except TypeError:
                spec.ordered_levels = list(df[col].dropna().unique())
        out[col] = spec
    return out


def print_schema_template(schema: dict[str, ColSpec]) -> None:
    """Print a paste-back-able Python dict literal of the schema."""
    lines = ["schema_overrides = {"]
    for col, spec in schema.items():
        extras = []
        if spec.ordered_levels is not None and spec.kind == "ordinal":
            extras.append(f"ordered_levels={spec.ordered_levels!r}")
        if spec.nulls:
            extras.append(f"nulls={spec.nulls!r}")
        if spec.replace:
            extras.append(f"replace={spec.replace!r}")
        if not spec.keep:
            extras.append("keep=False")
        if spec.datetime_bin:
            extras.append(f"datetime_bin={spec.datetime_bin!r}")
        extras_str = (", " + ", ".join(extras)) if extras else ""
        lines.append(f'    {col!r}: ColSpec(name={col!r}, kind={spec.kind!r}{extras_str}),')
    lines.append("}")
    print("\n".join(lines))


def _fmt_val(v: Any) -> str:
    if pd.isna(v):
        return "∅"
    return repr(v)


def _sort_vals(vals) -> list:
    def key(v):
        if pd.isna(v):
            return (1, "")
        return (0, str(v))
    return sorted(vals, key=key)


def print_column_uniques(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    max_levels: int = 25,
) -> None:
    """Print per-column unique values to help fill nulls= and replace= in schema_overrides."""
    print("📋 Column uniques — for nulls=() and replace={} in schema_overrides below\n")

    for col in df.columns:
        spec = schema.get(col)
        kind = spec.kind if spec else "?"
        s = df[col]
        n_miss = int(s.isna().sum())

        print(f"▸ {col} · {kind}")

        if kind == "continuous":
            nn = s.dropna()
            if nn.empty:
                print("  · (all missing)\n")
                continue
            print(f"  · {nn.nunique()} unique · {nn.min()} … {nn.max()}")
            if n_miss:
                print(f"  · ∅ → {n_miss}")
            print()
            continue

        if kind in ("id", "text", "datetime"):
            nu = s.nunique(dropna=True)
            print(f"  · {nu} unique", end="")
            if n_miss:
                print(f" · ∅ → {n_miss}", end="")
            print()
            top_n = min(8, nu + (1 if n_miss else 0))
            for val, cnt in s.value_counts(dropna=False).head(top_n).items():
                print(f"  · {_fmt_val(val)} → {int(cnt)}")
            if nu > 8:
                print(f"  · … {nu - 8} more values")
            print()
            continue

        vals = _sort_vals(s.unique())
        if len(vals) > max_levels:
            head, tail = vals[:max_levels], vals[max_levels:]
            for val in head:
                cnt = int(s.isna().sum()) if pd.isna(val) else int((s == val).sum())
                print(f"  · {_fmt_val(val)} → {cnt}")
            print(f"  · … {len(tail)} more levels")
        else:
            for val in vals:
                cnt = int(s.isna().sum()) if pd.isna(val) else int((s == val).sum())
                print(f"  · {_fmt_val(val)} → {cnt}")
        print()


def _levels_for_spec(spec: ColSpec) -> list[Any] | None:
    """Levels for schema summary / report (ordinal order or nominal categories)."""
    if spec.kind == "ordinal":
        return list(spec.ordered_levels) if spec.ordered_levels else None
    if spec.kind == "nominal":
        if spec.ordered_levels:
            return list(spec.ordered_levels)
        if spec.replace:
            seen: list[Any] = []
            for val in spec.replace.values():
                if val not in seen:
                    seen.append(val)
            return seen or None
    return None


def schema_summary(schema: dict[str, ColSpec]) -> pd.DataFrame:
    """Tabular view of the schema (for sanity-checking in the notebook)."""
    rows = []
    for col, spec in schema.items():
        rows.append({
            "column": col,
            "kind": spec.kind,
            "keep": spec.keep,
            "datetime_bin": spec.datetime_bin,
            "levels": _levels_for_spec(spec),
            "nulls": list(spec.nulls) if spec.nulls else None,
            "note": spec.note,
        })
    return pd.DataFrame(rows)


def export_schema_summary(
    schema: dict[str, ColSpec],
    output_root: Path | str = "output",
) -> Path:
    """Write ``output/schema/schema_summary.csv`` for the HTML report."""
    from cleaning import format_table_for_csv

    out_dir = Path(output_root) / "schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "schema_summary.csv"
    format_table_for_csv(schema_summary(schema)).to_csv(path, index=False)
    return path
