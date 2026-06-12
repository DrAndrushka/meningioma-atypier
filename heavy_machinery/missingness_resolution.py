"""Missingness audit, MICE imputation (m=10), and modelling dataset export.

``simple_impute`` provides a single-frame fill for EDA; binary imaging columns
stay NaN unless explicitly allowed (missing ≠ confirmed absent).
Artifacts → ``output/missingness/``, ``output/datasets/``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from schema_infer import ColSpec


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "missingness" / "figures"
    tabs = root / "missingness" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


# ---------------------------------------------------------------------------
# 1. Missingness analysis
# ---------------------------------------------------------------------------

def analyze_missingness(df: pd.DataFrame, *, output_root: Path | str = "output") -> pd.DataFrame:
    """
    Per-column missing summary + co-missingness heatmap (saved as SVG).
    Returns the per-column table.
    """
    figs, tabs = _ensure_dirs(Path(output_root))

    miss = df.isna()
    per_col = pd.DataFrame({
        "column": df.columns,
        "n_missing": miss.sum().values,
        "pct_missing": (miss.mean() * 100).round(2).values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    per_col.to_csv(tabs / "missing_per_column.csv", index=False)

    # Bar chart
    if (per_col["pct_missing"] > 0).any():
        plot_df = per_col[per_col["pct_missing"] > 0]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_df))))
        sns.barplot(x="pct_missing", y="column", data=plot_df, ax=ax, color="#e76f51")
        ax.set_title("Missing % per column"); ax.set_xlabel("% missing")
        fig.tight_layout()
        fig.savefig(figs / "missing_per_column.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    # Co-missingness heatmap (Jaccard over missing rows)
    cols_with_miss = per_col[per_col["pct_missing"] > 0]["column"].tolist()
    if len(cols_with_miss) >= 2:
        m = miss[cols_with_miss].astype(int)
        inter = m.T @ m
        union = (m.values[:, :, None] | m.values[:, None, :]).sum(axis=0)
        jacc = pd.DataFrame(
            np.where(union > 0, inter.values / np.where(union == 0, 1, union), 0),
            index=cols_with_miss, columns=cols_with_miss,
        )
        jacc.to_csv(tabs / "co_missingness_jaccard.csv")
        fig, ax = plt.subplots(figsize=(0.6 * len(cols_with_miss) + 2,
                                        0.6 * len(cols_with_miss) + 2))
        # Lower triangle only — hide upper mirror and diagonal (self = always 1.00).
        tri_mask = np.triu(np.ones_like(jacc.values, dtype=bool), k=0)
        sns.heatmap(jacc, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar=True, mask=tri_mask)
        ax.set_title("Co-missingness (Jaccard)")
        fig.tight_layout()
        fig.savefig(figs / "co_missingness_heatmap.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    return per_col


# ---------------------------------------------------------------------------
# 2a. Structural-missing handling (NOT to be imputed)
# ---------------------------------------------------------------------------

def mark_structural_missing(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    groups: dict[str, dict],
) -> pd.DataFrame:
    """
    Tell the pipeline which NaNs are *structural* (the value does not exist —
    e.g. lesion_2_MRI_PIRADS is NaN because the MRI showed only one lesion)
    rather than truly missing.

    Structural columns are marked ``kind='skip'`` so they are excluded from
    MICE imputation, EDA screening, and the multivariable model. In their
    place this helper can derive two real features per group:

      - ``<group_name>``      : count of non-null columns (e.g. n_lesions=0..3)
      - ``<group_name>_max``  : max value across columns (the "dominant" item;
                                only sensible when the columns are ordinal/numeric)

    Parameters
    ----------
    groups : dict like ::

        {
          'n_lesions_MRI': {
              'cols':         ['lesion_1_MRI_PIRADS',
                               'lesion_2_MRI_PIRADS',
                               'lesion_3_MRI_PIRADS'],
              'derive_count': True,
              'derive_max':   True,
              'count_levels': [0, 1, 2, 3],
              'max_levels':   [1, 2, 3, 4, 5],
              'skip_after':   ['lesion_2_MRI_PIRADS',
                               'lesion_3_MRI_PIRADS'],
          },
        }

    Returns
    -------
    DataFrame with derived columns appended. ``schema`` is mutated in place
    (new ColSpecs added; ``skip_after`` columns flipped to ``kind='skip'``).
    """
    out = df.copy()
    for group_name, cfg in groups.items():
        cols = [c for c in cfg.get('cols', []) if c in out.columns]
        if not cols:
            continue

        if cfg.get('derive_count', True):
            out[group_name] = out[cols].notna().sum(axis=1).astype('Int64')
            levels = cfg.get('count_levels')
            schema[group_name] = ColSpec(
                name=group_name,
                kind='ordinal',
                ordered_levels=levels if levels is not None
                else sorted(out[group_name].dropna().unique().tolist()),
                note=f'structural count over {cols}',
            )

        if cfg.get('derive_max', True):
            max_name = f"{group_name}_max"
            try:
                numeric_view = out[cols].apply(pd.to_numeric, errors='coerce')
                out[max_name] = numeric_view.max(axis=1)
                levels = cfg.get('max_levels')
                schema[max_name] = ColSpec(
                    name=max_name,
                    kind='ordinal',
                    ordered_levels=levels if levels is not None
                    else sorted(out[max_name].dropna().unique().tolist()),
                    note=f'structural max over {cols}',
                )
            except Exception:
                pass  # non-numeric group — skip the max

        for c in cfg.get('skip_after', []):
            if c in schema:
                schema[c].kind = 'skip'
                schema[c].note = (schema[c].note or '') + ' [structural-missing, derived above]'

    return out


# ---------------------------------------------------------------------------
# 2b. Missing flags (for true MNAR columns)
# ---------------------------------------------------------------------------

def add_missing_flags(
    df: pd.DataFrame,
    cols: Sequence[str],
    schema: dict[str, ColSpec] | None = None,
) -> pd.DataFrame:
    """
    Add <col>_missing boolean columns for true-MNAR columns.
    If ``schema`` is passed, the new flag columns are registered as binary
    ColSpecs automatically.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        flag = f"{c}_missing"
        out[flag] = out[c].isna().astype("boolean")
        if schema is not None and flag not in schema:
            schema[flag] = ColSpec(name=flag, kind='binary',
                                   note=f'MNAR flag for {c}')
    return out


def drop_rows(
    df: pd.DataFrame,
    *,
    mask: "pd.Series | None" = None,
    where: "str | None" = None,
    reason: str = '',
    log: "list | None" = None,
) -> pd.DataFrame:
    """
    Remove rows from the dataframe with an audit trail.

    Provide EITHER a boolean ``mask`` (True = drop) OR a pandas ``query`` string
    in ``where`` (rows matching the query are dropped).

    If ``log`` is passed (a list), an entry is appended describing the drop —
    useful for reproducibility / methods-section reporting.

    Example::

        drop_log = []
        df = drop_rows(df, where='age < 18', reason='paediatric record',
                       log=drop_log)
        df = drop_rows(df, mask=df['preop_PSA'] < 0,
                       reason='negative PSA = data entry error',
                       log=drop_log)
        pd.DataFrame(drop_log)
    """
    if mask is None and where is None:
        raise ValueError("Pass either mask= or where=")
    if where is not None:
        drop_mask = df.eval(where)
    else:
        drop_mask = mask.reindex(df.index, fill_value=False).astype(bool)

    n_before = len(df)
    out = df.loc[~drop_mask].copy()
    n_after = len(out)
    if log is not None:
        log.append({
            'reason': reason,
            'criterion': where if where is not None else 'mask',
            'n_before': n_before,
            'n_dropped': n_before - n_after,
            'n_remaining': n_after,
        })
    return out


# ---------------------------------------------------------------------------
# 3. MICE imputation (multiple imputation)
# ---------------------------------------------------------------------------

def _macos_on_battery() -> bool:
    """
    Return True if running on macOS and currently on battery power.
    If detection fails, return False.
    """
    try:
        if platform.system() != "Darwin":
            return False

        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        text = (result.stdout or "").lower()
        return "battery power" in text

    except Exception:
        return False


def _apply_mice_slot_guardrail(
    n_jobs_imputations: int,
    n_jobs_rf: int,
    slot_limit: int,
) -> tuple[int, int]:
    """Reduce RF / imputation parallelism so total worker slots fit within limit."""
    estimated = n_jobs_imputations * n_jobs_rf
    if estimated <= slot_limit:
        return n_jobs_imputations, n_jobs_rf

    n_jobs_rf = max(1, slot_limit // n_jobs_imputations)
    estimated = n_jobs_imputations * n_jobs_rf
    if estimated > slot_limit:
        n_jobs_imputations = max(1, slot_limit // max(1, n_jobs_rf))

    return n_jobs_imputations, n_jobs_rf


def _print_macos_battery_safety_warning(max_iter: int, n_estimators: int) -> None:
    """Warn when a heavy MICE config runs on macOS battery."""
    if max_iter <= 25 and n_estimators <= 50:
        return
    print(
        "⚠️  macOS battery safety warning:\n"
        "You are running a heavy MICE configuration on battery.\n"
        "Recommended maximum on battery: max_iter <= 25 and n_estimators <= 50.\n"
        "Consider plugging in or using emergency_safe_mode=True.",
        flush=True,
    )


def choose_mice_parallel_settings(
    m: int,
    os_profile: str = "auto",
    safety_margin_cores: int = 2,
    max_worker_slots: int | None = None,
    emergency_safe_mode: bool = False,
) -> dict:
    """Choose conservative MICE parallelism based on OS and CPU availability."""
    del os_profile  # reserved; OS is detected via platform.system() only

    system = platform.system()
    on_battery = _macos_on_battery()

    if emergency_safe_mode:
        n_jobs_imputations = 1
        n_jobs_rf = 1
        backend = "loky"
    elif system == "Windows":
        n_jobs_imputations = min(m, 4)
        n_jobs_rf = 3
        backend = "loky"
    elif system == "Darwin":
        if on_battery:
            n_jobs_imputations = min(m, 2)
            n_jobs_rf = 1
        else:
            n_jobs_imputations = min(m, 3)
            n_jobs_rf = 2
        backend = "loky"
    else:
        n_jobs_imputations = min(m, 2)
        n_jobs_rf = 1
        backend = "loky"

    cpu_count = os.cpu_count() or 1
    available_cores = max(1, cpu_count - safety_margin_cores)

    effective_max_worker_slots = max_worker_slots
    if effective_max_worker_slots is None:
        if system == "Windows":
            effective_max_worker_slots = 12
        elif system == "Darwin":
            effective_max_worker_slots = 2 if on_battery else 6
        else:
            effective_max_worker_slots = 2

    if not emergency_safe_mode:
        n_jobs_imputations, n_jobs_rf = _apply_mice_slot_guardrail(
            n_jobs_imputations, n_jobs_rf, available_cores,
        )
        if effective_max_worker_slots is not None:
            n_jobs_imputations, n_jobs_rf = _apply_mice_slot_guardrail(
                n_jobs_imputations, n_jobs_rf, effective_max_worker_slots,
            )

    estimated_worker_slots = n_jobs_imputations * n_jobs_rf

    return {
        "os_detected": system,
        "macos_on_battery": on_battery,
        "cpu_count": cpu_count,
        "available_cores": available_cores,
        "m": m,
        "n_jobs_imputations": n_jobs_imputations,
        "n_jobs_rf": n_jobs_rf,
        "estimated_worker_slots": estimated_worker_slots,
        "max_worker_slots": effective_max_worker_slots,
        "backend": backend,
        "emergency_safe_mode": emergency_safe_mode,
    }


def _encode_for_impute(df: pd.DataFrame, schema: dict[str, ColSpec]):
    """Encode categoricals to integer codes for the imputer; remember mapping."""
    work = df.copy()
    decoders: dict[str, dict[int, object]] = {}
    cat_cols = []
    for col, spec in schema.items():
        if col not in work.columns:
            continue
        if spec.kind in ("ordinal", "nominal"):
            cats = pd.Categorical(work[col])
            decoders[col] = dict(enumerate(cats.categories))
            work[col] = pd.Series(cats.codes, index=work.index).replace(-1, np.nan)
            cat_cols.append(col)
        elif spec.kind == "binary":
            work[col] = work[col].astype("float")
            cat_cols.append(col)
    # keep only numeric / coded columns for imputation
    drop = [c for c, sp in schema.items()
            if sp.kind in ("id", "text", "datetime", "skip") and c in work.columns]
    work = work.drop(columns=drop, errors="ignore")
    return work, decoders, cat_cols, drop


def _decode_after_impute(
    imputed: pd.DataFrame,
    decoders: dict[str, dict[int, object]],
    cat_cols: list[str],
    schema: dict[str, ColSpec],
) -> pd.DataFrame:
    out = imputed.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        spec = schema[col]
        if spec.kind == "binary":
            out[col] = (out[col] >= 0.5).astype("boolean")
        else:
            codes = out[col].round().clip(lower=0, upper=max(decoders[col]) if decoders[col] else 0)
            out[col] = codes.map(decoders[col])
            levels = spec.ordered_levels if spec.kind == "ordinal" else list(decoders[col].values())
            out[col] = pd.Categorical(out[col], categories=levels, ordered=(spec.kind == "ordinal"))
    return out


def _restore_imputed_dtypes(original: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """Recast imputed columns to match original categorical / Float64 dtypes."""
    out = imputed.reindex(columns=original.columns).copy()
    for col in original.columns:
        orig_dtype = original[col].dtype
        if isinstance(orig_dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical(
                out[col],
                categories=orig_dtype.categories,
                ordered=orig_dtype.ordered,
            )
        elif isinstance(orig_dtype, pd.Float64Dtype):
            out[col] = out[col].astype("Float64")
    return out


def _validate_imputed_frames(original: pd.DataFrame, frames: list[pd.DataFrame]) -> None:
    """Post-imputation checks: columns, row count, categorical level integrity."""
    for idx, frame in enumerate(frames):
        assert list(frame.columns) == list(original.columns), (
            f"imputed frame {idx}: column mismatch"
        )
        assert len(frame) == len(original), (
            f"imputed frame {idx}: row count mismatch "
            f"({len(frame)} vs {len(original)})"
        )
        for col in original.columns:
            orig_dtype = original[col].dtype
            if isinstance(orig_dtype, pd.CategoricalDtype):
                allowed = set(orig_dtype.categories)
                observed = set(frame[col].dropna().unique())
                new_levels = observed - allowed
                assert not new_levels, (
                    f"imputed frame {idx}, column {col}: "
                    f"new categorical levels {new_levels}"
                )


def _mice_dataset_dir(output_root: Path | str) -> Path:
    return Path(output_root) / "missingness" / "mice"


def _datasets_dir(output_root: Path | str) -> Path:
    return Path(output_root) / "datasets"


UNIMPUTED_DATASET_NAME = "unimputed_df.parquet"
MICE_MODELING_DATASET_NAME = "mice_imputed_df.parquet"
SIMPLE_MODELING_DATASET_NAME = "simple_imputed_df.parquet"
DATASETS_MANIFEST_NAME = "manifest.json"


def prepare_datasets_dir(output_root: Path | str) -> Path:
    """Reset ``output/datasets/`` (delete if present, then recreate)."""
    datasets_dir = _datasets_dir(output_root)
    if datasets_dir.exists():
        shutil.rmtree(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def _read_datasets_manifest(output_root: Path | str) -> dict[str, Any]:
    manifest_path = _datasets_dir(output_root) / DATASETS_MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_datasets_manifest(output_root: Path | str, manifest: dict[str, Any]) -> None:
    manifest_path = _datasets_dir(output_root) / DATASETS_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def _save_dataset_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    context: str = "",
    dtype_reference: pd.DataFrame | None = None,
) -> None:
    """Write parquet and assert column dtypes survive the roundtrip."""
    label = context or path.name
    if dtype_reference is not None:
        _assert_frame_dtypes_match(
            dtype_reference, df, context=f"{label} pre-save",
        )

    dtype_spec = _dtype_manifest(df)
    df.to_parquet(path, index=False, engine="pyarrow")
    roundtrip = _apply_dtype_manifest(
        pd.read_parquet(path, engine="pyarrow"),
        dtype_spec,
    )
    _assert_frame_dtypes_match(
        df, roundtrip, context=f"{label} parquet roundtrip",
    )


def _load_dataset_parquet(path: Path, dtype_spec: dict[str, Any]) -> pd.DataFrame:
    frame = _apply_dtype_manifest(
        pd.read_parquet(path, engine="pyarrow"),
        dtype_spec,
    )
    template = _dtype_template(list(frame.columns), dtype_spec)
    _assert_frame_dtypes_match(template, frame, context=f"{path.name} post-load")
    return frame


def stage_unimputed_dataset(df: pd.DataFrame, output_root: Path | str) -> Path:
    """Wipe ``output/datasets/`` and write ``unimputed_df.parquet`` (DDA / EDA cohort)."""
    datasets_dir = prepare_datasets_dir(output_root)
    path = datasets_dir / UNIMPUTED_DATASET_NAME
    _save_dataset_parquet(
        df, path, context=UNIMPUTED_DATASET_NAME,
    )
    _write_datasets_manifest(output_root, {
        "saved_at": datetime.now(UTC).isoformat(),
        "unimputed": UNIMPUTED_DATASET_NAME,
        "dtypes": _dtype_manifest(df),
        "imputation_method": None,
        "modeling_dataset": None,
    })
    print(f"💾 Saved unimputed cohort → {path}", flush=True)
    return path


def save_modeling_dataset(
    df: pd.DataFrame,
    output_root: Path | str,
    *,
    method: Literal["mice", "simple"],
) -> Path:
    """Write ``mice_imputed_df.parquet`` or ``simple_imputed_df.parquet`` for modelling."""
    datasets_dir = _datasets_dir(output_root)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        MICE_MODELING_DATASET_NAME if method == "mice"
        else SIMPLE_MODELING_DATASET_NAME
    )
    path = datasets_dir / filename
    _save_dataset_parquet(df, path, context=filename)

    manifest = _read_datasets_manifest(output_root)
    manifest.update({
        "saved_at": datetime.now(UTC).isoformat(),
        "imputation_method": method,
        "modeling_dataset": filename,
        "dtypes": _dtype_manifest(df),
    })
    _write_datasets_manifest(output_root, manifest)
    print(f"💾 Saved {method} modelling cohort → {path}", flush=True)
    return path


def load_unimputed_dataset(output_root: Path | str = "output") -> pd.DataFrame:
    """Load ``output/datasets/unimputed_df.parquet``."""
    datasets_dir = _datasets_dir(output_root)
    path = datasets_dir / UNIMPUTED_DATASET_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No unimputed dataset at {path}. Run imputation staging first."
        )
    manifest = _read_datasets_manifest(output_root)
    dtype_spec = manifest.get("dtypes") or {}
    if not dtype_spec:
        raise FileNotFoundError(
            f"Missing dtype manifest in {datasets_dir / DATASETS_MANIFEST_NAME}."
        )
    return _load_dataset_parquet(path, dtype_spec)


def load_modeling_frames(output_root: Path | str = "output") -> list[pd.DataFrame]:
    """Load imputed cohort(s) for multivariable modelling.

    Prefers full MICE draws from ``output/missingness/mice/`` when present
    (Rubin pooling). Otherwise loads the single modelling parquet from
    ``output/datasets/``.
    """
    mice_manifest = _mice_dataset_dir(output_root) / "manifest.json"
    if mice_manifest.exists():
        return load_imputed_frames(output_root)

    datasets_dir = _datasets_dir(output_root)
    manifest = _read_datasets_manifest(output_root)
    dtype_spec = manifest.get("dtypes") or {}
    modeling_name = manifest.get("modeling_dataset")
    candidates = [modeling_name] if modeling_name else []
    candidates.extend([SIMPLE_MODELING_DATASET_NAME, MICE_MODELING_DATASET_NAME])
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = datasets_dir / name
        if path.exists():
            if not dtype_spec:
                raise FileNotFoundError(
                    f"Missing dtype manifest in {datasets_dir / DATASETS_MANIFEST_NAME}."
                )
            return [_load_dataset_parquet(path, dtype_spec)]

    raise FileNotFoundError(
        f"No modelling dataset in {datasets_dir} or {_mice_dataset_dir(output_root)}. "
        "Run mice_impute() or simple_impute_stage() first."
    )


def _serialize_category(value: Any) -> Any:
    if pd.isna(value):
        return {"__dtype__": "null"}
    if isinstance(value, bool):
        return {"__dtype__": "bool", "value": value}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"__dtype__": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"__dtype__": "float", "value": float(value)}
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return {"__dtype__": "datetime", "value": ts.isoformat()}
    return {"__dtype__": "str", "value": str(value)}


def _deserialize_category(item: Any) -> Any:
    if not isinstance(item, dict) or "__dtype__" not in item:
        return item
    kind = item["__dtype__"]
    if kind == "null":
        return pd.NA
    if kind == "bool":
        return bool(item["value"])
    if kind == "int":
        return int(item["value"])
    if kind == "float":
        return float(item["value"])
    if kind == "datetime":
        return pd.Timestamp(item["value"])
    return str(item["value"])


def _serialize_categories(categories: Sequence[Any]) -> list[Any]:
    return [_serialize_category(c) for c in categories]


def _deserialize_categories(items: Sequence[Any]) -> list[Any]:
    return [_deserialize_category(item) for item in items]


def _datetime_unit(dtype: Any) -> str:
    unit, _count = np.datetime_data(dtype)
    return str(unit)


def _dtype_manifest(df: pd.DataFrame) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    for col in df.columns:
        dt = df[col].dtype
        if isinstance(dt, pd.CategoricalDtype):
            spec[col] = {
                "kind": "categorical",
                "categories": _serialize_categories(dt.categories),
                "ordered": bool(dt.ordered),
            }
        elif dt == "boolean" or str(dt) == "boolean":
            spec[col] = {"kind": "boolean"}
        elif isinstance(dt, pd.Float64Dtype) or str(dt) == "Float64":
            spec[col] = {"kind": "Float64"}
        elif isinstance(dt, pd.Int64Dtype) or str(dt) == "Int64":
            spec[col] = {"kind": "Int64"}
        elif pd.api.types.is_datetime64_any_dtype(dt):
            spec[col] = {"kind": "datetime64", "unit": _datetime_unit(dt)}
        elif pd.api.types.is_string_dtype(dt):
            spec[col] = {"kind": "string"}
        elif dt == np.dtype("bool"):
            spec[col] = {"kind": "boolean"}
        else:
            spec[col] = {"kind": str(dt)}
    return spec


def _apply_dtype_manifest(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, info in spec.items():
        if col not in out.columns:
            continue
        kind = info.get("kind") if isinstance(info, dict) else info
        if kind == "categorical":
            cats = _deserialize_categories(info["categories"])
            out[col] = pd.Categorical(
                out[col],
                categories=cats,
                ordered=bool(info.get("ordered", False)),
            )
        elif kind == "Float64":
            out[col] = out[col].astype("Float64")
        elif kind == "Int64":
            out[col] = out[col].astype("Int64")
        elif kind == "boolean":
            out[col] = out[col].astype("boolean")
        elif kind == "datetime64":
            unit = info.get("unit", "ns")
            out[col] = pd.to_datetime(out[col]).astype(f"datetime64[{unit}]")
        elif kind == "string":
            out[col] = out[col].astype("string")
        elif kind.startswith("datetime64"):
            out[col] = pd.to_datetime(out[col])
        elif kind in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
            out[col] = out[col].astype(kind)
        elif kind == "float64":
            out[col] = out[col].astype("float64")
        elif kind == "object":
            out[col] = out[col].astype("object")
    return out


def _dtypes_equivalent(expected: Any, actual: Any) -> bool:
    if isinstance(expected, pd.CategoricalDtype) and isinstance(actual, pd.CategoricalDtype):
        return (
            expected.ordered == actual.ordered
            and len(expected.categories) == len(actual.categories)
            and all(
                (pd.isna(e) and pd.isna(a)) or e == a
                for e, a in zip(expected.categories, actual.categories)
            )
        )
    if pd.api.types.is_string_dtype(expected) and pd.api.types.is_string_dtype(actual):
        return True
    if expected == np.dtype("bool") or isinstance(expected, pd.BooleanDtype):
        return actual == np.dtype("bool") or isinstance(actual, pd.BooleanDtype)
    if pd.api.types.is_datetime64_any_dtype(expected) and pd.api.types.is_datetime64_any_dtype(actual):
        return _datetime_unit(expected) == _datetime_unit(actual)
    return expected == actual


def _assert_frame_dtypes_match(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    context: str = "",
) -> None:
    """Assert every column dtype matches ``reference`` (for multivariate safety)."""
    prefix = f"{context}: " if context else ""
    assert list(frame.columns) == list(reference.columns), (
        f"{prefix}column order mismatch"
    )
    mismatches: list[str] = []
    for col in reference.columns:
        expected = reference[col].dtype
        actual = frame[col].dtype
        if not _dtypes_equivalent(expected, actual):
            mismatches.append(f"{col}: expected {expected!r}, got {actual!r}")
    assert not mismatches, f"{prefix}dtype mismatch — " + "; ".join(mismatches)


def _dtype_template(columns: Sequence[str], spec: dict[str, Any]) -> pd.DataFrame:
    """Empty frame carrying the manifest dtypes for post-load validation."""
    data: dict[str, pd.Series] = {}
    for col in columns:
        info = spec.get(col)
        if not isinstance(info, dict):
            data[col] = pd.Series([], dtype="object")
            continue
        kind = info.get("kind")
        if kind == "categorical":
            data[col] = pd.Categorical(
                [],
                categories=_deserialize_categories(info["categories"]),
                ordered=bool(info.get("ordered", False)),
            )
        elif kind == "Float64":
            data[col] = pd.Series([], dtype="Float64")
        elif kind == "Int64":
            data[col] = pd.Series([], dtype="Int64")
        elif kind == "boolean":
            data[col] = pd.Series([], dtype="boolean")
        elif kind in ("datetime64",) or str(kind).startswith("datetime64"):
            unit = info.get("unit", "ns")
            data[col] = pd.Series([], dtype=f"datetime64[{unit}]")
        elif kind == "string":
            data[col] = pd.Series([], dtype="string")
        else:
            data[col] = pd.Series([], dtype=kind)
    return pd.DataFrame(data)


def save_imputed_frames(
    frames: Sequence[pd.DataFrame],
    output_root: Path | str = "output",
    *,
    source_df: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write MICE draws to ``output/missingness/mice/imputed_*.parquet``."""
    if not frames:
        raise ValueError("save_imputed_frames: no frames to save")

    mice_dir = _mice_dataset_dir(output_root)
    mice_dir.mkdir(parents=True, exist_ok=True)

    for old in mice_dir.glob("imputed_*.parquet"):
        old.unlink()

    reference = source_df if source_df is not None else frames[0]
    dtype_spec = _dtype_manifest(reference)
    frame_names: list[str] = []
    for i, frame in enumerate(frames, start=1):
        name = f"imputed_{i:03d}.parquet"
        path = mice_dir / name
        _save_dataset_parquet(
            frame,
            path,
            context=name,
            dtype_reference=reference,
        )
        frame_names.append(name)

    manifest: dict[str, Any] = {
        "saved_at": datetime.now(UTC).isoformat(),
        "m": len(frames),
        "n_rows": len(frames[0]),
        "columns": list(frames[0].columns),
        "frames": frame_names,
        "dtypes": dtype_spec,
    }
    if metadata:
        manifest.update(metadata)

    (mice_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"💾 Saved {len(frames)} imputed frames → {mice_dir}", flush=True)
    return mice_dir


def load_imputed_frames(output_root: Path | str = "output") -> list[pd.DataFrame]:
    """Load MICE draws saved by ``mice_impute`` / ``save_imputed_frames``."""
    mice_dir = _mice_dataset_dir(output_root)
    manifest_path = mice_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_names = manifest.get("frames") or []
        dtype_spec = manifest.get("dtypes") or {}
        columns = manifest.get("columns") or []
    else:
        frame_names = sorted(p.name for p in mice_dir.glob("imputed_*.parquet"))
        dtype_spec = {}
        columns = []

    if not frame_names:
        raise FileNotFoundError(
            f"No MICE parquet dataset at {mice_dir}. Run mice_impute() first."
        )
    if not dtype_spec:
        raise FileNotFoundError(
            f"Missing dtype manifest at {manifest_path}. Re-run mice_impute()."
        )

    dtype_template = _dtype_template(columns or list(dtype_spec), dtype_spec)
    frames: list[pd.DataFrame] = []
    for name in frame_names:
        path = mice_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing imputed frame: {path}")
        frame = _apply_dtype_manifest(
            pd.read_parquet(path, engine="pyarrow"),
            dtype_spec,
        )
        _assert_frame_dtypes_match(dtype_template, frame, context=f"{name} post-load")
        frames.append(frame)

    return frames


def _format_elapsed(seconds: float) -> str:
    """Human-readable duration for progress logs."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def _run_single_mice_imputation(
    i: int,
    work: pd.DataFrame,
    schema: dict[str, ColSpec],
    df: pd.DataFrame,
    decoders: dict[str, dict[int, object]],
    cat_cols: list[str],
    dropped: list[str],
    random_state: int,
    max_iter: int,
    n_estimators: int,
    n_jobs_rf: int,
    m_total: int,
    suppress_convergence_warnings: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Run one MICE imputation draw; returns (decoded frame, convergence_warned)."""
    draw = i + 1
    t0 = time.perf_counter()
    print(
        f"🔄 Imputation {draw}/{m_total} — "
        f"IterativeImputer (max_iter={max_iter}, n_estimators={n_estimators}, "
        f"seed={random_state + i})…",
        flush=True,
    )
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs_rf,
            random_state=random_state + i,
        ),
        max_iter=max_iter,
        sample_posterior=False,  # RF estimator doesn't support posterior
        random_state=random_state + i,
    )
    with warnings.catch_warnings(record=True) as caught:
        if suppress_convergence_warnings:
            warnings.simplefilter("ignore", category=ConvergenceWarning)
        arr = imputer.fit_transform(work)
    convergence_warned = any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    imp = pd.DataFrame(arr, columns=work.columns, index=work.index)
    decoded = _decode_after_impute(imp, decoders, cat_cols, schema)
    for c in dropped:
        if c in df.columns:
            decoded[c] = df[c].values
    decoded = decoded.reindex(columns=df.columns)
    elapsed = time.perf_counter() - t0
    print(f"✅ Imputation {draw}/{m_total} done ({_format_elapsed(elapsed)})", flush=True)
    return decoded, convergence_warned


def mice_impute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    m: int = 10,
    max_iter: int = 10,
    random_state: int = 42,
    output_root: Path | str = "output",
    os_profile: str = "auto",
    n_jobs_imputations: int | None = None,
    n_jobs_rf: int | None = None,
    n_estimators: int = 50,
    safety_margin_cores: int = 2,
    max_worker_slots: int | None = None,
    emergency_safe_mode: bool = False,
    enforce_macos_battery_safety: bool = True,
    suppress_convergence_warnings: bool = True,
    save_imputed: bool = True,
) -> list[pd.DataFrame]:
    """
    Generate `m` imputed datasets via sklearn IterativeImputer with different
    random seeds (sample_posterior=True). Returns a list of m DataFrames with
    the same columns as the original (datetime/id/text columns are passed
    through unchanged from the original df).

    Parallelism notes:
    - Windows uses conservative parallelism because process spawning and nested
      parallelism can be expensive.
    - macOS is intentionally conservative to protect laptop thermals and battery.
    - If macOS is detected on battery power, a much safer low-power profile is used.
    - Imputations run in parallel; random forest parallelism is capped to avoid
      CPU oversubscription.
    - ``enforce_macos_battery_safety`` (default True) caps worker slots to 2 on
      macOS battery and warns on heavy ``max_iter`` / ``n_estimators``.
    - ``suppress_convergence_warnings`` (default True) hides only
      ``ConvergenceWarning`` per draw; a post-run summary still reports counts.
      Pass ``False`` to surface raw sklearn convergence warnings.
    """
    stage_unimputed_dataset(df, output_root)
    figs, tabs = _ensure_dirs(Path(output_root))

    work, decoders, cat_cols, dropped = _encode_for_impute(df, schema)
    mice_meta = {
        "max_iter": max_iter,
        "random_state": random_state,
        "n_estimators": n_estimators,
    }

    if work.isna().sum().sum() == 0:
        # nothing to impute -> return m copies
        print(f"✨ No missing values — returning {m} identical copies (no MICE run).")
        imputed_frames = [_restore_imputed_dtypes(df, df.copy()) for _ in range(m)]
        _validate_imputed_frames(df, imputed_frames)
        pd.DataFrame([{"m": m, **mice_meta}]).to_csv(
            tabs / "mice_config.csv", index=False,
        )
        if save_imputed:
            save_imputed_frames(
                imputed_frames,
                output_root,
                source_df=imputed_frames[0],
                metadata={"m": m, **mice_meta},
            )
        save_modeling_dataset(imputed_frames[0], output_root, method="mice")
        return imputed_frames

    t_start = time.perf_counter()
    n_missing = int(work.isna().sum().sum())
    print(
        f"🧊 MICE starting — {m} imputations, {n_missing:,} coded missing cells "
        f"across {work.shape[1]} columns × {work.shape[0]} rows"
    )

    settings = choose_mice_parallel_settings(
        m=m,
        os_profile=os_profile,
        safety_margin_cores=safety_margin_cores,
        max_worker_slots=max_worker_slots,
        emergency_safe_mode=emergency_safe_mode,
    )

    if (
        settings["os_detected"] == "Darwin"
        and settings["macos_on_battery"]
    ):
        _print_macos_battery_safety_warning(max_iter, n_estimators)

    if emergency_safe_mode:
        settings["n_jobs_imputations"] = 1
        settings["n_jobs_rf"] = 1
    else:
        if n_jobs_imputations is not None:
            settings["n_jobs_imputations"] = n_jobs_imputations
        if n_jobs_rf is not None:
            settings["n_jobs_rf"] = n_jobs_rf

    if (
        enforce_macos_battery_safety
        and settings["os_detected"] == "Darwin"
        and settings["macos_on_battery"]
        and not emergency_safe_mode
    ):
        settings["max_worker_slots"] = 2
        if max_iter > 25:
            print(
                "⚠️  macOS battery safety (enforce_macos_battery_safety=True): "
                f"max_iter={max_iter} exceeds recommended 25 on battery.",
                flush=True,
            )
        if n_estimators > 50:
            print(
                "⚠️  macOS battery safety (enforce_macos_battery_safety=True): "
                f"n_estimators={n_estimators} exceeds recommended 50 on battery.",
                flush=True,
            )

    if not emergency_safe_mode:
        settings["n_jobs_imputations"], settings["n_jobs_rf"] = _apply_mice_slot_guardrail(
            settings["n_jobs_imputations"],
            settings["n_jobs_rf"],
            settings["available_cores"],
        )
        if settings["max_worker_slots"] is not None:
            settings["n_jobs_imputations"], settings["n_jobs_rf"] = _apply_mice_slot_guardrail(
                settings["n_jobs_imputations"],
                settings["n_jobs_rf"],
                settings["max_worker_slots"],
            )

    settings["estimated_worker_slots"] = (
        settings["n_jobs_imputations"] * settings["n_jobs_rf"]
    )

    print(
        "⚙️  MICE parallel settings:\n"
        f"   🖥️  OS detected: {settings['os_detected']}\n"
        f"   🔋 macOS on battery: {settings['macos_on_battery']}\n"
        f"   🧮 CPU count: {settings['cpu_count']}\n"
        f"   🛡️  available cores (after safety margin): {settings['available_cores']}\n"
        f"   📦 m imputations: {settings['m']}\n"
        f"   🔀 n_jobs_imputations: {settings['n_jobs_imputations']}\n"
        f"   🌲 n_jobs_rf: {settings['n_jobs_rf']}\n"
        f"   🎰 estimated worker slots: {settings['estimated_worker_slots']}\n"
        f"   🚦 max_worker_slots: {settings['max_worker_slots']}\n"
        f"   🔧 backend: {settings['backend']}\n"
        f"   🆘 emergency_safe_mode: {settings['emergency_safe_mode']}"
    )

    impute_kwargs = dict(
        work=work,
        schema=schema,
        df=df,
        decoders=decoders,
        cat_cols=cat_cols,
        dropped=dropped,
        random_state=random_state,
        max_iter=max_iter,
        n_estimators=n_estimators,
        n_jobs_rf=settings["n_jobs_rf"],
        m_total=m,
        suppress_convergence_warnings=suppress_convergence_warnings,
    )

    if settings["n_jobs_imputations"] == 1:
        print(f"🐢 Running {m} imputations serially (safest / emergency mode)…")
    else:
        print(
            f"🚀 Launching {m} imputations across "
            f"{settings['n_jobs_imputations']} parallel workers…"
        )

    if settings["n_jobs_imputations"] == 1:
        run_results = [
            _run_single_mice_imputation(i=i, **impute_kwargs)
            for i in range(m)
        ]
    else:
        run_results = Parallel(
            n_jobs=settings["n_jobs_imputations"],
            backend=settings["backend"],
        )(
            delayed(_run_single_mice_imputation)(i=i, **impute_kwargs)
            for i in range(m)
        )

    imputed_frames = [frame for frame, _ in run_results]
    n_convergence_warnings = sum(1 for _, warned in run_results if warned)
    if n_convergence_warnings > 0:
        print(
            f"⚠️  ConvergenceWarning occurred in "
            f"{n_convergence_warnings}/{m} imputations.\n"
            "Meaning: IterativeImputer reached max_iter before early-stopping "
            "criterion was satisfied.\n"
            "This is common with RandomForestRegressor-based MICE.\n"
            "Check downstream stability rather than chasing max_iter blindly.",
            flush=True,
        )

    elapsed = time.perf_counter() - t_start
    nan_first = int(imputed_frames[0].isna().sum().sum())
    print(
        f"🏁 MICE complete — {len(imputed_frames)} frames in "
        f"{_format_elapsed(elapsed)} "
        f"({_format_elapsed(elapsed / m)} avg per draw)"
    )
    print(f"📊 NaN count in first imputed frame: {nan_first}")

    imputed_frames = [
        _restore_imputed_dtypes(df, frame) for frame in imputed_frames
    ]
    _validate_imputed_frames(df, imputed_frames)

    pd.DataFrame([{"m": m, **mice_meta}]).to_csv(
        tabs / "mice_config.csv", index=False,
    )
    if save_imputed:
        save_imputed_frames(
            imputed_frames,
            output_root,
            source_df=imputed_frames[0],
            metadata={"m": m, **mice_meta},
        )
    save_modeling_dataset(imputed_frames[0], output_root, method="mice")
    return imputed_frames


# ---------------------------------------------------------------------------
# 4. Simple imputation (fallback / screening)
# ---------------------------------------------------------------------------

def simple_impute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    impute_binary: bool = False,
) -> pd.DataFrame:
    """Single-frame imputation for screening runs.

    - continuous/count → median
    - ordinal → mode of the declared category level
    - nominal → mode
    - binary → left as NaN unless ``impute_binary=True`` (then mode; legacy)
    """
    out = df.copy()
    for col, spec in schema.items():
        if col not in out.columns or out[col].isna().sum() == 0:
            continue
        if spec.kind in ("continuous", "count"):
            out[col] = out[col].fillna(out[col].median())
        elif spec.kind == "ordinal":
            cats = pd.Categorical(out[col])
            mode_code = pd.Series(cats.codes).replace(-1, np.nan).mode()
            if len(mode_code):
                fill = cats.categories[int(mode_code.iloc[0])]
                out[col] = out[col].fillna(fill)
        elif spec.kind == "nominal":
            mode = out[col].mode(dropna=True)
            if len(mode):
                out[col] = out[col].fillna(mode.iloc[0])
        elif spec.kind == "binary":
            if impute_binary:
                mode = out[col].mode(dropna=True)
                if len(mode):
                    out[col] = out[col].fillna(mode.iloc[0])
    return out


def simple_impute_stage(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    output_root: Path | str = "output",
    *,
    impute_binary: bool = False,
) -> pd.DataFrame:
    """Stage ``output/datasets/``, impute once, save ``simple_imputed_df.parquet``."""
    stage_unimputed_dataset(df, output_root)
    mice_dir = _mice_dataset_dir(output_root)
    if mice_dir.exists():
        shutil.rmtree(mice_dir)
    imputed = simple_impute(df, schema, impute_binary=impute_binary)
    save_modeling_dataset(imputed, output_root, method="simple")
    return imputed


def imputation_audit(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    schema: dict[str, ColSpec],
    columns: Sequence[str],
    *,
    impute_binary: bool = False,
) -> pd.DataFrame:
    """Per-column missingness summary before/after ``simple_impute``."""
    rows: list[dict] = []
    n = len(df_before)
    for col in columns:
        if col not in df_before.columns:
            continue
        spec = schema.get(col)
        kind = spec.kind if spec else "unknown"
        n_miss = int(df_before[col].isna().sum())
        pct = round(100 * n_miss / n, 1) if n else 0.0
        if n_miss == 0:
            method = "none (complete)"
        elif kind in ("continuous", "count"):
            method = "median"
        elif kind == "ordinal":
            method = "mode (ordinal level)"
        elif kind == "nominal":
            method = "mode"
        elif kind == "binary":
            method = "mode" if impute_binary else "none (left NaN)"
        else:
            method = "none"
        n_remain = int(df_after[col].isna().sum()) if col in df_after.columns else n_miss
        rows.append({
            "predictor": col,
            "kind": kind,
            "n_missing": n_miss,
            "pct_missing": pct,
            "imputation_method": method,
            "missing_after": n_remain,
        })
    return pd.DataFrame(rows)
