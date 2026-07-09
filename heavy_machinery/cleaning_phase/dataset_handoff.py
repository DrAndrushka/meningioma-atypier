"""Validate parquet handoff between cleaning and modelling notebooks.

Checks roundtrip integrity, dtypes, row counts, and imputation-method detection
for files under ``output/datasets/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from missingness_resolution import (
    MICE_MODELING_DATASET_NAME,
    SIMPLE_MODELING_DATASET_NAME,
    UNIMPUTED_DATASET_NAME,
    _assert_frame_dtypes_match,
    _datasets_dir,
    _load_dataset_parquet,
    _mice_dataset_dir,
    _read_datasets_manifest,
    load_imputed_frames,
    load_unimputed_dataset,
)
from schema_infer import ColSpec, export_schema_summary, load_schema_from_handoff


def _assert_missing_counts_match(
    original: pd.DataFrame,
    loaded: pd.DataFrame,
    *,
    context: str,
) -> None:
    miss_orig = original.isna().sum()
    miss_loaded = loaded.isna().sum()
    if not miss_orig.equals(miss_loaded):
        diff = (miss_orig - miss_loaded)[miss_orig != miss_loaded]
        raise AssertionError(
            f"{context}: missing-count mismatch — "
            + "; ".join(f"{col}: {miss_orig[col]} vs {miss_loaded[col]}" for col in diff.index)
        )


def _assert_numeric_stats_match(
    original: pd.DataFrame,
    loaded: pd.DataFrame,
    *,
    context: str,
) -> None:
    mismatches: list[str] = []
    for col in original.columns:
        if not (pd.api.types.is_numeric_dtype(original[col]) or pd.api.types.is_bool_dtype(original[col])):
            continue
        o = original[col]
        l = loaded[col]
        stats = {
            "count": (o.count(), l.count()),
            "mean": (o.mean(), l.mean()),
            "std": (o.std(ddof=0), l.std(ddof=0)),
            "min": (o.min(), l.min()),
            "max": (o.max(), l.max()),
        }
        for name, (a, b) in stats.items():
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or not np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
                mismatches.append(f"{col}.{name}: {a!r} vs {b!r}")
    if mismatches:
        raise AssertionError(f"{context}: numeric stat mismatch — " + "; ".join(mismatches))


def assert_dataset_equivalent(
    original: pd.DataFrame,
    loaded: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Assert shape, columns, dtypes, missingness, and numeric summaries match."""
    if list(loaded.columns) != list(original.columns):
        raise AssertionError(f"{context}: column order mismatch")
    if loaded.shape != original.shape:
        raise AssertionError(
            f"{context}: shape mismatch {loaded.shape} vs {original.shape}"
        )
    _assert_frame_dtypes_match(original, loaded, context=f"{context} dtypes")
    _assert_missing_counts_match(original, loaded, context=context)
    _assert_numeric_stats_match(original, loaded, context=context)


def _load_modeling_primary(output_root: Path | str) -> pd.DataFrame:
    manifest = _read_datasets_manifest(output_root)
    dtype_spec = manifest.get("dtypes") or {}
    modeling_name = manifest.get("modeling_dataset")
    if not modeling_name:
        mice_path = _datasets_dir(output_root) / MICE_MODELING_DATASET_NAME
        simple_path = _datasets_dir(output_root) / SIMPLE_MODELING_DATASET_NAME
        if mice_path.exists():
            modeling_name = MICE_MODELING_DATASET_NAME
        elif simple_path.exists():
            modeling_name = SIMPLE_MODELING_DATASET_NAME
        else:
            raise FileNotFoundError("No modelling dataset parquet found in output/datasets/")
    path = _datasets_dir(output_root) / modeling_name
    if not path.exists():
        raise FileNotFoundError(f"Missing modelling dataset: {path}")
    if not dtype_spec:
        raise FileNotFoundError("Missing dtype manifest in output/datasets/manifest.json")
    return _load_dataset_parquet(path, dtype_spec)


def validate_schema_against_frame(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    context: str = "modelling handoff",
) -> None:
    """Assert every dataframe column is declared in the handoff schema."""
    df_cols = set(df.columns)
    missing = sorted(df_cols - set(schema.keys()))
    if missing:
        raise ValueError(
            f"{context}: columns in dataframe without schema entry — {missing}"
        )
    skip_present = sorted(c for c in df_cols if schema[c].kind == "skip")
    if skip_present:
        raise ValueError(
            f"{context}: skip columns still present in dataframe — {skip_present}"
        )


def load_modelling_handoff(
    output_root: Path | str = "output",
) -> tuple[pd.DataFrame, dict[str, ColSpec], str]:
    """Load unimputed cohort, committed schema, and imputation method for modelling."""
    output_root = Path(output_root)
    imputation_method = detect_imputation_method(output_root)
    df = load_unimputed_dataset(output_root)
    schema = load_schema_from_handoff(output_root)
    validate_schema_against_frame(df, schema)
    return df, schema, imputation_method


def detect_imputation_method(output_root: Path | str) -> str:
    datasets_dir = _datasets_dir(output_root)
    mice_path = datasets_dir / MICE_MODELING_DATASET_NAME
    simple_path = datasets_dir / SIMPLE_MODELING_DATASET_NAME
    if mice_path.exists():
        return "mice"
    if simple_path.exists():
        return "simple"
    raise FileNotFoundError(
        f"No modelling dataset in {datasets_dir}. "
        "Run meningioma-cleaning.ipynb §10 first."
    )


def validate_handoff_datasets(
    output_root: Path | str,
    *,
    df_unimputed: pd.DataFrame,
    imputation_method: str,
    schema: dict[str, ColSpec] | None = None,
    imputed_frames: list[pd.DataFrame] | None = None,
    imputed_single: pd.DataFrame | None = None,
) -> None:
    """Reload saved parquets and assert equivalence to in-memory frames."""
    output_root = Path(output_root)

    if schema is not None:
        export_schema_summary(schema, output_root)
        print("💾 Re-exported schema/schema_summary.csv (final handoff schema)")

    loaded_unimputed = load_unimputed_dataset(output_root)
    assert_dataset_equivalent(
        df_unimputed, loaded_unimputed, context="unimputed_df.parquet",
    )
    print("✅ parquet roundtrip validated — unimputed_df.parquet")

    if imputation_method == "mice":
        if not imputed_frames:
            raise ValueError("imputed_frames required for MICE validation")
        loaded_primary = _load_modeling_primary(output_root)
        assert_dataset_equivalent(
            imputed_frames[0], loaded_primary, context="mice_imputed_df.parquet",
        )
        print("✅ parquet roundtrip validated — mice_imputed_df.parquet")

        loaded_draws = load_imputed_frames(output_root)
        if len(loaded_draws) != len(imputed_frames):
            raise AssertionError(
                f"MICE draw count mismatch: disk={len(loaded_draws)} "
                f"memory={len(imputed_frames)}"
            )
        for i, (orig, disk) in enumerate(zip(imputed_frames, loaded_draws), start=1):
            name = f"imputed_{i:03d}.parquet"
            assert_dataset_equivalent(orig, disk, context=name)
            print(f"✅ parquet roundtrip validated — {name}")
        return

    if imputation_method == "simple":
        if imputed_single is None:
            raise ValueError("imputed_single required for simple validation")
        loaded_primary = _load_modeling_primary(output_root)
        assert_dataset_equivalent(
            imputed_single, loaded_primary, context="simple_imputed_df.parquet",
        )
        print("✅ parquet roundtrip validated — simple_imputed_df.parquet")
        mice_dir = _mice_dataset_dir(output_root)
        if mice_dir.exists() and any(mice_dir.glob("imputed_*.parquet")):
            raise AssertionError(
                "simple imputation active but MICE draw parquets still exist "
                f"under {mice_dir}"
            )
        return

    raise ValueError(f"Unknown imputation_method: {imputation_method!r}")
