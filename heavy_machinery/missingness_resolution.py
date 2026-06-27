"""Handle the blanks in the table, then hand a clean dataset to the model.

The problem: many MRI/clinical fields have gaps. Dropping every patient with a
gap wastes data and can bias results. So we *fill* the gaps sensibly and carry
the "we're not 100% sure" honestly into the final odds ratios.

    raw cohort (has gaps)
        │  analyze_missingness()      → see where/why blanks cluster
        ▼
    fill the blanks ─┬─ proper_mice_impute()  ★ main method (formal MICE, in R)
                     ├─ rf_chained_impute()      sensitivity check only
                     └─ simple_impute()          quick median/mode for screening
        ▼
    output/datasets/  +  output/missingness/   → picked up by the model notebook

★ ``proper_mice_impute`` makes several complete copies of the table, each with the
  blanks filled a little differently. The spread between copies = the uncertainty,
  which the model then pools (Rubin's rules). This is the statistically valid one.

``rf_chained_impute`` (old name ``mice_impute``) is a robustness check, NOT formal
MICE — do not pool it. ``simple_impute`` is a fast single fill for exploration only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

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
from plot_style import PALETTE, apply_plot_style, prettify_label

apply_plot_style()


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
    """Map where the blanks are — before deciding how to fill them.

    Produces two things and saves them under ``output/missingness/``:

      • a per-column table: how many / what % of values are missing
      • a co-missingness heatmap: which fields tend to be blank *together*
        (e.g. ADC missing whenever DWI wasn't done)

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
        plot_df = per_col[per_col["pct_missing"] > 0].copy()
        plot_df["label"] = plot_df["column"].map(prettify_label)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(plot_df) + 0.8)))
        sns.barplot(x="pct_missing", y="label", data=plot_df, ax=ax,
                    color=PALETTE["accent"])
        ax.set_title("Missing values per column")
        ax.set_xlabel("% missing"); ax.set_ylabel("")
        ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=8.5, padding=3)
        ax.margins(x=0.12)
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
        pretty = [prettify_label(c) for c in cols_with_miss]
        jacc_disp = jacc.copy()
        jacc_disp.index = pretty
        jacc_disp.columns = pretty
        nlab = len(cols_with_miss)
        fig, ax = plt.subplots(figsize=(0.75 * nlab + 3, 0.75 * nlab + 3))
        # Lower triangle only — hide upper mirror and diagonal (self = always 1.00).
        tri_mask = np.triu(np.ones_like(jacc_disp.values, dtype=bool), k=0)
        annot_fs = 8 if nlab <= 12 else 6.5
        sns.heatmap(jacc_disp, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar=True,
                    mask=tri_mask, annot_kws={"fontsize": annot_fs},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "Jaccard overlap", "shrink": 0.6})
        ax.set_title("Co-missingness overlap (Jaccard)")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
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
    """Separate "blank because it doesn't exist" from "blank because unrecorded".

    Some blanks are not really missing — e.g. ``lesion_2`` is empty because there
    was only one lesion. Filling those would be nonsense. This marks such columns
    ``kind='skip'`` so they are left out of imputation, screening, and the model.

    In their place it can build two real, countable features per group:

      • ``<group_name>``      → how many slots were filled (e.g. n_lesions = 0..3)
      • ``<group_name>_max``  → the largest value across the slots (the dominant
                                lesion; only meaningful for numeric/ordinal slots)

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
    The dataframe with the count/max columns added. ``schema`` is updated in
    place: new entries added, and ``skip_after`` columns flipped to ``'skip'``.
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
    """Add a yes/no ``<col>_missing`` flag when *being blank* may itself matter.

    Sometimes a test isn't ordered precisely because the patient looked low-risk
    (or high-risk). There the blank carries information. This adds a column
    recording "was this value present?" so the model can use that signal.
    If ``schema`` is passed, each flag is registered as a binary column.
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
    """Remove patients (rows) on purpose, and keep a paper trail of why.

    Use this to exclude records by rule — e.g. paediatric cases or impossible
    values — so the exclusions are documented, not silent.

    Give EITHER a boolean ``mask`` (True = drop) OR a ``where`` query string
    (rows matching it are dropped). If ``log`` (a list) is passed, one entry is
    appended per call — ready to paste into the methods section.

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

def _decode_binary_bernoulli(
    predicted: pd.Series,
    original: pd.Series,
    rng: np.random.Generator,
    ) -> pd.Series:
    """Preserve observed booleans; Bernoulli-sample originally missing cells."""
    out = original.astype("boolean").copy()
    missing = original.isna()

    if not missing.any():
        return out

    probabilities = predicted.loc[missing].astype(float).clip(0.0, 1.0)

    draws = pd.Series(
        rng.binomial(1, probabilities.to_numpy()).astype(bool),
        index=probabilities.index,
        dtype="boolean",
    )

    out.loc[missing] = draws
    return out


def _decode_after_impute(
    imputed: pd.DataFrame,
    decoders: dict[str, dict[int, object]],
    cat_cols: list[str],
    schema: dict[str, ColSpec],
    original: pd.DataFrame,
    rng: np.random.Generator,
    ) -> pd.DataFrame:
    out = imputed.copy()

    for col in cat_cols:
        if col not in out.columns:
            continue

        spec = schema[col]

        if spec.kind == "binary":
            out[col] = _decode_binary_bernoulli(
                predicted=out[col],
                original=original[col],
                rng=rng,
            )
        else:
            codes = out[col].round().clip(
                lower=0,
                upper=max(decoders[col]) if decoders[col] else 0,
            )
            out[col] = codes.map(decoders[col])

            levels = (
                spec.ordered_levels
                if spec.kind == "ordinal"
                else list(decoders[col].values())
            )

            out[col] = pd.Categorical(
                out[col],
                categories=levels,
                ordered=(spec.kind == "ordinal"),
            )

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
        elif isinstance(orig_dtype, pd.Int64Dtype):
            out[col] = out[col].astype("Int64")
        elif isinstance(orig_dtype, pd.BooleanDtype):
            out[col] = out[col].astype("boolean")
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
            elif isinstance(orig_dtype, pd.BooleanDtype):
                observed = original[col].notna()

                assert frame[col].dtype == "boolean", (
                    f"imputed frame {idx}, column {col}: expected boolean dtype"
                )

                assert frame.loc[observed, col].equals(
                    original.loc[observed, col]
                ), (
                    f"imputed frame {idx}, column {col}: "
                    "observed binary values were modified"
                )

                assert not frame[col].isna().any(), (
                    f"imputed frame {idx}, column {col}: "
                    "binary missing values remain"
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
    """Hand the model notebook the dataset(s) to fit on.

    If the full set of MICE copies exists, returns all of them (so the model can
    pool). Otherwise returns the single simple-filled table as a one-item list.
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
    """Save all the filled copies (one parquet each) plus the manifest receipt."""
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
    """Re-load all the filled copies from disk, with original column types intact."""
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


def read_mice_manifest(output_root: Path | str = "output") -> dict[str, Any]:
    """Read the little receipt saved next to the imputed data (empty if none).

    The manifest records how the data was filled — which method, how many copies,
    and whether it's valid for pooling.
    """
    manifest_path = _mice_dataset_dir(output_root) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def assert_proper_multiple_imputation(output_root: Path | str = "output") -> dict[str, Any]:
    """Gatekeeper: only let proper MICE through to Rubin pooling.

    Raises if the data was filled by the sensitivity method (random forest) or if
    no receipt exists — so the model stage can't accidentally report a backup
    method as if it were the real one.
    """
    manifest = read_mice_manifest(output_root)
    if not manifest:
        raise FileNotFoundError(
            f"No MICE manifest in {_mice_dataset_dir(output_root)}. "
            "Run proper_mice_impute() before Rubin pooling."
        )
    if not (
        manifest.get("proper_multiple_imputation")
        and manifest.get("rubin_pooling_supported")
    ):
        raise ValueError(
            "Loaded imputation is a sensitivity method "
            f"(method={manifest.get('method')!r}); Rubin pooling requires "
            "proper_mice_impute() output "
            "(proper_multiple_imputation=True, rubin_pooling_supported=True)."
        )
    return manifest


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

    rng = np.random.default_rng(random_state + i)

    decoded = _decode_after_impute(
        imp,
        decoders,
        cat_cols,
        schema,
        original=df,
        rng=rng,
    )
    
    for c in dropped:
        if c in df.columns:
            decoded[c] = df[c].values
    decoded = decoded.reindex(columns=df.columns)
    elapsed = time.perf_counter() - t0
    print(f"✅ Imputation {draw}/{m_total} done ({_format_elapsed(elapsed)})", flush=True)
    return decoded, convergence_warned


def rf_chained_impute(
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
    """Backup gap-filler (random forest) — a robustness check, NOT the real one.

        ⚠ Sensitivity analysis only. Not formal MICE. Do NOT pool with Rubin's rules.

    Fills blanks with a random forest predicting each column from the rest, and
    flips a weighted coin for missing yes/no fields. It makes ``m`` copies, but
    it does not carry uncertainty correctly, so the manifest is flagged
    ``proper_multiple_imputation=False``. For real inference use
    :func:`proper_mice_impute`. Keep this around only to show results don't hinge
    on the imputation choice.

    Runtime is laptop-friendly by default: draws run in parallel, but the number
    of workers is capped per operating system and throttled further on macOS
    battery (``enforce_macos_battery_safety``) to protect heat/battery.
    ``suppress_convergence_warnings`` hides the noisy per-draw warnings and prints
    one summary count instead.
    """
    stage_unimputed_dataset(df, output_root)
    figs, tabs = _ensure_dirs(Path(output_root))

    work, decoders, cat_cols, dropped = _encode_for_impute(df, schema)
    mice_meta = {
        "method": "rf_chained_imputation_posthoc_bernoulli",
        "proper_multiple_imputation": False,
        "rubin_pooling_supported": False,
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


# Backward-compatibility alias. Older notebook cells call ``mice_impute``; it is
# the RF chained-imputation sensitivity method, NOT formal mixed-type MICE.
# Use :func:`proper_mice_impute` for inferential multiple imputation.
mice_impute = rf_chained_impute


# ---------------------------------------------------------------------------
# 4. Simple imputation (fallback / screening)
# ---------------------------------------------------------------------------

def simple_impute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    impute_binary: bool = False,
    ) -> pd.DataFrame:
    """Quick-and-dirty single fill — for exploration only, not the final model.

    Fills each blank with one obvious value (no uncertainty, one copy):

      • numbers  → median
      • category → most common value
      • yes / no → left blank on purpose ("unknown" ≠ "absent"),
                   unless ``impute_binary=True``

    Fine for fast EDA/screening; use :func:`proper_mice_impute` for real results.
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
    """Run the quick single fill end-to-end and save it for the model notebook.

    Resets ``output/datasets/``, applies :func:`simple_impute` once, and writes
    ``simple_imputed_df.parquet``. Screening shortcut — not for pooled inference.
    """
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
    """Before/after table: how many blanks each field had and how they got filled.

    A quick sanity check for the simple-fill path — confirms what was imputed,
    by which rule, and how many blanks remain.
    """
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


# ---------------------------------------------------------------------------
# 5. Formal mixed-type MICE (R ``mice`` engine, primary inferential method)
# ---------------------------------------------------------------------------
#
# Each incomplete variable receives a model appropriate to its declared kind:
#     continuous -> pmm        count   -> pmm        binary  -> logreg
#     nominal    -> polyreg    ordinal -> polr
# One fully conditional specification (FCS) chain produces ``m`` completed
# datasets and preserves between-imputation uncertainty for Rubin pooling.
# The numeric algorithm lives entirely in ``scripts/run_mice.R``; Python only
# orchestrates the exchange, restoration, validation, and manifest.

_MICE_METHOD_BY_KIND: dict[str, str] = {
    "continuous": "pmm",
    "count": "pmm",
    "binary": "logreg",
    "nominal": "polyreg",
    "ordinal": "polr",
}
_MICE_IMPUTABLE_KINDS = frozenset(_MICE_METHOD_BY_KIND)
_MICE_EXCLUDED_KINDS = frozenset({"id", "text", "datetime", "skip"})

MICE_ROW_ID = "__mice_row_id__"
_R_SCRIPT = Path(__file__).resolve().parent / "scripts" / "run_mice.R"
_MICE_RUN_SUBDIR = "r_run"


def mice_method_for_kind(kind: str) -> str:
    """Pick the right fill-in model for a field type (continuous → pmm, etc.)."""
    try:
        return _MICE_METHOD_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(
            f"No formal-MICE method for kind {kind!r}; expected one of "
            f"{sorted(_MICE_IMPUTABLE_KINDS)}"
        ) from exc


def _declared_levels(series: pd.Series, spec: ColSpec) -> list[Any]:
    """Declared category levels for a binary/nominal/ordinal column."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return list(dtype.categories)
    if spec.ordered_levels:
        return list(spec.ordered_levels)
    observed = series.dropna().unique().tolist()
    try:
        return sorted(observed)
    except TypeError:
        return list(observed)


def _r_token(value: Any) -> str:
    """String token written to CSV / declared as an R factor level."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _classify_mice_columns(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    analysis_outcome: str | None,
    derived_dependencies: dict[str, Sequence[str]],
    predictor_exclusions: Sequence[str],
) -> dict[str, Any]:
    """Partition columns into the R imputation matrix vs reattached extras.

    Returns a dict describing which columns go to R, which are dropped and
    reattached afterward, and the analysis-outcome source columns excluded as
    duplicate predictors.
    """
    derived_cols = [c for c in derived_dependencies if c in df.columns]
    non_outcome_derived = [c for c in derived_cols if c != analysis_outcome]

    outcome_source_cols: list[str] = []
    if analysis_outcome and analysis_outcome in derived_dependencies:
        outcome_source_cols = [
            c for c in derived_dependencies[analysis_outcome] if c in df.columns
        ]

    structural_excluded = [
        c for c, spec in schema.items()
        if spec.kind in _MICE_EXCLUDED_KINDS and c in df.columns
    ]
    # Columns dropped before R and reattached unchanged afterwards.
    dropped = []
    for col in df.columns:
        if (
            col in non_outcome_derived
            or col in outcome_source_cols
            or col in structural_excluded
        ):
            dropped.append(col)
    r_columns = [c for c in df.columns if c not in dropped]

    excl = [c for c in predictor_exclusions if c in r_columns]
    non_predictor_cols = [MICE_ROW_ID, *excl]

    return {
        "derived_cols": derived_cols,
        "non_outcome_derived": non_outcome_derived,
        "outcome_source_cols": outcome_source_cols,
        "structural_excluded": structural_excluded,
        "dropped": dropped,
        "r_columns": r_columns,
        "non_predictor_cols": non_predictor_cols,
    }


def _build_mice_spec(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    parts: dict[str, Any],
    *,
    m: int,
    max_iter: int,
    random_state: int,
    analysis_outcome: str | None,
    derived_dependencies: dict[str, Sequence[str]],
    predictor_exclusions: Sequence[str],
    input_sha256: str,
) -> dict[str, Any]:
    """Assemble the JSON specification consumed by ``run_mice.R``."""
    r_columns: list[str] = parts["r_columns"]
    kinds: dict[str, str] = {}
    r_types: dict[str, str] = {}
    levels: dict[str, list[str]] = {}
    methods: dict[str, str] = {}
    vars_with_missing: list[str] = []
    missing_counts: dict[str, int] = {}

    for col in r_columns:
        spec = schema.get(col)
        kind = spec.kind if spec else "continuous"
        kinds[col] = kind
        n_missing = int(df[col].isna().sum())
        missing_counts[col] = n_missing
        has_missing = n_missing > 0

        if kind in ("continuous", "count"):
            r_types[col] = "numeric"
        elif kind == "binary":
            r_types[col] = "factor"
            levels[col] = [_r_token(v) for v in _declared_levels(df[col], spec)]
        elif kind == "nominal":
            r_types[col] = "factor"
            levels[col] = [_r_token(v) for v in _declared_levels(df[col], spec)]
        elif kind == "ordinal":
            r_types[col] = "ordered"
            levels[col] = [_r_token(v) for v in _declared_levels(df[col], spec)]
        else:
            # Defensive: any non-imputable kind that slipped through stays inert.
            r_types[col] = "numeric"

        # Method: empty for complete columns, the analysis outcome (never
        # imputed), and the row id; otherwise mapped from the declared kind.
        if col == MICE_ROW_ID or col == analysis_outcome or not has_missing:
            methods[col] = ""
        elif kind in _MICE_IMPUTABLE_KINDS:
            methods[col] = mice_method_for_kind(kind)
            vars_with_missing.append(col)
        else:
            methods[col] = ""

    return {
        "row_id_col": MICE_ROW_ID,
        "columns": r_columns,
        "kinds": kinds,
        "r_types": r_types,
        "levels": levels,
        "methods": methods,
        "vars_with_missing": vars_with_missing,
        "missing_counts": missing_counts,
        "non_predictor_cols": parts["non_predictor_cols"],
        "analysis_outcome": analysis_outcome,
        "derived_dependencies": {k: list(v) for k, v in derived_dependencies.items()},
        "predictor_exclusions": list(predictor_exclusions),
        "m": int(m),
        "max_iter": int(max_iter),
        "seed": int(random_state),
        "input_sha256": input_sha256,
    }


def _write_r_input_csv(df: pd.DataFrame, r_columns: list[str], path: Path) -> str:
    """Write the R imputation matrix to CSV; return its sha256 hex digest."""
    out = df[r_columns].copy()
    out.insert(0, MICE_ROW_ID, np.arange(len(out), dtype="int64"))
    # Booleans must serialize as the declared factor tokens "True"/"False".
    for col in out.columns:
        if isinstance(out[col].dtype, pd.BooleanDtype) or out[col].dtype == bool:
            out[col] = out[col].map(
                lambda v: ("" if pd.isna(v) else ("True" if bool(v) else "False"))
            )
    out.to_csv(path, index=False, na_rep="")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_r_environment() -> str:
    """Verify Rscript + ``mice`` + ``jsonlite`` are available; return Rscript path."""
    rscript = shutil.which("Rscript")
    install_hint = (
        "\n\nFormal MICE requires R and the R packages mice and jsonlite.\n\n"
        "Install in R:\n"
        'install.packages(c("mice", "jsonlite"))'
    )
    if rscript is None:
        raise RuntimeError("Rscript was not found on PATH." + install_hint)

    probe = (
        'ok <- all(c('
        'requireNamespace("mice", quietly=TRUE), '
        'requireNamespace("jsonlite", quietly=TRUE)));'
        'cat(if (ok) "OK" else "MISSING")'
    )
    result = subprocess.run(
        [rscript, "-e", probe],
        capture_output=True, text=True,
    )
    if "OK" not in (result.stdout or ""):
        raise RuntimeError(
            "Required R packages are missing (mice / jsonlite)."
            + install_hint
            + f"\n\nRscript stdout:\n{result.stdout}\nRscript stderr:\n{result.stderr}"
        )
    return rscript


def _run_r_mice(rscript: str, run_dir: Path) -> None:
    """Invoke ``run_mice.R`` on ``run_dir``; surface stdout+stderr on failure."""
    if not _R_SCRIPT.exists():
        raise FileNotFoundError(f"Missing R engine script: {_R_SCRIPT}")
    proc = subprocess.run(
        [rscript, str(_R_SCRIPT), str(run_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_mice.R failed (exit {proc.returncode}).\n\n"
            f"--- stdout ---\n{proc.stdout}\n\n--- stderr ---\n{proc.stderr}"
        )
    if proc.stdout:
        print(proc.stdout, flush=True)


def _restore_r_frame(
    frame: pd.DataFrame,
    original: pd.DataFrame,
    r_columns: list[str],
) -> pd.DataFrame:
    """Restore Python dtypes for columns returned by R (string tokens → typed)."""
    out = frame.copy()
    for col in r_columns:
        if col not in out.columns:
            continue
        odt = original[col].dtype
        s = out[col]
        if isinstance(odt, pd.CategoricalDtype):
            cats = list(odt.categories)
            token_map = {_r_token(c): c for c in cats}
            mapped = s.map(lambda v: token_map.get(_r_token(v), pd.NA)
                           if not pd.isna(v) else pd.NA)
            out[col] = pd.Categorical(mapped, categories=cats, ordered=odt.ordered)
        elif isinstance(odt, pd.BooleanDtype) or odt == bool:
            out[col] = s.map(_to_bool).astype("boolean")
        elif isinstance(odt, pd.Int64Dtype):
            out[col] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif isinstance(odt, pd.Float64Dtype):
            out[col] = pd.to_numeric(s, errors="coerce").astype("Float64")
        elif pd.api.types.is_float_dtype(odt):
            out[col] = pd.to_numeric(s, errors="coerce").astype(odt)
        elif pd.api.types.is_integer_dtype(odt):
            out[col] = pd.to_numeric(s, errors="coerce").round().astype(odt)
    return out


def _to_bool(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in ("true", "1", "1.0", "yes"):
        return True
    if token in ("false", "0", "0.0", "no"):
        return False
    return pd.NA


def _validate_proper_mice_frame(
    original: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    idx: int,
    r_columns: list[str],
    vars_with_missing: list[str],
    derived_dependencies: dict[str, Sequence[str]],
) -> None:
    """Structural checks per completed frame (Pandera is run separately)."""
    tag = f"proper MICE frame {idx}"
    assert list(frame.columns) == list(original.columns), f"{tag}: column mismatch"
    assert len(frame) == len(original), f"{tag}: row count mismatch"
    assert frame.index.equals(original.index), f"{tag}: row identity changed"

    # Observed values unchanged; intended missing cells filled.
    for col in r_columns:
        observed = original[col].notna()
        if isinstance(original[col].dtype, pd.CategoricalDtype):
            same = (frame.loc[observed, col].astype("object")
                    .reset_index(drop=True)
                    .equals(original.loc[observed, col].astype("object")
                            .reset_index(drop=True)))
            assert same, f"{tag}, {col}: observed categorical values modified"
        else:
            o = original.loc[observed, col]
            f = frame.loc[observed, col]
            if pd.api.types.is_numeric_dtype(o):
                assert np.allclose(
                    pd.to_numeric(o, errors="coerce").astype(float),
                    pd.to_numeric(f, errors="coerce").astype(float),
                    equal_nan=True,
                ), f"{tag}, {col}: observed numeric values modified"
            else:
                assert f.astype("object").reset_index(drop=True).equals(
                    o.astype("object").reset_index(drop=True)
                ), f"{tag}, {col}: observed values modified"
        if col in vars_with_missing:
            assert frame[col].isna().sum() == 0, (
                f"{tag}, {col}: imputable column still has missing cells"
            )

    # Continuous/float values finite where present.
    for col in r_columns:
        if pd.api.types.is_float_dtype(frame[col].dtype) or isinstance(
            frame[col].dtype, pd.Float64Dtype
        ):
            vals = pd.to_numeric(frame[col], errors="coerce").dropna()
            assert np.isfinite(vals.astype(float)).all(), (
                f"{tag}, {col}: non-finite imputed value"
            )

    # Derived columns must be internally consistent with their sources: the
    # callback must have (re)computed a value wherever every source is present.
    # A stale or un-recreated derived column would leave gaps here.
    for col, sources in derived_dependencies.items():
        if col not in frame.columns:
            continue
        present_sources = [s for s in sources if s in frame.columns]
        if not present_sources:
            continue
        sources_known = frame[present_sources].notna().all(axis=1)
        derived_missing = frame[col].isna() & sources_known
        assert not derived_missing.any(), (
            f"{tag}, {col}: derived value missing where sources "
            f"{present_sources} are present (not recreated from imputed source)"
        )


def _imputed_cell_variation(
    original: pd.DataFrame,
    frames: list[pd.DataFrame],
    schema: dict[str, ColSpec],
    r_columns: list[str],
    vars_with_missing: list[str],
) -> pd.DataFrame:
    """Per originally-missing cell: how imputed values vary across the m draws."""
    rows: list[dict[str, Any]] = []
    m = len(frames)
    for col in vars_with_missing:
        spec = schema.get(col)
        kind = spec.kind if spec else "continuous"
        missing_mask = original[col].isna().to_numpy()
        missing_positions = np.flatnonzero(missing_mask)
        for pos in missing_positions:
            draws = [frame[col].iloc[pos] for frame in frames]
            row: dict[str, Any] = {
                "original_row_id": int(pos),
                "variable": col,
                "kind": kind,
                "m": m,
            }
            if kind in ("continuous", "count"):
                numeric = pd.to_numeric(pd.Series(draws), errors="coerce")
                row.update({
                    "mean_imputed": float(numeric.mean()),
                    "median_imputed": float(numeric.median()),
                    "sd_across_draws": float(numeric.std(ddof=0)),
                    "min_imputed": float(numeric.min()),
                    "max_imputed": float(numeric.max()),
                    "values_across_draws": json.dumps(
                        [None if pd.isna(v) else float(v) for v in numeric]
                    ),
                })
            elif kind == "binary":
                bools = [bool(v) for v in draws if not pd.isna(v)]
                n_true = int(sum(bools))
                n_false = int(len(bools) - n_true)
                row.update({
                    "n_true": n_true,
                    "n_false": n_false,
                    "proportion_true": (n_true / len(bools)) if bools else None,
                    "values_across_draws": json.dumps(
                        [None if pd.isna(v) else bool(v) for v in draws]
                    ),
                })
            else:  # nominal / ordinal
                tokens = [_r_token(v) for v in draws if not pd.isna(v)]
                counts = pd.Series(tokens).value_counts()
                total = int(counts.sum())
                row.update({
                    "modal_level": counts.index[0] if len(counts) else None,
                    "level_counts": json.dumps(
                        {str(k): int(v) for k, v in counts.items()}
                    ),
                    "level_proportions": json.dumps(
                        {str(k): (int(v) / total) for k, v in counts.items()}
                    )
                    if total else json.dumps({}),
                    "values_across_draws": json.dumps(
                        [None if pd.isna(v) else _r_token(v) for v in draws]
                    ),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def proper_mice_impute(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    m: int = 20,
    max_iter: int = 20,
    random_state: int = 42,
    output_root: Path | str = "output",
    analysis_outcome: str | None = None,
    derived_dependencies: dict[str, Sequence[str]] | None = None,
    post_impute_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    predictor_exclusions: Sequence[str] = (),
    save_imputed: bool = True,
) -> list[pd.DataFrame]:
    """★ The proper way to fill the blanks (formal MICE, run in R).

    Think of it as filling the table several times over. Each blank is estimated
    from the other columns, using the right tool for that kind of field:

        continuous / count  →  pmm      (borrow a real value from a similar patient)
        yes / no            →  logreg   (logistic regression)
        unordered category  →  polyreg
        ordered grade       →  polr     (ordinal regression)

    It cycles through the incomplete fields ``max_iter`` times and produces ``m``
    completed tables. Where the tables disagree = the honest uncertainty, which
    the model later pools with Rubin's rules. This is the version valid for
    publication.

        df ─► write CSV+spec ─► Rscript run_mice.R ─► m filled tables
                                                       ─► restore types ─► recreate
                                                          derived cols ─► validate

    Needs R with the ``mice`` and ``jsonlite`` packages; Python calls it for you.
    Never silently falls back to the random-forest method.

    Note: for analysis, the known outcome (e.g. high_grade) is allowed to help
    predict missing findings. This must NOT be reused unchanged for a deployment
    calculator, where the outcome is exactly what's unknown.

    Returns the list of ``m`` completed dataframes.
    """
    if df.empty:
        raise ValueError("proper_mice_impute: input dataframe is empty")
    if m < 1 or max_iter < 1:
        raise ValueError("proper_mice_impute: m and max_iter must be >= 1")
    if analysis_outcome is not None and analysis_outcome not in df.columns:
        raise ValueError(
            f"proper_mice_impute: analysis_outcome {analysis_outcome!r} not in df"
        )
    derived_dependencies = dict(derived_dependencies or {})

    parts = _classify_mice_columns(
        df, schema,
        analysis_outcome=analysis_outcome,
        derived_dependencies=derived_dependencies,
        predictor_exclusions=predictor_exclusions,
    )
    r_columns: list[str] = parts["r_columns"]
    if not r_columns:
        raise ValueError("proper_mice_impute: no columns left for the R matrix")

    stage_unimputed_dataset(df, output_root)
    mice_dir = _mice_dataset_dir(output_root)
    mice_dir.mkdir(parents=True, exist_ok=True)
    run_dir = mice_dir / _MICE_RUN_SUBDIR
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.csv"
    input_sha256 = _write_r_input_csv(df, r_columns, input_path)

    spec = _build_mice_spec(
        df, schema, parts,
        m=m, max_iter=max_iter, random_state=random_state,
        analysis_outcome=analysis_outcome,
        derived_dependencies=derived_dependencies,
        predictor_exclusions=predictor_exclusions,
        input_sha256=input_sha256,
    )
    (run_dir / "mice_spec.json").write_text(
        json.dumps(spec, indent=2, default=str), encoding="utf-8",
    )

    rscript = _check_r_environment()
    print(
        f"🧬 Formal mixed-type MICE (R mice) — m={m}, maxit={max_iter}, "
        f"{len(spec['vars_with_missing'])} incomplete variables…",
        flush=True,
    )
    _run_r_mice(rscript, run_dir)

    # --- reload completed datasets -----------------------------------------
    imputed_paths = sorted(run_dir.glob("imputed_*.csv"))
    if len(imputed_paths) != m:
        raise RuntimeError(
            f"proper_mice_impute: expected {m} imputed CSVs, found "
            f"{len(imputed_paths)} in {run_dir}"
        )

    # Reattach only columns that never went to R and are NOT recreated by the
    # derivation callback: structural (id/text/datetime/skip) and the analysis
    # outcome's source columns. Non-outcome derived columns are recomputed from
    # their imputed sources by ``post_impute_transform`` (Phase 5). When no
    # callback is supplied, reattach them unchanged so the frame stays complete.
    reattach_cols: list[str] = [
        c for c in parts["dropped"] if c not in parts["non_outcome_derived"]
    ]
    if post_impute_transform is None:
        reattach_cols = list(parts["dropped"])
    frames: list[pd.DataFrame] = []
    for path in imputed_paths:
        raw = pd.read_csv(path)
        if MICE_ROW_ID not in raw.columns:
            raise RuntimeError(f"{path.name}: missing {MICE_ROW_ID} column")
        raw = raw.sort_values(MICE_ROW_ID).reset_index(drop=True)
        order = raw[MICE_ROW_ID].astype(int).to_numpy()
        raw = raw.drop(columns=[MICE_ROW_ID])
        frame = _restore_r_frame(raw, df, r_columns)
        frame.index = df.index[order]
        # Reattach columns that never went to R, aligned by original row order.
        # Use the original Series (re-indexed positionally) so categorical /
        # Int64 / boolean / datetime dtypes survive — .to_numpy() would flatten
        # categoricals to plain str and break Pandera/dtype checks.
        for col in reattach_cols:
            reattached = df[col].iloc[order].copy()
            reattached.index = frame.index
            frame[col] = reattached
        frame = frame.reindex(columns=[c for c in df.columns if c in frame.columns])
        if post_impute_transform is not None:
            frame = post_impute_transform(frame)
        frame = frame.reindex(columns=df.columns)
        frames.append(frame)

    for idx, frame in enumerate(frames, start=1):
        _validate_proper_mice_frame(
            df, frame, idx=idx,
            r_columns=r_columns,
            vars_with_missing=spec["vars_with_missing"],
            derived_dependencies=derived_dependencies,
        )

    # --- diagnostics --------------------------------------------------------
    cell_variation = _imputed_cell_variation(
        df, frames, schema, r_columns, spec["vars_with_missing"],
    )
    cell_variation_path = mice_dir / "imputed_cell_variation.csv"
    cell_variation.to_csv(cell_variation_path, index=False)

    logged_events_count = 0
    for artifact in (
        "methods.csv", "predictor_matrix.csv", "logged_events.csv",
        "chain_diagnostics.png", "r_session.json",
    ):
        src = run_dir / artifact
        if src.exists():
            shutil.copy2(src, mice_dir / artifact)
    logged_path = mice_dir / "logged_events.csv"
    if logged_path.exists():
        try:
            logged_events_count = int(len(pd.read_csv(logged_path)))
        except Exception:
            logged_events_count = 0
    if logged_events_count:
        print(
            f"⚠️  R mice recorded {logged_events_count} logged event(s) — "
            f"see {logged_path}",
            flush=True,
        )

    r_session: dict[str, Any] = {}
    r_session_path = mice_dir / "r_session.json"
    if r_session_path.exists():
        try:
            r_session = json.loads(r_session_path.read_text(encoding="utf-8"))
        except Exception:
            r_session = {}

    # --- save + manifest ----------------------------------------------------
    metadata = {
        "method": "mice_fcs_mixed_type_r",
        "engine": "R mice",
        "proper_multiple_imputation": True,
        "rubin_pooling_supported": True,
        "assumption": "MAR conditional on included predictors",
        "methods_by_column": spec["methods"],
        "derived_dependencies": spec["derived_dependencies"],
        "analysis_outcome": analysis_outcome,
        "predictor_matrix_file": "predictor_matrix.csv",
        "cell_variation_file": "imputed_cell_variation.csv",
        "m": m,
        "max_iter": max_iter,
        "seed": random_state,
        "r_version": r_session.get("r_version"),
        "mice_version": r_session.get("mice_version"),
        "jsonlite_version": r_session.get("jsonlite_version"),
        "input_sha256": input_sha256,
        "logged_events_count": logged_events_count,
    }
    if save_imputed:
        save_imputed_frames(
            frames, output_root, source_df=frames[0], metadata=metadata,
        )
    save_modeling_dataset(frames[0], output_root, method="mice")
    print(
        f"🏁 Formal MICE complete — {len(frames)} completed datasets, "
        f"diagnostics in {mice_dir}",
        flush=True,
    )
    return frames
