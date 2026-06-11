"""Missingness audit, then MICE (m=10) for modelling.

simple_impute is the quick single-frame fill for eda — binary cols stay NaN
unless you say otherwise, because blank on the form isn't the same as absent.
output/missingness/.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Sequence

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
    if effective_max_worker_slots is None and system == "Windows":
        effective_max_worker_slots = 12

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
) -> pd.DataFrame:
    """Run one MICE imputation draw; mirrors the original per-iteration logic."""
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
    arr = imputer.fit_transform(work)
    imp = pd.DataFrame(arr, columns=work.columns, index=work.index)
    decoded = _decode_after_impute(imp, decoders, cat_cols, schema)
    for c in dropped:
        if c in df.columns:
            decoded[c] = df[c].values
    decoded = decoded.reindex(columns=df.columns)
    elapsed = time.perf_counter() - t0
    print(f"✅ Imputation {draw}/{m_total} done ({_format_elapsed(elapsed)})", flush=True)
    return decoded


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
    """
    figs, tabs = _ensure_dirs(Path(output_root))

    work, decoders, cat_cols, dropped = _encode_for_impute(df, schema)
    if work.isna().sum().sum() == 0:
        # nothing to impute -> return m copies
        print(f"✨ No missing values — returning {m} identical copies (no MICE run).")
        imputed_frames = [_restore_imputed_dtypes(df, df.copy()) for _ in range(m)]
        _validate_imputed_frames(df, imputed_frames)
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

    if emergency_safe_mode:
        settings["n_jobs_imputations"] = 1
        settings["n_jobs_rf"] = 1
    else:
        if n_jobs_imputations is not None:
            settings["n_jobs_imputations"] = n_jobs_imputations
        if n_jobs_rf is not None:
            settings["n_jobs_rf"] = n_jobs_rf

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
    )

    if settings["n_jobs_imputations"] == 1:
        print(f"🐢 Running {m} imputations serially (safest / emergency mode)…")
    else:
        print(
            f"🚀 Launching {m} imputations across "
            f"{settings['n_jobs_imputations']} parallel workers…"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", module=r"sklearn\.utils\.parallel")

        if settings["n_jobs_imputations"] == 1:
            imputed_frames = [
                _run_single_mice_imputation(i=i, **impute_kwargs)
                for i in range(m)
            ]
        else:
            imputed_frames = Parallel(
                n_jobs=settings["n_jobs_imputations"],
                backend=settings["backend"],
            )(
                delayed(_run_single_mice_imputation)(i=i, **impute_kwargs)
                for i in range(m)
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

    pd.DataFrame([{"m": m, "max_iter": max_iter, "random_state": random_state}]) \
      .to_csv(tabs / "mice_config.csv", index=False)
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
