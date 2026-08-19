"""Rubin-pooled multivariable logistic regression on MICE-imputed cohorts.

Fits one model per variant (literature + experimental lists from ``config/07``).
VIF pruning, forest plots, calculator JSON, EPV / complete-case tables, and
Streamlit-ready JSON under ``output/inferential/model_artifacts/``.
Removes stale per-variant tables, figures, and model JSON at the start of each run.
Artifacts → ``output/inferential/`` (``tables/``, ``figures/``, ``model_artifacts/``).
"""

from __future__ import annotations

import re
import warnings
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from schema_infer import ColSpec
from scales import is_log_scaled, to_model_scale
from scales import z_params as z_params_for
from cleaning import format_table_for_csv as _format_table_for_csv  # CSV display-only rounding
from plot_style import (
    FIG_WIDTH_MEDIUM,
    apply_plot_style,
    forest_lr,
    set_figure_legend,
    prettify_label,
    save_figure,
)

apply_plot_style()


@dataclass(frozen=True)
class InferentialModelVariant:
    """One multivariable model specification (outcome + predictor set + report label)."""

    model_id: str
    title: str
    link: str
    target: str
    predictors: tuple[str, ...]
    experimental: bool = False


def _slug_model_id(model_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", model_id.strip().lower()).strip("_")
    if not s:
        raise ValueError(f"Invalid model id: {model_id!r}")
    return s


def _variant_targets(
    variant: InferentialModelVariant,
    fallback_targets: Sequence[str],
) -> list[str]:
    """Outcome column(s) to fit for one variant."""
    if variant.target:
        return [variant.target]
    return list(fallback_targets)


def normalize_inferential_variants(
    predictors: Sequence[str] | None = None,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    *,
    default_target: str = "",
) -> list[InferentialModelVariant]:
    """
    Build variant list from either ``variants=`` (multi-model) or ``predictors=`` (single).

    ``variants`` entries may be:
    - ``InferentialModelVariant``
    - ``{\"id\": ..., \"title\": ..., \"link\": ..., \"target\": ..., \"predictors\": [...]}``
    - ``(id, title, link, target, [predictors])``
    - ``(id, title, target, [predictors])`` — no source link
    - ``(id, title, [predictors])`` — uses ``default_target`` (legacy)
    """
    if variants is not None:
        out: list[InferentialModelVariant] = []
        for v in variants:
            if isinstance(v, InferentialModelVariant):
                out.append(v)
            elif hasattr(v, "model_id") and hasattr(v, "predictors"):
                # Accept variants from a pre-reload inferential import (reload breaks isinstance).
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v.model_id)),
                    title=str(getattr(v, "title", "") or v.model_id).strip(),
                    link=str(getattr(v, "link", "") or "").strip(),
                    target=str(getattr(v, "target", "") or default_target).strip(),
                    predictors=tuple(v.predictors),
                    experimental=bool(getattr(v, "experimental", False)),
                ))
            elif isinstance(v, dict):
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v["id"])),
                    title=str(v.get("title") or v["id"]).strip(),
                    link=str(v.get("link") or "").strip(),
                    target=str(v.get("target") or default_target).strip(),
                    predictors=tuple(v["predictors"]),
                    experimental=bool(v.get("experimental", False)),
                ))
            elif isinstance(v, (tuple, list)) and len(v) >= 5 and isinstance(v[4], (list, tuple)):
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v[0])),
                    title=str(v[1]).strip(),
                    link=str(v[2]).strip(),
                    target=str(v[3]).strip(),
                    predictors=tuple(v[4]),
                ))
            elif isinstance(v, (tuple, list)) and len(v) >= 4 and isinstance(v[3], (list, tuple)):
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v[0])),
                    title=str(v[1]).strip(),
                    link="",
                    target=str(v[2]).strip(),
                    predictors=tuple(v[3]),
                ))
            elif isinstance(v, (tuple, list)) and len(v) == 3 and isinstance(v[2], (list, tuple)):
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v[0])),
                    title=str(v[1]).strip(),
                    link="",
                    target=default_target,
                    predictors=tuple(v[2]),
                ))
            else:
                raise ValueError(
                    "Each variant must be InferentialModelVariant, dict, "
                    "(id, title, link, target, predictors), (id, title, target, predictors), "
                    "or legacy (id, title, predictors)."
                )
        if not out:
            raise ValueError("variants= must contain at least one model.")
        return out
    if predictors is not None:
        return [InferentialModelVariant("", "", "", default_target, tuple(predictors))]
    raise ValueError("Provide predictors= or variants=.")


def artifact_base(target: str, model_id: str = "") -> str:
    """Filename stem prefix: ``high_grade`` or ``high_grade__bondo_et_al``."""
    return f"{target}__{model_id}" if model_id else target


def parse_artifact_base(base: str, known_targets: set[str] | None = None) -> tuple[str, str]:
    """Split artifact stem into ``(target, model_id)``."""
    known_targets = known_targets or set()
    if base in known_targets:
        return base, ""
    for t in sorted(known_targets, key=len, reverse=True):
        prefix = f"{t}__"
        if base.startswith(prefix):
            return t, base[len(prefix):]
    if "__" in base:
        target, model_id = base.split("__", 1)
        return target, model_id
    return base, ""


def model_key(target: str, model_id: str = "") -> str:
    """Internal dict key for report loading."""
    return target if not model_id else f"{target}::{model_id}"


def parse_model_key(key: str) -> tuple[str, str]:
    if "::" in key:
        target, model_id = key.split("::", 1)
        return target, model_id
    return key, ""


def _pool_df_for_display(val: Any) -> object:
    """Format pooled df for CSV/report (avoids 1e+300 style junk when B ≪ W)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return val
    if not np.isfinite(x) or x >= 9999:
        return "∞"
    if x == int(x):
        return int(x)
    return round(x, 1)


def _format_inferential_table(df: pd.DataFrame) -> pd.DataFrame:
    """Display-only copy for saved CSVs (in-memory tables stay numeric)."""
    out = df.copy()
    if "df" in out.columns:
        out["df"] = out["df"].map(_pool_df_for_display)
    return _format_table_for_csv(out)


def _safe_z_denominator(sd: float) -> float:
    """Return sd when finite and positive; otherwise 1.0 (no scaling)."""
    if pd.isna(sd) or not np.isfinite(sd) or sd == 0:
        return 1.0
    return float(sd)


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "inferential" / "figures"
    tabs = root / "inferential" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


_PER_MODEL_TABLE_SUFFIXES = (
    "__multivariable.csv",
    "__vif.csv",
    "__calculator.json",
)

# Written by the comparison stage. Cleared here for the same reason the
# per-model tables are: a run that skips or fails must not leave the
# previous run's headline numbers on disk for the report to pick up.
_COMPARISON_TABLES = (
    "single_predictor_reference.csv",
    "model_vs_single_auc.csv",
    "top_variable_selection.csv",
)


def _clear_inferential_artifacts(figs_dir: Path, tabs_dir: Path) -> None:
    """Drop per-variant inferential outputs so a re-run cannot leave stale models."""
    for path in tabs_dir.iterdir():
        if path.is_file() and (
            any(path.name.endswith(s) for s in _PER_MODEL_TABLE_SUFFIXES)
            or path.name in _COMPARISON_TABLES
        ):
            path.unlink()
    for pattern in ("*__forest.png", "*__forest.tif", "*__forest.eps", "*__forest.svg"):
        for path in figs_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    model_dir = tabs_dir.parent / "model_artifacts"
    if model_dir.is_dir():
        for path in model_dir.glob("*_model.json"):
            if path.is_file():
                path.unlink()


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def _build_design(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    predictors: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, dict[str, float]]]:
    """
    Build X with:
      - continuous/count: z-scored numeric
      - ordinal:          numeric ordinal codes (kept as-is, NOT one-hot)
      - nominal:          one-hot, drop_first=True
      - binary:           0/1
    Returns (X, mapping {original_predictor: [columns_in_X]}, z_params).
    z_params maps continuous/count predictor names to {mu, sd} from that frame.
    """
    pieces = []
    mapping: dict[str, list[str]] = {}
    z_params: dict[str, dict[str, float]] = {}
    for p in predictors:
        spec = schema[p]
        s = df[p]
        if spec.kind in ("continuous", "count"):
            # Which measurements are logged is declared in ``scales``, so the
            # adjusted odds ratios here are on the same scale as the univariable
            # ones in the EDA table and the cut-point tables. ``mu``/``sd`` are
            # stored on that same scale, and ``log`` tells the risk calculator
            # to transform a patient's raw value before applying them.
            log_x = is_log_scaled(p)
            mu, sd = z_params_for(s.to_numpy(dtype=float), log_x)
            u = pd.Series(to_model_scale(s.to_numpy(dtype=float), log_x),
                          index=s.index)
            z = (u - mu) / _safe_z_denominator(sd)
            z.name = p
            pieces.append(z)
            mapping[p] = [p]
            z_params[p] = {"mu": float(mu), "sd": float(sd), "log": bool(log_x)}
        elif spec.kind == "ordinal":
            cats = pd.Categorical(s,
                                  categories=spec.ordered_levels if spec.ordered_levels else None,
                                  ordered=True)
            num = pd.Series(cats.codes.astype(float), index=s.index, name=p).replace(-1, np.nan)
            pieces.append(num)
            mapping[p] = [p]
        elif spec.kind == "binary":
            num = s.astype("float")
            num.name = p
            pieces.append(num)
            mapping[p] = [p]
        elif spec.kind == "nominal":
            dummies = pd.get_dummies(s, prefix=p, drop_first=True, dtype=float)
            pieces.append(dummies)
            mapping[p] = list(dummies.columns)
        # other kinds skipped
    X = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    # Coerce to plain numpy float64: MICE-imputed frames carry nullable
    # extension dtypes (Float64/Int64/boolean) which statsmodels rejects
    # (np.asarray on an extension array yields object dtype). pd.NA -> np.nan.
    if len(X.columns):
        X = X.astype("float64")
    return X, mapping, z_params


def _prune_by_vif(X: pd.DataFrame, threshold: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Iteratively drop the column with the highest VIF until all <= threshold."""
    cols = list(X.columns)
    if not cols:
        return X, pd.DataFrame(columns=["predictor", "vif"])

    # VIF requires a complete numeric design (same rows as logistic .dropna()).
    X_fit = X.dropna()
    if len(X_fit) < max(3, len(cols) + 1):
        return X, pd.DataFrame({"predictor": cols, "vif": np.nan})

    vif_df = pd.DataFrame({"predictor": cols, "vif": np.nan})
    while cols:
        Xc = sm.add_constant(X_fit[cols], has_constant="add")
        vifs = []
        for i, c in enumerate(Xc.columns):
            if c == "const":
                continue
            try:
                v = variance_inflation_factor(Xc.values, i)
            except Exception:
                v = np.nan
            vifs.append((c, v))
        vif_df = pd.DataFrame(vifs, columns=["predictor", "vif"])
        worst = vif_df.loc[vif_df["vif"].idxmax()] if vif_df["vif"].notna().any() else None
        if worst is None or worst["vif"] <= threshold or pd.isna(worst["vif"]):
            break
        cols.remove(worst["predictor"])
    return X[cols], vif_df


# ---------------------------------------------------------------------------
# Rubin pooling
# ---------------------------------------------------------------------------

def _rubin_pool(thetas: np.ndarray, ses: np.ndarray) -> dict[str, float]:
    """
    Rubin's rules pooling for a single coefficient across m imputations.
    thetas, ses: length-m arrays.
    """
    m = len(thetas)
    theta_bar = float(np.mean(thetas))
    within = float(np.mean(ses ** 2))
    between = float(np.var(thetas, ddof=1)) if m > 1 else 0.0
    total = within + (1 + 1 / m) * between
    se = float(np.sqrt(total))
    # Barnard–Rubin degrees of freedom (cap pathological blow-ups when B << W)
    _DF_CAP = 9999.0
    _REL_INCR_FLOOR = 1e-6
    if between > 0 and m > 1 and within > 0:
        rel_incr = (1 + 1 / m) * between / within
        if rel_incr < _REL_INCR_FLOOR:
            df = np.inf
        else:
            df = (m - 1) * (1 + 1 / rel_incr) ** 2
            if not np.isfinite(df) or df > _DF_CAP:
                df = np.inf
    else:
        df = np.inf
    if df == np.inf:
        from scipy.stats import norm
        z = theta_bar / se if se > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else np.nan
        crit = norm.ppf(0.975)
    else:
        tstat = theta_bar / se if se > 0 else np.nan
        p = float(2 * (1 - t_dist.cdf(abs(tstat), df))) if se > 0 else np.nan
        crit = t_dist.ppf(0.975, df)
    ci_lo = theta_bar - crit * se
    ci_hi = theta_bar + crit * se
    return {"coef": theta_bar, "se": se, "df": df, "p": p,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "or": float(np.exp(theta_bar)),
            "or_ci_lo": float(np.exp(ci_lo)),
            "or_ci_hi": float(np.exp(ci_hi))}


# ---------------------------------------------------------------------------
# Per-target multivariable model with MICE pooling
# ---------------------------------------------------------------------------

def _target_is_binary(y: pd.Series, spec: ColSpec | None) -> bool:
    """Logistic regression requires a two-level outcome."""
    nn = y.dropna()
    if int(nn.nunique()) != 2:
        return False
    if spec is not None and spec.kind in ("continuous", "count"):
        return False
    return True


def _encode_target(y: pd.Series, positive_class) -> tuple[pd.Series, object]:
    if positive_class is None:
        nn = y.dropna().unique()
        if len(nn) != 2:
            raise ValueError(f"Target '{y.name}' not binary; values={nn}")
        positive_class = (
            True if True in nn
            else 1 if 1 in nn
            else y.dropna().value_counts().idxmin()  # rarer class
        )
    out = pd.Series(
        np.where(y.isna(), np.nan, (y == positive_class).astype(float)),
        index=y.index,
        dtype=float,
    )
    return out, positive_class


def _logit_converged(model: Any) -> bool:
    mle = getattr(model, "mle_retvals", None)
    if not isinstance(mle, dict):
        params = getattr(model, "params", None)
        return params is not None and np.all(np.isfinite(params))
    return bool(mle.get("converged", False))


def _fit_logit_robust(y: pd.Series, Xc: pd.DataFrame) -> Any | None:
    """
    Fit binary logistic regression with convergence-aware retries.

    Tries Newton first, then BFGS, then L2-regularized logistic if MLE stalls.
    Suppresses ``ConvergenceWarning`` while retrying alternative optimizers.
    """
    logit = sm.Logit(y, Xc)
    strategies: list[dict[str, Any]] = [
        {"disp": False, "method": "newton", "maxiter": 200},
        {"disp": False, "method": "bfgs", "maxiter": 500},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for kwargs in strategies:
            try:
                model = logit.fit(**kwargs)
            except Exception:
                continue
            if _logit_converged(model):
                return model
        try:
            model = logit.fit_regularized(disp=False, alpha=1e-3)
            if np.all(np.isfinite(model.params)):
                return model
        except Exception:
            pass
    return None


def fit_multivariable_logistic(
    imputed_frames: list[pd.DataFrame],
    schema: dict[str, ColSpec],
    target: str,
    predictors: Sequence[str],
    *,
    positive_class=None,
    vif_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run logistic regression on each imputed frame, then Rubin-pool.

    Returns
    -------
    pooled_df : per predictor (column-in-X), OR with 95% CI and p.
    vif_df    : VIF table from the first imputation after pruning.
    """
    # 1. Build X for each imputation, using the SAME columns (use first imp to set design).
    X0, mapping, _ = _build_design(imputed_frames[0], schema, predictors)
    X0_pruned, vif_df = _prune_by_vif(X0, threshold=vif_threshold)
    keep_cols = list(X0_pruned.columns)
    zscore_cols = {
        c for c in keep_cols
        if c in schema and schema[c].kind in ("continuous", "count")
    }

    coefs_by_col: dict[str, list[float]] = {c: [] for c in keep_cols}
    ses_by_col: dict[str, list[float]] = {c: [] for c in keep_cols}
    mus_by_col: dict[str, list[float]] = {c: [] for c in zscore_cols}
    sds_by_col: dict[str, list[float]] = {c: [] for c in zscore_cols}
    intercept_thetas: list[float] = []
    intercept_ses: list[float] = []

    for frame in imputed_frames:
        X, _, z_params = _build_design(frame, schema, predictors)
        for c in zscore_cols:
            if c in z_params:
                mus_by_col[c].append(z_params[c]["mu"])
                sds_by_col[c].append(z_params[c]["sd"])
        X = X.reindex(columns=keep_cols, fill_value=0.0)
        y_enc, _ = _encode_target(frame[target], positive_class)
        sub = pd.concat([y_enc.rename("_y"), X], axis=1).dropna()
        if len(sub) < max(20, X.shape[1] + 5):
            continue
        Xc = sm.add_constant(sub[keep_cols], has_constant="add")
        model = _fit_logit_robust(sub["_y"], Xc)
        if model is None:
            continue
        for c in keep_cols:
            if c in model.params.index:
                coefs_by_col[c].append(float(model.params[c]))
                ses_by_col[c].append(float(model.bse[c]))
        if "const" in model.params.index:
            intercept_thetas.append(float(model.params["const"]))
            intercept_ses.append(float(model.bse["const"]))

    if intercept_thetas:
        intercept_coef = float(_rubin_pool(
            np.array(intercept_thetas), np.array(intercept_ses))["coef"])
        intercept_or = float(np.exp(intercept_coef))
    else:
        intercept_coef = intercept_or = np.nan

    rows = []
    for c in keep_cols:
        thetas = np.array(coefs_by_col[c])
        ses = np.array(ses_by_col[c])
        z_mu = float(np.mean(mus_by_col[c])) if mus_by_col.get(c) else np.nan
        z_sd = float(np.mean(sds_by_col[c])) if sds_by_col.get(c) else np.nan
        if len(thetas) == 0:
            rows.append({"predictor_col": c, "coef": np.nan, "se": np.nan,
                         "or": np.nan, "or_ci_lo": np.nan, "or_ci_hi": np.nan,
                         "p": np.nan, "n_models": 0,
                         "intercept_coef": intercept_coef, "intercept_or": intercept_or,
                         "z_mu": z_mu, "z_sd": z_sd, "z_log": is_log_scaled(c)})
            continue
        pooled = _rubin_pool(thetas, ses)
        rows.append({"predictor_col": c, **pooled, "n_models": len(thetas),
                     "intercept_coef": intercept_coef, "intercept_or": intercept_or,
                     "z_mu": z_mu, "z_sd": z_sd})
    pooled_df = pd.DataFrame(rows)
    pooled_df["target"] = target
    return pooled_df, vif_df


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def predictor_label(row: pd.Series) -> str:
    """Predictor name plus the contrast its odds ratio actually describes.

    Continuous predictors are z-standardised before fitting, so their OR is per
    one standard deviation, not per natural unit. Without saying so, a column
    labelled "Edema volume (cm³)" invites the reader to take OR 1.17 as
    per-cm³ and dismiss the strongest continuous effect in the model.

    For a log-scaled predictor the SD is in log units, so printing it next to a
    clinical label would be worse than printing nothing — "per 1 SD: 1.20" reads
    as 1.2 cm³. Those rows name the scale instead of the width.
    """
    label = prettify_label(row["predictor_col"])
    if is_log_scaled(row["predictor_col"]):
        return f"{label} (per 1 SD, log scale)"
    sd = row.get("z_sd")
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        return label
    if not np.isfinite(sd) or sd <= 0:
        return label
    digits = 0 if abs(sd) >= 10 else 2
    return f"{label} (per 1 SD: {sd:.{digits}f})"


def _forest_plot(
    pooled: pd.DataFrame,
    target: str,
    figs_dir: Path,
    *,
    model_id: str = "",
) -> None:
    """Rubin-pooled adjusted odds ratios for one model variant.

    Rows whose CI crosses 1 are drawn grey. All rows sort together by the
    odds ratio, largest first. Sample size, event count and EPV are not
    written onto the figure — the report states them once, in the EPV gauge
    above it, from ``multivariable_cases.csv``.
    """
    plot_df = pooled.dropna(subset=["or"]).copy()
    if plot_df.empty:
        return
    labels = [predictor_label(r) for _, r in plot_df.iterrows()]
    fig, ax = forest_lr(
        labels,
        plot_df["or"].to_numpy(dtype=float),
        plot_df["or_ci_lo"].to_numpy(dtype=float),
        plot_df["or_ci_hi"].to_numpy(dtype=float),
        ref=1.0,
        xlabel="Adjusted odds ratio (95% CI)",
        value_header="OR (95% CI)",
        width=FIG_WIDTH_MEDIUM,
    )
    del ax
    stem = artifact_base(target, model_id)
    set_figure_legend(fig, plain=(
        "One row per predictor. Squares right of the line raise the odds of high grade, left of it lower them — and if the bar touches the line, it may do nothing at all."))
    save_figure(fig, figs_dir / f"{stem}__forest", tight_layout=False)


# ---------------------------------------------------------------------------
# Performance figures (built from the validated Streamlit artifacts)
# ---------------------------------------------------------------------------

def _artifact_model_id(stem: str, target: str) -> str:
    """``high_grade_yao_et_al_2022_model`` → ``yao_et_al_2022``.

    Artifacts are named ``{target}_{model_id}_model.json``; the in-file
    ``model_name`` is a display title, not the id.
    """
    out = stem[: -len("_model")] if stem.endswith("_model") else stem
    if out == target:
        return ""  # single-model artifact: ``{target}_model.json``
    return out[len(target) + 1:] if out.startswith(f"{target}_") else out


def _outcome_label(definition: str | None) -> str:
    """``1 = high-grade meningioma, 0 = ...`` -> ``high-grade meningioma``.

    The figure footnotes name the outcome; taking it from the artifact means a
    new target names itself instead of inheriting a stale caption.
    """
    if not definition:
        return ""
    head = str(definition).split(",")[0]
    return head.split("=", 1)[-1].strip() if "=" in head else ""


def _write_performance_figures(
    artifact_paths: Sequence[Path],
    figs_dir: Path,
    model_variants: Sequence[InferentialModelVariant],
) -> tuple[list[Path], dict[str, list[dict]]]:
    """ROC / calibration / decision curve per variant, plus one comparison figure.

    Sourced from the enriched Streamlit artifacts so the figures and the
    calculator quote exactly the same bootstrap-validated numbers.
    """
    from performance_plots import (
        write_model_comparison_figure,
        write_performance_figures,
    )

    titles = {v.model_id: (v.title or v.model_id) for v in model_variants}
    written: list[Path] = []
    by_target: dict[str, list[dict]] = {}

    for path in artifact_paths:
        path = Path(path)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        validation = artifact.get("validation")
        target = str(artifact.get("target") or "")
        if not validation or not target:
            continue
        model_id = _artifact_model_id(path.stem, target)

        stem = artifact_base(target, model_id)
        label = titles.get(model_id, model_id)
        try:
            written.extend(write_performance_figures(validation, figs_dir, stem))
        except Exception as exc:  # one bad variant must not sink the stage
            warnings.warn(
                f"Performance figures skipped for {stem}: {exc}", stacklevel=2,
            )
            continue
        # n and events ride along so the figure footnotes can name the sample
        # they were fitted on instead of leaving the reader to find it.
        by_target.setdefault(target, []).append({
            "label": label,
            "model_id": model_id,
            "validation": validation,
            "n": artifact.get("n"),
            "events": artifact.get("events"),
            "outcome_label": _outcome_label(artifact.get("outcome_definition")),
        })

    # The model-overview figure is NOT built here. It quotes the same AUCs,
    # deltas and intervals as the all-models comparison table, and that table
    # does not exist yet at this point in the stage — see
    # ``_write_model_overview_figures``, called once the comparison stage has
    # written ``model_overview.csv``.
    for target, entries in by_target.items():
        try:
            path = write_model_comparison_figure(entries, figs_dir, target=target)
        except Exception as exc:
            warnings.warn(
                f"Model comparison figure skipped for {target}: {exc}", stacklevel=2,
            )
            continue
        if path is not None:
            written.append(path)
    return written, by_target


def _write_model_overview_figures(
    by_target: dict[str, list[dict]],
    figs_dir: Path,
    tabs_dir: Path,
) -> list[Path]:
    """This figure IS the all-models comparison table, drawn.

    It therefore has to run after ``run_comparison_stage`` has written
    ``model_overview.csv``: every AUC, delta and 95% CI it draws is read from
    that file rather than recomputed, so the figure and the table cannot publish
    two different answers to the same question. Without the file there is
    nothing to draw and the figure is skipped rather than guessed at.
    """
    from performance_plots import write_model_performance_overview_figure

    overview_csv = Path(tabs_dir) / "model_overview.csv"
    if not overview_csv.is_file():
        warnings.warn(
            "model_overview.csv not found — the model overview figure is "
            "skipped rather than drawn from recomputed numbers.",
            stacklevel=2,
        )
    written: list[Path] = []
    for target, entries in by_target.items():
        try:
            path = write_model_performance_overview_figure(
                entries, figs_dir, target=target, overview_csv=overview_csv,
            )
        except Exception as exc:
            warnings.warn(
                f"Model overview figure skipped for {target}: {exc}",
                stacklevel=2,
            )
            continue
        if path is not None:
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_INFERENTIAL_COLS = [
    "target", "model_id", "model_title", "model_link", "predictor_col", "or", "or_ci_lo", "or_ci_hi",
    "coef", "se", "p", "df", "n_models",
    "intercept_coef", "intercept_or", "z_mu", "z_sd", "z_log",
]


def _empty_inferential_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_INFERENTIAL_COLS)


def summarize_multivariable_cases(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    *,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
) -> pd.DataFrame:
    """Complete-case counts for multivariable logistic (design + VIF prune + dropna)."""
    positive_class = positive_class or {}
    model_variants = normalize_inferential_variants(predictors, variants)
    rows: list[dict] = []
    n_total = len(df)

    for variant in model_variants:
        for target in _variant_targets(variant, targets):
            if target not in df.columns:
                continue
            if not _target_is_binary(df[target], schema.get(target)):
                continue
            X, _, _ = _build_design(df, schema, variant.predictors)
            Xp, _ = _prune_by_vif(X, threshold=vif_threshold)
            y_enc, _ = _encode_target(df[target], positive_class.get(target))
            sub = pd.concat([y_enc.rename("_y"), Xp], axis=1).dropna()
            n_used = len(sub)
            # EPV power uses the minority class count, not the encoded positive class.
            if n_used:
                y_counts = sub["_y"].value_counts()
                n_events = int(y_counts.min()) if len(y_counts) >= 2 else 0
            else:
                n_events = 0
            n_params = len(Xp.columns)
            rows.append({
                "target": target,
                "model_id": variant.model_id,
                "model_title": variant.title,
                "model_link": variant.link,
                "experimental": variant.experimental,
                "n_rows_total": n_total,
                "n_complete_cases": n_used,
                "n_rows_dropped": n_total - n_used,
                "n_outcome_events": n_events,
                "n_design_columns": n_params,
                "epv": round(n_events / n_params, 1) if n_params else np.nan,
            })
    return pd.DataFrame(rows)


def export_calculator_meta(
    imputed_frames: list[pd.DataFrame],
    schema: dict[str, ColSpec],
    predictors: Sequence[str],
    pooled_df: pd.DataFrame,
    *,
    vif_threshold: float = 5.0,
) -> dict[str, Any]:
    """Structured model spec for the Streamlit calculator (reference levels + z-params)."""
    X, mapping, z_params = _build_design(imputed_frames[0], schema, predictors)
    Xp, _ = _prune_by_vif(X, threshold=vif_threshold)
    keep = set(Xp.columns)
    coef_by_col = {
        str(row["predictor_col"]): float(row["coef"])
        for _, row in pooled_df.iterrows()
        if pd.notna(row.get("coef"))
    }
    intercept = float(pooled_df["intercept_coef"].iloc[0])

    terms: list[dict[str, Any]] = []
    for pred in predictors:
        spec = schema.get(pred)
        if spec is None:
            continue
        cols = [c for c in mapping.get(pred, []) if c in keep]
        if not cols and pred in keep:
            cols = [pred]
        if not cols:
            continue

        if spec.kind in ("continuous", "count"):
            if pred not in coef_by_col:
                continue
            terms.append({
                "name": pred,
                "kind": "continuous",
                "coef": coef_by_col[pred],
                "z_mu": z_params[pred]["mu"],
                "z_sd": z_params[pred]["sd"],
                "z_log": bool(z_params[pred].get("log", False)),
            })
        elif spec.kind == "binary":
            if pred not in coef_by_col:
                continue
            terms.append({
                "name": pred,
                "kind": "binary",
                "coef": coef_by_col[pred],
            })
        elif spec.kind == "ordinal":
            if pred not in coef_by_col:
                continue
            s = imputed_frames[0][pred].dropna()
            if spec.ordered_levels:
                levels = [str(x) for x in spec.ordered_levels]
            else:
                cat = pd.Categorical(s, ordered=True)
                levels = [str(x) for x in cat.categories]
            terms.append({
                "name": pred,
                "kind": "ordinal",
                "coef": coef_by_col[pred],
                "levels": levels,
            })
        elif spec.kind == "nominal":
            dummy_cols = [c for c in cols if c in coef_by_col]
            if not dummy_cols:
                continue
            s = imputed_frames[0][pred].dropna().astype(str)
            full_dummies = pd.get_dummies(s, prefix=pred, drop_first=False, dtype=float)
            ref_col = sorted(full_dummies.columns)[0]
            reference = ref_col[len(pred) + 1:]
            levels = sorted(set(s.unique()) | {reference}, key=str)
            dummies = {
                col[len(pred) + 1:]: coef_by_col[col]
                for col in dummy_cols
            }
            terms.append({
                "name": pred,
                "kind": "categorical",
                "reference": reference,
                "levels": levels,
                "dummies": dummies,
            })

    return {
        "target": str(pooled_df["target"].iloc[0]),
        "intercept": intercept,
        "terms": terms,
    }


def run_inferential(
    imputed_frames: list[pd.DataFrame] | None,
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
    output_root: Path | str = "output",
    selection_candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Run multivariable logistic regression per target × model variant with Rubin
    pooling over the MICE imputations. Returns combined long-format results and
    writes tables, forest plot SVGs, and Streamlit JSON under ``output/inferential/``.

    Pass ``predictors=`` for a single model (legacy filenames), or ``variants=``
    for multiple named predictor sets (e.g. literature-based calculators).
    Each variant may set its own ``target``; otherwise all ``targets=`` are fit.

    Pass ``imputed_frames=None`` to load draws from
    ``output/missingness/mice/`` or ``output/datasets/`` (written by imputation).

    Stale per-variant CSV/SVG/JSON in ``output/inferential/`` is removed before
    each run so renamed or dropped models cannot linger in the report or calculator.

    ``selection_candidates`` is the EDA predictor pool handed to
    ``model_comparison.run_comparison_stage`` for its own top-k variable
    selection (combined-vs-single comparison tables). It is independent of
    ``predictors=``/``variants=``, which choose the multivariable models
    actually fitted here.
    """
    output_root = Path(output_root)
    if imputed_frames is None:
        from missingness_resolution import load_modeling_frames, read_mice_manifest

        imputed_frames = load_modeling_frames(output_root)
        # Rubin pooling is only valid on proper multiple imputation. RF chained
        # imputation is a labelled sensitivity method — warn loudly so pooled ORs
        # from it are never reported as formal MI without acknowledgement.
        manifest = read_mice_manifest(output_root)
        if manifest and not (
            manifest.get("proper_multiple_imputation")
            and manifest.get("rubin_pooling_supported")
        ):
            warnings.warn(
                "Rubin pooling on a sensitivity-analysis imputation "
                f"(method={manifest.get('method')!r}); these draws do not "
                "support formal multiple imputation. Use proper_mice_impute() "
                "for inferential results.",
                stacklevel=2,
            )

    figs_dir, tabs_dir = _ensure_dirs(output_root)
    _clear_inferential_artifacts(figs_dir, tabs_dir)
    positive_class = positive_class or {}
    model_variants = normalize_inferential_variants(predictors, variants)

    # Complete-case counts and EPV per variant. Written to
    # ``multivariable_cases.csv`` (the report's EPV gauge) and handed to
    # ``write_streamlit_artifacts`` so each calculator carries its own n/events.
    cases_df = summarize_multivariable_cases(
        imputed_frames[0], schema,
        targets=targets,
        variants=model_variants,
        positive_class=positive_class,
        vif_threshold=vif_threshold,
    )

    all_rows = []
    for variant in model_variants:
        for target in _variant_targets(variant, targets):
            if target not in imputed_frames[0].columns:
                continue
            spec = schema.get(target)
            if not _target_is_binary(imputed_frames[0][target], spec):
                warnings.warn(
                    f"Inferential skipped for '{target}' / '{variant.model_id or 'default'}': "
                    f"multivariable logistic requires a binary outcome "
                    f"(got kind={getattr(spec, 'kind', '?')}).",
                    stacklevel=2,
                )
                continue
            pooled_df, vif_df = fit_multivariable_logistic(
                imputed_frames, schema, target, variant.predictors,
                positive_class=positive_class.get(target),
                vif_threshold=vif_threshold,
            )
            pooled_df = pooled_df.copy()
            pooled_df["model_id"] = variant.model_id
            pooled_df["model_title"] = variant.title
            pooled_df["model_link"] = variant.link
            pooled_df["experimental"] = variant.experimental
            stem = artifact_base(target, variant.model_id)
            table_df = pooled_df.drop(columns=["model_title", "model_link"], errors="ignore")
            _format_inferential_table(table_df).to_csv(
                tabs_dir / f"{stem}__multivariable.csv", index=False)
            _format_table_for_csv(vif_df).to_csv(
                tabs_dir / f"{stem}__vif.csv", index=False)
            meta = export_calculator_meta(
                imputed_frames, schema, variant.predictors, pooled_df,
                vif_threshold=vif_threshold,
            )
            meta["model_id"] = variant.model_id
            meta["model_title"] = variant.title
            meta["model_link"] = variant.link
            meta["experimental"] = variant.experimental
            (tabs_dir / f"{stem}__calculator.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8",
            )
            _forest_plot(pooled_df, target, figs_dir, model_id=variant.model_id)
            all_rows.append(pooled_df)

    if not all_rows:
        empty = _empty_inferential_df()
        _format_inferential_table(empty).to_csv(
            tabs_dir / "inferential_summary.csv", index=False)
        return empty

    _format_table_for_csv(cases_df).to_csv(
        tabs_dir / "multivariable_cases.csv", index=False)

    from model_calculator import write_streamlit_artifacts

    artifact_paths = write_streamlit_artifacts(
        output_root,
        cases_df=cases_df,
        cohort_df=imputed_frames[0],
        schema=schema,
        vif_threshold=vif_threshold,
        # Same candidate pool and same data-selected model id as
        # run_comparison_stage below, so a data-selected model's calculator
        # artifact carries the *same* optimism-corrected AUC as the
        # combined-vs-single comparison table publishes for it, rather than
        # a second, uncorrected one (see model_calculator.write_streamlit_artifacts).
        selection_candidates=selection_candidates,
        selected_model_ids={"top_6_variables", "top_1_variable"},
    )
    _, figure_entries = _write_performance_figures(
        artifact_paths, figs_dir, model_variants)

    if selection_candidates is None:
        # Only run when the caller supplies a real candidate pool (the
        # notebook's EDA predictor list). Legacy/unit-test call sites that
        # fit a couple of ad-hoc predictors on a 4-row fixture have no
        # meaningful "does combining beat the best single" question to
        # answer, and ``vs.assert_reference`` would reject their top pick
        # against ``analysis.REFERENCE_VARIABLE`` (a real cohort column)
        # regardless of which toy predictor won. ``_clear_inferential_artifacts``
        # already removed any comparison tables from a previous run, so a
        # skip here leaves no stale numbers behind — just none at all.
        warnings.warn(
            "No selection_candidates given — the combined-vs-single comparison "
            "stage is skipped and its three tables are not written.",
            stacklevel=2,
        )
    else:
        from model_comparison import run_comparison_stage

        run_comparison_stage(
            imputed_frames[0], schema, model_variants, targets[0], tabs_dir,
            frames=imputed_frames,
            candidates=selection_candidates,   # EDA predictor pool, from the notebook
            selected_model_ids={"top_6_variables", "top_1_variable"},
        )

    # After the comparison stage, always: the figure reads its deltas and
    # intervals off the table that stage just wrote.
    _write_model_overview_figures(figure_entries, figs_dir, tabs_dir)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[[c for c in _INFERENTIAL_COLS if c in combined.columns]]
    _format_inferential_table(combined).to_csv(
        tabs_dir / "inferential_summary.csv", index=False)
    return combined


def run_inferential_stage(
    schema: dict[str, ColSpec],
    *,
    imputed_frames: list[pd.DataFrame] | None = None,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
    output_root: Path | str = "output",
    selection_candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Notebook entry point: load MICE parquets, fit Rubin-pooled models, write artifacts.

    Writes ``output/inferential/tables/``, ``figures/``, and
    ``output/inferential/model_artifacts/*.json`` (Streamlit calculator).

    ``selection_candidates`` is passed through to ``run_inferential`` for the
    combined-vs-single comparison stage's own variable selection.
    """
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SMConvergenceWarning)
        return run_inferential(
            imputed_frames,
            schema,
            targets=targets,
            predictors=predictors,
            variants=variants,
            positive_class=positive_class,
            vif_threshold=vif_threshold,
            output_root=output_root,
            selection_candidates=selection_candidates,
        )
