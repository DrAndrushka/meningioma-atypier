"""Rubin-pooled multivariable logistic regression on MICE-imputed cohorts.

Fits one model per variant (literature + experimental lists from ``config/07``).
VIF pruning, forest plots, calculator JSON, EPV / complete-case tables.
Removes stale per-variant artifacts at the start of each run.
Artifacts → ``output/inferential/``.
"""

from __future__ import annotations

import re
import warnings
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from schema_infer import ColSpec
from cleaning import format_table_for_csv as _format_table_for_csv  # CSV display-only rounding


@dataclass(frozen=True)
class InferentialModelVariant:
    """One multivariable model specification (outcome + predictor set + report label)."""

    model_id: str
    title: str
    link: str
    target: str
    predictors: tuple[str, ...]


EXPERIMENTAL_MODEL_ID = "experimental"
LEGACY_EXPERIMENTAL_MODEL_IDS = frozenset({"experimental_model"})


def is_experimental_model_id(model_id: str) -> bool:
    mid = str(model_id or "")
    if mid in LEGACY_EXPERIMENTAL_MODEL_IDS:
        return True
    if mid == EXPERIMENTAL_MODEL_ID:
        return True
    return mid.startswith(f"{EXPERIMENTAL_MODEL_ID}_")


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
                ))
            elif isinstance(v, dict):
                out.append(InferentialModelVariant(
                    model_id=_slug_model_id(str(v["id"])),
                    title=str(v.get("title") or v["id"]).strip(),
                    link=str(v.get("link") or "").strip(),
                    target=str(v.get("target") or default_target).strip(),
                    predictors=tuple(v["predictors"]),
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


def _clear_inferential_artifacts(figs_dir: Path, tabs_dir: Path) -> None:
    """Drop per-variant inferential outputs so a re-run cannot leave stale models."""
    for path in tabs_dir.iterdir():
        if path.is_file() and any(path.name.endswith(s) for s in _PER_MODEL_TABLE_SUFFIXES):
            path.unlink()
    for path in figs_dir.glob("*__forest.svg"):
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
            mu, sd = s.mean(), s.std(ddof=0)
            z = (s - mu) / _safe_z_denominator(sd)
            z.name = p
            pieces.append(z)
            mapping[p] = [p]
            z_params[p] = {"mu": float(mu), "sd": float(sd)}
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
        positive_class = True if True in nn else 1 if 1 in nn else sorted(nn, key=str)[-1]
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
                         "z_mu": z_mu, "z_sd": z_sd})
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

def _forest_plot(
    pooled: pd.DataFrame,
    target: str,
    figs_dir: Path,
    *,
    model_id: str = "",
    model_title: str = "",
) -> None:
    plot_df = pooled.dropna(subset=["or"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("or")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_df))))
    y = np.arange(len(plot_df))
    ax.errorbar(plot_df["or"], y,
                xerr=[plot_df["or"] - plot_df["or_ci_lo"],
                      plot_df["or_ci_hi"] - plot_df["or"]],
                fmt="o", color="#264653", ecolor="#2a9d8f", capsize=3)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(plot_df["predictor_col"])
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted Odds Ratio (95% CI, log scale)")
    title = f"Multivariable logistic — {target}"
    if model_title:
        title += f"\n{model_title}"
    ax.set_title(title)
    fig.tight_layout()
    stem = artifact_base(target, model_id)
    fig.savefig(figs_dir / f"{stem}__forest.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_INFERENTIAL_COLS = [
    "target", "model_id", "model_title", "model_link", "predictor_col", "or", "or_ci_lo", "or_ci_hi",
    "coef", "se", "p", "df", "n_models",
    "intercept_coef", "intercept_or", "z_mu", "z_sd",
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
            n_events = int(sub["_y"].sum()) if n_used else 0
            n_params = len(Xp.columns)
            rows.append({
                "target": target,
                "model_id": variant.model_id,
                "model_title": variant.title,
                "model_link": variant.link,
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
) -> pd.DataFrame:
    """
    Run multivariable logistic regression per target × model variant with Rubin
    pooling over the MICE imputations. Returns combined long-format results and
    writes tables + forest plot SVGs.

    Pass ``predictors=`` for a single model (legacy filenames), or ``variants=``
    for multiple named predictor sets (e.g. literature-based calculators).
    Each variant may set its own ``target``; otherwise all ``targets=`` are fit.

    Pass ``imputed_frames=None`` to load draws from
    ``output/missingness/mice/`` or ``output/datasets/`` (written by imputation).
    """
    output_root = Path(output_root)
    if imputed_frames is None:
        from missingness_resolution import load_modeling_frames

        imputed_frames = load_modeling_frames(output_root)

    figs_dir, tabs_dir = _ensure_dirs(output_root)
    _clear_inferential_artifacts(figs_dir, tabs_dir)
    positive_class = positive_class or {}
    model_variants = normalize_inferential_variants(predictors, variants)

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
            (tabs_dir / f"{stem}__calculator.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8",
            )
            _forest_plot(
                pooled_df, target, figs_dir,
                model_id=variant.model_id,
                model_title=variant.title,
            )
            all_rows.append(pooled_df)

    if not all_rows:
        empty = _empty_inferential_df()
        _format_inferential_table(empty).to_csv(
            tabs_dir / "inferential_summary.csv", index=False)
        return empty

    cases_df = summarize_multivariable_cases(
        imputed_frames[0], schema,
        targets=targets,
        variants=model_variants,
        positive_class=positive_class,
        vif_threshold=vif_threshold,
    )
    _format_table_for_csv(cases_df).to_csv(
        tabs_dir / "multivariable_cases.csv", index=False)

    from model_calculator import write_streamlit_artifacts

    write_streamlit_artifacts(
        output_root,
        cases_df=cases_df,
        cohort_df=imputed_frames[0],
        schema=schema,
        vif_threshold=vif_threshold,
    )

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[[c for c in _INFERENTIAL_COLS if c in combined.columns]]
    _format_inferential_table(combined).to_csv(
        tabs_dir / "inferential_summary.csv", index=False)
    return combined


def preview_multivariable_cases(
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
    output_root: Path | str = "output",
) -> pd.DataFrame:
    """EPV / complete-case preview table — loads modelling cohort from ``output/datasets/``."""
    from missingness_resolution import load_modeling_frames

    return summarize_multivariable_cases(
        load_modeling_frames(output_root)[0],
        schema,
        targets=targets,
        predictors=predictors,
        variants=variants,
        positive_class=positive_class,
        vif_threshold=vif_threshold,
    )


def run_inferential_stage(
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    variants: Sequence[InferentialModelVariant | dict | tuple | list] | None = None,
    positive_class: dict | None = None,
    vif_threshold: float = 5.0,
    output_root: Path | str = "output",
) -> pd.DataFrame:
    """Notebook entry point: load MICE parquets, fit Rubin-pooled models, write artifacts."""
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SMConvergenceWarning)
        return run_inferential(
            None,
            schema,
            targets=targets,
            predictors=predictors,
            variants=variants,
            positive_class=positive_class,
            vif_threshold=vif_threshold,
            output_root=output_root,
        )
