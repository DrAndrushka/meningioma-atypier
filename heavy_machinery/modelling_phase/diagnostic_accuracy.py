"""Radiology-style diagnostic accuracy for binary MRI signs vs binary outcomes.

Sensitivity, specificity, PPV, NPV, Wilson 95% CIs, and paper-style AUC.
Complements EDA; not multivariable modelling. Artifacts → ``output/eda/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportion_confint

from cleaning import format_table_for_csv as _format_table_for_csv
from eda import _chi2_row, _encode_binary_target, _infer_target_kind, benjamini_hochberg
from schema_infer import ColSpec

_SKIP_NOTE = "Skipped: requires predefined cutoff"

# Clinically meaningful binary contrasts derived from nominal predictors.
_DERIVED_CONTRASTS: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "tumor_location_non_skull_base": (
        "tumor_location",
        lambda s: s == "non_skull_base",
    ),
    "tumor_margin_irregular": (
        "tumor_margin",
        lambda s: s == "irregular",
    ),
    "sex_male": (
        "sex",
        lambda s: s == "male",
    ),
}

_PRESENT_VALUES = {True, 1, 1.0, "True", "true", "1", "yes", "Yes", "Y", "y"}


def _ensure_dirs(root: Path) -> Path:
    tabs = root / "eda" / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    return tabs


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def _wilson_ci(successes: int, nobs: int) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if nobs <= 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(successes, nobs, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def _odds_ratio_ci(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float]:
    """Odds ratio for the 2×2, with a Woolf (log-scale) 95% interval.

    The effect size the meningioma literature quotes for a dichotomised
    imaging feature — Magill 2018 reports OR 1.69 for > 3 cm and 3.01 for
    > 6 cm — so a cut-point here can be read next to a published one.

    A zero cell makes the OR 0 or infinite and the log interval undefined, so
    the Haldane–Anscombe correction (add 0.5 to every cell) is applied in that
    case. It is a real change to the estimate, which is why it only happens
    when the alternative is no estimate at all.
    """
    cells = [tp, fp, fn, tn]
    if any(c < 0 for c in cells) or (tp + fn) == 0 or (fp + tn) == 0:
        return np.nan, np.nan, np.nan
    a, b, c, d = (float(v) + 0.5 for v in cells) if 0 in cells else map(float, cells)
    if b == 0 or c == 0:  # both margins non-empty, so this cannot happen
        return np.nan, np.nan, np.nan
    odds_ratio = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = np.log(odds_ratio)
    return (float(odds_ratio),
            float(np.exp(log_or - 1.96 * se)),
            float(np.exp(log_or + 1.96 * se)))


def _encode_feature_present(x: pd.Series) -> pd.Series:
    """Map predictor to 1 = feature present, 0 = absent; preserve NaN."""
    if pd.api.types.is_bool_dtype(x):
        return x.astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    mask = x.notna()
    if not mask.any():
        return out
    vals = x[mask]
    if vals.isin(_PRESENT_VALUES).any() or (~vals.isin(_PRESENT_VALUES)).any():
        out.loc[mask] = vals.isin(_PRESENT_VALUES).astype(float)
    return out


def _contingency_pvalue(tp: int, fp: int, fn: int, tn: int) -> tuple[float, str]:
    """χ²/Fisher for the 2×2, or nothing when there is nothing to test.

    A rule that flags nobody — an AND of two rare signs, say — gives a table
    with an empty row. There is no association to test in that table, and
    ``chi2_contingency`` raises rather than saying so. The accuracy numbers
    around it are still real, so the p-value is reported missing and the row
    survives.
    """
    table = np.array([[tp, fp], [fn, tn]], dtype=float)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return float("nan"), "not applicable"
    row = _chi2_row(table)
    return float(row["p"]), str(row["test"])


def binary_diagnostic_metrics(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    *,
    positive_class=None,
    predictor_series: pd.Series | None = None,
) -> dict:
    """
    Diagnostic accuracy for one binary target × one binary predictor.

    Parameters
    ----------
    df               : analysis-ready dataframe.
    target           : binary outcome column name.
    predictor        : label for the predictor row (may be a derived contrast name).
    positive_class   : value coded as outcome-positive (1); auto-detected if None.
    predictor_series : optional 0/1 (or boolean) series when ``predictor`` is derived.

    Returns
    -------
    dict with keys: predictor, n_used, TP, FP, FN, TN, sensitivity, specificity,
    PPV, NPV, accuracy, AUC, p, test, note.
    """
    y_enc, _ = _encode_binary_target(df[target], positive_class)
    x_raw = predictor_series if predictor_series is not None else df[predictor]
    x_enc = _encode_feature_present(x_raw)

    pair = pd.concat([y_enc.rename("_y"), x_enc.rename("_x")], axis=1).dropna()
    n_used = len(pair)
    if n_used == 0:
        return _empty_metrics(predictor, note="No complete cases")

    y = pair["_y"].astype(int)
    x = pair["_x"].astype(int)

    tp = int(((x == 1) & (y == 1)).sum())
    fp = int(((x == 1) & (y == 0)).sum())
    fn = int(((x == 0) & (y == 1)).sum())
    tn = int(((x == 0) & (y == 0)).sum())

    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    accuracy = _safe_ratio(tp + tn, n_used)
    if np.isnan(sensitivity) or np.isnan(specificity):
        auc = np.nan
    else:
        auc = (sensitivity + specificity) / 2.0

    p, test = _contingency_pvalue(tp, fp, fn, tn)
    odds_ratio, or_lo, or_hi = _odds_ratio_ci(tp, fp, fn, tn)

    sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
    spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
    ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp)
    npv_lo, npv_hi = _wilson_ci(tn, tn + fn)
    acc_lo, acc_hi = _wilson_ci(tp + tn, n_used)

    return {
        "predictor": predictor,
        "n_used": n_used,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sensitivity,
        "sensitivity_lo": sens_lo,
        "sensitivity_hi": sens_hi,
        "specificity": specificity,
        "specificity_lo": spec_lo,
        "specificity_hi": spec_hi,
        "PPV": ppv,
        "PPV_lo": ppv_lo,
        "PPV_hi": ppv_hi,
        "NPV": npv,
        "NPV_lo": npv_lo,
        "NPV_hi": npv_hi,
        "accuracy": accuracy,
        "accuracy_lo": acc_lo,
        "accuracy_hi": acc_hi,
        "AUC": auc,
        "OR": odds_ratio,
        "OR_lo": or_lo,
        "OR_hi": or_hi,
        "p": p,
        "test": test,
        "note": "",
    }


def _empty_metrics(predictor: str, *, note: str = "") -> dict:
    nan_cols = (
        "sensitivity", "sensitivity_lo", "sensitivity_hi",
        "specificity", "specificity_lo", "specificity_hi",
        "PPV", "PPV_lo", "PPV_hi",
        "NPV", "NPV_lo", "NPV_hi",
        "accuracy", "accuracy_lo", "accuracy_hi",
        "AUC", "OR", "OR_lo", "OR_hi",
    )
    row = {
        "predictor": predictor,
        "n_used": 0,
        "TP": np.nan,
        "FP": np.nan,
        "FN": np.nan,
        "TN": np.nan,
        "p": np.nan,
        "test": "",
        "note": note,
    }
    for col in nan_cols:
        row[col] = np.nan
    return row


def _skipped_row(target: str, predictor: str, kind: str, note: str = _SKIP_NOTE) -> dict:
    row = _empty_metrics(predictor, note=note)
    row.update({"target": target, "kind": kind})
    return row


def _derived_series(df: pd.DataFrame, derived_name: str) -> pd.Series | None:
    if derived_name in df.columns:
        return df[derived_name]
    if derived_name not in _DERIVED_CONTRASTS:
        return None
    source, fn = _DERIVED_CONTRASTS[derived_name]
    if source not in df.columns:
        return None
    return fn(df[source])


def _evaluation_plan(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    predictors: Sequence[str],
) -> list[tuple[str, str, str | None, pd.Series | None]]:
    """
    Return (predictor_label, source_col, kind, optional_series) rows to evaluate.
    """
    plan: list[tuple[str, str, str | None, pd.Series | None]] = []
    for pred in predictors:
        if pred not in df.columns:
            continue
        spec = schema.get(pred)
        kind = spec.kind if spec is not None else "unknown"

        if kind == "binary":
            plan.append((pred, pred, kind, None))
            continue

        if kind == "nominal":
            derived = [
                name for name, (source, _) in _DERIVED_CONTRASTS.items()
                if source == pred
            ]
            if derived:
                for name in derived:
                    series = _derived_series(df, name)
                    if series is not None:
                        plan.append((name, pred, "derived_binary", series))
            else:
                plan.append((pred, pred, kind, None))
            continue

        plan.append((pred, pred, kind, None))

    return plan


def screen_diagnostic_accuracy(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    positive_class: dict | None = None,
    fdr_alpha: float = 0.05,
    output_root: Path | str = "output",
) -> pd.DataFrame:
    """
    Compute univariate diagnostic accuracy for each (binary target × predictor).

    Ordinal / continuous / count predictors are retained with a skip note.
    Nominal predictors emit predefined derived binary contrasts when available.
    """
    output_root = Path(output_root)
    tabs_dir = _ensure_dirs(output_root)
    positive_class = positive_class or {}

    testable_kinds = {"continuous", "count", "ordinal", "nominal", "binary", "datetime"}
    if predictors is None or len(predictors) == 0:
        predictors = [
            c for c, sp in schema.items()
            if c in df.columns and sp.keep and sp.kind in testable_kinds and c not in targets
        ]

    rows: list[dict] = []
    for target in targets:
        if target not in df.columns:
            continue
        target_spec = schema.get(target)
        target_mode = _infer_target_kind(df[target], target_spec)
        if target_mode != "binary":
            warnings.warn(
                f"Diagnostic accuracy skipped target '{target}': requires binary outcome "
                f"(kind={target_mode}).",
                stacklevel=2,
            )
            continue

        pos_cfg = positive_class.get(target)
        plan = _evaluation_plan(df, schema, predictors)

        for label, source, kind, series in plan:
            if kind in ("continuous", "count", "ordinal", "datetime"):
                rows.append(_skipped_row(target, label, kind))
                continue
            if kind == "nominal":
                rows.append(_skipped_row(target, label, kind))
                continue
            if kind == "unknown":
                rows.append(_skipped_row(target, label, kind, note="Skipped: unknown predictor kind"))
                continue

            if series is None and label not in df.columns:
                rows.append(_skipped_row(target, label, kind or "binary", note="Skipped: column missing"))
                continue

            metrics = binary_diagnostic_metrics(
                df,
                target,
                label,
                positive_class=pos_cfg,
                predictor_series=series,
            )
            metrics.update({"target": target, "kind": kind or "binary"})
            rows.append(metrics)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "target", "predictor", "kind", "n_used", "TP", "FP", "FN", "TN",
            "sensitivity", "sensitivity_lo", "sensitivity_hi",
            "specificity", "specificity_lo", "specificity_hi",
            "PPV", "PPV_lo", "PPV_hi",
            "NPV", "NPV_lo", "NPV_hi",
            "accuracy", "accuracy_lo", "accuracy_hi",
            "AUC", "p", "test", "p_fdr", "fdr_significant", "note",
        ])
        _format_table_for_csv(out).to_csv(tabs_dir / "diagnostic_accuracy.csv", index=False)
        return out

    out["p_fdr"] = np.nan
    for t in out["target"].unique():
        mask = (out["target"] == t) & out["note"].eq("")
        if mask.any():
            out.loc[mask, "p_fdr"] = benjamini_hochberg(out.loc[mask, "p"]).values
    out["fdr_significant"] = out["p_fdr"] < fdr_alpha

    cols = [
        "target", "predictor", "kind", "n_used", "TP", "FP", "FN", "TN",
        "sensitivity", "sensitivity_lo", "sensitivity_hi",
        "specificity", "specificity_lo", "specificity_hi",
        "PPV", "PPV_lo", "PPV_hi",
        "NPV", "NPV_lo", "NPV_hi",
        "accuracy", "accuracy_lo", "accuracy_hi",
        "AUC", "p", "test", "p_fdr", "fdr_significant", "note",
    ]
    out = out[cols].sort_values(
        ["target", "p_fdr", "sensitivity"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    _format_table_for_csv(out).to_csv(tabs_dir / "diagnostic_accuracy.csv", index=False)
    return out
