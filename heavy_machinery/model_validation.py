"""Bootstrap internal validation + shrink coefficients before they go in the json artifact."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from schema_infer import ColSpec


def build_complete_case_frame(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    predictors: Sequence[str],
    target: str,
    *,
    vif_threshold: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Complete-case modeling frame with the same design matrix as inferential."""
    from inferential import _build_design, _prune_by_vif

    X, _, _ = _build_design(df, schema, predictors)
    Xp, _ = _prune_by_vif(X, threshold=vif_threshold)
    design_cols = list(Xp.columns)
    if not design_cols:
        raise ValueError("No design columns available for validation")
    out = pd.concat([df[[target]], Xp], axis=1).dropna()
    if out.empty:
        raise ValueError("No complete cases available for validation")
    return out, design_cols


def _predict_logistic(
    X: pd.DataFrame,
    design_cols: list[str],
    coefficients: dict[str, float],
) -> np.ndarray:
    mat = sm.add_constant(X[design_cols].astype(float), has_constant="add")
    coef_vec = np.array([coefficients[c] for c in mat.columns], dtype=float)
    logit = mat.values @ coef_vec
    return 1.0 / (1.0 + np.exp(-logit))


def _calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    logit_pred = np.log(p / (1 - p))
    y = pd.Series(y_true).reset_index(drop=True).astype(int)
    X = sm.add_constant(pd.DataFrame({"logit_pred": logit_pred}), has_constant="add")
    cal_model = sm.Logit(y, X).fit(disp=False)
    return float(cal_model.params["logit_pred"])


def _round_metric(x: float, decimals: int = 3) -> float:
    return round(float(x), decimals)


def _validation_interpretation(
    auc_corr: float,
    brier_corr: float,
    baseline_brier: float,
    slope_corr: float,
) -> str:
    if auc_corr >= 0.75:
        disc = "good"
    elif auc_corr >= 0.70:
        disc = "moderate"
    elif auc_corr >= 0.60:
        disc = "fair"
    else:
        disc = "limited"

    brier_clause = (
        "remained better than the prevalence-only baseline"
        if brier_corr < baseline_brier
        else "did not clearly beat the prevalence-only baseline"
    )
    if slope_corr >= 0.95:
        cal_clause = "acceptable calibration for an exploratory research risk calculator"
    elif slope_corr >= 0.85:
        cal_clause = "mild overconfidence but acceptable calibration for an exploratory research risk calculator"
    else:
        cal_clause = "notable overconfidence; interpret predicted probabilities cautiously"

    return (
        f"The model showed {disc} discrimination after bootstrap internal validation. "
        f"The optimism-corrected Brier score {brier_clause}. "
        f"The optimism-corrected calibration slope suggested {cal_clause}."
    )


def bootstrap_internal_validation(
    model_df: pd.DataFrame,
    target: str,
    design_cols: list[str],
    coefficients: dict[str, float],
    *,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Apparent + optimism-corrected metrics and ROC points for the development sample."""
    y_orig = model_df[target].astype(int)
    X_orig = sm.add_constant(model_df[design_cols].astype(float), has_constant="add")
    apparent_fit = sm.Logit(y_orig, X_orig).fit(disp=False)
    y_pred_apparent = apparent_fit.predict(X_orig).values
    y_arr = y_orig.values

    auc_app = roc_auc_score(y_arr, y_pred_apparent)
    brier_app = brier_score_loss(y_arr, y_pred_apparent)
    baseline_brier = brier_score_loss(y_arr, np.repeat(y_arr.mean(), len(y_arr)))
    # Development-sample logistic has apparent calibration slope ≈ 1 by construction.
    slope_app = 1.0

    auc_optimisms: list[float] = []
    brier_optimisms: list[float] = []
    slope_optimisms: list[float] = []

    for i in range(n_bootstrap):
        boot_df = model_df.sample(n=len(model_df), replace=True, random_state=i).reset_index(drop=True)
        y_boot = boot_df[target].astype(int)
        X_boot = sm.add_constant(boot_df[design_cols].astype(float), has_constant="add")

        try:
            boot_result = sm.Logit(y_boot, X_boot).fit(disp=False)
            pred_boot = boot_result.predict(X_boot)
            pred_orig = boot_result.predict(X_orig)

            auc_optimisms.append(
                roc_auc_score(y_boot, pred_boot) - roc_auc_score(y_orig, pred_orig)
            )
            brier_optimisms.append(
                brier_score_loss(y_orig, pred_orig) - brier_score_loss(y_boot, pred_boot)
            )
            slope_optimisms.append(
                _calibration_slope(y_boot.values, pred_boot.values)
                - _calibration_slope(y_orig.values, pred_orig.values)
            )
        except Exception:
            continue

    if not auc_optimisms:
        raise RuntimeError("Bootstrap internal validation failed for all resamples")

    auc_corr = auc_app - float(np.mean(auc_optimisms))
    brier_corr = brier_app + float(np.mean(brier_optimisms))
    slope_corr = slope_app - float(np.mean(slope_optimisms))

    fpr, tpr, _ = roc_curve(y_arr, y_pred_apparent)

    metrics = [
        {
            "metric": "AUC",
            "apparent": _round_metric(auc_app),
            "optimism_corrected": _round_metric(auc_corr),
        },
        {
            "metric": "Brier score",
            "apparent": _round_metric(brier_app),
            "optimism_corrected": _round_metric(brier_corr),
        },
        {
            "metric": "Calibration slope",
            "apparent": _round_metric(slope_app),
            "optimism_corrected": _round_metric(slope_corr),
        },
    ]

    return {
        "method": "bootstrap internal validation",
        "bootstrap_resamples": n_bootstrap,
        "successful_bootstraps": len(auc_optimisms),
        "metrics": metrics,
        "baseline_brier": _round_metric(baseline_brier),
        "interpretation": _validation_interpretation(
            auc_corr, brier_corr, baseline_brier, slope_corr
        ),
        "auc_chart": {
            "title": "AUC before and after bootstrap internal validation",
            "series": [
                {
                    "value_field": "apparent",
                    "label": "Apparent (development sample)",
                    "color": "#2E86AB",
                },
                {
                    "value_field": "optimism_corrected",
                    "label": "Optimism-corrected",
                    "color": "#E94F37",
                },
            ],
        },
        "roc_curves": {
            "title": "ROC curve (apparent model, development sample)",
            "curves": [
                {
                    "series": "apparent",
                    "label": "Apparent",
                    "color": "#2E86AB",
                    "auc": _round_metric(auc_app),
                    "fpr": [round(float(x), 6) for x in fpr],
                    "tpr": [round(float(x), 6) for x in tpr],
                }
            ],
        },
        "corrected_calibration_slope": slope_corr,
    }


def shrink_and_recalibrate_coefficients(
    coefficients: dict[str, float],
    design_cols: list[str],
    model_df: pd.DataFrame,
    target: str,
    shrinkage_factor: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Uniform slope shrinkage on predictors + intercept recalibration to prevalence."""
    shrunk = dict(coefficients)
    for col in design_cols:
        if col in shrunk:
            shrunk[col] = float(shrunk[col]) * float(shrinkage_factor)

    X = model_df[design_cols].astype(float)
    y = model_df[target].astype(int)
    linear_part = X.values @ np.array([shrunk[c] for c in design_cols], dtype=float)
    prevalence = float(y.mean())

    def mean_prediction_error(intercept: float) -> float:
        p = 1.0 / (1.0 + np.exp(-(intercept + linear_part)))
        return float(p.mean() - prevalence)

    shrunk["const"] = float(brentq(mean_prediction_error, -20.0, 20.0))

    processing = {
        "shrinkage_applied": True,
        "shrinkage_factor": _round_metric(shrinkage_factor),
        "intercept_recalibrated": True,
        "notes": (
            "Non-intercept coefficients were uniformly shrunk by the optimism-corrected "
            "calibration slope. The intercept was recalibrated so mean predicted risk "
            "matched observed cohort prevalence."
        ),
    }
    return shrunk, processing


def enrich_streamlit_artifact(
    artifact: dict[str, Any],
    model_df: pd.DataFrame,
    design_cols: list[str],
    *,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Add validation charts, prose, shrinkage, and recalibrated coefficients."""
    target = str(artifact["target"])
    validation = bootstrap_internal_validation(
        model_df,
        target,
        design_cols,
        artifact["coefficients"],
        n_bootstrap=n_bootstrap,
    )
    shrunk_coefs, processing = shrink_and_recalibrate_coefficients(
        artifact["coefficients"],
        design_cols,
        model_df,
        target,
        validation["corrected_calibration_slope"],
    )

    out = dict(artifact)
    out["coefficients"] = shrunk_coefs
    out["created_from"] = "bootstrap-shrunken multivariable logistic regression"
    out["version"] = out.get("version", "v1")
    out["validation"] = {
        k: v for k, v in validation.items() if k != "corrected_calibration_slope"
    }
    out["coefficient_processing"] = processing
    out["missing_data_policy"] = (
        "Binary imaging variables were not imputed because missing values represented "
        "unknown/unrecorded findings rather than confirmed absence. The model was fitted "
        "on complete cases for included predictors."
    )
    out["clinical_note"] = (
        "This is an internally validated exploratory research calculator and not a "
        "standalone clinical decision-making tool. External validation on an independent "
        "cohort is required before clinical use."
    )
    if target == "high_grade":
        out["model_name"] = "High-grade meningioma risk calculator"
        out["outcome_definition"] = (
            "1 = high-grade meningioma, 0 = non-high-grade meningioma"
        )
    return out
