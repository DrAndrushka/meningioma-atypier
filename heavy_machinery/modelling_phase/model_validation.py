"""Bootstrap internal validation and coefficient shrinkage for calculator export.

Optimism-corrected AUC/Brier, calibration slope, and shrunken coefficients merged into
``output/inferential/model_artifacts/<target>_<model_id>_model.json`` by
``model_calculator.write_streamlit_artifacts``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from schema_infer import ColSpec

# Chart colours travel with the artifact so the Streamlit calculator matches the
# report; both come from the pipeline's shared Okabe–Ito palette.
_APPARENT_COLOR = "#666666"
_CORRECTED_COLOR = "#0072B2"


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


def _calibration_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calibration-in-the-large: offset of the logit with the slope fixed at 1."""
    eps = 1e-6
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    offset = np.log(p / (1 - p))
    y = pd.Series(y_true).reset_index(drop=True).astype(int)
    X = sm.add_constant(pd.DataFrame({"_": np.zeros(len(y))}), has_constant="add")
    fit = sm.Logit(y, X[["const"]], offset=offset).fit(disp=False)
    return float(fit.params["const"])


def _calibration_data(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int = 10,
) -> dict[str, Any]:
    """Binned observed-vs-predicted risk plus a LOESS smooth of the same data.

    Bins are risk quantiles, so each carries a comparable number of patients;
    equal-width bins would leave the high-risk end nearly empty at this
    prevalence. Counts travel with each bin so the reader can weight them.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    n = int(y.size)

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    bins: list[dict[str, Any]] = []
    if edges.size >= 2:
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, edges.size - 2)
        for b in range(edges.size - 1):
            mask = idx == b
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            bins.append({
                "predicted": _round_metric(float(p[mask].mean()), 4),
                "observed": _round_metric(float(y[mask].mean()), 4),
                "events": int(y[mask].sum()),
                "n": n_b,
            })

    smooth: dict[str, list[float]] = {}
    if n >= 20 and float(np.std(p)) > 0:
        # it=0 is required for a binary outcome: LOWESS's robustness iterations
        # treat the 1s as outliers among their mostly-0 neighbours and drag the
        # whole curve to the floor.
        curve = lowess(y, p, frac=0.66, it=0, return_sorted=True)
        smooth = {
            "predicted": [round(float(v), 4) for v in curve[:, 0]],
            "observed": [round(float(np.clip(v, 0.0, 1.0)), 4) for v in curve[:, 1]],
        }

    return {
        "title": "Calibration (apparent, development sample)",
        "bins": bins,
        "smooth": smooth,
        "observed_rate": _round_metric(float(y.mean()), 4),
    }


def _net_benefit_data(
    y_true: np.ndarray, y_pred: np.ndarray, *, max_threshold: float = 0.8,
) -> dict[str, Any]:
    """Decision-curve net benefit for the model, treat-all, and treat-none.

    Net benefit = TP/n − FP/n · t/(1−t): the share of true positives found,
    penalised by false positives at the exchange rate a clinician implies by
    acting at threshold ``t``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    n = int(y.size)
    if n == 0:
        return {"thresholds": [], "model": [], "treat_all": []}

    prevalence = float(y.mean())
    thresholds = np.round(np.arange(0.01, float(max_threshold) + 1e-9, 0.01), 4)
    odds = thresholds / (1.0 - thresholds)

    model_nb, all_nb = [], []
    for t, w in zip(thresholds, odds):
        flagged = p >= t
        tp = float((flagged & (y == 1)).sum())
        fp = float((flagged & (y == 0)).sum())
        model_nb.append(round(tp / n - (fp / n) * w, 5))
        all_nb.append(round(prevalence - (1.0 - prevalence) * w, 5))

    return {
        "title": "Decision curve (apparent, development sample)",
        "thresholds": [float(t) for t in thresholds],
        "model": model_nb,
        "treat_all": all_nb,
        "prevalence": _round_metric(prevalence, 4),
    }


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
    # Development-sample logistic has apparent calibration slope ≈ 1 and apparent
    # calibration-in-the-large ≈ 0 by construction. Both are measured rather than
    # asserted for the intercept, because a near-separable fit can drift off zero.
    slope_app = 1.0
    intercept_app = _calibration_intercept(y_arr, y_pred_apparent)

    auc_optimisms: list[float] = []
    brier_optimisms: list[float] = []
    slope_optimisms: list[float] = []
    intercept_optimisms: list[float] = []

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
            # Calibration-in-the-large drifts under resampling for the same
            # reason the slope does, and a model can be well-calibrated in
            # slope while systematically over- or under-predicting. Correcting
            # only the slope reports half the calibration.
            intercept_optimisms.append(
                _calibration_intercept(y_boot.values, pred_boot.values)
                - _calibration_intercept(y_orig.values, pred_orig.values)
            )
        except Exception:
            continue

    if not auc_optimisms:
        raise RuntimeError("Bootstrap internal validation failed for all resamples")

    auc_corr = auc_app - float(np.mean(auc_optimisms))
    brier_corr = brier_app + float(np.mean(brier_optimisms))
    slope_corr = slope_app - float(np.mean(slope_optimisms))
    intercept_corr = (intercept_app - float(np.mean(intercept_optimisms))
                      if intercept_optimisms else float("nan"))

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
        {
            "metric": "Calibration intercept",
            "apparent": _round_metric(intercept_app),
            "optimism_corrected": _round_metric(intercept_corr),
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
                    "color": _APPARENT_COLOR,
                },
                {
                    "value_field": "optimism_corrected",
                    "label": "Optimism-corrected",
                    "color": _CORRECTED_COLOR,
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
        "calibration": {
            **_calibration_data(y_arr, y_pred_apparent),
            "slope_apparent": _round_metric(slope_app),
            "slope_corrected": _round_metric(slope_corr),
            "intercept_apparent": _round_metric(intercept_app),
            "intercept_corrected": _round_metric(intercept_corr),
        },
        "decision_curve": _net_benefit_data(y_arr, y_pred_apparent),
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
