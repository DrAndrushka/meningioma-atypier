"""Bootstrap internal validation and coefficient shrinkage for calculator export.

Optimism-corrected AUC/Brier, calibration slope, and shrunken coefficients merged into
``output/inferential/model_artifacts/<target>_<model_id>_model.json`` by
``model_calculator.write_streamlit_artifacts``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from plot_style import roc_auc as _auc
from schema_infer import ColSpec

from heavy_machinery.config import load

analysis = load("analysis")

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


# ---------------------------------------------------------------------------
# Inner-loop primitives
# ---------------------------------------------------------------------------
# Everything below runs ~1000 times per model. The pandas/sklearn wrappers cost
# far more than the arithmetic at this sample size, so the loop works on plain
# float64 arrays. The numbers are unchanged: statsmodels unwraps a DataFrame to
# the same ndarray before fitting, and ``plot_style.roc_auc``/``_brier`` are the
# closed forms of the sklearn metrics they replace (see test_model_validation.py).


def _brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error of the predicted risk — ``brier_score_loss`` verbatim."""
    return float(np.mean((y_pred - y_true) ** 2))


def _logit_clipped(y_pred: np.ndarray) -> np.ndarray:
    """Log-odds of a predicted risk, clipped off 0 and 1 so the log is finite."""
    eps = 1e-6
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _quiet_logit_fit(y, X, **kwargs):
    """``sm.Logit(y, X).fit()`` with ConvergenceWarning silenced at the source.

    Every logistic fit in this module goes through here. The stage-level guard
    in ``inferential.run_inferential_stage`` is per-process state and does not
    reach the joblib/loky workers this module also runs inside, so a guard that
    lives with the caller silences some fits and not others — which is exactly
    how a couple of bare statsmodels warnings used to escape an otherwise
    silent pipeline run. Non-convergence is still *observable*: callers read
    ``mle_retvals["converged"]`` on the returned fit, and
    ``bootstrap_internal_validation`` counts and reports it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # check_rank=False skips the QR-based rank check statsmodels runs in
        # every model constructor. At the ~200k fits a pipeline run makes, that
        # check cost more than the Newton iterations it guards, and the fitted
        # params are bit-for-bit the same with it off.
        return sm.Logit(y, X, check_rank=False, **kwargs).fit(disp=False)


def _calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    logit_pred = _logit_clipped(y_pred)
    y = np.asarray(y_true).astype(int)
    # [const, logit_pred] — the column order sm.add_constant(prepend=True) uses.
    X = np.column_stack([np.ones(y.size), logit_pred])
    return float(_quiet_logit_fit(y, X).params[1])


def _calibration_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calibration-in-the-large: offset of the logit with the slope fixed at 1."""
    offset = _logit_clipped(y_pred)
    y = np.asarray(y_true).astype(int)
    X = np.ones((y.size, 1))
    return float(_quiet_logit_fit(y, X, offset=offset).params[0])


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


# One run asks for the same model's bootstrap more than once: the calculator
# export (``enrich_streamlit_artifact``) and the combined-vs-single comparison
# (``model_comparison.run_comparison_stage``) validate the same 11 models, from
# the same frame, on the same pre-drawn resample indices. The second answer is
# the first answer recomputed a thousand logistic fits at a time.
#
# Keyed on everything the result depends on — outcome, design columns, the
# bytes of the frame those columns are read from, the resample count, the
# selector, and whether calibration was asked for — so a hit is the object the
# recomputation would have produced. ``coefficients`` is deliberately not in
# the key: it is an unused parameter (the function refits the apparent model
# from the data), and keying on it would miss every legitimate hit.
_VALIDATION_CACHE: dict[Any, dict[str, Any]] = {}


def clear_validation_cache() -> None:
    """Empty the memo. For tests, and for a caller that mutates a frame in place."""
    _VALIDATION_CACHE.clear()


def _validation_key(
    model_df: pd.DataFrame,
    target: str,
    design_cols: Sequence[str],
    *,
    n_bootstrap: int,
    select,
    calibration: bool,
):
    # With a selector the resamples re-pick from the WHOLE frame, so the whole
    # frame is what the answer depends on; without one, only the model columns.
    frame = model_df if select is not None else model_df[[target, *design_cols]]
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()
    sel = None
    if select is not None:
        fn = getattr(select, "func", select)
        sel = (getattr(fn, "__qualname__", repr(fn)),
               repr(getattr(select, "args", ())),
               repr(sorted(getattr(select, "keywords", {}).items())))
    return (target, tuple(design_cols), int(n_bootstrap), digest, sel,
            bool(calibration))


def bootstrap_internal_validation(
    model_df: pd.DataFrame,
    target: str,
    design_cols: list[str],
    coefficients: dict[str, float],
    *,
    n_bootstrap: int = analysis.BOOTSTRAP_RESAMPLES,
    return_resample_aucs: bool = False,
    select=None,
    calibration: bool = True,
) -> dict[str, Any]:
    """Apparent + optimism-corrected metrics and ROC points for the development sample.

    ``calibration=False`` drops the calibration slope and intercept and their
    optimism, leaving both ``nan``. They cost four extra logistic fits in every
    resample — most of the loop — and the single-predictor yardstick in
    ``model_comparison.fit_single_predictors`` reads neither.

    Results are memoised (see ``_VALIDATION_CACHE``): asking twice for the same
    model on the same frame returns the first answer rather than recomputing it.
    """
    key = _validation_key(model_df, target, design_cols,
                          n_bootstrap=n_bootstrap, select=select,
                          calibration=calibration)
    hit = _VALIDATION_CACHE.get(key)
    if hit is not None and (not return_resample_aucs or "resample_aucs" in hit):
        cached = copy.deepcopy(hit)
        if not return_resample_aucs:
            cached.pop("resample_aucs", None)
            cached.pop("resample_ids", None)
        return cached
    y_arr = model_df[target].astype(int).to_numpy()
    n_rows = y_arr.size
    # Design matrix built once as float64 with the constant prepended, matching
    # sm.add_constant(..., has_constant="add"). Resampling then indexes rows of
    # this array instead of rebuilding a DataFrame 1000 times.
    X_orig = np.column_stack(
        [np.ones(n_rows), model_df[design_cols].astype(float).to_numpy()]
    )
    apparent_fit = _quiet_logit_fit(y_arr, X_orig)
    y_pred_apparent = np.asarray(apparent_fit.predict(X_orig), dtype=float)

    auc_app = _auc(y_arr, y_pred_apparent)
    brier_app = _brier(y_arr, y_pred_apparent)
    baseline_brier = _brier(y_arr, np.repeat(y_arr.mean(), n_rows))
    # Development-sample logistic has apparent calibration slope ≈ 1 and apparent
    # calibration-in-the-large ≈ 0 by construction. Both are measured rather than
    # asserted for the intercept, because a near-separable fit can drift off zero.
    slope_app = 1.0
    intercept_app = (_calibration_intercept(y_arr, y_pred_apparent)
                     if calibration else float("nan"))

    auc_optimisms: list[float] = []
    brier_optimisms: list[float] = []
    slope_optimisms: list[float] = []
    intercept_optimisms: list[float] = []
    resample_aucs: list[float] = []
    # Which resample each entry of `resample_aucs` came from. Appended in
    # lockstep with it, so a caller differencing two models' vectors can
    # align on the draw rather than on list position. Position alone is not
    # enough: a resample that selects nothing or whose fit raises is skipped
    # here, and one arm skipping a draw the other kept shifts everything
    # after it while leaving both lists a plausible length.
    resample_ids: list[int] = []
    selection_counts: dict[str, int] = {}
    # Denominator for `selection_counts`, sourced from THIS loop and no other:
    # how many resamples the selector actually ran on and returned a non-empty
    # pick for (the same event that increments selection_counts below, so the
    # two can never drift apart). This is deliberately NOT `n_bootstrap` (a
    # resample whose selector returns nothing is never tallied) and NOT
    # `successful_bootstraps` (a resample can be tallied here and still have
    # its fit fail afterwards — see the comment at the tally site). Reusing
    # either of those, or a resample count from an unrelated bootstrap loop
    # such as the paired patient-resampling one in model_comparison.py, is
    # exactly the "two numbers, one name" pattern that has already caused
    # separate defects on this branch.
    selection_resamples = 0
    # Resamples whose refit hit the iteration limit instead of converging. The
    # `except` below only catches fits that RAISE; a non-converged fit returns
    # normally and its optimism still lands in the means above, so nothing else
    # here would notice. Counted rather than dropped: on 352 patients a handful
    # of resamples separate by luck, and discarding them would bias the
    # optimism estimate toward the well-behaved draws. Surfaced below so the
    # count cannot climb from 2 to 500 invisibly.
    n_not_converged = 0

    # One index matrix for every resample, drawn from the single master seed
    # (see BOOTSTRAP_SEED) instead of reseeding per-resample. This is what lets
    # two different models validated in the same run see identical patients at
    # resample i, so their AUCs at i can be differenced as a paired sample.
    idx_matrix = _resample_indices(n_rows, n_bootstrap)

    for i in range(n_bootstrap):
        idx = idx_matrix[i]
        y_boot = y_arr[idx]
        if select is None:
            X_boot, X_score = X_orig[idx], X_orig
        else:
            # Re-run the *choosing* on this resample, not just the fitting.
            boot_frame = model_df.iloc[idx].reset_index(drop=True)
            cols_i = select(boot_frame, y_boot)
            if not cols_i:
                continue
            # Tallied on *attempt*, before the fit `try` below: a variable
            # this resample chose is counted here even if the subsequent fit
            # then fails and the resample is skipped. So sum(selection_counts
            # .values()) counts attempted selections, not successful ones —
            # it is not, in general, successful_bootstraps * k.
            selection_resamples += 1
            for c in cols_i:
                selection_counts[c] = selection_counts.get(c, 0) + 1
            X_boot = np.column_stack(
                [np.ones(n_rows), boot_frame[cols_i].astype(float).to_numpy()])
            X_score = np.column_stack(
                [np.ones(n_rows), model_df[cols_i].astype(float).to_numpy()])

        try:
            boot_result = _quiet_logit_fit(y_boot, X_boot)
            if not boot_result.mle_retvals.get("converged", True):
                n_not_converged += 1
            pred_boot = np.asarray(boot_result.predict(X_boot), dtype=float)
            pred_orig = np.asarray(boot_result.predict(X_score), dtype=float)

            # Computed once, used twice: the optimism needs this AUC and so
            # does the resample vector. Collected unconditionally so a cached
            # validation can also serve a later caller that wants the vector.
            auc_orig = _auc(y_arr, pred_orig)
            resample_aucs.append(_round_metric(auc_orig, 6))
            resample_ids.append(i)

            auc_optimisms.append(
                _auc(y_boot, pred_boot) - auc_orig
            )
            brier_optimisms.append(
                _brier(y_arr, pred_orig) - _brier(y_boot, pred_boot)
            )
            if calibration:
                slope_optimisms.append(
                    _calibration_slope(y_boot, pred_boot)
                    - _calibration_slope(y_arr, pred_orig)
                )
            # Calibration-in-the-large drifts under resampling for the same
            # reason the slope does, and a model can be well-calibrated in
            # slope while systematically over- or under-predicting. Correcting
            # only the slope reports half the calibration.
            if calibration:
                intercept_optimisms.append(
                    _calibration_intercept(y_boot, pred_boot)
                    - _calibration_intercept(y_arr, pred_orig)
                )
        except Exception:
            continue

    if not auc_optimisms:
        raise RuntimeError("Bootstrap internal validation failed for all resamples")

    auc_corr = auc_app - float(np.mean(auc_optimisms))
    brier_corr = brier_app + float(np.mean(brier_optimisms))
    slope_corr = (slope_app - float(np.mean(slope_optimisms))
                  if slope_optimisms else float("nan"))
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

    result = {
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
            **(_calibration_data(y_arr, y_pred_apparent) if calibration else {}),
            "slope_apparent": _round_metric(slope_app),
            "slope_corrected": _round_metric(slope_corr),
            "intercept_apparent": _round_metric(intercept_app),
            "intercept_corrected": _round_metric(intercept_corr),
        },
        "decision_curve": _net_benefit_data(y_arr, y_pred_apparent),
        "corrected_calibration_slope": slope_corr,
    }
    # 6 decimals, not the usual 3 — these get differenced against another
    # model's resample_aucs, and 3 decimals would quantise the difference.
    # Always attached to the cached copy; stripped below from the returned dict
    # when the caller did not ask, so what the caller sees is unchanged.
    result["resample_aucs"] = resample_aucs
    result["resample_ids"] = resample_ids
    if select is not None:
        # How many resamples *attempted* each variable — see the tally site
        # above for why a resample whose fit later fails is still counted.
        result["selection_counts"] = selection_counts
        # The denominator for `selection_counts`, from this same loop — never
        # `bootstrap_resamples`/`successful_bootstraps` above, and never a
        # count from a caller's own, unrelated resampling loop. See the
        # comment where `selection_resamples` is declared.
        result["selection_resamples"] = selection_resamples
    result["n_not_converged"] = n_not_converged
    # One summary per model instead of one raw statsmodels warning per fit. A
    # few non-converged resamples out of a thousand move a mean optimism by
    # nothing and are normal on a cohort this size; a large share means the
    # design is near-separable and the correction itself is unreliable, which
    # is worth interrupting for. 1% of attempted resamples is the line.
    if n_not_converged > max(1, len(auc_optimisms) // 100):
        warnings.warn(
            f"{n_not_converged} of {len(auc_optimisms)} bootstrap resamples "
            f"failed to converge for a model on {', '.join(design_cols)}. "
            "Their optimism estimates are still included; treat the corrected "
            "metrics for this model as unreliable if the share is large.",
            stacklevel=2,
        )
    _VALIDATION_CACHE[key] = copy.deepcopy(result)
    if not return_resample_aucs:
        result.pop("resample_aucs", None)
        result.pop("resample_ids", None)
    return result


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


# ---------------------------------------------------------------------------
# Missing-data policy sentence
# ---------------------------------------------------------------------------
# Clinicians read this text in the Streamlit calculator and it is phrased like a
# Methods sentence, so it has to describe the run that produced the artifact. It
# is generated from the MICE manifest rather than typed by hand: a hand-typed
# sentence is exactly what went stale and claimed complete-case analysis for a
# fully imputed cohort.

_MICE_METHOD_NAMES = {
    "pmm": "predictive mean matching",
    "logreg": "logistic regression",
    "polyreg": "polytomous regression",
    "polr": "proportional-odds regression",
}
#: Used only when no manifest is at hand (unit tests, ad-hoc calls).
_MICE_FALLBACK_M = 20
_MICE_FALLBACK_SEED = 42


def _join_names(names: Sequence[str]) -> str:
    """``a, b and c`` — readable prose rather than a bracketed list."""
    names = list(names)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def missing_data_policy_text(manifest: Mapping[str, Any] | None = None) -> str:
    """Plain-language account of how missing values were handled, from the manifest.

    ``manifest`` is ``missingness_resolution.read_mice_manifest(output_root)``.
    Without one it falls back to a short statement that claims no per-column
    detail — better silent than wrong.
    """
    manifest = dict(manifest or {})
    m = int(manifest.get("m") or _MICE_FALLBACK_M)
    seed = int(manifest.get("seed") or _MICE_FALLBACK_SEED)
    methods = {
        str(col): str(meth)
        for col, meth in (manifest.get("methods_by_column") or {}).items()
        if str(meth)
    }
    if not methods:
        return (
            f"Missing values were filled in by multiple imputation ({m} completed "
            f"datasets, seed {seed}) rather than by dropping patients, so every "
            "model was fitted on the full cohort."
        )

    by_method: dict[str, list[str]] = {}
    for col, meth in methods.items():
        by_method.setdefault(meth, []).append(col)
    clauses = [
        f"{len(cols)} by {_MICE_METHOD_NAMES.get(meth, meth)} "
        f"({_join_names(sorted(cols))})"
        for meth, cols in sorted(by_method.items())
    ]

    # A derived column counts as recreated-from-imputed when ANY ancestor was
    # imputed: edema_index_ge0.0617 sits two hops above edema_volume_cm3.
    deps = {
        str(name): [str(s) for s in sources]
        for name, sources in (manifest.get("derived_dependencies") or {}).items()
    }
    touched: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, sources in deps.items():
            if name in touched:
                continue
            if any(s in methods or s in touched for s in sources):
                touched.add(name)
                changed = True
    derived_clause = ""
    if touched:
        derived_clause = (
            f" Derived columns were never imputed directly: "
            f"{_join_names(sorted(touched))} were removed before imputation and "
            "recomputed from their imputed parent measurement in each completed "
            "dataset, which is why they are complete here."
        )

    return (
        "Missing values were filled in by multiple imputation, not by dropping "
        f"patients. The {len(methods)} variables that had missing values were "
        f"imputed in one R mice chain (m = {m} completed datasets, seed {seed}): "
        f"{_join_names(clauses)}. Every other variable was already complete."
        f"{derived_clause}"
        f" Coefficients were fitted in all {m} datasets and pooled with Rubin's "
        "rules, so n is the whole cohort in every dataset and no patient was "
        "excluded — it is not a complete-case count. Internal validation and "
        "shrinkage were run on the first completed dataset. Imputation assumes "
        "values were missing at random given the other recorded variables."
    )


def enrich_streamlit_artifact(
    artifact: dict[str, Any],
    model_df: pd.DataFrame,
    design_cols: list[str],
    *,
    n_bootstrap: int = analysis.BOOTSTRAP_RESAMPLES,
    missing_data_policy: str | None = None,
    select=None,
) -> dict[str, Any]:
    """Add validation charts, prose, shrinkage, and recalibrated coefficients.

    ``select``, when given, is forwarded to :func:`bootstrap_internal_validation`
    so a data-selected model (e.g. ``top_6_variables``) re-runs its own
    variable picking inside every resample here too — the same correction
    ``model_comparison.run_comparison_stage`` applies for its own bootstrap of
    the same model. Without this, the two paths would validate the same model
    two different ways and publish two different "optimism-corrected AUC"
    numbers for it.
    """
    target = str(artifact["target"])
    validation = bootstrap_internal_validation(
        model_df,
        target,
        design_cols,
        artifact["coefficients"],
        n_bootstrap=n_bootstrap,
        select=select,
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
        missing_data_policy or missing_data_policy_text()
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


# ---------------------------------------------------------------------------
# One master seed for every bootstrap in the modelling phase
# ---------------------------------------------------------------------------
# Matches the cut-point phase's ``wobble.SEED``. Every model validated in a
# run draws its resamples from the same index matrix, so two models' AUCs at
# resample i come from the same patients and their difference is a paired
# difference rather than noise from two unrelated resamplings.

BOOTSTRAP_SEED: int = 20260801


def _resample_indices(n_rows: int, n_bootstrap: int) -> np.ndarray:
    """``n_bootstrap`` x ``n_rows`` index matrix, fixed by ``BOOTSTRAP_SEED``."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    return rng.choice(n_rows, size=(n_bootstrap, n_rows), replace=True)


# ---------------------------------------------------------------------------
# Running several model validations at once
# ---------------------------------------------------------------------------
# Each model's bootstrap is self-contained — same resamples, same fits, same
# numbers whether it runs alone or beside the others. Only the wall clock
# changes, which makes this the one speedup that cannot move a published
# figure. The caps below are what keep it laptop-friendly rather than maximal.

MAX_VALIDATION_WORKERS = 4
#: Cores deliberately left for the OS, the browser, and the rest of the pipeline.
RESERVED_CORES = 2
#: Override the worker count, e.g. ATYPIER_VALIDATION_WORKERS=1 to force sequential.
WORKERS_ENV = "ATYPIER_VALIDATION_WORKERS"


def _on_battery() -> bool:
    """True on macOS running unplugged. Detection failure counts as plugged in."""
    try:
        from missingness_resolution import _macos_on_battery

        return _macos_on_battery()
    except Exception:
        return False


def validation_workers(n_tasks: int) -> int:
    """Processes to use for ``n_tasks`` model validations — conservative by design.

    One process on battery (spinning up every core is what empties it), never
    more than :data:`MAX_VALIDATION_WORKERS`, and always ``RESERVED_CORES``
    left free so the machine stays responsive.

    ``ATYPIER_VALIDATION_WORKERS`` overrides all of that, still bounded by the
    task count, for a plugged-in batch run or to force the sequential path.
    """
    if n_tasks <= 1:
        return 1
    override = os.environ.get(WORKERS_ENV, "").strip()
    if override:
        try:
            return max(1, min(int(override), n_tasks))
        except ValueError:
            raise ValueError(
                f"{WORKERS_ENV}={override!r} is not a whole number of workers"
            ) from None
    if _on_battery():
        return 1
    usable = (os.cpu_count() or 1) - RESERVED_CORES
    return max(1, min(MAX_VALIDATION_WORKERS, usable, n_tasks))


def _enrich_or_keep(
    artifact: dict[str, Any],
    model_df: pd.DataFrame,
    design_cols: list[str],
    *,
    n_bootstrap: int,
    missing_data_policy: str | None = None,
    select=None,
) -> dict[str, Any]:
    """Enrich one artifact, falling back to the plain artifact if validation fails.

    Module-level (not a closure) so it can be shipped to a worker process.
    ``select`` (when given) must itself be picklable for the same reason —
    see ``variable_selection.bootstrap_reselect``, the only hook this is
    used with today.
    """
    try:
        return enrich_streamlit_artifact(
            artifact, model_df, design_cols,
            n_bootstrap=n_bootstrap,
            missing_data_policy=missing_data_policy,
            select=select,
        )
    except (ValueError, RuntimeError):
        return artifact


def _enrich_and_return_cache(*args, **kwargs) -> tuple[dict[str, Any], dict]:
    """``_enrich_or_keep`` plus the memo entries it filled, for a worker process.

    ``_VALIDATION_CACHE`` is per-process, so a validation computed inside a
    joblib worker is invisible to the parent — and the comparison stage back in
    the parent would then recompute all eleven of them. The worker hands its
    entries back with the artifact and the parent merges them, so the memo
    works the same whether the batch ran on one process or four.
    """
    out = _enrich_or_keep(*args, **kwargs)
    return out, dict(_VALIDATION_CACHE)


def enrich_streamlit_artifacts(
    jobs: Sequence[
        tuple[dict[str, Any], pd.DataFrame, list[str]]
        | tuple[dict[str, Any], pd.DataFrame, list[str], Any]
    ],
    *,
    n_bootstrap: int = analysis.BOOTSTRAP_RESAMPLES,
    missing_data_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Validate several models, concurrently when that is safe. Order is preserved.

    Each job is a 3-tuple ``(artifact, model_df, design_cols)`` or, for a
    model whose validation must re-run variable selection inside every
    resample (see ``model_comparison.run_comparison_stage``), a 4-tuple with
    a picklable ``select`` callable as the last element. The 4th element
    defaults to ``None`` when omitted, so existing 3-tuple call sites are
    unaffected.
    """
    if not jobs:
        return []
    normalized = [
        (job[0], job[1], job[2], job[3] if len(job) > 3 else None)
        for job in jobs
    ]
    n_workers = validation_workers(len(normalized))
    if n_workers == 1:
        # No pool, no process startup — the same code path as before.
        return [
            _enrich_or_keep(
                a, m, d,
                n_bootstrap=n_bootstrap,
                missing_data_policy=missing_data_policy,
                select=s,
            )
            for a, m, d, s in normalized
        ]

    from joblib import Parallel, delayed, parallel_config

    print(
        f"🔀 Validating {len(normalized)} models on {n_workers} processes "
        f"({os.cpu_count()} cores, {RESERVED_CORES} held back)…",
        flush=True,
    )
    # inner_max_num_threads=1: each worker's BLAS must not also fan out, or
    # n_workers x BLAS threads oversubscribes the machine and runs slower hot.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        returned = list(
            Parallel(n_jobs=n_workers)(
                delayed(_enrich_and_return_cache)(
                    a, m, d,
                    n_bootstrap=n_bootstrap,
                    missing_data_policy=missing_data_policy,
                    select=s,
                )
                for a, m, d, s in normalized
            )
        )
    for _, worker_cache in returned:
        _VALIDATION_CACHE.update(worker_cache)
    return [artifact for artifact, _ in returned]
