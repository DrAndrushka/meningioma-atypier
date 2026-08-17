"""Combined-versus-single model comparison for the multivariable phase.

Zhang 2020 and Peng 2021 are the only published meningioma-grading papers that
report a combined model against each of its own single predictors. This module
reproduces that comparison on our cohort: every literature model against each
predictor it is built from, on the optimism-corrected scale.

Differences are paired. Two models validated in the same run see the same
bootstrap index sets (``model_validation.BOOTSTRAP_SEED``), so their AUCs can be
differenced within each resample. Comparing two independent CIs by eye instead
overstates how different two models are, because a patient who is easy to
classify is easy for both.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_delta_auc(
    aucs_combined: Sequence[float],
    aucs_single: Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Within-resample AUC difference, with a percentile CI.

    Both vectors must come from the same run, so element *i* of each is the
    same resample. The CI is the percentiles of the difference distribution,
    not the difference of two separate CIs.
    """
    a = np.asarray(aucs_combined, dtype=float)
    b = np.asarray(aucs_single, dtype=float)
    if a.size != b.size:
        raise ValueError(
            "paired_delta_auc needs the same number of resamples for both "
            f"models; got {a.size} and {b.size}."
        )
    if a.size == 0:
        raise ValueError("paired_delta_auc needs at least one resample.")
    diff = a - b
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(np.mean(diff)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_resamples": int(a.size),
    }


def d2_pool(chi2_stats: Sequence[float], k: int) -> dict[str, Any]:
    """Pool ``m`` chi-square statistics across imputations — Rubin's D2.

    Li KH, Meng XL, Raghunathan TE, Rubin DB. Significance levels from repeated
    p-values with multiply-imputed data. Statistica Sinica 1991;1:65-92.

    A likelihood-ratio test run separately on each of the 20 MICE draws gives 20
    chi-square statistics that cannot simply be averaged: doing so ignores the
    between-draw variance and reports a test that is too confident. D2 combines
    them with a variance correction. It is the cheap, standard alternative to
    Meng-Rubin D3, which needs the per-draw likelihoods rather than the
    statistics alone.

    ``k`` is the number of parameters being tested — for a combined model
    against one of its own predictors, the count of the extra terms.
    """
    from scipy.stats import f as f_dist

    d = np.asarray(chi2_stats, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("d2_pool needs at least one chi-square statistic.")
    if k < 1:
        raise ValueError("d2_pool needs k >= 1 parameters under test.")
    m = d.size
    d_bar = float(np.mean(d))
    # Relative increase in variance due to nonresponse, on the sqrt scale the
    # D2 derivation works on.
    r = (1.0 + 1.0 / m) * float(np.var(np.sqrt(d), ddof=1)) if m > 1 else 0.0
    stat = (d_bar / k - (m + 1.0) / (m - 1.0) * r) / (1.0 + r) if m > 1 else d_bar / k
    stat = max(stat, 0.0)
    df_den = k ** (-3.0 / m) * (m - 1) * (1.0 + 1.0 / r) ** 2 if (m > 1 and r > 0) else 1e6
    p = float(f_dist.sf(stat, k, df_den))
    return {
        "statistic": float(stat),
        "df_num": int(k),
        "df_den": float(df_den),
        "p": p,
        "m": int(m),
        "method": "D2 (Li, Meng, Raghunathan & Rubin 1991)",
    }


def fit_single_predictors(
    cohort_df,
    schema,
    predictors: Sequence[str],
    target: str,
    *,
    n_bootstrap: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit and bootstrap-validate each predictor on its own.

    These exist only to supply the yardstick a combined model is measured
    against, so they emit AUC and nothing else — no forest plot, no VIF table,
    no calculator artifact, no report fold of their own.
    """
    import statsmodels.api as sm
    from heavy_machinery.config import load
    from model_validation import (
        build_complete_case_frame,
        bootstrap_internal_validation,
    )

    if n_bootstrap is None:
        n_bootstrap = load("analysis").BOOTSTRAP_RESAMPLES

    out: dict[str, dict[str, Any]] = {}
    for pred in predictors:
        if pred not in cohort_df.columns:
            continue
        try:
            model_df, design_cols = build_complete_case_frame(
                cohort_df, schema, [pred], target)
        except (ValueError, RuntimeError):
            continue
        y = model_df[target].astype(int).to_numpy()
        X = sm.add_constant(model_df[design_cols].astype(float), has_constant="add")
        fit = sm.Logit(y, X).fit(disp=False)
        coefs = {"const": float(fit.params["const"])}
        coefs.update({c: float(fit.params[c]) for c in design_cols})
        model_pred = np.asarray(fit.predict(X), dtype=float)
        val = bootstrap_internal_validation(
            model_df, target, design_cols, coefs,
            n_bootstrap=n_bootstrap, return_resample_aucs=True)
        auc_row = next(m for m in val["metrics"] if m["metric"] == "AUC")
        out[pred] = {
            "auc_apparent": float(auc_row["apparent"]),
            "auc_corrected": float(auc_row["optimism_corrected"]),
            "n": int(len(model_df)),
            "events": int(y.sum()),
            "resample_aucs": val["resample_aucs"],
            "pred": [float(v) for v in model_pred],
            "y": [int(v) for v in y],
        }
    return out


def bootstrap_auc_vector(
    y: Sequence[int],
    pred: Sequence[float],
    *,
    n_bootstrap: int | None = None,
) -> list[float]:
    """AUC of a FIXED model, recomputed on each patient resample.

    This is not the optimism bootstrap. Nothing is refitted: the model stays
    as it was fitted on the full cohort, and only the patients are resampled.
    The spread of the resulting AUCs is sampling error of the AUC, which is
    what a confidence interval for a difference between two models needs.

    The optimism bootstrap answers a different question — how much does this
    model flatter itself — and for a one-predictor model its per-resample AUC
    is constant, because AUC is rank-based and cannot see the size of a
    coefficient. Feeding that vector into a difference would produce an
    interval with no variance from one side.

    Uses ``model_validation._resample_indices``, so two models scored in the
    same run see identical patient draws and their difference is paired.

    A resample containing only one outcome class has no defined AUC; it is
    dropped rather than recorded as 0.5.
    """
    from sklearn.metrics import roc_auc_score

    from heavy_machinery.config import load
    from model_validation import _resample_indices

    if n_bootstrap is None:
        n_bootstrap = load("analysis").BOOTSTRAP_RESAMPLES
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(pred, dtype=float)
    if y_arr.size != p_arr.size:
        raise ValueError(
            "bootstrap_auc_vector needs one prediction per patient; got "
            f"{y_arr.size} outcomes and {p_arr.size} predictions."
        )
    idx_matrix = _resample_indices(y_arr.size, n_bootstrap)
    out: list[float] = []
    for i in range(n_bootstrap):
        idx = idx_matrix[i]
        yy = y_arr[idx]
        if yy.min() == yy.max():
            continue
        # Not rounded, unlike the optimism vector's resample_aucs: this vector
        # is checked against roc_auc_score at rel=1e-9 for pairing (see the
        # tests), a tolerance six-decimal rounding would fail on any AUC that
        # isn't a round number.
        out.append(float(roc_auc_score(yy, p_arr[idx])))
    return out
