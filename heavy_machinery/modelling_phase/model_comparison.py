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

import warnings
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


def _nested_chi2_per_imputation(
    frames, schema, target: str, full_cols: Sequence[str], single: str,
) -> tuple[list[float], int]:
    """Per-draw LR chi-square for the full model against one of its predictors."""
    import statsmodels.api as sm
    from model_validation import build_complete_case_frame

    stats: list[float] = []
    k = max(len(full_cols) - 1, 1)
    for frame in frames:
        try:
            df_full, cols_full = build_complete_case_frame(
                frame, schema, list(full_cols), target)
            df_red, cols_red = build_complete_case_frame(
                frame, schema, [single], target)
        except (ValueError, RuntimeError):
            continue
        y = df_full[target].astype(int).to_numpy()
        ll_full = sm.Logit(
            y, sm.add_constant(df_full[cols_full].astype(float), has_constant="add")
        ).fit(disp=False).llf
        ll_red = sm.Logit(
            y, sm.add_constant(df_red[cols_red].astype(float), has_constant="add")
        ).fit(disp=False).llf
        stats.append(2.0 * (ll_full - ll_red))
        k = max(len(cols_full) - len(cols_red), 1)
    return stats, k


def run_comparison_stage(
    cohort_df,
    schema,
    variants: Sequence[Any],
    target: str,
    tabs_dir,
    *,
    n_bootstrap: int | None = None,
    candidates: Sequence[str] | None = None,
    k_top: int = 6,
    frames: Sequence[Any] | None = None,
    assert_reference: bool = True,
    selected_model_ids: set[str] | None = None,
):
    """Fit the singles, difference every model against its own, write 3 tables."""
    from pathlib import Path

    import pandas as pd
    import variable_selection as vs
    from heavy_machinery.config import load
    from model_validation import (
        build_complete_case_frame,
        bootstrap_internal_validation,
    )

    analysis = load("analysis")
    if n_bootstrap is None:
        n_bootstrap = analysis.BOOTSTRAP_RESAMPLES
    tabs_dir = Path(tabs_dir)
    tabs_dir.mkdir(parents=True, exist_ok=True)
    frames = list(frames) if frames is not None else [cohort_df]

    def _pred(v):
        return list(v["predictors"] if isinstance(v, dict) else v.predictors)

    def _mid(v):
        return str(v["model_id"] if isinstance(v, dict) else v.model_id)

    singles = sorted({p for v in variants for p in _pred(v)})
    # ``candidates`` is the selection pool and is NOT the literature union: the
    # six that win include adc_value and cystic_component, which appear in no
    # literature model. The notebook passes its EDA predictor list.
    fitted = fit_single_predictors(
        cohort_df, schema, singles, target, n_bootstrap=n_bootstrap)

    single_tbl = pd.DataFrame([
        {"predictor": p, "n": d["n"], "events": d["events"],
         "auc_apparent": round(d["auc_apparent"], 3),
         "auc_corrected": round(d["auc_corrected"], 3)}
        for p, d in sorted(fitted.items())
    ])
    single_tbl.to_csv(tabs_dir / "single_predictor_reference.csv", index=False)

    y_all = cohort_df[target].astype(int).to_numpy()
    cand = list(candidates) if candidates is not None else singles
    picked, audit = vs.select_variables(
        cohort_df, y_all, cand, k=k_top,
        rho_max=0.8, cutpoint_parent=analysis.CUTPOINT_PARENT)
    if assert_reference:
        vs.assert_reference(picked)
    sel_tbl = pd.DataFrame(audit)
    sel_tbl.to_csv(tabs_dir / "top_variable_selection.csv", index=False)

    rows = []
    for v in variants:
        mid, preds = _mid(v), _pred(v)
        try:
            model_df, design_cols = build_complete_case_frame(
                cohort_df, schema, preds, target)
        except (ValueError, RuntimeError):
            continue
        import statsmodels.api as sm
        y = model_df[target].astype(int).to_numpy()
        X = sm.add_constant(model_df[design_cols].astype(float), has_constant="add")
        fit = sm.Logit(y, X).fit(disp=False)
        coefs = {"const": float(fit.params.iloc[0])}
        coefs.update({c: float(fit.params[c]) for c in design_cols})
        # The one data-selected model re-runs its own selection inside every
        # resample, so its optimism covers the picking and not just the fitting.
        selector = None
        if selected_model_ids and mid in selected_model_ids:
            def selector(frame, y_boot, _cand=cand, _k=k_top):
                sub, _ = vs.select_variables(
                    frame, y_boot, [c for c in _cand if c in frame.columns],
                    k=_k, rho_max=0.8,
                    cutpoint_parent=analysis.CUTPOINT_PARENT)
                return sub
        val = bootstrap_internal_validation(
            model_df, target, design_cols, coefs,
            n_bootstrap=n_bootstrap, return_resample_aucs=True, select=selector)
        combined_auc = next(
            m for m in val["metrics"] if m["metric"] == "AUC")["optimism_corrected"]
        # Predictions of the full-cohort fit, held fixed. The confidence
        # interval comes from resampling PATIENTS against these, not from the
        # optimism vector: a one-predictor model's optimism AUC is constant
        # (AUC is rank-based and cannot see a coefficient's size), so
        # differencing optimism vectors would leave the interval with no
        # variance from the single side at all.
        y_full = model_df[target].astype(int).to_numpy()
        pred_combined = np.asarray(fit.predict(X), dtype=float)
        boot_combined = bootstrap_auc_vector(
            y_full, pred_combined, n_bootstrap=n_bootstrap)
        auc_model_apparent = next(
            m for m in val["metrics"] if m["metric"] == "AUC")["apparent"]
        for single in preds:
            if single not in fitted:
                warnings.warn(
                    f"No single-predictor fit for {single!r}; the "
                    f"{mid} vs {single} comparison row is missing.",
                    stacklevel=2,
                )
                continue
            boot_single = bootstrap_auc_vector(
                fitted[single]["y"], fitted[single]["pred"], n_bootstrap=n_bootstrap)
            if len(boot_combined) != len(boot_single):
                raise RuntimeError(
                    f"Unpaired resample vectors for {mid} vs {single}: "
                    f"{len(boot_combined)} and {len(boot_single)}. Both models "
                    "must be scored over the same patient draws."
                )
            d = paired_delta_auc(boot_combined, boot_single)
            chi2s, k = _nested_chi2_per_imputation(
                frames, schema, target, preds, single)
            p = d2_pool(chi2s, k)["p"] if chi2s else float("nan")
            # Two deltas on purpose, each meaning what it says. The corrected
            # one is the point estimate and is exactly the difference of the
            # two corrected AUC columns, so a reader can check the arithmetic.
            # The apparent one is what the confidence interval is centred on,
            # because the interval measures sampling error, not optimism.
            rows.append({
                "model_id": mid, "single": single,
                "auc_model_corrected": round(float(combined_auc), 3),
                "auc_single_corrected": round(fitted[single]["auc_corrected"], 3),
                "delta_auc_corrected": round(
                    float(combined_auc) - fitted[single]["auc_corrected"], 3),
                "delta_auc_apparent": round(
                    float(auc_model_apparent) - fitted[single]["auc_apparent"], 3),
                "delta_ci_lo": round(d["ci_lo"], 3),
                "delta_ci_hi": round(d["ci_hi"], 3),
                "n_resamples": d["n_resamples"],
                "d2_p": p,
            })
    vs_tbl = pd.DataFrame(rows)
    vs_tbl.to_csv(tabs_dir / "model_vs_single_auc.csv", index=False)
    return {"single_predictor_reference": single_tbl,
            "model_vs_single_auc": vs_tbl,
            "top_variable_selection": sel_tbl,
            "top_variables": picked}
