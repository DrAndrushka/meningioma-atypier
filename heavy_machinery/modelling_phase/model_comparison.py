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
