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
