"""How well one cut-point performs — sensitivity, specificity, and what follows.

Every number here comes with an interval, because a cut-point quoted bare is a
cut-point that cannot be judged.

**Sensitivity** — of the patients who really are high grade, what share does the
test flag? **Specificity** — of the patients who really are low grade, what
share does it correctly leave alone. These two trade off: any cut-point that
catches more of one misses more of the other.

**PPV and NPV** answer the question a clinician actually asks — *this patient
tested positive, what is the chance they are high grade?* But they depend on how
common high grade is in the group being tested. This cohort is a surgical
series, roughly 30% high grade; in an unselected imaging population the rate is
far lower and the PPV would fall with it. So PPV and NPV are reported here as
descriptions of *this* cohort and must not be quoted as transferable.

**Likelihood ratios** are the ones that do transfer. LR+ is how many times more
likely a positive result is in a high-grade patient than a low-grade one, and it
does not depend on prevalence. Above 10 is usually called strong evidence,
around 5 moderate, below 2 barely worth acting on.

Intervals: Wilson for the proportions (a symmetric interval around a specificity
of 0.95 would run past 1.0), and the log scale for likelihood ratios, which are
ratios and so are skewed on their natural scale.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from intervals import wilson_ci
from measurements import LOWER


def flag(x: np.ndarray, cutoff: float, direction: str) -> np.ndarray:
    """Which patients this cut-point calls suspicious.

    The comparison is inclusive on the suspicious side — ``<=`` for a
    measurement where low values are suspicious, ``>=`` otherwise — so a patient
    sitting exactly on the cut-point is flagged. That matches how the rule reads
    in a sentence ("ADC of 0.72 or below") and keeps the sweep in
    :mod:`criteria` consistent with the rule it prints.
    """
    x = np.asarray(x, dtype=float)
    return (x <= cutoff) if direction == LOWER else (x >= cutoff)


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    """The four counts everything else is computed from."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    return {"tp": int(np.sum(flagged & y)),
            "fp": int(np.sum(flagged & ~y)),
            "fn": int(np.sum(~flagged & y)),
            "tn": int(np.sum(~flagged & ~y))}


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int, *,
                              alpha: float = 0.05) -> tuple[float, float, float]:
    """LR+ with a log-scale interval.

    ``LR+ = Se / (1 - Sp)``. When specificity is a perfect 1.0 the ratio is
    infinite and no interval exists — reported as ``inf`` with blank bounds
    rather than quietly nudged, because an infinite likelihood ratio is a signal
    that the cut-point sits beyond the observed data, not a strong result.
    """
    n_pos, n_neg = tp + fn, fp + tn
    if not n_pos or not n_neg:
        return float("nan"), float("nan"), float("nan")
    se, sp = tp / n_pos, tn / n_neg
    if sp >= 1.0:
        return float("inf"), float("nan"), float("nan")
    est = _ratio(se, 1 - sp)
    if se <= 0:
        return est, float("nan"), float("nan")
    z = float(norm.ppf(1 - alpha / 2))
    se_log = np.sqrt((1 - se) / (se * n_pos) + sp / ((1 - sp) * n_neg))
    return est, float(est * np.exp(-z * se_log)), float(est * np.exp(z * se_log))


def likelihood_ratio_negative(tp: int, fp: int, fn: int, tn: int, *,
                              alpha: float = 0.05) -> tuple[float, float, float]:
    """LR- with a log-scale interval. ``LR- = (1 - Se) / Sp``."""
    n_pos, n_neg = tp + fn, fp + tn
    if not n_pos or not n_neg:
        return float("nan"), float("nan"), float("nan")
    se, sp = tp / n_pos, tn / n_neg
    if sp <= 0:
        return float("inf"), float("nan"), float("nan")
    est = _ratio(1 - se, sp)
    if se >= 1.0:
        return est, float("nan"), float("nan")
    z = float(norm.ppf(1 - alpha / 2))
    se_log = np.sqrt(se / ((1 - se) * n_pos) + (1 - sp) / (sp * n_neg))
    return est, float(est * np.exp(-z * se_log)), float(est * np.exp(z * se_log))


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, *,
                        alpha: float = 0.05) -> dict[str, float]:
    """Every accuracy measure this phase reports, each with its interval."""
    n_pos, n_neg = tp + fn, fp + tn
    n_flagged, n_clear = tp + fp, fn + tn

    se_lo, se_hi = wilson_ci(tp, n_pos, alpha=alpha)
    sp_lo, sp_hi = wilson_ci(tn, n_neg, alpha=alpha)
    ppv_lo, ppv_hi = wilson_ci(tp, n_flagged, alpha=alpha)
    npv_lo, npv_hi = wilson_ci(tn, n_clear, alpha=alpha)
    lr_pos, lr_pos_lo, lr_pos_hi = likelihood_ratio_positive(
        tp, fp, fn, tn, alpha=alpha)
    lr_neg, lr_neg_lo, lr_neg_hi = likelihood_ratio_negative(
        tp, fp, fn, tn, alpha=alpha)

    sensitivity, specificity = _ratio(tp, n_pos), _ratio(tn, n_neg)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": tp + fp + fn + tn, "n_high": n_pos, "n_low": n_neg,
        "n_flagged": n_flagged,
        "sensitivity": sensitivity, "sensitivity_lo": se_lo, "sensitivity_hi": se_hi,
        "specificity": specificity, "specificity_lo": sp_lo, "specificity_hi": sp_hi,
        "ppv": _ratio(tp, n_flagged), "ppv_lo": ppv_lo, "ppv_hi": ppv_hi,
        "npv": _ratio(tn, n_clear), "npv_lo": npv_lo, "npv_hi": npv_hi,
        "lr_pos": lr_pos, "lr_pos_lo": lr_pos_lo, "lr_pos_hi": lr_pos_hi,
        "lr_neg": lr_neg, "lr_neg_lo": lr_neg_lo, "lr_neg_hi": lr_neg_hi,
        "youden_j": (sensitivity + specificity - 1.0
                     if np.isfinite(sensitivity) and np.isfinite(specificity)
                     else float("nan")),
    }


def accuracy_at(y: np.ndarray, x: np.ndarray, cutoff: float, direction: str, *,
                alpha: float = 0.05) -> dict[str, float]:
    """Performance of one cut-point, on the patients who have the measurement.

    Rows missing either the measurement or the outcome are dropped, and the
    surviving count is reported as ``n`` — so a denominator that shrank is
    visible in the table rather than inferred.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    counts = confusion(y[ok].astype(int), flag(x[ok], cutoff, direction))
    out = metrics_from_counts(**counts, alpha=alpha)
    out["cutoff"] = float(cutoff)
    return out
