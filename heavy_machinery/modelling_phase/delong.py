"""The one place that turns a score into an AUC and a 95% interval.

Two phases publish the same univariate AUC for the same five measurements — the
modelling phase in the EDA table on ``report.html``, the cut-point phase in the
separation table on ``cutpoint_report.html``. While each computed its own
interval the reader got max diameter as 0.67 (0.62–0.74) on one page and
0.67 (0.61–0.73) on the other, with nothing on either page naming the method.
Both were arithmetically defensible, which is why nothing caught it. The
estimator is now declared once, here, and both phases read it — the same
arrangement :mod:`scales` uses for the log scale.

**DeLong**, computed exactly rather than by resampling. Resampling an AUC gives
a slightly different answer every run: the 400-draw percentile bootstrap this
replaced moved max diameter's lower bound between 0.61 and 0.62 on the random
seed alone, and only converged on the DeLong interval past ~20 000 draws.
DeLong is a closed formula, so the same data always yields the same interval,
and it produces the pieces needed to compare two AUCs measured on the same
patients — which is what the cut-point phase's step 10 does with them.

*The interval is built on the logit scale.* An AUC near 0.5 with a wide interval
can otherwise be reported as reaching above 1.0, which is not a possible value.
Working on the log-odds scale and transforming back keeps both ends inside
[0, 1] without distorting the middle.

Orientation is the one thing this module does not decide. The cut-point phase
declares a direction per measurement and passes it in; the EDA screen runs over
whatever the associations table holds and has no such declaration to read, so it
calls :func:`auc_ci_auto` and takes the orientation from the data.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm


class AucError(Exception):
    """The AUC cannot be computed from what was passed in."""


class DelongResult(NamedTuple):
    """An AUC, its variance, and the per-patient pieces that variance is made of.

    ``v_pos`` and ``v_neg`` are kept because comparing two AUCs on the same
    patients needs their covariance, and that is built from these — step 10.
    """

    auc: float
    var: float
    v_pos: np.ndarray
    v_neg: np.ndarray
    n_pos: int
    n_neg: int


def oriented_score(x: np.ndarray, direction: str) -> np.ndarray:
    """Flip the measurement so that larger always means more suspicious."""
    x = np.asarray(x, dtype=float)
    return -x if direction == "lower" else x


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties sharing their average — the tie handling DeLong needs.

    Many patients here share a value exactly (edema volume 0.0 above all), and
    breaking those ties arbitrarily would understate the variance.
    """
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(y: np.ndarray, score: np.ndarray) -> DelongResult:
    """AUC and its variance by the Sun & Xu (2014) closed form.

    ``y`` is 1 for high grade, 0 for low grade; ``score`` is already oriented so
    that larger means more suspicious.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise AucError("Outcome and score have different lengths.")
    if np.isnan(score).any():
        raise AucError("Score contains missing values — subset before scoring.")

    pos, neg = score[y == 1], score[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise AucError(
            f"Need both grades to compute an AUC — got {m} high, {n} low.")

    t_pos = _midrank(pos)
    t_neg = _midrank(neg)
    t_all = _midrank(np.concatenate([pos, neg]))
    auc = (t_all[:m].sum() / m - (m + 1) / 2) / n

    v_pos = (t_all[:m] - t_pos) / n
    v_neg = 1.0 - (t_all[m:] - t_neg) / m
    var = (v_pos.var(ddof=1) / m if m > 1 else 0.0) + \
          (v_neg.var(ddof=1) / n if n > 1 else 0.0)
    return DelongResult(float(auc), float(var), v_pos, v_neg, m, n)


def logit_ci(auc: float, var: float, *, alpha: float = 0.05
             ) -> tuple[float, float]:
    """DeLong interval carried through the log-odds scale and back.

    ``Var(logit AUC) = Var(AUC) / [AUC·(1−AUC)]²`` — the variance is transformed
    too, not just the estimate. Transforming only the point estimate is a common
    slip and produces an interval that is not the one it claims to be.
    """
    if not np.isfinite(auc) or not np.isfinite(var) or var < 0:
        return float("nan"), float("nan")
    if auc <= 0 or auc >= 1 or var == 0:
        return float(auc), float(auc)
    z = float(norm.ppf(1 - alpha / 2))
    eta = np.log(auc / (1 - auc))
    se_eta = np.sqrt(var) / (auc * (1 - auc))
    lo, hi = eta - z * se_eta, eta + z * se_eta
    return float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))


def auc_with_ci(y: np.ndarray, x: np.ndarray, direction: str, *,
                alpha: float = 0.05) -> dict[str, float]:
    """AUC, its 95% interval, and the counts it rests on."""
    result = fast_delong(y, oriented_score(x, direction))
    lo, hi = logit_ci(result.auc, result.var, alpha=alpha)
    return {
        "auc": result.auc,
        "auc_lo": lo,
        "auc_hi": hi,
        "n": result.n_pos + result.n_neg,
        "n_high": result.n_pos,
        "n_low": result.n_neg,
        "auc_var": result.var,
    }


def auc_ci_auto(y: np.ndarray, scores: np.ndarray, *, alpha: float = 0.05
                ) -> tuple[float, float, float]:
    """AUC and DeLong interval when no direction has been declared.

    The screening table on ``report.html`` runs over every predictor in the
    associations screen, not only the five measurements the cut-point phase
    declares a direction for, so the orientation has to come from the data:
    an AUC below 0.5 is reported as 1 − AUC. That floors an uninformative
    predictor at 0.50, which is acceptable in a screen where the sign is read
    off the odds ratio in the next column and never acceptable as a published
    claim of separation — which is why :func:`auc_with_ci`, the one the
    cut-point phase publishes through, takes the direction as an argument.

    Flipping the estimate leaves the variance alone: negating the score negates
    ``v_pos``/``v_neg`` about their means, so ``Var(AUC)`` is unchanged and the
    logit interval simply mirrors about 0.5.

    Missing values are dropped pairwise. Returns three NaNs when fewer than five
    patients or only one grade survive, where an AUC means nothing.
    """
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    result = fast_delong(y, scores)
    auc = result.auc if result.auc >= 0.5 else 1.0 - result.auc
    lo, hi = logit_ci(auc, result.var, alpha=alpha)
    return float(auc), lo, hi


def delong_compare(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, *,
                   alpha: float = 0.05) -> dict[str, float]:
    """Compare two AUCs measured on the *same* patients.

    Two scores on one cohort are not independent — a patient who is easy to
    classify is easy for both. Comparing their intervals by eye, or with a test
    that assumes independence, overstates how different they are. DeLong uses
    the per-patient pieces from each score to work out how much they agree, and
    subtracts that shared agreement from the variance of the difference.

    Both scores must already be oriented so larger means more suspicious.
    """
    y = np.asarray(y).astype(int)
    a = fast_delong(y, np.asarray(score_a, dtype=float))
    b = fast_delong(y, np.asarray(score_b, dtype=float))
    m, n = a.n_pos, a.n_neg

    def _cov(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.cov(u, v, ddof=1)[0, 1]) if u.size > 1 else 0.0

    var = ((a.v_pos.var(ddof=1) - 2 * _cov(a.v_pos, b.v_pos)
            + b.v_pos.var(ddof=1)) / m if m > 1 else 0.0) + \
          ((a.v_neg.var(ddof=1) - 2 * _cov(a.v_neg, b.v_neg)
            + b.v_neg.var(ddof=1)) / n if n > 1 else 0.0)
    diff = a.auc - b.auc
    # Not ``var <= 0``: two identical scores leave a variance of order 1e-19
    # rather than exactly zero, which would otherwise be reported as an interval
    # a billionth of a point wide instead of no interval at all. AUC variances
    # in this phase are of order 1e-4, so this floor cannot swallow a real one.
    if not np.isfinite(var) or var <= 1e-16:
        return {"auc_a": a.auc, "auc_b": b.auc, "difference": float(diff),
                "difference_lo": float(diff), "difference_hi": float(diff),
                "z": np.nan, "p": np.nan, "correlated": True}
    z = float(norm.ppf(1 - alpha / 2))
    se = float(np.sqrt(var))
    stat = diff / se
    return {"auc_a": a.auc, "auc_b": b.auc, "difference": float(diff),
            "difference_lo": float(diff - z * se),
            "difference_hi": float(diff + z * se),
            "z": stat, "p": float(2 * norm.sf(abs(stat))), "correlated": True}
