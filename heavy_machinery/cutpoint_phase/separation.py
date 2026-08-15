"""Step 3 — does each measurement separate grade 1 from grade 2-3 at all?

The answer is the **AUC**: line the patients up by one measurement, pick a
high-grade and a low-grade patient at random, and ask how often the high-grade
one is on the suspicious side. 0.50 is a coin flip — the measurement carries no
information. 1.00 is perfect separation.

The interval comes from **DeLong**, computed exactly rather than by resampling.
Resampling an AUC gives a slightly different answer every run and costs seconds
per measurement; DeLong is a closed formula, so the same data always yields the
same interval. It also produces the pieces needed to compare two AUCs measured
on the same patients, which is what step 10 does with them.

Two details that change the numbers:

*Direction is applied, not discovered.* ADC is negated before scoring, because
low ADC is the suspicious side. Letting the code choose the orientation would
floor every AUC at 0.50 and quietly convert "this measurement does nothing"
into "this measurement does a little".

*The interval is built on the logit scale.* An AUC near 0.5 with a wide interval
can otherwise be reported as reaching above 1.0, which is not a possible value.
Working on the log-odds scale and transforming back keeps both ends inside
[0, 1] without distorting the middle.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from measurements import (MEASUREMENTS, Measurement, STRATUM_ALL,
                          STRATUM_PRESENT, stratum_mask)
from intervals import wilson_ci

OUTCOME = "high_grade"


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


def separation_table(df: pd.DataFrame,
                     measurements: Sequence[Measurement] = MEASUREMENTS,
                     *, alpha: float = 0.05) -> pd.DataFrame:
    """One row per measurement per stratum, ranked best-separating first.

    The zero-inflated measurements contribute two rows each, so the ranking
    shows directly whether the edema signal comes from having any edema or from
    how much — the two rows sit next to each other with different denominators.
    """
    rows = []
    for m in measurements:
        for stratum in m.strata:
            mask = stratum_mask(df, m, stratum)
            sub = df.loc[mask]
            y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
            x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
            row = {
                "measurement": m.stratum_label(stratum),
                "col": m.col,
                "stratum": stratum,
                "direction": m.direction,
            }
            try:
                row.update(auc_with_ci(y, x, m.direction, alpha=alpha))
            except AucError as exc:
                row.update({"auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                            "n": int(mask.sum()), "n_high": np.nan,
                            "n_low": np.nan, "auc_var": np.nan,
                            "note": str(exc)})
            rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("auc", ascending=False,
                             na_position="last").reset_index(drop=True)


def presence_effect(df: pd.DataFrame, m: Measurement, *,
                    alpha: float = 0.05) -> dict[str, float]:
    """Does *having* the finding at all predict high grade?

    The first half of the zero-inflation question, and the one an AUC cannot
    answer: a yes/no predictor has no ranking to measure. So this is two
    high-grade rates with Wilson intervals, and the gap between them.
    """
    if not m.zero_inflated:
        raise AucError(f"{m.col} is not declared zero-inflated.")
    observed = stratum_mask(df, m, STRATUM_ALL)
    present = stratum_mask(df, m, STRATUM_PRESENT)
    absent = observed & ~present

    out: dict[str, float] = {"measurement": m.label}
    for name, mask in (("absent", absent), ("present", present)):
        y = pd.to_numeric(df.loc[mask, OUTCOME], errors="coerce")
        n, k = int(len(y)), int(y.sum())
        lo, hi = wilson_ci(k, n, alpha=alpha)
        out[f"n_{name}"] = n
        out[f"high_grade_{name}"] = k
        out[f"rate_{name}"] = k / n if n else np.nan
        out[f"rate_{name}_lo"] = lo
        out[f"rate_{name}_hi"] = hi
    out["rate_difference"] = out["rate_present"] - out["rate_absent"]
    return out


def presence_table(df: pd.DataFrame,
                   measurements: Sequence[Measurement] = MEASUREMENTS,
                   *, alpha: float = 0.05) -> pd.DataFrame:
    """`presence_effect` for every zero-inflated measurement."""
    rows = [presence_effect(df, m, alpha=alpha)
            for m in measurements if m.zero_inflated]
    return pd.DataFrame(rows)


def describe_separation(table: pd.DataFrame) -> str:
    """One line: the best measurement, and whether anything clears a coin flip.

    "Clears a coin flip" means the lower end of the interval sits above 0.50. A
    measurement whose interval includes 0.50 has not been shown to carry any
    information at all, and no cut-point drawn on it can be defended.
    """
    scored = table.dropna(subset=["auc"])
    if scored.empty:
        return "No measurement could be scored."
    best = scored.iloc[0]
    informative = scored[scored["auc_lo"] > 0.5]
    line = (f"Best separation: {best['measurement']}, AUC "
            f"{best['auc']:.2f} (95% CI {best['auc_lo']:.2f}-{best['auc_hi']:.2f}), "
            f"n={int(best['n'])}.")
    if informative.empty:
        return line + (" No measurement's interval clears 0.50 — none is shown "
                       "to separate the grades at all.")
    names = ", ".join(informative["measurement"])
    return line + (f" Clearing 0.50 at the lower bound: {names}"
                   f" ({len(informative)} of {len(scored)}).")
