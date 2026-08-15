"""Step 14 — would acting on the cut-point beat the two obvious alternatives?

Every step before this one asks a statistical question: does risk bend, where
does it break, how far does the number move. None of them asks the question that
decides whether anyone uses the rule, which is *would following it lead to better
decisions than the two things a clinician can already do for free* — treat
everyone as high grade, or treat no one as high grade.

**Net benefit** is how that comparison is made. Count the high-grade tumors the
rule catches, subtract the benign ones it flags by mistake, and weight the
mistake by how bad it is:

    net benefit = TP/n − (FP/n) × t/(1 − t)

``t`` is the **threshold probability** — the risk at which a clinician would
switch from watching to acting. It is not a statistic; it is a value judgment,
and it is the point of the whole method that it is stated out loud. A clinician
who would act at a 20% chance of high grade is saying that missing one high-grade
tumor is four times worse than over-calling one benign tumor, because
``t/(1−t) = 0.2/0.8 = 1/4``. Plot net benefit across every ``t`` a reasonable
clinician might hold and you no longer have to guess which one your reader has.

Three lines are drawn for each measurement, and a rule is worth using only where
its line sits **above both** of the others:

*Treat none* is flat at zero. Nothing found, nothing wrongly flagged.
*Treat all* starts at the prevalence and falls, steeply, as ``t`` rises.
*The rule* is a straight line too, because a yes/no test has one fixed pair of
TP and FP counts and only the exchange rate changes with ``t``.

The rules are **corrected for optimism**, the same way and in the same resamples
as the Youden J in step 7: the cut-point is re-derived inside each bootstrap
sample and scored on the patients that sample left out. Without it a rule chosen
on these patients is credited with luck it will not have elsewhere, and net
benefit inherits every bit of that flattery.

The continuous measurement is corrected the same way, so the comparison between
"use the number" and "use the yes/no rule" is like for like. Its logistic fit is
refitted inside each resample too, which is why this module carries its own
two-parameter fitter — statsmodels is exact but too slow to call five thousand
times, and ``_irls`` reproduces it to six decimals (asserted in the tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from accuracy import confusion, flag
from ajnr_format import join_names
from criteria import YOUDEN, select, sweep
from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import auc_with_ci
from wobble import (MIN_HELD_OUT, MIN_PER_ARM, N_BOOTSTRAP, SEED,
                    FrozenCutpointError)

OUTCOME = "high_grade"

# Above roughly twice the prevalence nothing in this cohort stays above treat-all,
# and the curves are drawn where a clinician might plausibly sit rather than out
# to a threshold no one holds. 1% is the lowest the grid can carry: t/(1−t) at
# t = 0 is zero, which makes every rule look free.
MIN_THRESHOLD = 0.01
DEFAULT_MAX_THRESHOLD = 0.60
GRID_STEP = 0.01

# Net benefit differences smaller than this are noise on 352 patients: it is one
# extra true positive in a thousand, and the third decimal of a net benefit is
# not something this cohort can resolve.
NEGLIGIBLE = 0.001


def threshold_grid(max_threshold: float = DEFAULT_MAX_THRESHOLD) -> np.ndarray:
    """The threshold probabilities the curves are evaluated at."""
    return np.round(np.arange(MIN_THRESHOLD, float(max_threshold) + 1e-9,
                              GRID_STEP), 4)


def net_benefit(tp: float, fp: float, n: int,
                thresholds: np.ndarray) -> np.ndarray:
    """``TP/n − (FP/n)·t/(1−t)`` — the share of true positives, minus the price.

    Reported per patient in the cohort, not per positive, so the three lines on
    one plot share a denominator and can be read against each other.
    """
    t = np.asarray(thresholds, dtype=float)
    if n <= 0:
        return np.full(t.shape, np.nan)
    return tp / n - (fp / n) * (t / (1.0 - t))


def treat_all_curve(y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Flag every patient: catches everything, pays for every benign tumor."""
    y = np.asarray(y, dtype=int)
    return net_benefit(float(y.sum()), float((1 - y).sum()), int(y.size),
                       thresholds)


def rule_curve(y: np.ndarray, x: np.ndarray, cutoff: float, direction: str,
               thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of one fixed yes/no rule across every threshold."""
    if not np.isfinite(cutoff) or y.size == 0:
        return np.full(np.shape(thresholds), np.nan)
    k = confusion(np.asarray(y, dtype=int), flag(x, cutoff, direction))
    return net_benefit(float(k["tp"]), float(k["fp"]), int(y.size), thresholds)


# --------------------------------------------------------------------------
# The continuous comparator
# --------------------------------------------------------------------------
def _irls(y: np.ndarray, x: np.ndarray, *, max_iter: int = 40,
          tol: float = 1e-10) -> tuple[float, float] | None:
    """Univariable logistic fit by iteratively reweighted least squares.

    Two parameters and a 2×2 solve per iteration, which is the whole reason it
    exists: the bootstrap refits this five thousand times, and a general-purpose
    fitter spends most of its time on machinery a one-predictor model does not
    need. Returns ``(intercept, slope)``, or None if it fails to converge or the
    data separate perfectly.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 10 or y.sum() == 0 or y.sum() == y.size:
        return None
    b = np.zeros(2)
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(max_iter):
        eta = np.clip(design @ b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        if not np.all(np.isfinite(w)) or w.sum() < 1e-12:
            return None
        hessian = design.T @ (design * w[:, None])
        gradient = design.T @ (y - p)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if np.max(np.abs(step)) < tol:
            return (float(b[0]), float(b[1]))
    return None


def _model_scale(x: np.ndarray, log_x: bool) -> np.ndarray:
    """The scale the rest of the pipeline models this measurement on."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0.0, None)) if log_x else x


def model_curve(y: np.ndarray, x: np.ndarray, log_x: bool,
                thresholds: np.ndarray, *,
                coefficients: tuple[float, float] | None = None,
                score_on: tuple[np.ndarray, np.ndarray] | None = None
                ) -> np.ndarray:
    """Net benefit of using the measurement as a number rather than a cut-point.

    The number is turned into a risk by univariable logistic regression, and a
    patient is flagged when that risk reaches ``t``. This is the honest
    comparator for a cut-point: both arms then answer the same clinical question
    on the same scale, and the difference between the lines is exactly what
    dichotomising costs a decision rather than what it costs an AUC.

    ``coefficients`` and ``score_on`` exist for the bootstrap: fit on the
    patients drawn, score on the patients left out.
    """
    t = np.asarray(thresholds, dtype=float)
    fitted = coefficients if coefficients is not None else _irls(
        np.asarray(y, dtype=float), _model_scale(x, log_x))
    if fitted is None:
        return np.full(t.shape, np.nan)
    b0, b1 = fitted
    y_s, x_s = (score_on if score_on is not None else (y, x))
    y_s = np.asarray(y_s, dtype=int)
    if y_s.size == 0:
        return np.full(t.shape, np.nan)
    p = 1.0 / (1.0 + np.exp(-np.clip(b0 + b1 * _model_scale(x_s, log_x),
                                     -30.0, 30.0)))
    out = np.empty(t.shape, dtype=float)
    for i, thr in enumerate(t):
        flagged = p >= thr
        out[i] = net_benefit(float((flagged & (y_s == 1)).sum()),
                             float((flagged & (y_s == 0)).sum()),
                             int(y_s.size), np.array([thr]))[0]
    return out


# --------------------------------------------------------------------------
# Optimism correction
# --------------------------------------------------------------------------
def corrected_curves(y: np.ndarray, x: np.ndarray, direction: str, log_x: bool,
                     thresholds: np.ndarray, *, cutoff: float | None = None,
                     criterion: str = YOUDEN, n_boot: int = N_BOOTSTRAP,
                     seed: int = SEED) -> dict[str, object]:
    """Apparent and optimism-corrected net benefit for the rule and the number.

    One loop, one seed, the same resamples as step 7. In each replicate the
    cut-point is re-derived and the logistic model refitted on the patients
    drawn, then both are scored on the patients that replicate left out. The
    average gap between in-bag and out-of-bag net benefit is the optimism, and
    it is subtracted from the apparent curve threshold by threshold — the
    flattery is not the same size at every ``t``, so a single scalar correction
    would misstate the ends of the curve.

    ``cutoff`` is the published, rounded cut-point. The apparent curve is scored
    at exactly that value so this table describes the same rule as every other
    table in the phase; the resamples still re-derive their own, because the
    optimism being measured is the optimism of *having chosen* it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    n = y.size
    t = np.asarray(thresholds, dtype=float)
    blank = {"rule": np.full(t.shape, np.nan),
             "rule_corrected": np.full(t.shape, np.nan),
             "model": np.full(t.shape, np.nan),
             "model_corrected": np.full(t.shape, np.nan),
             "treat_all": np.full(t.shape, np.nan),
             "cutpoint": np.nan, "derived_cutpoint": np.nan,
             "n": int(n), "n_valid": 0,
             "n_skipped": int(n_boot), "prevalence": np.nan}
    if n == 0 or y.sum() < MIN_PER_ARM or (n - y.sum()) < MIN_PER_ARM:
        return blank

    derived = select(sweep(y, x, direction), criterion,
                     auc=auc_with_ci(y, x, direction)["auc"])
    if not np.isfinite(derived):
        return blank

    # The rule scored here must be the rule the manuscript prints, which is the
    # rounded one. Scoring the criterion's raw optimum instead makes this table
    # describe a different rule from every other table in the phase: on the
    # edema index the two differ by 0.0000155, one high-grade patient sits
    # between them, and the published net benefit moves by 0.003.
    #
    # Whether the value handed in is the right one is checked by the caller,
    # which knows the measurement's own printing precision. Here it is simply
    # honoured, and the raw optimum is returned beside it so it can be.
    observed = float(cutoff) if cutoff is not None and np.isfinite(cutoff) else derived

    apparent_rule = rule_curve(y, x, observed, direction, t)
    apparent_model = model_curve(y, x, log_x, t)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    rule_gaps, model_gaps = [], []
    skipped = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        y_in, x_in = y[idx], x[idx]
        held = np.setdiff1d(all_idx, np.unique(idx), assume_unique=False)
        if (held.size < MIN_HELD_OUT or y_in.sum() < MIN_PER_ARM
                or (n - y_in.sum()) < MIN_PER_ARM
                or y[held].sum() < 1 or y[held].size - y[held].sum() < 1):
            skipped += 1
            continue
        y_out, x_out = y[held], x[held]
        try:
            c_b = select(sweep(y_in, x_in, direction), criterion,
                         auc=auc_with_ci(y_in, x_in, direction)["auc"])
        except Exception:
            skipped += 1
            continue
        if np.isfinite(c_b):
            gap = (rule_curve(y_in, x_in, c_b, direction, t)
                   - rule_curve(y_out, x_out, c_b, direction, t))
            if np.all(np.isfinite(gap)):
                rule_gaps.append(gap)
        fitted = _irls(y_in.astype(float), _model_scale(x_in, log_x))
        if fitted is not None:
            gap = (model_curve(y_in, x_in, log_x, t, coefficients=fitted)
                   - model_curve(y_in, x_in, log_x, t, coefficients=fitted,
                                 score_on=(y_out, x_out)))
            if np.all(np.isfinite(gap)):
                model_gaps.append(gap)

    def _corrected(apparent: np.ndarray, gaps: list) -> np.ndarray:
        if not gaps:
            return np.full(t.shape, np.nan)
        return apparent - np.mean(np.vstack(gaps), axis=0)

    return {
        "rule": apparent_rule,
        "rule_corrected": _corrected(apparent_rule, rule_gaps),
        "model": apparent_model,
        "model_corrected": _corrected(apparent_model, model_gaps),
        "treat_all": treat_all_curve(y, t),
        "cutpoint": float(observed), "derived_cutpoint": float(derived),
        "n": int(n),
        "n_valid": len(rule_gaps), "n_skipped": int(skipped),
        "prevalence": float(y.mean()),
    }


# --------------------------------------------------------------------------
# Reading the curves
# --------------------------------------------------------------------------
def useful_range(curve: np.ndarray, treat_all: np.ndarray,
                 thresholds: np.ndarray) -> tuple[float, float]:
    """The span of thresholds where a rule beats both treating all and none.

    The widest *contiguous* run, not every threshold that happens to qualify: a
    rule useful at 10% and again at 40% but not between is not a rule anyone can
    follow, and reporting "10% to 40%" would hide that.
    """
    t = np.asarray(thresholds, dtype=float)
    ok = (np.asarray(curve) > np.asarray(treat_all) + NEGLIGIBLE) & (
        np.asarray(curve) > NEGLIGIBLE)
    if not ok.any():
        return (np.nan, np.nan)
    best = (0, -1)
    start = None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    return (float(t[best[0]]), float(t[best[1]]))


def at_threshold(curve: np.ndarray, thresholds: np.ndarray,
                 target: float) -> float:
    """The curve's value at the grid point nearest ``target``."""
    t = np.asarray(thresholds, dtype=float)
    if t.size == 0 or not np.isfinite(target):
        return np.nan
    return float(np.asarray(curve)[int(np.argmin(np.abs(t - target)))])


def _nearest(thresholds: np.ndarray, target: float) -> float:
    """The grid threshold a value is read at — the one the curves were built on."""
    t = np.asarray(thresholds, dtype=float)
    if t.size == 0 or not np.isfinite(target):
        return np.nan
    return float(t[int(np.argmin(np.abs(t - target)))])


def net_reduction_per_100(curve: np.ndarray, treat_all: np.ndarray,
                          thresholds: np.ndarray, target: float) -> float:
    """How many fewer patients get flagged per 100, for the same tumors found.

    Net benefit is in units of true positives, which nobody has an intuition
    for. Dividing the gain over treat-all by the exchange rate ``t/(1−t)``
    converts it into the currency the reader does have one for: patients not
    flagged unnecessarily, at no cost in high-grade tumors missed.

    The exchange rate uses the grid threshold the gain was actually read at, not
    the raw target. Reading a gain at t = .30 and dividing it by the odds at
    t = .2983 mixes two thresholds inside one division, and the reader who
    checks the footnote's formula against the printed number gets a different
    answer — 14 where the table says 15.
    """
    t_used = _nearest(thresholds, target)
    gain = at_threshold(np.asarray(curve) - np.asarray(treat_all), thresholds,
                        target)
    if not np.isfinite(gain) or not np.isfinite(t_used) or t_used >= 1.0:
        return np.nan
    odds = t_used / (1.0 - t_used)
    return float(100.0 * gain / odds) if odds > 0 else np.nan


def decision_table(df: pd.DataFrame, eligible: pd.DataFrame,
                   cutpoints: dict[str, float], *,
                   max_threshold: float = DEFAULT_MAX_THRESHOLD,
                   n_boot: int = N_BOOTSTRAP, seed: int = SEED
                   ) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Decision curves for every measurement that carried forward.

    Returns the summary table and the curves themselves, because the table
    cannot show a line crossing another line and the figure is drawn from the
    same arrays the table was read from.
    """
    t = threshold_grid(max_threshold)
    rows, curves = [], {}
    for _, row in eligible.iterrows():
        if row["col"] not in cutpoints:
            continue
        m: Measurement = MEASUREMENTS_BY_COL[row["col"]]
        sub = df.loc[stratum_mask(df, m, row["stratum"])]
        out = corrected_curves(
            pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy(),
            pd.to_numeric(sub[m.col], errors="coerce").to_numpy(),
            m.direction, m.log_x, t, cutoff=m.round(cutpoints[row["col"]]),
            n_boot=n_boot, seed=seed)

        # Same contract as step 7, checked the same way: the criterion must still
        # land on the published number once rounded to the precision that
        # measurement prints at. Rounding may move a cut-point; nothing else may.
        published = m.round(cutpoints[row["col"]])
        if (np.isfinite(out["derived_cutpoint"])
                and not np.isclose(m.round(out["derived_cutpoint"]), published)):
            raise FrozenCutpointError(
                f"{m.label}: the decision curve prints {published} but the "
                f"criterion re-derives {m.round(out['derived_cutpoint'])} on "
                "this cohort. The curve may be corrected; the point estimate "
                "may not move.")
        out["thresholds"] = t
        out["measurement"] = row["measurement"]
        curves[row["col"]] = out

        prevalence = out["prevalence"]
        rule_lo, rule_hi = useful_range(out["rule_corrected"], out["treat_all"], t)
        model_lo, model_hi = useful_range(out["model_corrected"],
                                          out["treat_all"], t)
        # Where comparing the two forms means anything. Outside the span in
        # which one of them is useful at all, the continuous model has degraded
        # into treat-all or treat-none and the difference between it and the
        # cut-point is a difference against a strategy that needs no scan.
        lo = np.nanmin([rule_lo, model_lo])
        hi = np.nanmax([rule_hi, model_hi])
        out["decidable"] = ((t >= lo) & (t <= hi) if np.isfinite(lo)
                            else np.zeros(t.shape, dtype=bool))
        rule_nb = at_threshold(out["rule_corrected"], t, prevalence)
        model_nb = at_threshold(out["model_corrected"], t, prevalence)
        rows.append({
            "measurement": row["measurement"], "col": row["col"],
            "stratum": row["stratum"], "claim": row.get("claim", ""),
            "n": out["n"], "prevalence": prevalence,
            "cutpoint": m.round(cutpoints[row["col"]]),
            "rule_useful_from": rule_lo, "rule_useful_to": rule_hi,
            "model_useful_from": model_lo, "model_useful_to": model_hi,
            "nb_rule": rule_nb, "nb_model": model_nb,
            "nb_treat_all": at_threshold(out["treat_all"], t, prevalence),
            "nb_rule_apparent": at_threshold(out["rule"], t, prevalence),
            "reduction_per_100": net_reduction_per_100(
                out["rule_corrected"], out["treat_all"], t, prevalence),
            "rule_beats_alternatives": bool(np.isfinite(rule_lo)),
            "number_beats_rule": bool(
                np.isfinite(rule_nb) and np.isfinite(model_nb)
                and model_nb > rule_nb + NEGLIGIBLE),
            "n_valid": out["n_valid"], "n_skipped": out["n_skipped"],
        })
    return pd.DataFrame(rows), curves


def describe_decision(table: pd.DataFrame) -> str:
    """One line: which rules are worth following, and over what range."""
    if table is None or table.empty:
        return "No measurement could be put on a decision curve."
    useful = table[table["rule_beats_alternatives"]]
    parts = []
    if useful.empty:
        parts.append(
            "No yes/no rule beats treating everyone as high grade at any "
            "threshold a clinician would plausibly hold, once the rule is "
            "charged for having been chosen on these patients.")
    else:
        named = join_names(
            f"{r['measurement']} from {r['rule_useful_from']:.0%} to "
            f"{r['rule_useful_to']:.0%} ({r['reduction_per_100']:.0f} fewer "
            f"patients flagged per 100 at the base rate)"
            for _, r in useful.iterrows())
        parts.append(f"Worth acting on — {named}.")
    better_as_number = table[table["number_beats_rule"]]
    if not better_as_number.empty:
        parts.append(
            "Kept as a number rather than cut in two, these do better at the "
            f"base rate: {join_names(better_as_number['measurement'])}.")
    return " ".join(parts)
