"""Step 4 — does risk climb smoothly, or is there a genuine bend?

A threshold is only meaningful if risk *changes character* somewhere. If risk
rises steadily across the whole range, any line drawn through it is an
administrative convenience, not a biological boundary — and the manuscript
should say that rather than dress a convenience up as a discovery.

The test fits two models and compares them:

*straight*  risk rises at a constant rate on the log-odds scale
*bent*      a restricted cubic spline — three straight pieces joined smoothly at
            knots placed where the patients are, and forced straight beyond the
            outermost knots so the tails cannot flap

Because the spline's first column is the variable itself, dropping the extra
columns gives back exactly the straight model. That nesting is what makes the
**likelihood-ratio test** valid: it asks whether the extra flexibility bought
enough fit to be worth its degrees of freedom. A small P value says the bend is
real; a large one says the straight line already explained everything.

**Scale matters, and it is not a detail.** Whether a curve looks bent depends on
what you plot it against. Tumour volume in this cohort tests p = 0.97 for a bend
on the log scale and p = 0.02 on the raw scale — the same patients, opposite
conclusions. So the primary test runs in **clinical units**, because that is the
scale the cut-point is quoted in and testing on one scale while reporting on
another would be incoherent. The log-scale fit runs alongside as a sensitivity
analysis, so the dependence is visible rather than chosen quietly.

Note this is a *different* use of the log transform from the one declared in
:mod:`measurements`. There, ``log_x`` controls standardisation for the odds
ratio per 1-SD. Here it selects the sensitivity fit only. The two must not be
collapsed into one flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from measurements import MEASUREMENTS, Measurement, stratum_mask

OUTCOME = "high_grade"

# Harrell, *Regression Modeling Strategies*, Table 2.3 — knot quantiles that put
# the knots where the data are, not where the axis is.
KNOT_QUANTILES: dict[int, tuple[float, ...]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}

DEFAULT_N_KNOTS = 3
MIN_PATIENTS = 20
MIN_PER_ARM = 5


def default_knots(x: np.ndarray, n_knots: int = DEFAULT_N_KNOTS) -> np.ndarray:
    """Harrell's quantile knots, reduced until they are distinct.

    A zero-inflated measurement collapses the lower quantiles onto the same
    value, which makes the basis rank-deficient. Dropping to fewer knots is the
    graceful failure; the caller finds out through ``n_knots``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    for k in range(int(n_knots), 2, -1):
        qs = KNOT_QUANTILES.get(k)
        if qs is None:
            continue
        knots = np.unique(np.quantile(x, qs))
        if knots.size == k:
            return knots
    return np.array([], dtype=float)


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis with ``len(knots) - 1`` columns.

    Column 0 is ``x`` itself, so dropping the rest gives exactly the straight
    model — the nesting the likelihood-ratio test depends on.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    k = knots.size
    if k < 3:
        return x.reshape(-1, 1)

    t_last, t_prev, t_first = knots[-1], knots[-2], knots[0]
    denom = t_last - t_prev
    scale = (t_last - t_first) ** 2
    if denom <= 0 or scale <= 0:
        return x.reshape(-1, 1)

    def cube_plus(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v ** 3, 0.0)

    cols = [x]
    for j in range(k - 2):
        tj = knots[j]
        term = (cube_plus(x - tj)
                - cube_plus(x - t_prev) * (t_last - tj) / denom
                + cube_plus(x - t_last) * (t_prev - tj) / denom)
        cols.append(term / scale)
    return np.column_stack(cols)


def _design(x: np.ndarray, knots: np.ndarray, *,
            linear_only: bool = False) -> np.ndarray:
    basis = x.reshape(-1, 1) if linear_only else rcs_basis(x, knots)
    return sm.add_constant(basis, has_constant="add")


def _fit_glm(design: np.ndarray, y: np.ndarray):
    """Binomial GLM, or None when it will not converge."""
    if not np.all(np.isfinite(design)):
        return None
    try:
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return None
    except np.linalg.LinAlgError:
        return None
    try:
        return sm.GLM(y, design, family=sm.families.Binomial()).fit()
    except Exception:      # separation, singular matrix — caller falls back
        return None


@dataclass(frozen=True)
class SplineFit:
    """One fitted curve and the test of whether it needed to bend.

    ``grid`` and ``risk`` are always on the measurement's **original** scale,
    even when the fit ran on ``log1p``, because that is the scale a radiologist
    reads off a scan. Step 5 reads the bend location off these.
    """

    column: str
    stratum: str
    n: int
    events: int
    log_fitted: bool
    knots: np.ndarray
    n_knots: int
    spline_fitted: bool
    lr_stat: float
    lr_df: int
    lr_p: float
    grid: np.ndarray
    risk: np.ndarray
    risk_lo: np.ndarray
    risk_hi: np.ndarray
    prevalence: float
    note: str

    @property
    def bent(self) -> bool:
        """Did the extra flexibility earn its keep at the 5% level?"""
        return bool(np.isfinite(self.lr_p) and self.lr_p < 0.05)


def fit_spline(x: np.ndarray, y: np.ndarray, *, column: str = "",
               stratum: str = "all", log_fit: bool = False,
               n_knots: int = DEFAULT_N_KNOTS, grid_n: int = 200,
               grid_quantiles: tuple[float, float] = (0.025, 0.975)) -> SplineFit:
    """Fit the bent and straight models and test one against the other.

    The grid is trimmed to the inner 95% of observed values: a risk curve means
    nothing where there are no patients, and a spline is least trustworthy at
    its ends.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    n, events = int(x.size), int(y.sum())
    empty = np.array([], dtype=float)

    def _blank(note: str) -> SplineFit:
        return SplineFit(
            column=column, stratum=stratum, n=n, events=events,
            log_fitted=log_fit, knots=empty, n_knots=0, spline_fitted=False,
            lr_stat=np.nan, lr_df=0, lr_p=np.nan,
            grid=empty, risk=empty, risk_lo=empty, risk_hi=empty,
            prevalence=float(y.mean()) if n else np.nan, note=note)

    if n < MIN_PATIENTS or events < MIN_PER_ARM or (n - events) < MIN_PER_ARM:
        return _blank("too few patients or events for a spline")
    if log_fit and x.min() <= -1.0:
        return _blank("log fit needs values above -1")

    # log1p rather than log, because edema volume is legitimately zero.
    u = np.log1p(x) if log_fit else x
    knots = default_knots(u, n_knots)

    full = _fit_glm(_design(u, knots), y) if knots.size >= 3 else None
    reduced = _fit_glm(_design(u, knots, linear_only=True), y)
    if reduced is None:
        return _blank("logistic fit did not converge")

    spline_fitted = full is not None
    model = full if spline_fitted else reduced
    used_knots = knots if spline_fitted else empty
    if spline_fitted:
        lr_df = int(full.df_model - reduced.df_model)
        lr_stat = float(2.0 * (full.llf - reduced.llf))
        lr_p = float(chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan
    else:
        lr_df, lr_stat, lr_p = 0, np.nan, np.nan

    lo, hi = np.quantile(x, grid_quantiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _blank("observed range is degenerate")
    grid = np.linspace(float(lo), float(hi), int(grid_n))
    grid_u = np.log1p(grid) if log_fit else grid
    pred = model.get_prediction(
        _design(grid_u, used_knots, linear_only=not spline_fitted)
    ).summary_frame(alpha=0.05)

    return SplineFit(
        column=column, stratum=stratum, n=n, events=events,
        log_fitted=log_fit, knots=used_knots, n_knots=int(used_knots.size),
        spline_fitted=spline_fitted,
        lr_stat=lr_stat, lr_df=lr_df, lr_p=lr_p,
        grid=grid,
        risk=pred["mean"].to_numpy(dtype=float),
        risk_lo=pred["mean_ci_lower"].to_numpy(dtype=float),
        risk_hi=pred["mean_ci_upper"].to_numpy(dtype=float),
        prevalence=float(y.mean()),
        note="" if spline_fitted else "spline rank-deficient - straight fit used")


def fit_all(df: pd.DataFrame,
            measurements: Sequence[Measurement] = MEASUREMENTS,
            *, n_knots: int = DEFAULT_N_KNOTS
            ) -> dict[tuple[str, str, bool], SplineFit]:
    """Every fit step 4 and step 5 need, keyed ``(column, stratum, log_fitted)``.

    Both scales are fitted for every measurement, not only the skewed ones. The
    sensitivity analysis is only informative if it was run everywhere it could
    have disagreed.
    """
    fits: dict[tuple[str, str, bool], SplineFit] = {}
    for m in measurements:
        for stratum in m.strata:
            sub = df.loc[stratum_mask(df, m, stratum)]
            x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
            y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
            for log_fit in (False, True):
                fits[(m.col, stratum, log_fit)] = fit_spline(
                    x, y, column=m.col, stratum=stratum, log_fit=log_fit,
                    n_knots=n_knots)
    return fits


def nonlinearity_table(df: pd.DataFrame,
                       measurements: Sequence[Measurement] = MEASUREMENTS,
                       *, fits: dict | None = None,
                       n_knots: int = DEFAULT_N_KNOTS) -> pd.DataFrame:
    """One row per measurement per stratum, both scales side by side.

    ``scales_agree`` is the column that decides how the finding is written up.
    Where the two scales disagree, no claim about a bend is scale-free, and the
    manuscript has to report the dependence instead of picking the answer it
    prefers.
    """
    fits = fits if fits is not None else fit_all(df, measurements,
                                                 n_knots=n_knots)
    rows = []
    for m in measurements:
        for stratum in m.strata:
            clinical = fits[(m.col, stratum, False)]
            logged = fits[(m.col, stratum, True)]
            agree = (np.isfinite(clinical.lr_p) and np.isfinite(logged.lr_p)
                     and clinical.bent == logged.bent)
            rows.append({
                "measurement": m.stratum_label(stratum),
                "col": m.col,
                "stratum": stratum,
                "n": clinical.n,
                "events": clinical.events,
                "n_knots": clinical.n_knots,
                "lr_stat": clinical.lr_stat,
                "lr_df": clinical.lr_df,
                "lr_p": clinical.lr_p,
                "bent_clinical": clinical.bent,
                "lr_p_log": logged.lr_p,
                "bent_log": logged.bent,
                "scales_agree": bool(agree),
                "note": clinical.note,
            })
    return pd.DataFrame(rows)


def describe_nonlinearity(table: pd.DataFrame) -> str:
    """Two sentences: what bent, and where the scales disagreed."""
    scored = table.dropna(subset=["lr_p"])
    if scored.empty:
        return "No measurement could be fitted."
    bent = scored[scored["bent_clinical"]]
    if bent.empty:
        first = ("No measurement shows a bend in clinical units — every "
                 "association is consistent with a steady climb, so any "
                 "cut-point is a convenience rather than a boundary.")
    else:
        named = ", ".join(f"{r['measurement']} (P={r['lr_p']:.3f})"
                          for _, r in bent.iterrows())
        first = f"Bend detected in clinical units: {named}."
    disagree = scored[~scored["scales_agree"]]
    if disagree.empty:
        return first + " Both scales agree everywhere."
    names = ", ".join(disagree["measurement"])
    return first + (f" Scale-dependent, so no claim about a bend is scale-free: "
                    f"{names}.")
