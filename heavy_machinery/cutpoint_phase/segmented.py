"""Segmented logistic regression — a breakpoint estimated, not read off a curve.

The spline in :mod:`nonlinearity` answers *does risk bend*. It cannot answer
*where does it break*, because a spline has no break: it curves smoothly, and
the "steepest point" is a place on that curve rather than a parameter with a
standard error. A threshold claim needs the second thing.

So this fits the model the claim actually implies — two straight lines meeting
at a point:

    logit(p) = β₀ + β₁·x + β₂·(x − ψ)⁺

``ψ`` is the breakpoint: below it the slope is β₁, above it β₁ + β₂. Unlike the
spline's steepest point, ψ is estimated, and an interval can be put on it.

**How ψ is estimated.** By profile likelihood over a grid of candidate values
rather than by Muggeo's iterative linearisation. The likelihood surface for ψ is
not smooth and can have local maxima; an iterative method can settle into one
and report it confidently. A grid sees the whole surface, and the same surface
gives the interval for free — every ψ whose deviance is within 3.84 of the best
belongs in the 95% interval.

**Why the P value is not a likelihood-ratio test.** Under the null hypothesis of
no breakpoint, β₂ is zero — and when β₂ is zero, ψ has no effect on the model at
all. A parameter that does not exist under the null breaks the assumptions the
χ² distribution rests on, so the usual likelihood-ratio P value is
anti-conservative: it finds breakpoints in straight lines. This is Davies'
problem, and it is the single most common error in applied segmented regression.

Both P values are returned. ``lr_p`` is the naive one, kept only so the
difference is visible; ``davies_p`` is the one to report, an upper bound that
accounts for having searched every candidate ψ for the best one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask

OUTCOME = "high_grade"

# Candidate breakpoints are drawn from the inner range only. A break in the
# outermost few percent is fitted to a handful of patients and is an artefact of
# the tail, not a threshold anyone could apply.
INNER_QUANTILES = (0.10, 0.90)
N_GRID = 60
MIN_PER_SIDE = 20
MIN_EVENTS_PER_SIDE = 5

# 95% profile-likelihood interval: every ψ whose deviance is within this of the
# best. One degree of freedom, because only ψ is being profiled.
DEVIANCE_CUTOFF = float(chi2.ppf(0.95, df=1))


class SegmentedError(Exception):
    """The segmented model cannot be fitted to what was passed in."""


@dataclass(frozen=True)
class SegmentedFit:
    """A fitted breakpoint, its interval, and the two P values that bracket it."""

    column: str
    n: int
    events: int
    breakpoint: float
    ci_lo: float
    ci_hi: float
    slope_below: float
    slope_above: float
    slope_change: float
    lr_stat: float
    lr_p: float          # naive, anti-conservative — for comparison only
    davies_p: float      # the one to report
    delta_aic: float
    grid: np.ndarray
    profile_llf: np.ndarray
    note: str

    @property
    def supported(self) -> bool:
        """A breakpoint worth reporting, judged on Davies' P, not the naive one."""
        return bool(np.isfinite(self.davies_p) and self.davies_p < 0.05)

    @property
    def ci_width(self) -> float:
        return self.ci_hi - self.ci_lo


def _design(x: np.ndarray, psi: float) -> np.ndarray:
    """Intercept, slope, and the extra slope that switches on above ``psi``."""
    return sm.add_constant(
        np.column_stack([x, np.clip(x - psi, 0.0, None)]), has_constant="add")


def _fit(design: np.ndarray, y: np.ndarray):
    try:
        return sm.GLM(y, design, family=sm.families.Binomial()).fit()
    except Exception:
        return None


def candidate_breakpoints(x: np.ndarray, *, n_grid: int = N_GRID,
                          inner: tuple[float, float] = INNER_QUANTILES
                          ) -> np.ndarray:
    """Evenly spaced candidates across the inner range, deduplicated."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=float)
    lo, hi = np.quantile(x, inner)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([], dtype=float)
    return np.unique(np.linspace(float(lo), float(hi), int(n_grid)))


def fit_segmented(x: np.ndarray, y: np.ndarray, *, column: str = "",
                  n_grid: int = N_GRID, min_per_side: int = MIN_PER_SIDE,
                  min_events_per_side: int = MIN_EVENTS_PER_SIDE
                  ) -> SegmentedFit:
    """Estimate the breakpoint by profiling the likelihood over candidate ψ."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    n, events = int(x.size), int(y.sum())
    empty = np.array([], dtype=float)

    def _blank(note: str) -> SegmentedFit:
        return SegmentedFit(
            column=column, n=n, events=events, breakpoint=np.nan,
            ci_lo=np.nan, ci_hi=np.nan, slope_below=np.nan, slope_above=np.nan,
            slope_change=np.nan, lr_stat=np.nan, lr_p=np.nan, davies_p=np.nan,
            delta_aic=np.nan, grid=empty, profile_llf=empty, note=note)

    if n < 2 * min_per_side or events < 2 * min_events_per_side:
        return _blank("too few patients or events for a segmented fit")

    linear = _fit(sm.add_constant(x.reshape(-1, 1), has_constant="add"), y)
    if linear is None:
        return _blank("straight-line fit did not converge")

    grid = candidate_breakpoints(x, n_grid=n_grid)
    if grid.size < 3:
        return _blank("observed range is degenerate")

    llf = np.full(grid.size, -np.inf)
    z_stats = np.full(grid.size, np.nan)
    for i, psi in enumerate(grid):
        below, above = int(np.sum(x < psi)), int(np.sum(x >= psi))
        if below < min_per_side or above < min_per_side:
            continue
        if (int(y[x < psi].sum()) < min_events_per_side
                or int(y[x >= psi].sum()) < min_events_per_side):
            continue
        fit = _fit(_design(x, psi), y)
        if fit is None or not np.isfinite(fit.llf):
            continue
        llf[i] = float(fit.llf)
        # The slope-change statistic, kept for Davies' correction: it is the
        # process whose maximum over ψ the correction has to account for.
        try:
            z_stats[i] = float(fit.params[2] / fit.bse[2])
        except (IndexError, ZeroDivisionError, ValueError):
            z_stats[i] = np.nan

    if not np.isfinite(llf).any():
        return _blank("no candidate breakpoint leaves enough patients on both sides")

    best = int(np.nanargmax(np.where(np.isfinite(llf), llf, -np.inf)))
    psi = float(grid[best])
    fit = _fit(_design(x, psi), y)
    if fit is None:
        return _blank("segmented fit did not converge at the best breakpoint")

    lr_stat = float(2.0 * (llf[best] - linear.llf))
    # Two extra parameters over the straight line: the slope change and the
    # breakpoint itself. AIC has to be charged for both.
    delta_aic = float(4.0 - lr_stat)

    inside = grid[llf >= llf[best] - DEVIANCE_CUTOFF / 2.0]
    ci_lo = float(inside.min()) if inside.size else np.nan
    ci_hi = float(inside.max()) if inside.size else np.nan

    slope_below = float(fit.params[1])
    slope_change = float(fit.params[2])
    return SegmentedFit(
        column=column, n=n, events=events, breakpoint=psi,
        ci_lo=ci_lo, ci_hi=ci_hi,
        slope_below=slope_below, slope_above=slope_below + slope_change,
        slope_change=slope_change,
        lr_stat=lr_stat,
        lr_p=float(chi2.sf(lr_stat, 2)) if np.isfinite(lr_stat) else np.nan,
        davies_p=davies_p(z_stats),
        delta_aic=delta_aic, grid=grid, profile_llf=llf,
        note="" if inside.size > 1 else
        "profile interval collapsed to a single candidate")


def davies_p(z_stats: np.ndarray) -> float:
    """Davies' upper bound on the P value for a parameter absent under the null.

    The naive test asks "is the best breakpoint significant?" while ignoring
    that the best was chosen by looking at every candidate. Davies' correction
    charges for that search: it adds a term for how much the test statistic
    *wanders* across the candidates, so a jagged profile — one that found its
    maximum by luck — is penalised more than a smooth one.

    Returns an upper bound, so it is conservative by construction. Reporting a
    bound and calling it a P value is honest; reporting the uncorrected value is
    not, and it will find breakpoints in straight lines.
    """
    z = np.asarray(z_stats, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return float("nan")
    peak = float(np.max(np.abs(z)))
    total_variation = float(np.sum(np.abs(np.diff(np.abs(z)))))
    tail = float(2 * norm.sf(peak))          # two-sided, uncorrected
    bound = tail + total_variation * np.exp(-(peak ** 2) / 2.0) / np.sqrt(2 * np.pi)
    return float(min(1.0, bound))


def segmented_table(df: pd.DataFrame, eligible: pd.DataFrame, *,
                    n_grid: int = N_GRID) -> pd.DataFrame:
    """One row per eligible measurement, on the clinical scale it is quoted in."""
    rows = []
    for _, row in eligible.iterrows():
        m: Measurement = MEASUREMENTS_BY_COL[row["col"]]
        sub = df.loc[stratum_mask(df, m, row["stratum"])]
        fit = fit_segmented(
            pd.to_numeric(sub[m.col], errors="coerce").to_numpy(),
            pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy(),
            column=m.col, n_grid=n_grid)
        rows.append({
            "measurement": m.label, "col": m.col, "stratum": row["stratum"],
            "n": fit.n, "events": fit.events,
            "breakpoint": m.round(fit.breakpoint),
            "ci_lo": m.round(fit.ci_lo), "ci_hi": m.round(fit.ci_hi),
            "ci_width": fit.ci_width,
            "slope_below": fit.slope_below, "slope_above": fit.slope_above,
            "slope_change": fit.slope_change,
            "lr_stat": fit.lr_stat, "lr_p_naive": fit.lr_p,
            "davies_p": fit.davies_p, "delta_aic": fit.delta_aic,
            "breakpoint_supported": fit.supported, "note": fit.note,
        })
    return pd.DataFrame(rows)


def describe_segmented(table: pd.DataFrame) -> str:
    """One line: which breakpoints survive the correction, and which do not."""
    if table.empty:
        return "No segmented model could be fitted."
    supported = table[table["breakpoint_supported"]]
    if supported.empty:
        return ("No breakpoint survives Davies' correction: every apparent break "
                "is within what searching every candidate would produce by "
                "chance.")
    named = "; ".join(
        f"{r['measurement']} at {r['breakpoint']:g} "
        f"({r['ci_lo']:g}-{r['ci_hi']:g}, P={r['davies_p']:.3f})"
        for _, r in supported.iterrows())
    return f"Breakpoint supported — {named}."
