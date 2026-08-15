"""Step 5 — if risk bends, where does it bend?

Step 4 asked *whether* the climb changes character. This asks *where*, and
answers it two ways, because they are different clinical questions:

**The knee** — the value at which risk is climbing fastest. This is the
dose-response answer: the point where one more cm³ buys the most extra risk.

**The risk crossing** — the value at which predicted risk reaches 30% or 50%.
This is the counselling answer: "above this number, one patient in two is high
grade." It does not depend on the curve's shape, only on where it passes a line
that matters to a clinician.

Two guards on reading either of them.

*A steepest point exists on every curve, including a perfectly straight one.*
On a straight line it is wherever the arithmetic happens to wobble, and it means
nothing. So the knee is reported alongside step 4's verdict, and only quoted for
measurements whose bend survived the test.

*A steepest point at the edge of the data is an artefact.* Splines are least
stable at their ends. Interiority is judged in **patients**, not axis units:
edema volume runs to 197 cm³ but half the cohort sits below 4.5, so a knee at
3.5 cm³ is 2% along the axis and at the 48th percentile of patients. Judging it
by axis position would throw away a knee sitting in the middle of the data.

The knee has no interval here. That comes from step 7, which re-derives it in
each of two thousand resampled cohorts — and a knee whose interval spans most
of the observed range is not a landmark, however precise this point estimate
looks.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from intervals import wilson_ci
from measurements import MEASUREMENTS, LOWER, HIGHER, Measurement, stratum_mask
from nonlinearity import SplineFit, fit_all

OUTCOME = "high_grade"

# A steepest point with fewer than this percent of patients on either side of it
# is a boundary artefact rather than an interior knee.
BOUNDARY_PERCENTILE = 5.0

DEFAULT_RISK_LEVELS = (0.30, 0.50)


def risk_crossing(grid: np.ndarray, risk: np.ndarray, level: float,
                  direction: str) -> float:
    """Measurement value at the outer edge of the region where risk >= ``level``.

    For a measurement where high values are suspicious this is the *lowest*
    value still carrying that risk; where low values are suspicious, the
    *highest*. Taking the outer boundary rather than the first crossing keeps
    the answer stable if the fitted curve wobbles across the line twice.
    """
    grid = np.asarray(grid, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if grid.size == 0:
        return float("nan")
    above = risk >= level
    if not above.any() or above.all():
        return float("nan")   # never reaches that risk, or never drops below it

    idx = np.flatnonzero(above)
    edge = idx.min() if direction == HIGHER else idx.max()
    neighbour = edge - 1 if direction == HIGHER else edge + 1
    if neighbour < 0 or neighbour >= grid.size:
        return float(grid[edge])
    r0, r1 = risk[neighbour], risk[edge]
    if r1 == r0:
        return float(grid[edge])
    frac = (level - r0) / (r1 - r0)
    return float(grid[neighbour] + frac * (grid[edge] - grid[neighbour]))


def steepest_point(fit: SplineFit, direction: str,
                   x_observed: np.ndarray) -> dict[str, float]:
    """Where the curve climbs fastest, and whether that point is trustworthy.

    The slope is taken on the measurement's **original** scale — "risk added per
    extra cm³" — even when the spline was fitted on the log scale, because that
    is the quantity the question is about.
    """
    blank = {"knee": np.nan, "knee_risk": np.nan, "knee_slope": np.nan,
             "knee_percentile": np.nan, "knee_at_boundary": True}
    if fit.grid.size < 3:
        return blank
    slope = np.gradient(fit.risk, fit.grid)
    signed = slope if direction == HIGHER else -slope
    if not np.isfinite(signed).any():
        return blank
    i = int(np.nanargmax(signed))
    x_observed = np.asarray(x_observed, dtype=float)
    x_observed = x_observed[np.isfinite(x_observed)]
    pct = (float((x_observed < fit.grid[i]).mean() * 100.0)
           if x_observed.size else np.nan)
    at_boundary = bool(not np.isfinite(pct)
                       or pct < BOUNDARY_PERCENTILE
                       or pct > 100.0 - BOUNDARY_PERCENTILE)
    return {"knee": float(fit.grid[i]),
            "knee_risk": float(fit.risk[i]),
            "knee_slope": float(slope[i]),
            "knee_percentile": pct,
            "knee_at_boundary": at_boundary}


def observed_bins(x: np.ndarray, y: np.ndarray, *, n_bins: int = 8,
                  min_per_bin: int = 10) -> pd.DataFrame:
    """Observed high-grade proportion in equal-count bins, with Wilson intervals.

    The honesty check under the fitted curve: if these dots do not track the
    spline, the curve is telling a story the patients do not support. Equal
    *count* bins rather than equal width, so no bin rests on three patients.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    columns = ["x", "observed", "lo", "hi", "n", "events"]
    if x.size < 2 * min_per_bin:
        return pd.DataFrame(columns=columns)
    n_bins = min(int(n_bins), max(2, x.size // max(min_per_bin, 1)))
    try:
        bins = pd.qcut(x, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame({"x": x, "y": y, "bin": bins})
    rows = []
    for _, grp in frame.groupby("bin", observed=True):
        n, k = int(len(grp)), int(grp["y"].sum())
        if n < min_per_bin:
            continue
        lo, hi = wilson_ci(k, n)
        rows.append({"x": float(grp["x"].median()), "observed": k / n,
                     "lo": lo, "hi": hi, "n": n, "events": k})
    return pd.DataFrame(rows, columns=columns)


def bend_table(df: pd.DataFrame,
               measurements: Sequence[Measurement] = MEASUREMENTS, *,
               fits: dict | None = None,
               risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS
               ) -> pd.DataFrame:
    """Knee and risk crossings for every measurement, with the caveats attached.

    Computed for all rows, quotable only for some. ``bend_is_real`` carries step
    4's verdict forward so no reader has to hold two tables in their head to
    know whether a knee means anything.
    """
    fits = fits if fits is not None else fit_all(df, measurements)
    rows = []
    for m in measurements:
        for stratum in m.strata:
            clinical = fits[(m.col, stratum, False)]
            logged = fits[(m.col, stratum, True)]
            sub = df.loc[stratum_mask(df, m, stratum)]
            x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
            row = {
                "measurement": m.stratum_label(stratum),
                "col": m.col,
                "stratum": stratum,
                "n": clinical.n,
                "bend_is_real": clinical.bent,
                "lr_p": clinical.lr_p,
                "scales_agree": bool(np.isfinite(clinical.lr_p)
                                     and np.isfinite(logged.lr_p)
                                     and clinical.bent == logged.bent),
            }
            row.update(steepest_point(clinical, m.direction, x))
            for level in risk_levels:
                row[f"risk_{int(round(level * 100))}"] = risk_crossing(
                    clinical.grid, clinical.risk, float(level), m.direction)
            row["quotable"] = bool(row["bend_is_real"]
                                   and not row["knee_at_boundary"])
            rows.append(row)
    return pd.DataFrame(rows)


def describe_bend(table: pd.DataFrame,
                  measurements: Sequence[Measurement] = MEASUREMENTS) -> str:
    """One line naming the knees that may be quoted, and why the rest may not."""
    by_col = {m.col: m for m in measurements}
    quotable = table[table["quotable"]]
    if quotable.empty:
        return ("No knee can be quoted: every curve either did not bend or bent "
                "only at the edge of the data.")
    named = "; ".join(
        f"{r['measurement']} {by_col[r['col']].op} "
        f"{by_col[r['col']].round(r['knee']):g} "
        f"{by_col[r['col']].unit}".strip()
        + (" (scale-dependent)" if not r["scales_agree"] else "")
        for _, r in quotable.iterrows())
    withheld = len(table) - len(quotable)
    line = f"Knees that may be quoted — {named}."
    if withheld:
        line += (f" {withheld} of {len(table)} withheld: no bend, or a bend "
                 "sitting at the edge of the data.")
    return line
