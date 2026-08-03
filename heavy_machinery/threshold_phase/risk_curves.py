"""Risk curves — where does the probability of high grade climb most steeply?

This is the "šķēre" question, and it is **not** a ROC question. Youden's J asks
which cut-point separates the two grades best under a symmetric cost; it says
nothing about the *shape* of the risk. A metric whose risk rises perfectly
smoothly still has a Youden cut-point, and quoting it as "the point where risk
jumps" would be wrong.

What is fitted here is a logistic regression of the outcome on the metric,
entered as a **restricted cubic spline** — a curve built from cubic pieces
joined at a few knots and forced straight beyond the outer ones, so it bends
where the data are dense and does not fly off in the tails.

Three things come out, in decreasing order of how much you can trust them:

1. **The curve itself**, with a confidence band and the observed proportions
   in bins laid over it. This is the figure; it needs no threshold at all.
2. **Risk-level crossings** — "predicted risk passes 50% at X". Stable, and
   what a clinician actually asks for.
3. **The steepest-rise point** — the maximum of the curve's slope. This is the
   literal answer to the question, and the least stable number in the module:
   it is a maximum of a derivative of a fitted curve. It is only reported as a
   knee when the non-linearity test is significant *and* the maximum falls in
   the interior of the observed range. Otherwise the honest finding is that
   risk rises steadily and no cut-point marks a jump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

import plot_style as ps
from thresholds import HIGHER, LOWER, Metric, metric_arrays, metric_auc

# Harrell, *Regression Modeling Strategies*, Table 2.3 — knot quantiles that put
# the knots where the data are, not where the axis is.
KNOT_QUANTILES: dict[int, tuple[float, ...]] = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}

DEFAULT_N_KNOTS = 3
DEFAULT_RISK_LEVELS = (0.30, 0.50)
# A steepest point with fewer than this percent of patients on either side of it
# is a boundary artefact rather than an interior knee.
#
# Measured in *patients*, not in axis units. Edema volume runs 0–116 cc but half
# the cohort sits below 4.5 cc; a peak at 3.5 cc is 3% along the axis and at the
# 48th percentile of patients. Judging it by axis position would discard a knee
# sitting in the middle of the data purely because the variable has a long tail.
BOUNDARY_PERCENTILE = 5.0


# --------------------------------------------------------------------------
# Restricted cubic spline basis
# --------------------------------------------------------------------------
def default_knots(x: np.ndarray, n_knots: int = DEFAULT_N_KNOTS) -> np.ndarray:
    """Harrell's quantile knots, reduced until they are distinct.

    A zero-inflated metric (a quarter of this cohort has exactly no edema)
    collapses the lower quantiles onto the same value, which would make the
    basis rank-deficient. Dropping to fewer knots is the graceful failure;
    the caller finds out via ``RiskCurve.n_knots``.
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

    Column 0 is ``x`` itself, so dropping the remaining columns gives exactly
    the linear model — which is what makes the nested likelihood-ratio test for
    non-linearity valid.
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
        term = (
            cube_plus(x - tj)
            - cube_plus(x - t_prev) * (t_last - tj) / denom
            + cube_plus(x - t_last) * (t_prev - tj) / denom
        )
        cols.append(term / scale)
    return np.column_stack(cols)


def _design(x: np.ndarray, knots: np.ndarray, *, linear_only: bool = False) -> np.ndarray:
    basis = x.reshape(-1, 1) if linear_only else rcs_basis(x, knots)
    return sm.add_constant(basis, has_constant="add")


def _fit_glm(design: np.ndarray, y: np.ndarray):
    """Binomial GLM, or None when it will not converge (separation, rank deficiency)."""
    # A non-finite cell makes matrix_rank raise LinAlgError rather than return
    # a small number, so it has to be caught before the rank check.
    if not np.all(np.isfinite(design)):
        return None
    try:
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return None
    except np.linalg.LinAlgError:
        return None
    try:
        return sm.GLM(y, design, family=sm.families.Binomial()).fit()
    except Exception:  # separation, singular matrix — caller falls back
        return None


# --------------------------------------------------------------------------
# The fitted curve
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskCurve:
    """A fitted risk curve and the few numbers that can be read off it.

    ``grid``/``risk`` are always on the metric's **original** scale (cc, ADC
    units) even when the spline was fitted on ``log1p(x)``, because that is the
    scale a radiologist reads off a scan.
    """

    column: str
    n: int
    events: int
    auc: float
    grid: np.ndarray
    risk: np.ndarray
    risk_lo: np.ndarray
    risk_hi: np.ndarray
    slope: np.ndarray
    knots: np.ndarray
    n_knots: int
    log_fitted: bool
    spline_fitted: bool
    lr_stat: float
    lr_df: int
    lr_p: float
    steepest_x: float
    steepest_risk: float
    steepest_slope: float
    steepest_at_boundary: bool
    steepest_pct: float          # percentile of patients below the steepest point
    prevalence: float
    crossings: dict[float, float] = field(default_factory=dict)
    note: str = ""

    @property
    def nonlinear(self) -> bool:
        """Did the spline beat a straight line on the log-odds scale?"""
        return bool(self.spline_fitted and np.isfinite(self.lr_p) and self.lr_p < 0.05)

    @property
    def knee_found(self) -> bool:
        """A defensible "šķēre": real curvature *and* an interior steepest point.

        Both conditions matter, and the curvature one is the subtle half.
        Even a perfectly straight log-odds relationship produces an S-shaped
        *probability* curve whose slope peaks in the middle — at exactly the
        50% crossing. That peak is real but it is not a threshold effect: it
        says nothing the 50% crossing does not already say. Requiring the
        non-linearity test to fire first is what separates "risk changes
        behaviour here" from "risk passes one half here".

        Interiority is the cruder half: a maximum at the edge of the observed
        range means risk was still accelerating (or already flattening) when
        the data ran out, so the location is an artefact of where recruitment
        stopped.
        """
        return bool(self.nonlinear and not self.steepest_at_boundary)

    def verdict(self, metric: Metric) -> str:
        """One sentence, in the words you would put under the figure."""
        if not self.spline_fitted:
            return f"No spline fitted ({self.note or 'insufficient distinct values'})."
        if not self.nonlinear:
            return (
                f"Risk rises steadily with {metric.label} — no curvature "
                f"detected (p {ps.format_p(self.lr_p)}), so no threshold marks a jump."
            )
        if self.steepest_at_boundary:
            return (
                f"Risk is non-linear (p {ps.format_p(self.lr_p)}) but rises fastest at "
                f"the edge of the observed range — no interior threshold."
            )
        where = f"{self.steepest_x:.3g} {metric.prose_unit}".strip()
        return (
            f"Risk climbs most steeply near {metric.label} {where} "
            f"(non-linearity p {ps.format_p(self.lr_p)})."
        )


def fit_risk_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    column: str = "",
    direction: str = HIGHER,
    log_fit: bool = False,
    n_knots: int = DEFAULT_N_KNOTS,
    grid_n: int = 200,
    grid_quantiles: tuple[float, float] = (0.025, 0.975),
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
) -> RiskCurve:
    """Spline logistic risk of the outcome against one continuous metric.

    The grid is trimmed to the inner 95% of observed values: a risk curve is
    only meaningful where there are patients, and the endpoints of a spline are
    where it is least trustworthy.

    ``log_fit`` fits the spline on ``log1p(x)`` instead of ``x``. **This changes
    the question**, so it defaults off: "is the risk bent?" is not a
    scale-invariant claim, and a relationship that is a straight line in
    log(volume) is curved in volume. Tumour volume in this cohort is p = 0.97
    for non-linearity on the log scale and p = 0.02 on the raw scale — same
    patients, opposite conclusion.

    The threshold is reported in the metric's own clinical units, so the test
    has to be run in those units too; testing on the log scale and quoting a
    cut-point in cc would be incoherent. Harrell's quantile knots already place
    the flexibility where the patients are, which is what the log transform was
    there for. ``risk_curve_summary`` runs the log fit anyway and reports it as
    a sensitivity analysis, so the scale dependence is visible rather than a
    choice made quietly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n, events = int(x.size), int(y.sum())

    def _blank(note: str) -> RiskCurve:
        empty = np.array([], dtype=float)
        return RiskCurve(
            column=column, n=n, events=events, auc=np.nan,
            grid=empty, risk=empty, risk_lo=empty, risk_hi=empty, slope=empty,
            knots=empty, n_knots=0, log_fitted=log_fit, spline_fitted=False,
            lr_stat=np.nan, lr_df=0, lr_p=np.nan,
            steepest_x=np.nan, steepest_risk=np.nan, steepest_slope=np.nan,
            steepest_at_boundary=True, steepest_pct=np.nan,
            prevalence=float(y.mean()) if n else np.nan,
            crossings={}, note=note,
        )

    if n < 20 or events < 5 or (n - events) < 5:
        return _blank("too few patients or events for a spline")
    if log_fit and x.size and x.min() <= -1.0:
        # log1p is undefined there. Every clinical metric here is non-negative,
        # so this only guards mirrored or centred inputs.
        return _blank("log fit needs values above −1")

    # Fit on log1p for right-skewed metrics; report everywhere on the raw scale.
    # log1p rather than log because edema volume is legitimately zero.
    u = np.log1p(x) if log_fit else x
    knots = default_knots(u, n_knots)

    full = _fit_glm(_design(u, knots), y) if knots.size >= 3 else None
    reduced = _fit_glm(_design(u, knots, linear_only=True), y)
    if reduced is None:
        return _blank("logistic fit did not converge")

    spline_fitted = full is not None
    model = full if spline_fitted else reduced
    used_knots = knots if spline_fitted else np.array([], dtype=float)
    if spline_fitted:
        lr_df = int(full.df_model - reduced.df_model)
        lr_stat = float(2.0 * (full.llf - reduced.llf))
        lr_p = float(chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan
    else:
        lr_df, lr_stat, lr_p = 0, np.nan, np.nan

    lo, hi = np.quantile(x, grid_quantiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _blank("observed range is degenerate")
    grid = np.linspace(lo, hi, int(grid_n))
    grid_u = np.log1p(grid) if log_fit else grid
    design = _design(grid_u, used_knots, linear_only=not spline_fitted)

    pred = model.get_prediction(design).summary_frame(alpha=0.05)
    risk = pred["mean"].to_numpy(dtype=float)
    risk_lo = pred["mean_ci_lower"].to_numpy(dtype=float)
    risk_hi = pred["mean_ci_upper"].to_numpy(dtype=float)

    # Slope on the ORIGINAL scale: "risk added per extra cc", which is the
    # quantity the question is about, even when the fit happened in log space.
    slope = np.gradient(risk, grid)
    signed = slope if direction == HIGHER else -slope
    i_max = int(np.nanargmax(signed))
    # Interiority is judged against the patients, not the axis — see
    # BOUNDARY_PERCENTILE.
    pct = float((x < grid[i_max]).mean() * 100.0)
    at_boundary = bool(pct < BOUNDARY_PERCENTILE or pct > 100.0 - BOUNDARY_PERCENTILE)

    return RiskCurve(
        column=column, n=n, events=events,
        auc=metric_auc(x, y, direction),
        grid=grid, risk=risk, risk_lo=risk_lo, risk_hi=risk_hi, slope=slope,
        knots=used_knots, n_knots=int(used_knots.size),
        log_fitted=log_fit, spline_fitted=spline_fitted,
        lr_stat=lr_stat, lr_df=lr_df, lr_p=lr_p,
        steepest_x=float(grid[i_max]),
        steepest_risk=float(risk[i_max]),
        steepest_slope=float(slope[i_max]),
        steepest_at_boundary=at_boundary,
        steepest_pct=pct,
        prevalence=float(y.mean()),
        crossings={
            float(level): risk_crossing(grid, risk, float(level), direction)
            for level in risk_levels
        },
        note="" if spline_fitted else "spline rank-deficient — linear fit used",
    )


def risk_crossing(
    grid: np.ndarray, risk: np.ndarray, level: float, direction: str,
) -> float:
    """Metric value at the outer boundary of the region where risk ≥ ``level``.

    For a metric where high values are suspicious this is the *lowest* value
    still carrying that risk; for one where low values are suspicious it is the
    *highest*. Taking the boundary rather than a first crossing keeps the answer
    meaningful if the fitted curve wobbles.
    """
    grid = np.asarray(grid, dtype=float)
    risk = np.asarray(risk, dtype=float)
    above = risk >= level
    if not above.any() or above.all():
        return np.nan  # never reaches that risk, or never drops below it

    idx = np.flatnonzero(above)
    edge = idx.min() if direction == HIGHER else idx.max()
    neighbour = edge - 1 if direction == HIGHER else edge + 1
    if neighbour < 0 or neighbour >= grid.size:
        return float(grid[edge])
    # Linear interpolation between the last point below and the first above.
    r0, r1 = risk[neighbour], risk[edge]
    if r1 == r0:
        return float(grid[edge])
    frac = (level - r0) / (r1 - r0)
    return float(grid[neighbour] + frac * (grid[edge] - grid[neighbour]))


# --------------------------------------------------------------------------
# Observed proportions — the reality check under the fitted curve
# --------------------------------------------------------------------------
def observed_bins(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 8, min_per_bin: int = 10,
) -> pd.DataFrame:
    """Observed high-grade proportion in equal-count bins, with Wilson intervals.

    These dots are the honesty check on the fitted curve: if they do not track
    it, the spline is telling a story the patients do not support.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.size < n_bins * min_per_bin:
        n_bins = max(2, x.size // max(min_per_bin, 1))
    try:
        bins = pd.qcut(x, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["x", "observed", "lo", "hi", "n", "events"])

    frame = pd.DataFrame({"x": x, "y": y, "bin": bins})
    rows = []
    for _, grp in frame.groupby("bin", observed=True):
        k, n = int(grp["y"].sum()), int(len(grp))
        if n < min_per_bin:
            continue
        lo, hi = ps.wilson_ci(k, n)
        rows.append({
            "x": float(grp["x"].median()),
            "observed": k / n, "lo": float(lo), "hi": float(hi),
            "n": n, "events": k,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Bootstrap for the two numbers worth quoting
# --------------------------------------------------------------------------
def bootstrap_risk_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    direction: str = HIGHER,
    log_fit: bool = False,
    n_knots: int = DEFAULT_N_KNOTS,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    n_boot: int = 500,
    seed: int = 20260801,
) -> dict:
    """Percentile intervals for the steepest-rise point and the risk crossings.

    Refits the whole spline on each resample, so the interval covers knot
    placement and curve shape, not just the position of a maximum on one fixed
    curve. Expect the steepest-point interval to be wide — a maximum of a
    derivative of a fitted curve is about as noisy as an estimate gets, and
    reporting that width is what keeps the number defensible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = x.size

    steepest: list[float] = []
    cross: dict[float, list[float]] = {float(l): [] for l in risk_levels}
    n_ok = 0
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        xb, yb = x[take], y[take]
        if yb.sum() < 5 or (n - yb.sum()) < 5:
            continue
        curve = fit_risk_curve(
            xb, yb, direction=direction, log_fit=log_fit,
            n_knots=n_knots, risk_levels=risk_levels,
        )
        if not curve.spline_fitted:
            continue
        n_ok += 1
        # Gate on the full knee test, not just interiority. A risk curve that is
        # straight on the log-odds scale is still S-shaped as a probability, so
        # its slope always peaks somewhere in the middle — at exactly the 50%
        # crossing, carrying no information that crossing does not already give.
        # Counting those as knees would make every metric look like it had one.
        if curve.knee_found:
            steepest.append(curve.steepest_x)
        for level, value in curve.crossings.items():
            if np.isfinite(value):
                cross[level].append(value)

    def _pi(values: list[float]) -> tuple[float, float]:
        if len(values) < 20:  # too few to quote a percentile interval from
            return np.nan, np.nan
        return (float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)))

    steep_lo, steep_hi = _pi(steepest)
    return {
        "n_boot_ok": n_ok,
        "steepest_lo": steep_lo,
        "steepest_hi": steep_hi,
        # How often a resample produced a knee at all. Low values are the real
        # finding: the "šķēre" is not reproducible in this cohort.
        "knee_rate_boot": (len(steepest) / n_ok) if n_ok else np.nan,
        "crossing_ci": {
            level: _pi(vals) for level, vals in cross.items()
        },
        "crossing_found_frac": {
            level: (len(vals) / n_ok if n_ok else np.nan)
            for level, vals in cross.items()
        },
    }


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def risk_curve_summary(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    n_knots: int = DEFAULT_N_KNOTS,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    n_boot: int = 500,
    seed: int = 20260801,
) -> tuple[pd.DataFrame, dict[str, RiskCurve]]:
    """One row per metric: curvature test, steepest-rise point, risk crossings.

    Returns the table and the fitted curves, so the figures do not refit.
    """
    rows: list[dict] = []
    curves: dict[str, RiskCurve] = {}

    for metric in metrics:
        x, y = metric_arrays(df, metric, target)
        # Primary fit is on the metric's own clinical units, because that is the
        # scale the threshold gets reported in.
        curve = fit_risk_curve(
            x, y, column=metric.col, direction=metric.direction,
            log_fit=False, n_knots=n_knots, risk_levels=risk_levels,
        )
        curves[metric.col] = curve
        # Sensitivity analysis: the same test on log1p(x). Reported next to the
        # primary result so a disagreement is visible rather than a scale
        # quietly chosen to suit the answer.
        log_curve = fit_risk_curve(
            x, y, column=metric.col, direction=metric.direction,
            log_fit=True, n_knots=n_knots, risk_levels=risk_levels,
        )
        boot = bootstrap_risk_curve(
            x, y, direction=metric.direction, log_fit=False,
            n_knots=n_knots, risk_levels=risk_levels, n_boot=n_boot, seed=seed,
        ) if curve.spline_fitted else {}

        row = {
            "metric": metric.label,
            "column": metric.col,
            "unit": metric.unit,
            "direction": metric.direction,
            "n": curve.n,
            "events": curve.events,
            "AUC": curve.auc,
            "fitted_on": "raw x (clinical units)",
            "n_knots": curve.n_knots,
            "nonlinearity_chi2": curve.lr_stat,
            "nonlinearity_df": curve.lr_df,
            "nonlinearity_p": curve.lr_p,
            "nonlinear": curve.nonlinear,
            "steepest_x": curve.steepest_x,
            "steepest_lo": boot.get("steepest_lo", np.nan),
            "steepest_hi": boot.get("steepest_hi", np.nan),
            "steepest_risk": curve.steepest_risk,
            "steepest_slope_per_unit": curve.steepest_slope,
            "steepest_at_boundary": curve.steepest_at_boundary,
            "steepest_pct_of_patients": curve.steepest_pct,
            "knee_rate_boot": boot.get("knee_rate_boot", np.nan),
            # Sensitivity to the fitting scale — see fit_risk_curve's docstring.
            "nonlinearity_p_log_scale": log_curve.lr_p,
            "nonlinear_log_scale": log_curve.nonlinear,
            "scale_sensitive": bool(curve.nonlinear != log_curve.nonlinear),
            "knee_found": curve.knee_found,
            "prevalence": curve.prevalence,
            "n_bootstrap": boot.get("n_boot_ok", 0),
            "verdict": curve.verdict(metric),
            "note": curve.note,
        }
        for level, value in curve.crossings.items():
            tag = f"risk_{int(round(level * 100))}"
            lo, hi = boot.get("crossing_ci", {}).get(level, (np.nan, np.nan))
            row[f"{tag}_x"] = value
            row[f"{tag}_lo"] = lo
            row[f"{tag}_hi"] = hi
            row[f"{tag}_found_frac"] = boot.get(
                "crossing_found_frac", {}
            ).get(level, np.nan)
        rows.append(row)

    return pd.DataFrame(rows), curves


def risk_curve_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """The plain-language version: is there a threshold, and where?"""
    def _val_ci(row, stem: str, digits: str = ".3g") -> str:
        v, lo, hi = row.get(f"{stem}_x"), row.get(f"{stem}_lo"), row.get(f"{stem}_hi")
        if v is None or pd.isna(v):
            return "never reached"
        text = format(v, digits)
        if lo is not None and not pd.isna(lo) and not pd.isna(hi):
            text += f" ({format(lo, digits)}–{format(hi, digits)})"
        return text

    out = pd.DataFrame({
        "Metric": table["metric"],
        "n / events": table["n"].astype(str) + " / " + table["events"].astype(str),
        "Non-linear?": [
            "—" if pd.isna(p) else f"{'yes' if nl else 'no'} (p {ps.format_p(p)})"
            for nl, p in zip(table["nonlinear"], table["nonlinearity_p"])
        ],
        "Steepest rise": [
            _val_ci(r, "steepest") if r["knee_found"] else "no interior threshold"
            for _, r in table.iterrows()
        ],
        # A knee at the 7th percentile is real but rests on few patients; one at
        # the 47th sits in the bulk of the cohort. The reader needs to see which.
        "Patients below it": [
            "" if pd.isna(p) else f"{p:.0f}%"
            for p in table.get("steepest_pct_of_patients",
                               pd.Series(np.nan, index=table.index))
        ],
        "Risk reaches 30%": [_val_ci(r, "risk_30") for _, r in table.iterrows()],
        "Risk reaches 50%": [_val_ci(r, "risk_50") for _, r in table.iterrows()],
        "Reading": table["verdict"],
    })
    return out


# --------------------------------------------------------------------------
# Zero inflation — "no edema at all" is a category, not a small number
# --------------------------------------------------------------------------
# A metric with at least this share of exact zeros is treated as zero-inflated
# and gets the three-way analysis below rather than one whole-cohort spline.
ZERO_INFLATION_FLOOR = 0.10


def zero_share(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    floor: float = ZERO_INFLATION_FLOOR,
) -> pd.DataFrame:
    """How many patients sit at exactly zero, and how their risk differs.

    A spline fitted across a zero-inflated metric puts its 10th-percentile knot
    *at* zero and then reports the resulting bend as a threshold. On this
    cohort a third of patients have no peritumoral edema at all, so a "threshold
    at 3.5 cc" is substantially detecting edema present versus absent — a
    binary distinction a radiologist makes by looking, not by measuring.

    Reported before the curve so the reader can see which claim is which.
    """
    rows: list[dict] = []
    y_all = df[target].astype("boolean")
    for metric in metrics:
        x = pd.to_numeric(df[metric.col], errors="coerce")
        keep = x.notna() & y_all.notna()
        x, y = x[keep], y_all[keep].astype(bool)
        n = int(len(x))
        if n == 0:
            continue
        at_zero = x == 0
        n_zero = int(at_zero.sum())

        def _risk(mask: pd.Series) -> tuple[float, float, float, int, int]:
            k, m = int(y[mask].sum()), int(mask.sum())
            if m == 0:
                return np.nan, np.nan, np.nan, 0, 0
            lo, hi = ps.wilson_ci(k, m)
            return k / m, float(lo), float(hi), k, m

        r0, lo0, hi0, k0, m0 = _risk(at_zero)
        r1, lo1, hi1, k1, m1 = _risk(~at_zero)
        rows.append({
            "metric": metric.label,
            "column": metric.col,
            "n_analysed": n,
            "n_zero": n_zero,
            # Percent, not a fraction: the shared CSV formatter rounds a column
            # named pct_* on a 0–100 scale, and 36.6% would export as "0.4".
            "pct_zero": (100.0 * n_zero / n) if n else np.nan,
            "n_positive": m1,
            "smallest_positive": float(x[x > 0].min()) if (x > 0).any() else np.nan,
            "events_zero": k0, "risk_zero": r0,
            "risk_zero_lo": lo0, "risk_zero_hi": hi0,
            "events_positive": k1, "risk_positive": r1,
            "risk_positive_lo": lo1, "risk_positive_hi": hi1,
            "risk_ratio": (r1 / r0) if (np.isfinite(r0) and r0 > 0) else np.nan,
            "zero_inflated": bool(n_zero / n >= floor) if n else False,
            "zero_inflation_floor_pct": 100.0 * floor,
        })
    return pd.DataFrame(rows)


def presence_rules(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
) -> pd.DataFrame:
    """"Is there any at all?" scored as a yes/no test, with the full 2×2.

    The comparator the 3.5 cc cut-point has to beat. If a rule that needs no
    measurement at all performs as well, the measured cut-point is not what is
    doing the work.
    """
    from diagnostic_accuracy import binary_diagnostic_metrics

    rows: list[dict] = []
    for metric in metrics:
        x = pd.to_numeric(df[metric.col], errors="coerce")
        present = pd.Series(pd.NA, index=df.index, dtype="boolean")
        present[x.notna()] = (x[x.notna()] > 0)
        # A metric nobody has a zero of flags everyone, which is not a test:
        # the 2x2 has an empty column and the chi-square is undefined.
        if present.dropna().nunique() < 2:
            continue
        label = f"{metric.label} present (> 0)"
        row = binary_diagnostic_metrics(df, target, label, predictor_series=present)
        row.update({
            "metric": metric.label,
            "column": metric.col,
            "rule": "presence",
            "rule_label": label,
            "youden_J": row["sensitivity"] + row["specificity"] - 1.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def positive_only(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """The patients with a non-zero value — where the metric is a measurement."""
    x = pd.to_numeric(df[metric.col], errors="coerce")
    return df.loc[x.notna() & (x > 0)].copy()


def zero_inflation_comparison(
    whole: pd.DataFrame,
    presence: pd.DataFrame,
    positive: pd.DataFrame,
    metric: Metric,
) -> pd.DataFrame:
    """The three fits side by side, so the reader can pick the defensible claim.

    ``whole`` and ``positive`` are ``risk_curve_summary`` tables; ``presence``
    is a :func:`presence_rules` table. One row each, same columns, so the
    report can print them as one block.
    """
    def _curve_row(table: pd.DataFrame, what: str) -> dict:
        sub = table[table["column"] == metric.col]
        if sub.empty:
            return {"Fitted on": what, "n": "—", "Non-linearity p": "—",
                    "Steepest rise": "—", "95% CI": "—"}
        r = sub.iloc[0]
        knee = bool(r.get("knee_found"))
        return {
            "Fitted on": what,
            "n": f"{int(r['n'])} ({int(r['events'])} high grade)",
            "Non-linearity p": ps.format_p(r["nonlinearity_p"]),
            "Steepest rise": f"{r['steepest_x']:.3g}" if knee else "no interior threshold",
            "95% CI": (f"{r['steepest_lo']:.3g}–{r['steepest_hi']:.3g}"
                       if knee and pd.notna(r.get("steepest_lo")) else "—"),
        }

    rows = [_curve_row(whole, "Whole cohort (zeros included)")]

    pres = presence[presence["column"] == metric.col]
    if not pres.empty:
        p = pres.iloc[0]
        rows.append({
            "Fitted on": "Present vs absent (no measurement needed)",
            "n": f"{int(p['n_used'])} ({int(p['TP'] + p['FN'])} high grade)",
            "Non-linearity p": "n/a — binary",
            "Steepest rise": (f"sens {p['sensitivity'] * 100:.0f}% / "
                              f"spec {p['specificity'] * 100:.0f}%, J {p['youden_J']:.2f}"),
            "95% CI": "—",
        })

    rows.append(_curve_row(positive, "Only patients with a non-zero value"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style_metric_axis(ax: plt.Axes, metric: Metric, values: np.ndarray) -> None:
    """Log axis only when every plotted value is positive.

    Edema volume is legitimately zero for a quarter of this cohort, and a log
    axis would silently drop that quarter off the left of the figure — the most
    important part of the curve for a metric whose story is "no edema at all".
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if metric.log_x and finite.size and finite.min() > 0:
        ax.set_xscale("log")
    ax.set_xlabel(metric.axis_label)


def risk_curve_figure(
    df: pd.DataFrame,
    metric: Metric,
    target: str,
    curve: RiskCurve,
    *,
    boot: dict | None = None,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
) -> plt.Figure:
    """Left: risk against the metric. Right: the slope of that risk — the "šķēre".

    The right panel is the direct answer to "where does risk increase most":
    its peak *is* that point. A right panel with no peak — flat, or highest at
    an edge — is the finding that no such point exists.
    """
    x, y = metric_arrays(df, metric, target)
    boot = boot or {}

    fig, (ax_risk, ax_slope) = plt.subplots(
        1, 2, figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.46),
    )

    # ---- Left: the risk curve -------------------------------------------
    if curve.grid.size:
        ax_risk.fill_between(curve.grid, curve.risk_lo, curve.risk_hi,
                             color=ps.PALETTE["primary"], alpha=0.15, zorder=1,
                             label="95% CI")
        ax_risk.plot(curve.grid, curve.risk, color=ps.PALETTE["primary"],
                     linewidth=1.8, zorder=3, label="Fitted risk (spline)")

    obs = observed_bins(x, y)
    if len(obs):
        ax_risk.errorbar(
            obs["x"], obs["observed"],
            yerr=ps.errorbar_lengths(obs["observed"], obs["lo"], obs["hi"]),
            fmt="o", markersize=3.4, color=ps.PALETTE["neutral"],
            ecolor=ps.PALETTE["neutral"], elinewidth=0.8, capsize=1.8,
            linestyle="none", zorder=4, label="Observed (equal-count bins)",
        )

    ax_risk.axhline(curve.prevalence, color=ps.PALETTE["neutral"], linewidth=0.8,
                    linestyle="-.", zorder=1,
                    label=f"Cohort rate ({curve.prevalence * 100:.0f}%)")

    for level, color in zip(risk_levels, (ps.PALETTE["good"], ps.PALETTE["accent"])):
        cross = curve.crossings.get(float(level), np.nan)
        if not np.isfinite(cross):
            continue
        ax_risk.axhline(level, color=color, linewidth=0.7, linestyle=":", zorder=1)
        ax_risk.scatter([cross], [level], s=24, color=color, zorder=6,
                        label=f"Risk {level * 100:.0f}% at {cross:.3g}")

    ax_risk.set_ylim(0, 1)
    ax_risk.set_ylabel("Probability of high grade")
    _style_metric_axis(ax_risk, metric, curve.grid)
    ps.place_legend(ax_risk, loc="best", scale=0.62)
    ps.set_titles(ax_risk, "Risk against the metric",
                  ps.n_subtitle(curve.n, extra=f"{curve.events} high grade"))

    # ---- Right: the slope ------------------------------------------------
    if curve.grid.size:
        signed = curve.slope if metric.direction == HIGHER else -curve.slope
        ax_slope.plot(curve.grid, signed, color=ps.PALETTE["accent"],
                      linewidth=1.6, zorder=3, label="Rise in risk per unit")
        ax_slope.axhline(0.0, color=ps.PALETTE["neutral"], linewidth=0.8, zorder=1)

        lo, hi = boot.get("steepest_lo", np.nan), boot.get("steepest_hi", np.nan)
        if curve.knee_found and np.isfinite(lo) and np.isfinite(hi):
            ax_slope.axvspan(lo, hi, color=ps.PALETTE["accent"], alpha=0.12,
                             zorder=0, label="Steepest point 95% boot")
        if curve.knee_found:
            ax_slope.axvline(curve.steepest_x, color=ps.PALETTE["accent"],
                             linewidth=1.3, zorder=4,
                             label=f"Steepest at {curve.steepest_x:.3g} "
                                   f"({curve.steepest_pct:.0f}% below)")
        else:
            ax_slope.axvline(curve.steepest_x, color=ps.PALETTE["neutral"],
                             linewidth=1.0, linestyle="--", zorder=4,
                             label="Max slope (not a threshold)")

    _style_metric_axis(ax_slope, metric, curve.grid)
    ax_slope.set_ylabel("d(risk) / d(metric)")
    ps.place_legend(ax_slope, loc="best", scale=0.62)
    ps.set_titles(ax_slope, "Where risk climbs fastest",
                  f"Non-linearity p {ps.format_p(curve.lr_p)}")

    fig.suptitle(curve.verdict(metric),
                 fontsize=plt.rcParams["font.size"] * 0.95)
    return fig


def risk_curve_panel(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    curves: dict[str, RiskCurve],
) -> plt.Figure:
    """All risk curves as small multiples — the one figure for a poster.

    No cut-points, no rules: just how the probability of high grade moves with
    each measurement, and how wide the uncertainty around that is.
    """
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.30 * nrows + 0.06),
    )

    for ax, metric in zip(axes.ravel(), metrics):
        curve = curves.get(metric.col)
        x, y = metric_arrays(df, metric, target)
        if curve is not None and curve.grid.size:
            ax.fill_between(curve.grid, curve.risk_lo, curve.risk_hi,
                            color=ps.PALETTE["primary"], alpha=0.15, zorder=1)
            ax.plot(curve.grid, curve.risk, color=ps.PALETTE["primary"],
                    linewidth=1.7, zorder=3)
            ax.axhline(curve.prevalence, color=ps.PALETTE["neutral"],
                       linewidth=0.7, linestyle="-.", zorder=1)
            if curve.knee_found:
                ax.axvline(curve.steepest_x, color=ps.PALETTE["accent"],
                           linewidth=1.1, zorder=4)
        obs = observed_bins(x, y)
        if len(obs):
            ax.errorbar(
                obs["x"], obs["observed"],
                yerr=ps.errorbar_lengths(obs["observed"], obs["lo"], obs["hi"]),
                fmt="o", markersize=3.0, color=ps.PALETTE["neutral"],
                ecolor=ps.PALETTE["neutral"], elinewidth=0.7, capsize=1.5,
                linestyle="none", zorder=4,
            )
        ax.set_ylim(0, 1)
        _style_metric_axis(ax, metric, curve.grid if curve is not None else x)
        ax.set_ylabel("P(high grade)")
        # The percentile is on the subtitle because a threshold on a skewed
        # metric looks pinned to the left spine even when half the cohort sits
        # below it — edema volume's knee is at 3.5 cc of a 0–116 cc axis.
        subtitle = (
            f"steepest {curve.steepest_x:.3g} "
            f"({curve.steepest_pct:.0f}% of patients below) · p {ps.format_p(curve.lr_p)}"
            if curve is not None and curve.knee_found
            else (f"no threshold · p {ps.format_p(curve.lr_p)}"
                  if curve is not None else "")
        )
        ps.set_titles(ax, metric.label, subtitle)

    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()

    fig.suptitle("Risk of high-grade meningioma across the observed range",
                 fontsize=plt.rcParams["font.size"] * 1.05)
    return fig
