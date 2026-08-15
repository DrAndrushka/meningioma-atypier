"""The main figure — three panels that carry the argument in order.

**A. Where each measurement stands.** ROC curves for all five, with the
published cut-point marked on each. Answers "how well does any of this
separate the grades?" and the honest answer is: modestly.

**B. Why the cut-point is hard to pin down.** Youden J against the cut-point's
position in the cohort, so five different units share one axis. A sharp peak
means the data pick out one value; a broad plateau means hundreds of candidate
values score almost identically, and the winner won by a rounding error. This
panel is the reason panel C looks the way it does.

**C. What that costs.** Each cut-point with its bootstrap interval, drawn on the
same percentile axis so five measurements are comparable, and annotated with the
value in its own units. A bar spanning half the axis is not a threshold.

A second figure, :func:`decision_figure`, asks the question none of these three
do: whether following the rule would lead to better decisions than treating
everyone or no one as high grade. That is the only one of the questions here a
clinician can act on, and it lives in its own figure because its horizontal axis
is a value judgment rather than a measurement.

Read left to right, the three panels say: there is a signal, the optimum is
flat, and so the number is uncertain. Any one of them alone would overstate or
understate the case.

The percentile axis in B and C is what makes five measurements comparable at
all. Plotting cm³ against ADC units on one axis is meaningless; plotting "the
value below which 40% of patients fall" is not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import ajnr_style as aj
from criteria import sweep
from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import oriented_score

OUTCOME = "high_grade"


def to_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Where each value sits in the cohort, as a percentage of patients below it."""
    reference = np.asarray(reference, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if reference.size == 0:
        return np.full(np.shape(values), np.nan)
    return 100.0 * np.searchsorted(reference, values, side="left") / reference.size


def _series(df: pd.DataFrame, row: pd.Series) -> tuple[Measurement, np.ndarray,
                                                       np.ndarray]:
    m = MEASUREMENTS_BY_COL[row["col"]]
    sub = df.loc[stratum_mask(df, m, row["stratum"])]
    x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
    y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    return m, x[ok], y[ok].astype(int)


def panel_roc(ax, df: pd.DataFrame, eligible: pd.DataFrame,
              cutpoints: dict[str, float]) -> None:
    """ROC curves for every eligible measurement, cut-point marked."""
    from sklearn.metrics import roc_curve

    ax.plot([0, 1], [0, 1], color=aj.REFERENCE, ls=":", lw=0.9, zorder=0)
    for i, (_, row) in enumerate(eligible.iterrows()):
        m, x, y = _series(df, row)
        if not x.size:
            continue
        fpr, tpr, _ = roc_curve(y, oriented_score(x, m.direction))
        # The name alone. The shared legend keys all three panels, and an AUC
        # sitting in it would look like it applied to B and C as well. AUCs with
        # their intervals belong in the table, where they have room.
        ax.plot(fpr, tpr, **aj.series_style(i, lw=1.3), label=m.short_label,
                zorder=2)

        if m.col in cutpoints:
            from accuracy import accuracy_at
            perf = accuracy_at(y, x, cutpoints[m.col], m.direction)
            ax.plot([1 - perf["specificity"]], [perf["sensitivity"]],
                    marker=aj.MARKER, ms=4.0, color=aj.SHADES[i % len(aj.SHADES)],
                    markeredgecolor=aj.INK, markeredgewidth=0.5, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    # No equal aspect: it shrinks the axes to a square inside a wider slot.
    # No legend here either — see main_figure. A key placed inside these axes
    # either sits on top of five curves or hides them behind an opaque patch,
    # and both cost the reader more than the space saved.
    ax.set_title("A", loc="left", fontweight="bold", pad=6)


def panel_youden(ax, df: pd.DataFrame, eligible: pd.DataFrame,
                 cutpoints: dict[str, float]) -> None:
    """Youden J across every candidate cut-point, on a shared percentile axis."""
    for i, (_, row) in enumerate(eligible.iterrows()):
        m, x, y = _series(df, row)
        if not x.size:
            continue
        table = sweep(y, x, m.direction)
        if table.empty:
            continue
        pct = to_percentile(table["cutoff"].to_numpy(), x)
        ax.plot(pct, table["youden_j"].to_numpy(),
                **aj.series_style(i, lw=1.1), zorder=2)
        best = table["youden_j"].idxmax()
        ax.plot([pct[table.index.get_loc(best)]], [table.loc[best, "youden_j"]],
                marker=aj.MARKER, ms=3.4,
                color=aj.SHADES[i % len(aj.SHADES)],
                markeredgecolor=aj.INK, markeredgewidth=0.4, zorder=3)

    ax.axhline(0.0, color=aj.REFERENCE, ls=":", lw=0.9, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of cohort")
    ax.set_ylabel("Youden J")
    ax.set_title("B", loc="left", fontweight="bold", pad=6)


def panel_forest(ax, df: pd.DataFrame, wobble: pd.DataFrame) -> None:
    """Each cut-point with its bootstrap interval, on the percentile axis."""
    rows = list(wobble.iterrows())[::-1]      # first row at the top
    for slot, (_, row) in enumerate(rows):
        m = MEASUREMENTS_BY_COL[row["col"]]
        reference = pd.to_numeric(
            df.loc[stratum_mask(df, m, row["stratum"]), m.col],
            errors="coerce").to_numpy()
        point, lo, hi = (to_percentile(np.array([row[k]]), reference)[0]
                         for k in ("cutpoint", "ci_lo", "ci_hi"))
        wide = row.get("stability_ratio", np.nan) > 1.0
        colour = aj.MUTED if wide else aj.INK
        if slot % 2 == 0:
            ax.axhspan(slot - 0.5, slot + 0.5, color=aj.ROW_BAND,
                       alpha=aj.ROW_BAND_ALPHA, lw=0, zorder=0)
        ax.plot([lo, hi], [slot, slot], color=colour, lw=aj.CI_LINEWIDTH,
                solid_capstyle="butt", zorder=2)
        ax.plot([point], [slot], marker=aj.MARKER, ms=aj.MARKER_SIZE,
                color=colour, zorder=3)
        # axis_unit, not unit: Arial has no superscript-minus, so the plain
        # ADC unit would render as an empty box.
        ax.text(104, slot, f"{m.op} {row['cutpoint']:g} {m.axis_unit}".strip(),
                va="center", ha="left", fontsize=8.0, color=colour)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MEASUREMENTS_BY_COL[r["col"]].short_label
                        for _, r in rows], fontsize=8.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of cohort")
    ax.set_title("C", loc="left", fontweight="bold", pad=6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def verdict_thumbnail(df: pd.DataFrame, m: Measurement, fits: dict,
                      segmented_row=None, *, width_in: float = 2.6,
                      height_in: float = 1.0) -> str:
    """A two-panel sparkline for a dashboard card, as a base64 PNG.

    Left, risk against the measurement in its own units; right, the same fit
    against the log scale. Two panels because the commonest failure in this
    cohort is a bend that exists on one scale and not the other, and no single
    curve can show that — the reader has to see the same patients twice.

    Where a breakpoint was estimated, its confidence interval is shaded. A band
    covering most of the panel is the point: the model always returns a
    breakpoint, and the width is what reveals it located nothing.

    Deliberately unlabelled. At an inch high, axis numbers are unreadable and
    the shape is the whole message; the numbers are in the card beside it.
    """
    import io

    import matplotlib.pyplot as plt

    aj.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))
    for ax, log_fit in zip(axes, (False, True)):
        fit = fits.get((m.col, "all", log_fit))
        if fit is None or fit.grid.size == 0:
            ax.set_axis_off()
            continue
        ax.plot(fit.grid, fit.risk, color=aj.INK, lw=1.2)
        if log_fit:
            ax.set_xscale("log")
        elif segmented_row is not None and np.isfinite(
                segmented_row.get("ci_lo", np.nan)):
            ax.axvspan(segmented_row["ci_lo"], segmented_row["ci_hi"],
                       color=aj.INK, alpha=0.10, lw=0)
            if np.isfinite(segmented_row.get("breakpoint", np.nan)):
                ax.axvline(segmented_row["breakpoint"], color=aj.INK,
                           lw=0.9, ls="--")
        lo, hi = float(fit.grid.min()), float(fit.grid.max())
        if log_fit:
            # Edema is zero for a third of this cohort, so the grid starts at 0
            # and a log axis cannot take it. Start at the smallest positive
            # value instead of letting matplotlib ignore the limit and pick its
            # own, which would silently show a different range from the panel
            # beside it.
            positive = fit.grid[fit.grid > 0]
            lo = float(positive.min()) if positive.size else hi / 100.0
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, max(0.8, float(np.nanmax(fit.risk)) * 1.15))
        # Both major and minor: a log axis adds labelled minor ticks of its own,
        # and clearing only the major ones leaves stray digits under the curve.
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(side == "bottom")
        ax.spines["bottom"].set_linewidth(0.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10,
                        wspace=0.14)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, facecolor="white")
    plt.close(fig)
    import base64
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def panel_decision(ax, curves: dict, order: Sequence[str]) -> None:
    """Net benefit of each yes/no rule against treating everyone and no one."""
    treat_all, thresholds = None, None
    for i, col in enumerate(order):
        entry = curves.get(col)
        if entry is None:
            continue
        thresholds = entry["thresholds"]
        treat_all = entry["treat_all"] if treat_all is None else treat_all
        m = MEASUREMENTS_BY_COL[col]
        ax.plot(thresholds, entry["rule_corrected"],
                **aj.series_style(i, lw=1.3), label=m.short_label, zorder=2)

    if thresholds is None:
        ax.set_axis_off()
        return
    # Treat-all differs by a fraction of a percent between measurements because
    # each is analysed on its own complete cases. One line is drawn, from the
    # largest set: five overlapping grey lines would read as a band that means
    # something, and it does not.
    widest = max((c for c in curves.values() if c is not None),
                 key=lambda c: c["n"])
    ax.plot(thresholds, widest["treat_all"], color=aj.REFERENCE, lw=1.0,
            ls=(0, (3, 2)), zorder=1, label="Treat all")
    # Labelled, though it is only the zero line: a reader who does not already
    # know decision curves cannot be expected to infer that the axis itself is
    # one of the two strategies every other line has to beat.
    ax.axhline(0.0, color=aj.REFERENCE, ls=":", lw=0.9, zorder=0,
               label="Treat none")

    finite = np.concatenate([c["rule_corrected"][np.isfinite(c["rule_corrected"])]
                             for c in curves.values() if c is not None])
    top = float(np.nanmax(finite)) if finite.size else 0.1
    ax.set_xlim(float(thresholds.min()), float(thresholds.max()))
    ax.set_ylim(-0.05, max(0.12, top * 1.25))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("A", loc="left", fontweight="bold", pad=6)


def panel_decision_difference(ax, curves: dict, order: Sequence[str]) -> None:
    """Number minus yes/no rule: above zero, dichotomising costs a decision.

    Drawn as a difference rather than as ten lines on one pair of axes. The
    continuous and dichotomised curves for one measurement sit within a few
    thousandths of each other over most of the range, and the eye cannot resolve
    that gap on an axis wide enough to hold five measurements — but the gap is
    the entire question this panel exists to answer.

    Each line is drawn only across the span where one of the two forms is useful
    at all — ``decidable``, computed in :mod:`decision_curve` so the figure and
    Table S4 cannot disagree about it — and that restriction is not cosmetic. At
    a threshold of 5% the fitted model assigns every patient a risk above it, so
    "use the number" *is* treat-all wearing a regression; at 55% it assigns
    almost nobody one, so it is treat-none. The difference at either end is
    large, real, and about nothing: it compares the cut-point against a strategy
    that needs no measurement at all. Plotting those thresholds would read as
    "the number wins everywhere", which is the opposite of what is happening.
    """
    thresholds = None
    for i, col in enumerate(order):
        entry = curves.get(col)
        if entry is None:
            continue
        thresholds = entry["thresholds"]
        gap = (np.asarray(entry["model_corrected"], dtype=float)
               - np.asarray(entry["rule_corrected"], dtype=float))
        decidable = entry.get("decidable")
        if decidable is not None:
            gap = np.where(np.asarray(decidable, dtype=bool), gap, np.nan)
        ax.plot(thresholds, gap, **aj.series_style(i, lw=1.3), zorder=2)

    if thresholds is None:
        ax.set_axis_off()
        return
    ax.axhline(0.0, **aj.NULL_LINE, zorder=1)
    ax.set_xlim(float(thresholds.min()), float(thresholds.max()))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit, number − rule")
    ax.set_title("B", loc="left", fontweight="bold", pad=6)


def decision_figure(curves: dict, order: Sequence[str], *,
                    height_in: float = 3.0):
    """The two-panel decision-curve figure. Caller saves and closes it."""
    import matplotlib.pyplot as plt

    aj.apply_style()
    fig = plt.figure(figsize=(aj.WIDTH_IN, height_in))
    grid = fig.add_gridspec(1, 2, wspace=0.30, left=0.088, right=0.98,
                            bottom=0.30, top=0.92)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    panel_decision(axes[0], curves, order)
    panel_decision_difference(axes[1], curves, order)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=8.0,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.4)
    return fig, axes


def main_figure(df: pd.DataFrame, eligible: pd.DataFrame,
                separation: pd.DataFrame, wobble: pd.DataFrame,
                cutpoints: dict[str, float], *, height_in: float = 3.3):
    """The three-panel main figure. Caller saves and closes it.

    Panel C is given the widest slot: it carries a right-hand annotation column
    in native units that sits outside its axes, and the remaining panels are
    plots that shrink gracefully.

    One legend for the whole figure, along the bottom. All three panels use the
    same five series with the same shade and dash, so a key inside any one of
    them would be both redundant and in the way — a legend placed over a plot
    either covers the curves or hides them behind an opaque patch.
    """
    import matplotlib.pyplot as plt

    aj.apply_style()
    fig = plt.figure(figsize=(aj.WIDTH_IN, height_in))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.15),
                            wspace=0.55, left=0.075, right=0.80,
                            bottom=0.26, top=0.92)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    panel_roc(axes[0], df, eligible, cutpoints)
    panel_youden(axes[1], df, eligible, cutpoints)
    panel_forest(axes[2], df, wobble)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=8.0,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.5)
    return fig, axes
