"""Does the answer survive the missing data?

Everything in :mod:`thresholds`, :mod:`risk_curves` and :mod:`combinations` is
complete-case: each analysis quietly drops the patients missing its own inputs.
If those patients differ from the ones who kept their measurement, the answer is
shifted, and **no amount of bootstrapping will reveal it** — the bootstrap
resamples the patients who are already there.

So every headline number is re-derived on each of the m MICE draws. Two
different uncertainties then sit side by side:

============  ==========================  ==========================================
Source        Measured by                 Question it answers
============  ==========================  ==========================================
Sampling      bootstrap CI                would a different 350 patients change it?
Missing data  spread across the m draws   would knowing the missing values change it?
============  ==========================  ==========================================

⚠️ This is a **stability check, not Rubin pooling**. Rubin's rules combine
estimates with an approximately normal sampling distribution and a within-
imputation variance; a cut-point picked by maximising J over hundreds of
candidates has neither. The spread across draws is between-imputation variance
only — a legitimate statement about how much the missing data matter, and not a
confidence interval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from missingness_resolution import load_modeling_frames
import plot_style as ps
from combinations import (
    CutPoint,
    count_score_table,
    count_threshold_table,
    flag_frame,
)
from risk_curves import DEFAULT_N_KNOTS, DEFAULT_RISK_LEVELS, fit_risk_curve
from thresholds import (
    Metric,
    RULES,
    metric_arrays,
    metric_auc,
    operating_point,
    roc_table,
    select_cutoff,
)


def load_imputed_draws(
    output_root: Path,
    *,
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> list[pd.DataFrame]:
    """The m MICE draws, with any notebook-local derived columns rebuilt per draw.

    ``derive`` runs inside each draw rather than once on the pooled data. For a
    ratio like the edema index that is the only correct order — imputing the
    ratio directly would not respect its two parents, and computing it once
    from averaged parents would erase the between-draw variation this module
    exists to measure.
    """
    frames = []
    for frame in load_modeling_frames(output_root):
        frame = frame.copy()
        if derive is not None:
            frame = derive(frame)
        frames.append(frame)
    return frames


# --------------------------------------------------------------------------
# Cut-points across draws
# --------------------------------------------------------------------------
def draw_cutoffs(
    frames: Sequence[pd.DataFrame],
    metrics: Sequence[Metric],
    target: str,
    *,
    rules: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per draw × metric × rule: the cut-point that draw would choose."""
    rules = list(RULES) if rules is None else list(rules)
    rows = []
    for draw, frame in enumerate(frames, start=1):
        for metric in metrics:
            x, y = metric_arrays(frame, metric, target)
            tab = roc_table(x, y, metric.direction)
            auc = metric_auc(x, y, metric.direction)
            for rule in rules:
                idx = select_cutoff(tab, rule)
                if idx is None:
                    continue
                cutoff = float(tab.loc[idx, "cutoff"])
                sens, spec = operating_point(x, y, metric.direction, cutoff)
                rows.append({
                    "draw": draw, "metric": metric.label, "column": metric.col,
                    "rule": rule, "cutoff": cutoff,
                    "sensitivity": sens, "specificity": spec,
                    "youden_J": sens + spec - 1.0, "AUC": auc, "n": len(y),
                })
    return pd.DataFrame(rows)


def imputation_stability(
    draws: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    metrics: Sequence[Metric],
    target: str,
    complete_case: pd.DataFrame,
) -> pd.DataFrame:
    """Across-draw summary per metric × rule, next to the complete-case cut-point.

    ``sens_at_mean``/``spec_at_mean`` apply the **single averaged cut-point**
    back to every draw. That is what the cut-point would actually do in
    practice, and it is always a little worse than the per-draw optimum, each
    of which was chosen with hindsight on its own draw.
    """
    by_col = {m.col: m for m in metrics}
    # Only the selection rules can be compared with a per-draw cut-point, and
    # only they are unique per (column, rule). A measurement can carry several
    # *published* cut-points — three for max diameter — which share the key
    # "literature", and looking that up returns a frame where a value is
    # expected. They are dropped here rather than silently deduplicated: a
    # published cut-point is not re-derived inside a draw, so there is nothing
    # for this table to say about it.
    cc = None
    if len(complete_case):
        selection_rules = set(draws["rule"].unique()) if "rule" in draws else set(RULES)
        cc = (complete_case[complete_case["rule"].isin(selection_rules)]
              .set_index(["column", "rule"]).sort_index())

    rows = []
    for (col, rule), grp in draws.groupby(["column", "rule"], sort=False):
        metric = by_col.get(col)
        if metric is None:
            continue
        cuts = grp["cutoff"].to_numpy(dtype=float)
        mean_cut = float(np.mean(cuts))

        sens_at, spec_at = [], []
        for frame in frames:
            x, y = metric_arrays(frame, metric, target)
            s, p = operating_point(x, y, metric.direction, mean_cut)
            sens_at.append(s)
            spec_at.append(p)

        row = {
            "metric": metric.label, "column": col, "rule": rule,
            "operator": metric.op, "m_draws": len(cuts),
            "cutoff_mean": mean_cut,
            # A flat J plateau makes the argmax hop between local maxima, so the
            # draws can be bimodal and the mean lands in the empty middle.
            "cutoff_median": float(np.median(cuts)),
            "cutoff_sd": float(np.std(cuts, ddof=1)) if len(cuts) > 1 else np.nan,
            "cutoff_min": float(np.min(cuts)),
            "cutoff_max": float(np.max(cuts)),
            "cutoff_cv": (float(np.std(cuts, ddof=1) / mean_cut)
                          if len(cuts) > 1 and mean_cut else np.nan),
            "sens_mean": float(grp["sensitivity"].mean()),
            "spec_mean": float(grp["specificity"].mean()),
            "J_mean": float(grp["youden_J"].mean()),
            "AUC_mean": float(grp["AUC"].mean()),
            "sens_at_mean": float(np.nanmean(sens_at)) if sens_at else np.nan,
            "spec_at_mean": float(np.nanmean(spec_at)) if spec_at else np.nan,
        }
        if cc is not None and (col, rule) in cc.index:
            # List-of-one key: always a frame, so .iloc[0] is a row whatever the
            # index does. Plain .loc[(col, rule)] returns a Series or a frame
            # depending on how many rows match, and the frame reaches float().
            cc_row = cc.loc[[(col, rule)]].iloc[0]
            row["cutoff_complete_case"] = float(cc_row["cutoff"])
            row["cutoff_boot_lo"] = float(cc_row.get("cutoff_boot_lo", np.nan))
            row["cutoff_boot_hi"] = float(cc_row.get("cutoff_boot_hi", np.nan))
            row["n_complete_case"] = cc_row.get("n_used", np.nan)
        rows.append(row)

    out = pd.DataFrame(rows)
    if "cutoff_complete_case" in out.columns:
        out["shift_vs_complete_case"] = out["cutoff_mean"] - out["cutoff_complete_case"]
    return out.sort_values(["column", "rule"]).reset_index(drop=True)


def stability_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """Complete-case vs imputation-averaged cut-point, with both spreads."""
    def _fmt(op, value):
        return "" if pd.isna(value) else f"{op}{value:.3g}"

    m_draws = int(table["m_draws"].max()) if len(table) else 0
    return pd.DataFrame({
        "Metric": table["metric"],
        "Rule": table["rule"],
        "Complete-case": [
            _fmt(op, c) for op, c in
            zip(table["operator"], table.get("cutoff_complete_case", np.nan))
        ],
        "Bootstrap 95%": [
            "" if pd.isna(lo) or pd.isna(hi) else f"{lo:.3g}–{hi:.3g}"
            for lo, hi in zip(table.get("cutoff_boot_lo", np.nan),
                              table.get("cutoff_boot_hi", np.nan))
        ],
        f"MICE mean (m={m_draws})": [
            _fmt(op, c) for op, c in zip(table["operator"], table["cutoff_mean"])
        ],
        "MICE median": [
            _fmt(op, c) for op, c in zip(table["operator"], table["cutoff_median"])
        ],
        "MICE SD": table["cutoff_sd"].map(lambda v: "" if pd.isna(v) else f"{v:.3g}"),
        "MICE range": [
            f"{lo:.3g}–{hi:.3g}"
            for lo, hi in zip(table["cutoff_min"], table["cutoff_max"])
        ],
        "Sens @ mean cut": (table["sens_at_mean"] * 100).round(0).map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}%"),
        "Spec @ mean cut": (table["spec_at_mean"] * 100).round(0).map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}%"),
    })


# --------------------------------------------------------------------------
# Risk curves across draws
# --------------------------------------------------------------------------
def draw_risk_curves(
    frames: Sequence[pd.DataFrame],
    metrics: Sequence[Metric],
    target: str,
    *,
    n_knots: int = DEFAULT_N_KNOTS,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
) -> pd.DataFrame:
    """Refit the risk curve inside every draw — one row per draw × metric."""
    rows = []
    for draw, frame in enumerate(frames, start=1):
        for metric in metrics:
            x, y = metric_arrays(frame, metric, target)
            curve = fit_risk_curve(
                x, y, column=metric.col, direction=metric.direction,
                log_fit=False,  # clinical units, matching risk_curve_summary
                n_knots=n_knots, risk_levels=risk_levels,
            )
            row = {
                "draw": draw, "metric": metric.label, "column": metric.col,
                "n": curve.n, "AUC": curve.auc,
                "nonlinearity_p": curve.lr_p,
                "nonlinear": curve.nonlinear,
                "steepest_x": curve.steepest_x,
                "knee_found": curve.knee_found,
            }
            for level, value in curve.crossings.items():
                row[f"risk_{int(round(level * 100))}_x"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def risk_curve_stability(draws: pd.DataFrame) -> pd.DataFrame:
    """How often is the "šķēre" even there, and where does it move to?

    ``knee_rate`` is the headline: the fraction of imputed datasets in which a
    non-linear risk curve with an interior steepest point was found at all.
    A rate near 1 means the threshold is a property of the data; a rate near 0
    means it was a property of which patients happened to have that scan.
    """
    rows = []
    cross_cols = [c for c in draws.columns
                  if c.startswith("risk_") and c.endswith("_x")]
    for col, grp in draws.groupby("column", sort=False):
        knees = grp.loc[grp["knee_found"], "steepest_x"].to_numpy(dtype=float)
        row = {
            "metric": grp["metric"].iloc[0],
            "column": col,
            "m_draws": len(grp),
            "nonlinear_rate": float(grp["nonlinear"].mean()),
            "knee_rate": float(grp["knee_found"].mean()),
            "median_nonlinearity_p": float(grp["nonlinearity_p"].median()),
            "AUC_mean": float(grp["AUC"].mean()),
            "steepest_median": float(np.median(knees)) if knees.size else np.nan,
            "steepest_min": float(np.min(knees)) if knees.size else np.nan,
            "steepest_max": float(np.max(knees)) if knees.size else np.nan,
        }
        for cc in cross_cols:
            vals = grp[cc].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            stem = cc[:-2]
            row[f"{stem}_median"] = float(np.median(vals)) if vals.size else np.nan
            row[f"{stem}_min"] = float(np.min(vals)) if vals.size else np.nan
            row[f"{stem}_max"] = float(np.max(vals)) if vals.size else np.nan
            row[f"{stem}_reached_rate"] = float(vals.size / len(grp))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Combination rules across draws
# --------------------------------------------------------------------------
def draw_count_scores(
    frames: Sequence[pd.DataFrame],
    cutpoints: Sequence[CutPoint],
    target: str,
) -> pd.DataFrame:
    """Count-score risk table averaged over the draws — every patient included.

    The complete-case version drops anyone missing a single metric, which on
    this cohort is most of the missingness put together. Inside a draw nothing
    is missing, so every patient gets a count.
    """
    per_draw = []
    for draw, frame in enumerate(frames, start=1):
        tab = count_score_table(frame, cutpoints, target, complete_only=True)
        tab["draw"] = draw
        per_draw.append(tab)
    if not per_draw:
        return pd.DataFrame()

    stacked = pd.concat(per_draw, ignore_index=True)
    out = (
        stacked.groupby("n_criteria_met", as_index=False)
        .agg(
            n=("n", "mean"),
            n_high_grade=("n_high_grade", "mean"),
            risk=("risk", "mean"),
            risk_min=("risk", "min"),
            risk_max=("risk", "max"),
        )
    )
    # Wilson interval from the averaged counts: the draws agree closely on the
    # denominators, and quoting no interval at all would be worse.
    lo, hi = ps.wilson_ci(out["n_high_grade"].to_numpy(), out["n"].to_numpy())
    out["risk_lo"], out["risk_hi"] = lo, hi
    out.attrs["k"] = len(cutpoints)
    out.attrs["n_scored"] = int(round(out["n"].sum()))
    out.attrs["m_draws"] = len(frames)
    return out


def draw_count_rules(
    frames: Sequence[pd.DataFrame],
    cutpoints: Sequence[CutPoint],
    target: str,
) -> pd.DataFrame:
    """"≥k criteria" rules scored inside every draw, then averaged."""
    per_draw = []
    for draw, frame in enumerate(frames, start=1):
        tab = count_threshold_table(frame, cutpoints, target, complete_only=True)
        tab["draw"] = draw
        per_draw.append(tab)
    if not per_draw:
        return pd.DataFrame()
    stacked = pd.concat(per_draw, ignore_index=True)
    return (
        stacked.groupby("rule_label", as_index=False, sort=False)
        .agg(
            n_used=("n_used", "mean"),
            sensitivity=("sensitivity", "mean"),
            sens_min=("sensitivity", "min"),
            sens_max=("sensitivity", "max"),
            specificity=("specificity", "mean"),
            spec_min=("specificity", "min"),
            spec_max=("specificity", "max"),
            PPV=("PPV", "mean"),
            NPV=("NPV", "mean"),
            youden_J=("youden_J", "mean"),
            J_min=("youden_J", "min"),
            J_max=("youden_J", "max"),
        )
    )


def flag_missingness(df: pd.DataFrame, cutpoints: Sequence[CutPoint]) -> pd.DataFrame:
    """How many patients the complete-case count score can actually score."""
    flags = flag_frame(df, cutpoints)
    observed = flags.notna().sum(axis=1)
    k = len(cutpoints)
    return pd.DataFrame([{
        "n_patients": len(df),
        "n_all_flags_observed": int((observed == k).sum()),
        "n_some_flag_missing": int((observed < k).sum()),
        "pct_scorable": float((observed == k).mean() * 100) if len(df) else np.nan,
        "k_criteria": k,
    }])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def stability_figure(
    draw_table: pd.DataFrame,
    stability_table: pd.DataFrame,
    metrics: Sequence[Metric],
    *,
    rule: str = "youden",
) -> plt.Figure:
    """The two instabilities side by side, one panel per metric.

    Shaded band = bootstrap interval from the complete cases (sampling noise).
    Dots = the m MICE draws (missing-data uncertainty). Read it as a ratio:
    dots tight inside a wide band means sample size is the problem, not
    missingness; dots scattered across or beyond it means a complete-case
    number quoted alone would be overconfident.
    """
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.31 * nrows),
    )
    rng = ps.deterministic_rng("imputation_stability", rule)

    for ax, metric in zip(axes.ravel(), metrics):
        sub = draw_table[
            (draw_table["column"] == metric.col) & (draw_table["rule"] == rule)
        ]
        row = stability_table[
            (stability_table["column"] == metric.col)
            & (stability_table["rule"] == rule)
        ]
        if sub.empty or row.empty:
            ax.set_axis_off()
            continue
        row = row.iloc[0]

        lo, hi = row.get("cutoff_boot_lo"), row.get("cutoff_boot_hi")
        if pd.notna(lo) and pd.notna(hi):
            ax.axvspan(lo, hi, color=ps.PALETTE["accent"], alpha=0.12, zorder=0,
                       label="Bootstrap 95% (complete case)")
        if pd.notna(row.get("cutoff_complete_case")):
            ax.axvline(row["cutoff_complete_case"], color=ps.PALETTE["accent"],
                       linewidth=1.3, zorder=3, label="Complete-case cut-point")

        cuts = sub["cutoff"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=cuts.size)
        ax.scatter(cuts, jitter, s=13, color=ps.PALETTE["primary"], alpha=0.75,
                   edgecolors="none", zorder=4, label=f"MICE draws (m = {cuts.size})")
        if pd.notna(row.get("cutoff_sd")):
            ax.errorbar(
                [row["cutoff_mean"]], [0.34],
                xerr=[[row["cutoff_sd"]], [row["cutoff_sd"]]],
                fmt="o", markersize=4, color=ps.PALETTE["good"],
                ecolor=ps.PALETTE["good"], elinewidth=1.2, capsize=2.5,
                zorder=5, label="MICE mean ± SD",
            )

        if metric.log_x:
            ax.set_xscale("log")
        ax.set_ylim(-0.55, 0.62)
        ax.set_yticks([])
        ax.set_xlabel(f"Cut-point ({metric.unit})" if metric.unit else "Cut-point")
        ps.set_titles(ax, metric.label, f"{metric.op} cut-point · {rule}")

    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        fontsize=plt.rcParams["font.size"] * 0.72, bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Cut-point stability: sampling noise vs missing-data uncertainty",
                 fontsize=plt.rcParams["font.size"] * 1.05)
    return fig


def knee_stability_figure(
    draws: pd.DataFrame,
    metrics: Sequence[Metric],
) -> plt.Figure:
    """Where the steepest-rise point landed in each draw, and how often it existed.

    Shaded band = the middle half of the draws (25th–75th percentile), so a
    narrow band means the draws agree on where the bend is. The dots are every
    draw that found one, including the outliers the band deliberately excludes.

    An empty panel is a result: in those draws the risk curve had no interior
    steepest point at all, so there was no "šķēre" to locate.
    """
    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.31 * nrows),
    )
    rng = ps.deterministic_rng("knee_stability")

    for ax, metric in zip(axes.ravel(), metrics):
        sub = draws[draws["column"] == metric.col]
        if sub.empty:
            ax.set_axis_off()
            continue
        found = sub[sub["knee_found"]]
        rate = float(sub["knee_found"].mean())

        if len(found):
            vals = found["steepest_x"].to_numpy(dtype=float)
            lo, hi = (float(x) for x in np.percentile(vals, [25, 75]))
            # A degenerate span is a zero-width polygon: invisible, and it would
            # claim a band where every draw in fact landed on the same value.
            if hi > lo:
                ax.axvspan(lo, hi, color=ps.PALETTE["accent"], alpha=0.12,
                           zorder=0, label="Middle half of draws")
            ax.scatter(vals, rng.uniform(-0.2, 0.2, size=vals.size), s=15,
                       color=ps.PALETTE["accent"], alpha=0.8, edgecolors="none",
                       zorder=4, label="One imputed dataset")
            ax.axvline(float(np.median(vals)), color=ps.PALETTE["accent"],
                       linewidth=1.2, zorder=3, label="Median across draws")
        else:
            ax.text(0.5, 0.5, "no interior threshold\nin any draw",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=plt.rcParams["font.size"] * 0.8,
                    color=ps.PALETTE["neutral"])
            # No points means the x range is matplotlib's arbitrary 0–1, and a
            # tick reading "0.6 cc" under an empty panel invites misreading.
            ax.set_xticks([])

        if metric.log_x and len(found):
            ax.set_xscale("log")
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel(metric.axis_label)
        ps.set_titles(ax, metric.label,
                      f"threshold detected in {rate * 100:.0f}% of draws")

    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()

    # Panels differ in what they drew — an empty one has no artists at all, and a
    # unanimous one has no band — so the legend is pooled over every panel.
    pooled: dict[str, Any] = {}
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            pooled.setdefault(label, handle)
    if pooled:
        fig.legend(
            list(pooled.values()), list(pooled), loc="lower center", ncol=3,
            frameon=False, fontsize=plt.rcParams["font.size"] * 0.72,
            bbox_to_anchor=(0.5, -0.04),
        )
    fig.suptitle("Is the steepest-rise point reproducible across imputations?",
                 fontsize=plt.rcParams["font.size"] * 1.05)
    return fig
