"""Model-performance figures rendered from a validation artifact.

One implementation shared by the HTML report, the SVG exports under
``output/inferential/figures/``, and the Streamlit calculator, so a ROC curve
looks the same wherever it appears.

Every figure here is drawn from the *development sample*, which is optimistic by
construction. Each one says so, and quotes the optimism-corrected statistic
next to the apparent one rather than showing the apparent value alone.

Three per-model figures answer three different questions:

- ROC — can the model rank patients? (discrimination)
- Calibration — is a stated 30% risk really 30%? (accuracy of the number)
- Decision curve — is acting on it better than scanning everyone or no one?

plus one across-model figure that ranks the variants side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (
    CATEGORICAL_COLORS,
    FIG_WIDTH_MEDIUM,
    FIG_WIDTH_SINGLE,
    PALETTE,
    apply_plot_style,
    errorbar_lengths,
    figure_size,
    place_legend,
    prettify_label,
    save_figure,
    set_titles,
    wilson_ci,
)

apply_plot_style()

# Apparent (in-sample) vs optimism-corrected, used consistently everywhere.
APPARENT_COLOR = PALETTE["neutral"]
CORRECTED_COLOR = PALETTE["primary"]
REFERENCE_COLOR = "#9a9a9a"


def _metric_row(validation: dict[str, Any], name: str) -> dict | None:
    for row in validation.get("metrics") or []:
        if str(row.get("metric", "")).lower() == name.lower():
            return row
    return None


def _metric_value(validation: dict[str, Any], name: str, field: str) -> float:
    row = _metric_row(validation, name)
    if not row or row.get(field) is None:
        return float("nan")
    try:
        return float(row[field])
    except (TypeError, ValueError):
        return float("nan")


def _sample_note(validation: dict[str, Any]) -> str:
    n_boot = validation.get("successful_bootstraps") or validation.get(
        "bootstrap_resamples"
    )
    return f"apparent fit, {int(n_boot)} bootstrap resamples" if n_boot else "apparent fit"


# ---------------------------------------------------------------------------
# Per-model figures
# ---------------------------------------------------------------------------

def roc_figure(
    validation: dict[str, Any],
    *,
    title: str = "",
    width: float = FIG_WIDTH_SINGLE,
) -> plt.Figure | None:
    """ROC of the apparent model, with the optimism-corrected AUC alongside.

    The curve is in-sample; quoting only its AUC would overstate the model, so
    the corrected value shares the legend.
    """
    curves = (validation.get("roc_curves") or {}).get("curves") or []
    drawn = False
    fig, ax = plt.subplots(figsize=figure_size(width, aspect=1.0))
    for curve, color in zip(curves, CATEGORICAL_COLORS):
        fpr, tpr = curve.get("fpr"), curve.get("tpr")
        if not fpr or not tpr or len(fpr) != len(tpr):
            continue
        label = str(curve.get("label", curve.get("series", "Model")))
        if curve.get("auc") is not None:
            label = f"{label} (AUC {float(curve['auc']):.3f})"
        ax.plot(fpr, tpr, color=color, linewidth=1.8, zorder=3, label=label)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None

    auc_corr = _metric_value(validation, "AUC", "optimism_corrected")
    if np.isfinite(auc_corr):
        ax.plot(
            [], [], linestyle="none",
            label=f"Optimism-corrected AUC {auc_corr:.3f}",
        )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOR, linewidth=1.0,
        zorder=1, label="No discrimination",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    place_legend(ax, loc="lower right", scale=0.78)
    set_titles(ax, title or "Discrimination", _sample_note(validation))
    return fig


def calibration_figure(
    validation: dict[str, Any],
    *,
    title: str = "",
    width: float = FIG_WIDTH_SINGLE,
) -> plt.Figure | None:
    """Observed vs predicted risk, by risk decile, against the ideal diagonal.

    The question a clinician actually asks of a risk calculator: when it says
    30%, is it 30%? Bin points carry Wilson intervals and their denominators,
    and the subtitle reports the optimism-corrected slope — the apparent slope
    is 1.0 by construction on the development sample and means nothing alone.
    """
    cal = validation.get("calibration") or {}
    bins = cal.get("bins") or []
    if not bins:
        return None

    pred = np.array([float(b["predicted"]) for b in bins])
    obs = np.array([float(b["observed"]) for b in bins])
    counts = np.array([int(b["n"]) for b in bins], dtype=float)
    events = np.array([int(b["events"]) for b in bins], dtype=float)
    lo, hi = wilson_ci(events, counts)

    fig, ax = plt.subplots(figsize=figure_size(width, aspect=1.0))
    upper = float(max(pred.max(), obs.max(), float(np.nanmax(hi)))) * 1.1
    upper = float(np.clip(upper, 0.2, 1.0))
    ax.plot(
        [0, upper], [0, upper], linestyle="--", color=REFERENCE_COLOR,
        linewidth=1.0, zorder=1, label="Perfect calibration",
    )

    smooth = cal.get("smooth") or {}
    if smooth.get("predicted"):
        ax.plot(
            smooth["predicted"], smooth["observed"], color=PALETTE["accent"],
            linewidth=1.6, zorder=2, label="Smoothed (LOESS)",
        )
    ax.errorbar(
        pred, obs, yerr=errorbar_lengths(obs, lo, hi),
        fmt="o", color=CORRECTED_COLOR, markersize=5, capsize=2.5,
        linestyle="none", elinewidth=1.0,
        markeredgecolor="white", markeredgewidth=0.8, zorder=3,
        label="Risk decile (95% CI)",
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed proportion")
    place_legend(ax, loc="upper left", scale=0.78)

    slope = cal.get("slope_corrected")
    # Prefer the corrected intercept now that validation produces one; older
    # artifacts carry only the apparent value, and those still render.
    intercept = cal.get("intercept_corrected")
    intercept_label = "corrected intercept"
    if intercept is None or not np.isfinite(float(intercept)):
        intercept, intercept_label = cal.get("intercept_apparent"), "apparent intercept"
    parts = [_sample_note(validation)]
    if slope is not None:
        parts.append(f"corrected slope {float(slope):.2f}")
    if intercept is not None:
        # Three decimals: calibration-in-the-large is anchored to the sample's
        # event rate and lands within a few thousandths of zero, so two decimals
        # round -0.005 to "-0.01" and read as ten times the real value.
        # ``or 0.0`` collapses -0.0 so a null intercept never prints as "-0.000".
        parts.append(f"{intercept_label} {round(float(intercept), 3) or 0.0:+.3f}")
    set_titles(ax, title or "Calibration", " · ".join(parts))
    return fig


def decision_curve_figure(
    validation: dict[str, Any],
    *,
    title: str = "",
    width: float = FIG_WIDTH_MEDIUM,
) -> plt.Figure | None:
    """Net benefit of using the model versus treating everyone or no one.

    The model is only worth acting on where its curve sits above both
    references; discrimination and calibration cannot show that.
    """
    dca = validation.get("decision_curve") or {}
    thresholds = dca.get("thresholds") or []
    model_nb = dca.get("model") or []
    all_nb = dca.get("treat_all") or []
    if not thresholds or len(thresholds) != len(model_nb):
        return None

    t = np.asarray(thresholds, dtype=float)
    nb = np.asarray(model_nb, dtype=float)
    nb_all = np.asarray(all_nb, dtype=float)

    fig, ax = plt.subplots(figsize=figure_size(width, aspect=0.62))
    ax.plot(t, nb, color=CORRECTED_COLOR, linewidth=1.8, zorder=3, label="Model")
    ax.plot(
        t, nb_all, color=PALETTE["accent"], linewidth=1.3, linestyle="-.",
        zorder=2, label="Treat all",
    )
    ax.axhline(
        0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0,
        zorder=1, label="Treat none",
    )

    # Show only the range where a decision is still meaningful.
    useful = t[(nb > 0) & (nb >= nb_all)]
    x_hi = float(min(t.max(), (useful.max() + 0.1) if useful.size else 0.5))
    ax.set_xlim(float(t.min()), max(x_hi, 0.2))
    top = float(np.nanmax(nb)) if np.isfinite(nb).any() else 0.1
    ax.set_ylim(min(-0.02, float(np.nanmin(nb)) * 0.2), max(top * 1.25, 0.05))
    ax.set_xlabel("Risk threshold for acting")
    ax.set_ylabel("Net benefit")
    place_legend(ax, loc="upper right", scale=0.78)

    prevalence = dca.get("prevalence")
    extra = (
        f"outcome rate {float(prevalence):.0%}" if prevalence is not None else None
    )
    parts = [_sample_note(validation)] + ([extra] if extra else [])
    set_titles(ax, title or "Clinical usefulness", " · ".join(parts))
    return fig


# ---------------------------------------------------------------------------
# Across-model comparison
# ---------------------------------------------------------------------------

# (metric name in the validation table, axis label, reference line, better direction)
_COMPARISON_METRICS: tuple[tuple[str, str, float | None, str], ...] = (
    ("AUC", "AUC (discrimination)", 0.5, "higher"),
    ("Brier score", "Brier score (error)", None, "lower"),
    ("Calibration slope", "Calibration slope", 1.0, "target"),
)


def model_comparison_figure(
    entries: Sequence[dict[str, Any]],
    *,
    target: str = "",
) -> plt.Figure | None:
    """Rank every model variant on discrimination, error, and calibration.

    One panel per metric, one row per variant, apparent (hollow) beside
    optimism-corrected (filled) — so the gap between the two markers *is* the
    overfitting, readable at a glance. Ordered by corrected AUC, best on top.
    """
    usable = [e for e in entries if e.get("validation")]
    if not usable:
        return None

    usable = sorted(
        usable,
        key=lambda e: (
            -1e9
            if not np.isfinite(_metric_value(e["validation"], "AUC", "optimism_corrected"))
            else _metric_value(e["validation"], "AUC", "optimism_corrected")
        ),
    )
    labels = [str(e.get("label") or e.get("model_id") or "model") for e in usable]
    y = np.arange(len(usable), dtype=float)

    n_panels = len(_COMPARISON_METRICS)
    fig, axes = plt.subplots(
        1, n_panels, sharey=True, squeeze=False,
        figsize=figure_size(
            2.5 * n_panels + 2.4, height=max(0.42 * len(usable) + 1.9, 2.4),
        ),
    )

    for ax, (metric, axis_label, reference, direction) in zip(
        axes[0], _COMPARISON_METRICS,
    ):
        apparent = np.array(
            [_metric_value(e["validation"], metric, "apparent") for e in usable]
        )
        corrected = np.array(
            [_metric_value(e["validation"], metric, "optimism_corrected") for e in usable]
        )
        for yi, a, c in zip(y, apparent, corrected):
            if np.isfinite(a) and np.isfinite(c):
                ax.plot(
                    [a, c], [yi, yi], color=REFERENCE_COLOR, linewidth=0.9, zorder=2,
                )
        ax.scatter(
            apparent, y, s=26, facecolors="white", edgecolors=APPARENT_COLOR,
            linewidths=1.1, zorder=3, label="Apparent",
        )
        ax.scatter(
            corrected, y, s=30, color=CORRECTED_COLOR, edgecolors="white",
            linewidths=0.8, zorder=4, label="Optimism-corrected",
        )
        if reference is not None:
            ax.axvline(
                reference, color=REFERENCE_COLOR, linestyle="--",
                linewidth=1.0, zorder=1,
            )
        hint = {"higher": "higher is better",
                "lower": "lower is better",
                "target": "1.0 is ideal"}[direction]
        ax.set_xlabel(f"{axis_label}\n({hint})")
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.2)
        ax.grid(axis="y", visible=False)

    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(labels)
    axes[0][0].set_ylim(-0.7, len(usable) - 0.3)
    place_legend(axes[0][-1], loc="lower right", scale=0.75)

    heading = "Model variants compared"
    if target:
        heading += f" — {prettify_label(target)}"
    fig.suptitle(
        f"{heading}\nGap between markers is the optimism removed by bootstrap "
        "internal validation",
        y=1.02,
    )
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

_PER_MODEL_FIGURES = (
    ("roc", roc_figure, "Discrimination"),
    ("calibration", calibration_figure, "Calibration"),
    ("decision_curve", decision_curve_figure, "Clinical usefulness"),
)


def write_performance_figures(
    validation: dict[str, Any],
    figs_dir: Path,
    stem: str,
) -> list[Path]:
    """Write ROC / calibration / decision-curve SVGs for one model variant."""
    figs_dir = Path(figs_dir)
    written: list[Path] = []
    for suffix, builder, title in _PER_MODEL_FIGURES:
        fig = builder(validation, title=title)
        if fig is None:
            continue
        written.append(save_figure(fig, figs_dir / f"{stem}__{suffix}.svg"))
    return written


def write_model_comparison_figure(
    entries: Sequence[dict[str, Any]],
    figs_dir: Path,
    *,
    target: str,
) -> Path | None:
    """Write the across-variant comparison SVG for one target."""
    fig = model_comparison_figure(entries, target=target)
    if fig is None:
        return None
    return save_figure(
        fig, Path(figs_dir) / f"{target}__model_comparison.svg", tight_layout=False,
    )
