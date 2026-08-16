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
    calibration_plot,
    decision_curve,
    figure_size,
    place_legend,
    prettify_label,
    roc_panel,
    save_figure,
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
    drawn: list[dict[str, Any]] = []
    auc_corr = _metric_value(validation, "AUC", "optimism_corrected")
    for curve, color in zip(curves, CATEGORICAL_COLORS):
        fpr, tpr = curve.get("fpr"), curve.get("tpr")
        if not fpr or not tpr or len(fpr) != len(tpr):
            continue
        name = str(curve.get("label", curve.get("series", "Model")))
        extra = ""
        if np.isfinite(auc_corr):
            extra = f"\nOptimism-corrected AUC {auc_corr:.3f}"
        drawn.append({
            "name": name + extra,
            "fpr": fpr,
            "tpr": tpr,
            "auc": curve.get("auc"),
            "color": color,
        })
    if not drawn:
        return None
    fig, ax = roc_panel(drawn, title=title or None, figsize=(width, width))
    del ax
    return fig


def calibration_figure(
    validation: dict[str, Any],
    *,
    title: str = "",
    width: float = FIG_WIDTH_SINGLE,
) -> plt.Figure | None:
    """Observed vs predicted risk, by risk decile, against the ideal diagonal."""
    cal = validation.get("calibration") or {}
    bins = cal.get("bins") or []
    if not bins:
        return None
    slope = cal.get("slope_corrected")
    intercept = cal.get("intercept_corrected")
    if intercept is None or not np.isfinite(float(intercept)):
        intercept = cal.get("intercept_apparent")
    metrics = {
        "slope": slope,
        "intercept": intercept,
        "brier": _metric_value(validation, "Brier score", "optimism_corrected"),
    }
    fig, ax = calibration_plot(
        bins=bins,
        smooth=cal.get("smooth") or {},
        metrics=metrics,
        title=title or None,
        figsize=(width, width + 0.4),
        show_hist=False,
    )
    del ax
    return fig


def decision_curve_figure(
    validation: dict[str, Any],
    *,
    title: str = "",
    width: float = FIG_WIDTH_MEDIUM,
) -> plt.Figure | None:
    """Net benefit of using the model versus treating everyone or no one."""
    dca = validation.get("decision_curve") or {}
    thresholds = dca.get("thresholds") or []
    model_nb = dca.get("model") or []
    all_nb = dca.get("treat_all") or []
    if not thresholds or len(thresholds) != len(model_nb):
        return None
    t = np.asarray(thresholds, dtype=float)
    series = {
        "Model": (t, np.asarray(model_nb, dtype=float)),
        "Treat all": (t, np.asarray(all_nb, dtype=float) if all_nb else t * 0),
        "Treat none": (t, np.zeros_like(t)),
    }
    fig, ax = decision_curve(
        series=series,
        title=title or None,
        figsize=(width, width * 0.62),
        prevalence=dca.get("prevalence"),
    )
    del ax
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
        written.append(save_figure(fig, figs_dir / f"{stem}__{suffix}"))
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
        fig, Path(figs_dir) / f"{target}__model_comparison", tight_layout=False,
    )
