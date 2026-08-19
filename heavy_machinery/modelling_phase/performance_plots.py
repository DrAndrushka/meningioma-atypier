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

import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_style import (
    CATEGORICAL_COLORS,
    FIG_WIDTH_DOUBLE,
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
    set_figure_legend,
)

apply_plot_style()

# The AJNR palette is imported, not re-declared, so this figure cannot drift
# from the OR forest and the cut-point figures a reader sees a page earlier.
try:
    import ajnr_style as aj
except ModuleNotFoundError:  # cutpoint_phase is not on sys.path yet
    from heavy_machinery.config import load as _load_config  # noqa: F401
    import ajnr_style as aj

# Apparent (in-sample) vs optimism-corrected, used consistently everywhere.
# Nothing here is distinguished by hue: filled/hollow and shade carry the
# meaning, so the figure survives greyscale printing and colour blindness.
APPARENT_COLOR = aj.MUTED
CORRECTED_COLOR = aj.INK
REFERENCE_COLOR = aj.REFERENCE


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


def _baseline_brier(entries) -> float | None:
    """Prevalence-only Brier score, if every variant agrees on it.

    Without it "lower is better" has no anchor: a reader cannot tell whether
    0.19 is a good error or barely better than predicting the base rate.
    """
    values = {
        round(float(e["validation"]["baseline_brier"]), 6)
        for e in entries
        if e.get("validation", {}).get("baseline_brier") is not None
    }
    return values.pop() if len(values) == 1 else None


# What each dashed line marks, spelled out for the footnote rather than
# crammed into a legend entry.
_REFERENCE_NAMES = {
    "AUC": "chance discrimination (AUC {value:.2f})",
    "Brier score": "the prevalence-only Brier score ({value:.3f})",
    "Calibration slope": "perfect calibration (slope {value:.1f})",
}


# What the outcome is called in print. ``prettify_label`` produces column
# names ("High-grade"), which is right for an axis and wrong for a title: it
# left the figures announcing "high-grade disease" for a meningioma cohort.
#
# Grade wording follows WHO CNS5 / cIMPACT-NOW update 8, which write grades as
# "CNS WHO grade 1/2/3"; "high-grade meningioma" for CNS WHO grade 2-3 is the
# form AJNR's own 2025 meningioma review uses.
_OUTCOME_PHRASES: dict[str, tuple[str, str, str]] = {
    # target: (singular, plural, definition for the footnote)
    "high_grade": ("high-grade meningioma", "high-grade meningiomas",
                   "CNS WHO grade 2\u20133"),
}


def _outcome(target: str, entries) -> tuple[str, str, str]:
    """Singular, plural and definition for the outcome being predicted.

    Falls back to the artifact's own ``outcome_definition`` and finally to the
    column label, so a target nobody has named yet degrades to something true
    rather than to something wrong.
    """
    if target in _OUTCOME_PHRASES:
        return _OUTCOME_PHRASES[target]
    for entry in entries:
        stated = str(entry.get("outcome_label") or "").strip()
        if stated:
            return stated, stated + "s", ""
    label = prettify_label(target).lower() if target else "the outcome"
    return label, label, ""


def _join(items) -> str:
    """``a``, ``a and b``, ``a, b, and c`` - serial comma, as the journal sets it."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _cohort_note(entries, plural: str = "events", definition: str = "") -> str:
    """``the development sample (n = 352), of which 105 (29.8%) are ...``.

    Variants fitted on different samples have no single cohort to quote, so the
    sentence is dropped rather than quietly reporting the first one's.
    """
    sizes = {(e.get("n"), e.get("events")) for e in entries}
    if len(sizes) != 1:
        return ""
    n, events = sizes.pop()
    if not n:
        return ""
    text = f"the development sample (n = {int(n)})"
    if events:
        text += (f", of which {int(events)} ({100 * events / n:.1f}%) are "
                 f"{plural}")
        if definition:
            text += f" ({definition})"
    return text


def _resamples(entries) -> int | None:
    counts = {
        int(e["validation"].get("bootstrap_resamples")
            or e["validation"].get("successful_bootstraps") or 0)
        for e in entries
    }
    counts.discard(0)
    return counts.pop() if len(counts) == 1 else None


def model_comparison_figure(
    entries: Sequence[dict[str, Any]],
    *,
    target: str = "",
) -> plt.Figure | None:
    """Rank every model variant on discrimination, error, and calibration.

    One panel per metric, one row per variant, apparent (hollow) beside
    optimism-corrected (filled) — so the gap between the two markers *is* the
    overfitting, readable at a glance. Ordered by corrected AUC, best on top.

    Drawn in the AJNR ink palette: black is the value that counts (the
    optimism-corrected estimate), grey is the one that flatters, and the dashed
    black line is the value being tested against — never a background guide.
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
    labels = [_short_label(e.get("label") or e.get("model_id") or "model")
              for e in usable]
    y = np.arange(len(usable), dtype=float)
    brier_baseline = _baseline_brier(usable)
    references: list[str] = []

    n_panels = len(_COMPARISON_METRICS)
    width = FIG_WIDTH_DOUBLE

    # The note is composed before the figure exists: its wrapped height decides
    # how much room the layout must leave, and guessing wrong is what lands a
    # footnote on top of an axis label.
    resamples = _resamples(usable)
    singular, plural, definition = _outcome(target, usable)
    cohort = _cohort_note(usable, plural, definition)
    note = "Note:\u2014Estimates are from "
    note += f"{cohort}." if cohort else "the development sample."
    note += (" Hollow squares are apparent (in-sample) values, filled squares are "
             "optimism-corrected values")
    note += f" from {resamples} bootstrap resamples" if resamples else ""
    note += (", and the connecting line is the optimism removed. Models are "
             "ordered by optimism-corrected AUC. Dashed lines mark the reference "
             "value in each panel.")

    heading = (f"Performance of {len(usable)} candidate models for predicting "
               f"{singular}")
    # The heading and the note go to the report as text, so the canvas keeps
    # only a thin top margin instead of a title band and a footnote block.
    title_in = 0.12
    rows_in = max(0.38 * len(usable), 2.0)
    xlabel_in, legend_in = 0.62, 0.30
    height = rows_in + title_in + xlabel_in + legend_in

    fig, axes = plt.subplots(
        1, n_panels, sharey=True, squeeze=False, figsize=(width, height),
    )
    fig.subplots_adjust(
        left=0.175, right=0.985, wspace=0.18,
        top=1 - title_in / height,
        bottom=(xlabel_in + legend_in) / height,
    )

    for ax, (metric, axis_label, reference, direction) in zip(
        axes[0], _COMPARISON_METRICS,
    ):
        # Alternating row shading, the same device the OR forest uses: with
        # three panels sharing one set of row labels, the eye needs a rail.
        for yi in y[::2]:
            ax.axhspan(
                yi - 0.5, yi + 0.5, facecolor=aj.ROW_BAND,
                alpha=aj.ROW_BAND_ALPHA, linewidth=0, zorder=0,
            )

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
        # The hollow marker is deliberately larger than the filled one: where a
        # model had almost no optimism the two estimates coincide, and equal
        # sizes would let the filled square swallow the hollow one entirely —
        # "no optimism" and "apparent value missing" would look identical.
        ax.scatter(
            apparent, y, s=46, marker=aj.MARKER, facecolors="white",
            edgecolors=APPARENT_COLOR, linewidths=1.0, zorder=3,
            label="Apparent (in-sample)",
        )
        ax.scatter(
            corrected, y, s=15, marker=aj.MARKER, color=CORRECTED_COLOR,
            linewidths=0.0, zorder=4, label="Optimism-corrected",
        )

        line = reference
        if metric == "Brier score" and line is None and brier_baseline is not None:
            line = brier_baseline
        if line is not None:
            ax.axvline(line, zorder=1, **aj.NULL_LINE)
            references.append(_REFERENCE_NAMES[metric].format(value=line))

        hint = {"higher": "higher is better",
                "lower": "lower is better",
                "target": "1.0 is ideal"}[direction]
        ax.set_xlabel(f"{axis_label}\n({hint})")
        ax.margins(x=0.18)
        ax.grid(False)

    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(labels)
    # Flush with the row bands: a half-height band at either end reads as a
    # twelfth model whose markers failed to draw.
    axes[0][0].set_ylim(-0.5, len(usable) - 0.5)

    # The legend sits below the panels, not inside one. In the panel it landed
    # on the bottom row's shading and on the lowest-ranked model's markers —
    # the one row a reader is most likely to be checking.
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if references:
        handles.append(Line2D([], [], **aj.NULL_LINE))
        legend_labels.append("Reference value")
    fig.legend(
        handles, legend_labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.03 / height),
        ncol=len(handles), frameon=False, handletextpad=0.4, columnspacing=1.6,
    )

    # Named after the panels are built, because only then is it known which
    # reference lines were actually drawn.
    if references:
        note = note.replace("the reference value in each panel",
                            _join(references))
    plain = (
        "One row per model. Hollow is the score a model gives itself, filled is "
        "the honest one — the gap is how much it was kidding itself."
    )
    set_figure_legend(fig, title=heading, plain=plain, note=note)
    return fig


# ---------------------------------------------------------------------------
# All variants against the single best variable
# ---------------------------------------------------------------------------

def _short_label(label: str) -> str:
    """``Funari et al. 2023 | imaging score components`` -> ``Funari 2023``.

    The descriptor after the pipe is useful in a table and fatal in a figure:
    eleven of them set the width of the whole panel. Citations collapse to the
    first author and the year, the form the caption and the reference list use
    anyway; "et al.", co-authors and the trailing qualifier all go.
    """
    text = str(label).split("|")[0].strip() or str(label)
    text = re.sub(r"\s+by discrimination", "", text)
    match = re.match(r"^([A-Z][\w'\u2019-]+).*?((?:19|20)\d{2})$", text)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return text[:1].upper() + text[1:]


def _overview_rows(
    overview_csv: Path | None,
) -> tuple[dict[str, dict[str, Any]], float | None, str | None]:
    """Every number the overview figure draws, straight from ``model_overview.csv``.

    Read rather than recomputed, so the figure and the all-models table cannot
    publish two different answers to the same question. A missing file is not an
    error — the figure simply does not draw.

    Each value is ``model_id -> {auc_apparent, auc_corrected, ref, own}`` where
    ``ref`` and ``own`` are ``(delta, ci_lo, ci_hi)`` on the optimism-corrected
    scale. ``ref`` compares every model with the same prespecified single
    predictor; ``own`` compares each with the strongest single variable *it
    contains*. They disagree by design — a model built on a weak ingredient can
    clear its own single comfortably and still lose to tumour volume — which is
    why panel B draws both rather than choosing one.
    """
    if overview_csv is None or not Path(overview_csv).is_file():
        return {}, None, None
    import csv

    def _f(row: dict, key: str) -> float | None:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    out: dict[str, dict[str, Any]] = {}
    ref_auc: float | None = None
    ref_name: str | None = None
    with Path(overview_csv).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if ref_auc is None and row.get("reference_auc_corrected"):
                ref_auc = _f(row, "reference_auc_corrected")
                ref_name = (row.get("reference") or "").strip() or None
            out[row["model_id"]] = {
                "auc_apparent": _f(row, "auc_apparent"),
                "auc_corrected": _f(row, "auc_corrected"),
                # Blank by design in two places: the reference model's own
                # ``delta_ref``, and a one-predictor model's ``delta_own`` —
                # a single variable has no combination to test against itself.
                "ref": (_f(row, "delta_ref_corrected"),
                        _f(row, "delta_ref_ci_lo_corrected"),
                        _f(row, "delta_ref_ci_hi_corrected")),
                "own": (_f(row, "delta_own_corrected"),
                        _f(row, "delta_own_ci_lo_corrected"),
                        _f(row, "delta_own_ci_hi_corrected")),
            }
    label = ref_name.replace("_", " ").capitalize() if ref_name else None
    return out, ref_auc, label


def model_performance_overview_figure(
    entries: Sequence[dict[str, Any]],
    *,
    target: str = "",
    overview: dict[str, dict[str, Any]] | None = None,
    reference_auc: float | None = None,
    reference_label: str | None = None,
    groups: dict[str, str] | None = None,
    group_order: Sequence[str] | None = None,
) -> plt.Figure | None:
    """Where every candidate model lands, and what it bought (A and B).

    A answers "how well does it discriminate": each model's AUC as an
    optimism-corrected value beside the apparent one it was shrunk from, so the
    gap between the two squares *is* that model's overfitting. The dashed line
    is the single prespecified predictor, which several multivariable models do
    not clear.

    B answers "was the combination worth it", against both baselines at once
    because they disagree and the disagreement is the finding. Filled squares
    compare with the shared reference — is this model worth more than the best
    one variable in the cohort at all. Hollow squares compare with the strongest
    single variable the model itself contains — did combining *those* predictors
    help, which is the comparison the source papers published. A model can sit
    left of zero on one and right of zero on the other.

    Rows are grouped into our models and the literature refits, each block
    ordered by optimism-corrected AUC, so a reader is never comparing a model we
    built with one we merely refit without being told which is which.
    """
    stats = dict(overview or {})
    if not stats:
        return None
    labels = {str(e.get("model_id", "")): _short_label(e.get("label", ""))
              for e in entries}
    rows = [(mid, labels.get(mid) or _pretty_variable(mid).capitalize(), st)
            for mid, st in stats.items()
            if st.get("auc_corrected") is not None]
    if len(rows) < 2:
        return None

    # Rows run bottom-to-top, so a block is laid out in reverse and the block
    # listed first in ``group_order`` ends up at the top — where the eye lands.
    gmap = dict(groups or {})
    if gmap:
        order = list(group_order or ())
        for mid, _, _ in rows:
            g = gmap.get(mid)
            if g and g not in order:
                order.append(g)
        blocks = [(g, [r for r in rows if gmap.get(r[0]) == g]) for g in order]
    else:
        blocks = [("", list(rows))]
    blocks = [(g, sorted(rs, key=lambda r: r[2]["auc_corrected"], reverse=True))
              for g, rs in blocks if rs]

    y_of: dict[str, float] = {}
    banded: list[float] = []
    headers: list[tuple[float, str]] = []
    cursor = 0.0
    for g, rs in reversed(blocks):
        # Banding restarts inside each block: carrying the alternation across
        # the gap makes the first row of a block look shifted.
        for rank, (mid, _, _) in enumerate(reversed(rs)):
            y_of[mid] = cursor
            if rank % 2:
                banded.append(cursor)
            cursor += 1.0
        if g:
            headers.append((cursor - 0.40, "\n".join(textwrap.wrap(g, 22))))
            cursor += 1.05
    rows.sort(key=lambda r: y_of[r[0]])
    ys = [y_of[r[0]] for r in rows]

    ref_short = (reference_label or "the single predictor").strip()
    width = FIG_WIDTH_DOUBLE
    singular, plural, definition = _outcome(target, entries)
    cohort = _cohort_note(entries, plural, definition) or "the development sample"
    ref_txt = "" if reference_auc is None else f" ({reference_auc:.2f})"
    note = (
        f"Note:—A, Area under the receiver operating characteristic curve for "
        f"each of the {len(rows)} candidate models in {cohort}. Hollow squares "
        "indicate apparent (in-sample) values; filled squares, optimism-corrected "
        "values, and the connecting line is the optimism removed; dashed line, "
        f"{ref_short.lower()} alone{ref_txt}. "
        f"B, Difference in optimism-corrected AUC against {ref_short.lower()} "
        "alone (filled) and against the strongest single predictor each model "
        "itself contains (hollow); bars, 95% CIs. Values to the right of the "
        "dashed line favor the multivariable model."
    )

    heading = f"Discrimination of {len(rows)} candidate models for predicting {singular}"
    # The heading and the note go to the report as text, so the canvas reserves
    # only the strip the panel letters sit in.
    title_in = 0.24
    rows_in = max(0.315 * cursor, 3.0)
    xlabel_in, legend_in = 0.46, 0.78
    height = rows_in + title_in + xlabel_in + legend_in

    fig, axes = plt.subplots(
        1, 2, squeeze=True, figsize=(width, height),
        gridspec_kw={"width_ratios": (1.0, 1.0), "wspace": 0.82},
    )
    fig.subplots_adjust(
        left=0.135, right=0.985,
        top=1 - title_in / height,
        bottom=(xlabel_in + legend_in) / height,
    )
    top_y = cursor - 1.05 if headers else max(ys) + 0.5
    for ax in axes:
        for yi in banded:
            ax.axhspan(yi - 0.5, yi + 0.5, facecolor=aj.ROW_BAND,
                       alpha=aj.ROW_BAND_ALPHA, linewidth=0, zorder=0)
        ax.set_ylim(-0.5, top_y + 0.55)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)

    # --- A: where each model lands ----------------------------------------
    ax = axes[0]
    if reference_auc is not None:
        ax.axvline(reference_auc, zorder=1, **aj.NULL_LINE)
    for mid, _, st in rows:
        a, c = st["auc_apparent"], st["auc_corrected"]
        if a is not None and c is not None:
            ax.plot([c, a], [y_of[mid]] * 2, color=aj.REFERENCE,
                    linewidth=0.9, zorder=2)
    ax.scatter([r[2]["auc_corrected"] for r in rows], ys,
               s=18, marker=aj.MARKER, color=aj.INK, zorder=3)
    app = [(r[2]["auc_apparent"], y) for r, y in zip(rows, ys)
           if r[2]["auc_apparent"] is not None]
    if app:
        ax.scatter([v for v, _ in app], [y for _, y in app], s=18,
                   marker=aj.MARKER, facecolors="none", edgecolors=aj.INK,
                   linewidths=0.9, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows])
    # The whole panel spans about 0.15 AUC, so two decimals would print three
    # ticks and hide how tightly the middle of the field is packed.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2.5, 5]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("AUC")
    ax.set_title("A", loc="left", fontweight="bold")
    ax.margins(x=0.12)

    # --- B: what the combination bought ------------------------------------
    ax = axes[1]
    ax.axvline(0.0, zorder=1, **aj.NULL_LINE)
    offset = 0.19
    for mid, _, st in rows:
        for key, dy, filled in (("ref", offset, True), ("own", -offset, False)):
            d, lo, hi = st[key]
            if d is None:
                continue
            yy = y_of[mid] + dy
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [yy, yy], color=aj.REFERENCE,
                        linewidth=0.9, zorder=2)
                for x_end in (lo, hi):
                    ax.plot([x_end, x_end], [yy - 0.11, yy + 0.11],
                            color=aj.REFERENCE, linewidth=0.9, zorder=2)
            face = aj.INK if filled else "none"
            ax.scatter([d], [yy], s=16, marker=aj.MARKER, facecolors=face,
                       edgecolors=aj.INK, linewidths=0.9, zorder=3)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2.5, 5]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Δ AUC (optimism-corrected)")
    ax.set_title("B", loc="left", fontweight="bold")
    ax.margins(x=0.10)

    # The block headings sit in the gutter between the panels: outside on the
    # right they push the saved figure past the journal's 7.5-inch limit.
    # Centred in that gutter rather than hung off panel B's left edge, because
    # the heading names a block of rows that runs across both panels.
    box_a, box_b = axes[0].get_position(), axes[1].get_position()
    gutter_mid = -((box_b.x0 - box_a.x1) / 2) / box_b.width
    for y_head, text in headers:
        axes[1].text(gutter_mid, y_head, text,
                     transform=axes[1].get_yaxis_transform(),
                     ha="center", va="center", multialignment="center",
                     style="italic", fontweight="bold",
                     fontsize=plt.rcParams["ytick.labelsize"] * 0.95)

    def _mark(filled: bool) -> Line2D:
        return Line2D([], [], linestyle="none", marker=aj.MARKER,
                      markerfacecolor=aj.INK if filled else "none",
                      markeredgecolor=aj.INK, markeredgewidth=0.9,
                      markersize=aj.MARKER_SIZE)

    legends = (
        (axes[0], [_mark(True), _mark(False), Line2D([], [], **aj.NULL_LINE)],
         ["Optimism-corrected", "Apparent", f"{ref_short} alone"]),
        (axes[1], [_mark(True), _mark(False), Line2D([], [], **aj.NULL_LINE)],
         [f"vs {ref_short.lower()}", "vs its strongest single predictor",
          "No difference"]),
    )
    for ax, handles, texts in legends:
        box = ax.get_position()
        fig.legend(
            handles, texts, loc="upper center",
            bbox_to_anchor=(box.x0 + box.width / 2, legend_in / height),
            bbox_transform=fig.transFigure, ncol=1, frameon=False,
            handletextpad=0.6, borderaxespad=0.0,
        )
    plain = (
        f"One row per model. Left: how well it does, further right is better — "
        f"the dashed line is {ref_short.lower()} alone. Right: it beat a single "
        "measurement if the square is past the dashed line."
    )
    set_figure_legend(fig, title=heading, plain=plain, note=note)
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

# (file suffix, builder, how to read it in ordinary words). No title: the
# report's name-derived caption already says which model and which chart.
_PER_MODEL_FIGURES = (
    ("roc", roc_figure,
     "The higher the curve bulges above the diagonal, the better the model "
     "tells the two groups apart. The diagonal is pure guessing."),
    ("calibration", calibration_figure,
     "Dots on the diagonal mean the predicted risk was right. Below the line, "
     "the model warned of more risk than actually happened."),
    ("decision_curve", decision_curve_figure,
     "Whichever line is highest is the best thing to do at that level of "
     "concern. If the model's line is on top, using it helps."),
)


def write_performance_figures(
    validation: dict[str, Any],
    figs_dir: Path,
    stem: str,
) -> list[Path]:
    """Write ROC / calibration / decision-curve SVGs for one model variant."""
    figs_dir = Path(figs_dir)
    written: list[Path] = []
    for suffix, builder, plain in _PER_MODEL_FIGURES:
        fig = builder(validation)
        if fig is None:
            continue
        set_figure_legend(fig, plain=plain)
        written.append(save_figure(fig, figs_dir / f"{stem}__{suffix}"))
    return written


_UNIT_SUFFIXES = {"_cm3": " (cm\u00b3)"}


def _pretty_variable(name: str) -> str:
    """``edema_volume_cm3`` -> ``edema volume (cm\u00b3)``."""
    out = str(name or "").strip()
    for suffix, replacement in _UNIT_SUFFIXES.items():
        if out.endswith(suffix):
            out = out[: -len(suffix)] + replacement
            break
    return out.replace("_", " ")


_GROUP_ORDER = ("Models developed in this study", "Previously published models, refit")
_PUBLISHED_ID = re.compile(r"_(19|20)\d{2}$")


def _study_groups(entries: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Split our models from the literature refits by model id.

    A published model carries its source year in its id (``funari_2023``,
    ``lin_2014``); ours do not (``experimental_model_1``, ``top_6_variables``).
    The reviewer's first question about this panel is which rows are the
    authors' own, and no amount of legend text answers it as fast as putting
    them in their own block.
    """
    own, pub = _GROUP_ORDER
    return {
        str(e.get("model_id", "")):
            (pub if _PUBLISHED_ID.search(str(e.get("model_id", ""))) else own)
        for e in entries
    }


def write_model_performance_overview_figure(
    entries: Sequence[dict[str, Any]],
    figs_dir: Path,
    *,
    target: str,
    overview_csv: Path | None = None,
    group_rows: bool = True,
) -> Path | None:
    """Write the two-panel model overview SVG/PNG for one target."""
    overview, ref_auc, ref_label = _overview_rows(overview_csv)
    groups = _study_groups(entries) if group_rows else None
    fig = model_performance_overview_figure(
        entries, target=target, overview=overview,
        reference_auc=ref_auc, reference_label=ref_label,
        groups=groups, group_order=_GROUP_ORDER if groups else None,
    )
    if fig is None:
        return None
    return save_figure(
        fig, Path(figs_dir) / f"{target}__model_performance_overview",
        tight_layout=False,
    )


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
