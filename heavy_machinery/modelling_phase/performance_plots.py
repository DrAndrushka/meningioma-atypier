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
from matplotlib.patches import Rectangle
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
    MaxNLocator,
)

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
    import ajnr_format as afmt
except ModuleNotFoundError:  # cutpoint_phase is not on sys.path yet
    from heavy_machinery.config import load as _load_config  # noqa: F401
    import ajnr_style as aj
    import ajnr_format as afmt

# The published models' citations and their manuscript reference numbers. Read
# lazily and tolerantly: the overview figure is the only caller, and a checkout
# without the config must still draw it (with plainer row labels).
def _published_models() -> dict[str, dict]:
    try:
        from heavy_machinery.config import load as _load
        return dict(_load("published_models").PUBLISHED_MODELS)
    except Exception:  # pragma: no cover - config absent or unreadable
        return {}

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
    """How many resamples every entry agrees on, or None.

    An entry with no validation block contributes nothing rather than raising:
    the overview figure is drawn from ``model_overview.csv`` and only reaches
    for the entries to name the cohort, so a caller that passes bare rows must
    get a legend with the sentence dropped, not a traceback.
    """
    counts = {
        int((e.get("validation") or {}).get("bootstrap_resamples")
            or (e.get("validation") or {}).get("successful_bootstraps") or 0)
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
    eleven of them set the width of the whole panel.

    This is the fallback, and author-year is not the journal's citation style —
    the overview figure prefers :func:`_citation_label`, which names a paper the
    way AJNR's running text does and carries its reference number. What is left
    here is the across-variant comparison figure, whose rows are our own model
    variants rather than published papers.
    """
    text = str(label).split("|")[0].strip() or str(label)
    text = re.sub(r"\s+by discrimination", "", text)
    match = re.match(r"^([A-Z][\w'\u2019-]+).*?((?:19|20)\d{2})$", text)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return text[:1].upper() + text[1:]


def _surname(person: str) -> str:
    """``De la Garza Ramos R`` -> ``De la Garza Ramos``: drop trailing initials."""
    return re.sub(r"\s+[A-Z]{1,3}$", "", str(person).strip())


def _author_short(citation: str) -> str | None:
    """How AJNR's running text names a paper: ``Spille et al``.

    AMA — which AJNR follows — names one author, two joined by "and", and three
    or more as the first plus "et al" with no comma before it. There is no year:
    the superscript reference number is the citation. Derived from the stored
    citation rather than stored a second time, so correcting a citation cannot
    leave a stale label behind it.
    """
    head = str(citation or "").split(". ")[0].strip()
    if not head:
        return None
    if "et al" in head:
        return f"{_surname(head.split(',')[0])} et al"
    names = [_surname(part) for part in head.split(",") if part.strip()]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al"


def _citation_label(model_id: str, published: dict[str, dict]) -> str | None:
    """``Spille et al$^{7}$`` — the name, then the reference number in superscript.

    Mathtext rather than the Unicode superscript characters: Arial carries only
    U+00B9/B2/B3, so "et al\u2077" prints a tofu box where the 7 should be.
    :mod:`plot_style` points mathtext at the body font, so the digits come out
    in the same face as the name beside them.
    """
    entry = published.get(str(model_id))
    if not entry:
        return None
    name = _author_short(entry.get("citation", ""))
    if not name:
        return None
    number = entry.get("reference_number")
    if number is None:
        return name
    return f"{name}$^{{{int(number)}}}$"


def _published_roll_call(rows, published: dict[str, dict]) -> str:
    """``Published models are those of Spille et al,7 ... refit in this cohort.``

    Plain digits, not superscript characters: this string is the legend the
    manuscript prints, and the superscript is applied there. Ordered by
    reference number, which is the order AJNR numbers references in anyway.

    The number follows the comma — ``Spille et al,7`` — so the items are joined
    by a space rather than by ", ", and the one before "and" drops its comma.
    """
    named = []
    for mid, _, _ in rows:
        entry = published.get(str(mid))
        if not entry:
            continue
        name = _author_short(entry.get("citation", ""))
        number = entry.get("reference_number")
        if name and number is not None:
            named.append((int(number), name))
    if not named:
        return ""
    named.sort()
    items = [f"{name},{number}" for number, name in named]
    if len(items) > 1:
        number, name = named[-2]
        items[-2] = f"{name}{number}"
        listing = " ".join(items[:-1]) + f" and {items[-1]}"
    else:
        listing = items[0]
    return f"Published models are those of {listing} refit in this cohort. "


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
                # Read here rather than taken off the entry: the count belongs
                # to the fitted model the CSV describes, and an entry assembled
                # for the report does not always carry it.
                "n_predictors": _f(row, "n_predictors"),
                "auc_apparent": _f(row, "auc_apparent"),
                "auc_corrected": _f(row, "auc_corrected"),
                # Blank by design in two places: the reference model's own
                # ``delta_ref``, and a one-predictor model's ``delta_own`` —
                # a single variable has no combination to test against itself.
                # Drawn under each model's name in the gain figure, so the
                # reader can see WHICH ingredient the left panel measured it
                # against rather than being told one exists.
                "best_own_single": (row.get("best_own_single") or "").strip()
                                   or None,
                # Carried alongside the name because the two panels do not
                # score the same variable the same way: the left comparator is
                # taken as the model specifies it (fitting charged, search not),
                # the right one is the winner of a search (both charged). Two
                # rows can name "tumor volume" on the left and still sit at
                # different distances from it on the right, and without the
                # number beside the name that looks like an error.
                "best_own_auc_corrected": _f(row, "best_own_auc_corrected"),
                "ref": (_f(row, "delta_ref_corrected"),
                        _f(row, "delta_ref_ci_lo_corrected"),
                        _f(row, "delta_ref_ci_hi_corrected")),
                "own": (_f(row, "delta_own_corrected"),
                        _f(row, "delta_own_ci_lo_corrected"),
                        _f(row, "delta_own_ci_hi_corrected")),
            }
    label = ref_name.replace("_", " ").capitalize() if ref_name else None
    return out, ref_auc, label


def _rows_and_blocks(entries, overview, groups, group_order):
    """Shared row list: ``(model_id, label, stats)`` grouped and ranked."""
    stats = dict(overview or {})
    if not stats:
        return None, None
    published = _published_models()
    labels = {str(e.get("model_id", "")): _short_label(e.get("label", ""))
              for e in entries}
    # "Top 1 variable" is the pipeline's name for a procedure, not something a
    # reader can act on. What the row IS, is the single predictor the search
    # kept — and the grey line beneath says it was selected here.
    labels[SELECTED_SINGLE_MODEL_ID] = "Best single variable, selected"
    rows = [(mid,
             _citation_label(mid, published)
             or labels.get(mid)
             or _pretty_variable(mid).capitalize(),
             st)
            for mid, st in stats.items()
            if st.get("auc_corrected") is not None]
    if len(rows) < 2:
        return None, None
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
    return rows, blocks


def _stack(blocks, *, top_in, row_h=0.323, head_h=0.208, block_gap=0.104):
    """Lay blocks out downward; return ``(placed, banded, bottom_in)``."""
    y = top_in
    placed, banded = [], []
    for gi, (gname, rs) in enumerate(blocks):
        head_y = y
        y += head_h
        laid = []
        for i, row in enumerate(rs):
            if i % 2 == 0:
                banded.append(y)
            laid.append((y + row_h / 2, row))
            y += row_h
        placed.append((gname, head_y, laid))
        if gi < len(blocks) - 1:
            y += block_gap
    return placed, banded, y


def _band_rows(fig, banded, x0, x1, width, height, row_h=0.323):
    for y0 in banded:
        fig.add_artist(Rectangle(
            (x0 / width, 1 - (y0 + row_h) / height),
            (x1 - x0) / width, row_h / height,
            transform=fig.transFigure, facecolor=aj.ROW_BAND,
            alpha=aj.ROW_BAND_ALPHA, linewidth=0, zorder=0))


def _rule(fig, x0, x1, y_in, width, height, lw=1.0):
    fig.add_artist(plt.Line2D(
        [x0 / width, x1 / width], [1 - y_in / height] * 2,
        transform=fig.transFigure, color=aj.INK, linewidth=lw, zorder=3))


def _n_pred_map(entries, stats):
    n_pred = {str(e.get("model_id", "")): e.get("n_predictors") for e in entries}
    n_pred.update({mid: st["n_predictors"] for mid, st in stats.items()
                   if st.get("n_predictors") is not None})
    return n_pred


def _sub_line(mid, npred):
    """The grey line under a model's name in the discrimination figure."""
    if mid == SELECTED_SINGLE_MODEL_ID:
        return "1 predictor, selected in these data"
    if npred is None:
        return ""
    npred = int(npred)
    return f"{npred} predictor" + ("" if npred == 1 else "s")


SELECTED_SINGLE_MODEL_ID = "top_1_variable"

# 174 mm — the journal's double-column measure, and the width the figures
# reworked for submission are drawn at. Deliberately local to these two plates
# rather than pushed into ``plot_style.FIG_WIDTH_DOUBLE`` (7.2 in = 183 mm),
# which every already-exported figure is built on: moving that constant would
# silently redraw all of them.
OVERVIEW_WIDTH_IN = 174.0 / 25.4

# The discrimination plate is narrower on purpose. It carries three columns —
# name, one plot, one number — and at the full double-column measure the space
# between the longest model name and the start of the plot was dead gutter,
# with the marks themselves crowded into the right half. The gain plate keeps
# 174 mm: two side-by-side panels genuinely need it.
DISCRIMINATION_WIDTH_IN = 150.0 / 25.4


def model_discrimination_figure(
    entries: Sequence[dict[str, Any]],
    *,
    target: str = "",
    overview: dict[str, dict[str, Any]] | None = None,
    groups: dict[str, str] | None = None,
    group_order: Sequence[str] | None = None,
) -> plt.Figure | None:
    """How well each candidate model separates the two grades — and no more.

    One question per figure. The old single plate answered two on every row —
    where a model lands, and whether it beat a comparator — with two markers
    and two number columns per row, and the radiologists it was written for
    could not read it. The comparator question is now :func:`model_gain_figure`.

    Each model is drawn twice: a hollow square at the apparent AUC, a filled
    one at the optimism-corrected AUC, and an arrow between them pointing at
    what the correction took away. That arrow is the whole of "bootstrap-
    corrected" made visible, which is what the word alone never managed.

    No reference line. Tumour volume is not a benchmark drawn here: it is the
    winner of a search, it has its own row like any other candidate, and every
    comparison against it belongs in the gain figure.
    """
    stats = dict(overview or {})
    rows, blocks = _rows_and_blocks(entries, stats, groups, group_order)
    if not blocks:
        return None
    n_pred = _n_pred_map(entries, stats)

    W = DISCRIMINATION_WIDTH_IN
    X_NAME, AX = 0.10, (1.60, 5.30)
    X_AUC_R = W - 0.10
    ROW_H = 0.323
    TOP_PAD, HDR_H, AXIS_H, LEGEND_H = 0.104, 0.205, 0.46, 0.34

    top_in = TOP_PAD + HDR_H
    placed, banded, bot_in = _stack(blocks, top_in=top_in, row_h=ROW_H)
    height = bot_in + AXIS_H + LEGEND_H

    fig = plt.figure(figsize=(W, height))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((AX[0] / W, 1 - bot_in / height,
                       (AX[1] - AX[0]) / W, (bot_in - top_in) / height))
    ax.set_ylim(bot_in, top_in)
    ax.set_yticks([])
    ax.set_facecolor("none")
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    aucs = [r[2]["auc_corrected"] for r in rows]
    aucs += [r[2]["auc_apparent"] for r in rows if r[2]["auc_apparent"] is not None]
    pad = max(0.006, 0.08 * (max(aucs) - min(aucs)))
    lo_x, hi_x = min(aucs) - pad, max(aucs) + pad
    ax.set_xlim(lo_x, hi_x)

    _band_rows(fig, banded, X_NAME, X_AUC_R, W, height, ROW_H)

    small = plt.rcParams["xtick.labelsize"]
    body = plt.rcParams["ytick.labelsize"]

    def _fx(x_in):
        return (x_in - AX[0]) / (AX[1] - AX[0])

    def _text(x_in, y_in, s, *, ha="left", size=None, weight="normal",
              style="normal", color=aj.INK):
        return ax.text(_fx(x_in), y_in, s, transform=ax.get_yaxis_transform(),
                       ha=ha, va="center", fontsize=size or small,
                       fontweight=weight, fontstyle=style, color=color,
                       clip_on=False, zorder=4)

    hdr_y = TOP_PAD + HDR_H * 0.42
    _text(X_NAME, hdr_y, "Model (No. of predictors)", weight="bold")
    _text(X_AUC_R, hdr_y, "AUC", ha="right", weight="bold")
    _rule(fig, X_NAME, X_AUC_R, TOP_PAD + HDR_H - 0.055, W, height)

    for gname, head_y, laid in placed:
        if gname:
            _text(X_NAME, head_y + 0.104, gname, size=body, weight="bold",
                  style="italic")
        for cy, (mid, label, st) in laid:
            _text(X_NAME, cy - 0.055, label, size=body)
            sub = _sub_line(mid, n_pred.get(mid))
            if sub:
                _text(X_NAME, cy + 0.062, sub, size=small, color="#5A5A5A")

            app, cor = st["auc_apparent"], st["auc_corrected"]
            if app is not None:
                span = (app - cor) / (hi_x - lo_x)
                ax.plot([cor, app], [cy, cy], color=aj.REFERENCE,
                        linewidth=0.9, zorder=2)
                # Only where the two markers are far enough apart for a head to
                # read as a head rather than as ink on the square.
                if span > 0.035:
                    ax.annotate(
                        "", xy=(cor + 0.14 * (app - cor), cy),
                        xytext=(cor + 0.42 * (app - cor), cy),
                        arrowprops=dict(arrowstyle="-|>", color=aj.REFERENCE,
                                        linewidth=0.9, shrinkA=0, shrinkB=0),
                        annotation_clip=False, zorder=2)
                ax.plot([app], [cy], marker=aj.MARKER, markersize=aj.MARKER_SIZE,
                        markerfacecolor="white", markeredgecolor=aj.INK,
                        markeredgewidth=0.9, linestyle="none", zorder=3)
            ax.plot([cor], [cy], marker=aj.MARKER, markersize=aj.MARKER_SIZE,
                    color=aj.INK, linestyle="none", zorder=3)
            _text(X_AUC_R, cy, afmt.fmt_est(cor, 3), ha="right", size=body)

    ax.spines["bottom"].set_position(("outward", 4))
    ax.tick_params(axis="x", labelsize=small, length=3.2, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    rule_y = bot_in + AXIS_H - 0.03
    _rule(fig, X_NAME, X_AUC_R, rule_y, W, height)
    fig.text((AX[0] + AX[1]) / 2 / W, 1 - (bot_in + 0.32) / height,
             "Area under the ROC curve — further right, better separation →",
             ha="center", va="center", fontsize=small, color=aj.INK)

    def _mark(filled):
        return Line2D([], [], linestyle="none", marker=aj.MARKER,
                      markerfacecolor=aj.INK if filled else "white",
                      markeredgecolor=aj.INK, markeredgewidth=0.9,
                      markersize=aj.MARKER_SIZE)

    fig.legend(
        [_mark(True), _mark(False),
         Line2D([], [], color=aj.REFERENCE, linewidth=0.9, marker="4",
                markersize=6)],
        ["Optimism-corrected AUC", "Apparent AUC", "Optimism removed"],
        loc="upper left", bbox_to_anchor=(X_NAME / W, 1 - (rule_y + 0.11) / height),
        bbox_transform=fig.transFigure, ncol=3, frameon=False, fontsize=small,
        handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)

    singular, plural, definition = _outcome(target, entries)
    cohort = _cohort_note(entries, plural) or "the development sample"
    resamples = _resamples(entries) or 1000
    grade = f" ({definition})" if definition else ""
    roll = _published_roll_call(rows, _published_models())
    note = (
        f"Note:—Discrimination of {len(rows)} candidate models for {singular}"
        f"{grade} in {cohort}. Open squares indicate apparent area under the "
        "receiver operating characteristic curve (AUC); filled squares, "
        f"optimism-corrected AUC over {resamples} bootstrap resamples; the "
        "arrow, the optimism removed. For the single-variable model the "
        f"correction also covers selection. {roll}"
    )
    plain = ("One row per model, ordered by the honest score. The hollow square "
             "is what the model claims on the data that built it; the arrow is "
             "what the check takes back.")
    heading = (f"Bootstrap-corrected discrimination of {len(rows)} candidate "
               f"models for {singular}")
    set_figure_legend(fig, title=heading, plain=plain, note=note)
    return fig


def _searched_cue(reference_auc: float | None) -> str:
    """Column-level cue for the shared comparator, which every row shares.

    Safe to state once here precisely because this comparator does not vary by
    row — unlike the left panel's, which is named and scored per row.
    """
    if reference_auc is None:
        return "selected by search, not in advance"
    return f"selected by search (AUC {afmt.fmt_est(reference_auc, 3)})"


def model_gain_figure(
    entries: Sequence[dict[str, Any]],
    *,
    target: str = "",
    overview: dict[str, dict[str, Any]] | None = None,
    reference_label: str | None = None,
    reference_auc: float | None = None,
    groups: dict[str, str] | None = None,
    group_order: Sequence[str] | None = None,
) -> plt.Figure | None:
    """Was combining predictors worth it? Two yardsticks, one marker each.

    The two comparators disagree and the disagreement is the finding, so both
    are drawn — but in their own panels, one marker to a row, rather than
    stacked two-deep on a single line with two number columns beside them.

    They are not the same kind of thing, which is why only one of them is
    charged for having been searched for. The model's own strongest predictor
    is handed over by the model; nothing was chosen. The shared comparator is
    the winner of a selection across the candidate pool, so its correction —
    and the vectors behind its interval — carry the cost of that choosing.
    """
    stats = dict(overview or {})
    rows, blocks = _rows_and_blocks(entries, stats, groups, group_order)
    if not blocks:
        return None
    # A model with nothing to compare has no row here: the selected single
    # predictor IS the shared comparator, and a model that contains only one
    # variable has no combination to test against its own ingredient.
    blocks = [(g, [r for r in rs
                   if r[2]["ref"][0] is not None or r[2]["own"][0] is not None])
              for g, rs in blocks]
    blocks = [(g, rs) for g, rs in blocks if rs]
    if not blocks:
        return None
    drawn = [r for _, rs in blocks for r in rs]

    W = OVERVIEW_WIDTH_IN
    X_NAME = 0.10
    AX_L, AX_R = (2.20, 4.45), (4.85, W - 0.10)
    ROW_H = 0.323
    # HDR_H carries three stacked lines over each panel — two bold, one
    # grey cue — and the rule sits 0.055 above the first row. Too small
    # and the cue's descenders land on the rule.
    TOP_PAD, HDR_H, AXIS_H, LEGEND_H = 0.104, 0.54, 0.50, 0.10

    top_in = TOP_PAD + HDR_H
    placed, banded, bot_in = _stack(blocks, top_in=top_in, row_h=ROW_H)
    height = bot_in + AXIS_H + LEGEND_H

    fig = plt.figure(figsize=(W, height))
    fig.patch.set_facecolor("white")

    def _panel(span):
        a = fig.add_axes((span[0] / W, 1 - bot_in / height,
                          (span[1] - span[0]) / W, (bot_in - top_in) / height))
        a.set_ylim(bot_in, top_in)
        a.set_yticks([])
        a.set_facecolor("none")
        a.grid(False)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.tick_params(axis="y", length=0)
        return a

    ax_l, ax_r = _panel(AX_L), _panel(AX_R)

    deltas = [v for _, _, st in drawn for key in ("ref", "own")
              for v in st[key] if v is not None]
    pad = max(0.004, 0.06 * (max(deltas) - min(deltas))) if deltas else 0.05
    lo_b, hi_b = min(deltas + [0.0]) - pad, max(deltas + [0.0]) + pad
    for a in (ax_l, ax_r):
        a.set_xlim(lo_b, hi_b)

    _band_rows(fig, banded, X_NAME, W - 0.10, W, height, ROW_H)

    small = plt.rcParams["xtick.labelsize"]
    body = plt.rcParams["ytick.labelsize"]
    ref_short = (reference_label or "the single predictor").strip().lower()

    def _fig_text(x_in, y_in, s, *, ha="center", size=None, weight="normal",
                  color=aj.INK):
        fig.text(x_in / W, 1 - y_in / height, s, ha=ha, va="center",
                 fontsize=size or small, fontweight=weight, color=color)

    for span, l1, l2, cue in (
            (AX_L, "Does it beat its own", "best single predictor?",
             "as the model specifies it, not searched for"),
            (AX_R, "Does it beat our dataset's",
             f"best predictor ({ref_short})?",
             _searched_cue(reference_auc))):
        cx = (span[0] + span[1]) / 2
        _fig_text(cx, TOP_PAD + 0.09, l1, weight="bold")
        _fig_text(cx, TOP_PAD + 0.22, l2, weight="bold")
        _fig_text(cx, TOP_PAD + 0.34, cue, color="#5A5A5A")
    _fig_text(X_NAME, TOP_PAD + 0.34, "Model", ha="left", weight="bold")
    _rule(fig, X_NAME, W - 0.10, top_in - 0.055, W, height)

    for gname, head_y, laid in placed:
        if gname:
            fig.text(X_NAME / W, 1 - (head_y + 0.104) / height, gname,
                     ha="left", va="center", fontsize=body, fontweight="bold",
                     fontstyle="italic", color=aj.INK)
        for cy, (mid, label, st) in laid:
            fig.text(X_NAME / W, 1 - (cy - 0.055) / height, label, ha="left",
                     va="center", fontsize=body, color=aj.INK)
            own_name = st.get("best_own_single")
            if own_name:
                # The score belongs on the row, not over the column: the left
                # comparator changes with the model, so a single AUC in the
                # heading would be true of one row and wrong for the rest.
                own_auc = st.get("best_own_auc_corrected")
                sub = f"vs {_pretty_variable(own_name)}"
                if own_auc is not None:
                    sub += f" ({afmt.fmt_est(own_auc, 3)})"
                fig.text(X_NAME / W, 1 - (cy + 0.062) / height, sub, ha="left",
                         va="center", fontsize=small, color="#5A5A5A")
            for a, key in ((ax_l, "own"), (ax_r, "ref")):
                d, lo, hi = st[key]
                if d is None:
                    continue
                if lo is not None and hi is not None:
                    a.plot([lo, hi], [cy, cy], color=aj.REFERENCE,
                           linewidth=1.0, zorder=2)
                    for x_end in (lo, hi):
                        a.plot([x_end, x_end], [cy - 0.033, cy + 0.033],
                               color=aj.REFERENCE, linewidth=1.0, zorder=2)
                a.plot([d], [cy], marker=aj.MARKER, markersize=aj.MARKER_SIZE,
                       color=aj.INK, linestyle="none", zorder=3)

    step = 0.05
    majors = np.arange(np.ceil(lo_b / (2 * step)) * 2 * step, hi_b, 2 * step)
    for a, span in ((ax_l, AX_L), (ax_r, AX_R)):
        a.axvline(0.0, zorder=1, **aj.NULL_LINE)
        a.spines["bottom"].set_position(("outward", 4))
        a.tick_params(axis="x", labelsize=small, length=3.2, pad=2)
        a.xaxis.set_major_locator(FixedLocator(majors))
        a.xaxis.set_minor_locator(FixedLocator(
            [t for t in np.arange(np.ceil(lo_b / step) * step, hi_b, step)
             if not np.any(np.isclose(t, majors))]))
        a.tick_params(axis="x", which="minor", length=1.9)
        a.xaxis.set_major_formatter(
            FuncFormatter(lambda t, _pos: afmt.fmt_signed(t, 2)))
        _fig_text((span[0] + span[1]) / 2, bot_in + 0.36,
                  "Δ AUC (95% CI)")

    _rule(fig, X_NAME, W - 0.10, bot_in + AXIS_H - 0.03, W, height)

    singular, plural, definition = _outcome(target, entries)
    ref_auc_txt = ("" if reference_auc is None
                   else f" (corrected AUC, {afmt.fmt_est(reference_auc, 3)})")
    note = (
        "Note:—Δ optimism-corrected AUC between each model and 2 "
        "comparators: the strongest single predictor the model itself contains "
        f"(named beneath each model) and {ref_short} alone, selected in this "
        f"same sample as the strongest single predictor{ref_auc_txt}. Squares "
        "indicate point estimates; bars, 95% CIs; the dashed line, no "
        "difference. Values right of it favor the model."
    )
    plain = ("One row per model. Right of the dashed line the model is ahead; "
             "a bar that touches the line means the difference could be chance.")
    heading = f"Gain over a single predictor for {singular}"
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
) -> list[Path]:
    """Write the two model-overview plates for one target.

    Two files, not one. The single plate this replaced answered "how well does
    it do?" and "did it beat a comparator?" on the same row, and the
    radiologists it was written for could not read it; each question now has a
    figure to itself. Both are drawn from ``model_overview.csv`` — never
    recomputed — so neither can disagree with the table.
    """
    overview, ref_auc, ref_label = _overview_rows(overview_csv)
    groups = _study_groups(entries) if group_rows else None
    order = _GROUP_ORDER if groups else None
    written: list[Path] = []
    for suffix, fig in (
            ("model_discrimination",
             model_discrimination_figure(
                 entries, target=target, overview=overview,
                 groups=groups, group_order=order)),
            ("model_gain",
             model_gain_figure(
                 entries, target=target, overview=overview,
                 reference_label=ref_label, reference_auc=ref_auc,
                 groups=groups, group_order=order)),
    ):
        if fig is None:
            continue
        written.append(save_figure(
            fig, Path(figs_dir) / f"{target}__{suffix}", tight_layout=False,
            # Exported as drawn, so the plate really is 174 mm across.
            bbox_inches="figure"))
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
