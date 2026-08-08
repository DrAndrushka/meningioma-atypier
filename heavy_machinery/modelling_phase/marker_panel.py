"""Which MRI markers, and do they combine? — the two study aims, in one place.

The report's EDA section already scores every marker on its own, and the
threshold phase already knows how to compare a combined rule against a single
one. What is missing is a place where those two answers sit side by side on one
patient set, in the report that actually gets read.

Nothing here re-implements an estimator that already exists. The combination
machinery is :mod:`combinations` from the threshold phase, reached through a
nine-line adapter; the model scoring is :mod:`model_calculator`. The only new
statistic is the positive likelihood ratio, which is what turns "most specific"
into a question with a defensible answer — a sign that is never seen is
perfectly specific and perfectly useless.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import combinations as cb
import plot_style as ps
from cleaning import format_table_for_csv
from diagnostic_accuracy import binary_diagnostic_metrics
from model_calculator import load_model_artifact, predict_from_artifact
from thresholds import format_pct_ci

_Z95 = 1.959963984540054

BINARY_KINDS = ("binary", "derived_binary")


def likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int) -> dict:
    """How much more likely this sign is in a high-grade tumor than a benign one.

    ``LR+ = sensitivity / (1 - specificity)``. An LR+ of 10 means seeing the
    sign makes high grade ten times more likely; an LR+ of 1 means it says
    nothing. The interval is Katz's, computed on the log scale because a ratio
    bounded below by zero and unbounded above is not symmetric.

    A zero in the flagged column makes the ratio infinite and its interval
    undefined. Half a patient is added to every cell in that case
    (Haldane-Anscombe) and ``continuity_corrected`` says so, because the
    resulting interval is honestly enormous and the reader should see why.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
           "chance_overlap": False, "continuity_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = tp == 0 or fp == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    sens = a / (a + c)
    fpr = b / (b + d)
    if fpr <= 0 or sens <= 0:
        return nan

    lr = sens / fpr
    se = math.sqrt(1.0 / a - 1.0 / (a + c) + 1.0 / b - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_pos": float(lr),
        "lr_pos_lo": float(lo),
        "lr_pos_hi": float(hi),
        "chance_overlap": bool(lo <= 1.0 <= hi),
        "continuity_corrected": bool(corrected),
    }


class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a ``CutPoint`` so ``combinations`` accepts it.

    :mod:`combinations` touches exactly four members of a cut-point — ``col``,
    ``label``, ``short_label`` and ``flag`` — so supplying those is enough to
    run its whole rule machinery on markers that were never continuous. This is
    why ``combinations.py`` needs no change: the threshold phase's estimators
    and this section's are the same code, and cannot drift apart.
    """

    col: str
    label: str

    @property
    def short_label(self) -> str:
        """No cut-point to name, so the short form is the label itself."""
        return self.label

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].astype("boolean")


def markers_from_diagnostic_accuracy(
    table: pd.DataFrame,
    *,
    target: str,
    exclude: Collection[str] = (),
) -> list[BinaryMarker]:
    """The marker panel, read from the EDA table rather than hard-coded.

    Whatever is activated or dropped in the cleaning notebook's ``DERIVATIONS``
    flows through here without an edit, which is the point: the panel cannot
    silently fall out of step with the cohort it describes.

    ``exclude`` is the caller's, and belongs in the notebook. The accuracy
    table carries non-imaging predictors too — ``sex_male`` is
    ``derived_binary`` and would otherwise walk into a section about MRI signs.
    """
    if table is None or table.empty:
        return []
    drop = set(exclude) | {target}
    rows = table[
        (table["target"].astype(str) == str(target))
        & (table["kind"].astype(str).isin(BINARY_KINDS))
        & (~table["predictor"].astype(str).isin(drop))
    ]
    return [
        BinaryMarker(str(r["predictor"]), ps.prettify_label(str(r["predictor"])))
        for _, r in rows.iterrows()
    ]


_PANEL_COLUMNS = [
    "marker", "label", "n_used", "present_n", "catches", "n_high_grade",
    "TP", "FP", "FN", "TN",
    "sensitivity", "sensitivity_lo", "sensitivity_hi",
    "specificity", "specificity_lo", "specificity_hi",
    "PPV", "PPV_lo", "PPV_hi", "NPV", "NPV_lo", "NPV_hi",
    "AUC", "OR", "OR_lo", "OR_hi",
    "lr_pos", "lr_pos_lo", "lr_pos_hi", "chance_overlap", "continuity_corrected",
    "p", "test",
]


def marker_panel_table(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """Every marker on its own, ranked by how hard a positive finding argues.

    Ranked by LR+, but ``catches`` is what keeps the ranking honest: the most
    specific sign in a cohort is usually the one nobody ever sees, and a table
    sorted on specificity alone puts it first. Markers whose interval covers 1
    are sorted to the bottom instead of being given a rank they have not
    earned.
    """
    if not markers:
        return pd.DataFrame(columns=_PANEL_COLUMNS)

    rows = []
    for marker in markers:
        row = binary_diagnostic_metrics(
            df, target, marker.col, predictor_series=marker.flag(df),
        )
        row["marker"] = marker.col
        row["label"] = marker.label
        tp, fp = row.get("TP"), row.get("FP")
        fn, tn = row.get("FN"), row.get("TN")
        if any(pd.isna(v) for v in (tp, fp, fn, tn)):
            row.update({"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
                        "chance_overlap": False, "continuity_corrected": False})
            row.update({"present_n": 0, "catches": 0, "n_high_grade": 0})
        else:
            row.update(likelihood_ratio_positive(tp, fp, fn, tn))
            row["present_n"] = int(tp) + int(fp)
            row["catches"] = int(tp)
            row["n_high_grade"] = int(tp) + int(fn)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["chance_overlap"] = out["chance_overlap"].astype(bool)
    out = out.sort_values(
        ["chance_overlap", "lr_pos"], ascending=[True, False], kind="mergesort",
    ).reset_index(drop=True)
    cols = [c for c in _PANEL_COLUMNS if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def _format_lr(row: pd.Series) -> str:
    """``2.79 (1.68–4.63)`` — always the number, never a verdict.

    An interval covering 1 is left to speak for itself. Printing a phrase in
    place of the estimate substitutes our reading for the reader's, and a
    journal table is expected to carry the number either way.

    Two decimals, because one is not enough to tell the reader what the
    footnote asks them to check: at one decimal, mass effect (1.04–1.21,
    excludes 1) and dural tail (0.99–1.21, crosses it) both print as
    ``1.1 (1.0–1.2)``.
    """
    if pd.isna(row.get("lr_pos")):
        return "—"
    text = f"{row['lr_pos']:.2f} ({row['lr_pos_lo']:.2f}–{row['lr_pos_hi']:.2f})"
    return text + "*" if bool(row.get("continuity_corrected")) else text


def _format_present(row: pd.Series) -> str:
    """``59/309 (19%)`` — the epidemiological convention, prevalence scannable."""
    present, used = int(row["present_n"]), int(row["n_used"])
    pct = 100.0 * present / used if used else 0.0
    return f"{present}/{used} ({pct:.0f}%)"


def marker_panel_reading_view(panel: pd.DataFrame) -> pd.DataFrame:
    """The aim-1 table in the columns a clinician reads.

    Ordered by LR+ descending — a single stated sort rule, rather than the
    panel's own "informative first, then the rest", so the table's footnote
    can describe the order in one line.
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=[
            "Marker", "n/N (%)",
            "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
        ])
    ranked = panel.sort_values("lr_pos", ascending=False, na_position="last",
                               kind="mergesort")
    return pd.DataFrame({
        "Marker": ranked["label"],
        "n/N (%)": [_format_present(r) for _, r in ranked.iterrows()],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in ranked.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in ranked.iterrows()],
        "LR+ (95% CI)": [_format_lr(r) for _, r in ranked.iterrows()],
    }).reset_index(drop=True)


_FOREST_BAND_COLOR = "#F1F1F1"
_FOREST_TICKS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)


def _forest_ticks(lo: float, hi: float) -> list[float]:
    """Round multipliers a reader can name, inside the plotted range."""
    inside = [t for t in _FOREST_TICKS if lo <= t <= hi]
    return inside if inside else [1.0]


def lr_forest_figure(panel: pd.DataFrame) -> plt.Figure:
    """LR+ per marker with its interval, on a log axis with a line at 1.

    Log scale because a likelihood ratio is a multiplier: 0.5 and 2 are the
    same distance from "says nothing", and a linear axis hides that. Markers
    whose interval crosses the line at 1 are drawn in the neutral colour and
    on a shaded band, so the ones carrying no information read as a block
    rather than as the bottom of a ranking.

    The values repeat in a right-hand column the way a journal forest plot
    prints them: the dots carry the ranking, the column carries the number,
    and the reader never has to leave the figure to quote one.
    """
    usable = panel[panel["lr_pos"].notna()] if len(panel) else panel
    if usable is None or usable.empty:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No marker has an estimable likelihood ratio",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ordered = usable.iloc[::-1]
    n = len(ordered)
    y = np.arange(n, dtype=float)
    values = ordered["lr_pos"].to_numpy(dtype=float)
    lows = ordered["lr_pos_lo"].to_numpy(dtype=float)
    highs = ordered["lr_pos_hi"].to_numpy(dtype=float)
    xerr = ps.errorbar_lengths(values, ordered["lr_pos_lo"], ordered["lr_pos_hi"])
    crosses = ordered["chance_overlap"].to_numpy(dtype=bool)
    informative = ps.PALETTE["high_grade"]
    muted = ps.PALETTE["neutral"]

    base_size = plt.rcParams["font.size"]
    height = max(2.6, 0.34 * n + 1.5)
    fig, (ax, tax) = plt.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [3.0, 1.15], "wspace": 0.03},
        figsize=(ps.FIG_WIDTH_DOUBLE, height),
    )

    # One band per uninformative row; neighbours merge into a single block.
    # Only inside the plot frame — a band under the value column would float
    # unframed, and the muted number colour already groups those rows.
    for i in np.flatnonzero(crosses):
        ax.axhspan(i - 0.5, i + 0.5, color=_FOREST_BAND_COLOR,
                   linewidth=0, zorder=0)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.5)
    for i in range(n):
        color = muted if crosses[i] else informative
        ax.errorbar(values[i], y[i], xerr=xerr[:, i: i + 1], fmt="o",
                    color=color, ecolor=color, elinewidth=1.1, capsize=2.5,
                    markersize=4, zorder=3)

    ax.set_xscale("log")
    span_lo, span_hi = float(np.min(lows)) * 0.82, float(np.max(highs)) * 1.30
    ax.set_xlim(span_lo, span_hi)
    ticks = _forest_ticks(span_lo, span_hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(-0.7, n - 0.5 + 1.9)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"].astype(str))
    ax.set_xlabel("Positive likelihood ratio (log scale)")

    # Headroom strip: what the line at 1 means, and how to read the top row.
    label_y = n - 0.5 + 0.55
    # Drawn in data coordinates so it stops short of its own label.
    ax.plot([1.0, 1.0], [-0.7, label_y - 0.12], color="#444444",
            linewidth=0.9, linestyle="-.", zorder=2, clip_on=False)
    ax.text(1.0, label_y, "1× = the finding\nchanges nothing",
            ha="center", va="bottom", fontsize=base_size * 0.74,
            color="#444444", linespacing=1.25)
    if not crosses[-1]:
        ax.annotate(
            f"{values[-1]:.1f}× more likely\nwhen this sign is present",
            xy=(values[-1], y[-1]), xytext=(values[-1], label_y),
            ha="center", va="bottom", fontsize=base_size * 0.74,
            color=informative, linespacing=1.25,
            arrowprops={"arrowstyle": "-", "color": informative,
                        "linewidth": 0.6, "shrinkB": 4.0},
        )
        top_label = ax.get_yticklabels()[-1]
        top_label.set_fontweight("bold")
        top_label.set_color(informative)

    tax.set_xlim(0.0, 1.0)
    tax.set_axis_off()
    tax.text(0.0, n - 0.5 + 0.12, "LR+ (95% CI)", ha="left", va="bottom",
             fontsize=base_size * 0.78, color="#444444")
    for i in range(n):
        tax.text(0.0, y[i], f"{values[i]:.2f} ({lows[i]:.2f}–{highs[i]:.2f})",
                 ha="left", va="center", fontsize=base_size * 0.78,
                 color=muted if crosses[i] else "#222222",
                 fontweight="normal" if crosses[i] or i < n - 1 else "bold")

    handles = [
        plt.Line2D([], [], color=informative, marker="o", markersize=4,
                   linewidth=1.1, label="Interval excludes 1 — argues for high grade"),
        plt.Line2D([], [], color=muted, marker="o", markersize=4,
                   linewidth=1.1, label="Interval crosses 1 — says nothing on its own"),
    ]
    # Anchored a fixed 0.42 inch under the axes, so the gap does not grow with
    # the number of markers the way an axes-fraction offset would.
    axes_height = height - 1.35
    ax.legend(handles=handles, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, -0.42 / axes_height),
              frameon=False, handletextpad=0.5, columnspacing=1.6,
              borderaxespad=0.0, fontsize=base_size * 0.76)

    ps.set_titles(
        ax, "How much a positive finding argues for high grade",
        "Positive likelihood ratio with 95% CI — how many times more often the "
        "sign appears in a high-grade tumour than in a benign one",
    )
    # Hand-set margins: the value column needs a fixed narrow gutter, which
    # tight_layout refuses to honour once wspace is set. Marker labels and the
    # legend sit outside these margins and are recovered by ``bbox_inches``.
    fig.subplots_adjust(left=0.26, right=0.995,
                        top=1.0 - 0.52 / height, bottom=0.55 / height)
    return fig


def usable_markers(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> tuple[list[BinaryMarker], list[dict]]:
    """Markers that vary. A column that is always the same answers nothing.

    An all-absent sign has no true positives and an undefined likelihood ratio;
    an always-present one has no true negatives. Both would enter the rule
    search and produce cells of zeros, so they are dropped here with the reason
    recorded rather than silently.
    """
    kept: list[BinaryMarker] = []
    dropped: list[dict] = []
    y = df[target].astype("boolean")
    for marker in markers:
        flags = marker.flag(df)[y.notna()]
        n_true = int((flags == True).sum())   # noqa: E712 - nullable boolean
        n_false = int((flags == False).sum())  # noqa: E712
        if n_true == 0:
            dropped.append({"marker": marker.col,
                            "reason": "never present in this cohort"})
        elif n_false == 0:
            dropped.append({"marker": marker.col,
                            "reason": "always present in this cohort"})
        else:
            kept.append(marker)
    return kept, dropped


def shared_cohort_frame(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The patients every rule is scored on — threshold-phase logic, reused."""
    if not markers:
        return df.iloc[0:0].copy()
    return cb.shared_cohort(df, markers, target)


def shared_cohort_audit(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    dropped: Sequence[dict] = (),
) -> pd.DataFrame:
    """What the shared set cost, per marker, so the denominator is auditable.

    A restriction to complete cases is defensible; a restriction nobody can
    check is not. This is the table that lets a reader see the loss is one or
    two measurements rather than the marker panel as a whole.
    """
    shared = shared_cohort_frame(df, markers, target)
    y = df[target].astype("boolean")
    rows: list[dict] = [
        {"item": "Patients in the cohort", "value": int(len(df)), "note": ""},
        {"item": "Patients in the shared set", "value": int(len(shared)),
         "note": "every marker observed and the outcome known"},
        {"item": "High grade in the shared set",
         "value": int(shared[target].astype("boolean").sum()) if len(shared) else 0,
         "note": ""},
        {"item": "Markers required", "value": int(len(markers)),
         "note": ", ".join(m.col for m in markers)},
    ]
    for marker in markers:
        missing = int((marker.flag(df).isna() & y.notna()).sum())
        rows.append({"item": marker.col, "value": missing,
                     "note": "patients this marker alone was missing for"})
    for entry in dropped:
        rows.append({"item": entry["marker"], "value": 0,
                     "note": f"excluded — {entry['reason']}"})
    return pd.DataFrame(rows)


def count_score(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """Observed high-grade rate at each number of signs present.

    The answer to the study aim that involves no choosing. Every other
    comparison in this section picks a winner and then has to pay for having
    picked it; this one asks a question with no winner in it, so the number it
    produces is the number.
    """
    return cb.count_score_table(df, markers, target, complete_only=True)


def count_thresholds(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The count used as a test: "at least one sign", "at least two", …"""
    return cb.count_threshold_table(df, markers, target, complete_only=True)


MAX_SUBTITLE_MARKERS = 5


def count_score_figure(
    counts: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    *,
    prevalence: float | None = None,
) -> plt.Figure:
    """Threshold-phase figure, drawn on markers instead of cut-points.

    The subtitle lists the criteria by name, which works for the three or four
    cut-points the threshold phase feeds it and does not work here: sixteen
    marker names is a 200-character run-on line, and matplotlib gives up on the
    layout rather than wrapping it. Past
    :data:`MAX_SUBTITLE_MARKERS` the names are dropped and the axis label —
    "Criteria met (of 16)" — carries the count instead. The marker names are
    not lost; they are the rows of the aim-1 table on the same page.
    """
    counts = counts.copy()
    counts.attrs.setdefault("k", len(markers))
    listed = tuple(markers) if len(markers) <= MAX_SUBTITLE_MARKERS else ()
    return cb.count_score_figure(counts, cutpoints=listed, prevalence=prevalence)


MIN_HEADLINE_N = 10

_HEADLINE_COLUMNS = [
    "k_markers", "min_n", "n_bins_usable", "direction",
    "low_count", "low_n", "low_risk",
    "high_count", "high_n", "high_risk", "note",
]


def count_headline(
    counts: pd.DataFrame,
    *,
    k: int | None = None,
    min_n: int = MIN_HEADLINE_N,
) -> pd.DataFrame:
    """The two bins the headline sentence is allowed to quote, and which way it went.

    The count-score table has a row for every possible count, and the rows at
    the two ends are almost always the thinnest: on this cohort the highest
    occupied bin holds a single patient, whose outcome sets that bin's "risk"
    to 0% or 100% with nothing behind it. A sentence built from the first and
    last *occupied* rows therefore reports two coin flips and calls it a trend.

    So the endpoints are chosen from bins with at least ``min_n`` patients, and
    the direction is measured rather than assumed. ``direction`` is
    ``"rises"``, ``"falls"`` or ``"flat"``, and the renderer picks its wording
    from that instead of hard-coding "rises" — a sentence that is true only
    when the data cooperates is not a finding, it is a hope.

    If no two bins clear ``min_n`` the floor is relaxed to "occupied at all"
    and ``note`` says so, because a thin honest sentence beats no sentence.
    """
    if counts is None or counts.empty or "n" not in counts.columns:
        return pd.DataFrame(columns=_HEADLINE_COLUMNS)

    ordered = counts.sort_values("n_criteria_met", kind="mergesort")
    occupied = ordered[(ordered["n"] > 0) & ordered["risk"].notna()]

    note = ""
    used_min = int(min_n)
    usable = occupied[occupied["n"] >= used_min]
    if len(usable) < 2:
        usable = occupied
        used_min = 1
        note = (f"no two counts reached {int(min_n)} patients — the endpoints "
                "below rest on whichever counts were occupied at all")
    if len(usable) < 2:
        return pd.DataFrame(columns=_HEADLINE_COLUMNS)

    if k is None:
        k = int(counts.attrs.get("k", int(ordered["n_criteria_met"].max())))
    low, high = usable.iloc[0], usable.iloc[-1]
    low_risk, high_risk = float(low["risk"]), float(high["risk"])
    direction = (
        "rises" if high_risk > low_risk
        else "falls" if high_risk < low_risk
        else "flat"
    )
    return pd.DataFrame([{
        "k_markers": int(k),
        "min_n": used_min,
        "n_bins_usable": int(len(usable)),
        "direction": direction,
        "low_count": int(low["n_criteria_met"]),
        "low_n": int(low["n"]),
        "low_risk": low_risk,
        "high_count": int(high["n_criteria_met"]),
        "high_n": int(high["n"]),
        "high_risk": high_risk,
        "note": note,
    }])


DEFAULT_SEED = 20260801
DEFAULT_N_BOOT = 500
DEFAULT_DRAW_N_BOOT = 200


def rule_menu(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    max_size: int = 2,
) -> pd.DataFrame:
    """Singles, AND/OR pairs and count rules, all on one patient set.

    Pairs only. ``max_size=3`` is available but an AND of three signs on this
    many events lands in single figures, and a sensitivity computed from eight
    patients is not a number worth printing.
    """
    return cb.full_rule_menu(df, markers, target, max_size=max_size)


def rule_reading_view(menu: pd.DataFrame, *, top: int | None = 12) -> pd.DataFrame:
    """Rules ranked by Youden J, in clinician-facing columns."""
    return cb.combination_reading_view(menu, top=top)


def selection_correction(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> pd.DataFrame:
    """What the winner's advantage is worth once you pay for having picked it.

    With a dozen markers there are hundreds of candidate rules, and the best of
    hundreds beats the best single marker on the data that chose it even when
    no rule is genuinely better. :func:`combinations.bootstrap_best_rule`
    measures that gap by rebuilding the whole menu on each resample, taking its
    winner, and scoring that same rule back on the original cohort.

    It is run **twice**, with the same budget and seed. Correcting only the
    combination and comparing it against an uncorrected single is the error
    recorded in ``CHANGES.md``: it reported a gain of +0.008 that was really
    +0.050, because choosing the best of the singles costs almost exactly as
    much as choosing the best of the combinations.

    Both sides come out of one resample loop. They always used the same seed,
    so they were already visiting the same resamples; the second run was only
    re-scoring the same menu to look at a different part of it.
    """
    sides = {
        "best single": ("single",),
        "best combination": ("and", "or", "count"),
    }
    results = cb.bootstrap_best_rules(
        df, markers, target, sides=sides, n_boot=n_boot, seed=seed,
        max_size=max_size,
    )
    rows = [
        {
            "side": label,
            "best_rule": results[label].get("best_rule", ""),
            "J_apparent": results[label].get("J_apparent", np.nan),
            "optimism": results[label].get("optimism", np.nan),
            "J_corrected": results[label].get("J_corrected", np.nan),
            "winner_stability": results[label].get("winner_stability", np.nan),
            "n_bootstrap": results[label].get("n_bootstrap", 0),
        }
        for label in sides
    ]
    out = pd.DataFrame(rows)
    gain = float(out.loc[1, "J_corrected"] - out.loc[0, "J_corrected"])
    gain_apparent = float(out.loc[1, "J_apparent"] - out.loc[0, "J_apparent"])
    out["gain_apparent"] = gain_apparent
    out["gain_corrected"] = gain
    out["correction_effect"] = _correction_effect(gain_apparent, gain)
    return out


def _correction_effect(gain_apparent: float, gain_corrected: float) -> str:
    """Which way correcting both sides moved the gap — measured, not assumed.

    The intuition is that correction shrinks a gap, and often it does. It is
    not a law. Correction subtracts each side's *own* selection optimism, and
    on this cohort the best-of-16-singles side pays more of it than the
    best-of-210-combinations side, so the corrected gap is the **larger** one.
    Prose that asserts "the uncorrected gap is larger" is therefore a claim
    about the data, and belongs in a column where it can be wrong out loud.
    """
    if not (np.isfinite(gain_apparent) and np.isfinite(gain_corrected)):
        return ""
    if gain_corrected > gain_apparent:
        return "widens"
    if gain_corrected < gain_apparent:
        return "narrows"
    return "unchanged"


def rule_space_figure(menu: pd.DataFrame, *, top: int = 12) -> plt.Figure:
    """Every rule in sensitivity-specificity space, singles marked apart."""
    return cb.combination_figure(menu, top=top)


def _artifact_auc(artifact: dict, key: str) -> float:
    """Read one stored AUC (``apparent`` or ``optimism_corrected``) from a model."""
    metrics = (artifact.get("validation") or {}).get("metrics") or []
    for entry in metrics:
        if str(entry.get("metric")) == "AUC":
            value = entry.get(key)
            return float(value) if value is not None else np.nan
    return np.nan


def score_model_on(df: pd.DataFrame, artifact: dict) -> pd.Series:
    """Apply a saved model's coefficients to this cohort, row by row.

    Re-scoring, not refitting. The coefficients are the ones the modelling
    phase already fitted and shrank; only the patient set changes, which is the
    whole point — a model AUC computed on 352 patients and a marker's Youden J
    computed on 301 are not comparable, and the fix is to move the model, not
    to hope the difference is small.

    A patient missing any predictor scores ``NaN``. Filling one in here would
    be imputation smuggled into a scoring helper, and the section's whole claim
    is that its accuracy numbers describe findings someone actually saw.
    """
    features = artifact.get("features") or []
    names = [str(f["name"]) for f in features]
    missing_cols = [n for n in names if n not in df.columns]
    if missing_cols:
        raise KeyError(f"cohort is missing model predictors: {missing_cols}")

    out: list[float] = []
    for _, row in df.iterrows():
        values = {n: row[n] for n in names}
        if any(pd.isna(v) for v in values.values()):
            out.append(np.nan)
            continue
        inputs = {
            n: (bool(v) if str(f.get("type")) == "binary" else v)
            for (n, v), f in zip(values.items(), features)
        }
        out.append(float(predict_from_artifact(inputs, artifact)))
    return pd.Series(out, index=df.index, dtype="float64")


_MODEL_COLUMNS = [
    "model", "n_scored", "n_complete_own", "denominator",
    "auc_shared_apparent", "auc_artifact_corrected", "auc_artifact_apparent",
    "best_single_rule", "n_best_single", "best_single_auc_corrected",
    "best_single_J_corrected", "note", "source_link",
]

DENOM_SHARED = "the patients every model could score"
DENOM_OWN = "this model's own complete cases"


def _complete_case_mask(df: pd.DataFrame, artifact: dict) -> tuple[pd.Series | None, list[str]]:
    """Which patients have every one of this model's predictors recorded."""
    names = [str(f["name"]) for f in (artifact.get("features") or [])]
    missing_cols = [n for n in names if n not in df.columns]
    if missing_cols:
        return None, missing_cols
    if not names:
        return pd.Series(True, index=df.index), []
    return df[names].notna().all(axis=1), []


def model_vs_single(
    df: pd.DataFrame,
    artifacts: dict[str, dict],
    target: str,
    correction: pd.DataFrame,
    links: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Each multivariable model against the best single marker, on one patient set.

    ``links`` maps a model key to the paper its predictor set came from, and
    lands in ``source_link``. A reader comparing our AUC against a published
    one should be one click from the published one. Models with no entry — the
    experimental variants are ours — get an empty string rather than a guess;
    inventing a citation is worse than omitting it.

    **One denominator, for the models too.** Restricting the markers to a
    shared set and then letting each model score on whatever patients happened
    to have its own predictors would reinstate exactly the error this section
    exists to prevent: an AUC on 344 patients and an AUC on 301 are two
    different questions, and putting them in one column invites the reader to
    subtract them. Every model is therefore scored on the intersection of the
    shared set with *all* models' complete cases — one number in ``n_scored``,
    the same for every row, named in ``denominator``. ``n_complete_own`` keeps
    what each model could have scored, so the cost of the restriction is
    visible rather than asserted.

    If that intersection is empty or has only one outcome class — a degenerate
    run, not a normal one — each model falls back to its own complete cases and
    ``denominator`` says so on every row, so the table never silently mixes.

    Four accuracy columns, deliberately not collapsed:

    ``auc_shared_apparent``   the model re-scored on the shared set — the
                              like-for-like comparison, and *apparent*, because
                              correcting it would mean re-running the bootstrap
                              on this set, which is refitting.
    ``auc_artifact_corrected`` the model's own optimism-corrected AUC from its
                              artifact, on its own patients. The gap between
                              this and ``auc_artifact_apparent`` bounds how
                              optimistic the re-scored column is.
    ``best_single_auc_corrected`` the single-marker side as an **AUC**, which is
                              the column that compares with the three above. For
                              a yes/no rule ``AUC = (J + 1) / 2``, so a Youden J
                              of 0.14 is an AUC of 0.57 — not 0.14. Printing the
                              J beside an AUC and letting the reader compare
                              them makes a modest model look five times better
                              than the best sign.
    ``best_single_J_corrected`` the same quantity on the Youden scale, kept
                              because it is what the rule menu is ranked by. It
                              compares with the rule table, not with an AUC.

    ``n_best_single`` records that the single-marker side is scored on the
    whole marker shared set, which the models' set is a subset of. The two are
    not forced together: the marker shared set is the denominator the rest of
    the section is built on, and re-running the selection bootstrap on the
    smaller set would produce a second, different "best single" on the same
    page. Both counts are written down so the residual difference is visible.
    """
    if not artifacts:
        return pd.DataFrame(columns=_MODEL_COLUMNS)

    single = correction[correction["side"] == "best single"]
    best_rule = str(single["best_rule"].iloc[0]) if len(single) else ""
    best_j = float(single["J_corrected"].iloc[0]) if len(single) else np.nan
    best_auc = (best_j + 1.0) / 2.0 if np.isfinite(best_j) else np.nan
    y = df[target].astype("boolean")
    known = y.notna()

    masks: dict[str, pd.Series] = {}
    missing_by_model: dict[str, list[str]] = {}
    for name, artifact in artifacts.items():
        mask, missing_cols = _complete_case_mask(df, artifact)
        missing_by_model[name] = missing_cols
        if mask is not None:
            masks[name] = mask.fillna(False).astype(bool)

    common = known.copy()
    for mask in masks.values():
        common = common & mask
    use_common = bool(masks) and int(common.sum()) > 0 and \
        y[common].astype(int).nunique() == 2

    rows: list[dict] = []
    for name, artifact in artifacts.items():
        note = ""
        auc = np.nan
        n_scored = 0
        n_own = 0
        denominator = ""

        missing_cols = missing_by_model[name]
        if missing_cols:
            note = f"not scorable on this set — cohort is missing model predictors: {missing_cols}"
        else:
            own = masks[name] & known
            n_own = int(own.sum())
            scored_on = common if use_common else own
            denominator = DENOM_SHARED if use_common else DENOM_OWN
            probs = score_model_on(df, artifact)
            usable = scored_on & probs.notna()
            n_scored = int(usable.sum())
            truth = y[usable].astype(int)
            if n_scored == 0:
                note = "not scorable on this set — no patient had every predictor recorded"
            elif truth.nunique() == 2:
                auc = float(roc_auc_score(truth, probs[usable]))
            else:
                note = "not scorable on this set — one outcome class only"

        rows.append({
            "model": name,
            "n_scored": n_scored,
            "n_complete_own": n_own,
            "denominator": denominator,
            "auc_shared_apparent": auc,
            "auc_artifact_corrected": _artifact_auc(artifact, "optimism_corrected"),
            "auc_artifact_apparent": _artifact_auc(artifact, "apparent"),
            "best_single_rule": best_rule,
            "n_best_single": int(known.sum()),
            "best_single_auc_corrected": best_auc,
            "best_single_J_corrected": best_j,
            "note": note,
            "source_link": str((links or {}).get(name, "") or ""),
        })
    return pd.DataFrame(rows)[_MODEL_COLUMNS]


def _fmt_num(value, digits: int = 3) -> str:
    """A number for a reading view, or an em dash where there is none."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{digits}f}"


def _fmt_count(value) -> str:
    """A patient count, or an em dash when there is none to report."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) or v <= 0 else f"{int(round(v))}"


def _model_label(name: str) -> str:
    """``amano_et_al_2021`` → ``Amano et al 2021``.

    Not :func:`plot_style.prettify_label`, which expands acronyms and turns
    ``et_al`` into ``ET AL``. The model keys are already author-and-year
    citations; they only need their underscores back as spaces.
    """
    text = str(name).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


_MODEL_VIEW_COLUMNS = [
    "Model", "Patients scored", "Model AUC here (apparent)",
    "Model AUC, own patients (corrected)", "Model AUC, own patients (apparent)",
    "Best single sign", "Best single AUC (corrected)",
    "Best single Youden J (corrected)", "Note",
]


def _model_note(row: pd.Series) -> str:
    """The row's note and its source link, in that order, one cell.

    Kept as a bare URL rather than an anchor: this cell is written to CSV as
    well as rendered, and a CSV full of ``<a href>`` is a CSV nobody can open
    in a spreadsheet. ``report.py`` turns the URL into a link at render time.
    """
    note = str(row.get("note") or "").strip()
    link = str(row.get("source_link") or "").strip()
    return " ".join(part for part in (note, link) if part)


def model_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """The model comparison with column headings a reader can act on.

    Machine names are fine in a CSV and wrong in a report: a page that prints
    ``auc_shared_apparent`` beside ``best_single_J_corrected`` asks the reader
    to know which of the two is on a 0.5–1 scale. The headings here say which
    patients each number is on and whether it has been corrected, so the two
    columns that are comparable look comparable and the one that is not is
    named as a Youden J.
    """
    if table is None or table.empty:
        return pd.DataFrame(columns=_MODEL_VIEW_COLUMNS)
    return pd.DataFrame({
        "Model": [_model_label(r["model"]) for _, r in table.iterrows()],
        "Patients scored": [_fmt_count(r.get("n_scored"))
                            for _, r in table.iterrows()],
        "Model AUC here (apparent)": [
            _fmt_num(r.get("auc_shared_apparent")) for _, r in table.iterrows()
        ],
        "Model AUC, own patients (corrected)": [
            _fmt_num(r.get("auc_artifact_corrected")) for _, r in table.iterrows()
        ],
        "Model AUC, own patients (apparent)": [
            _fmt_num(r.get("auc_artifact_apparent")) for _, r in table.iterrows()
        ],
        "Best single sign": [str(r.get("best_single_rule") or "")
                             for _, r in table.iterrows()],
        "Best single AUC (corrected)": [
            _fmt_num(r.get("best_single_auc_corrected")) for _, r in table.iterrows()
        ],
        "Best single Youden J (corrected)": [
            _fmt_num(r.get("best_single_J_corrected")) for _, r in table.iterrows()
        ],
        "Note": [_model_note(r) for _, r in table.iterrows()],
    })


def imputation_stability(
    draws: Sequence[pd.DataFrame],
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    n_boot: int = 200,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """How often the observed-data story survives filling in the missing scans.

    Reported as reproduction rates, not pooled estimates. Rubin's rules average
    an *estimate*; they cannot average a *choice*, and both headline answers
    here are choices — which marker ranks first, which rule wins. A pooled
    "winner" would be an average of things that are not on the same scale.

    The modal answer is carried alongside each rate so a low rate is
    interpretable: "the same rule won in 40% of draws, and the runner-up was
    this one" says something; "40%" alone does not.
    """
    if not draws:
        return pd.DataFrame([{"item": "Draws", "value": 0,
                              "note": "no MICE draws were found"}])

    top_markers: list[str] = []
    winners: list[str] = []
    combo_wins = 0
    scored = 0

    for i, draw in enumerate(draws):
        kept, _ = usable_markers(draw, markers, target)
        if len(kept) < 2:
            continue
        panel = marker_panel_table(draw, kept, target)
        if not panel.empty:
            top_markers.append(str(panel.iloc[0]["label"]))
        corr = selection_correction(draw, kept, target, n_boot=n_boot, seed=seed + i)
        winners.append(str(corr.loc[1, "best_rule"]))
        if float(corr.loc[0, "gain_corrected"]) > 0:
            combo_wins += 1
        scored += 1

    def _rate(values: Sequence[str]) -> tuple[float, str]:
        if not values:
            return (np.nan, "")
        counts = pd.Series(values).value_counts()
        return (float(counts.iloc[0] / len(values)), str(counts.index[0]))

    top_rate, top_mode = _rate(top_markers)
    win_rate, win_mode = _rate(winners)
    return pd.DataFrame([
        {"item": "Draws", "value": int(len(draws)), "note": f"{scored} scorable"},
        {"item": "Top marker reproduced", "value": top_rate,
         "note": f"most often: {top_mode}" if top_mode else ""},
        {"item": "Winning rule reproduced", "value": win_rate,
         "note": f"most often: {win_mode}" if win_mode else ""},
        {"item": "Combination still beat the best single",
         "value": (combo_wins / scored) if scored else np.nan,
         "note": "share of draws with a positive corrected gain"},
    ])


_STABILITY_VIEW_COLUMNS = ["What was checked", "Result", "Detail"]


def stability_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """The MICE-draw check with its rates shown as rates.

    ``value`` holds two different kinds of number — a count of draws and three
    proportions — so a raw dump prints ``0.4`` where the sentence is "in 40% of
    draws". The count row keeps its integer; everything else is a percentage.
    """
    if table is None or table.empty:
        return pd.DataFrame(columns=_STABILITY_VIEW_COLUMNS)

    results: list[str] = []
    for _, row in table.iterrows():
        raw = row.get("value")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            results.append("—" if raw is None else str(raw))
            continue
        if not np.isfinite(v):
            results.append("—")
        elif str(row.get("item")) == "Draws":
            results.append(f"{int(round(v))}")
        else:
            results.append(f"{v * 100:.0f}%")
    return pd.DataFrame({
        "What was checked": [str(r.get("item") or "") for _, r in table.iterrows()],
        "Result": results,
        "Detail": [str(r.get("note") or "") for _, r in table.iterrows()],
    })


TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"


def _write_table(tables: dict, root: Path, stem: str, frame: pd.DataFrame) -> None:
    tables[stem] = frame
    (root / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    format_table_for_csv(frame).to_csv(
        root / TABLES_DIRNAME / f"{stem}.csv", index=False,
    )


def panel_key(name: str, target: str) -> str:
    """Artifact filenames and variant ids, reduced to the same key.

    ``high_grade_experimental_model_1_model.json`` becomes
    ``experimental_model_1``, so the variant id ``experimental_model_1`` has
    to lose the same affixes or the two never meet. The key is also the model
    name :func:`_model_label` prints in the report.

    Delegates to :func:`inferential._artifact_model_id` rather than doing its
    own string surgery. The notebook's version used ``str.replace``, which
    strips ``_model`` wherever it appears rather than only as a suffix; the
    two sides agreed only because both were mangled the same way, and an id
    containing ``_model`` in the middle would have silently mismatched.

    ``_artifact_model_id`` deliberately returns ``""`` for the single-model
    artifact shape ``{target}_model.json`` — there is no model id to strip
    off, only the target. Falling back to ``target`` here means that shape
    still gets a real key instead of an empty one, so a reading-view row
    built from it carries a ``Model`` cell rather than a blank.
    """
    from inferential import _artifact_model_id

    return _artifact_model_id(name, target) or target


def load_panel_artifacts(output_root: Path | str, target: str) -> dict[str, dict]:
    """Every fitted model artifact under ``output/inferential/model_artifacts/``.

    Empty when the inferential stage has not run — a panel without models
    still answers aim 1, so this is a missing section, not an error.

    Filtered on each artifact's own ``target`` field, not just its filename.
    The glob below matches every ``*_model.json`` in the directory regardless
    of target; without this filter, a second outcome added to the notebook's
    model lists would write e.g. ``brain_invasion_xyz_model.json`` next to
    these, and it would be loaded here, keyed by filename, and scored against
    the wrong target. Worse, :func:`model_vs_single` intersects every loaded
    model's complete-case mask into one shared denominator, so a foreign
    model would quietly shrink ``n_scored`` for every row in the table.
    """
    art_dir = Path(output_root) / "inferential" / "model_artifacts"
    if not art_dir.exists():
        return {}
    out: dict[str, dict] = {}
    for path in sorted(art_dir.glob("*_model.json")):
        artifact = load_model_artifact(path)
        if str(artifact.get("target")) != str(target):
            continue
        out[panel_key(path.stem, target)] = artifact
    return out


def model_links_from_variants(variants: Sequence, target: str) -> dict[str, str]:
    """The paper each published predictor set came from, keyed like the artifacts.

    So the model comparison table links out to what it is being compared
    against. Our own experimental variants carry an empty link and get no
    citation — inventing one is worse than leaving the cell blank.
    """
    from inferential import normalize_inferential_variants

    if not variants:
        return {}
    return {
        panel_key(var.model_id, target): var.link
        for var in normalize_inferential_variants(variants=list(variants),
                                                  default_target=target)
        if var.link
    }


def _panel_draws(output_root: Path | str) -> list[pd.DataFrame]:
    """The MICE draws, or none when the cohort was filled by simple imputation.

    The draws are a stability check, not an input to any published number, so
    a cohort without them loses one table rather than the whole panel.

    Guards on the ``missingness/mice/`` directory existing, not on the
    manifest inside it. A simple-imputation cohort has no such directory at
    all, so it still returns ``[]`` here — no error, one missing table. But a
    MICE run that crashed after writing its parquets and before its manifest
    leaves the directory present and broken, and that distinction matters: it
    used to be swallowed by checking for the manifest directly, which turned
    a loud, useful ``FileNotFoundError`` from :func:`load_imputed_frames` into
    a silently empty stability table. Checking the directory instead lets
    that error surface, because a half-written run is a real failure worth
    raising, not a cohort shape worth handling quietly.
    """
    from missingness_resolution import load_imputed_frames

    mice_dir = Path(output_root) / "missingness" / "mice"
    if not mice_dir.exists():
        return []
    return load_imputed_frames(output_root)


def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    variants: Sequence = (),
    artifacts: dict[str, dict] | None = None,
    draws: Sequence[pd.DataFrame] | None = None,
    model_links: Mapping[str, str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    draw_n_boot: int = DEFAULT_DRAW_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> dict[str, pd.DataFrame]:
    """Compute the whole panel and write it to ``output/panel/``.

    Two patient sets on purpose. The aim-1 marker table uses each marker's own
    complete cases, because it ranks markers and each row stands alone. Every
    aim-2 comparison uses the shared set, because a Youden J compared across
    two different groups of patients is not a comparison.

    Two bootstrap budgets on purpose too. ``n_boot`` buys the two corrections
    on the shared set, run once each; ``draw_n_boot`` buys the one inside
    :func:`imputation_stability`, which runs **per MICE draw** — twenty times
    over. They stayed separate parameters after the menu scoring moved to
    :mod:`rule_matrix`, because they still buy different things: the shared-set
    correction is a number the report prints, and the per-draw one only has to
    settle a reproduction rate.

    Nothing here renders. ``report.py`` reads what this writes.

    ``artifacts``, ``draws`` and ``model_links`` left at ``None`` are found
    rather than passed: the fitted models under ``output_root``, the MICE
    draws beside them, and the paper links carried by ``variants``. A caller
    that passes an empty dict or list means empty, and gets empty.
    """
    root = Path(output_root) / "panel"
    tables: dict[str, pd.DataFrame] = {}

    if artifacts is None:
        artifacts = load_panel_artifacts(output_root, target)
    if draws is None:
        draws = _panel_draws(output_root)
    if model_links is None:
        model_links = model_links_from_variants(variants, target)

    markers = markers_from_diagnostic_accuracy(
        accuracy_table, target=target, exclude=exclude,
    )
    markers = [m for m in markers if m.col in df.columns]
    kept, dropped = usable_markers(df, markers, target)

    panel = marker_panel_table(df, kept, target)
    _write_table(tables, root, "01_marker_panel", panel)
    _write_table(tables, root, "02_marker_panel_reading_view",
                 marker_panel_reading_view(panel))

    shared = shared_cohort_frame(df, kept, target)
    _write_table(tables, root, "03_shared_cohort",
                 shared_cohort_audit(df, kept, target, dropped))

    empty = pd.DataFrame()
    if len(kept) >= 2 and not shared.empty:
        menu = rule_menu(shared, kept, target, max_size=max_size)
        correction = selection_correction(
            shared, kept, target, n_boot=n_boot, seed=seed, max_size=max_size,
        )
        counts = count_score(shared, kept, target)
        models = model_vs_single(shared, artifacts, target, correction,
                                 links=model_links)
        stability = imputation_stability(list(draws), kept, target,
                                          n_boot=draw_n_boot, seed=seed)
        _write_table(tables, root, "05_rule_menu", menu)
        _write_table(tables, root, "06_rule_reading_view", rule_reading_view(menu))
        _write_table(tables, root, "07_count_score", counts)
        _write_table(tables, root, "08_count_thresholds",
                     count_thresholds(shared, kept, target))
        _write_table(tables, root, "09_selection_correction", correction)
        _write_table(tables, root, "10_model_vs_single", models)
        _write_table(tables, root, "11_imputation_stability", stability)
        _write_table(tables, root, "12_count_headline",
                     count_headline(counts, k=len(kept)))
        _write_table(tables, root, "13_model_reading_view",
                     model_reading_view(models))
        _write_table(tables, root, "14_stability_reading_view",
                     stability_reading_view(stability))
    else:
        for stem in ("05_rule_menu", "06_rule_reading_view", "07_count_score",
                     "08_count_thresholds", "09_selection_correction",
                     "10_model_vs_single", "11_imputation_stability",
                     "12_count_headline", "13_model_reading_view",
                     "14_stability_reading_view"):
            _write_table(tables, root, stem, empty)
        menu, counts = empty, empty

    fig_dir = root / FIGURES_DIRNAME
    ps.save_figure(lr_forest_figure(panel), fig_dir / "lr_forest.svg",
                   tight_layout=False)
    prevalence = (
        float(shared[target].astype("boolean").mean()) if len(shared) else None
    )
    if len(counts):
        ps.save_figure(count_score_figure(counts, kept, prevalence=prevalence),
                       fig_dir / "count_score.svg")
        ps.save_figure(rule_space_figure(menu), fig_dir / "rule_space.svg")
    else:
        for name in ("count_score.svg", "rule_space.svg"):
            fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM,
                                                          aspect=0.4))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Not enough markers for a combination",
                    ha="center", va="center", transform=ax.transAxes)
            ps.save_figure(fig, fig_dir / name)

    return tables
