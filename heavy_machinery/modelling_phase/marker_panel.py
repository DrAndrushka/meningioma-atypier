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
from collections.abc import Collection, Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import combinations as cb
import plot_style as ps
from diagnostic_accuracy import binary_diagnostic_metrics
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
    """``2.8 (1.7-4.6)``, or a sentence when the interval covers 1."""
    if pd.isna(row.get("lr_pos")):
        return "—"
    if bool(row.get("chance_overlap")):
        return "not distinguishable from chance"
    text = f"{row['lr_pos']:.1f} ({row['lr_pos_lo']:.1f}–{row['lr_pos_hi']:.1f})"
    return text + "*" if bool(row.get("continuity_corrected")) else text


def marker_panel_reading_view(panel: pd.DataFrame) -> pd.DataFrame:
    """The aim-1 table in the columns a clinician reads."""
    if panel is None or panel.empty:
        return pd.DataFrame(columns=[
            "Marker", "Present in", "Catches",
            "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
        ])
    return pd.DataFrame({
        "Marker": panel["label"],
        "Present in": [f"{int(r['present_n'])}/{int(r['n_used'])}"
                       for _, r in panel.iterrows()],
        "Catches": [f"{int(r['catches'])} of {int(r['n_high_grade'])}"
                    for _, r in panel.iterrows()],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in panel.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in panel.iterrows()],
        "LR+ (95% CI)": [_format_lr(r) for _, r in panel.iterrows()],
    })


def lr_forest_figure(panel: pd.DataFrame) -> plt.Figure:
    """LR+ per marker with its interval, on a log axis with a line at 1.

    Log scale because a likelihood ratio is a multiplier: 0.5 and 2 are the
    same distance from "says nothing", and a linear axis hides that. Markers
    whose interval crosses the line at 1 are drawn in the neutral colour, so
    the ones carrying no information are visible as a group rather than as a
    ranking.
    """
    usable = panel[panel["lr_pos"].notna()] if len(panel) else panel
    if usable is None or usable.empty:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No marker has an estimable likelihood ratio",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ordered = usable.iloc[::-1]
    y = np.arange(len(ordered), dtype=float)
    values = ordered["lr_pos"].to_numpy(dtype=float)
    xerr = ps.errorbar_lengths(values, ordered["lr_pos_lo"], ordered["lr_pos_hi"])
    colors = [
        ps.PALETTE["neutral"] if bool(flag) else ps.PALETTE["high_grade"]
        for flag in ordered["chance_overlap"]
    ]

    height = max(2.0, 0.32 * len(ordered) + 1.0)
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_MEDIUM, height))
    ax.axvline(1.0, color=ps.PALETTE["neutral"], linewidth=0.9, linestyle="-.", zorder=1)
    for i, color in enumerate(colors):
        ax.errorbar(values[i], y[i], xerr=xerr[:, i: i + 1], fmt="o",
                    color=color, ecolor=color, elinewidth=1.1, capsize=2.5,
                    markersize=4, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"].astype(str))
    ax.set_xlabel("Positive likelihood ratio (log scale)")
    ps.set_titles(
        ax, "How much a positive finding argues for high grade",
        "A ratio of 1 says nothing; grey intervals cross it",
    )
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
