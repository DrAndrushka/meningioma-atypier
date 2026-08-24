"""Which MRI markers, and do they combine? — the two study aims, in one place.

The report's EDA section already scores every marker on its own, and
:mod:`combinations` already knows how to compare a combined rule against a
single one. What is missing is a place where those two answers sit side by side
on one patient set, in the report that actually gets read.

Nothing here re-implements an estimator that already exists. The combination
machinery is :mod:`combinations`, reached through a nine-line adapter. The only new statistic is the positive likelihood ratio,
which is what turns "most specific" into a question with a defensible answer —
a sign that is never seen is perfectly specific and perfectly useless.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import combinations as cb
import plot_style as ps
from cleaning import format_table_for_csv
from diagnostic_accuracy import binary_diagnostic_metrics
from eda import benjamini_hochberg
from marker_rules import format_pct_ci

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


def likelihood_ratio_negative(tp: int, fp: int, fn: int, tn: int) -> dict:
    """What *not* seeing the sign is worth — the other half of the 2×2.

    ``LR- = (1 - sensitivity) / specificity``. An LR- of 0.1 means the absence
    of the sign makes high grade ten times *less* likely; an LR- of 1 means
    absence says nothing. Same Katz interval on the log scale as LR+, with the
    miss column standing where the hit column stood.

    The continuity correction fires on ``FN == 0`` or ``TN == 0`` — a different
    row than the LR+ one, which is why it reports its own flag instead of
    sharing ``continuity_corrected``.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_neg": np.nan, "lr_neg_lo": np.nan, "lr_neg_hi": np.nan,
           "lr_neg_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = fn == 0 or tn == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    miss = c / (a + c)
    spec = d / (b + d)
    if spec <= 0 or miss <= 0:
        return nan

    lr = miss / spec
    se = math.sqrt(1.0 / c - 1.0 / (a + c) + 1.0 / d - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_neg": float(lr),
        "lr_neg_lo": float(lo),
        "lr_neg_hi": float(hi),
        "lr_neg_corrected": bool(corrected),
    }


class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a ``CutPoint`` so ``combinations`` accepts it.

    :mod:`combinations` touches exactly four members of a cut-point — ``col``,
    ``label``, ``short_label`` and ``flag`` — so supplying those is enough to
    run its whole rule machinery on markers that were never continuous. This is
    why ``combinations.py`` needs no change: the cut-point estimators and this
    section's are the same code, and cannot drift apart.
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
    "lr_neg", "lr_neg_lo", "lr_neg_hi", "lr_neg_corrected",
    "p", "p_fdr", "test", "origin",
]

# A derived flag is a native column with a line drawn through it —
# ``adc_value_le0.72`` is ``adc_value`` dichotomised. Correcting the two in one
# family counts the same information twice and shifts every native q, so the
# panel is split and each half is corrected on its own.
NATIVE, DERIVED = "native", "derived"


def classify_origin(
    markers: Collection[str],
    *,
    derivation_sources: dict[str, str],
    hidden_parents: Collection[str],
) -> dict[str, str]:
    """Which side of the panel each marker belongs to, and why.

    The test is not "was this column derived" but **is the column it restates
    still in the table**:

    *Parent hidden* — the flag replaced its parent outright. Nothing in the
    table restates anything, so the flag *is* the native variable and is
    corrected with the others. ``male`` replaced ``sex``; there is no ``sex``
    row left for it to duplicate.

    *Parent present* — the flag and the measurement it was cut from are both
    here. ``adc_value_le0.72`` is ``adc_value`` with a line drawn through it,
    and correcting them together tests one thing twice. The flag goes to the
    derived table and takes no part in the native family.

    Driving this off ``hide_parent`` rather than off a hand-kept list is what
    keeps the two in step: change the derivation and the panel follows.
    """
    hidden = {str(h) for h in hidden_parents}
    out: dict[str, str] = {}
    for marker in (str(m) for m in markers):
        parent = derivation_sources.get(marker)
        out[marker] = (
            DERIVED if parent and parent not in hidden else NATIVE
        )
    return out


def _hidden_parents(output_root: Path | str) -> frozenset[str]:
    """Columns a derivation replaced outright, from the cleaning handoff."""
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    if not path.exists():
        return frozenset()
    tbl = pd.read_csv(path)
    if "column" not in tbl.columns:
        return frozenset()
    return frozenset(str(c) for c in tbl["column"].dropna())


def _derivation_parents(output_root: Path | str) -> dict[str, str]:
    """Each derived column's single source, from the cleaning derivation log.

    Only single-source derivations count. A column computed from two others
    (``edema_index``) restates neither of them on its own, and one that lists
    itself is a repair rule rather than a restatement.
    """
    path = Path(output_root) / "cleaning" / "derivation_log.csv"
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    if not {"derivation", "source"} <= set(log.columns):
        return {}
    out: dict[str, str] = {}
    for _, row in log.iterrows():
        name = str(row["derivation"]).strip()
        sources = [s.strip() for s in str(row.get("source", "")).split(",")
                   if s.strip()]
        if len(sources) == 1 and sources[0] != name:
            out[name] = sources[0]
    return out


def marker_panel_table(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    origin_by_marker: dict[str, str] | None = None,
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
            row.update({"lr_neg": np.nan, "lr_neg_lo": np.nan,
                        "lr_neg_hi": np.nan, "lr_neg_corrected": False})
            row.update({"present_n": 0, "catches": 0, "n_high_grade": 0})
        else:
            row.update(likelihood_ratio_positive(tp, fp, fn, tn))
            row.update(likelihood_ratio_negative(tp, fp, fn, tn))
            row["present_n"] = int(tp) + int(fp)
            row["catches"] = int(tp)
            row["n_high_grade"] = int(tp) + int(fn)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["chance_overlap"] = out["chance_overlap"].astype(bool)
    # Corrected across this table's own rows, so the footnote's claim about
    # which family the q-values belong to is true by construction rather than
    # by a comment. Borrowing them from the EDA screen would silently break
    # the moment a marker is excluded here but not there.
    origins = origin_by_marker or {}
    out["origin"] = out["marker"].astype(str).map(
        lambda m: origins.get(m, NATIVE))
    # One BH per origin, not one across the panel. Each q is therefore a rank
    # within its own family, and the two are not comparable with each other —
    # which the table footnote has to say, because nothing in the column does.
    out["p_fdr"] = np.nan
    if "p" in out.columns:
        for origin in (NATIVE, DERIVED):
            mask = out["origin"] == origin
            if mask.any():
                out.loc[mask, "p_fdr"] = benjamini_hochberg(
                    out.loc[mask, "p"]).values
        # Stated, then checked. A BH q is m/rank × p, so if a derived row had
        # slipped into the native family every native q would be wrong by the
        # ratio of the two family sizes — a silent, uniform inflation that no
        # single number on the page would look wrong.
        native_mask = out["origin"] == NATIVE
        if native_mask.any():
            expected = benjamini_hochberg(out.loc[native_mask, "p"]).values
            if not np.allclose(out.loc[native_mask, "p_fdr"].to_numpy(float),
                               np.asarray(expected, dtype=float),
                               equal_nan=True):
                raise ValueError(
                    "The native FDR family is not the native rows alone.")
    out = out.sort_values(
        ["chance_overlap", "lr_pos"], ascending=[True, False], kind="mergesort",
    ).reset_index(drop=True)
    cols = [c for c in _PANEL_COLUMNS if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def _format_lr(row: pd.Series, *, key: str = "lr_pos",
               corrected_key: str = "continuity_corrected") -> str:
    """``2.79 (1.68–4.63)`` — always the number, never a verdict.

    An interval covering 1 is left to speak for itself. Printing a phrase in
    place of the estimate substitutes our reading for the reader's, and a
    journal table is expected to carry the number either way.

    Two decimals, because one is not enough to tell the reader what the
    footnote asks them to check: at one decimal, mass effect (1.04–1.21,
    excludes 1) and dural tail (0.99–1.21, crosses it) both print as
    ``1.1 (1.0–1.2)``.
    """
    if pd.isna(row.get(key)):
        return "—"
    text = f"{row[key]:.2f} ({row[f'{key}_lo']:.2f}–{row[f'{key}_hi']:.2f})"
    return text + "*" if bool(row.get(corrected_key)) else text


def _format_q(value) -> str:
    """``0.003`` / ``<0.001`` — an adjusted p, never in scientific notation."""
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    return "<0.001" if v < 0.001 else f"{v:.3f}"


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

    Predictive values sit next to sensitivity and specificity in the order a
    radiology table is read, and carry the cohort's own prevalence with them:
    they answer "the sign is there, now what?" where LR+ answers "how much
    does seeing it move the odds?".
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=[
            "Variable", "n/N (%)",
            "Sens (95% CI)", "Spec (95% CI)",
            "PPV (95% CI)", "NPV (95% CI)", "FDR p", "LR+ (95% CI)",
            "LR− (95% CI)",
        ])
    ranked = panel.sort_values("lr_pos", ascending=False, na_position="last",
                               kind="mergesort")
    return pd.DataFrame({
        "Variable": ranked["label"],
        "n/N (%)": [_format_present(r) for _, r in ranked.iterrows()],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in ranked.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in ranked.iterrows()],
        "PPV (95% CI)": [format_pct_ci(r, "PPV") for _, r in ranked.iterrows()],
        "NPV (95% CI)": [format_pct_ci(r, "NPV") for _, r in ranked.iterrows()],
        "FDR p": [_format_q(r.get("p_fdr")) for _, r in ranked.iterrows()],
        "LR+ (95% CI)": [_format_lr(r) for _, r in ranked.iterrows()],
        # Beside LR+, because the two answer opposite questions about the same
        # row and a reader deciding whether a negative scan is reassuring
        # should not have to compute one from the other.
        "LR− (95% CI)": [
            _format_lr(r, key="lr_neg", corrected_key="lr_neg_corrected")
            for _, r in ranked.iterrows()
        ],
        # Carried so the report can split the table without re-deriving which
        # side each row belongs to. Dropped before display.
        "origin": (ranked["origin"] if "origin" in ranked.columns
                   else pd.Series(NATIVE, index=ranked.index)),
    }).reset_index(drop=True)


def lr_forest_figure(panel: pd.DataFrame) -> plt.Figure:
    """LR+ per marker with its interval, on a log axis with a line at 1.

    Log scale because a likelihood ratio is a multiplier: 0.5 and 2 are the
    same distance from "says nothing", and a linear axis hides that. Markers
    whose interval crosses the line at 1 are drawn grey on a shaded band.
    Rows sort together by LR+, largest first.
    """
    usable = panel[panel["lr_pos"].notna()] if len(panel) else panel
    if usable is None or usable.empty:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No marker has an estimable likelihood ratio",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ordered = usable.sort_values("lr_pos", ascending=False)
    fig, ax = ps.forest_lr(
        ordered["label"].astype(str).tolist(),
        ordered["lr_pos"].to_numpy(dtype=float),
        ordered["lr_pos_lo"].to_numpy(dtype=float),
        ordered["lr_pos_hi"].to_numpy(dtype=float),
        ref=1.0,
        xlabel="Positive likelihood ratio (95% CI)",
        value_header="LR+ (95% CI)",
        width=ps.FIG_WIDTH_DOUBLE,
    )
    del ax
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
    """The combination figure, drawn on markers instead of cut-points.

    The subtitle lists the criteria by name, which works for the three or four
    cut-points :mod:`combinations` feeds it and does not work here: sixteen
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


TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"


def _write_table(tables: dict, root: Path, stem: str, frame: pd.DataFrame) -> None:
    tables[stem] = frame
    (root / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    format_table_for_csv(frame).to_csv(
        root / TABLES_DIRNAME / f"{stem}.csv", index=False,
    )


def count_score_panel(
    df: pd.DataFrame,
    signs: Sequence[str],
    target: str,
) -> dict[str, pd.DataFrame]:
    """Does counting several signs beat using the best one alone? — NOT YET BUILT.

    The dose-response argument. A radiologist checks each sign in ``signs`` and
    counts how many are present; this asks whether risk climbs with that count.
    The shape of the answer is what makes it defensible: if the signs were all
    restating one underlying thing, risk would jump once and then flatten, so a
    **monotone rise is the evidence that combining helps** and a flat or
    wandering curve is the evidence that it does not.

    ``signs`` is passed in, never mined. Picking the best-scoring combination
    out of the data and then reporting how well it scores is the objection a
    committee raises first, and a pre-specified list is the answer to it. The
    existing ``selection_correction`` stays alongside as the guard for the
    "best single vs best combination" claim, which *is* selected and so has to
    carry its optimism correction and its winner stability.

    Planned outputs, all on one patient set — complete cases across every sign,
    because a count compared across different groups of patients is not a
    comparison:

    ``risk_by_count``   one row per k: n, events, risk with a Wilson interval.
    ``rule_by_count``   "≥ k of n signs" as a rule: Se, Sp, PPV, NPV, LR+, J.
    ``figure``          risk against k, greyscale, points with intervals rather
                        than bars, and an at-risk row under the axis instead of
                        floating labels that collide.

    Two things to settle before this is trusted:

    * **Collinear members double-count.** Tumor volume and max diameter
      correlate at rho ~= 0.92 in this cohort, edema volume and edema index at
      ~= 0.91. A count over all five cut-points rises partly for arithmetic
      reasons, so either the list de-duplicates or the caveat is printed.
    * **Sparse counts.** The k = 0 and k = n bins are often a handful of
      patients; a minimum n has to be set and the dropped bins named, not
      silently omitted.
    """
    raise NotImplementedError(
        "count_score_panel is a placeholder — populate COUNT_SCORE_SIGNS in "
        "the modelling notebook and implement this before calling it."
    )


def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    derived_cols: Collection[str] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> dict[str, pd.DataFrame]:
    """Compute the whole panel and write it to ``output/panel/``.

    Two patient sets on purpose. The aim-1 marker table uses each marker's own
    complete cases, because it ranks markers and each row stands alone. Every
    aim-2 comparison uses the shared set, because a Youden J compared across
    two different groups of patients is not a comparison.

    Nothing here renders. ``report.py`` reads what this writes.
    """
    root = Path(output_root) / "panel"
    tables: dict[str, pd.DataFrame] = {}

    markers = markers_from_diagnostic_accuracy(
        accuracy_table, target=target, exclude=exclude,
    )
    markers = [m for m in markers if m.col in df.columns]
    kept, dropped = usable_markers(df, markers, target)

    hidden = frozenset(derived_cols) if derived_cols else _hidden_parents(output_root)
    origins = classify_origin(
        [m.col for m in kept],
        derivation_sources=_derivation_parents(output_root),
        hidden_parents=hidden,
    )
    panel = marker_panel_table(df, kept, target, origin_by_marker=origins)
    _write_table(tables, root, "01_marker_panel", panel)
    _write_table(tables, root, "02_marker_panel_reading_view",
                 marker_panel_reading_view(panel))

    shared = shared_cohort_frame(df, kept, target)
    _write_table(tables, root, "03_shared_cohort",
                 shared_cohort_audit(df, kept, target, dropped))

    empty = pd.DataFrame()
    if len(kept) >= 2 and not shared.empty:
        correction = selection_correction(
            shared, kept, target, n_boot=n_boot, seed=seed, max_size=max_size,
        )
        counts = count_score(shared, kept, target)
        _write_table(tables, root, "07_count_score", counts)
        _write_table(tables, root, "08_count_thresholds",
                     count_thresholds(shared, kept, target))
        _write_table(tables, root, "09_selection_correction", correction)
        _write_table(tables, root, "12_count_headline",
                     count_headline(counts, k=len(kept)))
    else:
        for stem in ("07_count_score", "08_count_thresholds",
                     "09_selection_correction", "12_count_headline"):
            _write_table(tables, root, stem, empty)
        counts = empty

    fig_dir = root / FIGURES_DIRNAME
    # One forest per family, because the two are corrected separately and a
    # single axis invites the reader to rank a derived cut-point against a
    # native sign as though one q had ordered them both.
    for origin in (NATIVE, DERIVED):
        part = panel[panel["origin"] == origin] if "origin" in panel else panel
        if part.empty:
            continue
        # The forest is scaled and ordered by the rows it is given, so a derived
        # row reaching the native panel would move the axis and the ranking, not
        # just add a line. Checked here because the filter above is the only
        # thing keeping them apart.
        if "origin" in part.columns and not (part["origin"] == origin).all():
            raise ValueError(
                f"The {origin} forest was handed rows from the other family.")
        _fig = lr_forest_figure(part.reset_index(drop=True))
        ps.set_figure_legend(_fig, plain=(
            "One row per finding. The further right, the more it points to a high-grade tumour; a bar touching the line means it tells you nothing."))
        ps.save_figure(_fig, fig_dir / f"lr_forest_{origin}",
                       tight_layout=False, kind="halftone")
    _fig = lr_forest_figure(panel)
    ps.set_figure_legend(_fig, plain=(
        "One row per finding. The further right, the more it points to a high-grade tumour; a bar touching the line means it tells you nothing."))
    ps.save_figure(_fig, fig_dir / "lr_forest", tight_layout=False)
    prevalence = (
        float(shared[target].astype("boolean").mean()) if len(shared) else None
    )
    if len(counts):
        ps.save_figure(count_score_figure(counts, kept, prevalence=prevalence),
                       fig_dir / "count_score")
    else:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM,
                                                      aspect=0.4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Not enough markers for a combination",
                ha="center", va="center", transform=ax.transAxes)
        ps.save_figure(fig, fig_dir / "count_score")

    return tables
