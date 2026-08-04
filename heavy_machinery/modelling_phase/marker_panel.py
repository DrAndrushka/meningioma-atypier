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
from model_calculator import predict_from_artifact
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
    """
    sides = [
        ("best single", ("single",)),
        ("best combination", ("and", "or", "count")),
    ]
    rows: list[dict] = []
    for label, kinds in sides:
        result = cb.bootstrap_best_rule(
            df, markers, target, n_boot=n_boot, seed=seed,
            max_size=max_size, kinds=kinds,
        )
        rows.append({
            "side": label,
            "best_rule": result.get("best_rule", ""),
            "J_apparent": result.get("J_apparent", np.nan),
            "optimism": result.get("optimism", np.nan),
            "J_corrected": result.get("J_corrected", np.nan),
            "winner_stability": result.get("winner_stability", np.nan),
            "n_bootstrap": result.get("n_bootstrap", 0),
        })
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
    "best_single_J_corrected", "note",
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
) -> pd.DataFrame:
    """Each multivariable model against the best single marker, on one patient set.

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
        "Note": [str(r.get("note") or "") for _, r in table.iterrows()],
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


def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    artifacts: dict[str, dict] | None = None,
    draws: Sequence[pd.DataFrame] = (),
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

    Two bootstrap budgets on purpose too. ``n_boot`` buys the two corrections on
    the shared set — run once each, so 500 resamples costs about four minutes.
    ``draw_n_boot`` buys the one inside :func:`imputation_stability`, which runs
    **per MICE draw**: at twenty draws, forwarding a shared-set budget of 500
    turns a four-minute correction into an eighty-five-minute one for a
    stability check whose answer is a reproduction rate. They are separate
    parameters because they buy different things.

    Nothing here renders. ``report.py`` reads what this writes.
    """
    root = Path(output_root) / "panel"
    tables: dict[str, pd.DataFrame] = {}

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
        models = model_vs_single(shared, artifacts or {}, target, correction)
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
    ps.save_figure(lr_forest_figure(panel), fig_dir / "lr_forest.svg")
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
