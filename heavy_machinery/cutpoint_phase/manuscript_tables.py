"""The supplemental tables, each answering one question the main text raises.

**S1 — can you even locate the cut-point?** Three independent ways of asking how
firmly the data pin down a single value: do different selection rules agree, does
resampling the patients move it, does filling in the missing scans move it. Each
is expressed against the measurement's own IQR so five different units can be
compared on one page, and each shows the raw span as well as the ratio, because
a ratio alone tells a reader nothing about where the cut-point actually sits.

**S2 — what does the yes/no rule cost?** Discrimination as a number against
discrimination as a rule, on the same patients, compared by DeLong. A cut-point
that keeps almost everything is nearly free; one that loses a quarter is buying
convenience with information.

**S3 — is edema a sign or a dose?** A third of this cohort has no edema at all,
so one odds ratio across everyone answers two questions at once. Split apart,
the two can disagree, and where they do a cut-point quoted in cm³ is really a
test for presence wearing a number.

**S4 — would following the rule lead to better decisions?** Net benefit against
the two alternatives a clinician already has for free. This is the only table
here whose horizontal axis is a value judgment rather than a measurement, and
the only one that can fail a rule everything else passed.

Each table carries a graded final column, and every grading rule is stated in
the footnote. That is the condition for a verdict column existing at all: a
reader must be able to disagree with the threshold and recompute the verdict
themselves. No column grades a measurement — only whether one specific, named
test was met.

The order matters. S1 asks whether a number can be located; S2 asks what using
it costs; S4 asks whether it is worth using at all. A measurement that fails S1
has nothing for S2 to price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ajnr_format import (BLANK, EN_DASH, fmt_est, fmt_est_ci, fmt_p, fmt_pct,
                         fmt_ratio, fmt_signed, fmt_span, fmt_value, join_names,
                         yes_no)
from measurements import MEASUREMENTS_BY_COL

# The thresholds each graded column applies. Named here, printed in the
# footnote, and used by the code — one definition, three appearances.
SPREAD_LIMIT = 0.50        # × IQR, across the optimum-seeking selection rules
BOOTSTRAP_LIMIT = 1.00     # × IQR, width of the resampling interval
RETAINED_FLOOR = 0.90      # share of discrimination surviving the cut
LOSS_ALPHA = 0.05          # DeLong P below this is a real loss, not noise
BEND_ALPHA = 0.05          # spline likelihood-ratio test
BREAK_ALPHA = 0.05         # Davies-corrected test for the breakpoint

T1_TITLE = ("Table 1. Evidence for a threshold effect in each quantitative "
            "measurement")
S1_TITLE = ("Table S1. Stability of the derived cut-point for each "
            "quantitative measurement")
S2_TITLE = ("Table S2. Discrimination retained after dichotomisation at the "
            "derived cut-point")
S3_TITLE = ("Table S3. Presence versus amount for the two edema measurements")
S4_TITLE = ("Table S4. Decision-curve analysis: clinical usefulness of each "
            "cut-point across plausible threshold probabilities")

# Table 1 is nine columns wide. Left unspecified they would divide evenly and
# the intervals would wrap to three lines each; the names, the intervals and the
# conclusion get the room, the P values need very little.
T1_WIDTHS = (0.95, 0.30, 0.95, 0.50, 0.55, 0.95, 0.47, 0.43, 1.30)

# The criteria Table 1 applies, in the order a reader meets them, each with the
# name its failure carries. Ordered, not scored: a measurement is only asked the
# next question if it passed this one, because every criterion after the first
# is about the *shape* of an association the measurement may already have failed
# to show. So the table reports where a measurement stopped, not how many boxes
# it ticked, and one vocabulary serves the table and the report's cards alike.
# Each failure is named for what the data *is*, not for which test object.
# "Risk never bends" describes a test; "linear, no curve" describes the
# measurement, and only the second tells a radiologist what they are looking at.
CRITERIA: tuple[tuple[str, str], ...] = (
    ("the odds ratio excludes 1", "no association with grade"),
    ("risk bends", "linear, no curve"),
    ("the bend survives a change of scale", "near-linear on the other scale"),
    ("the breakpoint survives Davies' correction", "no definite break point"),
    ("the break pays for the parameters it adds", "break no better than a line"),
)
CRITERIA_TOTAL = len(CRITERIA)

# How a row that met everything opens, and how :func:`describe_t1` recognises
# one: named once rather than matched by eye.
MET_ALL = f"{CRITERIA_TOTAL}/{CRITERIA_TOTAL}"
T1_CONCLUSION = "Conclusion"


def _criteria_flags(dich_row, bend_row, segmented_row) -> list[bool] | None:
    """Whether each criterion in :data:`CRITERIA` was met, in that order.

    ``None`` when the measurement could not be assessed at all: a missing spline
    or segmented fit leaves nothing to grade, and a list of ``False`` would claim
    the tests were run and failed.
    """
    if bend_row is None or segmented_row is None:
        return None
    aic = segmented_row["delta_aic"]
    return [
        (_excludes_one(dich_row["or_per_sd_lo"], dich_row["or_per_sd_hi"])
         if dich_row is not None else False),
        bool(bend_row["bent_clinical"]),
        bool(bend_row["scales_agree"]),
        bool(segmented_row["breakpoint_supported"]),
        bool(np.isfinite(aic) and aic < 0),
    ]


def _reached(flags) -> int:
    """How many criteria in a row were met before the first that was not."""
    if flags is None:
        return -1
    return next((i for i, met in enumerate(flags) if not met), CRITERIA_TOTAL)


def _failure_phrase(index: int, bend_row) -> str:
    """What a failed criterion means for the measurement.

    Criterion 3 is the one that cannot be named in advance. ``scales_agree`` is
    symmetric — it fires whenever the two scales disagree, in either direction —
    so which scale is the flat one is read off the row rather than assumed.
    """
    # ``Series.get`` rather than indexing: an older nonlinearity table has no
    # ``bent_log`` column, and the generic phrase is still true without it.
    bent_log = None if bend_row is None else bend_row.get("bent_log")
    if index == 2 and bent_log is not None:
        return f"near-linear on {'log' if not bool(bent_log) else 'clinical'} scale"
    return CRITERIA[index][1]


def _conclusion(flags, measurement, segmented_row, bend_row) -> str:
    """Where the evidence stopped, in the few words a reader will actually read.

    The column this replaces held Yes or No, which told a reader nothing about
    *which* of five quite different failures they were looking at, nor whether a
    larger study would change it. It reads how many criteria were met, the one
    that stopped the row, and what that failure means for the measurement.
    """
    if flags is None:
        return BLANK
    stopped = _reached(flags)
    if stopped < CRITERIA_TOTAL:
        return (f"{stopped}/{CRITERIA_TOTAL} \u2014 fails at {stopped + 1}: "
                f"{_failure_phrase(stopped, bend_row)}")
    breakpoint_ = fmt_value(segmented_row["breakpoint"], measurement.decimals)
    return f"{MET_ALL} \u2014 threshold at {breakpoint_}"


def _by_col(table: pd.DataFrame) -> dict:
    """Whole-cohort rows keyed by column, so a lookup cannot pick a stratum."""
    if table is None or table.empty or "col" not in table.columns:
        return {}
    sub = table
    if "stratum" in sub.columns:
        sub = sub[sub["stratum"] == "all"]
    return {row["col"]: row for _, row in sub.iterrows()}


def table_one(eligible: pd.DataFrame, *, dichotomy: pd.DataFrame,
              nonlinearity: pd.DataFrame,
              segmented: pd.DataFrame) -> pd.DataFrame:
    """Table 1: does a threshold exist at all, for each measurement.

    Five things have to line up before a threshold claim is safe, and the
    columns are numbered so a reader meets them in the order they matter. Each
    is a gate rather than a box to tick: the next question is only asked of a
    measurement that passed the last one, and the final column names the gate a
    measurement stopped at.

    *Does the measurement do anything as a number* — the odds ratio per 1 SD.
    Placed first, and deliberately, because everything after it is about the
    *shape* of an association, and shape is not evidence when the association
    itself is null. A measurement can bend convincingly and be worthless.

    *Does risk bend* — the spline test, on the clinical scale and again on the
    log scale. Reported separately rather than combined: whether a curve looks
    bent depends on the axis it is drawn against, and a claim that survives only
    one scale is a claim about the axis.

    *Where does it break, and is the break real* — the segmented model, which
    unlike a spline estimates the breakpoint as a parameter with an interval.
    Its P value carries Davies' correction, because the breakpoint does not
    exist under the null and the uncorrected test finds breaks in straight lines.

    *Is the extra complexity worth it* — ΔAIC, charging for both parameters the
    segmented model adds. Negative favours a break; between −2 and 0 is not a
    meaningful improvement whatever the P value says.
    """
    # The nonlinearity table rather than the bend table: only the former
    # carries the log-scale repeat of the test, and the two scales have to be
    # shown side by side for the column to mean anything.
    dich, bnd, seg = _by_col(dichotomy), _by_col(nonlinearity), _by_col(segmented)
    rows, reached = [], []
    for _, row in eligible.iterrows():
        col = row["col"]
        m = MEASUREMENTS_BY_COL[col]
        d, b, s = dich.get(col), bnd.get(col), seg.get(col)

        flags = _criteria_flags(d, b, s)

        rows.append({
            "Measurement": m.label,
            "n": int(d["n"]) if d is not None else (int(s["n"]) if s is not None
                                                    else 0),
            "1. Associated with grade? OR per 1 SD (95% CI)": (
                fmt_est_ci(d["or_per_sd"], d["or_per_sd_lo"], d["or_per_sd_hi"])
                if d is not None else BLANK),
            "2. Bend on risk curve? Spline P": (fmt_p(b["lr_p"]) if b is not None
                                        else BLANK),
            "3. Survives log-scale? Spline P, log": (
                fmt_p(b["lr_p_log"]) if b is not None else BLANK),
            "Cutpoint location Breakpoint (95% CI)": (
                f"{fmt_value(s['breakpoint'], m.decimals)} "
                f"({fmt_span(s['ci_lo'], s['ci_hi'], m.decimals)})"
                if s is not None and np.isfinite(s["breakpoint"]) else BLANK),
            "4. Better than chance? Davies P": (fmt_p(s["davies_p"])
                                               if s is not None else BLANK),
            "5. Beats a straight line? ΔAIC < 0": (fmt_signed(s["delta_aic"], 1)
                                              if s is not None else BLANK),
            T1_CONCLUSION: _conclusion(flags, m, s, b),
        })
        # Ordered by how far each measurement got, so the numbered criteria in
        # the last column descend down the page and a reader can see at a glance
        # how many survived each one. Ties keep the alphabetical order the rest
        # of the phase sorts by.
        reached.append((-_reached(flags), m.label))
    frame = pd.DataFrame(rows).set_index("Measurement")
    return frame.iloc[[i for i, _ in sorted(enumerate(reached),
                                            key=lambda pair: pair[1])]]


def _excludes_one(lo, hi) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 1.0 or hi < 1.0))


def _better_as(dich_row) -> str:
    """Which form discriminates better — decided by the test, not the estimate.

    The two AUCs are compared by DeLong on the same patients, so "better" means
    a difference the study can actually detect. Where the comparison is not
    significant the honest answer is that neither form is better, and saying so
    is more useful than ranking two numbers a hundredth apart.
    """
    if dich_row is None:
        return ""
    # ``Series.get`` yields None for an absent column, and np.isfinite raises on
    # None rather than returning False — so coerce before testing.
    loss = pd.to_numeric(dich_row.get("auc_loss"), errors="coerce")
    p = pd.to_numeric(dich_row.get("auc_loss_p"), errors="coerce")
    if not (np.isfinite(loss) and np.isfinite(p)) or p >= LOSS_ALPHA:
        return ", with no measurable difference between the two"
    return (", better as a number" if loss > 0 else ", better as a yes/no")


def _works_line(dich_row, presence_row) -> str:
    """Whether the measurement works as a number, as a yes/no, or only one.

    The distinction a threshold paper most often loses. A measurement can be
    strongly predictive as a rule and null as a value — meaning the rule is
    detecting a *sign* rather than measuring a *dose* — and quoting a cut-point
    in cm³ without saying so invites the reader to believe the amount matters.
    """
    if dich_row is None:
        return ""
    as_number = _excludes_one(dich_row["or_per_sd_lo"], dich_row["or_per_sd_hi"])
    as_rule = _excludes_one(dich_row["or_binary_lo"], dich_row["or_binary_hi"])

    if as_number and as_rule:
        return f"Works both ways{_better_as(dich_row)}."
    if as_rule and not as_number:
        # Where the split is available, say which half of the yes/no is doing
        # the work — presence or amount — rather than leaving the reader to
        # infer it.
        if presence_row is not None and presence_row["presence_matters"] \
                and not presence_row["amount_matters"]:
            return ("Works as a yes/no only — having any predicts, how much "
                    "does not.")
        return "Works as a yes/no only, not as a number."
    if as_number and not as_rule:
        return "Works as a number only, not as a yes/no."
    return "Predicts neither as a number nor as a yes/no."


def _verdict_detail(bend_row, segmented_row) -> str:
    """The numbers behind a verdict, compressed to one line for a card."""
    parts = []
    if bend_row is not None:
        parts.append(f"bend P {fmt_p(bend_row['lr_p'])} clinical, "
                     f"{fmt_p(bend_row['lr_p_log'])} log")
    if segmented_row is not None:
        parts.append(f"break P {fmt_p(segmented_row['davies_p'])}")
        if np.isfinite(segmented_row["delta_aic"]):
            # fmt_signed, as in the table: a hyphen here reads as a dash between
            # the two numbers either side of it on the same line.
            parts.append(f"ΔAIC {fmt_signed(segmented_row['delta_aic'], 1)}")
    return " · ".join(parts)


def threshold_verdicts(eligible: pd.DataFrame, *, dichotomy: pd.DataFrame,
                       nonlinearity: pd.DataFrame, segmented: pd.DataFrame,
                       wobble: pd.DataFrame | None = None,
                       presence: pd.DataFrame | None = None) -> list[dict]:
    """One entry per measurement: supported or not, and the reason either way.

    The reason matters more than the verdict. "Not supported" covers four quite
    different failures — no bend at all, a bend that exists only on one scale, a
    breakpoint that does not survive correction, and a break too small to pay
    for itself — and a reader told only "No" cannot tell which they are looking
    at, nor whether a larger study would change it.

    ``works`` carries the opposite case: a measurement that meets every
    condition here while being null as a continuous value. The conditions test
    the *shape* of an association, and shape is not evidence when there is no
    association to shape.
    """
    dich, bnd, seg = _by_col(dichotomy), _by_col(nonlinearity), _by_col(segmented)
    wob = _by_col(wobble) if wobble is not None else {}
    pres = _by_col(presence) if presence is not None else {}
    entries = []
    for _, row in eligible.iterrows():
        col = row["col"]
        m = MEASUREMENTS_BY_COL[col]
        d, b, s, w = dich.get(col), bnd.get(col), seg.get(col), wob.get(col)
        pa = pres.get(col)

        # Named from the same :data:`CRITERIA` the table grades with, in the
        # same order, so a card and its row cannot describe a failure
        # differently. The most fundamental failure leads: a measurement with
        # no association has nothing for the later tests to be about.
        flags = _criteria_flags(d, b, s)
        failures = ([] if flags is None else
                    [_failure_phrase(i, b)
                     for i, met in enumerate(flags) if not met])

        supported = flags is not None and not failures
        breakpoint_text = (
            f"{m.op} {fmt_value(s['breakpoint'], m.decimals)} {m.unit}".strip()
            if s is not None and np.isfinite(s["breakpoint"]) else BLANK)
        entries.append({
            "measurement": m.label,
            "col": col,
            "supported": supported,
            "breakpoint": breakpoint_text,
            "breakpoint_ci": (fmt_span(s["ci_lo"], s["ci_hi"], m.decimals)
                              if s is not None else BLANK),
            "cutpoint": (f"{m.op} {fmt_value(w['cutpoint'], m.decimals)} "
                         f"{m.unit}".strip() if w is not None else BLANK),
            "odds_ratio": (fmt_est_ci(d["or_per_sd"], d["or_per_sd_lo"],
                                      d["or_per_sd_hi"])
                           if d is not None else BLANK),
            # Three short lines rather than a paragraph: what was met, how the
            # measurement works, and the numbers. A card that opens with four P
            # values is a card nobody reads.
            "criteria_met": CRITERIA_TOTAL - len(failures),
            "criteria_total": CRITERIA_TOTAL,
            "criteria_line": (f"All {CRITERIA_TOTAL} criteria met" if not failures
                              else f"{CRITERIA_TOTAL - len(failures)} of "
                                   f"{CRITERIA_TOTAL} criteria met"),
            "failures": failures,
            "reason": ("" if not failures else
                       "Why: " + "; ".join(failures) + "."),
            "works": _works_line(d, pa),
            "detail": _verdict_detail(b, s),
        })
    return sorted(entries, key=lambda e: (not e["supported"], e["measurement"]))


def t1_footnote() -> str:
    """What the columns are and what passes each of them, in under 100 words.

    Every threshold is read from the constants the code grades with, so the note
    cannot drift from the table above it. The definitions this note no longer
    carries — Davies' correction, what ΔAIC charges for — belong in Methods; a
    note long enough to hold them is a note nobody reads.
    """
    bend = f"{BEND_ALPHA:.2f}".lstrip("0")
    brk = f"{BREAK_ALPHA:.2f}".lstrip("0")
    return (
        "Note:—The numbered columns are sequential criteria, each evaluated "
        "only if the preceding one was met; the final column reports where "
        f"evaluation stopped. Criteria in order: CI excludes 1; P < {bend}; "
        f"P < {bend}; Davies P < {brk}; ΔAIC < 0. Spline P is the "
        "likelihood-ratio test of a 3-knot restricted cubic spline against the "
        "linear model, repeated after log transformation. Odds ratios are per "
        "1 SD; tumor volume, edema volume and edema index were log-transformed."
    )


def describe_t1(table: pd.DataFrame) -> str:
    """One line: which measurements support a threshold, and which do not."""
    if table.empty:
        return "No measurement could be assessed."
    clears = table[T1_CONCLUSION].astype(str).str.startswith(MET_ALL)
    supported = table[clears].index.tolist()
    if not supported:
        return (f"No measurement meets all {CRITERIA_TOTAL} criteria for a "
                "threshold effect.")
    return (f"Threshold effect supported: {', '.join(supported)}. "
            f"Not supported: {', '.join(table[~clears].index)}.")


def supplemental_s1(eligible: pd.DataFrame, *, agreement: pd.DataFrame,
                    wobble: pd.DataFrame, imputation: pd.DataFrame
                    ) -> pd.DataFrame:
    """S1: can the cut-point be located, and how firmly."""
    agree, wob, imput = _by_col(agreement), _by_col(wobble), _by_col(imputation)
    rows = []
    for _, row in eligible.iterrows():
        col = row["col"]
        m = MEASUREMENTS_BY_COL[col]
        a, w, i = agree.get(col), wob.get(col), imput.get(col)

        spread = a["spread_vs_iqr"] if a is not None else np.nan
        width = w["stability_ratio"] if w is not None else np.nan
        diverges = bool(i["diverges"]) if i is not None else None

        # Every one of the three has to hold. They fail independently — a
        # cut-point can be reproducible across imputations and still move
        # halfway across the cohort when the patients are resampled.
        checks = [np.isfinite(spread) and spread <= SPREAD_LIMIT,
                  np.isfinite(width) and width <= BOOTSTRAP_LIMIT,
                  diverges is False]
        locatable = None if diverges is None else all(checks)

        rows.append({
            "Measurement": m.label,
            "Published cut-point": (
                f"{m.op} {fmt_value(w['cutpoint'], m.decimals)} {m.unit}".strip()
                if w is not None else BLANK),
            "Spread across selection methods": (
                f"{fmt_span(a['cutoff_min'], a['cutoff_max'], m.decimals)} "
                f"({fmt_ratio(spread)} IQR)" if a is not None else BLANK),
            "Bootstrap 95% interval": (
                f"{fmt_span(w['ci_lo'], w['ci_hi'], m.decimals)} "
                f"({fmt_ratio(width)} IQR)" if w is not None else BLANK),
            "Range across 20 imputations": (
                fmt_span(i["draw_min"], i["draw_max"], m.decimals)
                if i is not None else BLANK),
            "Cut-point locatable": yes_no(locatable),
        })
    return pd.DataFrame(rows).set_index("Measurement")


def supplemental_s2(eligible: pd.DataFrame, *, dichotomy: pd.DataFrame
                    ) -> pd.DataFrame:
    """S2: what dichotomising costs, against the number it replaces."""
    dich = _by_col(dichotomy)
    rows = []
    for _, row in eligible.iterrows():
        col = row["col"]
        m = MEASUREMENTS_BY_COL[col]
        d = dich.get(col)
        if d is None:
            rows.append({"Measurement": m.label})
            continue

        retained, p = d["information_retained"], d["auc_loss_p"]
        # Both conditions, because they catch different failures: a large loss
        # that the sample is too small to call significant, and a small loss
        # that is nonetheless real.
        acceptable = (None if not np.isfinite(retained) else
                      bool(retained >= RETAINED_FLOOR
                           and (not np.isfinite(p) or p >= LOSS_ALPHA)))
        rows.append({
            "Measurement": m.label,
            "Cut-point": f"{m.op} {fmt_value(d['cutoff'], m.decimals)}",
            "n": int(d["n"]),
            "AUC as a number": fmt_est_ci(d["auc_continuous"],
                                          d["auc_continuous_lo"],
                                          d["auc_continuous_hi"]),
            "AUC as yes/no": fmt_est_ci(d["auc_binary"], d["auc_binary_lo"],
                                        d["auc_binary_hi"]),
            "Discrimination retained": fmt_pct(retained),
            "DeLong P": fmt_p(p),
            "Acceptable loss": yes_no(acceptable),
        })
    return pd.DataFrame(rows).set_index("Measurement")


def supplemental_s3(presence: pd.DataFrame) -> pd.DataFrame:
    """S3: for the measurements with a pile at zero, presence against amount.

    A third of this cohort has no edema. Asked of everyone, "does edema predict
    high grade?" answers two questions at once — whether *having* it matters and
    whether *how much* matters — and a single odds ratio cannot separate them.
    Split apart, the two can disagree, and where they do, a cut-point quoted in
    cm³ is really a test for presence wearing a number.
    """
    if presence is None or presence.empty:
        return pd.DataFrame()
    rows = []
    for _, r in presence.iterrows():
        rows.append({
            "Measurement": r["measurement"],
            "n": int(r["n"]),
            "None present": (f"{int(r['events_absent'])}/{int(r['n_absent'])} "
                             f"({fmt_pct(r['rate_absent'], 1)})"),
            "Some present": (f"{int(r['events_present'])}/{int(r['n_present'])} "
                             f"({fmt_pct(r['rate_present'], 1)})"),
            "OR for presence (95% CI)": fmt_est_ci(
                r["presence_or"], r["presence_lo"], r["presence_hi"]),
            "P": fmt_p(r["presence_p"]),
            "OR per 1 SD among those with it (95% CI)": fmt_est_ci(
                r["amount_or"], r["amount_lo"], r["amount_hi"]),
            "P ": fmt_p(r["amount_p"]),
            "OR per 1 SD, whole cohort (95% CI)": fmt_est_ci(
                r["overall_or"], r["overall_lo"], r["overall_hi"]),
        })
    return pd.DataFrame(rows).set_index("Measurement")


def s3_footnote() -> str:
    """What the split is, and why the third column can contradict the first two."""
    return (
        "Note:—Both measurements are zero in about a third of patients, so an "
        "odds ratio computed across the whole cohort answers two questions at "
        "once. The two are separated here. Presence compares patients with any "
        "edema against those with none. Amount is the odds ratio per 1 SD "
        "computed only among patients who have edema, so presence is held "
        "constant and the estimate reflects how much rather than whether. Both "
        "are from univariable logistic regression with Wald intervals; edema "
        "volume and edema index were log-transformed before standardization. "
        "The final column is the odds ratio across the whole cohort and is "
        "shown for comparison only: where presence and amount disagree, it is "
        "an average of the two and describes neither. Percentages are the "
        "proportion of high-grade tumors in each group.")


def describe_s3(presence: pd.DataFrame) -> str:
    """One line naming any measurement that is a sign rather than a dose."""
    if presence is None or presence.empty:
        return "No zero-inflated measurement to split."
    sign_only = presence[presence["presence_matters"].astype(bool)
                         & ~presence["amount_matters"].astype(bool)]
    both = presence[presence["presence_matters"].astype(bool)
                    & presence["amount_matters"].astype(bool)]
    parts = []
    if not sign_only.empty:
        named = "; ".join(
            f"{r['measurement']} (presence OR "
            f"{fmt_est_ci(r['presence_or'], r['presence_lo'], r['presence_hi'])}, "
            f"amount OR "
            f"{fmt_est_ci(r['amount_or'], r['amount_lo'], r['amount_hi'])})"
            for _, r in sign_only.iterrows())
        parts.append(f"Presence predicts but amount does not — {named}.")
    if not both.empty:
        named = "; ".join(f"{r['measurement']}" for _, r in both.iterrows())
        parts.append(f"Both presence and amount predict — {named}.")
    return " ".join(parts) if parts else "Neither presence nor amount predicts."


def supplemental_s4(decision: pd.DataFrame) -> pd.DataFrame:
    """S4: is the rule worth following, and would the number do better?

    Every other table in this set scores the cut-point as a measurement. This
    one scores it as a decision, which is a different and harder test: a rule
    can discriminate above chance and still never beat "call every tumor high
    grade" at any threshold a clinician would actually hold.

    The net benefit column is optimism-corrected, so it is not the number a
    reader would get by applying the formula to the published 2×2 table. The
    apparent value is shown beside it for exactly that reason — the gap between
    them is what selection on these patients was worth.
    """
    if decision is None or decision.empty:
        return pd.DataFrame()
    rows = []
    for _, r in decision.iterrows():
        m = MEASUREMENTS_BY_COL[r["col"]]
        useful = (f"{fmt_pct(r['rule_useful_from'], 0)}"
                  f"{EN_DASH}{fmt_pct(r['rule_useful_to'], 0)}"
                  if np.isfinite(r["rule_useful_from"]) else "Never")
        rows.append({
            "Measurement": r["measurement"],
            "n": int(r["n"]),
            "Cut-point": f"{m.op} {fmt_value(r['cutpoint'], m.decimals)}",
            "Net benefit at base rate, apparent": fmt_est(
                r["nb_rule_apparent"], 3),
            "Net benefit at base rate, corrected": fmt_est(r["nb_rule"], 3),
            "Useful range of threshold probabilities": useful,
            "Patients not flagged per 100, vs treating all": fmt_est(
                r["reduction_per_100"], 0),
            "Measurement does better as a number": yes_no(
                bool(r["number_beats_rule"])),
        })
    return pd.DataFrame(rows).set_index("Measurement")


def s4_footnote() -> str:
    """What net benefit is, what the threshold probability means, and the limits."""
    return (
        "Note:—Net benefit is TP/n − (FP/n) × t/(1 − t), where t is the "
        "threshold probability: the probability of high-grade histology at "
        "which a clinician would change management. It is a value judgment "
        "rather than an estimate, so net benefit is computed across the whole "
        "range of t a reader might hold rather than at one chosen value. "
        "Treating no patient as high grade has a net benefit of 0 at every t; "
        "treating every patient has a net benefit of prevalence − (1 − "
        "prevalence) × t/(1 − t). The useful range is the widest contiguous "
        "span of t over which the cut-point exceeds both. Net benefit at the "
        "base rate is evaluated at t equal to the observed prevalence of "
        "high-grade tumors in the patients analyzed for that measurement. Both "
        "the cut-point and the continuous comparator were corrected for "
        "optimism over 1000 patient-level bootstrap resamples (seed 20260801) "
        "by re-deriving the cut-point and refitting the logistic model within "
        "each resample and scoring both on the patients that resample omitted; "
        "the apparent column is the uncorrected value. Patients not flagged "
        "per 100 converts the gain over treating all into patients, by "
        "dividing it by t/(1 − t): it is the number of patients per 100 who "
        "would not be flagged, with no additional high-grade tumors missed. "
        "The final column compares the cut-point against the same measurement "
        "kept continuous and turned into a risk by univariable logistic "
        "regression, both corrected identically, at the base rate.")


def describe_s4(table: pd.DataFrame) -> str:
    """One line: which rules earn their place in a decision, and which do not."""
    if table is None or table.empty:
        return "No measurement could be put on a decision curve."
    useful = table[table["rule_beats_alternatives"].astype(bool)]
    parts = []
    if useful.empty:
        parts.append("No cut-point beats treating every patient as high grade "
                     "at any threshold probability a clinician would hold.")
    else:
        named = join_names(
            f"{r['measurement']} between {fmt_pct(r['rule_useful_from'], 0)} "
            f"and {fmt_pct(r['rule_useful_to'], 0)}"
            for _, r in useful.iterrows())
        parts.append(f"Each cut-point is worth acting on over a band of "
                     f"threshold probabilities — {named}.")
    better = table[table["number_beats_rule"].astype(bool)]
    if not better.empty:
        parts.append(f"At the base rate the measurement itself does better "
                     f"than its cut-point for "
                     f"{join_names(better['measurement'])}, so the cut-point "
                     f"buys convenience rather than accuracy there.")
    return " ".join(parts)


def s1_footnote() -> str:
    """What each column measures and what earns a Yes."""
    return (
        "Note:—Cut-points were derived by the Youden criterion. Spread across "
        "selection methods is the range of the cut-points chosen by the Youden, "
        "closest-to-perfect, sensitivity-equals-specificity, and index-of-union "
        "criteria. The bootstrap interval is the 2.5th to 97.5th percentile of "
        "1000 patient-level resamples with the cut-point re-derived in each "
        "(seed 20260801). Both are also expressed as a multiple of that "
        "measurement's interquartile range, so measurements in different units "
        "can be compared. The imputation range is the smallest and largest "
        "cut-point obtained by re-deriving it in each of 20 multiply imputed "
        f"datasets. Cut-point locatable: Yes when the spread is {SPREAD_LIMIT:.2f} "
        f"× IQR or less, the bootstrap interval is {BOOTSTRAP_LIMIT:.2f} × IQR "
        "or less, and the published cut-point lies within the imputation range; "
        "No when any of the three fails. Continuous measurements were imputed "
        "by predictive mean matching, which fills a gap by copying a value "
        "already observed for that measurement, so the filled-in values pile "
        "up on the observed ones instead of spreading evenly and a cut-point "
        "re-derived in them can only land where the observed data already "
        "had a value. The imputation range therefore checks whether the "
        "missing scans move the cut-point; it is not itself evidence for the "
        "threshold. Denominators differ between measurements because of "
        "missing data.")


def s2_footnote() -> str:
    """What each column measures and what earns a Yes."""
    return (
        "Note:—Each measurement is scored twice on the same patients: as a "
        "continuous value and as a yes/no rule at its cut-point. For a yes/no "
        "rule the AUC equals (sensitivity + specificity) / 2. Discrimination "
        "retained is (AUC as yes/no − 0.50) ÷ (AUC as a number − 0.50); values "
        "above 100% indicate that dichotomising improved discrimination, which "
        "occurs when the association is a step rather than a gradient. The two "
        "AUCs are compared by the DeLong test for correlated curves, which "
        "accounts for both being measured in the same patients. Acceptable "
        f"loss: Yes when at least {RETAINED_FLOOR:.0%} of discrimination is "
        f"retained and the DeLong P is {LOSS_ALPHA:.2f} or greater; No when "
        "either fails. Denominators differ between measurements because of "
        "missing data.")


def describe_s1(table: pd.DataFrame) -> str:
    """One line: which cut-points can be located, and which cannot."""
    if table.empty:
        return "No cut-point could be assessed."
    locatable = table[table["Cut-point locatable"] == "Yes"].index.tolist()
    not_locatable = table[table["Cut-point locatable"] == "No"].index.tolist()
    parts = []
    if locatable:
        parts.append(f"Locatable: {', '.join(locatable)}.")
    if not_locatable:
        parts.append(f"Not locatable: {', '.join(not_locatable)}.")
    return " ".join(parts)


def describe_s2(table: pd.DataFrame) -> str:
    """One line: what dichotomising costs, and where it costs too much."""
    if table.empty:
        return "No cut-point could be priced."
    acceptable = table[table["Acceptable loss"] == "Yes"].index.tolist()
    too_costly = table[table["Acceptable loss"] == "No"].index.tolist()
    parts = []
    if acceptable:
        parts.append(f"Loss acceptable: {', '.join(acceptable)}.")
    if too_costly:
        parts.append(f"Loss too large to accept: {', '.join(too_costly)}.")
    return " ".join(parts)
