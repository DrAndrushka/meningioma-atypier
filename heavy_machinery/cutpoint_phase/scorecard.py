"""The evidence table — every measurement against every criterion, met or not.

A ROC curve cannot show that a threshold exists. It shows how well a measurement
*ranks* patients, and a gradual relationship draws exactly the same curve as a
sharp step. What settles whether a cut-point deserves to be published is a list
of questions, asked of every measurement, each with a stated rule and a number
behind it. That is this table.

Six criteria in two families:

**Is a threshold real?**       5, 6 — does risk change character, and does that
                               conclusion survive changing the scale.
**Is the cut-point solid?**    8, 9, 10, 11 — do different rules agree on it,
                               does it survive resampling, does it survive the
                               scans we never measured, does cutting cost much.

The numbering has gaps, kept deliberately so a criterion cited in a draft keeps
meaning the same thing. What was removed, and where it lives instead:

``1`` discrimination (AUC) and ``2`` the odds ratio per 1 SD — reported for
every measurement in ``report.html``, from ``eda_paper_tables``.
``3`` collinearity and ``4`` significance after adjustment — multivariable
questions, answered in the multivariable analysis rather than here.
``7`` whether the bend sat inside the data — part of assessing the bend itself,
not a separate test of the cut-point.

**This table is now narrow on purpose, and narrower than the evidence.** Every
remaining criterion asks about the *threshold and its cut-point*. Nothing here
asks whether the measurement carries any signal in the first place, whether it
duplicates another, or whether it survives adjustment. Those questions have
answers, and the answers change the ranking — a measurement can meet all six
criteria below while being null as a number and 0.91 correlated with a second
measurement in the same table. Read alongside the multivariable analysis, not
instead of it. The footnote says so.

**No verdict word.** The table reports which criteria are met and the value
behind each one. It does not grade a measurement "strong" or "fragile" — those
words carry an authority the underlying numbers do not.

**Every cell carries its number.** A Yes alone is an assertion; a Yes with
``bend test P 0.009`` next to it is evidence. The long form of the table holds
both, and it is what belongs in Supplemental Data.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd

from measurements import MEASUREMENTS_BY_COL

FAMILY_THRESHOLD = "Is a threshold real?"
FAMILY_CUTPOINT = "Is the cut-point solid?"


class Criterion(NamedTuple):
    """One criterion: its name, the statistic behind it, and how it is graded.

    ``formula`` and ``yes_when``/``no_when`` exist so the footnote can be
    generated from the same object the arithmetic uses. A footnote maintained
    separately from the code drifts away from it within one revision, and the
    reader has no way to tell which of the two is describing the analysis.
    """

    number: int
    key: str
    name: str
    family: str
    question: str
    formula: str
    yes_when: str
    no_when: str
    step: str


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        5, "bends", "Nonlinearity", FAMILY_THRESHOLD,
        "Does risk actually change character somewhere?",
        "likelihood-ratio test of a restricted cubic spline (3 knots at the "
        "10th, 50th, 90th percentiles) against the linear model; "
        "χ² = 2(ℓ_spline − ℓ_linear), df = 1",
        "P below .05",
        "P at or above .05", "4"),
    Criterion(
        6, "scale_free", "Scale invariance", FAMILY_THRESHOLD,
        "Is that conclusion free of the scale it was tested on?",
        "the same likelihood-ratio test repeated on log1p(x)",
        "clinical and log scales reach the same verdict at P = .05",
        "the two scales disagree", "4"),
    Criterion(
        8, "criteria_agree", "Criterion concordance", FAMILY_CUTPOINT,
        "Do different selection rules land in the same place?",
        "range of the cut-points chosen by Youden, closest-to-perfect, "
        "Se = Sp, and index of union, divided by the measurement's IQR",
        "range at or below 0.50 × IQR",
        "range above 0.50 × IQR", "6"),
    Criterion(
        9, "survives_resampling", "Sampling stability", FAMILY_CUTPOINT,
        "Would it be the same number in a different sample?",
        "width of the 2.5th–97.5th percentile interval from 1000 patient-level "
        "bootstrap resamples, the cut-point re-derived within each, divided by "
        "the measurement's IQR",
        "width at or below 1.00 × IQR",
        "width above 1.00 × IQR", "7"),
    Criterion(
        10, "survives_missingness", "Imputation stability", FAMILY_CUTPOINT,
        "Does it survive the scans that were never measured?",
        "cut-point re-derived independently in each of the 20 MICE imputations; "
        "min–max range across them",
        "the published cut-point lies inside that range",
        "it lies outside the range every imputation produced", "8"),
    Criterion(
        11, "cut_costs_little", "Dichotomisation cost", FAMILY_CUTPOINT,
        "Does cutting throw away little of what the number knew?",
        "(AUC of the yes/no rule − 0.50) ÷ (AUC of the raw number − 0.50), "
        "the two AUCs compared by DeLong on the same patients",
        "0.90 or more retained",
        "below 0.90 retained", "9"),
)

CRITERIA_BY_KEY = {c.key: c for c in CRITERIA}


class ScorecardError(Exception):
    """A criterion could not be evaluated from the tables supplied."""


def _row(table: pd.DataFrame, col: str) -> pd.Series | None:
    """The whole-cohort row for one measurement, or None if it is absent."""
    if table is None or table.empty or "col" not in table.columns:
        return None
    sub = table[table["col"] == col]
    if "stratum" in sub.columns:
        sub = sub[sub["stratum"] == "all"]
    return sub.iloc[0] if len(sub) else None


def _fmt(value: float, decimals: int = 3) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"


def _evaluators() -> dict[str, Callable[[dict, str], tuple[bool, str]]]:
    """One function per criterion: given the tables and a column, met and why.

    Each returns the *value* that decided it, not a restatement of the rule, so
    a reader can check the arithmetic rather than take the tick on trust.
    """

    def bends(t, col):
        r = _row(t["bend"], col)
        if r is None:
            return False, "not scored"
        return bool(r["bend_is_real"]), f"bend test P {r['lr_p']:.3f}"

    def scale_free(t, col):
        r = _row(t["bend"], col)
        if r is None:
            return False, "not scored"
        return bool(r["scales_agree"]), ("clinical and log scales agree"
                                         if r["scales_agree"]
                                         else "clinical and log scales disagree")

    def criteria_agree(t, col):
        r = _row(t["agreement"], col)
        if r is None:
            return False, "not scored"
        return (bool(r["spread_vs_iqr"] <= 0.5),
                f"rules span {r['cutoff_min']:g}-{r['cutoff_max']:g}, "
                f"{r['spread_vs_iqr']:.2f} of the IQR")

    def survives_resampling(t, col):
        r = _row(t["wobble"], col)
        if r is None:
            return False, "not scored"
        return (bool(r["stability_ratio"] <= 1.0),
                f"95% CI {r['ci_lo']:g}-{r['ci_hi']:g}, "
                f"{r['stability_ratio']:.2f} of the IQR")

    def survives_missingness(t, col):
        r = _row(t["imputation"], col)
        if r is None:
            return False, "not scored"
        return (not bool(r["diverges"]),
                f"imputations span {r['draw_min']:g}-{r['draw_max']:g}")

    def cut_costs_little(t, col):
        r = _row(t["dichotomy"], col)
        if r is None:
            return False, "not scored"
        return (bool(r["information_retained"] >= 0.90),
                f"{r['information_retained']:.0%} of discrimination retained")

    return {"bends": bends, "scale_free": scale_free,
            "criteria_agree": criteria_agree,
            "survives_resampling": survives_resampling,
            "survives_missingness": survives_missingness,
            "cut_costs_little": cut_costs_little}


def scorecard_long(eligible: pd.DataFrame, *, separation: pd.DataFrame,
                   bend: pd.DataFrame, agreement: pd.DataFrame,
                   wobble: pd.DataFrame, imputation: pd.DataFrame,
                   dichotomy: pd.DataFrame, pairs: pd.DataFrame,
                   coefficients: pd.DataFrame,
                   adjustment_model: str = "Five cut-points",
                   criteria: Sequence[Criterion] = CRITERIA) -> pd.DataFrame:
    """One row per measurement per criterion, with the value that decided it.

    The long form is the one that goes in Supplemental Data: a reviewer can read
    down it and check every tick against its number without opening the code.
    """
    tables = {"separation": separation, "bend": bend, "agreement": agreement,
              "wobble": wobble, "imputation": imputation,
              "dichotomy": dichotomy, "pairs": pairs,
              "coefficients": coefficients,
              "adjustment_model": adjustment_model}
    evaluators = _evaluators()
    rows = []
    for _, row in eligible.iterrows():
        col = row["col"]
        m = MEASUREMENTS_BY_COL[col]
        for criterion in criteria:
            if criterion.key not in evaluators:
                raise ScorecardError(f"No evaluator for {criterion.key!r}.")
            met, evidence = evaluators[criterion.key](tables, col)
            rows.append({
                "measurement": m.label, "col": col,
                "criterion_number": criterion.number,
                "criterion": criterion.name,
                "question": criterion.question,
                "family": criterion.family,
                "formula": criterion.formula,
                "graded": f"Yes, {criterion.yes_when}; No, {criterion.no_when}",
                "step": criterion.step,
                "met": bool(met),
                "evidence": evidence,
            })
    return pd.DataFrame(rows)


def scorecard_wide(long: pd.DataFrame, *, mark: str = "Yes",
                   blank: str = "No") -> pd.DataFrame:
    """The manuscript table: criteria down the side, measurements across.

    This orientation rather than its transpose because the criteria are the
    thing being explained. Each row is one named test with one footnote entry,
    and a reader comparing measurements reads across a row — which is the
    comparison the table exists to support.

    Columns are ordered by how many criteria each measurement meets, so the
    best-supported one sits first without the table having to say so.
    """
    if long.empty:
        return pd.DataFrame()
    grid = long.pivot_table(index="criterion_number", columns="measurement",
                            values="met", aggfunc="first")
    counts = long.groupby("measurement")["met"].sum()
    grid = grid[counts.sort_values(ascending=False).index]

    names = {r["criterion_number"]: r["criterion"]
             for _, r in long.drop_duplicates("criterion_number").iterrows()}
    out = grid.replace({True: mark, False: blank})
    out.index = [f"{int(i)}. {names.get(i, '')}".rstrip(". ") for i in out.index]
    out.index.name = "Criterion"
    out.loc["Criteria met"] = [f"{int(counts[c])} of {len(grid.index)}"
                               for c in grid.columns]
    return out


def footnote(criteria: Sequence[Criterion] = CRITERIA) -> str:
    """The `Note:—` block: the statistic behind each row, and how it was graded.

    Generated from the same :class:`Criterion` objects the arithmetic uses, so
    the footnote cannot end up describing an analysis the code no longer
    performs — a drift that otherwise takes one revision to appear and a
    reviewer to notice.
    """
    parts = [f"{c.number}, {c.name}: {c.formula}. Yes, {c.yes_when}; "
             f"No, {c.no_when}" for c in criteria]
    return ("Note:—Rows are numbered criteria, each graded Yes or No against a "
            "stated rule. " + ". ".join(parts) + ". Numbering is not "
            "consecutive: discrimination, the odds ratio per 1 SD, "
            "collinearity, and significance after adjustment are reported in "
            "the main and multivariable analyses and are not repeated here. "
            "This table therefore addresses the cut-point only, and should be "
            "read together with those analyses: a measurement can meet every "
            "criterion below while carrying no information as a continuous "
            "value, since cutting something uninformative costs nothing. All "
            "intervals are 95%. Denominators differ between measurements "
            "because of missing data.")


def describe_scorecard(long: pd.DataFrame) -> str:
    """One line: what met the most criteria, and what each leader failed."""
    if long.empty:
        return "No measurement could be scored."
    counts = long.groupby("measurement")["met"].sum().sort_values(ascending=False)
    total = long["criterion_number"].nunique()
    best = counts.index[0]
    failed = long[(long["measurement"] == best) & ~long["met"]]
    line = f"Most criteria met: {best}, {int(counts.iloc[0])} of {total}."
    if not failed.empty:
        names = "; ".join(f"{r['criterion'].rstrip('?')} ({r['evidence']})"
                          for _, r in failed.iterrows())
        line += f" Not met: {names}."
    else:
        line += " It meets every criterion."
    return line
