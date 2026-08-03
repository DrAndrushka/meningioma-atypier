"""What the study is: the part the pipeline can answer, and the part only you can.

Most "methods" facts a reviewer asks for are already recorded somewhere in this
repo — the WHO edition sits in the source column header, the inclusion criteria
are the row filters the cleaning notebook applied, the edema index carries the
paper it comes from. Those are read back out of ``output/cleaning/`` here rather
than retyped, so they cannot drift from what the pipeline actually did.

What is left is four questions about how the images were acquired and read.
They are not paperwork: each one decides whether a number in this report means
anything outside this hospital. They are asked in the notebook, carried into
``46_study_facts.csv``, and printed in the report either as an answer or as an
open question.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STATUS_ANSWERED = "answered from the data"
STATUS_OPEN = "needs you"


@dataclass(frozen=True)
class Question:
    """One fact the pipeline cannot derive, with the reason it is needed."""

    key: str
    item: str
    prompt: str
    why: str


# Four, not ten. Each one changes what the numbers mean or whether they
# transfer; ethics numbers, reporting checklists and literature-comparison
# prose were dropped because none of them moves a threshold or its defence.
QUESTIONS: tuple[Question, ...] = (
    Question(
        "histology_reading",
        "Who read the histology, and were they blinded",
        "How many pathologists reported the grade, and whether they saw the MRI.",
        "The grade is the reference standard every number here is measured against. "
        "If it was read with the imaging in view, the design is circular."),
    Question(
        "dwi_acquisition",
        "DWI acquisition",
        "Field strength, scanner, and the b-values the ADC maps were computed from. "
        "Note any scanner or protocol change during accrual.",
        "ADC values are not comparable between b-value schemes. Without them the ADC "
        "cut-point cannot be used by anyone else, or checked by a reviewer."),
    Question(
        "adc_roi",
        "How the ADC ROI was placed",
        "Whole-tumour or single-slice; solid portion only or whole mass; how many "
        "readers, and whether they knew the grade.",
        "Single-slice and whole-tumour ADC differ systematically — the leading "
        "explanation for published ADC cut-points not reproducing."),
    Question(
        "volumetry",
        "How the volumes were measured",
        "Manual, semi-automated or automated; the software and version; on which "
        "sequence; by whom.",
        "Both volume cut-points are quoted in cc against published ones. They are "
        "only comparable if the segmentation is."),
)

QUESTIONS_BY_KEY = {q.key: q for q in QUESTIONS}


# ---------------------------------------------------------------------------
# What the pipeline already knows
# ---------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def pipeline_answers(output_root: Path = Path("output")) -> list[dict[str, str]]:
    """Methods facts read back out of the cleaning run, never retyped.

    Silently returns fewer rows when the cleaning artifacts are absent — a
    threshold run against an old output folder still works, it just has more
    open questions.
    """
    cleaning = Path(output_root) / "cleaning"
    summary = _read(cleaning / "cleaning_summary.csv")
    derivations = _read(cleaning / "derivation_log.csv")
    rows: list[dict[str, str]] = []

    def add(key: str, item: str, answer: str, why: str) -> None:
        if answer:
            rows.append({"key": key, "item": item, "answer": answer,
                         "status": STATUS_ANSWERED, "why": why})

    # ---- outcome definition, from the derivation that built it -----------
    if not derivations.empty and "derivation" in derivations.columns:
        d = derivations.set_index("derivation")
        if "high_grade" in d.index:
            row = d.loc["high_grade"]
            add("who_edition", "Outcome and WHO edition",
                f"{_text(row.get('reason'))} Rule as applied: "
                f"{_text(row.get('rule'))}.",
                "A high-grade rate cannot be interpreted without the edition that "
                "produced it.")
        if "edema_index" in d.index:
            row = d.loc["edema_index"]
            add("edema_index", "Definition of the edema index",
                "Edema volume (cc) ÷ tumour volume (cc), left missing where tumour "
                f"volume is zero. Source: {_text(row.get('reason'))}",
                "The index is analysed throughout and is defined nowhere else in "
                "the document.")
        published = [
            f"{_text(r.get('rule'))} — {_text(r.get('reason'))}"
            for _, r in derivations.iterrows()
            if _text(r.get("reason")) and any(
                token in _text(r.get("reason")) for token in ("doi", "20"))
            and _text(r.get("derivation")) not in ("high_grade", "edema_index")
            and _text(r.get("kind")) == "binary"
        ]
        if published:
            add("published_cutpoints", "Published cut-points already in the data",
                "; ".join(published),
                "These are the external cut-points this cohort validates rather "
                "than re-estimates.")

    # ---- how the cohort was assembled, from the filters that assembled it -
    if not summary.empty and {"step", "detail", "n_rows"} <= set(summary.columns):
        raw = summary[summary["step"] == "raw_data"]
        final = summary[summary["step"] == "final"]
        drops = summary[summary["step"] == "drop_rows"]
        if not drops.empty and not raw.empty and not final.empty:
            # Only the filters that actually removed someone. The summary holds
            # no "rows removed" column, so it is the drop in n_rows; a rule that
            # excluded nobody (the year filter here) is noise in a methods
            # sentence and stays in cleaning_summary.csv for anyone auditing.
            applied, previous = [], int(raw["n_rows"].iloc[0])
            for _, r in drops.iterrows():
                n_rows = int(r["n_rows"])
                if n_rows < previous and _text(r.get("criterion")):
                    applied.append(f"{_text(r.get('criterion'))} → {n_rows}")
                previous = n_rows
            steps = "; ".join(applied)
            add("inclusion", "Inclusion and exclusion, as applied",
                f"{int(raw['n_rows'].iloc[0])} records in the source file → "
                f"{int(final['n_rows'].iloc[0])} analysed. {steps}.",
                "Selection bias cannot be judged without the rule that built the "
                "cohort. This is the STARD flow diagram in one line.")
    # The accrual window is deliberately not repeated here — the report's
    # header card already carries it, derived from the same cohort summary.
    return rows


def _text(value: object) -> str:
    """CSV round-tripping turns an empty cell into NaN, and str(nan) is "nan"."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# The table the report reads
# ---------------------------------------------------------------------------
def study_facts_table(
    answers: dict[str, str] | None = None,
    output_root: Path = Path("output"),
) -> pd.DataFrame:
    """Answered facts first, then whatever is still open.

    ``answers`` is the notebook's fill-in dict, keyed by ``Question.key``. A
    blank value keeps the question open rather than printing an empty answer,
    which is the whole point: a plausible-looking blank is worse than a visible
    hole.
    """
    answers = {k: _text(v) for k, v in (answers or {}).items()}
    unknown = set(answers) - set(QUESTIONS_BY_KEY)
    if unknown:
        raise KeyError(
            f"STUDY_FACTS has keys that are not questions: {sorted(unknown)} — "
            f"expected any of {sorted(QUESTIONS_BY_KEY)}")

    rows = pipeline_answers(output_root)
    for question in QUESTIONS:
        given = answers.get(question.key, "")
        rows.append({
            "key": question.key,
            "item": question.item,
            "answer": given or question.prompt,
            "status": STATUS_ANSWERED if given else STATUS_OPEN,
            "why": question.why,
        })
    return pd.DataFrame(rows, columns=["key", "item", "answer", "status", "why"])


def open_questions(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty or "status" not in facts.columns:
        return pd.DataFrame()
    return facts[facts["status"] == STATUS_OPEN]


def literature_sources_table(
    cutoffs: dict[str, list[tuple[float, str]]],
    links: dict[str, str] | None = None,
) -> pd.DataFrame:
    """The published cut-points being validated, with somewhere to read them.

    A cut-point taken from a paper is only checkable if the paper is findable,
    so the link travels with the number instead of living in someone's inbox.
    """
    links = links or {}
    rows = []
    for column, entries in (cutoffs or {}).items():
        for cutoff, source in entries:
            rows.append({
                "column": column,
                "cutoff": float(cutoff),
                "source": source,
                "link": links.get(source, ""),
            })
    return pd.DataFrame(rows, columns=["column", "cutoff", "source", "link"])
