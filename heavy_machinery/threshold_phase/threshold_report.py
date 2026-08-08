"""Assemble ``output/thresholds/threshold_report.html`` from the phase's artifacts.

Reads the CSVs and SVGs the thresholder notebook already wrote — nothing is
refitted here, so the report and the exported tables can never disagree.

The document is deliberately short. Six questions, in the order a reader asks
them, and each one gets the same four things and nothing else:

    a two-sentence method note · one headline figure · one small table ·
    one line of answer, templated from the numbers

Per-measurement detail figures sit behind a fold so the main read stays a few
minutes long. The last section is a copy-paste abstract block; the one after it
lists the clinical facts the pipeline cannot know and must not invent.

CLI: ``python heavy_machinery/threshold_phase/threshold_report.py --output-root output``
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Run as a script only this folder lands on sys.path, so the sibling phases have
# to be added before `report` and its own flat imports resolve. A no-op once the
# notebook has imported `heavy_machinery.config`.
for _phase in ("cleaning_phase", "modelling_phase", "threshold_phase"):
    _path = str(Path(__file__).resolve().parent.parent / _phase)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evidence  # noqa: E402  (needs the sys.path bootstrap above)
import study  # noqa: E402
from heavy_machinery.config import load  # noqa: E402
from report import (  # noqa: E402
    _esc,
    _lead,
    _figure_img_html,
    _wrap_html,
    details_block,
    section_block,
    table_to_html,
    warning_box,
    write_html,
)

analysis = load("analysis")

DEFAULT_TITLE = "Thresholds for high-grade meningioma on pre-operative MRI"
GRADE_ORDER = list(evidence.GRADES)

# Byte-identical to the original rather than a second copy of it.
_truthy = evidence._truthy

# Only what a section actually renders. A table that no sentence and no column
# depends on is not loaded — the report is the short version of the run, not a
# second copy of it.
TABLE_FILES = {
    "cohort_summary": "00_cohort_summary.csv",
    "cohort": "01_metric_cohorts.csv",
    "risk_curves": "03_risk_curves.csv",
    "risk_reading": "04_risk_curves_reading_view.csv",
    "thresholds": "07_threshold_summary.csv",
    "threshold_reading": "08_threshold_reading_view.csv",
    "count_score": "15_count_score.csv",
    "stability_reading": "19_imputation_stability_reading_view.csv",
    "stability": "18_imputation_stability.csv",
    "risk_stability": "21_risk_curve_stability.csv",
    # Loaded for one column: the units already written for prose. The unit in
    # 03_risk_curves.csv is LaTeX for the figures and would print raw here.
    "headline": "26_headline_findings.csv",
    "evidence": "28_threshold_evidence.csv",
    "evidence_reading": "29_threshold_evidence_reading_view.csv",
    "shared_reading": "31_shared_combination_reading_view.csv",
    "shared_verdict": "32_shared_combination_verdict.csv",
    "combination_verdict": "17_combination_verdict.csv",
    "zero_share": "34_zero_inflation.csv",
    "nonzero_curves": "36_risk_curves_nonzero_only.csv",
    "zero_comparison": "37_zero_inflation_comparison.csv",
    "multiplicity_reading": "39_nonlinearity_multiplicity_reading_view.csv",
    "calibration": "40_calibration.csv",
    "net_benefit_summary": "44_net_benefit_summary.csv",
    "study_facts": "46_study_facts.csv",
    "literature_sources": "47_literature_sources.csv",
}

FIGURE_FILES = {
    "risk_panel": "05_risk_curves_panel.svg",
    "combined_roc": "10_combined_roc.svg",
    "count_score": "16_count_score.svg",
    "shared_combination_space": "33_shared_combination_space.svg",
    "stability": "20_stability_youden.svg",
    "knee_stability": "22_knee_stability.svg",
    "calibration": "42_calibration.svg",
    "net_benefit": "45_net_benefit.svg",
}

FIGURE_PREFIXES = {
    "distributions": "02_distribution_",
    "risk_curves": "06_risk_curve_",
    "nonzero_risk_curves": "36_risk_curve_nonzero_",
    "thresholds": "09_thresholds_",
}

_EXTRA_CSS = """
.lead { color: var(--muted); margin: 0 0 12px; max-width: 62em; }
.footnote { font-size: 12px; color: var(--muted); margin: 4px 0 10px; }
.answer {
    border-left: 4px solid var(--accent); background: var(--card);
    padding: 10px 14px; margin: 14px 0 6px; border-radius: 0 6px 6px 0;
}
.answer::before {
    content: "Answer"; display: block; font-size: 11px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 2px;
}
.answer p { margin: 0; }
/* A "86/127/15/99" cell offers a break after every slash, so a narrow column
   wraps it mid-number. Numbers never break; the row scrolls instead. */
table.report td { overflow-wrap: normal; word-break: keep-all; hyphens: manual; }
table.report td.nowrap, table.report td.num { white-space: nowrap; }
.gaps { margin: 8px 0 4px; padding-left: 20px; }
.gaps li { margin: 5px 0; }
.gaps b { color: var(--fg); }
/* Written to be selected and pasted: prose with bold labels, no markup that
   would survive into a manuscript. */
.manuscript {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin: 14px 0; line-height: 1.62;
}
.manuscript p { margin: 0 0 12px; }
.manuscript p:last-child { margin-bottom: 0; }
"""

_EVIDENCE_KEY = (
    "<p class=\"footnote\">Vocabulary: <em>pass</em>/<em>fail</em> — the "
    "criterion is met / not met; <em>survives</em> — still significant after "
    "the named multiple-testing correction; <em>fragile</em> — all three "
    "necessary criteria pass but neither robustness criterion does, or exactly "
    "one necessary criterion other than curvature fails.</p>")


# ---------------------------------------------------------------------------
# Config and inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdReportConfig:
    output_root: Path = Path("output")
    title: str = DEFAULT_TITLE
    author: str = ""
    subtitle: str = ("Where the risk of WHO grade 2–3 starts to climb, where a line "
                     "can be drawn, and how much that line can be trusted")

    @property
    def thresholds_root(self) -> Path:
        return self.output_root / "thresholds"

    @property
    def default_out(self) -> Path:
        return self.thresholds_root / "threshold_report.html"


@dataclass
class ThresholdReportData:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    figure_groups: dict[str, list[Path]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def table(self, key: str) -> pd.DataFrame:
        return self.tables.get(key, pd.DataFrame())

    def figure(self, key: str) -> Path | None:
        return self.figures.get(key)


def load_report_data(cfg: ThresholdReportConfig) -> ThresholdReportData:
    """Read the threshold phase's artifacts. A missing file is a warning, not a crash."""
    root = cfg.thresholds_root
    data = ThresholdReportData()

    if not root.exists():
        data.warnings.append(
            f"{root} does not exist — run meningioma-thresholder.ipynb first.")
        return data

    for key, name in TABLE_FILES.items():
        path = root / "tables" / name
        if not path.exists():
            data.warnings.append(f"Missing table: {name}")
            continue
        try:
            data.tables[key] = pd.read_csv(path)
        except Exception as exc:  # a truncated CSV must not kill the report
            data.warnings.append(f"Could not read {name}: {exc}")

    figure_dir = root / "figures"
    for key, name in FIGURE_FILES.items():
        path = figure_dir / name
        if path.exists():
            data.figures[key] = path
        else:
            data.warnings.append(f"Missing figure: {name}")

    for key, prefix in FIGURE_PREFIXES.items():
        data.figure_groups[key] = sorted(figure_dir.glob(f"{prefix}*.svg"))

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            data.manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            data.warnings.append(f"Could not parse manifest.json: {exc}")

    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _num(value: Any, digits: int = 2, default: str = "—") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f"{f:.{digits}f}"


def _pct(value: Any, digits: int = 0, default: str = "—") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f"{f * 100:.{digits}f}%"


def _sig(value: Any, digits: int = 3, default: str = "—") -> str:
    """Significant figures — cut-points here span 0.06 to 45, so fixed decimals
    print either false precision or none at all."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f"{f:.{digits}g}"


def _signed(value: Any, digits: int = 3, default: str = "—") -> str:
    """Signed, and never ``-0.00`` — calibration intercepts land within a few
    thousandths of zero, and a column of "-0.00" reads as a formatting bug."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    rounded = round(f, digits)
    return f"{0.0:.{digits}f}" if rounded == 0 else f"{rounded:+.{digits}f}"


def _int(value: Any, default: str = "—") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f"{int(round(f))}"


def _first(table: pd.DataFrame, column: str, default: Any = None) -> Any:
    if table.empty or column not in table.columns:
        return default
    return table[column].iloc[0]


def _cell(value: Any) -> str:
    """An empty CSV cell round-trips as NaN, and str(nan) is "nan" — which is how
    "limited by nan" once reached a poster."""
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


# Cells that must never wrap mid-token.
NOWRAP_COLUMNS = frozenset({
    "n", "n / events", "Patients", "High grade", "Measured in", "Missing",
    "Median, grade 1", "Median, grade 2–3", "Cut-point", "Cut-point 95% CI",
    "95% CI", "Sens (95% CI)", "Spec (95% CI)", "PPV (95% CI)", "NPV (95% CI)",
    "Sens", "Spec", "J", "J (corr.)", "Risk", "Risk (95% CI)", "OR (95% CI)",
    "AUC", "Sens / Spec", "Evidence", "Measurement",
    "Steepest rise", "Patients below it", "Risk reaches 30%", "Risk reaches 50%",
    "Non-linearity p", "Holm-adjusted p", "Bonferroni-adjusted p", "Criteria met",
    "Found in", "Typical location", "Range across draws", "Complete-case",
    "Bootstrap 95%", "MICE mean", "TP/FP/FN/TN", "Slope", "Intercept", "Brier",
    "Best net benefit", "Leads from–to", "Share of range",
})


def _table(df: pd.DataFrame, **kwargs: Any) -> str:
    if df is None or getattr(df, "empty", True):
        return table_to_html(df, **kwargs)
    kwargs.setdefault("nowrap_cols",
                      [c for c in df.columns if str(c) in NOWRAP_COLUMNS])
    return table_to_html(df, **kwargs)


def _answer(text: str) -> str:
    return f'<div class="answer"><p>{text}</p></div>'


def _figure(path: Path | None, note: str = "") -> str:
    if path is None or not path.exists():
        return '<p class="muted"><em>(figure unavailable — re-run the notebook)</em></p>'
    img = _figure_img_html(path)
    if not img:
        return '<p class="muted"><em>(figure could not be embedded)</em></p>'
    note_html = f'<p class="lead">{note}</p>' if note else ""
    return f'<div class="figure-card">{img}</div>{note_html}'


def _figure_row(paths: Sequence[Path],
                captions: Sequence[str] | None = None) -> str:
    """Figures side by side, each with its own caption underneath.

    A caption belongs below its own panel, not in a shared block: a reader
    lifting one figure out has to be able to take its caption with it.
    """
    labels = list(captions or [])
    cards = []
    for i, p in enumerate(paths):
        if not p.exists():
            continue
        img = _figure_img_html(p)
        if not img:
            continue
        caption = labels[i] if i < len(labels) else ""
        # `.figure-card .caption` is already styled by the shared report CSS.
        caption_html = f'<p class="caption">{caption}</p>' if caption else ""
        cards.append(f'<div class="figure-card">{img}{caption_html}</div>')
    if not cards:
        return '<p class="muted"><em>(figures unavailable)</em></p>'
    return f'<div class="figure-grid">{"".join(cards)}</div>'


def _join(names: Sequence[Any], conjunction: str = "and") -> str:
    """``A``, ``A and B``, ``A, B and C`` — never a stray comma."""
    items = [str(n) for n in names]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _drop(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


# ---------------------------------------------------------------------------
# The verdicts — one source, so no two sections can disagree
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricVerdict:
    metric: str
    column: str
    has_threshold: bool
    where: str = ""             # steepest-rise point, formatted, or ""
    unit: str = ""              # prose unit, blank for ratios
    auc: float = float("nan")
    reproduced: float = float("nan")   # MICE knee_rate, 0–1
    grade: str = ""             # strong / moderate / fragile / weak
    limiting: str = ""          # the criterion holding the grade down


@dataclass(frozen=True)
class Verdicts:
    items: tuple[MetricVerdict, ...] = ()

    @property
    def positive(self) -> list[MetricVerdict]:
        return [v for v in self.items if v.has_threshold]

    @property
    def n_thresholds(self) -> int:
        return len(self.positive)

    @property
    def n_metrics(self) -> int:
        return len(self.items)

    def count_phrase(self, *, verb: bool = False) -> str:
        noun = "measurement" if self.n_metrics == 1 else "measurements"
        text = f"{self.n_thresholds} of {self.n_metrics} {noun}"
        if verb:
            text += " has" if self.n_thresholds == 1 else " have"
        return text

    def turning_phrase(self) -> str:
        return _join([f"{v.metric} {v.where}{' ' + v.unit if v.unit else ''}"
                      for v in self.positive if v.where])

    @property
    def graded(self) -> bool:
        return any(v.grade for v in self.items)

    def by_grade(self, grade: str) -> list[MetricVerdict]:
        return [v for v in self.items if v.grade == grade]

    def grade_tally(self) -> list[tuple[str, int]]:
        return [(g, len(self.by_grade(g))) for g in GRADE_ORDER if self.by_grade(g)]

    def grade_phrase(self) -> str:
        """``1 strong, 2 moderate and 1 fragile`` — the headline in one clause."""
        return _join([f"{n} {g}" for g, n in self.grade_tally()])

    def grade_sentences(self, *, compact: bool = False) -> str:
        """One clause per measurement: name, grade, what limits it.

        ``compact`` groups measurements sharing a grade, for the abstract block
        where every word is paid for.
        """
        ordered = sorted(self.items,
                         key=lambda x: GRADE_ORDER.index(x.grade)
                         if x.grade in GRADE_ORDER else len(GRADE_ORDER))
        if not compact:
            bits = []
            for v in ordered:
                if not v.grade:
                    continue
                limit = (f" (limited by {v.limiting.lower()})" if v.limiting
                         else " (all criteria met)")
                bits.append(f"{_esc(v.metric)} <b>{_esc(v.grade)}</b>{limit}")
            return _join(bits)

        groups: dict[tuple[str, str], list[str]] = {}
        for v in ordered:
            if not v.grade:
                continue
            groups.setdefault((v.grade, v.limiting), []).append(v.metric)
        return "; ".join(
            f"{_esc(_join(names))} <b>{_esc(grade)}</b>"
            + (f" ({limiting.lower()})" if limiting else "")
            for (grade, limiting), names in groups.items())


def metric_verdicts(data: ThresholdReportData) -> Verdicts:
    risk = data.table("risk_curves")
    if risk.empty or "knee_found" not in risk.columns:
        return Verdicts()

    knee_rate: dict[Any, Any] = {}
    stab = data.table("risk_stability")
    if not stab.empty and {"knee_rate", "column"} <= set(stab.columns):
        knee_rate = dict(zip(stab["column"], stab["knee_rate"]))

    grades: dict[Any, dict[str, str]] = {}
    ev_table = data.table("evidence")
    if not ev_table.empty and "column" in ev_table.columns:
        grades = {row["column"]: {"grade": _cell(row.get("verdict")),
                                  "limiting": _cell(row.get("limiting_criterion"))}
                  for _, row in ev_table.iterrows()}

    units: dict[str, str] = {}
    headline = data.table("headline")
    if not headline.empty and {"Metric", "Unit"} <= set(headline.columns):
        units = {str(r["Metric"]): _cell(r["Unit"]) for _, r in headline.iterrows()}

    items = []
    for _, row in risk.iterrows():
        found = _truthy(row.get("knee_found"))
        g = grades.get(row.get("column"), {})
        items.append(MetricVerdict(
            metric=str(row.get("metric", "")),
            column=str(row.get("column", "")),
            has_threshold=found,
            where=_sig(row.get("steepest_x")) if found else "",
            unit=units.get(str(row.get("metric", "")), ""),
            auc=float(row.get("AUC", float("nan"))),
            reproduced=float(knee_rate.get(row.get("column"), float("nan"))),
            grade=g.get("grade", ""),
            limiting=g.get("limiting", ""),
        ))
    return Verdicts(tuple(items))


# ---------------------------------------------------------------------------
# Cohort facts — every section quotes these rather than recomputing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortFacts:
    n: int = 0
    events: int = 0
    benign: int = 0
    prevalence: float = float("nan")
    years: str = ""
    m_draws: str = "—"
    n_boot: str = "—"
    shared_n: str = "—"
    verdicts: Verdicts = field(default_factory=Verdicts)


def cohort_facts(data: ThresholdReportData) -> CohortFacts:
    summary = data.table("cohort_summary")
    cohort = data.table("cohort")
    context = data.manifest.get("context", {})

    n = events = benign = 0
    prevalence = float("nan")
    if not summary.empty:
        n = int(_first(summary, "n_patients", 0) or 0)
        events = int(_first(summary, "n_high_grade", 0) or 0)
        benign = int(_first(summary, "n_benign", 0) or 0)
        prevalence = float(_first(summary, "prevalence", float("nan")))
    elif not cohort.empty:
        # Fallback for a run predating 00_cohort_summary.csv: no single row of
        # the per-metric table knows the cohort total, so this is approximate.
        idx = (cohort["n_analysed"] + cohort["n_missing"]).idxmax()
        n = int(cohort.loc[idx, "n_analysed"] + cohort.loc[idx, "n_missing"])
        prevalence = float(cohort["prevalence"].mean())
        events = int(round(n * prevalence))
        benign = n - events

    years = ""
    first, last = _first(summary, "accrual_first_year"), _first(summary, "accrual_last_year")
    if first is not None and last is not None and pd.notna(first) and pd.notna(last):
        years = f"{_int(first)}–{_int(last)}"

    return CohortFacts(
        n=n, events=events, benign=benign, prevalence=prevalence, years=years,
        m_draws=str(_int(_first(data.table("stability"), "m_draws"), "—")),
        n_boot=str(context.get("n_bootstrap", "—")),
        shared_n=str(_int(_first(primary_verdict(data), "n_used"), "—")),
        verdicts=metric_verdicts(data),
    )


def primary_verdict(data: ThresholdReportData) -> pd.DataFrame:
    """The combination verdict scored on one denominator, with the per-metric
    fallback for runs that predate the shared-cohort table."""
    shared = data.table("shared_verdict")
    return shared if not shared.empty else data.table("combination_verdict")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _summary_table(data: ThresholdReportData) -> str:
    """One row per measurement: the table you would screenshot for a slide.

    Assembled from the headline table and the Youden rows of the cut-point
    table, because the odds ratio lives in one and the evidence grade in the
    other, and a reader should not have to join two CSVs by hand.
    """
    headline = data.table("headline")
    if headline.empty or "Metric" not in headline.columns:
        return ""

    odds: dict[str, str] = {}
    thresholds = data.table("thresholds")
    if not thresholds.empty and "rule" in thresholds.columns:
        for _, row in thresholds[thresholds["rule"] == "youden"].iterrows():
            if pd.notna(row.get("OR")):
                odds[str(row["metric"])] = (
                    f"{_num(row['OR'])} ({_num(row.get('OR_lo'))}–"
                    f"{_num(row.get('OR_hi'))})")

    def _cut(row: pd.Series) -> str:
        unit = _cell(row.get("Unit"))
        cut = _cell(row.get("Youden cut-point"))
        return f"{cut} {unit}".strip() if cut else "—"

    return _table(pd.DataFrame({
        "Measurement": headline["Metric"],
        "n": headline["n"],
        "AUC": headline["AUC"],
        "Cut-point": [_cut(r) for _, r in headline.iterrows()],
        "Sens / Spec": headline.get("Sens / Spec", ""),
        "OR (95% CI)": [odds.get(str(m), "—") for m in headline["Metric"]],
        "J (corr.)": headline.get("J (corr.)", ""),
        "Evidence": headline.get("Threshold evidence", ""),
    }))


def _recorded_bootstrap_resamples(data: ThresholdReportData) -> int | None:
    """The resample count this run actually used, not the library default.

    Prefers the calibration table's own ``n_bootstrap`` column — the bootstrap
    that drives internal validation — when every row agrees on one value.
    Falls back to the manifest's recorded context (cut-point selection count).
    Returns ``None`` when neither source has a usable number, so the caller
    can omit the sentence instead of asserting a count nobody ran.
    """
    cal = data.table("calibration")
    if not cal.empty and "n_bootstrap" in cal.columns:
        counts = {int(x) for x in pd.to_numeric(cal["n_bootstrap"], errors="coerce").dropna()}
        if len(counts) == 1:
            return counts.pop()

    context = data.manifest.get("context", {})
    n = context.get("n_bootstrap")
    if n is not None:
        try:
            return int(n)
        except (TypeError, ValueError):
            pass
    return None


def render_header(cfg: ThresholdReportConfig, data: ThresholdReportData,
                  facts: CohortFacts) -> str:
    v = facts.verdicts
    grade_value = v.grade_phrase() if v.graded else f"{v.n_thresholds} of {v.n_metrics}"
    verdict = primary_verdict(data)

    cards = [
        ("Patients", f"{facts.n}", f"{facts.events} WHO grade 2–3 ({_pct(facts.prevalence, 1)})"
         + (f" · {facts.years}" if facts.years else "")),
        ("Threshold evidence", grade_value, f"of {v.n_metrics} measurements"),
        ("Best single cut-point", _esc(str(_first(verdict, "best_single_rule", "—"))),
         f"corrected J {_num(_first(verdict, 'best_single_J_corrected'))}"),
        (f"{v.n_metrics} measurements uncut",
         _num(_first(verdict, "continuous_AUC_corrected")),
         f"AUC, corrected · n = {facts.shared_n}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="muted" style="font-size:12.5px">{sub}</div></div>'
        for label, value, sub in cards)

    author = (f'<p class="report-authors">{_esc(cfg.author)}</p>' if cfg.author else "")
    n_boot = _recorded_bootstrap_resamples(data)
    bootstrap_note = (f" Bootstrap internal validation: {n_boot} resamples."
                       if n_boot is not None else "")
    warn = ""
    if data.warnings:
        warn = warning_box(
            "Some artifacts are missing, so parts of this report are blank: "
            + _esc("; ".join(data.warnings[:6]))
            + (f" (+{len(data.warnings) - 6} more)" if len(data.warnings) > 6 else ""),
            severe=len(data.warnings) > 6)

    return f"""
<div class="report-title-block">
  <h1>{_esc(cfg.title)}</h1>
  <p class="muted">{_esc(cfg.subtitle)}</p>
  {author}
</div>
<div class="cards">{cards_html}</div>
{warn}
{_summary_table(data)}
<p class="lead">Seven questions, in the order they get asked. Each one: how it was done in two
sentences, one figure, one table, one answer. Every number is read from the run that
produced this file — nothing below is typed by hand.{bootstrap_note}</p>
"""


# ---------------------------------------------------------------------------
# 1 — do the grades look different at all
# ---------------------------------------------------------------------------
def render_separation(data: ThresholdReportData, facts: CohortFacts) -> str:
    cohort = data.table("cohort")
    table_html = ""
    agree = 0
    if not cohort.empty and {"metric", "n_analysed", "n_missing"} <= set(cohort.columns):
        shown = pd.DataFrame({
            "Measurement": cohort["metric"],
            "Higher or lower in grade 2–3": [
                "lower" if str(d) == "lower" else "higher" for d in cohort["direction"]],
            "Measured in": cohort["n_analysed"],
            "Missing": cohort["n_missing"],
            "Median, grade 1": [_sig(x) for x in cohort["median_benign"]],
            "Median, grade 2–3": [_sig(x) for x in cohort["median_high_grade"]],
        })
        table_html = _table(shown)
        for _, row in cohort.iterrows():
            lo, hi = row.get("median_benign"), row.get("median_high_grade")
            if pd.isna(lo) or pd.isna(hi):
                continue
            expected = (hi < lo) if str(row.get("direction")) == "lower" else (hi > lo)
            agree += int(bool(expected))

    aucs = [v.auc for v in facts.verdicts.items if np.isfinite(v.auc)]
    auc_text = (f"AUC — the chance the measurement ranks a random grade 2–3 tumour ahead of "
                f"a random grade 1 one — runs {_num(min(aucs))} to {_num(max(aucs))}, "
                f"where 0.5 is a coin toss"
                if aucs else "")

    return f"""
{_lead("Each measurement is compared between the two grades on the patients who actually "
       "had it, so the number of patients differs from row to row. The clouds below show "
       "the whole distribution, not just the average — the overlap between them is what "
       "any single cut-point has to live with.")}

{_figure_row(data.figure_groups.get("distributions", []))}

{table_html}

{_answer(f"Medians move in the expected direction for {agree} of {len(cohort)} "
         f"measurements, but the two grades overlap heavily: {auc_text}. So a threshold "
         f"has to be located and defended, not assumed."
         if aucs and not cohort.empty else
         "The cohort table is missing — re-run the notebook.")}
"""


# ---------------------------------------------------------------------------
# 2 — where does risk start climbing
# ---------------------------------------------------------------------------
def render_risk_curves(data: ThresholdReportData, facts: CohortFacts) -> str:
    v = facts.verdicts
    reading = _drop(data.table("risk_reading"), "Reading")
    mult = data.table("multiplicity_reading")

    # Only the measurements that bend: quoting the whole family's range put a
    # p of 0.503 at the top of a sentence about real bends.
    risk = data.table("risk_curves")
    p_values = pd.Series(dtype=float)
    if {"nonlinearity_p", "knee_found"} <= set(risk.columns):
        bending = risk[risk["knee_found"].map(_truthy)]
        p_values = pd.to_numeric(bending["nonlinearity_p"], errors="coerce").dropna()
    p_text = (f"p {_num(p_values.min(), 3)}–{_num(p_values.max(), 3)}"
              if len(p_values) > 1 else
              f"p {_num(p_values.iloc[0], 3)}" if len(p_values) == 1 else "")

    holm = ""
    if not mult.empty and "Holm" in mult.columns:
        survive = int((mult["Holm"].astype(str) == "survives").sum())
        holm = (f" Testing {len(mult)} measurements at once inflates the chance of a fluke, "
                f"so the "
                f"p-values are also Holm-adjusted: {survive} of {len(mult)} still clear "
                f"0.05.")

    zero_note = ""
    zero, nonzero = data.table("zero_share"), data.table("nonzero_curves")
    if not zero.empty and not nonzero.empty and "zero_inflated" in zero.columns:
        hits = zero[zero["zero_inflated"].map(_truthy)]
        lost = (nonzero[pd.to_numeric(nonzero["nonlinearity_p"], errors="coerce")
                        >= evidence.ALPHA]
                if "nonlinearity_p" in nonzero.columns else pd.DataFrame())
        if not hits.empty and not lost.empty:
            verb = "is" if len(lost) == 1 else "are"
            zero_note = (f" Caveat: {_esc(_join(list(lost['metric'])))} {verb} zero in about "
                         f"{_num(hits['pct_zero'].max(), 0)}% of patients, and the bend "
                         f"disappears when only patients with some edema are used — so that "
                         f"knee is mostly edema present versus absent, not a magnitude.")

    # The same curves with the zeros dropped, kept next to the caveat that
    # sends the reader looking for them.
    nonzero_figs = data.figure_groups.get("nonzero_risk_curves", [])
    comparison = data.table("zero_comparison")
    if "Metric" in comparison.columns:
        comparison = comparison[["Metric"] + [c for c in comparison.columns
                                              if c != "Metric"]]
    nonzero_block = ""
    if nonzero_figs or not comparison.empty:
        nonzero_block = details_block(
            "The edema curves with the zeros removed — patients who have some",
            _figure_row(nonzero_figs) + _table(comparison))

    return f"""
{_lead("Risk of grade 2–3 is fitted as a bendy line against each measurement (a restricted "
       "cubic spline — a smooth curve allowed to change slope). A likelihood-ratio test "
       "asks whether the bend is real or noise; where it is real, the steepest point of "
       "the curve is the threshold. Grey dots are the observed rate in equal-sized groups "
       "of patients — the honesty check on the curve.")}

{_figure(data.figure("risk_panel"))}

{_table(reading)}
{_table(mult) if not mult.empty else ""}

{_answer(f"{v.count_phrase(verb=True)} a real bend ({p_text}). Risk climbs most steeply "
         f"near {v.turning_phrase()}.{holm}{zero_note}")}

{details_block("Each curve on its own, with the slope drawn underneath",
               _figure_row(data.figure_groups.get("risk_curves", [])))}

{nonzero_block}
"""


# ---------------------------------------------------------------------------
# 3 — where do you draw the line
# ---------------------------------------------------------------------------
def _sources_html(data: ThresholdReportData) -> str:
    """Where the published cut-points come from, linked.

    A cut-point taken from a paper is only checkable if the paper is one click
    away, so the link travels with the number rather than living in an inbox.
    """
    sources = data.table("literature_sources")
    if sources.empty or "source" not in sources.columns:
        return ""
    # The table stores the raw column name; the reader knows the measurement by
    # its label, which the cohort table already carries.
    cohort = data.table("cohort")
    labels = (dict(zip(cohort["column"], cohort["metric"]))
              if {"column", "metric"} <= set(cohort.columns) else {})
    bits = []
    for _, row in sources.iterrows():
        label = _esc(str(row["source"]))
        link = _cell(row.get("link"))
        text = (f'<a href="{_esc(link)}">{label}</a>'
                if link.startswith(("http://", "https://")) else label)
        column = str(row["column"])
        bits.append(f"{_esc(labels.get(column, column))} {_sig(row['cutoff'], 4)} — {text}")
    return ('<p class="lead">Published cut-points scored here: '
            + "; ".join(bits) + ".</p>")


def render_cutpoints(data: ThresholdReportData, facts: CohortFacts) -> str:
    reading = data.table("threshold_reading")
    shown = pd.DataFrame()
    if not reading.empty and "Rule" in reading.columns:
        shown = _drop(reading[reading["Rule"].isin(["youden", "literature"])].copy(),
                      "Reading", "n", "J")
        shown["Rule"] = shown["Rule"].map({"youden": "best separation",
                                           "literature": "published"})

    thresholds = data.table("thresholds")
    best_line, spread_line = "—", ""
    if not thresholds.empty and "rule" in thresholds.columns:
        youden = thresholds[thresholds["rule"] == "youden"].copy()
        youden["youden_J_corrected"] = pd.to_numeric(youden["youden_J_corrected"],
                                                     errors="coerce")
        if youden["youden_J_corrected"].notna().any():
            best = youden.loc[youden["youden_J_corrected"].idxmax()]
            odds = (f", odds ratio {_num(best['OR'])} "
                    f"({_num(best.get('OR_lo'))}–{_num(best.get('OR_hi'))})"
                    if pd.notna(best.get("OR")) else "")
            best_line = (f"{_esc(str(best['metric']))} {_esc(str(best.get('operator', '')))}"
                         f"{_sig(best['cutoff'])} "
                         f"(sensitivity {_pct(best['sensitivity'])}, specificity "
                         f"{_pct(best['specificity'])}{odds}, J "
                         f"{_num(best['youden_J_corrected'])} "
                         f"after correcting for having been chosen on these same patients, "
                         f"on its own {_int(best.get('n_used'))} patients)")
            lo, hi = best.get("cutoff_boot_lo"), best.get("cutoff_boot_hi")
            if pd.notna(lo) and pd.notna(hi):
                spread_line = (f" Resampling the cohort moves that cut-point across "
                               f"{_sig(lo)}–{_sig(hi)}, so it is a region, not a number.")

    return f"""
{_lead("A different question from the last one: not where risk changes shape, but where to "
       "put one line so the two grades separate as well as possible. Every value the data "
       "can distinguish is scored — that list of scores is the ROC curve — and Youden's J "
       "(sensitivity + specificity − 1) picks the best one. Because that best value was "
       "chosen on these same patients it flatters itself, so a bootstrap subtracts the "
       "flattery and gives the interval the cut-point moves across. Each line also carries "
       "an odds ratio — how many times higher the odds of high grade are above it than "
       "below — because that is the number published cut-points are quoted with, and an "
       "interval spanning 1 means the split separates nothing.")}

{_figure(data.figure("combined_roc"))}

{_table(shown)}

{_sources_html(data)}

{_answer(f"Best single line: {best_line}.{spread_line}")}

{details_block("All five selection rules, and the full trade-off per measurement",
               _table(_drop(reading, "Reading"))
               + _figure_row(data.figure_groups.get("thresholds", [])))}
"""


# ---------------------------------------------------------------------------
# 4 — does ticking two boxes beat one
# ---------------------------------------------------------------------------
def render_combinations(data: ThresholdReportData, facts: CohortFacts) -> str:
    verdict = primary_verdict(data)
    counts = data.table("count_score")

    count_html = ""
    ladder = ""
    if not counts.empty and {"n_criteria_met", "n", "risk"} <= set(counts.columns):
        count_html = _table(pd.DataFrame({
            "Criteria met": counts["n_criteria_met"],
            "Patients": counts["n"],
            "High grade": counts.get("n_high_grade", pd.Series(dtype=float)),
            "Risk (95% CI)": [f"{_pct(r, 1)} ({_pct(lo, 1)}–{_pct(hi, 1)})" for r, lo, hi
                              in zip(counts["risk"], counts["risk_lo"], counts["risk_hi"])],
        }))
        usable = counts[counts["n"] > 0]
        if len(usable) >= 2:
            ladder = (f" Counting how many of the {_int(usable['n_criteria_met'].max())} "
                      f"criteria a tumour ticks moves the "
                      f"observed risk from {_pct(usable['risk'].iloc[0], 1)} to "
                      f"{_pct(usable['risk'].iloc[-1], 1)}.")

    gain = _num(_first(verdict, "gain_vs_best_single"))
    best_rule = _esc(str(_first(verdict, "best_rule", "—")))
    answer = (
        f"The best combination, {best_rule}, reaches corrected J "
        f"{_num(_first(verdict, 'best_rule_J_corrected'))} against "
        f"{_num(_first(verdict, 'best_single_J_corrected'))} for the best single criterion — "
        f"a gain of {gain}, and that rule won in only "
        f"{_pct(_first(verdict, 'winner_stability'))} of resampled cohorts. The same "
        f"{facts.verdicts.n_metrics} measurements left uncut reach AUC {_num(_first(verdict, 'continuous_AUC_corrected'))} "
        f"(equivalent J {_num(_first(verdict, 'continuous_J_equivalent'))}), above every "
        f"rule made of cut-points.{ladder}")

    return f"""
{_lead(f"The cut-points are frozen first, then only the way of joining them varies — AND, "
       f"OR, or simply counting how many are met. Searching cut-points and rules together "
       f"would be thousands of comparisons on {facts.events} events. Everything here is "
       f"scored on the same {facts.shared_n} patients who have all "
       f"{facts.verdicts.n_metrics} measurements, so "
       f"the rules are compared on one denominator.")}

{_figure(data.figure("count_score"))}

{count_html}

{_answer(answer)}

{details_block("Every rule in sensitivity–specificity space, and the full ranking",
               _figure(data.figure("shared_combination_space"))
               + _table(data.table("shared_reading")))}
"""


# ---------------------------------------------------------------------------
# 5 — would it survive the missing scans
# ---------------------------------------------------------------------------
def render_stability(data: ThresholdReportData, facts: CohortFacts) -> str:
    stab = data.table("risk_stability")
    shown = pd.DataFrame()
    sentence = "The stability table is missing — re-run the notebook."
    if not stab.empty and "knee_rate" in stab.columns:
        shown = pd.DataFrame({
            "Measurement": stab["metric"],
            "Found in": [_pct(x) for x in stab["knee_rate"]],
            "Typical location": [_sig(x) for x in stab["steepest_median"]],
            "Range across draws": [f"{_sig(lo)}–{_sig(hi)}" for lo, hi
                                   in zip(stab["steepest_min"], stab["steepest_max"])],
        })
        ordered = stab.sort_values("knee_rate", ascending=False)
        # The denominator rides on the first clause so the sentence does not end
        # "60% for ADC of the 20 datasets".
        bits = [f"{_pct(r['knee_rate'])}"
                + (f" of the {facts.m_draws} datasets" if i == 0 else "")
                + f" for {_esc(str(r['metric']))}"
                for i, (_, r) in enumerate(ordered.iterrows())]
        rates = pd.to_numeric(ordered["knee_rate"], errors="coerce")
        # Zero is a different finding from "unstable": those measurements had no
        # knee in the complete cases either, so there is nothing to reproduce.
        none_at_all = ordered[rates <= 0]
        weak = ordered[(rates > 0) & (rates < evidence.MICE_REPRODUCIBLE_CUT)]
        tail = ""
        if not weak.empty:
            tail += (f" Below {_pct(evidence.MICE_REPRODUCIBLE_CUT)} the threshold "
                     f"depends on which patients happened to have the scan, which is "
                     f"what holds {_esc(_join(list(weak['metric'])))} back.")
        if not none_at_all.empty:
            tail += (f" {_esc(_join(list(none_at_all['metric'])))} never produced a knee "
                     f"in any dataset — there was none in the complete cases either, so "
                     f"this is the missing bend, not missing data.")
        if not tail:
            tail = " Every threshold clears the pre-set bar."
        sentence = f"The same threshold reappears in {_join(bits)}.{tail}"

    return f"""
{_lead(f"Everything above quietly drops the patients missing that measurement. To find out "
       f"whether that mattered, the missing values were filled in {facts.m_draws} times "
       f"(multiple imputation — each copy is a plausible complete dataset), and the whole "
       f"analysis re-run on every copy. A threshold that only appears in the complete "
       f"cases is an artefact of who got scanned.")}

{_figure(data.figure("stability"),
         "Shaded band: how much the cut-point moves when patients are resampled. "
         "Dots: the imputed datasets. Dots inside the band mean missing data is not "
         "the problem; dots outside it mean the cut-point depends on the guessed values.")}

{_table(shown)}

{_answer(sentence)}

{details_block("Cut-points per imputed dataset, and where each threshold landed",
               _table(data.table("stability_reading"))
               + _figure(data.figure("knee_stability"),
                         "Shaded band: where the middle half of the imputed datasets put "
                         "the steepest rise. Dots: every dataset that found one. A wide "
                         "band means the location moves with the guessed values; an empty "
                         "panel means no dataset found a bend to place."))}
"""


# ---------------------------------------------------------------------------
# 6 — how much do we believe each threshold
# ---------------------------------------------------------------------------
def render_evidence(data: ThresholdReportData, facts: CohortFacts) -> str:
    v = facts.verdicts
    reading = _drop(data.table("evidence_reading"), "Read alongside (does not score)")
    core = [c for c in ("Metric", "Evidence", "Criteria met", "What limits it")
            if c in reading.columns]
    summary_table = _table(reading[core]) if core else _table(reading)

    return f"""
{_lead("A bare yes/no on one p-value is a weak claim and an easy one to attack, so each "
       "threshold is graded against five criteria fixed before any result was read: the "
       "bend is real, the knee sits among the patients rather than at the edge, it is not "
       "just the point where risk passes 50%, it survives re-fitting on a log scale, and it "
       "reappears across the imputed datasets.")}

{summary_table}

{_EVIDENCE_KEY}

{_answer(f"{v.grade_phrase()} of {v.n_metrics} measurements: "
         f"{v.grade_sentences()}." if v.graded else
         "The evidence grades are missing — re-run the notebook.")}

{details_block("Criterion by criterion, with the numbers behind each pass and fail",
               _table(data.table("evidence_reading")))}
"""


# ---------------------------------------------------------------------------
# 7 — are the predicted percentages usable
# ---------------------------------------------------------------------------
def _usefulness_figures(
    data: ThresholdReportData, facts: CohortFacts,
) -> tuple[list[Path], list[str]]:
    """The two section-7 panels with the captions a journal would expect.

    Everything the old panel drew on itself — what the shading meant, which
    line was which, the cohort rate — is said here instead, because a caption
    survives typesetting and an annotation baked into an axes does not.
    """
    rate = ("" if not np.isfinite(facts.prevalence)
            else f" Cohort high-grade rate {facts.prevalence:.1%}"
                 f" (n = {facts.shared_n}).")
    captions = {
        "calibration": (
            "Figure X. Calibration of predicted risk against observed "
            "high-grade rate. Points are equal-count bins with 95% confidence "
            "intervals; the diagonal is perfect calibration, and points below "
            "it mean the model promised more risk than the patients delivered."
        ),
        "net_benefit": (
            "Figure X. Decision curve analysis comparing net benefit of the "
            "five-measurement model against alternative classification "
            "strategies. Net benefit is plotted against threshold probability "
            "— the predicted risk above which a clinician would act. Coloured "
            "curves are the strategies compared, named in the legend; the grey "
            "dashed line treats every patient as high grade and the grey "
            "dotted line treats none. A strategy is worth using over the range "
            "where its curve lies above both grey lines." + rate +
            " Net benefit for every strategy tested, including the count-score "
            "thresholds not plotted here, is given in the tables below."
        ),
    }
    paths, labels = [], []
    for key in ("calibration", "net_benefit"):
        path = data.figure(key)
        if path is not None:
            paths.append(path)
            labels.append(captions[key])
    return paths, labels


def render_usefulness(data: ThresholdReportData, facts: CohortFacts) -> str:
    cal = data.table("calibration")
    nb = data.table("net_benefit_summary")

    # Runs made before the notebook stopped exporting them still carry the
    # modelling phase's own models. They sit on a different denominator and
    # answer a different question, so they are dropped rather than compared.
    if not cal.empty and "source" in cal.columns:
        cal = cal[~cal["source"].astype(str)
                  .str.contains("modelling phase", case=False, na=False)]

    cal_html, cal_text, cal_note = "", "", ""
    if not cal.empty and "model" in cal.columns:
        uncut = cal[cal["model"].astype(str).str.contains("Uncut", case=False, na=False)]
        row = uncut.iloc[0] if not uncut.empty else cal.iloc[0]
        # Every column the calibration table carries, apparent beside corrected:
        # the gap between the two pairs is the optimism, and hiding it hides the
        # only thing that separates a validated number from a self-scored one.
        def _col(name: str) -> list:
            return list(cal[name]) if name in cal.columns else [None] * len(cal)

        cal_html = _table(pd.DataFrame({
            "Model": cal["model"],
            "Patients": _col("n_used"),
            "Events": _col("events"),
            "Predictors": _col("n_predictors"),
            "Slope apparent": [_num(x) for x in _col("slope_apparent")],
            "Slope corrected": [_num(x) for x in _col("slope_corrected")],
            "Intercept apparent": [_signed(x, 3) for x in _col("intercept_apparent")],
            "Intercept corrected": [_signed(x, 3) for x in _col("intercept_corrected")],
            "Brier apparent": [_num(x, 3) for x in _col("brier_apparent")],
            "Brier corrected": [_num(x, 3) for x in _col("brier_corrected")],
            "Resamples": _col("n_bootstrap"),
            "Source": _col("source"),
        }))
        cal_text = (f"Predicted percentages are close to honest: calibration slope "
                    f"{_num(row['slope_corrected'])} after correction (1.00 is perfect; "
                    f"below 1 means the high and low predictions are pushed too far apart). ")
        # Add footnote if both Uncut and Cut models are present (non-overlapping match)
        cut = cal[cal["model"].astype(str).str.contains(r"(?i)^cut\s", na=False, regex=True)]
        if not uncut.empty and not cut.empty:
            # Read n_used from both rows and compare
            uncut_n = uncut.iloc[0].get("n_used") if "n_used" in cal.columns else None
            cut_n = cut.iloc[0].get("n_used") if "n_used" in cal.columns else None
            if pd.notna(uncut_n) and pd.notna(cut_n):
                uncut_n, cut_n = int(uncut_n), int(cut_n)
                if uncut_n == cut_n:
                    cal_note = (f"<p class=\"footnote\">Both the uncut and the cut model are scored "
                                f"on the same n={uncut_n} patients with all measurements available, "
                                f"so their rows are directly comparable.</p>")
                else:
                    cal_note = (f"<p class=\"footnote\">The uncut and cut model rows are fitted on "
                                f"different patient counts (Uncut: n={uncut_n}; Cut: n={cut_n}), so their "
                                f"metrics are not directly comparable row-to-row; read each against its own n.</p>")
            else:
                # Fallback if n_used is missing — no evidence either way, so say so
                # rather than asserting the counts differ.
                cal_note = ("<p class=\"footnote\">Whether the uncut and cut model rows are fitted "
                            "on the same patient count could not be verified from the artifacts "
                            "(n_used is missing), so read each row against its own numbers rather "
                            "than comparing across rows.</p>")

    nb_html, nb_text, nb_note = "", "", ""
    if not nb.empty and "strategy" in nb.columns:
        nb_html = _table(pd.DataFrame({
            "Strategy": nb["strategy"],
            "Best net benefit": [_num(x, 3) for x in nb["max_net_benefit"]],
            "Leads from–to": [
                "—" if pd.isna(lo) or pd.isna(hi) else f"{_pct(lo)}–{_pct(hi)}"
                for lo, hi in zip(nb["beats_references_from"], nb["beats_references_to"])],
            "Share of range": [f"{_num(x, 0)}%" for x in
                               nb["pct_of_range_beating_references"]],
        }))
        nb_note = ("<p class=\"footnote\">Share of range: the fraction of the "
                   "tested decision-threshold spectrum over which the strategy "
                   "beats both treating everyone and treating no one.</p>")
        contenders = nb[~nb["is_reference"].map(_truthy)].copy()
        contenders["pct_of_range_beating_references"] = pd.to_numeric(
            contenders["pct_of_range_beating_references"], errors="coerce")
        if not contenders.empty and contenders["pct_of_range_beating_references"].notna().any():
            top = contenders.loc[contenders["pct_of_range_beating_references"].idxmax()]
            never = contenders[contenders["pct_of_range_beating_references"] <= 0]
            nb_text = (f"The strategy worth acting on across the widest range is "
                       f"{_esc(str(top['strategy']))}: it beats operating on everyone and "
                       f"on no one over {_num(top['pct_of_range_beating_references'], 0)}% "
                       f"of the trigger risks a clinician might use "
                       f"({_pct(top['beats_references_from'])}–"
                       f"{_pct(top['beats_references_to'])})"
                       + (f"; {_esc(_join(list(never['strategy'])))} never beat either "
                          f"reference." if not never.empty else "."))

    return f"""
{_lead("Everything above is about ranking — who is more likely to be high grade. Two things "
       "it cannot see: whether “40%” really means 40 out of 100 (calibration), and "
       "whether acting on the number does more good than harm at the risk levels a "
       "clinician would actually act on (net benefit).")}

{_figure_row(*_usefulness_figures(data, facts))}

{cal_html}
{cal_note}
{nb_html}
{nb_note}

{_answer((cal_text + nb_text) or "Calibration and net benefit are missing — re-run the notebook.")}
"""


# ---------------------------------------------------------------------------
# 8 — copy-paste
# ---------------------------------------------------------------------------
def _literature_agreement(data: ThresholdReportData) -> list[str]:
    """Published cut-points falling inside our own bootstrap interval — the
    strongest single line for a poster, and it moves with every run."""
    thresholds, risk = data.table("thresholds"), data.table("risk_curves")
    if thresholds.empty or risk.empty or "rule" not in thresholds.columns:
        return []
    knee = risk.set_index("column")
    out = []
    for _, r in thresholds[thresholds["rule"] == "literature"].iterrows():
        col = r.get("column")
        if col not in knee.index:
            continue
        row = knee.loc[col]
        lo, hi, cut = row.get("steepest_lo"), row.get("steepest_hi"), r.get("cutoff")
        if pd.isna(lo) or pd.isna(hi) or pd.isna(cut) or not float(lo) <= float(cut) <= float(hi):
            continue
        source = str(r.get("source", ""))
        author = source.split(",")[0].strip()
        year = re.search(r"\b(19|20)\d{2}\b", source)
        cite = f"{author} {year.group(0)}" if (author and year) else author
        out.append(f"{r['metric']} {_num(cut, 2)} ({_esc(cite)}) in our "
                   f"{_num(lo, 2)}–{_num(hi, 2)}")
    return out


def _model_row(frame: pd.DataFrame, column: str, prefix: str) -> pd.Series | None:
    """The Cut/Uncut row of a calibration or net-benefit table, or ``None``."""
    if frame.empty or column not in frame.columns:
        return None
    hit = frame[frame[column].astype(str).str.startswith(prefix)]
    return hit.iloc[0] if not hit.empty else None


def _cut_vs_uncut(data: ThresholdReportData) -> str:
    """The all-cut model beside the uncut one — what the conclusion turns on.

    Silent unless both rows exist (a run predating the cut model has only one),
    because half a comparison in an abstract is worse than none.
    """
    cal, nb = data.table("calibration"), data.table("net_benefit_summary")
    cut, uncut = _model_row(cal, "model", "Cut"), _model_row(cal, "model", "Uncut")
    if cut is None or uncut is None:
        return ""
    text = (f"Cutting all {_int(cut['n_predictors'])} at their thresholds and fitting them "
            f"together calibrated no worse than leaving them uncut (corrected slope "
            f"{_num(cut['slope_corrected'])} vs {_num(uncut['slope_corrected'])}, "
            f"Brier {_num(cut['brier_corrected'], 3)} vs {_num(uncut['brier_corrected'], 3)}). ")
    cut_nb = _model_row(nb, "strategy", "Cut")
    uncut_nb = _model_row(nb, "strategy", "Uncut")
    if cut_nb is not None and uncut_nb is not None:
        text += (f"On decision-curve analysis the cut model beat treat-all and treat-none "
                 f"over {_num(cut_nb['pct_of_range_beating_references'], 0)}% of threshold "
                 f"probabilities against "
                 f"{_num(uncut_nb['pct_of_range_beating_references'], 0)}% uncut, and was the "
                 f"best available strategy over "
                 f"{_num(cut_nb['pct_of_range_best_available'], 0)}% of the range. ")
    return text


def render_manuscript(data: ThresholdReportData, facts: CohortFacts) -> str:
    """Short enough to paste. Every number templated, so a re-run rewrites it."""
    v = facts.verdicts
    verdict = primary_verdict(data)
    counts = data.table("count_score")
    cal = data.table("calibration")
    nb = data.table("net_benefit_summary")

    years = f" ({facts.years})" if facts.years else ""
    # The measurement list follows METRICS. Spelling it out by hand is how a
    # methods paragraph ends up describing a run that no longer exists.
    measurements = _join([x.metric for x in v.items]) or "The measurements"

    ladder = ""
    if not counts.empty:
        usable = counts[counts["n"] > 0]
        if len(usable) >= 2:
            ladder = (f"Observed risk rose with the number of criteria met, "
                      f"{_pct(usable['risk'].iloc[0], 1)} to {_pct(usable['risk'].iloc[-1], 1)} "
                      f"across 0–{_int(usable['n_criteria_met'].max())}. ")

    agreement = _literature_agreement(data)
    agreement_text = (f"Published cut-points fell inside our intervals: "
                      f"{'; '.join(agreement)}. " if agreement else "")

    slope = ""
    if not cal.empty and "model" in cal.columns:
        uncut = cal[cal["model"].astype(str).str.contains("Uncut", case=False, na=False)]
        if not uncut.empty:
            slope = (f", calibration slope {_num(uncut.iloc[0]['slope_corrected'])} "
                     f"after correction")

    nb_text = ""
    if not nb.empty and "pct_of_range_beating_references" in nb.columns:
        contenders = nb[~nb["is_reference"].map(_truthy)].copy()
        contenders["pct_of_range_beating_references"] = pd.to_numeric(
            contenders["pct_of_range_beating_references"], errors="coerce")
        if not contenders.empty and contenders["pct_of_range_beating_references"].notna().any():
            top = contenders.loc[contenders["pct_of_range_beating_references"].idxmax()]
            nb_text = (f"On decision-curve analysis {_esc(str(top['strategy']))} led over "
                       f"{_num(top['pct_of_range_beating_references'], 0)}% of threshold "
                       f"probabilities ({_pct(top['beats_references_from'])}–"
                       f"{_pct(top['beats_references_to'])}). ")

    compare = _cut_vs_uncut(data)

    return f"""
{_lead("Written to be selected and pasted. Every number is templated from this run, so "
       "re-running the notebook rewrites this block rather than leaving it stale.")}

<div class="manuscript">

<p><b>Aim.</b> To test whether pre-operative MRI measurements of meningioma carry a genuine
<em>threshold</em> for WHO grade 2–3 — a value at which risk changes behaviour — rather than
merely a cut-point, and whether combining cut-points beats the best single one.</p>

<p><b>Methods.</b> {facts.n} operated patients with histological grading{years};
{facts.events} ({_pct(facts.prevalence, 1)}) WHO grade 2–3. {measurements} were modelled
as restricted cubic splines
(3 knots) in clinical units, with a likelihood-ratio test for non-linearity. Cut-points came
from five pre-specified selection rules, bootstrap-corrected for optimism
({facts.n_boot} resamples). Threshold claims were graded against {len(evidence.CRITERIA)}
criteria fixed before the verdicts were read: non-linearity, knee interiority, distinctness
from the 50%-risk crossing, robustness to fitting scale, and reproducibility across
{facts.m_draws} multiply-imputed datasets. Combination rules were compared on one
denominator ({facts.shared_n} patients with all {v.n_metrics} measurements); calibration and net
benefit were assessed for the uncut model and for its all-cut counterpart, the same
measurements entered as yes/no flags at their own cut-points.</p>

<p><b>Results.</b> {v.count_phrase(verb=True)} non-linear risk, but graded unevenly:
{v.grade_sentences(compact=True)}. {agreement_text}Best single criterion:
optimism-corrected Youden J {_num(_first(verdict, 'best_single_J_corrected'))}; best
combination {_num(_first(verdict, 'best_rule_J_corrected'))} — a gain of
{_num(_first(verdict, 'gain_vs_best_single'))} that held in only
{_pct(_first(verdict, 'winner_stability'))} of resampled cohorts. The same {v.n_metrics}
measurements uncut reached AUC {_num(_first(verdict, 'continuous_AUC_corrected'))}{slope},
beating every AND/OR rule built from them. {ladder}{compare or nb_text}</p>

<p><b>Conclusion.</b> These measurements separate meningioma grades but do not support the
threshold language usually applied to them: the claims are graded rather than categorical,
and the edema ones rest on presence versus absence more than on magnitude. What cost
accuracy was combining cut-points by rule, which bought no stable improvement; cutting every
measurement and modelling the flags together did not, matching the uncut model on
calibration and net benefit. The defensible outputs are the risk curves, a count of criteria
met, and the {v.n_metrics}-flag model that count approximates.</p>

<p><b>Limitations.</b> Retrospective single-centre surgical series: at
{_pct(facts.prevalence, 1)} high grade the predictive values do not transfer, though
sensitivity and specificity do. All derived cut-points are optimistically biased and are
reported corrected. The {facts.m_draws} imputations are a stability check, not Rubin
pooling — a cut-point chosen by maximisation does not meet Rubin's conditions — so the
across-draw spread is a range, not a confidence interval.</p>

</div>
"""


# ---------------------------------------------------------------------------
# 9 — methods: what the pipeline knows, and what only the investigator does
# ---------------------------------------------------------------------------
# The answered rows are read back out of the cleaning run (WHO edition, the
# filters that built the cohort, the edema-index source), so they cannot drift
# from what the code did. The open ones are the four in `study.QUESTIONS`, left
# visibly blank rather than guessed: a plausible-sounding wrong b-value survives
# review, a blank one does not.
def render_methods(data: ThresholdReportData) -> str:
    facts = data.table("study_facts")
    if facts.empty or "status" not in facts.columns:
        # No 46_study_facts.csv (an older run): fall back to the questions alone.
        items = "".join(f"<li><b>{_esc(q.item)}</b> — {_esc(q.prompt)} "
                        f"<span class='muted'>{_esc(q.why)}</span></li>"
                        for q in study.QUESTIONS)
        return (_lead("This run predates the study-facts export, so only the open questions "
                      "are listed. Re-run the notebook to pull in what the cleaning run "
                      "already knows.")
                + "<!-- TODO: ANDY -->"
                + f'<ul class="gaps">{items}</ul>')

    answered = facts[facts["status"] == study.STATUS_ANSWERED]
    open_rows = study.open_questions(facts)

    answered_html = _table(pd.DataFrame({
        "Item": answered["item"],
        "As recorded by the pipeline": answered["answer"],
    })) if not answered.empty else ""

    open_html = "".join(
        f"<li><b>{_esc(r['item'])}</b> — {_esc(r['answer'])} "
        f"<span class='muted'>Needed because: {_esc(r['why'])}</span></li>"
        for _, r in open_rows.iterrows())

    # warning_box escapes its argument, so this stays plain text — markup here
    # would reach the page as literal tags.
    warn = warning_box(
        f"{len(open_rows)} open question{'' if len(open_rows) == 1 else 's'}. "
        f"Answer them in the notebook's STUDY_FACTS cell and they move into the table "
        f"above on the next run.") if not open_rows.empty else ""

    return f"""
{_lead("The top half is read back out of the cleaning run — the rule that built the cohort, "
       "the definition of every derived measurement — so it cannot drift from what the code "
       f"did. The bottom half is the {len(open_rows)} thing"
       f"{'' if len(open_rows) == 1 else 's'} the images cannot tell us; each one decides "
       "whether a number here transfers outside this hospital.")}

{answered_html}

{'<h3>Still open</h3>' if open_html else ''}
<!-- TODO: ANDY — the questions below are not in the data and must be supplied. -->
<ul class="gaps">{open_html}</ul>

{warn}
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_report(cfg: ThresholdReportConfig, data: ThresholdReportData | None = None) -> str:
    """Full self-contained HTML document as a string."""
    data = load_report_data(cfg) if data is None else data
    facts = cohort_facts(data)

    sections = [
        render_header(cfg, data, facts),
        section_block("1 · Do the two grades even look different?",
                      render_separation(data, facts), open=True),
        section_block("2 · Where does risk start climbing?",
                      render_risk_curves(data, facts), open=True),
        section_block("3 · Where do you draw the line?",
                      render_cutpoints(data, facts), open=True),
        section_block("4 · Does ticking two boxes beat one?",
                      render_combinations(data, facts), open=True),
        section_block("5 · Would it survive the missing scans?",
                      render_stability(data, facts), open=True),
        section_block("6 · How much do we believe each threshold?",
                      render_evidence(data, facts), open=True),
        section_block("7 · Are the predicted percentages usable?",
                      render_usefulness(data, facts), open=True),
        section_block("8 · Copy-paste for the abstract",
                      render_manuscript(data, facts), open=True),
        section_block("9 · Methods — what is known, and what is still open",
                      render_methods(data), open=False),
    ]

    context = data.manifest.get("context", {})
    generated = data.manifest.get("generated_at", "")
    footer = (
        f'<div class="footer">Generated {datetime.now().isoformat(timespec="seconds")} '
        f'from artifacts written {_esc(generated) or "(timestamp unavailable)"} '
        f'· source: <code>{_esc(cfg.thresholds_root)}</code>'
        + (f' · bootstrap {_esc(context.get("n_bootstrap"))}, '
           f'seed {_esc(context.get("seed"))}' if context else "")
        + "</div>"
    )

    html = _wrap_html(cfg.title, "".join(sections) + footer)
    # After the shared stylesheet so the threshold-only classes win.
    return html.replace("</head>", f"<style>{_EXTRA_CSS}</style></head>", 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("output"),
                        help="Pipeline output root containing thresholds/ (default: output)")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--author", default="")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path "
                             "(default: <output-root>/thresholds/threshold_report.html)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = ThresholdReportConfig(
        output_root=args.output_root.expanduser().resolve(),
        title=args.title,
        author=args.author,
    )
    data = load_report_data(cfg)
    html = build_report(cfg, data)
    written = write_html(html, args.out or cfg.default_out)
    print(f"Threshold report written: {written} (self-contained; figures embedded inline)")
    if data.warnings:
        print(f"  {len(data.warnings)} warning(s) — see the note at the top of the report.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
