"""Assemble ``output/thresholds/threshold_report.html`` from the phase's artifacts.

Reads the CSVs and SVGs the thresholder notebook already wrote — no refitting,
same contract as ``modelling_phase/report.py``. Run the notebook first, then
this, and the report reflects that run.

Two documents in one, because they are needed at different moments:

* a **step-by-step report** — what was done, what came out, what it means;
* a **reference section** — every term with a one-line definition, a concrete
  number from this cohort, and when it is used. Written for a radiologist to
  re-read before a presentation, not for a statistician.

Section 9 is the ESNR defence: the questions a commission actually asks, each
with a short answer backed by a number from this run.

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

# Run as a script (`python heavy_machinery/threshold_phase/threshold_report.py`)
# only this folder lands on sys.path, so the sibling phases have to be added
# before `report` and its own flat imports can resolve. A no-op when the
# notebook has already imported `heavy_machinery.config`.
for _phase in ("cleaning_phase", "modelling_phase", "threshold_phase"):
    _path = str(Path(__file__).resolve().parent.parent / _phase)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Shared with the main pipeline report so both documents look identical and the
# stylesheet has one home. The underscore names are private to `report`, not to
# the codebase.
import evidence  # noqa: E402  (needs the sys.path bootstrap above)
from report import (  # noqa: E402  (needs the sys.path bootstrap above)
    _CSS,  # noqa: F401  (imported for side-effect parity / future use)
    _esc,
    _figure_img_html,
    _wrap_html,
    details_block,
    info_box,
    section_block,
    table_to_html,
    warning_box,
    write_html,
)

DEFAULT_TITLE = "Threshold analysis — high-grade meningioma on pre-operative MRI"

# Pre-specified before the verdicts were read off: a threshold counts as
# reproducible when it is found again in at least this share of the MICE draws.
MICE_REPRODUCIBLE_CUT = evidence.MICE_REPRODUCIBLE_CUT
# Best-supported first — every ordering and tally in the document uses this.
GRADE_ORDER = list(evidence.GRADES)
GRADE_COLOUR = {
    evidence.GRADE_STRONG: "var(--green)",
    evidence.GRADE_MODERATE: "var(--yellow)",
    evidence.GRADE_FRAGILE: "var(--accent)",
    evidence.GRADE_WEAK: "var(--red)",
}

TABLE_FILES = {
    "cohort_summary": "00_cohort_summary.csv",
    "cohort": "01_metric_cohorts.csv",
    "risk_curves": "03_risk_curves.csv",
    "risk_reading": "04_risk_curves_reading_view.csv",
    "thresholds": "07_threshold_summary.csv",
    "threshold_reading": "08_threshold_reading_view.csv",
    "flag_missingness": "11_flag_missingness.csv",
    "combination_menu": "12_combination_menu.csv",
    "combination_reading": "13_combination_reading_view.csv",
    "count_score": "15_count_score.csv",
    "combination_verdict": "17_combination_verdict.csv",
    "stability": "18_imputation_stability.csv",
    "stability_reading": "19_imputation_stability_reading_view.csv",
    "risk_stability": "21_risk_curve_stability.csv",
    "count_score_imputed": "23_count_score_imputed.csv",
    "count_rules_imputed": "25_count_rules_imputed.csv",
    "headline": "26_headline_findings.csv",
    "evidence_criteria": "27_evidence_criteria.csv",
    "evidence": "28_threshold_evidence.csv",
    "evidence_reading": "29_threshold_evidence_reading_view.csv",
    "shared_menu": "30_shared_combination_menu.csv",
    "shared_reading": "31_shared_combination_reading_view.csv",
    "shared_verdict": "32_shared_combination_verdict.csv",
    "zero_share": "34_zero_inflation.csv",
    "presence_rules": "35_presence_rules.csv",
    "nonzero_curves": "36_risk_curves_nonzero_only.csv",
    "zero_comparison": "37_zero_inflation_comparison.csv",
    "multiplicity": "38_nonlinearity_multiplicity.csv",
    "multiplicity_reading": "39_nonlinearity_multiplicity_reading_view.csv",
    "calibration": "40_calibration.csv",
    "calibration_bins": "41_calibration_bins_uncut.csv",
    "net_benefit": "43_net_benefit.csv",
    "net_benefit_summary": "44_net_benefit_summary.csv",
}

FIGURE_FILES = {
    "risk_panel": "05_risk_curves_panel.svg",
    "combined_roc": "10_combined_roc.svg",
    "combination_space": "14_combination_space.svg",
    "shared_combination_space": "33_shared_combination_space.svg",
    "calibration": "42_calibration.svg",
    "net_benefit": "45_net_benefit.svg",
    "count_score": "16_count_score.svg",
    "stability": "20_stability_youden.svg",
    "knee_stability": "22_knee_stability.svg",
    "count_score_imputed": "24_count_score_imputed.svg",
}

FIGURE_PREFIXES = {
    "distributions": "02_distribution_",
    "risk_curves": "06_risk_curve_",
    "thresholds": "09_thresholds_",
}

_EXTRA_CSS = """
.keypoint {
    background: #f8fafc; border-left: 4px solid var(--accent);
    padding: 10px 14px; margin: 12px 0; border-radius: 0 6px 6px 0;
}
.keypoint b:first-child { color: var(--accent); }
.concrete {
    background: #fffbeb; border-left: 4px solid var(--yellow);
    padding: 10px 14px; margin: 12px 0; border-radius: 0 6px 6px 0;
}
.concrete::before { content: "On this cohort: "; font-weight: 700; color: #92400e; }
.verdict-list { list-style: none; padding: 0; margin: 12px 0; }
.verdict-list li {
    padding: 9px 13px; margin: 7px 0; border-radius: 6px;
    background: var(--card); border-left: 4px solid var(--border);
}
.verdict-list li.yes { border-left-color: var(--green); background: var(--green-bg); }
.verdict-list .metric-name { font-weight: 600; display: block; }
.answer-box {
    background: var(--green-bg); border: 1px solid var(--green);
    border-radius: 8px; padding: 12px 16px; margin: 14px 0;
}
.answer-box.negative { background: var(--grey-bg); border-color: var(--border); }
.answer-box h4 { margin: 0 0 4px; font-size: 12.5px; color: var(--fg);
                 text-transform: uppercase; letter-spacing: .05em; }
.answer-box p:last-child { margin-bottom: 0; }
.figure-note { font-size: 13px; color: var(--muted); margin: 4px 0 16px; }
.qa { margin: 0 0 18px; }
.qa .q {
    font-weight: 650; color: var(--fg); margin: 0 0 4px;
    padding-left: 22px; text-indent: -22px;
}
.qa .q::before { content: "Q "; color: var(--red); font-weight: 700; }
.qa .a { margin: 0 0 6px 22px; }
.qa .a::before { content: "A "; color: var(--green); font-weight: 700; }
.qa .ev { margin: 0 0 0 22px; font-size: 13px; color: var(--muted); }
.qa .ev::before { content: "→ evidence: "; font-style: italic; }
.ref-card {
    border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 15px; margin: 10px 0; background: var(--bg);
}
.ref-card h4 { margin: 0 0 6px; font-size: 15px; color: var(--fg);
               text-transform: none; letter-spacing: 0; font-weight: 650; }
.ref-card dl { margin: 0; display: grid; grid-template-columns: 92px 1fr;
               gap: 3px 12px; font-size: 14px; }
.ref-card dt { color: var(--muted); font-weight: 600; font-size: 12.5px;
               text-transform: uppercase; letter-spacing: .03em; padding-top: 2px; }
.ref-card dd { margin: 0; }
.ref-card dd.num { font-variant-numeric: tabular-nums; }
.tradeoff-table td:first-child { font-weight: 600; white-space: nowrap; }
/* A "86/127/15/99" cell offers a break opportunity after every slash, so a
   narrow column wraps it mid-number and it reads as "86/127/1  5/99". Numbers
   and intervals must never break; the row can be scrolled instead. */
table.report td { overflow-wrap: normal; word-break: keep-all; hyphens: manual; }
table.report td.nowrap, table.report td.num { white-space: nowrap; }
.grade {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .04em; color: #fff;
}
.grade-list { list-style: none; padding: 0; margin: 12px 0; }
.grade-list li {
    padding: 9px 13px; margin: 7px 0; border-radius: 6px;
    background: var(--card); border-left: 4px solid var(--border);
}
.grade-list .metric-name { font-weight: 600; }
.grade-list .limit { color: var(--muted); font-size: 13.5px; display: block;
                     margin-top: 2px; }
.todo {
    border: 2px dashed var(--red); border-radius: 8px;
    padding: 10px 14px 12px; margin: 12px 0; background: #fff7f7;
}
.todo-label {
    display: inline-block; background: var(--red); color: #fff;
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
    padding: 1px 8px; border-radius: 3px; margin-bottom: 6px;
}
.todo h4 { margin: 0 0 4px; font-size: 15px; text-transform: none;
           letter-spacing: 0; font-weight: 650; }
.todo p { margin: 0 0 6px; }
.todo .todo-why { font-size: 13px; color: var(--muted); margin-bottom: 0; }
/* Written to be selected and pasted, so: no badges, no pills, nothing that
   carries markup into a manuscript. Just prose with bold labels. */
.manuscript {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin: 14px 0;
    line-height: 1.62;
}
.manuscript p { margin: 0 0 12px; }
.manuscript p:last-child { margin-bottom: 0; }
.manuscript b { color: var(--fg); }
"""


# ---------------------------------------------------------------------------
# Config and inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdReportConfig:
    output_root: Path = Path("output")
    title: str = DEFAULT_TITLE
    author: str = ""
    subtitle: str = ("Where the risk of WHO grade 2–3 rises, where to draw a line, "
                     "and what that line costs")

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
    model_aucs: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    def table(self, key: str) -> pd.DataFrame:
        return self.tables.get(key, pd.DataFrame())


def load_report_data(cfg: ThresholdReportConfig) -> ThresholdReportData:
    """Read the threshold phase's artifacts, plus model AUCs if the modelling run exists."""
    root = cfg.thresholds_root
    data = ThresholdReportData()

    if not root.exists():
        data.warnings.append(f"{root} does not exist — run meningioma-thresholder.ipynb first.")
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

    data.model_aucs = _load_model_aucs(cfg.output_root, data.warnings)
    return data


def _load_model_aucs(output_root: Path, warnings: list[str]) -> pd.DataFrame:
    """Bootstrap-corrected AUCs from the modelling notebook, for the §7 comparison."""
    folder = output_root / "inferential" / "model_artifacts"
    if not folder.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(folder.glob("*_model.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = {m.get("metric"): m
                   for m in payload.get("validation", {}).get("metrics", [])}
        auc = metrics.get("AUC", {})
        if not auc:
            continue
        n_predictors = max(len(payload.get("coefficients", {})) - 1, 0)
        events = payload.get("events")
        rows.append({
            "model": path.stem.replace("high_grade_", "").replace("_model", ""),
            "n": payload.get("n"),
            "events": events,
            "n_predictors": n_predictors,
            # Events per variable. Below ~10 a logistic model is fitting noise;
            # printing it makes the overfitting visible instead of arguable.
            "EPV": (float(events) / n_predictors
                    if events and n_predictors else float("nan")),
            "AUC_apparent": auc.get("apparent"),
            "AUC_corrected": auc.get("optimism_corrected"),
        })
    if not rows:
        warnings.append("Model artifacts found but none carried an AUC.")
    return pd.DataFrame(rows).sort_values("AUC_corrected", ascending=False)


# ---------------------------------------------------------------------------
# Formatting helpers
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


def _truthy(value: Any) -> bool:
    """CSV round-tripping turns booleans into 'True'/'False' strings."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return bool(value)


# Cells that must never wrap. A "86/127/15/99" or a "1.53–12.2" offers the
# browser a break opportunity after every slash and dash, so a narrow column
# splits it mid-number: the published report showed "86/127/1 5/99" for what
# the CSV correctly holds as 86/127/15/99.
NOWRAP_COLUMNS = frozenset({
    "TP/FP/FN/TN", "n", "n / events", "Cut-point", "Cut-point 95% CI",
    "Sens (95% CI)", "Spec (95% CI)", "PPV (95% CI)", "NPV (95% CI)",
    "Sens / Spec", "J", "J (corr.)", "J as measured", "J after correction",
    "Steepest rise", "95% CI", "95% interval", "Patients below it",
    "Risk", "Risk reaches 30%", "Risk reaches 50%", "Risk 30% at", "Risk 50% at",
    "Non-linearity p", "Holm-adjusted p", "Bonferroni-adjusted p",
    "Criteria met", "AUC", "AUC as measured", "AUC corrected", "EPV",
    "Predictors", "Events", "Patients", "High grade", "Criteria met",
    "Slope (corrected)", "Intercept (corrected)", "Brier (corrected)",
    "Best net benefit", "Beats treat-all and treat-none",
    "…over this share of the range", "Best available strategy over",
    "Measured in", "Missing", "Median, benign", "Median, high grade",
    "Knee interval width", "Patients (avg)", "Threshold found in", "of datasets",
    "Typical location", "Range", "Criteria met", "Lowest of the datasets",
    "Highest of the datasets", "Youden cut-point", "MICE cut-point",
})


def _table(df: pd.DataFrame, **kwargs: Any) -> str:
    """``table_to_html`` with the numeric columns pinned against mid-token wraps."""
    if df is None or getattr(df, "empty", True):
        return table_to_html(df, **kwargs)
    kwargs.setdefault("nowrap_cols",
                      [c for c in df.columns if str(c) in NOWRAP_COLUMNS])
    return table_to_html(df, **kwargs)


def _key(label: str, text: str) -> str:
    return f'<div class="keypoint"><b>{_esc(label)}</b> {text}</div>'


def _concrete(text: str) -> str:
    """A claim anchored to a real number from this run."""
    return f'<div class="concrete">{text}</div>'


def _answer(heading: str, body: str, *, positive: bool = True) -> str:
    cls = "answer-box" if positive else "answer-box negative"
    return f'<div class="{cls}"><h4>{_esc(heading)}</h4>{body}</div>'


def _qa(question: str, answer: str, evidence: str = "") -> str:
    ev = f'<p class="ev">{evidence}</p>' if evidence else ""
    return (f'<div class="qa"><p class="q">{question}</p>'
            f'<p class="a">{answer}</p>{ev}</div>')


def _ref(term: str, is_: str, use: str, here: str = "") -> str:
    """One reference card: what it is, when you use it, what it was here."""
    here_html = (f"<dt>Here</dt><dd class='num'>{here}</dd>") if here else ""
    return (f'<div class="ref-card"><h4>{_esc(term)}</h4><dl>'
            f"<dt>Is</dt><dd>{is_}</dd>"
            f"<dt>Use it</dt><dd>{use}</dd>"
            f"{here_html}</dl></div>")


def _figure(path: Path | None, note: str = "") -> str:
    if path is None or not path.exists():
        return '<p class="muted"><em>(figure unavailable — re-run the notebook)</em></p>'
    img = _figure_img_html(path)
    if not img:
        return '<p class="muted"><em>(figure could not be embedded)</em></p>'
    note_html = f'<p class="figure-note">{note}</p>' if note else ""
    return f'<div class="figure-card">{img}</div>{note_html}'


def _figure_row(paths: Sequence[Path], note: str = "") -> str:
    cards = [f'<div class="figure-card">{_figure_img_html(p)}</div>'
             for p in paths if p.exists() and _figure_img_html(p)]
    if not cards:
        return '<p class="muted"><em>(figures unavailable)</em></p>'
    note_html = f'<p class="figure-note">{note}</p>' if note else ""
    return f'<div class="figure-grid">{"".join(cards)}</div>{note_html}'


# ---------------------------------------------------------------------------
# The verdict — one source, every sentence templated from it
# ---------------------------------------------------------------------------
# Section 3's answer, the header card, the section 9 defence, the reference
# card and the caveats all used to state the verdict as prose written by hand.
# When the numbers were refreshed the prose was not, and the document
# contradicted itself. Everything that says "does this measurement have a
# threshold" now reads one list built here, from the same CSV the tables use.
@dataclass(frozen=True)
class MetricVerdict:
    """What this run concluded about one measurement's threshold."""

    metric: str
    column: str
    has_threshold: bool
    where: str = ""            # steepest-rise point, formatted, or ""
    pct_below: float = float("nan")
    reproduced: float = float("nan")   # MICE knee_rate, 0–1
    grade: str = ""            # strong / moderate / fragile / weak (section 3)
    limiting: str = ""         # the criterion that holds the grade down
    grade_note: str = ""       # grade with its binding criterion spelled out


def _join_names(names: Sequence[str], conjunction: str = "and") -> str:
    """``A``, ``A and B``, ``A, B and C`` — never a stray comma."""
    items = [str(n) for n in names]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


@dataclass(frozen=True)
class Verdicts:
    """The per-measurement verdicts, plus the phrases every section needs.

    Nothing below returns a fixed number or a fixed measurement name, so a
    rerun that changes the answer changes every sentence with it.
    """

    items: tuple[MetricVerdict, ...] = ()

    @property
    def positive(self) -> list[MetricVerdict]:
        return [v for v in self.items if v.has_threshold]

    @property
    def negative(self) -> list[MetricVerdict]:
        return [v for v in self.items if not v.has_threshold]

    @property
    def n_thresholds(self) -> int:
        return len(self.positive)

    @property
    def n_metrics(self) -> int:
        return len(self.items)

    def count_phrase(self, *, verb: bool = False) -> str:
        """``3 of 4 measurements``, optionally with an agreeing verb."""
        noun = "measurement" if self.n_metrics == 1 else "measurements"
        text = f"{self.n_thresholds} of {self.n_metrics} {noun}"
        if verb:
            text += " has" if self.n_thresholds == 1 else " have"
        return text

    def positive_names(self) -> str:
        return _join_names([v.metric for v in self.positive])

    def negative_names(self) -> str:
        return _join_names([v.metric for v in self.negative])

    def turning_phrase(self) -> str:
        """``ADC (mean) turns near 0.662`` for each measurement that turns."""
        bits = [f"{v.metric} turns near {v.where}" for v in self.positive if v.where]
        return _join_names(bits, "and")

    # ---- the graded verdict (section 3's evidence hierarchy) -------------
    @property
    def graded(self) -> bool:
        return any(v.grade for v in self.items)

    def by_grade(self, grade: str) -> list[MetricVerdict]:
        return [v for v in self.items if v.grade == grade]

    def grade_tally(self) -> list[tuple[str, int]]:
        """``[("strong", 1), ("moderate", 2), …]`` — only grades that occur."""
        return [(g, len(self.by_grade(g))) for g in GRADE_ORDER if self.by_grade(g)]

    def grade_phrase(self) -> str:
        """``1 strong, 2 moderate and 1 fragile`` — the headline in one clause."""
        return _join_names([f"{n} {g}" for g, n in self.grade_tally()])

    def strongest(self) -> list[MetricVerdict]:
        """The measurements carrying the best-supported claim in this run."""
        for g in GRADE_ORDER:
            hits = self.by_grade(g)
            if hits:
                return hits
        return []

    def grade_sentences(self, *, compact: bool = False) -> str:
        """One clause per measurement: name, grade, and what limits it.

        ``compact`` groups measurements sharing a grade and drops the padding,
        for the manuscript block where every word is paid for.
        """
        ordered = sorted(self.items, key=lambda x: GRADE_ORDER.index(x.grade)
                         if x.grade in GRADE_ORDER else len(GRADE_ORDER))
        if not compact:
            bits = []
            for v in ordered:
                if not v.grade:
                    continue
                limit = (f" (limited by {v.limiting.lower()})" if v.limiting
                         else " (all criteria met)")
                bits.append(f"{_esc(v.metric)} <b>{_esc(v.grade)}</b>{limit}")
            return _join_names(bits)

        # Compact: one clause per (grade, limiting criterion) group.
        groups: dict[tuple[str, str], list[str]] = {}
        for v in ordered:
            if not v.grade:
                continue
            groups.setdefault((v.grade, v.limiting), []).append(v.metric)
        bits = []
        for (grade, limiting), names in groups.items():
            limit = f" ({limiting.lower()})" if limiting else ""
            bits.append(f"{_esc(_join_names(names))} <b>{_esc(grade)}</b>{limit}")
        return "; ".join(bits)


def metric_verdicts(data: ThresholdReportData) -> Verdicts:
    """Build the verdict list from the risk-curve table, MICE stability and grades."""
    risk = data.table("risk_curves")
    if risk.empty or "knee_found" not in risk.columns:
        return Verdicts()

    knee_rate = {}
    stab = data.table("risk_stability")
    if not stab.empty and "knee_rate" in stab.columns and "column" in stab.columns:
        knee_rate = dict(zip(stab["column"], stab["knee_rate"]))

    grades: dict[Any, dict[str, str]] = {}
    ev_table = data.table("evidence")
    if not ev_table.empty and "column" in ev_table.columns:
        # An empty cell round-trips through CSV as NaN, and str(nan) is "nan" —
        # which is how "limited by nan" reaches a poster.
        def _cell(value: Any) -> str:
            return "" if value is None or pd.isna(value) else str(value)

        grades = {row["column"]: {
            "grade": _cell(row.get("verdict")),
            "limiting": _cell(row.get("limiting_criterion")),
            "note": _cell(row.get("verdict_note")),
        } for _, row in ev_table.iterrows()}

    items = []
    for _, row in risk.iterrows():
        found = _truthy(row.get("knee_found"))
        g = grades.get(row.get("column"), {})
        items.append(MetricVerdict(
            metric=str(row.get("metric", "")),
            column=str(row.get("column", "")),
            has_threshold=found,
            where=_sig(row.get("steepest_x")) if found else "",
            pct_below=float(row.get("steepest_pct_of_patients", float("nan"))),
            reproduced=float(knee_rate.get(row.get("column"), float("nan"))),
            grade=g.get("grade", ""),
            limiting=g.get("limiting", ""),
            grade_note=g.get("note", ""),
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
    m_draws: str = "—"
    n_boot: str = "—"          # cut-point bootstraps
    n_boot_curve: str = "—"    # risk-curve bootstraps — a smaller, separate budget
    verdicts: Verdicts = field(default_factory=Verdicts)

    @property
    def n_thresholds(self) -> int:
        return self.verdicts.n_thresholds

    @property
    def n_metrics(self) -> int:
        return self.verdicts.n_metrics


def cohort_facts(data: ThresholdReportData) -> CohortFacts:
    cohort = data.table("cohort")
    context = data.manifest.get("context", {})

    n = events = benign = 0
    prevalence = float("nan")
    summary = data.table("cohort_summary")
    if not summary.empty:
        n = int(_first(summary, "n_patients", 0) or 0)
        events = int(_first(summary, "n_high_grade", 0) or 0)
        benign = int(_first(summary, "n_benign", 0) or 0)
        prevalence = float(_first(summary, "prevalence", float("nan")))
    elif not cohort.empty:
        # Fallback for a run predating 00_cohort_summary.csv. Approximate:
        # every metric drops its own rows, so no single row of this table
        # knows the cohort total, and averaging prevalences across metrics
        # with different denominators is off by a patient or two.
        idx = (cohort["n_analysed"] + cohort["n_missing"]).idxmax()
        n = int(cohort.loc[idx, "n_analysed"] + cohort.loc[idx, "n_missing"])
        prevalence = float(cohort["prevalence"].mean())
        events = int(round(n * prevalence))
        benign = n - events

    return CohortFacts(
        n=n, events=events, benign=benign, prevalence=prevalence,
        m_draws=str(_int(_first(data.table("stability"), "m_draws"), "—")),
        n_boot=str(context.get("n_bootstrap", "—")),
        n_boot_curve=str(context.get("n_bootstrap_curve",
                                     context.get("n_bootstrap", "—"))),
        verdicts=metric_verdicts(data),
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _threshold_card(v: Verdicts) -> tuple[str, str, str]:
    """The stat card for the headline finding.

    Deliberately not "4 of 4". A count of measurements that cleared one
    p-value is the claim reviewers attack first; the best-supported grade and
    the spread of grades is both a weaker-sounding and a stronger claim.
    """
    if not v.graded:
        return ("Real thresholds", f"{v.n_thresholds} of {v.n_metrics}",
                "measurements with a true turning point")
    best = v.strongest()
    grade = best[0].grade if best else "—"
    return ("Threshold evidence", f"{len(best)} {grade}",
            v.grade_phrase() + f", of {v.n_metrics}")


def render_header(cfg: ThresholdReportConfig, data: ThresholdReportData,
                  facts: CohortFacts) -> str:
    verdict = primary_verdict(data)
    v = facts.verdicts

    cards = [
        ("Patients", str(facts.n) if facts.n else "—", "operated, with histology"),
        ("High grade", _pct(facts.prevalence), f"{facts.events} of {facts.n} are WHO 2–3"),
        _threshold_card(v),
        ("Best cut-point", _num(_first(verdict, "best_single_J")),
         "Youden J, single criterion"),
        ("Best model", _num(_first(data.model_aucs, "AUC_corrected")),
         "AUC, full multivariable"),
        ("Imputed datasets", facts.m_draws, "missing-data check"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="card-label">{_esc(label)}</div>'
        f'<div class="card-value">{_esc(value)}</div>'
        f'<div class="card-sub">{_esc(sub)}</div></div>'
        for label, value, sub in cards
    )

    author_html = f'<p class="report-authors">{_esc(cfg.author)}</p>' if cfg.author else ""

    if v.n_metrics == 0:
        headline_answer = ""
    elif v.graded:
        # Graded, because "4 of 4 passed" is the claim a commission attacks
        # first — and it is not what the evidence actually supports.
        headline_answer = (
            f"<li><b>The threshold claims are not equally supported: "
            f"{_esc(v.grade_phrase())}.</b> {v.grade_sentences()}. Section 3 scores each "
            f"on five criteria fixed before the verdicts were read off.</li>")
    elif v.n_thresholds == 0:
        headline_answer = (
            f"<li><b>No measurement has a genuine threshold.</b> All {v.n_metrics} rise "
            f"smoothly: they have cut-points, but no turning point.</li>")
    else:
        rest = (f" {v.negative_names()} rise smoothly: cut-points, but no turning point."
                if v.negative else "")
        headline_answer = (
            f"<li><b>{v.count_phrase(verb=True)} a genuine threshold</b> — "
            f"{v.turning_phrase()}.{rest}</li>")
    gain = _first(verdict, "gain_vs_best_single")
    combo_answer = (
        f"<li><b>Combining criteria did not usefully help.</b> The best combination gains "
        f"J {_num(gain, 3)} over the best single criterion with both corrected the same "
        f"way, but wins in only {_pct(_first(verdict, 'winner_stability'))} of resampled "
        f"cohorts — so which combination is best is mostly noise.</li>")

    warn = ""
    if data.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in data.warnings)
        warn = details_block(f"⚠️ {len(data.warnings)} note(s) loading artifacts",
                             f"<ul>{items}</ul>")

    return (
        '<div class="report-title-block">'
        f"<h1>{_esc(cfg.title)}</h1>"
        f'<p class="muted">{_esc(cfg.subtitle)}</p>{author_html}</div>'
        f'<div class="cards">{card_html}</div>'
        f"""
<h3>The three answers, before the detail</h3>
<ul>
{headline_answer}
{combo_answer}
<li><b>The count score is the usable output.</b> Risk rises step by step with how many
criteria a tumour meets — no cut-point arithmetic at the workstation.</li>
</ul>

{_key("How this document is built:",
      "Sections 1–8 are the analysis, in order, each one stating its question, the method in "
      "two sentences, the figure, and the answer. <b>Section 9 is the ESNR defence</b> — the "
      "questions a commission asks, with the number that answers each. <b>Section 10 is a "
      "reference</b>: every term, one line each, with what it was on this cohort. Re-read 9 "
      "and 10 before you present.")}

<p class="muted">Nothing was fitted here. This assembles the tables and figures the thresholder
notebook wrote, so these numbers and the CSVs in <code>output/thresholds/tables/</code> are
always identical.</p>
"""
        f"{warn}"
    )


# ---------------------------------------------------------------------------
# 0 — methods and cohort
# ---------------------------------------------------------------------------
# Everything the pipeline cannot know. These are clinical and procedural facts
# that live in the study protocol and in the radiologists' heads, not in the
# data — so they are emitted as visible placeholders rather than guessed at.
# A guessed b-value or a guessed WHO edition in a conference document is worse
# than a blank one, because nobody checks a sentence that reads plausibly.
@dataclass(frozen=True)
class MethodsGap:
    key: str
    heading: str
    prompt: str
    why: str


METHODS_GAPS: tuple[MethodsGap, ...] = (
    MethodsGap(
        "who-version", "WHO CNS classification version",
        "Which edition was used to grade these tumours, and whether the cases were "
        "re-graded to the 2021 criteria (CDKN2A/B homozygous deletion, TERT promoter "
        "mutation) or graded as reported under 2016.",
        "The outcome variable is defined by this. A reviewer cannot interpret a "
        "high-grade rate without knowing which edition produced it, and the 2021 "
        "molecular criteria move cases across the grade 2 boundary."),
    MethodsGap(
        "histology-reading", "Who read the histology",
        "How many pathologists reported the grade, whether they were blinded to the "
        "imaging, and any inter-rater agreement statistic if two read the same slides.",
        "The reference standard's own reliability bounds every accuracy figure in "
        "this document. If the grade is not blinded to imaging, the whole design is "
        "circular."),
    MethodsGap(
        "dwi-acquisition", "DWI acquisition",
        "Field strength, scanner vendor and model, and the b-values used to compute "
        "the ADC maps. Note any change of scanner or protocol across the accrual "
        "window.",
        "ADC values are not comparable between b-value schemes. Publishing an ADC "
        "cut-point without them makes it unusable by anyone else — and the "
        "cut-point is the section 4 headline."),
    MethodsGap(
        "adc-roi", "ADC ROI protocol",
        "Whole-tumour volumetric ROI or single-slice; solid portion only or whole "
        "mass including cystic/necrotic areas; how many readers placed the ROIs and "
        "whether they were blinded to grade.",
        "Single-slice and whole-tumour ADC differ systematically, which is the "
        "leading explanation for why published ADC cut-points do not transfer."),
    MethodsGap(
        "volumetry", "Volumetry method and software",
        "How tumour and peritumoral edema volumes were measured — manual, "
        "semi-automated or automated — the named software and version, on which "
        "sequence, and who did it.",
        "Both volume thresholds are quoted in cc against published cut-points. "
        "Those are only comparable if the segmentation is."),
    MethodsGap(
        "edema-index", "Definition of the edema index",
        "The intended clinical definition, and the source it comes from if it is "
        "taken from the literature rather than defined here.",
        "This measurement is analysed throughout the document and its definition "
        "is stated nowhere in it. See the computed formula noted alongside — "
        "confirm that it is the definition you mean."),
    MethodsGap(
        "inclusion", "Inclusion and exclusion criteria",
        "How the cohort was assembled: consecutive cases or selected; which "
        "pre-operative sequences a patient had to have; what excluded a patient "
        "(prior surgery, prior radiotherapy, no pre-op MRI, non-diagnostic study).",
        "The accrual window below is derived from the data. The criteria are not "
        "in the data at all, and a reader cannot judge selection bias without them."),
    MethodsGap(
        "ethics", "Ethics approval and consent",
        "The IRB / ethics committee name and approval number, and the retrospective "
        "consent waiver.",
        "Required by every journal and by the ESNR abstract form."),
    MethodsGap(
        "prevalence-context", "Why the high-grade rate is above the literature rate",
        "The published population figure for the WHO 2–3 share of meningiomas with a "
        "citation, and one sentence on why a surgical series runs above it "
        "(operated cases are selected for size, symptoms and worrying imaging).",
        "The cohort's own rate is computed below. The comparison figure is a "
        "literature value and is not invented here. This is the sentence that "
        "supports the PPV/NPV transferability caveat in section 11."),
    MethodsGap(
        "stard", "STARD adherence",
        "Whether the study is reported to STARD 2015, and where the completed "
        "checklist and flow diagram will live.",
        "This is a diagnostic accuracy study. The first methodological question a "
        "reviewer asks is which reporting guideline it follows."),
)


def _todo_block(gap: MethodsGap) -> str:
    """One placeholder: visible on the page and greppable in the source."""
    return (
        f"<!-- TODO: ANDY — {_esc(gap.key)}: {_esc(gap.prompt)} -->"
        f'<div class="todo" id="todo-{_esc(gap.key)}">'
        f'<div class="todo-label">TODO: ANDY</div>'
        f"<h4>{_esc(gap.heading)}</h4>"
        f"<p>{_esc(gap.prompt)}</p>"
        f'<p class="todo-why"><b>Why it has to be here:</b> {_esc(gap.why)}</p>'
        f"</div>")


def render_methods(data: ThresholdReportData, facts: CohortFacts) -> str:
    """Section 0 — what the pipeline knows, and visible holes where it cannot."""
    summary = data.table("cohort_summary")
    cohort = data.table("cohort")

    accrual = ""
    first, last = _first(summary, "accrual_first_year"), _first(summary, "accrual_last_year")
    if pd.notna(first) and pd.notna(last):
        accrual = (f" Cases carry entry years from <b>{_int(first)}</b> to "
                   f"<b>{_int(last)}</b> ({_int(_first(summary, 'accrual_n_years'))} "
                   f"calendar years), which is the accrual window as the data record "
                   f"it — not necessarily the window the protocol specified.")

    missingness = ""
    if not cohort.empty and {"metric", "n_analysed", "n_missing"} <= set(cohort.columns):
        missingness = _table(pd.DataFrame({
            "Measurement": cohort["metric"],
            "Measured in": cohort["n_analysed"],
            "Missing": cohort["n_missing"],
            "Missing (%)": [
                _num(100.0 * m / (a + m), 1) + "%"
                for a, m in zip(cohort["n_analysed"], cohort["n_missing"])],
        }))

    return f"""
<p>What this analysis rests on. The numbers below come from the pipeline and refresh with
every run. <b>The lettered blocks are the facts the pipeline cannot know</b> — they live in
the study protocol and in the reading room, and they are left visibly blank rather than
filled with a plausible guess.</p>

{_key("Why blanks rather than sensible defaults:",
      "a guessed b-value, a guessed WHO edition or a guessed ROI protocol reads exactly "
      "like a real one. Nobody checks a sentence that sounds right, so a wrong one survives "
      "into the manuscript. Every gap below is marked <code>TODO: ANDY</code> in both the "
      "page and the HTML source, and they are listed together at the end of the "
      "notebook run.")}

<h3>What the pipeline does know</h3>

<p>The cohort is <b>{facts.n} operated patients with histology</b>, of whom
<b>{facts.events} ({_pct(facts.prevalence, 1)})</b> are WHO grade 2–3 and
<b>{facts.benign}</b> are WHO grade 1.{accrual}</p>

{missingness}

<p class="muted">Each measurement is analysed on its own complete cases, so <em>n</em>
differs by row. Section 6 checks whether that mattered.</p>

<h3>The edema index, as computed</h3>

<p>Stated here because it is used throughout the document and defined nowhere else in it.
As implemented in the pipeline the edema index is</p>

<p style="text-align:center"><code>edema index = edema volume (cc) ÷ tumour volume (cc)</code></p>

<p>with the ratio left missing where tumour volume is zero. That is the formula the numbers
in this report were produced from. Whether it is the definition intended clinically is the
first placeholder below.</p>

<h3>What has to be filled in before this is submitted</h3>

{"".join(_todo_block(g) for g in METHODS_GAPS)}

{warning_box(
    f"{len(METHODS_GAPS)} placeholders above. None of them can be derived from the data, "
    f"and none has been guessed. Search this document or its HTML source for "
    f"<code>TODO: ANDY</code> to find them all.")}
"""


# ---------------------------------------------------------------------------
# 1 — three questions
# ---------------------------------------------------------------------------
def render_questions(data: ThresholdReportData, facts: CohortFacts) -> str:
    v = facts.verdicts
    # The example of "cut-point without threshold" has to come from this run.
    # Naming a measurement that did turn, because an earlier run said it did
    # not, is exactly the contradiction this document had.
    if v.negative:
        example = (f"{_esc(v.negative_names())} in this series "
                   f"{'has' if len(v.negative) == 1 else 'have'} a perfectly reasonable "
                   f"cut-point and no threshold.")
    elif v.positive:
        example = (f"In this series all {v.n_metrics} measurements cleared the curvature "
                   f"test, so the distinction does not separate them here — but it is what "
                   f"the test in section 3 is for, and it is the reason a cut-point alone "
                   f"is never evidence of a threshold.")
    else:
        example = ""

    table = pd.DataFrame([
        {"Question": "Where does risk climb fastest?",
         "The clinical version": "Above what volume, or below what ADC, does this tumour "
                                 "become worrying?",
         "Section": "3",
         "Gives you": "A risk curve; sometimes one turning point"},
        {"Question": "Where do I draw the line?",
         "The clinical version": "I must write benign or atypical in the report. What value "
                                 "do I use?",
         "Section": "4",
         "Gives you": "One cut-point with sensitivity and specificity"},
        {"Question": "Do several criteria help?",
         "The clinical version": "Low ADC <em>and</em> a large tumour — better than either alone?",
         "Section": "5",
         "Gives you": "A combined rule, and whether it actually beats one"},
    ])

    return f"""
<p>These are three different questions. Most threshold papers answer the second and describe
it as the first.</p>

{_table(table, safe_html_cols=["The clinical version", "Gives you"])}

{_key("The difference that matters:",
      "a cut-point exists for any measurement that separates the groups at all — you can always "
      "draw a line. A <b>threshold</b> requires the risk itself to change behaviour at that "
      "point. " + example)}

{_key("Two words used throughout:",
      f"<b>sensitivity</b> — of the {facts.events} high-grade tumours, the share the rule "
      f"catches. <b>Specificity</b> — of the {facts.benign} benign ones, the share it correctly "
      f"leaves alone. Buying more of one always costs the other.")}
"""


# ---------------------------------------------------------------------------
# 2 — the measurements
# ---------------------------------------------------------------------------
def render_measurements(data: ThresholdReportData, facts: CohortFacts) -> str:
    cohort = data.table("cohort")

    display = pd.DataFrame()
    gap_note = ""
    if not cohort.empty:
        display = pd.DataFrame({
            "Measurement": cohort["metric"],
            "Suspicious when": cohort["direction"].map({"lower": "low", "higher": "high"}),
            "Measured in": cohort["n_analysed"],
            "Missing": cohort["n_missing"],
            "Median, benign": cohort.get("median_benign"),
            "Median, high grade": cohort.get("median_high_grade"),
        })
        rows = []
        for _, r in cohort.iterrows():
            lo, hi = r.get("median_benign"), r.get("median_high_grade")
            if pd.notna(lo) and pd.notna(hi):
                rows.append(f"{r['metric']} {_num(lo, 2)} vs {_num(hi, 2)}")
        if rows:
            gap_note = _concrete(
                "median benign vs high grade — " + "; ".join(rows) +
                ". These gaps are the entire raw material. Nothing downstream can "
                "manufacture separation that is not already here.")

    return f"""
<p>Four numbers off the pre-operative study. Each is analysed on the patients who have it, so
<em>n</em> differs by row — pooling to the patients who have all four would discard people
missing something unrelated.</p>

{_table(display) if not display.empty else '<p class="muted"><em>(cohort table unavailable)</em></p>'}

{gap_note}

<h3>The overlap</h3>

<p>Benign on the left, high grade on the right: density, box, and every patient. What decides
whether a rule can work is not whether the groups differ — it is how far they overlap.</p>

{_figure_row(data.figure_groups.get("distributions", []))}

{_key("Ignore the p-value here.",
      f"With {facts.n} patients a p-value will call a clinically meaningless difference "
      "significant. A test that has to classify one patient in front of you lives on the "
      "overlap. Read the AUC in each subtitle instead.")}

{_key("AUC, concretely:",
      "pick one high-grade and one benign tumour at random. AUC is the chance the high-grade "
      "one looks more suspicious. 0.50 is a coin flip. <b>0.63 means it gets it right 63 times "
      "in 100</b> — a real signal, not a usable standalone test.")}
"""


# ---------------------------------------------------------------------------
# 3 — risk curves
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MultiplicityFacts:
    """The pre-specified family and what survives each correction."""

    n_tests: int = 0
    n_holm: int = 0
    n_bonferroni: int = 0
    dropped_by_bonferroni: tuple[str, ...] = ()
    dropped_by_holm: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.n_tests > 0


def multiplicity_facts(data: ThresholdReportData) -> MultiplicityFacts:
    table = data.table("multiplicity")
    if table.empty or "survives_holm" not in table.columns:
        return MultiplicityFacts()
    holm = table["survives_holm"].map(_truthy)
    bonf = table["survives_bonferroni"].map(_truthy)
    raw = table["survives_raw"].map(_truthy)
    return MultiplicityFacts(
        n_tests=len(table),
        n_holm=int(holm.sum()),
        n_bonferroni=int(bonf.sum()),
        dropped_by_bonferroni=tuple(table.loc[holm & ~bonf, "metric"].astype(str)),
        dropped_by_holm=tuple(table.loc[raw & ~holm, "metric"].astype(str)),
    )


def render_multiplicity(data: ThresholdReportData) -> str:
    """Section 3's multiple-testing block — the family, then both corrections."""
    facts = multiplicity_facts(data)
    if not facts.available:
        return ""
    reading = data.table("multiplicity_reading")

    holm_line = (f"<b>{facts.n_holm} of {facts.n_tests}</b> survive Holm adjustment")
    bonf_line = f"<b>{facts.n_bonferroni} of {facts.n_tests}</b> survive Bonferroni"
    if facts.dropped_by_bonferroni:
        bonf_line += (f" — it additionally drops "
                      f"{_esc(_join_names(list(facts.dropped_by_bonferroni)))}")

    return f"""
<h3>Multiple testing, stated before it is asked about</h3>

<p>Four measurements, four non-linearity tests, one outcome. That family is
<b>pre-specified as the primary one</b>: it is the set of hypothesis tests this analysis
actually makes. The cut-point selection rules and the combination menu are deliberately
<em>not</em> in it — those are selections, not tests, no p-value is read off them, and they
are paid for with optimism correction in sections 4 and 5 instead. Correcting them as
though they were hypotheses would imply a family that was never posed.</p>

{_table(reading) if not reading.empty else ""}

{_concrete(f"{holm_line}; {bonf_line}. Holm controls the same family-wise error rate as "
           f"Bonferroni and is uniformly more powerful, so anything surviving Bonferroni "
           f"survives Holm. Both are printed because Bonferroni is the correction a "
           f"reviewer has in mind, and stating the pair pre-empts the question instead of "
           f"conceding it.")}
"""


def render_zero_inflation(data: ThresholdReportData) -> str:
    """Section 3's method note on zero-inflated metrics, and the three-way fit.

    Promoted here from the caveats list, because it is not a caveat: for a
    metric where a third of the cohort sits at exactly zero, the whole-cohort
    spline is answering a different question from the one the threshold claims
    to answer, and the reader has to know that before reading the knee.
    """
    zero = data.table("zero_share")
    if zero.empty or "zero_inflated" not in zero.columns:
        return ""
    hits = zero[zero["zero_inflated"].map(_truthy)]
    if hits.empty:
        return ""

    lines = []
    for _, r in hits.iterrows():
        lines.append(
            f"<li><b>{_esc(r['metric'])}</b> — {_int(r['n_zero'])} of "
            f"{_int(r['n_analysed'])} measured patients ({_num(r['pct_zero'], 1)}%) are "
            f"exactly zero. Risk {_pct(r['risk_zero'])} "
            f"({_pct(r['risk_zero_lo'])}–{_pct(r['risk_zero_hi'])}) with none against "
            f"{_pct(r['risk_positive'])} ({_pct(r['risk_positive_lo'])}–"
            f"{_pct(r['risk_positive_hi'])}) with any — a "
            f"{_num(r['risk_ratio'], 1)}-fold difference before a single cut-point is "
            f"drawn.</li>")

    comparison = data.table("zero_comparison")
    comparison_html = ""
    if not comparison.empty:
        cols = [c for c in comparison.columns if c != "Metric"]
        comparison_html = _table(comparison[["Metric"] + cols])

    # The one sentence that decides what the knee can be called.
    nonzero = data.table("nonzero_curves")
    survives = []
    if not nonzero.empty and "nonlinearity_p" in nonzero.columns:
        for _, r in nonzero.iterrows():
            survives.append((str(r["metric"]),
                             float(r["nonlinearity_p"]),
                             _truthy(r.get("knee_found"))))
    verdict_html = ""
    if survives:
        lost = [m for m, p, _ in survives if not (p < evidence.ALPHA)]
        kept = [m for m, p, _ in survives if p < evidence.ALPHA]
        parts = []
        if lost:
            parts.append(
                f"<b>{_esc(_join_names(lost))}</b> "
                f"{'loses' if len(lost) == 1 else 'lose'} the curvature entirely once "
                f"the zeros are removed (" +
                "; ".join(f"{_esc(m)} p {_num(p, 3)}" for m, p, _ in survives
                          if m in lost) + "). On those measurements the whole-cohort "
                "knee is substantially detecting <em>edema present versus absent</em>, "
                "not a volume above which risk turns.")
        if kept:
            parts.append(f"<b>{_esc(_join_names(kept))}</b> "
                         f"{'keeps' if len(kept) == 1 else 'keep'} the curvature among "
                         f"measured values alone, so the knee is a genuine claim about "
                         f"magnitude.")
        verdict_html = _answer("Which of the three is the defensible claim",
                               "<p>" + " ".join(parts) + "</p>",
                               positive=bool(kept))

    return f"""
<h3>Zero inflation — "none at all" is a category, not a small number</h3>

<p>The spline's knots go at the 10th, 50th and 90th percentile of the observed values. When
a large share of the cohort sits at <em>exactly</em> zero, the lowest knot lands on zero and
the bend the model then finds is substantially the difference between <b>present</b> and
<b>absent</b> — a distinction a radiologist makes by looking, not by measuring.</p>

<ul>{"".join(lines)}</ul>

<p>So the claim is split into three and reported side by side: the whole-cohort fit, the
presence/absence rule on its own, and the curve refitted on patients with a non-zero value.</p>

{comparison_html}

{verdict_html}
"""


def _grade_pill(grade: str) -> str:
    colour = GRADE_COLOUR.get(grade, "var(--muted)")
    return f'<span class="grade" style="background:{colour}">{_esc(grade)}</span>'


def render_evidence(data: ThresholdReportData, facts: CohortFacts) -> str:
    """Section 3's graded verdict: the hierarchy, the scoring, what limits each.

    The methods paragraph is not decoration. A hierarchy invented after the
    numbers were seen is worth nothing; stating that it was fixed first, and
    printing the rules next to the results, is the whole reason a graded
    verdict is more defensible than a bare p-value gate.
    """
    v = facts.verdicts
    if not v.graded:
        return info_box(
            "The graded evidence table was not found. Re-run the thresholder "
            "notebook (section 8) to produce <code>28_threshold_evidence.csv</code>; "
            "until then this section reports the bare non-linearity test only.")

    criteria = data.table("evidence_criteria")
    reading = data.table("evidence_reading")
    ev_table = data.table("evidence")

    context_notes = {}
    if not ev_table.empty and "context_note" in ev_table.columns:
        context_notes = {r["column"]: ("" if pd.isna(r["context_note"])
                                       else str(r["context_note"]))
                         for _, r in ev_table.iterrows()}

    items = []
    for x in sorted(v.items,
                    key=lambda i: GRADE_ORDER.index(i.grade)
                    if i.grade in GRADE_ORDER else len(GRADE_ORDER)):
        limit = (f'<span class="limit">Held back by: {_esc(x.limiting)}</span>'
                 if x.limiting else
                 f'<span class="limit">All {len(evidence.CRITERIA)} criteria met.</span>')
        note = context_notes.get(x.column, "")
        extra = (f'<span class="limit">Read alongside: {_esc(note)}.</span>'
                 if note else "")
        items.append(f'<li>{_grade_pill(x.grade)} '
                     f'<span class="metric-name">{_esc(x.metric)}</span>{limit}{extra}</li>')
    grade_list = f'<ul class="grade-list">{"".join(items)}</ul>'

    # The columns that are reported but do not score — see evidence.py.
    context_table = ""
    if not ev_table.empty and {"knee_ci_ratio", "AUC"} <= set(ev_table.columns):
        context = pd.DataFrame({
            "Measurement": ev_table["metric"],
            "Evidence": ev_table["verdict"],
            "Knee interval width": [
                "—" if pd.isna(r) else f"{_num(r, 1)}× (hi ÷ lo)"
                for r in ev_table["knee_ci_ratio"]],
            "AUC": [_num(a) for a in ev_table["AUC"]],
        })
        if "nonlinearity_p_nonzero_only" in ev_table.columns:
            context["Non-linearity p, non-zero values only"] = [
                "n/a — no zero inflation" if pd.isna(p) else _num(p, 3)
                for p in ev_table["nonlinearity_p_nonzero_only"]]
        context_table = _table(context)

    return f"""
<h3>How strong is each of those claims?</h3>

<p>A bare <em>yes</em> is a weaker result than a graded one, and an easier one to attack.
It rests on a single p-value clearing 0.05 and says nothing about whether the knee sits
where the patients are, whether it survives a change of fitting scale, or whether it is
the 50%-risk crossing under a new name.</p>

{_key("Pre-specified, and that is the point:",
      "these five criteria and the four grades they map onto were fixed <b>before any "
      "verdict was read off</b>. A hierarchy invented after seeing the numbers grades "
      "nothing — it just restates them. The rules are printed here next to the results "
      "so a reader can check that the grades follow from them.")}

{_table(criteria, safe_html_cols=["Rule", "Source"]) if not criteria.empty else ""}

<p>The first three are <b>necessary</b>: if one fails, the threshold claim as stated is not
supportable, whatever the others say. The last two are <b>robustness</b> checks — they ask
whether the claim survives a choice we were not forced into. All five pass →
<em>strong</em>. All three necessary pass and one robustness check →
<em>moderate</em>. One necessary criterion (other than curvature) fails, or neither
robustness check passes → <em>fragile</em>. The curvature test itself fails, or two
necessary criteria do → <em>weak</em>.</p>

{grade_list}

{_table(reading) if not reading.empty else ""}

{_concrete(f"the graded answer is {_esc(v.grade_phrase())} — "
           f"{v.grade_sentences()}. That is a more defensible headline than "
           f"&quot;{v.count_phrase(verb=True)} a threshold&quot;, and it is the "
           f"sentence to present.")}

<h3>Two numbers that sit next to the grade and do not score it</h3>

<p>The five criteria are pass/fail, so they cannot see how <em>precisely</em> a knee is
located or whether the measurement carries any signal at all. Both are reported here and
both should be read before quoting a grade.</p>

{context_table}

{warning_box(
    "A wide knee interval or a low AUC does not lower the grade, because neither was "
    "among the pre-specified criteria — changing that now would be exactly the "
    "after-the-fact rule-making the hierarchy exists to prevent. Read them as context. "
    "If they are to gate the verdict, they have to be specified before the next run, "
    "not after this one.")}
"""


def _risk_curve_answer(facts: CohortFacts) -> str:
    """Section 3's answer paragraph, entirely from :class:`Verdicts`.

    Three shapes, because the honest sentence is different in each: none
    passed, some passed, all passed. The old text asserted the middle shape
    whatever the numbers said.
    """
    v = facts.verdicts
    if v.n_metrics == 0:
        return "<p>No risk-curve table was available for this run.</p>"

    head = f"<p><b>{v.count_phrase(verb=True)} a genuine threshold.</b> "
    if v.n_thresholds == 0:
        return (head + "Risk rises steadily throughout, so the reportable numbers are the "
                "risk crossings above — <em>&quot;risk reaches 30% at …&quot;</em> — not a "
                "threshold. <b>A negative here is a finding.</b> It says the routine "
                "practice of quoting a cut-point as though it were a threshold does not "
                "hold on this cohort.</p>")

    body = f"{v.turning_phrase()}. "
    if v.negative:
        body += (f"For {v.negative_names()} risk rises steadily, and the reportable numbers "
                 f"are the risk crossings above — <em>&quot;risk reaches 30% at …&quot;</em> "
                 f"— not a threshold. ")
    else:
        body += ("Every measurement cleared the curvature test, so on this cohort the "
                 "answer is positive across the board — which raises the opposite "
                 "question, how <em>strong</em> each of those thresholds is. ")
    body += ("<b>The strength of the claim differs sharply between them</b>, and the "
             "evidence table above is what separates them.")
    return head + body + "</p>"


def render_risk_curves(data: ThresholdReportData, facts: CohortFacts) -> str:
    reading = data.table("risk_reading")
    raw = data.table("risk_curves")

    verdicts = ""
    if not raw.empty:
        items = []
        for _, row in raw.iterrows():
            found = _truthy(row.get("knee_found"))
            items.append(
                f'<li class="{"yes" if found else ""}">'
                f'<span class="metric-name">{"🎯" if found else "➖"} {_esc(row["metric"])}</span>'
                f'{_esc(row.get("verdict", ""))}</li>')
        verdicts = f'<ul class="verdict-list">{"".join(items)}</ul>'

    crossing_note = base_rate_note = ""
    if not raw.empty and "risk_50_x" in raw.columns:
        bits = []
        for _, r in raw.iterrows():
            if pd.notna(r.get("risk_50_x")):
                bits.append(f"{r['metric']} reaches 50% risk at {_num(r['risk_50_x'], 3)}")
            elif pd.notna(r.get("risk_30_x")):
                bits.append(f"{r['metric']} reaches 30% at {_num(r['risk_30_x'], 3)} "
                            f"but never 50%")
        if bits:
            crossing_note = _concrete("; ".join(bits) + ".")

    # The 30% crossing is nearly the cohort's own high-grade rate, so it marks
    # where the curve leaves the base rate — not where risk becomes high.
    if np.isfinite(facts.prevalence) and abs(facts.prevalence - 0.30) < 0.05:
        base_rate_note = warning_box(
            f"<b>Read the 30% crossing as the base-rate crossing.</b> This cohort is "
            f"{_pct(facts.prevalence)} high grade, so “risk reaches 30%” is very nearly "
            f"“risk reaches the rate you would have assumed without looking at the scan”. "
            f"It marks where the measurement starts adding information, "
            f"which is worth knowing — but it is not a level at which anything becomes "
            f"worrying. The 50% crossings are the informative ones, and the reason two "
            f"measurements never reach one is itself a finding.")

    return f"""
<p><b>Question:</b> at what value does the probability of high grade start climbing fast?</p>

{_key("What is done:",
      "instead of splitting patients into two groups, the probability of high grade is modelled "
      "as a smooth curve across the whole range, and the curve is then examined for a bend.")}

<h3>The method in two sentences</h3>

<p>The curve is a <b>restricted cubic spline</b> — think of a flexible drawing ruler pinned at
three points (the 10th, 50th and 90th percentile of the observed values) and left to bend
naturally in between, with the ends held straight. Three pins is the right amount of freedom
for {facts.events} high-grade cases; more would let the curve chase noise in the tails.</p>

<h3>Which scale the test runs on</h3>

<p><b>"Is the risk bent?" is not a scale-free question.</b> A relationship that is a straight
line in log(volume) is a curve in volume — same patients, same model, opposite answer. Tumour
volume here is p&nbsp;=&nbsp;0.97 for non-linearity when fitted on the log scale and
p&nbsp;=&nbsp;0.02 when fitted in cc.</p>

{_key("The rule:",
      "the threshold is <b>reported</b> in cc, so the test <b>runs</b> in cc. Testing on the "
      "log scale and quoting a cut-point in cc is incoherent. The knots are already placed at "
      "percentiles of the data, which is what a log transform would have been for. The "
      "log-scale test is run anyway and reported as <code>nonlinearity_p_log_scale</code>, so "
      "the scale dependence is visible rather than a choice made quietly.")}

<h3>The two tests a threshold must pass</h3>

<ol>
<li><b>The curve is genuinely bent</b> — a formal test says the flexible curve fits better than
a straight line (p &lt; 0.05). If risk climbs in a straight line there is no point at which it
"starts" rising; it was rising throughout.</li>
<li><b>The steepest point is interior — counted in patients, not in axis units.</b> At least 5%
of the cohort must lie on each side of it. Edema volume runs 0–116 cc but half the cohort sits
below 4.5 cc, so a knee at 3.5 cc is 3% along the axis and at the 48th percentile of patients.
Judging it by axis position would discard a threshold sitting in the middle of the data purely
because the measurement has a long tail.</li>
</ol>

{warning_box(
    "Test 1 is the one that keeps this defensible, and the reason most published "
    "“thresholds” are overstated. Even a perfectly straight relationship produces an "
    "S-shaped probability curve, and every S-curve has a steepest point in the middle — at the "
    "place where risk passes 50%. Reporting that as a threshold restates the 50% mark under a "
    "different name. Requiring real curvature first is what separates a threshold from a "
    "midpoint.")}

<h3>The curves</h3>

{_figure(data.figures.get("risk_panel"),
         "Blue: fitted probability with its 95% band. Grey dots: the observed proportion in "
         "equal-sized patient groups — the reality check. Dots that do not follow the blue "
         "line mean the curve is overreading the data. Dash-dotted line: the cohort's own "
         "high-grade rate; a curve that never leaves it carries no information.")}

<h3>Each measurement, with its slope</h3>

<p>Left panel: the risk curve again. <b>Right panel: the slope of that curve</b> — risk added
per extra cc, or per unit of ADC. Its peak is the steepest-rise point. A right panel that is
flat, or highest at an edge, is the finding that there is no such point.</p>

{_figure_row(data.figure_groups.get("risk_curves", []))}

<h3>What came out</h3>

{verdicts}

{_table(reading) if not reading.empty else ""}

{crossing_note}

{base_rate_note}

{render_multiplicity(data)}

{render_zero_inflation(data)}

{render_evidence(data, facts)}

{_key("The bracketed numbers are bootstrap intervals.",
      f"The whole curve was refitted on {facts.n_boot_curve} resampled cohorts; this is the "
      f"range the answer moved across. Expect them wide — a steepest point is the peak of "
      f"the slope of a fitted curve, and publishing that width is what makes the number "
      f"defensible. (The cut-points in section 4 get a larger budget, "
      f"{facts.n_boot} resamples, because refitting a spline is far more expensive than "
      f"re-picking a maximum off a ROC table.)")}

{_answer("Answer — Balodis's first question", _risk_curve_answer(facts),
         positive=facts.n_thresholds > 0)}
"""


# ---------------------------------------------------------------------------
# 4 — single cut-points
# ---------------------------------------------------------------------------
def render_cutpoints(data: ThresholdReportData, facts: CohortFacts) -> str:
    reading = data.table("threshold_reading")
    full = data.table("thresholds")

    rules = pd.DataFrame([
        {"Rule": "youden", "Picks": "Highest sensitivity + specificity",
         "Assumes": "A missed WHO 2–3 costs the same as a false alarm",
         "Use when": "You want the field convention — nearly every paper uses it"},
        {"Rule": "closest_01", "Picks": "Point nearest a perfect test",
         "Assumes": "Same as Youden", "Use when": "Rarely — included to show the rule matters"},
        {"Rule": "equal_sens_spec", "Picks": "Where the two curves cross",
         "Assumes": "You want the two error rates equal",
         "Use when": "You want symmetric errors. It is <b>not</b> an optimum"},
        {"Rule": "spec_ge_90", "Picks": "Best sensitivity with specificity ≥ 90%",
         "Assumes": "A false alarm is expensive",
         "Use when": "<b>Ruling in</b> — a positive will change management"},
        {"Rule": "sens_ge_90", "Picks": "Best specificity with sensitivity ≥ 90%",
         "Assumes": "A miss is unacceptable",
         "Use when": "<b>Ruling out</b> — you cannot afford to miss one"},
    ])

    youden = full[full["rule"] == "youden"] if "rule" in full.columns else pd.DataFrame()
    concrete = ""
    if not youden.empty:
        r = youden.iloc[0]
        tp, fp, fn, tn = (r.get("TP"), r.get("FP"), r.get("FN"), r.get("TN"))
        if all(pd.notna(v) for v in (tp, fp, fn, tn)):
            # "34 of the 34+62" printed the arithmetic instead of doing it.
            positives = float(tp) + float(fn)
            negatives = float(fp) + float(tn)
            concrete = _concrete(
                f"applying <b>{_esc(r['metric'])} {_esc(r['operator'])}"
                f"{_num(r['cutoff'], 3)}</b> to every patient flags {_int(tp)} of the "
                f"{_int(positives)} high-grade tumours and misses {_int(fn)}, while "
                f"wrongly flagging {_int(fp)} of the {_int(negatives)} benign ones. That "
                f"is what {_pct(r['sensitivity'])} sensitivity and "
                f"{_pct(r['specificity'])} specificity mean in patients.")

    lit_rows = full[full["rule"] == "literature"] if "rule" in full.columns else pd.DataFrame()
    lit_html = ""
    if not lit_rows.empty:
        lit_display = pd.DataFrame({
            "Published cut-point": [f"{r['metric']} {r['operator']}{_num(r['cutoff'], 2)}"
                                    for _, r in lit_rows.iterrows()],
            "Source": lit_rows.get("source", ""),
            "Sensitivity here": [_pct(v) for v in lit_rows["sensitivity"]],
            "Specificity here": [_pct(v) for v in lit_rows["specificity"]],
            "J here": [_num(v) for v in lit_rows["youden_J"]],
        })
        lit_html = f"""
<h3>Published cut-points, scored on our patients</h3>

<p>These were chosen on other cohorts, so they never saw our data. They are the only rows in
this section needing <b>no</b> correction — and the reason validating someone else's cut-point
is a stronger design than deriving your own.</p>

{_table(lit_display)}
"""

    return f"""
<p><b>Question:</b> if the report has to say yes or no from one measurement, what value?</p>

{_key("What is done:",
      "every value that could serve as a dividing line is tried and scored. That list "
      "<em>is</em> the ROC curve. Choosing a cut-point means choosing one row from it — and "
      "the rule you choose by is a clinical decision, not a statistical one.")}

<h3>Five rules, and what each assumes</h3>

{_table(rules, safe_html_cols=["Use when"])}

{_key("The last two rows are the honest ones.",
      "Youden assumes missing a WHO 2–3 meningioma costs exactly what over-calling a benign one "
      "costs. Nobody agrees with that when asked directly. The constrained rules let you state "
      "your actual position — <em>&quot;I will not go below 90% specificity&quot;</em> — and take "
      "the best cut-point that satisfies it.")}

{concrete}

<h3>Why every number here is flattering</h3>

<p>Hundreds of candidate cut-points were tried and the best-scoring one kept — on the same
patients it is then scored on. Part of that score is signal, part is luck that will not repeat.
This is the winner's curse, and its size is measured directly: the cohort is resampled
hundreds of times, a cut-point re-chosen on each resample, then scored back on the original
patients. The average gap is the <b>optimism</b>.</p>

{_key("Read J (corr.), not J.",
      "The gap between them is the part of the performance that will not survive the next "
      "patient. A reviewer who knows this field will ask for exactly that column.")}

<h3>The trade-off at every cut-point</h3>

<p>Left: the ROC, with Youden's point marked — J is the vertical distance from the curve up
off the diagonal. Right: sensitivity and specificity against the cut-point itself, so you can
read what any choice costs.</p>

{_key("Look at how flat the peak is on the right.",
      "If the dotted J line is flat across a wide band, the &quot;optimal&quot; value is one of many "
      "near-identical choices and quoting three decimal places is false precision. The shaded "
      "band says the same in cut-point units.")}

{_figure_row(data.figure_groups.get("thresholds", []))}

<h3>All four compared</h3>

{_figure(data.figures.get("combined_roc"),
         "Discrimination only — independent of which cut-point you pick. A curve hugging the "
         "diagonal belongs to a measurement with no cut-point worth arguing about.")}

<h3>The cut-points</h3>

{_table(reading) if not reading.empty else '<p class="muted"><em>(table unavailable)</em></p>'}

{lit_html}

{_answer("Answer — what to take from this section",
         "<p>Read the cut-point 95% CI before quoting any single number. Where that interval "
         "spans an order of magnitude, the defensible sentence is not <em>&quot;the optimal "
         "threshold was X&quot;</em> but <em>&quot;no stable cut-point exists in this cohort&quot;</em>.</p>",
         positive=False)}
"""


# ---------------------------------------------------------------------------
# 5 — combinations
# ---------------------------------------------------------------------------
def primary_verdict(data: ThresholdReportData) -> pd.DataFrame:
    """The combination verdict every section quotes: shared denominator if present.

    One accessor so section 5, section 7 and section 9 cannot end up quoting
    different denominators for the same claim.
    """
    shared = data.table("shared_verdict")
    return shared if not shared.empty else data.table("combination_verdict")


def _verdict_rows(verdict: pd.DataFrame) -> pd.DataFrame:
    """The three-way comparison in one table, both J columns corrected alike."""
    if verdict.empty:
        return pd.DataFrame()
    if "n_used" not in verdict.columns:
        # The all-available-data run has no single n — that is its whole problem.
        verdict = verdict.assign(n_used=np.nan)
    return pd.DataFrame([
        {"Compared": "Best single criterion",
         "Which": str(_first(verdict, "best_single_rule", "")),
         "n": _int(_first(verdict, "n_used"), "varies"),
         "J as measured": _num(_first(verdict, "best_single_J")),
         "J after correction": _num(_first(verdict, "best_single_J_corrected"), 3)},
        {"Compared": "Best combined rule",
         "Which": str(_first(verdict, "best_rule", "")),
         "n": _int(_first(verdict, "n_used"), "varies"),
         "J as measured": _num(_first(verdict, "best_rule_J")),
         "J after correction": _num(_first(verdict, "best_rule_J_corrected"), 3)},
        {"Compared": "All four, uncut",
         "Which": "Logistic model on the raw numbers",
         "n": _int(_first(verdict, "n_used_continuous")),
         "J as measured": "—",
         "J after correction": (f"{_num(_first(verdict, 'continuous_J_equivalent'))} "
                                f"(AUC {_num(_first(verdict, 'continuous_AUC_corrected'))})")},
    ])


def _denominator_note(data: ThresholdReportData, shared: bool, shared_n: str,
                      facts: CohortFacts) -> str:
    """Why the head-to-head runs on one patient set, with the numbers that force it."""
    if not shared:
        return warning_box(
            "The shared-denominator comparison was not found. Every rule below is scored "
            "on its own complete cases, so the denominators differ between rows and the "
            "Youden values are not directly comparable. Re-run the thresholder notebook "
            "(section 6) to produce <code>32_shared_combination_verdict.csv</code>.")

    spread = ""
    menu = data.table("combination_menu")
    if not menu.empty and {"n_used", "rule_label"} <= set(menu.columns):
        lo = menu.loc[menu["n_used"].idxmin()]
        hi = menu.loc[menu["n_used"].idxmax()]
        spread = (f" On this cohort that runs from {_int(lo['n_used'])} patients "
                  f"({_esc(str(lo['rule_label']))}) to {_int(hi['n_used'])} "
                  f"({_esc(str(hi['rule_label']))}).")

    return _key(
        "Every rule below is scored on the same patients.",
        f"A rule scored on its own complete cases has its own denominator: an OR needs one "
        f"of its two measurements, an AND needs both, a single needs one.{spread} "
        f"<b>A Youden J compared across non-identical patient sets is not a comparison</b> "
        f"— part of any difference between two rows is the difference between the two "
        f"groups of patients. So the head-to-head runs on the {shared_n} patients who have "
        f"all four measurements. The cut-points are unchanged; only the patient set is.")


def _secondary_denominator_block(data: ThresholdReportData, shared: bool,
                                 full_verdict: pd.DataFrame,
                                 full_reading: pd.DataFrame, shared_n: str) -> str:
    """The all-available-data version, kept but demoted and labelled as such."""
    if not shared or full_verdict.empty:
        return ""
    return details_block(
        "Secondary: all available data (denominator varies by rule — not comparable "
        "across rows)",
        "<p>The same menu with every rule scored on everyone who has its own inputs. "
        "More patients per row, which is why it is kept: it is the better estimate of "
        "each rule taken <em>on its own</em>. It is the worse basis for ranking them "
        f"against each other, which is why the {shared_n}-patient version above is "
        "primary.</p>"
        + _table(_verdict_rows(full_verdict))
        + (_table(full_reading, max_rows=40) if not full_reading.empty else ""))


def render_combinations(data: ThresholdReportData, facts: CohortFacts) -> str:
    # Primary comparison: every rule on the same patients. The all-available-data
    # version is kept below it, because comparing a J computed on 318 patients
    # with one computed on 329 is not a comparison.
    shared_verdict = data.table("shared_verdict")
    full_verdict = data.table("combination_verdict")
    verdict = shared_verdict if not shared_verdict.empty else full_verdict
    shared = not shared_verdict.empty

    reading = (data.table("shared_reading") if shared
               else data.table("combination_reading"))
    full_reading = data.table("combination_reading")
    counts = data.table("count_score")
    missing = data.table("flag_missingness")

    best_rule = _first(verdict, "best_rule", "")
    gain = _first(verdict, "gain_vs_best_single")
    gain_apparent = _first(verdict, "gain_apparent")
    stability = _first(verdict, "winner_stability")
    single_corr = _first(verdict, "best_single_J_corrected")
    single_stability = _first(verdict, "best_single_stability")
    cont_auc = _first(verdict, "continuous_AUC_corrected")
    cont_equiv = _first(verdict, "continuous_J_equivalent")

    try:
        helped = float(gain) > 0.05 and float(stability) > 0.5
    except (TypeError, ValueError):
        helped = False

    verdict_display = _verdict_rows(verdict)
    shared_n = _int(_first(verdict, "n_used"))

    count_display = pd.DataFrame()
    ladder = ""
    if not counts.empty:
        count_display = pd.DataFrame({
            "Criteria met": counts["n_criteria_met"],
            "Patients": counts["n"],
            "High grade": counts["n_high_grade"],
            "Risk": [_pct(v) for v in counts["risk"]],
            "95% interval": [f"{_pct(lo)}–{_pct(hi)}"
                             for lo, hi in zip(counts["risk_lo"], counts["risk_hi"])],
        })
        usable = counts[counts["n"] > 0]
        if len(usable) >= 2:
            ladder = _concrete(
                "risk runs " +
                " → ".join(_pct(v) for v in usable["risk"]) +
                f" as a tumour goes from meeting none of the criteria to all "
                f"{int(usable['n_criteria_met'].max())}. That is a "
                f"{_num(float(usable['risk'].iloc[-1]) / max(float(usable['risk'].iloc[0]), 1e-9), 1)}-fold "
                f"spread across the score.")

    scorable = ""
    if not missing.empty:
        scorable = _key(
            "How many patients can be scored:",
            f"{_int(_first(missing, 'n_all_flags_observed'))} of "
            f"{_int(_first(missing, 'n_patients'))} have all "
            f"{_int(_first(missing, 'k_criteria'))} measurements "
            f"({_num(_first(missing, 'pct_scorable'), 0)}%). The rest get no count at all — a "
            "practical limit of any multi-criteria rule. Section 6 recovers them.")

    return f"""
<p><b>Question:</b> Balodis's second — do several cut-points together beat the best single one?</p>

{_key("The design decision that makes this answerable:",
      f"the cut-points are <b>frozen before being combined</b>. They come from section 4 and "
      f"only the way of joining them varies. Searching for the best cut-points <em>and</em> the "
      f"best combination at once would be thousands of comparisons on {facts.events} events, "
      f"and every 'improvement' found that way is noise.")}

<h3>Three ways to combine</h3>

<ul>
<li><b>AND</b> — flag only if both criteria are met. Fewer false alarms, more misses. A
<em>rule-in</em> rule.</li>
<li><b>OR</b> — flag if either is met. Catches more, alarms more. A <em>rule-out</em> rule.</li>
<li><b>Count</b> — count how many criteria the tumour meets and use that as a score.</li>
</ul>

{_denominator_note(data, shared, shared_n, facts)}

<h3>The figure that answers it</h3>

{_figure(data.figures.get("shared_combination_space" if shared else "combination_space"),
         "Left: every rule by its sensitivity and specificity; diamonds are single criteria, "
         "dotted diagonals join rules of equal usefulness. Right: rules ranked, with the uncut "
         "model as a dashed line.")}

{_key("How to read the left panel in one glance:",
      "a combination that helps sits <b>above and right</b> of the diamonds — off the diagonal "
      "band. A combination that slides <em>along</em> a diagonal has improved nothing; it has "
      "traded sensitivity for specificity, which you could have done by moving one cut-point.")}

<h3>Did it help?</h3>

{_table(verdict_display) if not verdict_display.empty else ""}

{_key("Both rows are corrected the same way, and that matters.",
      f"The best single criterion was also <em>chosen</em> — the best of four — so it "
      f"carries its own selection optimism "
      f"(<b>{_num(_first(verdict, 'best_single_selection_optimism'), 3)}</b>), almost "
      f"exactly the same size as the combined rule's "
      f"(<b>{_num(_first(verdict, 'selection_optimism'), 3)}</b>). Scoring a corrected "
      f"combination against an <em>uncorrected</em> single would credit the combination "
      f"with the single's own winner's curse and make the gain look far smaller than it "
      f"is. Both rows here are best-of-menu, both corrected by the same bootstrap.")}

{_key("Check three things, in order:",
      f"<b>(1)</b> the raw gain: <b>{_num(gain_apparent, 3)}</b>. <b>(2)</b> the gain once "
      f"both sides are corrected for having been selected: <b>{_num(gain, 3)}</b>. "
      f"<b>(3)</b> how often that same rule wins when the cohort is resampled: "
      f"<b>{_pct(stability)}</b>, against <b>{_pct(single_stability)}</b> for the winning "
      f"single. Below about half, the 'best combination' is a coin toss between "
      f"near-identical rules — and <b>that, not the size of the gain, is what sinks it "
      f"here</b>.")}

{_answer("Answer — Balodis's second question",
         f"<p>The best combination ({_esc(str(best_rule))}) reaches a corrected J of "
         f"<b>{_num(_first(verdict, 'best_rule_J_corrected'), 3)}</b> against "
         f"<b>{_num(single_corr, 3)}</b> for the best single criterion corrected the same "
         f"way — a gain of <b>{_num(gain, 3)}</b>. But it wins in only "
         f"<b>{_pct(stability)}</b> of resampled cohorts, so which combination is 'best' is "
         f"largely noise. Meanwhile the model using all four measurements <em>without</em> "
         f"cutting them scores a J-equivalent of <b>{_num(cont_equiv)}</b> — better than "
         f"any cut rule here. The honest reading: combining cut-points buys a small, "
         f"unstable improvement, and chopping the measurements up costs more than combining "
         f"them gains.</p>",
         positive=helped)}

<h3>The one combination worth presenting</h3>

<p>Not AND/OR — the <b>count</b>. How many of the four does this tumour meet? No logic gate to
remember, no arithmetic at the workstation.</p>

{_figure(data.figures.get("count_score"),
         "Each bar: the observed share of high-grade tumours among patients meeting that many "
         "criteria, with a 95% interval and the raw counts printed above. Dash-dotted line: "
         "the cohort rate.")}

{_table(count_display) if not count_display.empty else ""}

{ladder}

{scorable}

{_key("Why this works where the fancier rules do not:",
      "AND/OR forces every patient into one of two boxes, discarding the difference between a "
      "tumour that barely met one criterion and one that met all four. The count keeps that "
      "gradient. It is not a better <em>test</em> by the usual metrics — it is a better way of "
      "<em>saying</em> what the measurements imply, which is what a clinician deciding how "
      "worried to be actually needs.")}

{details_block(f"All rules, ranked — one denominator ({shared_n} patients)"
               if shared else "All rules, ranked",
               _table(reading, max_rows=40) if not reading.empty else "")}

{_secondary_denominator_block(data, shared, full_verdict, full_reading, shared_n)}
"""


# ---------------------------------------------------------------------------
# 6 — missing data
# ---------------------------------------------------------------------------
def render_stability(data: ThresholdReportData, facts: CohortFacts) -> str:
    stability_reading = data.table("stability_reading")
    risk_stability = data.table("risk_stability")
    counts_imputed = data.table("count_score_imputed")
    m = facts.m_draws

    knee_html = knee_note = ""
    if not risk_stability.empty:
        knee_html = _table(pd.DataFrame({
            "Measurement": risk_stability["metric"],
            "Threshold found in": [_pct(v) for v in risk_stability["knee_rate"]],
            "of datasets": [_int(v) for v in risk_stability["m_draws"]],
            "Typical location": ["—" if pd.isna(v) else _num(v, 3)
                                 for v in risk_stability["steepest_median"]],
            "Range": ["—" if pd.isna(lo) or pd.isna(hi) else f"{_num(lo, 3)}–{_num(hi, 3)}"
                      for lo, hi in zip(risk_stability["steepest_min"],
                                        risk_stability["steepest_max"])],
        }))
        best = risk_stability.sort_values("knee_rate", ascending=False).iloc[0]
        if float(best["knee_rate"]) > 0:
            knee_note = _concrete(
                f"{_esc(str(best['metric']))}'s threshold was found again in "
                f"{_pct(best['knee_rate'])} of the {m} filled-in datasets, landing between "
                f"{_num(best['steepest_min'], 3)} and {_num(best['steepest_max'], 3)}. "
                f"That is what reproducible looks like.")

    count_html = ""
    if not counts_imputed.empty:
        count_html = _table(pd.DataFrame({
            "Criteria met": counts_imputed["n_criteria_met"],
            "Patients (avg)": [_num(v, 0) for v in counts_imputed["n"]],
            "Risk": [_pct(v) for v in counts_imputed["risk"]],
            "Lowest of the datasets": [_pct(v) for v in counts_imputed["risk_min"]],
            "Highest of the datasets": [_pct(v) for v in counts_imputed["risk_max"]],
        }))

    return f"""
<p><b>Question:</b> everything so far quietly ignored patients whose measurement was missing.
Does the answer change once they are accounted for?</p>

{_key("Why this is not paranoia:",
      "if the patients without an ADC differ from those with one — scanned elsewhere, different "
      "protocol, sicker — every number above is shifted, and <b>no amount of resampling will "
      "reveal it</b>. Resampling only redraws the patients already present.")}

<h3>What multiple imputation does</h3>

<p>Each missing value is filled in {m} separate times, each time with a different plausible
value drawn from what that patient's other measurements imply. The entire analysis then runs
{m} times. Same answer every time → the missing data did not matter. Answer jumps around →
it did.</p>

{_table(pd.DataFrame([
    {"Uncertainty": "Sampling", "Measured by": "The bootstrap intervals in sections 3–5",
     "Answers": f"Would a different {facts.n} patients have changed the answer?"},
    {"Uncertainty": "Missing data", "Measured by": f"Spread across the {m} filled-in datasets",
     "Answers": "Would knowing the missing values have changed it?"},
]))}

<h3>Did the cut-points move?</h3>

{_figure(data.figures.get("stability"),
         "Shaded band: sampling uncertainty from the patients who had the measurement. Dots: "
         "the cut-point chosen in each filled-in dataset. Solid line: the cut-point you would "
         "otherwise report. Dots tight inside a wide band → sample size is the limitation. "
         "Dots scattered across or beyond it → the complete-case number alone is overconfident.")}

{_table(stability_reading) if not stability_reading.empty else ""}

{_key("The trap in this table:",
      "watch for the MICE <b>mean and median disagreeing</b>. That is a flat peak — the "
      "cut-point hops between two values from dataset to dataset and the average lands in the "
      "empty gap where none of them pointed. When they disagree, neither is a cut-point, and "
      "the disagreement is the finding.")}

<h3>Is the threshold reproducible?</h3>

<p>The strictest test here. The risk curve is refitted from scratch in every filled-in dataset,
and we count how often a genuine threshold is found <em>at all</em>.</p>

{_figure(data.figures.get("knee_stability"),
         "Each dot is one filled-in dataset. An empty panel is a result: in those datasets "
         "there was no threshold to find.")}

{knee_html}
{knee_note}

{_key("Reading the percentage:",
      "near 100% → the threshold is a property of the disease, robust to who happened to get "
      "that sequence. Near 0% → it was a property of the complete cases, and quoting it from "
      "the complete-case analysis alone would be an artefact.")}

<h3>The count score with everyone included</h3>

<p>In section 5 the count could only be applied to patients with all four measurements. Inside
a filled-in dataset nothing is missing, so every patient gets a count.</p>

{count_html}

{_figure(data.figures.get("count_score_imputed"))}

{warning_box(
    "This is a stability check, not formal pooling. Rubin's rules — the proper machinery for "
    "combining results across filled-in datasets — need estimates that behave in a particular "
    "way, and a cut-point chosen by taking a maximum does not. So the spread across datasets "
    "is a legitimate statement about how much the missing data matter, but it must never be "
    "printed as a confidence interval. Quote the wider of the two spreads, in words.")}
"""


# ---------------------------------------------------------------------------
# 7 — trade-offs, calibration, net benefit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NetBenefitFacts:
    """What the decision curve concluded, in the form section 9 quotes."""

    available: bool = False
    lo: float = float("nan")          # low end of the plotted threshold range
    hi: float = float("nan")
    winner: str = ""                  # best available over most of the range
    winner_share: float = float("nan")
    useless: tuple[str, ...] = ()     # never beat treat-all / treat-none
    rows: tuple[tuple, ...] = ()


def net_benefit_facts(data: ThresholdReportData) -> NetBenefitFacts:
    summary = data.table("net_benefit_summary")
    curve = data.table("net_benefit")
    if summary.empty or "pct_of_range_best_available" not in summary.columns:
        return NetBenefitFacts()

    strategies = summary[~summary["is_reference"].map(_truthy)]
    if strategies.empty:
        return NetBenefitFacts()
    top = strategies.sort_values("pct_of_range_best_available", ascending=False).iloc[0]

    lo = hi = float("nan")
    if not curve.empty and "threshold" in curve.columns:
        lo, hi = float(curve["threshold"].min()), float(curve["threshold"].max())

    return NetBenefitFacts(
        available=True, lo=lo, hi=hi,
        winner=str(top["strategy"]),
        winner_share=float(top["pct_of_range_best_available"]),
        useless=tuple(strategies.loc[
            strategies["pct_of_range_beating_references"] <= 0, "strategy"].astype(str)),
        rows=tuple(strategies.itertuples(index=False)),
    )


def render_calibration(data: ThresholdReportData) -> str:
    """Is the probability the right size? — the question AUC cannot answer."""
    table = data.table("calibration")
    if table.empty or "slope_corrected" not in table.columns:
        return info_box(
            "No calibration table was found. Re-run the thresholder notebook "
            "(section 6) to produce <code>40_calibration.csv</code>.")

    display = pd.DataFrame({
        "Model": [str(m).replace("_", " ") for m in table["model"]],
        "n": [_int(v) for v in table["n_used"]],
        "Predictors": [_int(v) for v in table["n_predictors"]],
        "Slope (corrected)": [_num(v) for v in table["slope_corrected"]],
        "Intercept (corrected)": [
            _num(v) if pd.notna(v) else f"— (apparent {_num(a)})"
            for v, a in zip(table.get("intercept_corrected", pd.Series(dtype=float)),
                            table.get("intercept_apparent", pd.Series(dtype=float)))],
        "Brier (corrected)": [_num(v, 3) for v in table.get("brier_corrected",
                                                            pd.Series(dtype=float))],
        "Source": table.get("source", ""),
    })

    uncut = table[table["model"].astype(str).str.contains("Uncut", case=False, na=False)]
    concrete = ""
    if not uncut.empty:
        r = uncut.iloc[0]
        slope = float(r["slope_corrected"]) if pd.notna(r["slope_corrected"]) else np.nan
        shrink = ((1.0 - slope) * 100.0) if np.isfinite(slope) else np.nan
        concrete = _concrete(
            f"the uncut four-measurement model has a bootstrap-corrected calibration slope "
            f"of <b>{_num(slope)}</b>. A slope below 1 means the predictions are too "
            f"spread out: on new patients the high risks would come in lower and the low "
            f"risks higher, by roughly "
            f"{_num(shrink, 0) if np.isfinite(shrink) else '—'}% of the distance from the "
            f"cohort's own rate. It is a mild shrinkage, not a broken model — but it is the "
            f"reason a predicted 70% should be quoted as 'high', not as '70%'.")

    missing_intercept = ""
    if ("intercept_corrected" in table.columns
            and table["intercept_corrected"].isna().any()):
        missing_intercept = warning_box(
            "The multivariable models' artifacts carry a bootstrap-corrected calibration "
            "<em>slope</em> but only an apparent <em>intercept</em>, so that cell reports "
            "the apparent value and says so. Producing a corrected intercept means "
            "re-running the modelling phase's own validation loop; it is not derivable "
            "from the stored coefficients, and it is not invented here.")

    return f"""
<h3>Calibration — is the probability the right size?</h3>

<p>Everything above this heading is <b>discrimination</b>: AUC, Youden J, sensitivity,
specificity. None of it can see whether a predicted 30% happens 30% of the time. A model
that says 90% where the truth is 40% has exactly the same AUC as one that says 40% — and
this report presents its outputs as probabilities throughout.</p>

{_key("Two numbers, two different failures:",
      "the <b>slope</b> is the outcome regressed on the model's own log-odds. Below 1 means "
      "the predictions are too extreme — the usual direction for a model scored on the "
      "patients it was fitted on. The <b>intercept</b> comes from a separate fit with the "
      "slope held at 1; it asks whether the risks are systematically too high or too low. "
      "Both are bootstrap-corrected exactly the way the AUCs are.")}

{_figure(data.figures.get("calibration"),
         "Dots: observed high-grade rate against mean predicted risk in equal-count bins, "
         "with 95% intervals. The diagonal is perfect calibration; dots below it mean the "
         "model promises more risk than the patients delivered.")}

{_table(display)}

{concrete}

{missing_intercept}
"""


def render_net_benefit(data: ThresholdReportData, facts: CohortFacts) -> str:
    """The decision curve — the direct answer to "what does the line cost"."""
    nb = net_benefit_facts(data)
    if not nb.available:
        return info_box(
            "No decision curve was found. Re-run the thresholder notebook (section 6) to "
            "produce <code>44_net_benefit_summary.csv</code>.")

    summary = data.table("net_benefit_summary")
    strategies = summary[~summary["is_reference"].map(_truthy)]
    display = pd.DataFrame({
        "Strategy": strategies["strategy"],
        "Best net benefit": [_num(v, 3) for v in strategies["max_net_benefit"]],
        "Beats treat-all and treat-none": [
            "never" if pd.isna(a) else f"{_pct(a)}–{_pct(b)} threshold"
            for a, b in zip(strategies["beats_references_from"],
                            strategies["beats_references_to"])],
        "…over this share of the range": [
            _num(v, 0) + "%" for v in strategies["pct_of_range_beating_references"]],
        "Best available strategy over": [
            _num(v, 0) + "%" for v in strategies["pct_of_range_best_available"]],
    })

    losers = ""
    if nb.useless:
        losers = warning_box(
            f"{_esc(_join_names(list(nb.useless)))} never beats treating everyone or "
            f"treating no one anywhere in the plotted range. A rule that cannot clear "
            f"those two references is not worth applying, whatever its sensitivity and "
            f"specificity look like in isolation.")

    return f"""
<h3>Net benefit — what is each strategy actually worth?</h3>

<p>This section's title question — <em>what does the line cost</em> — has a direct answer,
and it is not an AUC. Declare the <b>threshold probability</b> <em>t</em>: the risk above
which you would act. That single declaration fixes the exchange rate between a missed
high-grade tumour and a false alarm, and net benefit follows from it:</p>

<p style="text-align:center"><code>net benefit = TP/n − (FP/n) × t/(1 − t)</code></p>

{_key("What the units mean:",
      f"true positives per patient. At <em>t</em> = 0.30 you are saying one missed WHO 2–3 "
      f"is worth about 2.3 unnecessary alarms, so each false positive is charged 1/2.3 of a "
      f"true one. The point of the measure is that <b>a cut-point, a count score, a model, "
      f"treating everyone and treating no one all land on one axis</b> — which sensitivity "
      f"and specificity never do.")}

{_figure(data.figures.get("net_benefit"),
         "Higher is better. The two grey references are the strategies that need no test at "
         "all: treat everyone, and treat no one. A curve below both of them over the range "
         "you care about is a strategy that would do harm.")}

{_table(display)}

{_concrete(f"across threshold probabilities from {_pct(nb.lo)} to {_pct(nb.hi)}, "
           f"<b>{_esc(nb.winner)}</b> is the best available strategy over "
           f"{_num(nb.winner_share, 0)}% of the range. Below the cohort's own high-grade "
           f"rate of {_pct(facts.prevalence)}, treating everyone is hard to beat — which is "
           f"itself worth saying out loud in a surgical series.")}

{losers}

{_key("Why this belongs in the paper and not just in the appendix:",
      "a reviewer who accepts that your AUC is 0.69 can still ask whether anyone should "
      "act on it. The decision curve answers that in the reviewer's own terms: name the "
      "threshold you would use, read off whether the rule beats doing nothing. It is the "
      "one figure here that converts a statistic into a decision.")}
"""


def render_tradeoffs(data: ThresholdReportData, facts: CohortFacts) -> str:
    verdict = primary_verdict(data)
    models = data.model_aucs

    cont_auc = _first(verdict, "continuous_AUC_corrected")
    best_single_j = _first(verdict, "best_single_J")
    best_rule_j = _first(verdict, "best_rule_J_corrected")

    def _balanced_accuracy(j: Any) -> str:
        """``J/2 + 0.5`` — which is ``(sensitivity + specificity) / 2``.

        Deliberately **not** called an AUC. It is the area under the two-segment
        ROC of a rule that has already been dichotomised, so it sits on the same
        0.5–1.0 axis as a real AUC and is not the same quantity: a genuine AUC
        integrates over every cut-point, this one is fixed at the one chosen.
        Printing it in a column headed "AUC" next to 0.69 and 0.73 invited a
        comparison that flatters the cut-point.
        """
        try:
            f = float(j)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(f) else _num(f / 2.0 + 0.5)

    best_model_auc = best_model_preds = None
    if not models.empty:
        top = models.iloc[0]
        best_model_auc = top["AUC_corrected"]
        best_model_preds = top.get("n_predictors")

    comparison = pd.DataFrame([
        {"Approach": "One cut-point",
         "What you get": "Yes/no from a single measurement",
         "What it costs": "A tumour just under the line and one far under it become identical",
         "At the workstation": "Nothing — one number, one comparison",
         "Here": f"Balanced accuracy {_balanced_accuracy(best_single_j)}*"},
        {"Approach": "Two cut-points (AND/OR)",
         "What you get": "A slightly different balance of the same trade-off",
         "What it costs": "The same information loss twice, plus a rule to remember",
         "At the workstation": "Two numbers and a logic gate",
         "Here": f"Balanced accuracy {_balanced_accuracy(best_rule_j)}* — no real gain"},
        {"Approach": "Count score (0–4)",
         "What you get": "A five-level risk ladder, easy to apply and explain",
         "What it costs": "Still loses within-category detail; needs all four measured",
         "At the workstation": "Four numbers, counted",
         "Here": "Monotone gradient, stable across datasets"},
        {"Approach": "Risk curve",
         "What you get": "A probability for <em>any</em> value, not just above/below",
         "What it costs": "Not doable in your head — needs the published chart",
         "At the workstation": "A lookup",
         "Here": "The only honest output where there is no threshold"},
        {"Approach": "All four, uncut",
         "What you get": "The best use of exactly these four measurements",
         "What it costs": "A calculator",
         "At the workstation": "Four numbers entered",
         "Here": f"AUC {_num(cont_auc)}"},
    ])

    if best_model_auc is not None:
        comparison = pd.concat([comparison, pd.DataFrame([{
            "Approach": "Full multivariable model",
            "What you get": "Best available estimate — imaging plus clinical features",
            "What it costs": f"{_int(best_model_preds)} predictors, all recorded",
            "At the workstation": "The Streamlit calculator",
            "Here": f"AUC {_num(best_model_auc)}",
        }])], ignore_index=True)

    ladder = ""
    if best_model_auc is not None and cont_auc is not None:
        ladder = _answer(
            "The trade-off in one paragraph",
            f"<p>Performance climbs as you stop simplifying: one cut-point reaches a "
            f"balanced accuracy of {_balanced_accuracy(best_single_j)}, two cut-points add "
            f"essentially nothing ({_balanced_accuracy(best_rule_j)}), the same four "
            f"measurements uncut reach an AUC of "
            f"{_num(cont_auc)}, and the full model reaches {_num(best_model_auc)}. "
            f"<b>Every step is paid for in applicability.</b> The cut-point works in your head "
            f"in front of the scanner; the model needs a calculator and every predictor "
            f"recorded. Which is right depends entirely on where it will be used — and the "
            f"reason to publish cut-points anyway is that a rule nobody applies has an "
            f"effective AUC of 0.50.</p>",
            positive=False)

    model_table = info_box(
        "Multivariable model artifacts not found, so the comparison stops at the "
        "four-measurement model. Run meningioma-modelling.ipynb and regenerate to include them.")
    if not models.empty:
        worst = models.sort_values("EPV").iloc[0] if "EPV" in models.columns else None
        epv_note = ""
        if worst is not None and np.isfinite(float(worst.get("EPV", np.nan))):
            shrink = float(worst["AUC_apparent"]) - float(worst["AUC_corrected"])
            epv_note = _concrete(
                f"<b>{_esc(str(worst['model']).replace('_', ' '))}</b> spends "
                f"{_int(worst['n_predictors'])} predictors on {_int(worst['events'])} "
                f"events — <b>EPV {_num(worst['EPV'], 1)}</b>, the lowest here — and loses "
                f"{_num(shrink, 3)} of AUC to optimism correction "
                f"({_num(worst['AUC_apparent'])} → {_num(worst['AUC_corrected'])}), the "
                f"largest drop of any model in the table. That is what a thin EPV costs, "
                f"shown rather than argued.")
        model_table = f"""
<h3>The multivariable models, for reference</h3>
{_table(pd.DataFrame({
    "Model": [str(m).replace("_", " ") for m in models["model"]],
    "Predictors": models["n_predictors"],
    "Events": [_int(v) for v in models.get("events", pd.Series(dtype=float))],
    "EPV": [_num(v, 1) for v in models.get("EPV", pd.Series(dtype=float))],
    "AUC as measured": [_num(v) for v in models["AUC_apparent"]],
    "AUC corrected": [_num(v) for v in models["AUC_corrected"]],
}))}
<p class="muted">Read the corrected column. These come from the modelling notebook, so the
comparison above is not against a number you have to take on trust. <b>EPV</b> is events per
variable — how many high-grade tumours the model has to spend on each predictor. Below about
10 a logistic model starts fitting noise, and the gap between the two AUC columns is where
that shows up.</p>

{epv_note}
"""

    return f"""
<p>Everything above turns MRI numbers into a judgement about grade. So does the multivariable
model. They are not competitors — they answer to different constraints, and this section is
about which constraint you are under.</p>

{_table(comparison, safe_html_cols=["What you get", "What it costs"])}

<p class="figure-note">* <b>Balanced accuracy (dichotomised)</b> is (sensitivity +
specificity) / 2, equivalently J/2 + 0.5. It is <em>not</em> an AUC: a real AUC integrates
over every possible cut-point, this one is fixed at the cut-point already chosen. The two
numbers live on the same 0.5–1.0 scale, which is why they are shown together — and why the
label matters. Comparing a dichotomised rule's 0.64 with a model's 0.69 as though both were
AUCs understates how much the cut-point actually costs.</p>

{_key("Why cutting a measurement up always loses something:",
      "turning ADC into &quot;below 0.72 / above 0.72&quot; asserts that 0.71 and 0.40 are the same "
      "kind of suspicious, and that 0.71 and 0.73 are different kinds. Neither is true. That "
      "lost detail is exactly the gap between the cut-point rows and the model rows above — "
      "not a flaw in how the cut-points were chosen, but the price of having one at all.")}

{ladder}

{render_calibration(data)}

{render_net_benefit(data, facts)}

{model_table}

<h3>What should actually be reported</h3>

<ul>
<li><b>The risk curves</b> are the scientific finding and need no threshold to be useful.</li>
<li><b>The count score</b> is the applicable output — usable tomorrow, without a tool.</li>
<li><b>Single cut-points</b> go in with their intervals and the correction, described as what
they are: a convenience, not an optimum.</li>
<li><b>The multivariable model</b> belongs in the discussion as the ceiling these simpler
rules approach.</li>
</ul>

{_key("This comparison is deliberately unflattering to the cut-points.",
      "That is the point. A threshold paper that does not show what its threshold costs against "
      "a proper model is not making a defensible claim — and this is increasingly the first "
      "thing reviewers in this field ask for.")}
"""


# ---------------------------------------------------------------------------
# 8 — bottom line
# ---------------------------------------------------------------------------
def render_bottom_line(data: ThresholdReportData, facts: CohortFacts) -> str:
    headline = data.table("headline")
    notes = data.manifest.get("notes", [])

    notes_html = ""
    if notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in notes)
        notes_html = f'<h3>What this run concluded</h3><ul class="verdict-list">{items}</ul>'

    return f"""
<p>One row per measurement, pulling every section together. Write the results paragraph from
this table.</p>

{_table(headline) if not headline.empty else '<p class="muted"><em>(headline table unavailable)</em></p>'}

{_key("The columns, in order:",
      "<b>Threshold evidence</b> is section 3's graded verdict — strong, moderate, fragile "
      "or weak — and <b>What limits it</b> names the criterion holding it back. "
      "<b>Reproducible</b> is section 6 — did it survive the missing data. "
      "<b>Risk 30% / 50% at</b> are what you quote where there is no threshold. "
      "<b>Youden cut-point</b> and its interval are section 4. <b>Cut-point stable?</b> flags "
      "measurements whose cut-point hops between filled-in datasets.")}

{notes_html}
"""


# ---------------------------------------------------------------------------
# 9 — ESNR defence
# ---------------------------------------------------------------------------
def _multiplicity_answer(m: MultiplicityFacts) -> str:
    """Both corrections in one answer, so neither has to be conceded."""
    if not m.available:
        return ("Yes for the primary family, but the adjusted table was not found in this "
                "run. Re-run the thresholder notebook (section 3) to regenerate it.")
    all_survive = m.n_holm == m.n_tests
    head = (f"Yes, for the family that matters. We pre-specified the {m.n_tests} "
            f"non-linearity likelihood-ratio tests — one per measurement, one outcome — as "
            f"the primary family and report Holm-adjusted p-values alongside the raw ones. ")
    if all_survive:
        head += f"<b>All {m.n_tests} survive Holm.</b> "
    else:
        head += (f"{m.n_holm} of {m.n_tests} survive Holm"
                 + (f"; Holm drops {_esc(_join_names(list(m.dropped_by_holm)))}. "
                    if m.dropped_by_holm else ". "))
    if m.dropped_by_bonferroni:
        head += (f"Under the stricter Bonferroni correction "
                 f"{_esc(_join_names(list(m.dropped_by_bonferroni)))} would fall below "
                 f"significance, and we say so rather than leaving it to be found. ")
    else:
        head += (f"All {m.n_bonferroni} also survive the stricter Bonferroni correction. ")
    head += ("The selection rules and the combination menu are deliberately outside the "
             "family: no p-value is read off them, so they are paid for with optimism "
             "correction, not with an alpha adjustment.")
    return head


def _calibration_answer(data: ThresholdReportData) -> str:
    """Section 9's calibration answer, from the table rather than from memory."""
    table = data.table("calibration")
    if table.empty or "slope_corrected" not in table.columns:
        return ("Yes, and it is reported in section 7 — but the calibration table was not "
                "found in this run. Re-run the thresholder notebook to regenerate it.")
    uncut = table[table["model"].astype(str).str.contains("Uncut", case=False, na=False)]
    row = uncut.iloc[0] if not uncut.empty else table.iloc[0]
    slope = _num(row.get("slope_corrected"))
    best = table.dropna(subset=["slope_corrected"])
    span = ""
    if len(best) > 1:
        span = (f" Across the multivariable variants the corrected slope runs "
                f"{_num(best['slope_corrected'].min())}–"
                f"{_num(best['slope_corrected'].max())}, and the model with the most "
                f"predictors has the worst of them — which is what overfitting looks "
                f"like in a calibration column rather than in an argument.")
    return (f"We report it, which most threshold papers do not. The uncut four-measurement "
            f"model has a bootstrap-corrected calibration slope of <b>{slope}</b> — mildly "
            f"over-extreme predictions, the expected direction, not a broken model. "
            f"Calibration is the reason we present the count score as a five-level ladder "
            f"rather than quoting a point probability off a curve.{span}")


def _net_benefit_answer(data: ThresholdReportData, facts: CohortFacts) -> str:
    nb = net_benefit_facts(data)
    if not nb.available:
        return ("That is what section 7's decision curve answers, but it was not found in "
                "this run. Re-run the thresholder notebook to regenerate it.")
    tail = ""
    if nb.useless:
        tail = (f" And we say which strategies fail it: "
                f"{_esc(_join_names(list(nb.useless)))} never beats treating everyone or "
                f"treating no one anywhere in that range.")
    return (f"We answer that directly rather than leaving it to the reader. A decision curve "
            f"puts every strategy on one axis in units of true positives per patient, across "
            f"threshold probabilities from {_pct(nb.lo)} to {_pct(nb.hi)}. "
            f"<b>{_esc(nb.winner)}</b> is the best available strategy over "
            f"{_num(nb.winner_share, 0)}% of that range. Below the cohort's high-grade rate "
            f"of {_pct(facts.prevalence)} treating everyone is hard to beat, and we state "
            f"that too.{tail}")


def _net_benefit_evidence(nb: NetBenefitFacts) -> str:
    if not nb.available:
        return "Section 7, decision curve."
    return (f"Section 7, decision curve and net-benefit table: {_esc(nb.winner)} best over "
            f"{_num(nb.winner_share, 0)}% of the {_pct(nb.lo)}–{_pct(nb.hi)} threshold "
            f"range.")


def _multiplicity_evidence(m: MultiplicityFacts) -> str:
    if not m.available:
        return "Section 3, multiple-testing block."
    return (f"Section 3, multiple-testing block: {m.n_holm}/{m.n_tests} survive Holm, "
            f"{m.n_bonferroni}/{m.n_tests} survive Bonferroni. Sections 4–5 carry optimism "
            f"correction instead.")


def render_defence(data: ThresholdReportData, facts: CohortFacts) -> str:
    verdict = primary_verdict(data)
    models = data.model_aucs
    thresholds = data.table("thresholds")

    v = facts.verdicts
    mult = multiplicity_facts(data)
    nb = net_benefit_facts(data)
    # Balanced accuracy, not an AUC — see render_tradeoffs' footnote.
    _single_j = _first(verdict, "best_single_J")
    single_balanced = (_num(float(_single_j) / 2.0 + 0.5)
                       if pd.notna(_single_j) else "—")
    widest = ""
    if "cutoff_boot_lo" in thresholds.columns:
        y = thresholds[thresholds["rule"] == "youden"].dropna(subset=["cutoff_boot_lo"])
        if not y.empty:
            y = y.assign(width=y["cutoff_boot_hi"] / y["cutoff_boot_lo"].replace(0, np.nan))
            worst = y.sort_values("width", ascending=False).iloc[0]
            widest = (f"{worst['metric']} — cut-point {worst['operator']}"
                      f"{_sig(worst['cutoff'])}, 95% CI "
                      f"{_sig(worst['cutoff_boot_lo'])}–{_sig(worst['cutoff_boot_hi'])}")

    best_model_auc = _first(models, "AUC_corrected")
    cont_auc = _first(verdict, "combination_verdict") or _first(verdict, "continuous_AUC_corrected")

    turn_ev = f"{v.turning_phrase()}." if v.positive else ""
    if v.negative:
        turn_ev += f" {v.negative_names()} did not turn."

    if v.graded:
        first_answer = (
            "We do report it — but we also tested whether it is a <em>threshold</em>, and "
            "that test is what most papers skip. We go further: rather than a single "
            f"p&nbsp;&lt;&nbsp;0.05 gate, each measurement is scored against "
            f"<b>{len(evidence.CRITERIA)} criteria fixed before any verdict was read "
            f"off</b> — curvature, whether the knee sits among the patients, whether it is "
            f"merely the 50%-risk crossing renamed, whether it survives a change of "
            f"fitting scale, and whether it reproduces across the imputed datasets. The "
            f"claims grade out as {_esc(v.grade_phrase())}, not as a uniform yes.")
        first_evidence = (
            f"Section 3, evidence hierarchy: {v.grade_sentences()}. "
            f"Underlying curvature test: {v.count_phrase()} passed. {turn_ev}")
    else:
        first_answer = (
            "We do report it — but we also tested whether it is a <em>threshold</em>, and "
            "that test is what most papers skip. Reporting a cut-point as though it marked "
            "a jump in risk, when the risk is a straight line, is the specific error we set "
            "out to avoid.")
        first_evidence = f"Section 3: {v.count_phrase()} passed the curvature test. {turn_ev}"

    qas = [
        _qa("Why not just report the optimal ADC cut-off, like every other paper?",
            first_answer, first_evidence),

        _qa("Your AUCs are only 0.6–0.7. Is that worth publishing?",
            "Those are single imaging measurements used alone, and that is the honest ceiling "
            "for them. We say so, and we show the multivariable model reaching higher. The "
            "contribution is not a strong single predictor — it is showing which of these four "
            "widely quoted measurements actually carries a threshold and which do not.",
            f"Section 7: single cut-point balanced accuracy {single_balanced}, four "
            f"measurements uncut AUC {_num(cont_auc)}, full model "
            f"{_num(best_model_auc)}."),

        _qa("How do we know your cut-points are not overfitted?",
            "We measured exactly that. Every cut-point was re-derived on hundreds of resampled "
            "cohorts and re-scored on the original patients; the average gap is the optimism, "
            "and we subtract it. Both the raw and corrected values are in the tables.",
            f"Section 4: J and J (corr.) reported side by side, "
            f"{facts.n_boot} bootstrap resamples."),

        _qa("Your cut-point differs from the published one. Which is right?",
            "Different cohorts, different ROI placement, different scanners — cut-points do not "
            "transfer cleanly, and that is a finding rather than a discrepancy to explain away. "
            "We scored the published cut-points on our patients so the comparison is direct.",
            "Section 4, 'Published cut-points, scored on our patients'."),

        _qa("Did you correct for multiple testing?",
            _multiplicity_answer(mult),
            _multiplicity_evidence(mult)),

        _qa("Is 352 patients enough for threshold analysis?",
            f"For a model, yes — {facts.events} events supports the multivariable work. For a "
            f"cut-point, it is thinner than people assume, because a cut-point is chosen by "
            f"taking a maximum and maxima are noisy. That is precisely why we publish the "
            f"bootstrap intervals rather than a bare number, and why for some measurements our "
            f"conclusion is that no stable cut-point exists.",
            f"Section 4: widest interval — {widest or 'see the cut-point table'}."),

        _qa("What about the missing data?",
            f"Every analysis was repeated on {facts.m_draws} multiply-imputed datasets, and for "
            f"the threshold we report how often it was found again at all. That separates "
            f"sampling noise from missing-data uncertainty, which a bootstrap alone cannot do.",
            "Section 6, including the reproducibility rate per measurement."),

        _qa("Dichotomising continuous variables is bad practice. Why do it?",
            "Agreed, and we quantify what it costs rather than defending it. The comparison "
            "table shows the loss explicitly. We publish the cut-points because a rule that "
            "needs a calculator will not be used at the scanner, and an unused rule has an "
            "effective AUC of 0.50 — but we do not pretend the cut-point is the better "
            "statistic.",
            "Section 7 comparison table."),

        _qa("You report discrimination everywhere. Is the model calibrated?",
            _calibration_answer(data),
            "Section 7, calibration figure and table — slope and intercept, "
            "bootstrap-corrected the same way the AUCs are."),

        _qa("Granted the AUC. Should anyone actually act on this?",
            _net_benefit_answer(data, facts),
            _net_benefit_evidence(nb)),

        _qa("Why the count score rather than a proper risk model?",
            "Both. The model is in the modelling arm of the study and performs better; we say "
            "so. The count score is offered as the applicable version — five levels, no "
            "arithmetic, and a monotone risk gradient that held across every imputed dataset.",
            "Section 5 count-score figure; section 6 for its stability."),

        _qa("Did combining thresholds improve accuracy? That was one of our questions.",
            f"Not usefully, and we can show why rather than just asserting it. The best "
            f"combination gained J {_num(_first(verdict, 'gain_vs_best_single'), 3)} over the "
            f"best single criterion once <em>both</em> were corrected for having been "
            f"selected off a menu — and the same combination won in only "
            f"{_pct(_first(verdict, 'winner_stability'))} of resampled cohorts, so the "
            f"ranking itself is largely noise. The uncut four-measurement model beats every "
            f"cut rule regardless.",
            f"Section 5 verdict table, all rules on the same "
            f"{_int(_first(verdict, 'n_used'), 'available')} patients: corrected J "
            f"{_num(_first(verdict, 'best_rule_J_corrected'), 3)} combined vs "
            f"{_num(_first(verdict, 'best_single_J_corrected'), 3)} single; uncut "
            f"J-equivalent {_num(_first(verdict, 'continuous_J_equivalent'))}."),
    ]

    return f"""
<p>The questions a commission actually asks, with the number that answers each. The evidence
line under every answer points at where in this document it lives.</p>

{"".join(qas)}

{_key("The general shape of the defence:",
      "every criticism a reviewer can make of threshold analysis — overfitting, multiplicity, "
      "dichotomisation, missing data, small n — is <b>already measured and reported here</b>. "
      "The strongest position is not that the cut-points are good; it is that we know exactly "
      "how good they are and have said so.")}

{_key("What not to say:",
      "do not claim a threshold for a measurement where section 3 said no. Do not quote "
      "sensitivity and specificity without the interval. Do not quote PPV or NPV as though "
      "they transfer outside a surgical series. Each of those is the question you will be "
      "asked next.")}
"""


# ---------------------------------------------------------------------------
# 10 — reference
# ---------------------------------------------------------------------------
def render_reference(data: ThresholdReportData, facts: CohortFacts) -> str:
    verdict = primary_verdict(data)
    nb_facts = net_benefit_facts(data)
    risk = data.table("risk_curves")
    thresholds = data.table("thresholds")

    youden_row = (thresholds[thresholds["rule"] == "youden"].iloc[0]
                  if "rule" in thresholds.columns
                  and not thresholds[thresholds["rule"] == "youden"].empty else None)
    sens = _pct(youden_row["sensitivity"]) if youden_row is not None else "—"
    spec = _pct(youden_row["specificity"]) if youden_row is not None else "—"
    ppv = _pct(youden_row["PPV"]) if youden_row is not None else "—"
    npv = _pct(youden_row["NPV"]) if youden_row is not None else "—"
    auc_best = _num(risk["AUC"].max()) if "AUC" in risk.columns else "—"

    cards = [
        _ref("Sensitivity",
             f"Of the {facts.events} tumours that really are WHO 2–3, the share the rule flags.",
             "Quote it when a miss is the error you care about.",
             f"{sens} for the best single cut-point."),
        _ref("Specificity",
             f"Of the {facts.benign} benign tumours, the share the rule correctly leaves alone.",
             "Quote it when a false alarm changes management.",
             f"{spec} at the same cut-point."),
        _ref("PPV",
             "If the rule fires, the chance the tumour really is high grade.",
             "Only quote it for a population with the same prevalence as this one. It falls "
             "sharply in an unselected population.",
             f"{ppv} in this surgical series ({_pct(facts.prevalence)} high grade)."),
        _ref("NPV",
             "If the rule is negative, the chance the tumour really is benign.",
             "Same prevalence caveat as PPV.",
             f"{npv} here."),
        _ref("ROC curve",
             "Every possible cut-point plotted as sensitivity against false alarms — one point "
             "per candidate line.",
             "Use it to compare measurements, not to pick a cut-point.",
             "Section 4, right-hand figures."),
        _ref("AUC",
             "The chance a random high-grade tumour looks more suspicious than a random benign "
             "one. 0.50 is a coin flip.",
             "The headline discrimination number. Independent of cut-point.",
             f"{auc_best} for the best single measurement here."),
        _ref("Youden's J",
             "Sensitivity + specificity − 1. Zero for a useless test, one for perfect.",
             "The default way of picking a cut-point. Assumes a miss and a false alarm cost "
             "the same — say so when you quote it.",
             f"{_num(_first(verdict, 'best_single_J'))} for the best single criterion."),
        _ref("Cut-point vs threshold",
             "<b>Cut-point</b>: a line you draw to get yes/no. <b>Threshold</b>: a place where "
             "the risk itself changes behaviour.",
             "Deliberately not synonyms in this study. A measurement can have the first "
             "without the second, which is why section 3 tests for the second separately.",
             (f"{facts.verdicts.grade_phrase()} on section 3's evidence hierarchy"
              if facts.verdicts.graded
              else f"{facts.verdicts.count_phrase()} passed the curvature test.")),
        _ref("Evidence hierarchy",
             f"{len(evidence.CRITERIA)} criteria — curvature, knee interiority, knee ≠ the "
             f"50%-risk point, scale robustness, MICE reproducibility — fixed before any "
             f"verdict was read off, mapping onto four grades.",
             "Replaces the single p &lt; 0.05 gate. Quote the grade, not the p-value, and "
             "name the criterion that limited it.",
             (facts.verdicts.grade_phrase() if facts.verdicts.graded else "—")),
        _ref("Restricted cubic spline",
             "A flexible curve made of cubic pieces joined at a few points, held straight beyond "
             "the outer ones.",
             "How the risk curve is fitted. The 'restricted' part stops it flying off in the "
             "tails where there are few patients.",
             "Three knots, at the 10th / 50th / 90th percentile."),
        _ref("Likelihood-ratio test",
             "A formal comparison of the flexible curve against a straight line.",
             "The gatekeeper for calling something a threshold. Small p → the relationship "
             "really is bent.",
             "Section 3, 'Non-linearity p' column."),
        _ref("Bootstrap",
             "Resampling the cohort with replacement hundreds of times and redoing the analysis "
             "each time.",
             "How every interval in this report was produced. The spread shows how much the "
             "answer depends on which patients you recruited.",
             f"{facts.n_boot} resamples per cut-point."),
        _ref("Optimism / winner's curse",
             "How much better a result looks on the data used to find it than it will on new "
             "patients.",
             "Always subtract it before quoting a cut-point you derived yourself. Never needed "
             "for a cut-point taken from someone else's paper.",
             "The gap between the J and J (corr.) columns."),
        _ref("Wilson interval",
             "A 95% interval for a percentage that stays sensible when the count is small or "
             "near 0% / 100%.",
             "Used on every sensitivity, specificity, PPV and NPV here.",
             "The brackets in every accuracy table."),
        _ref("Multiple imputation (MICE)",
             "Filling in missing values several times over with different plausible values, "
             "then running the analysis on each version.",
             "The only way to test whether missing data shifted your answer. A bootstrap "
             "cannot do it.",
             f"{facts.m_draws} imputed datasets."),
        _ref("Complete case",
             "An analysis using only patients who have every value it needs.",
             "The default, and the thing section 6 checks. Simple, but discards patients and "
             "can bias the answer if they differ.",
             "Sections 3–5 are complete-case; section 6 is not."),
        _ref("Prevalence",
             "How common the outcome is in the cohort you are working in.",
             "The reason PPV and NPV do not transfer between settings. Always state it next "
             "to them.",
             f"{_pct(facts.prevalence)} here — a surgical series, far above population rate."),
        _ref("Calibration slope",
             "The outcome regressed on the model's own predicted log-odds. 1.0 is perfect; "
             "below 1 means the predictions are too spread out.",
             "Quote it whenever you present a probability rather than a yes/no. AUC cannot "
             "see it at all.",
             f"{_num(_first(data.table('calibration'), 'slope_corrected'))} "
             f"(bootstrap-corrected) for the uncut four-measurement model."),
        _ref("Net benefit",
             "True positives per patient, minus false positives priced at t/(1−t), where "
             "t is the risk above which you would act.",
             "The one measure that puts a cut-point, a score, a model and doing nothing on "
             "the same axis. Use it to answer &quot;should anyone act on this?&quot;",
             (f"{_esc(nb_facts.winner)} best over {_num(nb_facts.winner_share, 0)}% of the "
              f"{_pct(nb_facts.lo)}–{_pct(nb_facts.hi)} range."
              if nb_facts.available else "Section 7, decision curve.")),
        _ref("Risk crossing",
             "The value at which the fitted risk curve passes a chosen level, e.g. 30% or 50%.",
             "What you quote instead of a threshold when the risk rises smoothly. Far more "
             "stable than a steepest point.",
             "Section 3, 'Risk reaches 30% / 50%' columns."),
    ]

    return f"""
<p>Every term used in this document: one line for what it is, one for when you use it, and
what it was on this cohort. Written to be re-read the morning of a presentation.</p>

{"".join(cards)}
"""


# ---------------------------------------------------------------------------
# 11 — caveats
# ---------------------------------------------------------------------------
def render_caveats(data: ThresholdReportData, facts: CohortFacts) -> str:
    v = facts.verdicts
    # Named from this run's grades, not from a remembered earlier answer.
    if v.graded:
        quotable = [x for x in v.items
                    if x.grade in (evidence.GRADE_STRONG, evidence.GRADE_MODERATE)]
        held_back = [x for x in v.items if x not in quotable]
        rule_head = ("Quote a threshold only where section 3's evidence hierarchy graded it "
                     "<b>strong</b> or <b>moderate</b>. ")
        if quotable:
            rule_head += f"On this run that is {_esc(_join_names([x.metric for x in quotable]))}."
        else:
            rule_head += "On this run no measurement reached either grade."
        if held_back:
            rule_head += (" Describe " +
                          _esc(_join_names([f"{x.metric} ({x.grade})" for x in held_back])) +
                          " as exploratory, and say what limited " +
                          ("it" if len(held_back) == 1 else "each") + ".")
        threshold_rule = rule_head
    else:
        reproduced = [x for x in v.positive
                      if np.isfinite(x.reproduced) and x.reproduced >= MICE_REPRODUCIBLE_CUT]
        threshold_rule = (
            "Only call something a threshold where section 3 said yes <em>and</em> section 6 "
            "said it reproduced. " +
            (f"On this run that is {_esc(_join_names([x.metric for x in reproduced]))}."
             if reproduced else "On this run no measurement cleared both."))

    m = multiplicity_facts(data)
    if m.available:
        multiplicity_caveat = (
            f"Many looks at one outcome, but only one pre-specified family. "
            f"The {m.n_tests} non-linearity tests are Holm-adjusted "
            f"({m.n_holm} of {m.n_tests} survive"
            + (f"; Bonferroni would drop "
               f"{_esc(_join_names(list(m.dropped_by_bonferroni)))}"
               if m.dropped_by_bonferroni else "") +
            "). The selection rules and the combination menu are not tests and carry no "
            "adjusted p-values — read their intervals and their optimism correction "
            "instead.")
    else:
        multiplicity_caveat = (
            "Many looks at one outcome. The adjusted table is missing from this run, so "
            "treat the p-values as descriptive and lean on the intervals.")

    # The old text said "a quarter of this cohort" from an earlier run. Read it.
    zero = data.table("zero_share")
    zero_caveat = ("<b>Check zero inflation before quoting any threshold.</b> No "
                   "zero-inflation table was found for this run.")
    if not zero.empty and "zero_inflated" in zero.columns:
        hits = zero[zero["zero_inflated"].map(_truthy)]
        if not hits.empty:
            share = "; ".join(
                f"{_esc(r['metric'])} {_num(r['pct_zero'], 1)}%" for _, r in hits.iterrows())
            zero_caveat = (
                f"{_esc(_join_names(list(hits['metric'])))} "
                f"{'is' if len(hits) == 1 else 'are'} zero-inflated ({share} of measured "
                f"patients sit at exactly zero). 'None at all' behaves like a category, not "
                f"a small number — see the three-way comparison in section 3 before quoting "
                f"a threshold for {'it' if len(hits) == 1 else 'either'}.")
        else:
            zero_caveat = ("No measurement in this run was zero-inflated, so the "
                           "whole-cohort splines answer the question they appear to.")

    return f"""
<ul>
<li><b>Every cut-point derived here is flattering.</b> Each was the best of hundreds of
candidates on the same patients it is scored on. Quote the corrected column, or validate
someone else's cut-point instead.</li>

<li><b>A threshold and a cut-point are different claims.</b> {threshold_rule}</li>

<li><b>PPV and NPV do not transfer.</b> This is a surgical series at {_pct(facts.prevalence)}
high grade. Sensitivity and specificity carry to other settings; predictive values do not.</li>

<li><b>Two spreads, not one.</b> Sections 3–5 measure sampling uncertainty only. Section 6 adds
missing-data uncertainty. Quote the wider.</li>

<li><b>{multiplicity_caveat}</b></li>

<li><b>{zero_caveat}</b></li>

<li><b>Nothing here feeds the model.</b> A cut-point estimated on this cohort has already seen
the answer, so it is never written back into the cleaned dataset. Published cut-points are the
exception — they came from other patients.</li>
</ul>
"""


# ---------------------------------------------------------------------------
# 12 — the manuscript block
# ---------------------------------------------------------------------------
def _literature_agreement(data: ThresholdReportData) -> list[str]:
    """Published cut-points that fall inside our own bootstrap interval.

    The strongest single result for a poster: a cut-point chosen on someone
    else's patients landing inside the range ours moved across. Computed here
    rather than remembered, because the interval moves with every run.
    """
    thresholds = data.table("thresholds")
    risk = data.table("risk_curves")
    if thresholds.empty or risk.empty or "rule" not in thresholds.columns:
        return []
    knee = risk.set_index("column")
    out = []
    for _, r in thresholds[thresholds["rule"] == "literature"].iterrows():
        col = r.get("column")
        if col not in knee.index:
            continue
        row = knee.loc[col]
        lo, hi, cut = (row.get("steepest_lo"), row.get("steepest_hi"), r.get("cutoff"))
        if pd.isna(lo) or pd.isna(hi) or pd.isna(cut):
            continue
        if float(lo) <= float(cut) <= float(hi):
            source = str(r.get("source", ""))
            author = source.split(",")[0].strip()
            year = re.search(r"\b(19|20)\d{2}\b", source)
            cite = f"{author} {year.group(0)}" if (author and year) else author
            out.append(f"{r['metric']} {_num(cut, 2)} ({_esc(cite)}) in our "
                       f"{_num(lo, 2)}–{_num(hi, 2)}")
    return out


def render_manuscript(data: ThresholdReportData, facts: CohortFacts) -> str:
    """A short, copy-paste block for the ESNR manuscript. Every number templated.

    Deliberately the only section written to be lifted verbatim. It carries no
    explanation — the rest of the document is the explanation — so it stays
    inside the word budget of an abstract or a methods paragraph.
    """
    v = facts.verdicts
    verdict = primary_verdict(data)
    mult = multiplicity_facts(data)
    nb = net_benefit_facts(data)
    counts = data.table("count_score")
    calibration = data.table("calibration")
    zero = data.table("zero_share")
    nonzero = data.table("nonzero_curves")
    summary = data.table("cohort_summary")

    years = ""
    first, last = _first(summary, "accrual_first_year"), _first(summary, "accrual_last_year")
    if pd.notna(first) and pd.notna(last):
        years = f" ({_int(first)}–{_int(last)})"

    ladder = ""
    if not counts.empty:
        usable = counts[counts["n"] > 0]
        if len(usable) >= 2:
            ladder = (f"Observed risk rose monotonically with the count of criteria met, "
                      f"{_pct(usable['risk'].iloc[0])} to {_pct(usable['risk'].iloc[-1])} "
                      f"across 0–{_int(usable['n_criteria_met'].max())}. ")

    # ADC's knee coinciding with the 50%-risk crossing is the sharpest negative
    # result in the run, so it is named rather than folded into the grades.
    restated = [x.metric for x in v.items if x.limiting == "Knee ≠ 50%-risk point"]
    restated_text = ""
    if restated:
        restated_text = (f"{_esc(_join_names(restated))}'s steepest-rise point fell inside "
                         f"the bootstrap interval of the 50%-risk crossing, restating that "
                         f"crossing rather than marking a threshold. ")

    zero_text = ""
    if not zero.empty and not nonzero.empty and "zero_inflated" in zero.columns:
        hits = zero[zero["zero_inflated"].map(_truthy)]
        lost = nonzero[nonzero["nonlinearity_p"] >= evidence.ALPHA] \
            if "nonlinearity_p" in nonzero.columns else pd.DataFrame()
        if not hits.empty and not lost.empty:
            share = _num(hits["pct_zero"].max(), 0)
            zero_text = (
                f"{_esc(_join_names(list(lost['metric'])))} are zero-inflated ({share}% "
                f"zero) and their non-linearity vanished when refitted on non-zero values "
                f"only ({', '.join('p ' + _num(p, 2) for p in lost['nonlinearity_p'])}), so "
                f"the knee largely reflects edema present versus absent. ")

    agreement = _literature_agreement(data)
    agreement_text = ""
    if agreement:
        agreement_text = (f"Published cut-points fell inside our intervals: "
                          f"{'; '.join(agreement)}. ")

    slope = ""
    if not calibration.empty:
        uncut = calibration[calibration["model"].astype(str)
                            .str.contains("Uncut", case=False, na=False)]
        if not uncut.empty:
            slope = (f", calibration slope {_num(uncut.iloc[0]['slope_corrected'])} "
                     f"after correction")

    nb_text = ""
    if nb.available:
        nb_text = (f"On decision-curve analysis the uncut model led over "
                   f"{_num(nb.winner_share, 0)}% of threshold probabilities "
                   f"({_pct(nb.lo)}–{_pct(nb.hi)}); no single cut-point led anywhere. ")

    holm = ""
    if mult.available:
        holm = (f"{mult.n_holm} of {mult.n_tests} non-linearity tests — the pre-specified "
                f"primary family — survived Holm adjustment"
                + (f"; Bonferroni would drop "
                   f"{_esc(_join_names(list(mult.dropped_by_bonferroni)))}"
                   if mult.dropped_by_bonferroni else "") + ". ")

    return f"""
<p class="muted">Short enough to paste. Every number below is templated from this run, so
re-running the notebook rewrites this block rather than leaving it stale. Nothing here is
explained — the rest of the document is the explanation.</p>

<div class="manuscript">

<p><b>Aim.</b> To test whether pre-operative MRI measurements of meningioma carry a genuine
<em>threshold</em> for WHO grade 2–3 — a value at which risk changes behaviour — rather
than merely a cut-point, and whether combining cut-points beats the best single one.</p>

<p><b>Methods.</b> {facts.n} operated patients with histological grading{years};
{facts.events} ({_pct(facts.prevalence, 1)}) WHO grade 2–3. Mean ADC, tumour volume,
peritumoral edema volume and the edema index were modelled as restricted cubic splines
(3 knots) in clinical units, with a likelihood-ratio test for non-linearity. Cut-points came
from five pre-specified selection rules, bootstrap-corrected for optimism
({facts.n_boot} resamples). Threshold claims were graded against
{len(evidence.CRITERIA)} criteria fixed before the verdicts were read: non-linearity, knee
interiority, distinctness from the 50%-risk crossing, robustness to fitting scale, and
reproducibility across {facts.m_draws} multiply-imputed datasets. Combination rules were
compared on one denominator (all four measurements present); calibration and net benefit
were assessed for the uncut model.</p>

<p><b>Results.</b> {v.count_phrase(verb=True)} non-linear risk, but graded unevenly:
{v.grade_sentences(compact=True)}. {restated_text}{zero_text}{agreement_text}Best single criterion:
optimism-corrected Youden J {_num(_first(verdict, 'best_single_J_corrected'), 2)}; best
combination {_num(_first(verdict, 'best_rule_J_corrected'), 2)} — a gain of
{_num(_first(verdict, 'gain_vs_best_single'), 2)} that held in only
{_pct(_first(verdict, 'winner_stability'))} of resampled cohorts. The same four
measurements uncut reached AUC {_num(_first(verdict, 'continuous_AUC_corrected'))}{slope},
beating every dichotomised rule. {ladder}{nb_text}{holm}</p>

<p><b>Conclusion.</b> These measurements separate meningioma grades but do not support the
threshold language usually applied to them: the claims are graded rather than categorical,
and the edema ones rest on presence versus absence more than on magnitude. Combining
cut-points bought no stable improvement, and dichotomising cost more than combining gained.
The defensible outputs are the risk curves and a count of criteria met.</p>

<p><b>Assumptions and limitations.</b> Retrospective single-centre surgical series: at
{_pct(facts.prevalence)} high grade the predictive values do not transfer, though
sensitivity and specificity do. All derived cut-points are optimistically biased and
reported corrected. The {facts.m_draws} imputations are a stability check, not Rubin
pooling — a cut-point chosen by maximisation does not meet Rubin's conditions — so the
across-draw spread is a range, not a confidence interval. Grades depend on the
pre-specified criteria; knee-interval width and AUC are reported but do not score.</p>

</div>

{warning_box(
    f"Before this block goes anywhere: section 0 carries {len(METHODS_GAPS)} "
    f"<code>TODO: ANDY</code> placeholders — WHO edition, b-values, ROI protocol, "
    f"volumetry, ethics approval and the rest. The Methods paragraph above is not "
    f"complete without them, and none of them can be derived from the data.")}
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
        section_block("0 · Methods and cohort", render_methods(data, facts), open=True),
        section_block("1 · Three questions that get confused",
                      render_questions(data, facts), open=True),
        section_block("2 · What we measured, and how much it overlaps",
                      render_measurements(data, facts), open=True),
        section_block("3 · Where does risk climb fastest?",
                      render_risk_curves(data, facts), open=True),
        section_block("4 · Where should the line be drawn?",
                      render_cutpoints(data, facts), open=True),
        section_block("5 · Do several criteria together do better?",
                      render_combinations(data, facts), open=True),
        section_block("6 · Does it survive the missing data?",
                      render_stability(data, facts), open=True),
        section_block("7 · Trade-offs against the other approaches",
                      render_tradeoffs(data, facts), open=True),
        section_block("8 · The bottom line", render_bottom_line(data, facts), open=True),
        section_block("9 · Defending this at ESNR", render_defence(data, facts), open=True),
        section_block("10 · Reference — every term, one line each",
                      render_reference(data, facts), open=False),
        section_block("11 · What not to claim", render_caveats(data, facts), open=False),
        section_block("12 · For the manuscript — copy-paste",
                      render_manuscript(data, facts), open=True),
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
    # Appended after the shared stylesheet so the threshold-only classes win
    # without touching report.py's CSS.
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
