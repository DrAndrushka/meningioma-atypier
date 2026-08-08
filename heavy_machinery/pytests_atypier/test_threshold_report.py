"""The threshold HTML report: loading, formatting, assembly, degradation.

The report never computes statistics — it reads what the notebook wrote. So the
tests that matter are about *robustness*: a missing table, a stale artifact set,
a boolean that survived a CSV round-trip as the string "True". Any of those
silently producing a wrong sentence in a document taken to a conference is the
failure mode worth guarding.

The second theme is that no sentence is hard-coded. Every verdict in the
document is templated from a CSV, so a rerun that changes the answer changes the
prose with it — the tests below rewrite the artifacts and check that it does.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

import study
import threshold_report as tr

SVG = (b'<?xml version="1.0"?>'
       b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
       b'<rect width="10" height="10"/></svg>')


def _defaults() -> dict[str, pd.DataFrame]:
    """A minimal but structurally faithful set of threshold-phase tables."""
    return {
        "00_cohort_summary.csv": pd.DataFrame([{
            "n_patients": 352, "n_high_grade": 105, "n_benign": 247,
            "n_outcome_missing": 0, "prevalence": 0.2983,
            "accrual_first_year": 2018, "accrual_last_year": 2026,
            "accrual_n_years": 9, "n_year_known": 352,
        }]),
        "01_metric_cohorts.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "direction": "lower", "n_analysed": 309,
             "n_missing": 43, "n_high_grade": 96, "n_benign": 213, "prevalence": 0.31,
             "median_benign": 0.83, "median_high_grade": 0.79},
            {"metric": "Volume", "column": "vol", "direction": "higher", "n_analysed": 329,
             "n_missing": 23, "n_high_grade": 101, "n_benign": 228, "prevalence": 0.31,
             "median_benign": 10.3, "median_high_grade": 26.2},
        ]),
        "03_risk_curves.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "unit": "$\\times 10^{-3}$ mm$^2$/s",
             "n": 309, "events": 96, "AUC": 0.63, "nonlinearity_p": 0.009,
             "nonlinear": True, "knee_found": True, "steepest_x": 0.662,
             "steepest_lo": 0.63, "steepest_hi": 0.70,
             "risk_30_x": 0.79, "risk_50_x": 0.66},
            {"metric": "Volume", "column": "vol", "unit": "cc",
             "n": 329, "events": 101, "AUC": 0.68, "nonlinearity_p": 0.97,
             "nonlinear": False, "knee_found": False, "steepest_x": 1.2,
             "steepest_lo": np.nan, "steepest_hi": np.nan,
             "risk_30_x": 16.2, "risk_50_x": 71.1},
        ]),
        "04_risk_curves_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Steepest rise": "0.662 (0.63–0.70)",
             "Reading": "Risk climbs most steeply near ADC 0.662."},
        ]),
        "07_threshold_summary.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "rule": "youden", "operator": "≤",
             "cutoff": 0.72, "cutoff_boot_lo": 0.69, "cutoff_boot_hi": 0.85,
             "n_used": 309, "sensitivity": 0.35, "specificity": 0.88,
             "OR": 4.12, "OR_lo": 2.28, "OR_hi": 7.45,
             "youden_J": 0.23, "youden_J_corrected": 0.21, "source": ""},
            {"metric": "Volume", "column": "vol", "rule": "literature", "operator": "≥",
             "cutoff": 13.95, "cutoff_boot_lo": np.nan, "cutoff_boot_hi": np.nan,
             "n_used": 329, "sensitivity": 0.69, "specificity": 0.58,
             "OR": 3.02, "OR_lo": 1.69, "OR_hi": 5.39,
             "youden_J": 0.27, "youden_J_corrected": np.nan,
             "source": "Shin et al., PLoS One 2021"},
        ]),
        "08_threshold_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Rule": "youden", "Cut-point": "≤0.72",
             "Cut-point 95% CI": "0.69–0.85", "n": 309, "J": 0.23, "Reading": ""},
            {"Metric": "Volume", "Rule": "literature", "Cut-point": "≥13.9",
             "Cut-point 95% CI": "", "n": 329, "J": 0.24, "Reading": ""},
        ]),
        "15_count_score.csv": pd.DataFrame([
            {"n_criteria_met": 0, "n": 73, "n_high_grade": 8, "risk": 0.11,
             "risk_lo": 0.06, "risk_hi": 0.20},
            {"n_criteria_met": 4, "n": 29, "n_high_grade": 19, "risk": 0.66,
             "risk_lo": 0.47, "risk_hi": 0.80},
        ]),
        "17_combination_verdict.csv": pd.DataFrame([{
            "best_single_rule": "Volume ≥ 15.1", "best_single_J": 0.274,
            "best_single_J_corrected": 0.232,
            "best_rule": "ADC ≤ 0.72 OR Volume ≥ 15.1", "best_rule_J": 0.326,
            "best_rule_J_corrected": 0.282, "winner_stability": 0.40,
            "gain_vs_best_single": 0.050, "continuous_AUC_corrected": 0.69,
            "continuous_J_equivalent": 0.383, "n_used_continuous": 304,
        }]),
        "18_imputation_stability.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "rule": "youden", "m_draws": 20,
             "cutoff_mean": 0.722, "cutoff_median": 0.72},
        ]),
        "19_imputation_stability_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Complete-case": "≤0.72", "MICE mean (m=20)": "≤0.722"},
        ]),
        "21_risk_curve_stability.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "m_draws": 20, "knee_rate": 0.85,
             "steepest_median": 0.652, "steepest_min": 0.63, "steepest_max": 0.67},
            {"metric": "Volume", "column": "vol", "m_draws": 20, "knee_rate": 0.30,
             "steepest_median": np.nan, "steepest_min": np.nan, "steepest_max": np.nan},
        ]),
        "26_headline_findings.csv": pd.DataFrame([
            {"Metric": "ADC", "Unit": "×10⁻³ mm²/s", "n": 309, "AUC": 0.63,
             "Threshold evidence": "moderate", "Youden cut-point": "≤0.72",
             "Sens / Spec": "35% / 88%", "J (corr.)": 0.21},
            {"Metric": "Volume", "Unit": "cm³", "n": 329, "AUC": 0.68,
             "Threshold evidence": "weak", "Youden cut-point": "≥15.1",
             "Sens / Spec": "67% / 60%", "J (corr.)": 0.23},
        ]),
        "28_threshold_evidence.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "verdict": "moderate",
             "n_criteria_passed": 4, "n_criteria": 5,
             "limiting_criterion": "MICE reproducibility", "AUC": 0.63},
            # Empty limiting_criterion round-trips as NaN — the cell that once
            # printed "limited by nan" on the front page.
            {"metric": "Volume", "column": "vol", "verdict": "weak",
             "n_criteria_passed": 1, "n_criteria": 5,
             "limiting_criterion": "", "AUC": 0.68},
        ]),
        "29_threshold_evidence_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Evidence": "moderate", "Criteria met": "4 of 5",
             "What limits it": "MICE reproducibility",
             "Read alongside (does not score)": "AUC 0.63"},
        ]),
        "31_shared_combination_reading_view.csv": pd.DataFrame([
            {"Rule": "ADC ≤ 0.72", "Type": "single", "n": 304, "J": 0.24},
        ]),
        "32_shared_combination_verdict.csv": pd.DataFrame([{
            "cohort": "shared denominator (all four measured)", "n_used": 304,
            "best_single_rule": "Edema volume ≥ 4.76", "best_single_J": 0.258,
            "best_single_J_corrected": 0.212,
            "best_rule": "ADC ≤ 0.72 OR Volume ≥ 15.1", "best_rule_J": 0.323,
            "best_rule_J_corrected": 0.281, "winner_stability": 0.396,
            "gain_vs_best_single": 0.069, "continuous_AUC_corrected": 0.69,
            "continuous_J_equivalent": 0.383, "n_used_continuous": 304,
        }]),
        "34_zero_inflation.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "n_analysed": 309, "n_zero": 0,
             "pct_zero": 0.0, "zero_inflated": False},
            {"metric": "Volume", "column": "vol", "n_analysed": 333, "n_zero": 122,
             "pct_zero": 36.6, "zero_inflated": True},
        ]),
        "36_risk_curves_nonzero_only.csv": pd.DataFrame([
            {"metric": "Volume", "column": "vol", "n": 211, "events": 76,
             "nonlinearity_p": 0.180, "knee_found": False},
        ]),
        "37_zero_inflation_comparison.csv": pd.DataFrame([
            {"Metric": "Volume", "All patients": "knee at 15.1",
             "Patients with some": "no knee", "n": 211},
        ]),
        "39_nonlinearity_multiplicity_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Non-linearity p": "= 0.009", "Holm-adjusted p": "= 0.018",
             "Holm": "survives"},
            {"Metric": "Volume", "Non-linearity p": "= 0.047",
             "Holm-adjusted p": "= 0.094", "Holm": "does not survive"},
        ]),
        "40_calibration.csv": pd.DataFrame([
            {"model": "Uncut four-measurement model", "n_used": 304, "events": 93,
             "slope_apparent": 1.0, "slope_corrected": 0.911,
             "intercept_apparent": 0.0, "intercept_corrected": -0.0002,
             "brier_apparent": 0.189, "brier_corrected": 0.195},
        ]),
        "46_study_facts.csv": pd.DataFrame([
            {"key": "who_edition", "item": "Outcome and WHO edition",
             "answer": "WHO 2021 CNS classification: grade 2/3 = high grade.",
             "status": "answered from the data",
             "why": "A high-grade rate needs the edition that produced it."},
            {"key": "dwi_acquisition", "item": "DWI acquisition",
             "answer": "Field strength, scanner and the b-values.",
             "status": "needs you",
             "why": "ADC values are not comparable between b-value schemes."},
        ]),
        "47_literature_sources.csv": pd.DataFrame([
            {"column": "vol", "cutoff": 13.95,
             "source": "Shin et al., PLoS One 2021",
             "link": "https://doi.org/10.1371/journal.pone.0252945"},
            {"column": "adc", "cutoff": 0.85, "source": "Unlinked et al. 2019",
             "link": ""},
        ]),
        "44_net_benefit_summary.csv": pd.DataFrame([
            {"strategy": "Uncut four-measurement model", "is_reference": False,
             "max_net_benefit": 0.27, "beats_references_from": 0.05,
             "beats_references_to": 0.60, "pct_of_range_beating_references": 76.8,
             "prevalence": 0.298},
            {"strategy": "Best single cut-point (Volume ≥ 15.1)", "is_reference": False,
             "max_net_benefit": 0.20, "beats_references_from": np.nan,
             "beats_references_to": np.nan, "pct_of_range_beating_references": 0.0,
             "prevalence": 0.298},
            {"strategy": "Treat all", "is_reference": True, "max_net_benefit": 0.27,
             "beats_references_from": np.nan, "beats_references_to": np.nan,
             "pct_of_range_beating_references": 0.0, "prevalence": 0.298},
        ]),
    }


def _write_artifacts(root, *, tables=None, figures=True, manifest=True):
    thresholds = root / "thresholds"
    (thresholds / "tables").mkdir(parents=True, exist_ok=True)
    (thresholds / "figures").mkdir(parents=True, exist_ok=True)

    for name, frame in (_defaults() if tables is None else tables).items():
        frame.to_csv(thresholds / "tables" / name, index=False)

    if figures:
        for name in list(tr.FIGURE_FILES.values()) + [
            "02_distribution_adc.svg", "06_risk_curve_adc.svg", "09_thresholds_adc.svg",
        ]:
            (thresholds / "figures" / name).write_bytes(SVG)

    if manifest:
        (thresholds / "manifest.json").write_text(json.dumps({
            "phase": "thresholds",
            "generated_at": "2026-08-02T20:00:00+00:00",
            "context": {"n_bootstrap": 2000, "seed": 20260801},
            "figures": [], "tables": [], "notes": [],
        }))
    return root


def _with(**replacements) -> dict[str, pd.DataFrame]:
    """The default artifact set with tables replaced (or dropped, with ``None``).

    Keys are ``TABLE_FILES`` keys: ``evidence=frame`` replaces
    ``28_threshold_evidence.csv``, ``evidence=None`` removes it.
    """
    tables = _defaults()
    for key, frame in replacements.items():
        name = tr.TABLE_FILES[key]
        if frame is None:
            tables.pop(name, None)
        else:
            tables[name] = frame
    return tables


@pytest.fixture
def artifacts(tmp_path):
    return _write_artifacts(tmp_path)


@pytest.fixture
def cfg(artifacts):
    return tr.ThresholdReportConfig(output_root=artifacts)


def _html(root) -> str:
    return tr.build_report(tr.ThresholdReportConfig(output_root=root))


def _prose(html: str) -> str:
    """The document with figures, styles and tables stripped — what a reader reads."""
    body = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<table.*?</table>", "", body, flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


def _section(html: str, heading: str) -> str:
    """Everything from a section's heading to the end of that section."""
    start = html.index(heading)
    end = html.index("</section>", start)
    return html[start:end]


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def test_truthy_handles_csv_round_tripped_booleans():
    """`knee_found` comes back from CSV as the string "True", which is the whole
    reason this helper exists — plain bool("False") is True."""
    assert tr._truthy("True") and tr._truthy(True) and tr._truthy("yes")
    assert not tr._truthy("False")
    assert not tr._truthy(np.nan)
    assert not tr._truthy("")


@pytest.mark.parametrize("value,expected", [
    (0.2983, "30%"), (1.0, "100%"), (np.nan, "—"), (None, "—"), ("x", "—"),
])
def test_pct(value, expected):
    assert tr._pct(value) == expected


def test_sig_drops_trailing_zeros_across_magnitudes():
    """Cut-points here run from 0.0617 to 61.4 — one fixed decimal count cannot
    serve both without printing false precision at one end."""
    assert tr._sig(0.06171) == "0.0617"
    assert tr._sig(61.4321) == "61.4"


def test_signed_collapses_negative_zero():
    """A calibration intercept of -0.0002 must not print as "-0.000"."""
    assert tr._signed(-0.0002) == "0.000"
    assert tr._signed(0.003) == "+0.003"
    assert tr._signed(np.nan) == "—"


def test_int_rounds_and_degrades():
    assert tr._int(2018.0) == "2018"
    assert tr._int(np.nan) == "—"


def test_cell_never_returns_the_string_nan():
    """An empty CSV cell becomes NaN, and str(nan) is "nan" — which is how
    "limited by nan" once reached a poster."""
    assert tr._cell(np.nan) == ""
    assert tr._cell(None) == ""
    assert tr._cell("scale robustness") == "scale robustness"


# --------------------------------------------------------------------------
# Loading and degradation
# --------------------------------------------------------------------------
def test_load_reads_every_expected_table_and_figure(cfg):
    data = tr.load_report_data(cfg)
    assert set(data.tables) == set(tr.TABLE_FILES)
    assert set(data.figures) == set(tr.FIGURE_FILES)
    assert data.figure_groups["distributions"] and data.figure_groups["risk_curves"]
    assert not data.warnings


def test_missing_output_folder_degrades_to_a_warning(tmp_path):
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=tmp_path))
    assert data.warnings and not data.tables


def test_missing_table_is_warned_not_raised(tmp_path):
    _write_artifacts(tmp_path, tables=_with(evidence=None))
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=tmp_path))
    assert any("28_threshold_evidence" in w for w in data.warnings)
    assert "cohort_summary" in data.tables


def test_report_renders_without_any_artifacts(tmp_path):
    """A report that raises on an empty folder is a report nobody can debug."""
    html = _html(tmp_path)
    assert "<html" in html and "re-run the notebook" in html


def test_report_degrades_without_calibration_or_net_benefit(tmp_path):
    _write_artifacts(tmp_path, tables=_with(calibration=None, net_benefit_summary=None))
    html = _html(tmp_path)
    assert "Are the predicted percentages usable?" in html
    assert "re-run the notebook" in html


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def test_report_is_a_self_contained_document(cfg):
    html = tr.build_report(cfg)
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "data:image/svg+xml;base64," in html   # figures embedded, not linked
    assert 'src="http' not in html


def test_report_contains_every_section(cfg):
    html = tr.build_report(cfg)
    for heading in ("Do the two grades even look different?",
                    "Where does risk start climbing?",
                    "Where do you draw the line?",
                    "Does ticking two boxes beat one?",
                    "Would it survive the missing scans?",
                    "How much do we believe each threshold?",
                    "Are the predicted percentages usable?",
                    "Copy-paste for the abstract",
                    "Methods — what is known, and what is still open"):
        assert heading in html, heading


def test_the_front_page_carries_one_row_per_measurement(cfg):
    """The summary a reader screenshots for a slide. It joins the headline table
    (AUC, grade) with the Youden rows of the cut-point table (odds ratio), so it
    has to appear before section 1 and carry both."""
    html = tr.build_report(cfg)
    front = html[:html.index("1 · Do the two grades")]
    assert "<table" in front
    for fragment in ("Measurement", "OR (95% CI)", "Evidence",
                     "≤0.72 ×10⁻³ mm²/s", "35% / 88%", "moderate"):
        assert fragment in front, fragment
    # The OR comes from the cut-point table, not from the headline CSV.
    assert "4.12" in front


def test_front_page_summary_degrades_without_the_headline_table(tmp_path):
    _write_artifacts(tmp_path, tables=_with(headline=None))
    html = _html(tmp_path)
    assert "<html" in html and "1 · Do the two grades" in html


def test_the_report_stays_short(cfg):
    """The point of the rewrite. Prose only — figures and tables do not count."""
    words = len(_prose(tr.build_report(cfg)).split())
    assert words < 2600, f"{words} words of prose — the long report is back"


def test_every_question_gets_exactly_one_answer_line(cfg):
    """Seven questions, seven answers: the structure is the document."""
    assert tr.build_report(cfg).count('<div class="answer">') == 7


def test_report_quotes_the_real_numbers(cfg):
    html = tr.build_report(cfg)
    assert "352" in html and "105" in html and "29.8%" in html
    assert "2018–2026" in html


def test_no_escaping_artifacts_reach_the_page(cfg):
    html = tr.build_report(cfg)
    assert "&amp;lt;" not in html and "&amp;gt;" not in html


def test_no_placeholder_values_leak_into_prose(cfg):
    prose = _prose(tr.build_report(cfg))
    leak = re.search(r"\b(nan|NaN|None|inf|Infinity)\b", prose)
    assert leak is None, prose[max(0, leak.start() - 60):leak.end() + 60]


def test_no_latvian_in_the_english_text(cfg):
    """The source CSVs carry Latvian column values; none may reach the document."""
    assert not re.search(r"[ēūīļķģšžčņā]", tr.build_report(cfg).lower())


def test_numeric_cells_cannot_wrap_mid_token(cfg):
    """"1.53–12.2" offers a break after the dash, and a narrow column took it."""
    html = tr.build_report(cfg)
    assert 'td class="nowrap"' in html
    assert "word-break: keep-all" in html


# --------------------------------------------------------------------------
# Nothing is hard-coded: change the artifacts, the prose changes
# --------------------------------------------------------------------------
def test_verdict_follows_the_artifacts_not_the_prose(tmp_path):
    evidence = pd.DataFrame([
        {"metric": "ADC", "column": "adc", "verdict": "strong",
         "n_criteria_passed": 5, "n_criteria": 5, "limiting_criterion": "", "AUC": 0.63},
        {"metric": "Volume", "column": "vol", "verdict": "strong",
         "n_criteria_passed": 5, "n_criteria": 5, "limiting_criterion": "", "AUC": 0.68},
    ])
    _write_artifacts(tmp_path, tables=_with(evidence=evidence))
    html = _html(tmp_path)
    assert "2 strong" in html
    assert "fragile" not in _prose(html)


def test_blank_limiting_criterion_never_prints_nan(cfg):
    """The default set has one metric with an empty limiting criterion."""
    prose = _prose(tr.build_report(cfg))
    assert "limited by nan" not in prose
    assert "all criteria met" in prose


def test_grade_tally_orders_best_supported_first():
    v = tr.Verdicts((
        tr.MetricVerdict("A", "a", True, grade="weak"),
        tr.MetricVerdict("B", "b", True, grade="strong"),
        tr.MetricVerdict("C", "c", True, grade="moderate"),
    ))
    assert [g for g, _ in v.grade_tally()] == ["strong", "moderate", "weak"]
    assert v.grade_phrase().startswith("1 strong")


def test_count_phrase_agrees_in_number():
    one = tr.Verdicts((tr.MetricVerdict("A", "a", True),
                       tr.MetricVerdict("B", "b", False)))
    assert one.count_phrase(verb=True) == "1 of 2 measurements has"
    two = tr.Verdicts((tr.MetricVerdict("A", "a", True),
                       tr.MetricVerdict("B", "b", True)))
    assert two.count_phrase(verb=True) == "2 of 2 measurements have"


def test_falls_back_to_the_bare_count_when_grades_are_missing(tmp_path):
    _write_artifacts(tmp_path, tables=_with(evidence=None, evidence_reading=None))
    html = _html(tmp_path)
    assert "1 of 2" in html          # the count of knees, since no grade exists
    assert "re-run the notebook" in html


def test_threshold_locations_carry_their_units(cfg):
    """A knee quoted as "0.662" with no unit is not a publishable number, and the
    unit in the risk-curve CSV is LaTeX written for the figures."""
    html = tr.build_report(cfg)
    assert "0.662 ×10⁻³ mm²/s" in html
    assert "$\\times" not in html


def test_unreproducible_thresholds_are_named_in_the_answer(cfg):
    """Volume's knee is found in 30% of draws — below the pre-specified bar, so
    the answer has to name it rather than average it away."""
    section = _section(tr.build_report(cfg), "Would it survive the missing scans?")
    assert "30%" in section and "Volume" in section


# --------------------------------------------------------------------------
# One denominator
# --------------------------------------------------------------------------
def test_shared_denominator_is_the_primary_comparison(cfg):
    """Rules scored on different patient sets are not comparable, so the
    shared-cohort verdict wins wherever both exist."""
    verdict = tr.primary_verdict(tr.load_report_data(cfg))
    assert verdict["n_used"].iloc[0] == 304
    # "all N measurements" counts METRICS, so adding one cannot leave a stale
    # "all four" in the prose.
    assert "304 patients with all 2 measurements" in tr.build_report(cfg)


def test_falls_back_to_the_per_metric_verdict_when_shared_is_missing(tmp_path):
    _write_artifacts(tmp_path, tables=_with(shared_verdict=None))
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=tmp_path))
    assert tr.primary_verdict(data)["best_rule"].iloc[0] == "ADC ≤ 0.72 OR Volume ≥ 15.1"
    assert "<html" in _html(tmp_path)


def test_cutpoint_answer_states_its_own_denominator(cfg):
    """Section 3 scores each cut-point on its own complete cases, so the number it
    quotes has to carry that n or it reads as a contradiction of section 4."""
    assert "on its own 309 patients" in tr.build_report(cfg)


# --------------------------------------------------------------------------
# The copy-paste block and the gaps
# --------------------------------------------------------------------------
def test_manuscript_block_carries_the_run_numbers(cfg):
    block = re.search(r'<div class="manuscript">.*?\n</div>',
                      tr.build_report(cfg), re.S)
    assert block
    for fragment in ("352 operated patients", "105 (29.8%)", "2000 resamples",
                     "20 multiply-imputed datasets", "304 patients"):
        assert fragment in block.group(0), fragment


def test_manuscript_limitations_prevalence_renders_one_decimal(tmp_path):
    """The Limitations sentence's cohort prevalence used to render at 0 dp
    (``30% high grade``) via the local ``_pct`` helper's default, even after the
    Methods paragraph two sentences above it was fixed to 1 dp. Both must agree."""
    _write_artifacts(tmp_path)
    html = _html(tmp_path)
    block = re.search(r'<div class="manuscript">.*?\n</div>', html, re.S)
    assert block
    assert "29.8% high grade" in block.group(0)
    assert "30% high grade" not in block.group(0)


def test_manuscript_block_stays_short(cfg):
    """It has to fit an abstract: long enough to be useful, short enough to paste."""
    block = re.search(r'<div class="manuscript">.*?\n</div>',
                      tr.build_report(cfg), re.S)
    words = len(re.sub(r"<[^>]+>", " ", block.group(0)).split())
    assert 200 < words < 500, words


def test_literature_agreement_is_computed_not_remembered(cfg):
    """A published cut-point landing inside our own interval is the strongest line
    on the poster — and it moves with every run, so it must be recomputed."""
    data = tr.load_report_data(cfg)
    # Volume has no knee in the fixture, so nothing qualifies yet.
    assert tr._literature_agreement(data) == []
    risk = data.table("risk_curves")
    risk.loc[risk["column"] == "vol",
             ["knee_found", "steepest_lo", "steepest_hi"]] = [True, 10.0, 17.0]
    assert any("Shin et al. 2021" in line for line in tr._literature_agreement(data))


def test_answered_methods_facts_are_shown_as_answers(cfg):
    """What the cleaning run already recorded is printed as fact, not asked for."""
    methods = _section(tr.build_report(cfg), "Methods — what is known")
    assert "WHO 2021 CNS classification" in methods
    assert "Still open" in methods


def test_open_questions_carry_their_reason_and_stay_greppable(cfg):
    methods = _section(tr.build_report(cfg), "Methods — what is known")
    assert "TODO: ANDY" in methods
    assert "1 open question." in methods          # singular, one open row in the fixture
    assert "Needed because:" in methods
    assert "not comparable between b-value schemes" in methods


def test_no_clinical_fact_is_invented_in_the_open_questions(cfg):
    """The pipeline cannot know the b-values or the field strength. A number next
    to one of these prompts would mean it was guessed."""
    methods = _section(tr.build_report(cfg), "Methods — what is known")
    assert not re.search(r"\b(1\.5|3)\s*T\b", methods)
    assert not re.search(r"b\s*=\s*\d", methods)


def test_methods_falls_back_to_the_questions_without_the_facts_table(tmp_path):
    """An older output/thresholds/ has no 46_study_facts.csv."""
    _write_artifacts(tmp_path, tables=_with(study_facts=None))
    methods = _section(_html(tmp_path), "Methods — what is known")
    assert "TODO: ANDY" in methods
    for question in study.QUESTIONS:
        assert question.item in methods, question.key


def test_published_cutpoints_are_linked_where_a_link_exists(cfg):
    """A cut-point taken from a paper is only checkable if the paper is."""
    html = tr.build_report(cfg)
    assert '<a href="https://doi.org/10.1371/journal.pone.0252945">' in html
    assert "Unlinked et al. 2019" in html          # no link, still named


def test_cli_writes_the_file(tmp_path):
    _write_artifacts(tmp_path)
    out = tmp_path / "report.html"
    assert tr.main(["--output-root", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text().startswith("<!DOCTYPE html>")
