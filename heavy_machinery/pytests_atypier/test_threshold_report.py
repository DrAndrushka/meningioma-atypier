"""The threshold HTML report: loading, formatting, assembly, degradation.

The report never computes statistics — it reads what the notebook wrote. So the
tests that matter are about *robustness*: a missing table, a stale artifact set,
a boolean that survived a CSV round-trip as the string "True". Any of those
silently producing a wrong sentence in a document taken to a conference is the
failure mode worth guarding.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import threshold_report as tr

SVG = (b'<?xml version="1.0"?>'
       b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
       b'<rect width="10" height="10"/></svg>')


def _write_artifacts(root, *, tables=None, figures=True, manifest=True, models=False):
    """A minimal but structurally faithful output/thresholds/ tree."""
    thresholds = root / "thresholds"
    (thresholds / "tables").mkdir(parents=True, exist_ok=True)
    (thresholds / "figures").mkdir(parents=True, exist_ok=True)

    defaults = {
        "00_cohort_summary.csv": pd.DataFrame([{
            "n_patients": 352, "n_high_grade": 105, "n_benign": 247,
            "n_outcome_missing": 0, "prevalence": 0.2983,
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
            {"metric": "ADC", "column": "adc", "AUC": 0.63, "nonlinearity_p": 0.009,
             "nonlinear": True, "knee_found": True, "steepest_x": 0.662,
             "steepest_lo": 0.63, "steepest_hi": 0.70,
             "risk_30_x": 0.79, "risk_50_x": 0.66,
             "verdict": "Risk climbs most steeply near ADC 0.662."},
            {"metric": "Volume", "column": "vol", "AUC": 0.68, "nonlinearity_p": 0.97,
             "nonlinear": False, "knee_found": False, "steepest_x": 1.2,
             "steepest_lo": np.nan, "steepest_hi": np.nan,
             "risk_30_x": 16.2, "risk_50_x": 71.1,
             "verdict": "Risk rises steadily with Volume."},
        ]),
        "04_risk_curves_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Steepest rise": "0.662 (0.63–0.70)", "Reading": "threshold"},
        ]),
        "07_threshold_summary.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "rule": "youden", "operator": "≤",
             "cutoff": 0.72, "cutoff_boot_lo": 0.69, "cutoff_boot_hi": 0.85,
             "n_used": 309, "TP": 34, "FP": 26, "FN": 62, "TN": 187,
             "sensitivity": 0.35, "specificity": 0.88, "PPV": 0.57, "NPV": 0.75,
             "youden_J": 0.23, "youden_J_corrected": 0.21, "source": ""},
            {"metric": "Volume", "column": "vol", "rule": "literature", "operator": "≥",
             "cutoff": 13.95, "cutoff_boot_lo": np.nan, "cutoff_boot_hi": np.nan,
             "n_used": 329, "TP": 70, "FP": 95, "FN": 31, "TN": 133,
             "sensitivity": 0.69, "specificity": 0.58, "PPV": 0.42, "NPV": 0.81,
             "youden_J": 0.27, "youden_J_corrected": np.nan,
             "source": "Shin et al., PLoS One 2021"},
        ]),
        "08_threshold_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Rule": "youden", "Cut-point": "≤0.72", "J": 0.23},
        ]),
        "11_flag_missingness.csv": pd.DataFrame([{
            "n_patients": 352, "n_all_flags_observed": 304, "n_some_flag_missing": 48,
            "pct_scorable": 86.4, "k_criteria": 4,
        }]),
        "12_combination_menu.csv": pd.DataFrame([
            {"rule_label": "ADC ≤ 0.72", "kind": "single", "youden_J": 0.23},
        ]),
        "13_combination_reading_view.csv": pd.DataFrame([
            {"Rule": "ADC ≤ 0.72", "Type": "single", "J": 0.23},
        ]),
        "15_count_score.csv": pd.DataFrame([
            {"n_criteria_met": 0, "n": 73, "n_high_grade": 8, "risk": 0.11,
             "risk_lo": 0.06, "risk_hi": 0.20},
            {"n_criteria_met": 4, "n": 29, "n_high_grade": 19, "risk": 0.66,
             "risk_lo": 0.47, "risk_hi": 0.80},
        ]),
        "17_combination_verdict.csv": pd.DataFrame([{
            "best_single_rule": "Volume ≥ 15.1", "best_single_J": 0.274,
            "best_rule": "ADC ≤ 0.72 OR Volume ≥ 15.1", "best_rule_J": 0.326,
            "best_rule_J_corrected": 0.282, "selection_optimism": 0.044,
            "winner_stability": 0.40, "gain_vs_best_single": 0.008,
            "continuous_AUC_corrected": 0.69, "continuous_J_equivalent": 0.383,
            "n_used_continuous": 304,
        }]),
        "18_imputation_stability.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "rule": "youden", "operator": "≤",
             "m_draws": 20, "cutoff_mean": 0.722, "cutoff_median": 0.72},
        ]),
        "19_imputation_stability_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Complete-case": "≤0.72", "MICE mean (m=20)": "≤0.722"},
        ]),
        "21_risk_curve_stability.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "m_draws": 20, "knee_rate": 0.85,
             "steepest_median": 0.652, "steepest_min": 0.63, "steepest_max": 0.67},
            {"metric": "Volume", "column": "vol", "m_draws": 20, "knee_rate": 0.0,
             "steepest_median": np.nan, "steepest_min": np.nan, "steepest_max": np.nan},
        ]),
        "23_count_score_imputed.csv": pd.DataFrame([
            {"n_criteria_met": 0, "n": 89, "risk": 0.109, "risk_min": 0.09,
             "risk_max": 0.12},
        ]),
        "25_count_rules_imputed.csv": pd.DataFrame([
            {"rule_label": "≥ 2 of 4 criteria", "sensitivity": 0.80, "specificity": 0.52},
        ]),
        "26_headline_findings.csv": pd.DataFrame([
            {"Metric": "ADC", "Threshold evidence": "moderate",
             "What limits it": "MICE reproducibility", "Non-linearity p": "= 0.009"},
            {"Metric": "Volume", "Threshold evidence": "weak",
             "What limits it": "Non-linearity", "Non-linearity p": "= 0.974"},
        ]),
        "27_evidence_criteria.csv": pd.DataFrame([
            {"Criterion": "Non-linearity", "Family": "necessary",
             "Source": "LRT (section 3)", "Rule": "p &lt; 0.05"},
        ]),
        "28_threshold_evidence.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "verdict": "moderate",
             "n_criteria_passed": 4, "n_criteria": 5,
             "limiting_criterion": "MICE reproducibility",
             "failed_criteria": "MICE reproducibility",
             "knee_ci_ratio": 1.11, "AUC": 0.63,
             "verdict_note": "moderate — MICE reproducibility fails"},
            # Empty limiting_criterion round-trips as NaN — the cell that once
            # printed "limited by nan" on the front page.
            {"metric": "Volume", "column": "vol", "verdict": "weak",
             "n_criteria_passed": 1, "n_criteria": 5,
             "limiting_criterion": "", "failed_criteria": "Non-linearity",
             "knee_ci_ratio": np.nan, "AUC": 0.68,
             "verdict_note": "weak — Non-linearity fails"},
        ]),
        "29_threshold_evidence_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Evidence": "moderate", "Criteria met": "4 of 5",
             "What limits it": "MICE reproducibility"},
        ]),
        "40_calibration.csv": pd.DataFrame([
            {"model": "Uncut four-measurement model", "n_used": 304, "events": 93,
             "n_predictors": 4, "slope_apparent": 1.0, "slope_corrected": 0.911,
             "intercept_apparent": 0.0, "intercept_corrected": 0.003,
             "brier_apparent": 0.189, "brier_corrected": 0.195, "n_bootstrap": 500,
             "source": "threshold phase (fitted here)"},
            # No corrected intercept — the modelling artifacts do not carry one.
            {"model": "experimental 2", "n_used": 352, "events": 105,
             "n_predictors": 10, "slope_apparent": 1.0, "slope_corrected": 0.773,
             "intercept_apparent": 0.0, "intercept_corrected": np.nan,
             "brier_apparent": 0.187, "brier_corrected": 0.199, "n_bootstrap": 1000,
             "source": "modelling phase artifact"},
        ]),
        "41_calibration_bins_uncut.csv": pd.DataFrame([
            {"predicted": 0.13, "observed": 0.10, "lo": 0.03, "hi": 0.28,
             "n": 30, "events": 3},
        ]),
        "43_net_benefit.csv": pd.DataFrame([
            {"strategy": "Treat all", "threshold": 0.05, "net_benefit": 0.27,
             "kind": "reference"},
            {"strategy": "Uncut four-measurement model", "threshold": 0.05,
             "net_benefit": 0.27, "kind": "model"},
            {"strategy": "Treat all", "threshold": 0.60, "net_benefit": -0.75,
             "kind": "reference"},
            {"strategy": "Uncut four-measurement model", "threshold": 0.60,
             "net_benefit": 0.01, "kind": "model"},
        ]),
        "44_net_benefit_summary.csv": pd.DataFrame([
            {"strategy": "Uncut four-measurement model", "is_reference": False,
             "max_net_benefit": 0.27, "threshold_at_max": 0.05,
             "beats_references_from": 0.05, "beats_references_to": 0.60,
             "pct_of_range_beating_references": 76.8,
             "pct_of_range_best_available": 58.9, "prevalence": 0.298},
            {"strategy": "Best single cut-point (Edema volume ≥ 4.76)",
             "is_reference": False, "max_net_benefit": 0.20, "threshold_at_max": 0.05,
             "beats_references_from": np.nan, "beats_references_to": np.nan,
             "pct_of_range_beating_references": 0.0,
             "pct_of_range_best_available": 0.0, "prevalence": 0.298},
            {"strategy": "Treat all", "is_reference": True, "max_net_benefit": 0.27,
             "threshold_at_max": 0.05, "beats_references_from": np.nan,
             "beats_references_to": np.nan, "pct_of_range_beating_references": 0.0,
             "pct_of_range_best_available": 12.5, "prevalence": 0.298},
        ]),
        "38_nonlinearity_multiplicity.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "family": "non-linearity LRT (primary)",
             "n_tests": 2, "p_raw": 0.009, "p_holm": 0.018, "p_bonferroni": 0.018,
             "survives_raw": True, "survives_holm": True, "survives_bonferroni": True},
            {"metric": "Volume", "column": "vol", "family": "non-linearity LRT (primary)",
             "n_tests": 2, "p_raw": 0.047, "p_holm": 0.047, "p_bonferroni": 0.094,
             "survives_raw": True, "survives_holm": True, "survives_bonferroni": False},
        ]),
        "39_nonlinearity_multiplicity_reading_view.csv": pd.DataFrame([
            {"Metric": "ADC", "Non-linearity p": "= 0.009", "Holm-adjusted p": "= 0.018",
             "Holm": "survives", "Bonferroni-adjusted p": "= 0.018",
             "Bonferroni": "survives"},
        ]),
        "34_zero_inflation.csv": pd.DataFrame([
            {"metric": "ADC", "column": "adc", "n_analysed": 309, "n_zero": 0,
             "pct_zero": 0.0, "n_positive": 309, "risk_zero": np.nan,
             "risk_zero_lo": np.nan, "risk_zero_hi": np.nan, "risk_positive": 0.31,
             "risk_positive_lo": 0.26, "risk_positive_hi": 0.36,
             "risk_ratio": np.nan, "zero_inflated": False},
            {"metric": "Volume", "column": "vol", "n_analysed": 333, "n_zero": 122,
             "pct_zero": 36.6, "n_positive": 211, "risk_zero": 0.189,
             "risk_zero_lo": 0.129, "risk_zero_hi": 0.267, "risk_positive": 0.36,
             "risk_positive_lo": 0.298, "risk_positive_hi": 0.427,
             "risk_ratio": 1.911, "zero_inflated": True},
        ]),
        "35_presence_rules.csv": pd.DataFrame([
            {"metric": "Volume", "column": "vol", "rule": "presence",
             "rule_label": "Volume present (> 0)", "n_used": 333,
             "TP": 76, "FP": 135, "FN": 23, "TN": 99,
             "sensitivity": 0.768, "specificity": 0.423, "youden_J": 0.191},
        ]),
        "36_risk_curves_nonzero_only.csv": pd.DataFrame([
            {"metric": "Volume", "column": "vol", "n": 211, "events": 76,
             "nonlinearity_p": 0.180, "knee_found": False, "steepest_x": 6.0,
             "steepest_lo": np.nan, "steepest_hi": np.nan},
        ]),
        "37_zero_inflation_comparison.csv": pd.DataFrame([
            {"Metric": "Volume", "Fitted on": "Whole cohort (zeros included)",
             "n": "333 (99 high grade)", "Non-linearity p": "= 0.007",
             "Steepest rise": "3.51", "95% CI": "1.7–8"},
        ]),
        "30_shared_combination_menu.csv": pd.DataFrame([
            {"rule_label": "ADC ≤ 0.72", "kind": "single", "n_used": 304,
             "youden_J": 0.24},
        ]),
        "31_shared_combination_reading_view.csv": pd.DataFrame([
            {"Rule": "ADC ≤ 0.72", "Type": "single", "n": 304, "J": 0.24},
        ]),
        "32_shared_combination_verdict.csv": pd.DataFrame([{
            "cohort": "shared denominator (all four measured)", "n_used": 304,
            "best_single_rule": "Edema volume ≥ 4.76", "best_single_J": 0.258,
            "best_single_J_corrected": 0.212,
            "best_single_selection_optimism": 0.046, "best_single_stability": 0.32,
            "best_rule": "ADC ≤ 0.72 OR Volume ≥ 15.1", "best_rule_J": 0.323,
            "best_rule_J_corrected": 0.281, "selection_optimism": 0.042,
            "winner_stability": 0.396, "gain_apparent": 0.065,
            "gain_vs_best_single": 0.069, "continuous_AUC_corrected": 0.69,
            "continuous_J_equivalent": 0.383, "n_used_continuous": 304,
        }]),
    }
    for name, frame in (tables if tables is not None else defaults).items():
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
            "figures": [], "tables": [],
            "notes": ["ADC has the only reproducible threshold."],
        }))

    if models:
        folder = root / "inferential" / "model_artifacts"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "high_grade_experimental_model_1_model.json").write_text(json.dumps({
            "n": 352, "events": 105,
            "coefficients": {"const": -1.0, "a": 0.5, "b": 0.3},
            "validation": {"metrics": [
                {"metric": "AUC", "apparent": 0.756, "optimism_corrected": 0.728},
            ]},
        }))
    return root


@pytest.fixture
def artifacts(tmp_path):
    return _write_artifacts(tmp_path)


@pytest.fixture
def cfg(artifacts):
    return tr.ThresholdReportConfig(output_root=artifacts)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def test_truthy_handles_csv_round_tripped_booleans():
    """`knee_found` comes back from CSV as the string "True", which is the whole
    reason this helper exists — plain bool("False") is True."""
    assert tr._truthy("True") and tr._truthy(True) and tr._truthy("yes")
    assert not tr._truthy("False")
    assert not tr._truthy(False)
    assert not tr._truthy(np.nan)
    assert not tr._truthy(None)


@pytest.mark.parametrize("value,expected", [
    (0.2983, "30%"), (1.0, "100%"), (np.nan, "—"), (None, "—"), ("x", "—"),
])
def test_pct(value, expected):
    assert tr._pct(value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.72, "0.72"), (np.nan, "—"), (None, "—"), ("x", "—"),
])
def test_num(value, expected):
    assert tr._num(value) == expected


def test_sig_drops_trailing_zeros_across_magnitudes():
    assert tr._sig(4.760) == "4.76"
    assert tr._sig(0.0617) == "0.0617"
    assert tr._sig(44.83) == "44.8"
    assert tr._sig(np.nan) == "—"


def test_int_rounds_and_degrades():
    assert tr._int(88.85) == "89"
    assert tr._int(np.nan) == "—"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_load_reads_every_expected_table_and_figure(cfg):
    data = tr.load_report_data(cfg)
    assert set(TABLE_KEYS := set(tr.TABLE_FILES)) <= set(data.tables) | {"cohort_summary"}
    assert not data.table("headline").empty
    assert set(tr.FIGURE_FILES) <= set(data.figures)
    assert data.figure_groups["distributions"]
    assert data.manifest["context"]["n_bootstrap"] == 2000


def test_missing_output_folder_degrades_to_a_warning(tmp_path):
    cfg = tr.ThresholdReportConfig(output_root=tmp_path)
    data = tr.load_report_data(cfg)
    assert data.tables == {}
    assert any("does not exist" in w for w in data.warnings)


def test_missing_table_is_warned_not_raised(tmp_path):
    root = _write_artifacts(tmp_path)
    (root / "thresholds" / "tables" / "26_headline_findings.csv").unlink()
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=root))
    assert data.table("headline").empty
    assert any("26_headline_findings" in w for w in data.warnings)


def test_model_aucs_are_optional(cfg, tmp_path):
    assert tr.load_report_data(cfg).model_aucs.empty
    root = _write_artifacts(tmp_path / "with_models", models=True)
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=root))
    assert len(data.model_aucs) == 1
    row = data.model_aucs.iloc[0]
    assert row["AUC_corrected"] == 0.728
    assert row["n_predictors"] == 2  # the intercept is not a predictor


# --------------------------------------------------------------------------
# Cohort facts
# --------------------------------------------------------------------------
def test_facts_come_from_the_exported_summary(cfg):
    facts = tr.cohort_facts(tr.load_report_data(cfg))
    assert facts.n == 352
    assert facts.events == 105          # exact, not reverse-engineered
    assert facts.benign == 247
    assert facts.prevalence == pytest.approx(0.2983)
    assert facts.n_thresholds == 1
    assert facts.m_draws == "20"


def test_facts_fall_back_when_the_summary_predates_this_report(tmp_path):
    """Older runs have no 00_cohort_summary.csv; the report must still render."""
    root = _write_artifacts(tmp_path)
    (root / "thresholds" / "tables" / "00_cohort_summary.csv").unlink()
    facts = tr.cohort_facts(tr.load_report_data(tr.ThresholdReportConfig(output_root=root)))
    assert facts.n == 352
    assert facts.events > 0  # approximate, but present


def test_threshold_count_survives_string_booleans(cfg):
    """knee_found arrives as "True"/"False" strings from the CSV."""
    facts = tr.cohort_facts(tr.load_report_data(cfg))
    assert facts.n_thresholds == 1
    assert facts.n_metrics == 2


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def test_report_is_a_self_contained_document(cfg):
    html = tr.build_report(cfg)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # Figures inlined as data URIs — no external files to lose.
    assert "data:image/svg+xml;base64," in html
    assert "src=\"http" not in html and "src='http" not in html


def test_no_latvian_in_the_english_text(cfg):
    """Kept in the library's own comments, dropped from the document."""
    assert "šķēre" not in tr.build_report(cfg)


def test_report_contains_every_section(cfg):
    html = tr.build_report(cfg)
    for heading in ("Three questions", "Where does risk climb fastest",
                    "Where should the line be drawn",
                    "Do several criteria", "survive the missing data",
                    "Trade-offs", "bottom line", "Defending this at ESNR",
                    "Reference", "What not to claim"):
        assert heading in html, heading


def test_report_quotes_the_real_numbers(cfg):
    html = tr.build_report(cfg)
    assert "105 of 352" in html          # from the exported summary
    assert "1 of 2" in html              # thresholds found / metrics tested
    assert "0.662" in html               # the steepest-rise point
    assert "40%" in html                 # winner stability


def test_no_escaping_artifacts_reach_the_page(cfg):
    """`warning_box` escapes its message, so a raw &quot; in it double-escapes."""
    html = tr.build_report(cfg)
    assert "&amp;quot;" not in html
    assert "&amp;lt;" not in html
    assert "\\&quot;" not in html


def test_no_placeholder_values_leak_into_prose(cfg):
    html = tr.build_report(cfg)
    for leak in (">nan<", "nan%", "NaN", "dtype:", "Series(", "None%"):
        assert leak not in html, leak


def test_report_renders_without_any_artifacts(tmp_path):
    """A report generated before the notebook has run must still be a valid page."""
    cfg = tr.ThresholdReportConfig(output_root=tmp_path)
    html = tr.build_report(cfg)
    assert html.startswith("<!DOCTYPE html>")
    assert "does not exist" in html
    assert "figure unavailable" in html or "figures unavailable" in html


def test_report_notes_when_models_are_absent(cfg):
    html = tr.build_report(cfg)
    assert "Multivariable model artifacts not found" in html


def test_report_includes_models_when_present(tmp_path):
    root = _write_artifacts(tmp_path, models=True)
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "0.73" in html  # the corrected AUC, in the header card and §7
    assert "Full multivariable model" in html


def test_literature_cutpoints_get_their_own_block(cfg):
    html = tr.build_report(cfg)
    assert "Published cut-points, scored on our patients" in html
    assert "Shin et al." in html


# --------------------------------------------------------------------------
# The verdict, which must never be a literal
# --------------------------------------------------------------------------
def test_no_verdict_sentence_is_hard_coded(cfg):
    """The exact prose that once contradicted the tables it sat next to."""
    html = tr.build_report(cfg)
    for stale in ("Only ADC", "only one passed", "the volumes do not turn",
                  "no threshold whatsoever", "Threshold effect?"):
        assert stale not in html, f"hard-coded verdict text survived: {stale!r}"


def test_verdict_follows_the_artifacts_not_the_prose(tmp_path):
    """Flip every grade in the CSV and the document must flip with it."""
    root = _write_artifacts(tmp_path)
    path = root / "thresholds" / "tables" / "28_threshold_evidence.csv"
    flipped = pd.read_csv(path)
    flipped["verdict"] = ["weak", "strong"]
    flipped["limiting_criterion"] = ["Non-linearity", ""]
    flipped.to_csv(path, index=False)

    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "1 strong" in html and "1 weak" in html
    assert "Volume <b>strong</b>" in html
    assert "ADC <b>weak</b>" in html


def test_count_phrase_agrees_in_number():
    """The verb follows the count of thresholds, which is the sentence's subject.

    "4 of 4 measurements has" was the published bug; "1 of 2 measurements has"
    is correct English and must survive.
    """
    yes, no = tr.MetricVerdict("ADC", "adc", True), tr.MetricVerdict("Vol", "vol", False)
    assert tr.Verdicts(()).count_phrase(verb=True) == "0 of 0 measurements have"
    assert tr.Verdicts((yes,)).count_phrase(verb=True) == "1 of 1 measurement has"
    assert tr.Verdicts((yes, no)).count_phrase(verb=True) == "1 of 2 measurements has"
    assert tr.Verdicts((yes, yes)).count_phrase(verb=True) == "2 of 2 measurements have"
    assert tr.Verdicts((no, no)).count_phrase(verb=True) == "0 of 2 measurements have"


def test_blank_limiting_criterion_never_prints_nan(tmp_path):
    root = _write_artifacts(tmp_path)
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "limited by nan" not in html
    assert "all criteria met" in html


def test_grade_tally_orders_best_supported_first():
    items = tuple(tr.MetricVerdict(g, g, True, grade=g)
                  for g in ("weak", "strong", "moderate"))
    assert [g for g, _ in tr.Verdicts(items).grade_tally()] == [
        "strong", "moderate", "weak"]
    assert tr.Verdicts(items).strongest()[0].grade == "strong"


def test_falls_back_to_the_bare_test_when_grades_are_missing(tmp_path):
    root = _write_artifacts(tmp_path)
    (root / "thresholds" / "tables" / "28_threshold_evidence.csv").unlink()
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "graded evidence table was not found" in html
    assert "measurements with a true turning point" in html


# --------------------------------------------------------------------------
# One denominator
# --------------------------------------------------------------------------
def test_shared_denominator_is_the_primary_comparison(cfg):
    html = tr.build_report(cfg)
    assert "Every rule below is scored on the same patients" in html
    assert "Edema volume ≥ 4.76" in html          # the shared-cohort winner
    assert "Secondary: all available data" in html


def test_every_section_quotes_the_same_denominator(cfg):
    """Sections 5, 7 and 9 must not disagree about which cohort they mean."""
    data = tr.load_report_data(cfg)
    verdict = tr.primary_verdict(data)
    assert int(verdict["n_used"].iloc[0]) == 304
    assert verdict["best_single_rule"].iloc[0] == "Edema volume ≥ 4.76"


def test_falls_back_to_all_available_data_when_shared_is_missing(tmp_path):
    root = _write_artifacts(tmp_path)
    (root / "thresholds" / "tables" / "32_shared_combination_verdict.csv").unlink()
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "shared-denominator comparison was not found" in html
    assert "Volume ≥ 15.1" in html                # the all-available-data winner


def test_all_available_data_verdict_reports_n_as_varying(tmp_path):
    root = _write_artifacts(tmp_path)
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "varies" in html


# --------------------------------------------------------------------------
# Zero inflation
# --------------------------------------------------------------------------
def test_zero_inflation_is_a_section_3_method_note(cfg):
    html = tr.build_report(cfg)
    assert "none at all" in html
    assert "36.6%" in html          # the real share, not "a quarter"
    assert "a quarter of this cohort" not in html
    assert "Which of the three is the defensible claim" in html


def test_caveat_names_the_zero_inflated_metrics_from_the_table(tmp_path):
    root = _write_artifacts(tmp_path)
    path = root / "thresholds" / "tables" / "34_zero_inflation.csv"
    flat = pd.read_csv(path)
    flat["zero_inflated"] = False
    flat.to_csv(path, index=False)
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "No measurement in this run was zero-inflated" in html


# --------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------
def test_multiplicity_prints_both_corrections(cfg):
    html = tr.build_report(cfg)
    assert "Multiple testing, stated before it is asked about" in html
    assert "2 of 2</b> survive Holm" in html
    assert "Bonferroni" in html
    # The old text conceded the point outright.
    assert "No, deliberately, and we state it" not in html


def test_multiplicity_answer_names_what_bonferroni_drops(cfg):
    data = tr.load_report_data(cfg)
    m = tr.multiplicity_facts(data)
    assert m.n_tests == 2 and m.n_holm == 2 and m.n_bonferroni == 1
    assert m.dropped_by_bonferroni == ("Volume",)
    assert "Volume" in tr._multiplicity_answer(m)


def test_multiplicity_degrades_when_the_table_is_missing(tmp_path):
    root = _write_artifacts(tmp_path)
    (root / "thresholds" / "tables" / "38_nonlinearity_multiplicity.csv").unlink()
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "adjusted table was not found" in html


# --------------------------------------------------------------------------
# Calibration and net benefit
# --------------------------------------------------------------------------
def test_calibration_reaches_section_7_and_section_9(cfg):
    html = tr.build_report(cfg)
    assert "Calibration — is the probability the right size?" in html
    assert "0.91" in html                       # the corrected slope
    assert "Is the model calibrated?" in html


def test_missing_corrected_intercept_is_shown_as_missing_not_filled_in(cfg):
    html = tr.build_report(cfg)
    assert "apparent 0.00" in html
    assert "carry a bootstrap-corrected calibration" in html


def test_net_benefit_reaches_section_7_and_section_9(cfg):
    html = tr.build_report(cfg)
    assert "Net benefit — what is each strategy actually worth?" in html
    assert "Should anyone actually act on this?" in html
    assert "59% of that range" in html


def test_net_benefit_names_the_strategies_that_never_beat_the_references(cfg):
    nb = tr.net_benefit_facts(tr.load_report_data(cfg))
    assert nb.available
    assert nb.winner == "Uncut four-measurement model"
    assert nb.useless == ("Best single cut-point (Edema volume ≥ 4.76)",)
    assert "never beats" in tr.build_report(cfg)


def test_report_degrades_without_calibration_or_net_benefit(tmp_path):
    root = _write_artifacts(tmp_path)
    for name in ("40_calibration.csv", "44_net_benefit_summary.csv"):
        (root / "thresholds" / "tables" / name).unlink()
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "No calibration table was found" in html
    assert "No decision curve was found" in html


# --------------------------------------------------------------------------
# Presentation correctness (P2)
# --------------------------------------------------------------------------
def test_high_grade_denominator_is_added_up_not_printed_as_arithmetic(cfg):
    html = tr.build_report(cfg)
    assert "34+62" not in html
    assert "of the 96 high-grade tumours" in html


def test_dichotomised_rules_are_not_labelled_auc(cfg):
    html = tr.build_report(cfg)
    assert "AUC ≈" not in html
    assert "Balanced accuracy" in html
    assert "Balanced accuracy (dichotomised)</b> is (sensitivity" in html


def test_numeric_cells_cannot_wrap_mid_token(cfg):
    """"86/127/15/99" broke after a slash and read as "86/127/1 5/99"."""
    html = tr.build_report(cfg)
    assert 'class="nowrap">≤0.72' in html


def test_bootstrap_counts_come_from_the_manifest_not_from_prose(tmp_path):
    root = _write_artifacts(tmp_path)
    path = root / "thresholds" / "manifest.json"
    payload = json.loads(path.read_text())
    payload["context"] = {"n_bootstrap": 4000, "n_bootstrap_curve": 250,
                          "seed": 20260801}
    path.write_text(json.dumps(payload))
    html = tr.build_report(tr.ThresholdReportConfig(output_root=root))
    assert "several hundred resamples" not in html
    assert "refitted on 250 resampled cohorts" in html
    assert "4000 resamples" in html


def test_thirty_percent_crossing_is_called_the_base_rate(cfg):
    html = tr.build_report(cfg)
    assert "base-rate crossing" in html


def test_epv_is_a_column_not_an_argument(tmp_path):
    root = _write_artifacts(tmp_path, models=True)
    data = tr.load_report_data(tr.ThresholdReportConfig(output_root=root))
    row = data.model_aucs.iloc[0]
    assert row["events"] == 105
    assert row["EPV"] == pytest.approx(105 / 2)
    assert ">EPV</th>" in tr.build_report(tr.ThresholdReportConfig(output_root=root))


def test_defence_section_cites_live_numbers(cfg):
    html = tr.build_report(cfg)
    assert "Is 352 patients enough" in html
    assert "105 events" in html
    assert "2000 bootstrap resamples" in html


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_writes_the_default_path(artifacts, capsys):
    assert tr.main(["--output-root", str(artifacts)]) == 0
    out = artifacts / "thresholds" / "threshold_report.html"
    assert out.exists()
    assert out.read_text().startswith("<!DOCTYPE html>")
    assert "Threshold report written" in capsys.readouterr().out


def test_cli_honours_out_and_author(artifacts, tmp_path):
    target = tmp_path / "custom" / "page.html"
    assert tr.main(["--output-root", str(artifacts), "--out", str(target),
                    "--author", "A. Zaguzovs"]) == 0
    assert "A. Zaguzovs" in target.read_text()
