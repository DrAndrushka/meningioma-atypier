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
            {"Metric": "ADC", "Threshold effect?": "yes", "Non-linearity p": "= 0.009"},
            {"Metric": "Volume", "Threshold effect?": "no — risk is linear",
             "Non-linearity p": "= 0.974"},
        ]),
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


def test_report_contains_every_section(cfg):
    html = tr.build_report(cfg)
    for heading in ("Three questions", "šķēre", "Where should the line be drawn",
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
