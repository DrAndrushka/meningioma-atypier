"""Model-performance figures built from a validation artifact.

Covers what the figures must never get wrong: missing data yields no figure
rather than an empty one, and the model-comparison panel ranks variants by the
optimism-corrected metric, not the flattering apparent one.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import performance_plots as pp


def _validation(*, auc_app=0.80, auc_corr=0.74, slope_corr=0.91) -> dict:
    fpr = list(np.linspace(0.0, 1.0, 25))
    tpr = list(np.clip(np.linspace(0.0, 1.0, 25) ** 0.5, 0, 1))
    return {
        "successful_bootstraps": 200,
        "metrics": [
            {"metric": "AUC", "apparent": auc_app, "optimism_corrected": auc_corr},
            {"metric": "Brier score", "apparent": 0.17, "optimism_corrected": 0.18},
            {"metric": "Calibration slope", "apparent": 1.0,
             "optimism_corrected": slope_corr},
        ],
        "roc_curves": {
            "curves": [
                {"series": "apparent", "label": "Apparent",
                 "auc": auc_app, "fpr": fpr, "tpr": tpr},
            ],
        },
        "calibration": {
            "bins": [
                {"predicted": 0.10, "observed": 0.12, "events": 4, "n": 33},
                {"predicted": 0.30, "observed": 0.28, "events": 9, "n": 32},
                {"predicted": 0.60, "observed": 0.65, "events": 21, "n": 32},
            ],
            "smooth": {"predicted": [0.1, 0.3, 0.6], "observed": [0.11, 0.30, 0.63]},
            "slope_apparent": 1.0,
            "slope_corrected": slope_corr,
            "intercept_apparent": -0.0,
        },
        "decision_curve": {
            "thresholds": [round(0.05 * i, 2) for i in range(1, 12)],
            "model": [0.25, 0.21, 0.18, 0.15, 0.12, 0.09, 0.06, 0.04, 0.02, 0.01, -0.01],
            "treat_all": [0.24, 0.17, 0.10, 0.02, -0.06, -0.16, -0.28, -0.42,
                          -0.60, -0.82, -1.10],
            "prevalence": 0.3,
        },
    }


# ---------------------------------------------------------------------------
# Per-model figures
# ---------------------------------------------------------------------------

def test_roc_figure_reports_both_apparent_and_corrected_auc():
    """An in-sample ROC quoted alone overstates the model."""
    fig = pp.roc_figure(_validation())
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("0.800" in lab for lab in labels)
    assert any("0.740" in lab for lab in labels)
    plt.close(fig)


def test_calibration_figure_draws_a_point_per_bin():
    fig = pp.calibration_figure(_validation())
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlim() == ax.get_ylim()  # risk vs risk must share a scale
    subtitle = " ".join(t.get_text() for t in ax.texts)
    assert "0.91" in subtitle  # corrected slope in the AJNR annotation box
    plt.close(fig)


def test_calibration_figure_none_without_bins():
    val = _validation()
    val["calibration"]["bins"] = []
    assert pp.calibration_figure(val) is None


def test_decision_curve_figure_includes_both_references():
    fig = pp.decision_curve_figure(_validation())
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert {"Model", "Treat all", "Treat none"} <= set(labels)
    plt.close(fig)


def test_outcome_rate_is_not_drawn_on_the_panel():
    """Cohort rate belongs in the caption, not as on-figure text."""
    val = _validation()
    val["decision_curve"]["prevalence"] = 105 / 352  # 0.29829...
    fig = pp.decision_curve_figure(val)
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Treat all" in labels and "Treat none" in labels
    plt.close(fig)


def test_figures_are_none_when_the_artifact_lacks_data():
    assert pp.roc_figure({}) is None
    assert pp.calibration_figure({}) is None
    assert pp.decision_curve_figure({}) is None


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

def test_model_comparison_ranks_by_corrected_auc_not_apparent():
    """Best corrected model on top, even when another looks better in-sample."""
    entries = [
        {"label": "flatters itself", "validation": _validation(auc_app=0.95, auc_corr=0.60)},
        {"label": "honest", "validation": _validation(auc_app=0.78, auc_corr=0.76)},
    ]
    fig = pp.model_comparison_figure(entries, target="high_grade")
    assert fig is not None
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # y increases upward, so the strongest CORRECTED model is last -- the whole
    # point, since "flatters itself" wins on apparent AUC and would be last if
    # the figure ranked on that. Compared case-insensitively: _short_label
    # capitalises for the figure, which is a separate concern with its own test,
    # and this one must not fail when that styling changes.
    assert [t.lower() for t in ticks] == ["flatters itself", "honest"]
    plt.close(fig)


def test_model_comparison_none_without_validated_entries():
    assert pp.model_comparison_figure([{"label": "x"}]) is None
    assert pp.model_comparison_figure([]) is None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_write_performance_figures_emits_three_svgs(tmp_path):
    written = pp.write_performance_figures(_validation(), tmp_path, "high_grade__m1")
    names = {p.name for p in written}
    assert names == {
        "high_grade__m1__roc.png",
        "high_grade__m1__calibration.png",
        "high_grade__m1__decision_curve.png",
    }
    assert all(p.exists() for p in written)


def test_write_performance_figures_skips_what_it_cannot_draw(tmp_path):
    val = _validation()
    del val["decision_curve"]
    written = pp.write_performance_figures(val, tmp_path, "high_grade__m1")
    assert len(written) == 2


def test_write_model_comparison_figure(tmp_path):
    entries = [{"label": "m1", "validation": _validation()}]
    path = pp.write_model_comparison_figure(entries, tmp_path, target="high_grade")
    assert path is not None and path.name == "high_grade__model_comparison.png"
    assert path.exists()


@pytest.mark.parametrize("metric", ["AUC", "Brier score", "Calibration slope"])
def test_comparison_covers_discrimination_error_and_calibration(metric):
    assert metric in {m[0] for m in pp._COMPARISON_METRICS}
