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


# ---------------------------------------------------------------------------
# Overview figure — citation labels and the two comparators
# ---------------------------------------------------------------------------

def _overview(auc_app, auc_corr, ref, own):
    return {"auc_apparent": auc_app, "auc_corrected": auc_corr,
            "ref": ref, "own": own}


@pytest.mark.parametrize("citation, expected", [
    ("Radeesri K, Lekhavat V. The role of pre-operative MRI.", "Radeesri and Lekhavat"),
    ("Peng S, Cheng Z, Guo Z. Diagnostic nomogram model.", "Peng et al"),
    ("Spille DC, Adeli A, Sporns PB, et al. Predicting the risk.", "Spille et al"),
    ("Funari A, De la Garza Ramos R, Cezayirli P, et al. Imaging score.",
     "Funari et al"),
    ("Kawahara Y. Solo work.", "Kawahara"),
])
def test_author_short_follows_ama(citation, expected):
    """One name, two joined by "and", three or more as "et al" — never a year.

    AJNR cites by superscript reference number, so an author-year label like
    "Funari 2023" is the wrong style *and* invites a year that disagrees with
    the reference list.
    """
    assert pp._author_short(citation) == expected


def test_citation_label_carries_the_reference_number():
    published = {"x_2020": {"citation": "Zhang S, Chiang GC, Knapp JM, et al. T.",
                            "reference_number": 9}}
    assert pp._citation_label("x_2020", published) == "Zhang et al$^{9}$"


def test_citation_label_without_a_number_is_still_named():
    published = {"x_2020": {"citation": "Zhang S, Chiang GC, Knapp JM, et al. T."}}
    assert pp._citation_label("x_2020", published) == "Zhang et al"
    assert pp._citation_label("absent", published) is None


def test_discrimination_labels_published_models_by_citation(monkeypatch):
    """The row is named the way the running text names it, not by model id."""
    monkeypatch.setattr(pp, "_published_models", lambda: {
        "lin_2014": {"citation": "Lin BJ, Chou KN, Kao HW, et al. Correlation.",
                     "reference_number": 12}})
    entries = [{"model_id": "lin_2014", "label": "Lin et al. 2014 | x",
                "n_predictors": 4},
               {"model_id": "top_6_variables", "label": "Top 6 variables",
                "n_predictors": 6}]
    overview = {"lin_2014": _overview(0.644, 0.628, (0.01, -0.02, 0.04),
                                      (0.02, -0.01, 0.05)),
                "top_6_variables": _overview(0.760, 0.725, (0.046, -0.005, 0.097),
                                             (0.046, -0.005, 0.097))}
    fig = pp.model_discrimination_figure(
        entries, target="high_grade", overview=overview)
    assert fig is not None
    drawn = {t.get_text() for t in fig.axes[0].texts}
    assert "Lin et al$^{12}$" in drawn
    assert "Top 6 variables" in drawn
    assert not any(t.startswith("Lin 2014") for t in drawn)
    plt.close(fig)


def test_discrimination_draws_no_reference_line():
    """Tumour volume is a searched-for winner, not a benchmark to rule a line at.

    Every comparison against it lives in the gain figure. A dashed rule here
    invited the reader to make that comparison by eye, on a scale where the
    comparator's own selection cost is not charged.
    """
    entries = [{"model_id": "a", "label": "A", "n_predictors": 6},
               {"model_id": "b", "label": "B", "n_predictors": 10}]
    overview = {"a": _overview(0.760, 0.725, (0.046, -0.005, 0.097),
                               (0.046, -0.005, 0.097)),
                "b": _overview(0.702, 0.662, (-0.017, -0.089, 0.056),
                               (0.071, 0.017, 0.122))}
    fig = pp.model_discrimination_figure(
        entries, target="high_grade", overview=overview)
    assert fig is not None
    dashed = [ln for ln in fig.axes[0].lines
              if ln.get_linestyle() in {"--", (0, (6.4, 1.6))}]
    assert dashed == []
    plt.close(fig)


def test_gain_draws_both_comparators_in_separate_panels():
    """Two yardsticks, one marker to a row in each — never stacked on one line.

    Stacking them, with two numeric columns beside, is what made the old plate
    unreadable. A model whose own strongest predictor happens to be the shared
    comparator now simply shows the same position in both panels.
    """
    same = _overview(0.760, 0.725, (0.046, -0.005, 0.097), (0.046, -0.005, 0.097))
    differ = _overview(0.702, 0.662, (-0.017, -0.089, 0.056), (0.071, 0.017, 0.122))
    entries = [{"model_id": "a", "label": "A", "n_predictors": 6},
               {"model_id": "b", "label": "B", "n_predictors": 10}]
    fig = pp.model_gain_figure(
        entries, target="high_grade", overview={"a": same, "b": differ},
        reference_auc=0.661, reference_label="tumor volume")
    assert fig is not None
    assert len(fig.axes) == 2
    for ax in fig.axes:
        # matplotlib normalises linestyle="none" to the string "None".
        markers = [ln for ln in ax.lines
                   if str(ln.get_linestyle()).lower() == "none"
                   and ln.get_marker() not in (None, "None", "")]
        assert len(markers) == 2
    plt.close(fig)


def test_gain_scores_the_own_comparator_per_row_not_per_column():
    """The left comparator changes with the model, so its AUC cannot be a heading.

    Both rows here name tumor volume, but one is scored as the model specifies
    it (0.679) and the other as the winner of a search (0.661). A single AUC
    over the column would be true of one row and wrong for the other; only the
    right panel's comparator is shared by every row, so only it gets a heading
    number.
    """
    a = _overview(0.760, 0.736, (0.075, 0.044, 0.128), (0.057, 0.010, 0.104))
    a |= {"best_own_single": "tumor_volume", "best_own_auc_corrected": 0.679}
    b = _overview(0.702, 0.662, (0.001, -0.039, 0.056), (0.071, 0.017, 0.122))
    b |= {"best_own_single": "irregular_tumor_margin",
          "best_own_auc_corrected": 0.625}
    entries = [{"model_id": "a", "label": "A", "n_predictors": 6},
               {"model_id": "b", "label": "B", "n_predictors": 10}]
    fig = pp.model_gain_figure(
        entries, target="high_grade", overview={"a": a, "b": b},
        reference_auc=0.661, reference_label="tumor volume")
    texts = [t.get_text() for t in fig.texts]
    assert "vs tumor volume (0.679)" in texts
    assert "vs irregular tumor margin (0.625)" in texts
    # The shared comparator's score is stated once, over its own panel only.
    assert sum("0.661" in t for t in texts) == 1
    plt.close(fig)


def test_gain_prints_a_real_minus_sign_not_a_hyphen():
    """A hyphen is a word-break; at 7 pt it does not read as a sign at all."""
    entries = [{"model_id": "a", "label": "A", "n_predictors": 2},
               {"model_id": "b", "label": "B", "n_predictors": 3}]
    overview = {"a": _overview(0.66, 0.65, (-0.03, -0.09, 0.03), (0.02, -0.01, 0.06)),
                "b": _overview(0.70, 0.69, (0.02, -0.06, 0.09), (0.07, 0.01, 0.13))}
    fig = pp.model_gain_figure(
        entries, target="high_grade", overview=overview,
        reference_auc=0.661, reference_label="tumor volume")
    joined = "".join(t.get_text() for ax in fig.axes
                     for t in ax.get_xticklabels())
    assert "−" in joined and "-0." not in joined
    plt.close(fig)


def test_both_overview_figures_need_two_models():
    one = {"a": _overview(0.7, 0.68, (None, None, None), (None, None, None))}
    for build in (pp.model_discrimination_figure, pp.model_gain_figure):
        assert build([{"model_id": "a", "label": "A"}], overview=one) is None
        assert build([{"model_id": "a"}], overview={}) is None


