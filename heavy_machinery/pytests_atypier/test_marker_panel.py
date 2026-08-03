"""Marker panel: LR+, the BinaryMarker adapter, rule menus, model re-scoring."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import combinations as cb
import marker_panel as mp
from thresholds import Metric

TARGET = "high_grade"


# --------------------------------------------------------------------------
# Positive likelihood ratio
# --------------------------------------------------------------------------
def test_lr_pos_matches_a_hand_computed_2x2():
    """27 of 105 high-grade flagged, 23 of 247 benign flagged.

    sens = 27/105 = 0.2571, spec = 224/247 = 0.9069, LR+ = sens / (1 - spec).
    Katz log-scale interval: exp(log LR+ ± 1.96 * sqrt(1/TP - 1/(TP+FN) + 1/FP - 1/(FP+TN))).
    """
    out = mp.likelihood_ratio_positive(tp=27, fp=23, fn=78, tn=224)
    assert out["lr_pos"] == pytest.approx(2.7615, abs=1e-4)
    assert out["lr_pos_lo"] == pytest.approx(1.6631, abs=1e-3)
    assert out["lr_pos_hi"] == pytest.approx(4.5854, abs=1e-3)
    assert out["chance_overlap"] is False
    assert out["continuity_corrected"] is False


def test_lr_pos_flags_a_marker_whose_interval_covers_one():
    """A sign that fires equally often in both groups carries no information."""
    out = mp.likelihood_ratio_positive(tp=20, fp=45, fn=85, tn=202)
    assert out["lr_pos_lo"] < 1.0 < out["lr_pos_hi"]
    assert out["chance_overlap"] is True


def test_lr_pos_survives_a_zero_cell_with_a_continuity_correction():
    """brain_invasion-shaped: never seen in a benign tumour, so FP = 0.

    Without a correction LR+ is infinite and its interval undefined. Adding 0.5
    to every cell (Haldane-Anscombe) gives a finite, very wide interval — which
    is the honest answer: a huge point estimate resting on five patients.
    """
    out = mp.likelihood_ratio_positive(tp=5, fp=0, fn=100, tn=247)
    assert np.isfinite(out["lr_pos"])
    assert out["lr_pos"] == pytest.approx(25.7358, abs=1e-3)
    assert out["lr_pos_lo"] == pytest.approx(1.4358, abs=1e-2)
    assert out["lr_pos_hi"] == pytest.approx(461.3, rel=1e-3)
    assert out["continuity_corrected"] is True


def test_lr_pos_returns_nan_when_a_margin_is_empty():
    """No high-grade patients at all: nothing to compute, and no crash."""
    out = mp.likelihood_ratio_positive(tp=0, fp=3, fn=0, tn=40)
    assert np.isnan(out["lr_pos"])
    assert out["chance_overlap"] is False


def marker_frame() -> pd.DataFrame:
    """Eight patients, three signs, one missing value in each of two signs."""
    return pd.DataFrame({
        "sign_a": pd.array([True, True, False, False, True, False, None, True],
                           dtype="boolean"),
        "sign_b": pd.array([True, False, True, False, None, False, True, True],
                           dtype="boolean"),
        "sign_c": pd.array([False, False, False, False, True, True, True, False],
                           dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, True],
                         dtype="boolean"),
    })


def accuracy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": TARGET, "predictor": TARGET, "kind": "binary"},
        {"target": TARGET, "predictor": "sign_a", "kind": "binary"},
        {"target": TARGET, "predictor": "sign_b", "kind": "derived_binary"},
        {"target": TARGET, "predictor": "sex_male", "kind": "derived_binary"},
        {"target": TARGET, "predictor": "adc_value", "kind": "continuous"},
        {"target": "other", "predictor": "sign_c", "kind": "binary"},
    ])


# --------------------------------------------------------------------------
# BinaryMarker — the adapter that lets combinations.py accept yes/no columns
# --------------------------------------------------------------------------
def test_binary_marker_flags_match_an_equivalent_cutpoint():
    """The reuse claim, verified: a marker and a 0.5 cut-point flag the same rows."""
    df = marker_frame()
    numeric = df.assign(sign_a=df["sign_a"].astype("Float64"))
    marker = mp.BinaryMarker("sign_a", "Sign A")
    cutpoint = cb.CutPoint(Metric("sign_a", "Sign A", "", "higher"), 0.5)

    from_marker = marker.flag(df)
    from_cutpoint = cutpoint.flag(numeric)
    pd.testing.assert_series_equal(
        from_marker.astype("boolean"), from_cutpoint.astype("boolean"),
        check_names=False,
    )


def test_binary_marker_short_label_is_the_label():
    """No cut-point to print, so the short form is just the name."""
    marker = mp.BinaryMarker("sign_a", "Sign A")
    assert marker.label == "Sign A"
    assert marker.short_label == "Sign A"


def test_combinations_accepts_binary_markers_unchanged():
    """single_rule_table is threshold-phase code, called here on plain columns."""
    df = marker_frame()
    markers = [mp.BinaryMarker("sign_a", "Sign A"), mp.BinaryMarker("sign_b", "Sign B")]
    table = cb.single_rule_table(df, markers, TARGET)
    assert list(table["rule_label"]) == ["Sign A", "Sign B"]
    assert table["youden_J"].notna().all()


# --------------------------------------------------------------------------
# Marker selection
# --------------------------------------------------------------------------
def test_markers_are_read_from_the_accuracy_table():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert [m.col for m in markers] == ["sign_a", "sign_b", "sex_male"]


def test_the_outcome_is_never_treated_as_a_marker():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert TARGET not in [m.col for m in markers]


def test_continuous_predictors_and_other_targets_are_left_out():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    cols = [m.col for m in markers]
    assert "adc_value" not in cols
    assert "sign_c" not in cols


def test_the_exclude_list_excludes():
    """sex_male is derived_binary and would otherwise enter a section on MRI signs."""
    markers = mp.markers_from_diagnostic_accuracy(
        accuracy_table(), target=TARGET, exclude={"sex_male"},
    )
    assert [m.col for m in markers] == ["sign_a", "sign_b"]


def test_marker_labels_are_prettified():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert markers[0].label == "Sign A"


# --------------------------------------------------------------------------
# Aim 1 — the marker table
# --------------------------------------------------------------------------
def test_marker_panel_reports_yield_alongside_specificity():
    """The guard against the specificity trap.

    ``rare`` is present in one patient and never in a benign tumor, so its
    specificity is 1.0 — and it catches 1 of 5 high-grade tumors. Both numbers
    must be in the row, or the table crowns a useless sign.
    """
    df = pd.DataFrame({
        "rare": pd.array([True, False, False, False, False, False, False, False],
                         dtype="boolean"),
        "common": pd.array([True, True, True, False, True, True, False, False],
                           dtype="boolean"),
        TARGET: pd.array([True, True, True, True, True, False, False, False],
                         dtype="boolean"),
    })
    markers = [mp.BinaryMarker("rare", "Rare"), mp.BinaryMarker("common", "Common")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    rare = panel[panel["marker"] == "rare"].iloc[0]
    assert rare["specificity"] == 1.0
    assert rare["present_n"] == 1
    assert rare["catches"] == 1
    assert rare["n_high_grade"] == 5


def test_markers_that_cannot_beat_chance_sort_last():
    """A ranked table must not open with a row that says nothing."""
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.3, 300).astype(bool)
    df = pd.DataFrame({
        "informative": pd.array(rng.binomial(1, 0.15 + 0.5 * y).astype(bool),
                                dtype="boolean"),
        "noise": pd.array(rng.binomial(1, 0.4, 300).astype(bool), dtype="boolean"),
        TARGET: pd.array(y, dtype="boolean"),
    })
    markers = [mp.BinaryMarker("noise", "Noise"),
               mp.BinaryMarker("informative", "Informative")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    assert panel.iloc[0]["marker"] == "informative"
    assert bool(panel.iloc[-1]["chance_overlap"]) is True


def test_marker_reading_view_says_so_instead_of_printing_a_rank():
    df = pd.DataFrame({
        "noise": pd.array([True, False, True, False, True, False, True, False],
                          dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, False],
                         dtype="boolean"),
    })
    panel = mp.marker_panel_table(df, [mp.BinaryMarker("noise", "Noise")], TARGET)
    view = mp.marker_panel_reading_view(panel)
    assert list(view.columns) == [
        "Marker", "Present in", "Catches",
        "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
    ]
    assert "not distinguishable from chance" in view.iloc[0]["LR+ (95% CI)"]


def test_marker_panel_is_empty_not_broken_when_there_are_no_markers():
    df = pd.DataFrame({TARGET: pd.array([True, False], dtype="boolean")})
    panel = mp.marker_panel_table(df, [], TARGET)
    assert panel.empty
    assert mp.marker_panel_reading_view(panel).empty


# --------------------------------------------------------------------------
# Aim 1 — the figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt


def test_lr_forest_draws_one_row_per_marker_on_a_log_axis():
    df = pd.DataFrame({
        "a": pd.array([True, True, False, False, True, False, False, False],
                      dtype="boolean"),
        "b": pd.array([True, False, True, False, True, True, False, False],
                      dtype="boolean"),
        TARGET: pd.array([True, True, True, True, True, False, False, False],
                         dtype="boolean"),
    })
    markers = [mp.BinaryMarker("a", "Sign A"), mp.BinaryMarker("b", "Sign B")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    fig = mp.lr_forest_figure(panel)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(ax.get_yticklabels()) == 2
    plt.close(fig)


def test_lr_forest_returns_a_figure_even_with_nothing_to_plot():
    """An empty panel must not crash the notebook cell that saves figures."""
    fig = mp.lr_forest_figure(pd.DataFrame(columns=["label", "lr_pos"]))
    assert fig is not None
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — one denominator
# --------------------------------------------------------------------------
def sparse_frame() -> pd.DataFrame:
    """Six patients; ``sign_b`` is missing for two of them."""
    return pd.DataFrame({
        "sign_a": pd.array([True, False, True, False, True, False], dtype="boolean"),
        "sign_b": pd.array([True, False, None, None, True, True], dtype="boolean"),
        "always_off": pd.array([False] * 6, dtype="boolean"),
        TARGET: pd.array([True, False, True, False, True, False], dtype="boolean"),
    })


def test_shared_cohort_keeps_only_patients_with_every_marker_observed():
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("sign_b", "B")]
    shared = mp.shared_cohort_frame(df, markers, TARGET)
    assert len(shared) == 4
    assert shared["sign_b"].notna().all()


def test_a_marker_that_never_fires_is_dropped_with_a_reason():
    """An all-false column has an undefined likelihood ratio and no rule value."""
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("always_off", "Off")]
    kept, dropped = mp.usable_markers(df, markers, TARGET)
    assert [m.col for m in kept] == ["sign_a"]
    assert dropped[0]["marker"] == "always_off"
    assert "never" in dropped[0]["reason"].lower()


def test_shared_cohort_audit_records_what_each_marker_cost():
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("sign_b", "B")]
    audit = mp.shared_cohort_audit(df, markers, TARGET, dropped=[])
    assert set(audit.columns) == {"item", "value", "note"}
    rows = dict(zip(audit["item"], audit["value"]))
    assert rows["Patients in the shared set"] == 4
    assert rows["sign_b"] == 2  # patients this marker cost


def test_shared_cohort_is_empty_not_broken_when_no_patient_has_everything():
    df = pd.DataFrame({
        "a": pd.array([True, None], dtype="boolean"),
        "b": pd.array([None, True], dtype="boolean"),
        TARGET: pd.array([True, False], dtype="boolean"),
    })
    markers = [mp.BinaryMarker("a", "A"), mp.BinaryMarker("b", "B")]
    assert mp.shared_cohort_frame(df, markers, TARGET).empty


# --------------------------------------------------------------------------
# Aim 2 — the count score
# --------------------------------------------------------------------------
def count_frame(n: int = 240, seed: int = 3) -> pd.DataFrame:
    """Three signs, each independently more common in high-grade tumors."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {
        f"sign_{i}": pd.array(rng.binomial(1, 0.15 + 0.45 * y).astype(bool),
                              dtype="boolean")
        for i in range(3)
    }
    cols[TARGET] = pd.array(y, dtype="boolean")
    return pd.DataFrame(cols)


COUNT_MARKERS = [mp.BinaryMarker(f"sign_{i}", f"Sign {i}") for i in range(3)]


def test_count_score_has_a_row_for_every_possible_count():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    assert list(counts["n_criteria_met"]) == [0, 1, 2, 3]
    assert counts["n"].sum() == counts.attrs["n_scored"]


def test_risk_climbs_with_the_number_of_signs_present():
    """The literal claim the section makes. If this fails, the claim is wrong."""
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    risks = counts[counts["n"] >= 10]["risk"].to_numpy(dtype=float)
    assert risks[0] < risks[-1]


def test_count_thresholds_are_scored_as_tests():
    rules = mp.count_thresholds(count_frame(), COUNT_MARKERS, TARGET)
    assert list(rules["rule_label"]) == [
        "≥ 1 of 3 criteria", "≥ 2 of 3 criteria", "≥ 3 of 3 criteria",
    ]
    assert rules["youden_J"].notna().all()


def test_count_score_figure_labels_the_axis_with_the_marker_count():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.count_score_figure(counts, COUNT_MARKERS)
    ax = fig.axes[0]
    assert "3" in ax.get_xlabel()
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — the rule menu, and paying for having picked a winner
# --------------------------------------------------------------------------
def test_rule_menu_holds_singles_and_combinations_together():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    kinds = set(menu["kind"])
    assert {"single", "and", "or", "count"} <= kinds
    assert (menu["n_used"] > 0).all()


def test_both_sides_of_the_head_to_head_are_corrected():
    """The CHANGES.md regression.

    A corrected combination scored against an *uncorrected* single flatters the
    combination by the whole of the single's own selection optimism. Picking the
    best of N single markers is a choice made on these patients too, so it costs
    something, and that cost must be non-zero and recorded.
    """
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    assert list(corr["side"]) == ["best single", "best combination"]
    assert corr["optimism"].notna().all()
    assert (corr["optimism"] > 0).all()
    assert corr["J_corrected"].notna().all()


def test_the_reported_gain_is_corrected_on_both_sides():
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    single = corr[corr["side"] == "best single"].iloc[0]
    combo = corr[corr["side"] == "best combination"].iloc[0]
    expected = combo["J_corrected"] - single["J_corrected"]
    assert corr["gain_corrected"].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_selection_correction_is_deterministic_for_a_seed():
    args = (count_frame(), COUNT_MARKERS, TARGET)
    a = mp.selection_correction(*args, n_boot=40, seed=7)
    b = mp.selection_correction(*args, n_boot=40, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_rule_reading_view_is_ranked_by_youden_j():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    view = mp.rule_reading_view(menu, top=5)
    assert len(view) == 5
    assert list(view["J"]) == sorted(view["J"], reverse=True)


def test_rule_space_figure_draws():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.rule_space_figure(menu)
    assert fig.axes
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — the multivariable comparison
# --------------------------------------------------------------------------
def tiny_artifact() -> dict:
    """A two-predictor logistic model in the shape the pipeline saves."""
    return {
        "model_name": "Tiny model",
        "target": TARGET,
        "coefficients": {"const": -1.0, "sign_0": 1.5, "sign_1": 0.8},
        "features": [
            {"name": "sign_0", "type": "binary",
             "encoding": {"sign_0": {"true": 1, "false": 0}}},
            {"name": "sign_1", "type": "binary",
             "encoding": {"sign_1": {"true": 1, "false": 0}}},
        ],
        "validation": {"metrics": [
            {"metric": "AUC", "apparent": 0.71, "optimism_corrected": 0.68},
        ]},
    }


def test_scoring_a_model_gives_one_probability_per_patient():
    df = count_frame()
    probs = mp.score_model_on(df, tiny_artifact())
    assert len(probs) == len(df)
    assert probs.between(0, 1).all()


def test_a_patient_missing_a_predictor_scores_nan_not_a_guess():
    """Imputing silently inside a scoring helper would be a lie by omission."""
    df = count_frame().copy()
    df.loc[df.index[0], "sign_0"] = pd.NA
    probs = mp.score_model_on(df, tiny_artifact())
    assert pd.isna(probs.iloc[0])
    assert probs.iloc[1:].notna().all()


def test_model_vs_single_keeps_the_two_aucs_in_separate_labelled_columns():
    """The artifact AUC and the re-scored AUC are different patients.

    Collapsing them into one column is the denominator mistake this section
    exists to avoid, so the table carries both and says which is which.
    """
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)

    row = table.iloc[0]
    assert row["model"] == "tiny"
    assert row["auc_artifact_corrected"] == 0.68
    assert row["auc_artifact_apparent"] == 0.71
    assert 0.0 <= row["auc_shared_apparent"] <= 1.0
    assert row["n_scored"] == len(df)
    assert row["best_single_rule"] == correction.iloc[0]["best_rule"]


def test_a_model_whose_predictors_are_absent_is_reported_not_dropped():
    df = count_frame().drop(columns=["sign_0"])
    correction = mp.selection_correction(df, COUNT_MARKERS[1:], TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    assert pd.isna(table.iloc[0]["auc_shared_apparent"])
    assert "sign_0" in table.iloc[0]["note"]


def test_model_vs_single_is_empty_when_there_are_no_artifacts():
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    assert mp.model_vs_single(df, {}, TARGET, correction).empty


def test_a_model_with_zero_scorable_patients_is_reported_not_mislabelled():
    """Distinct from the single-outcome-class note: here, every column exists but
    no patient has every predictor recorded, so nobody is scorable at all.

    ``truth.nunique()`` on an empty series is 0, not 2 — the same branch that
    ``one outcome class only`` used to catch, which named the wrong cause.
    """
    df = count_frame().copy()
    df["sign_0"] = pd.array([pd.NA] * len(df), dtype="boolean")
    correction = mp.selection_correction(df, COUNT_MARKERS[1:], TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    row = table.iloc[0]
    assert row["n_scored"] == 0
    assert pd.isna(row["auc_shared_apparent"])
    assert "no patient had every predictor recorded" in row["note"]


# --------------------------------------------------------------------------
# Does filling in the missing scans change the story?
# --------------------------------------------------------------------------
def test_imputation_stability_reports_reproduction_rates():
    """Rubin's rules can average an estimate, but not a choice.

    A different rule can win in each draw, so the honest output is "the same
    rule won in X% of draws", not a pooled winner.
    """
    draws = [count_frame(seed=s) for s in (1, 2, 3)]
    out = mp.imputation_stability(draws, COUNT_MARKERS, TARGET, n_boot=30)
    items = dict(zip(out["item"], out["value"]))
    assert items["Draws"] == 3
    assert 0.0 <= items["Top marker reproduced"] <= 1.0
    assert 0.0 <= items["Winning rule reproduced"] <= 1.0
    assert 0.0 <= items["Combination still beat the best single"] <= 1.0


def test_imputation_stability_says_so_when_there_are_no_draws():
    out = mp.imputation_stability([], COUNT_MARKERS, TARGET)
    assert dict(zip(out["item"], out["value"]))["Draws"] == 0
