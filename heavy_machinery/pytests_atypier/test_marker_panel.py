"""Marker panel: LR+, the BinaryMarker adapter, rule menus, model re-scoring."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import json

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


def test_marker_reading_view_prints_the_estimate_even_when_it_covers_one():
    df = pd.DataFrame({
        "noise": pd.array([True, False, True, False, True, False, True, False],
                          dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, False],
                         dtype="boolean"),
    })
    panel = mp.marker_panel_table(df, [mp.BinaryMarker("noise", "Noise")], TARGET)
    view = mp.marker_panel_reading_view(panel)
    assert list(view.columns) == [
        "Variable", "n/N (%)",
        "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
    ]
    # The interval covers 1; the table still carries the number, and the
    # reader draws the conclusion from the interval.
    assert bool(panel.iloc[0]["chance_overlap"])
    assert view.iloc[0]["LR+ (95% CI)"] == "3.00 (0.50–17.95)"
    assert view.iloc[0]["n/N (%)"] == "4/8 (50%)"


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


def test_a_short_marker_list_is_still_named_in_the_subtitle():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.count_score_figure(counts, COUNT_MARKERS)
    subtitle = " ".join(t.get_text() for t in fig.axes[0].texts) + \
        fig.axes[0].get_title()
    assert "Sign 0" in subtitle
    plt.close(fig)


def wide_frame(k: int = 16, n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {
        f"sign_{i}": pd.array(rng.binomial(1, 0.3 + 0.2 * y).astype(bool),
                              dtype="boolean")
        for i in range(k)
    }
    cols[TARGET] = pd.array(y, dtype="boolean")
    return pd.DataFrame(cols)


def test_a_long_marker_list_is_left_out_of_the_subtitle():
    """Sixteen names is a 200-character line and matplotlib abandons the layout.

    The count is still on the x-axis, and the names are the rows of the aim-1
    table on the same page, so nothing is lost by dropping them here.
    """
    markers = [mp.BinaryMarker(f"sign_{i}", f"Sign {i}") for i in range(16)]
    counts = mp.count_score(wide_frame(), markers, TARGET)
    fig = mp.count_score_figure(counts, markers)
    ax = fig.axes[0]
    text = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
    assert "Sign 15" not in text
    assert "16" in ax.get_xlabel()
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — which two counts the headline sentence may quote
# --------------------------------------------------------------------------
def _counts(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """``(count, n, events)`` triples in the shape ``count_score`` returns."""
    frame = pd.DataFrame([
        {"n_criteria_met": c, "n": n, "n_high_grade": e,
         "risk": (e / n) if n else np.nan,
         "risk_lo": np.nan, "risk_hi": np.nan}
        for c, n, e in rows
    ])
    frame.attrs["k"] = max(c for c, _, _ in rows)
    return frame


def test_the_headline_skips_counts_with_almost_nobody_in_them():
    """The real failure: the top bin held one patient, whose 0% became the claim.

    Occupied is not the same as usable. With the highest bin holding a single
    patient, a sentence built from the first and last occupied rows reported
    two coin flips as a trend.
    """
    head = mp.count_headline(_counts([
        (3, 9, 0), (4, 12, 0), (8, 60, 18), (10, 40, 17), (15, 1, 0),
    ]), min_n=10)
    row = head.iloc[0]
    assert row["low_count"] == 4
    assert row["high_count"] == 10
    assert row["direction"] == "rises"


def test_the_headline_reports_a_fall_as_a_fall():
    """The direction is measured. A hard-coded "rises" is a hope, not a finding."""
    head = mp.count_headline(_counts([
        (0, 40, 24), (1, 50, 20), (2, 60, 6),
    ]), min_n=10)
    assert head.iloc[0]["direction"] == "falls"


def test_the_headline_says_flat_when_the_two_ends_agree():
    head = mp.count_headline(_counts([
        (0, 40, 20), (1, 30, 12), (2, 60, 30),
    ]), min_n=10)
    assert head.iloc[0]["direction"] == "flat"


def test_the_headline_relaxes_its_floor_rather_than_going_silent():
    """A thin honest sentence beats no sentence — but it must say it is thin."""
    head = mp.count_headline(_counts([(0, 4, 0), (1, 5, 3)]), min_n=10)
    row = head.iloc[0]
    assert row["low_count"] == 0 and row["high_count"] == 1
    assert row["min_n"] == 1
    assert "no two counts" in row["note"]


def test_the_headline_is_empty_when_one_count_is_occupied():
    assert mp.count_headline(_counts([(2, 30, 9)]), min_n=10).empty
    assert mp.count_headline(pd.DataFrame()).empty


def test_the_headline_carries_the_denominators_it_quotes():
    head = mp.count_headline(_counts([(1, 25, 2), (4, 40, 20)]), min_n=10)
    row = head.iloc[0]
    assert row["low_n"] == 25 and row["high_n"] == 40
    assert row["k_markers"] == 4


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


def test_the_correction_records_both_gaps_and_which_way_it_moved_them():
    """Whether correction shrinks the gap is data, not doctrine.

    Correction subtracts each side's *own* selection optimism. When the
    best-of-N-singles side pays more of it than the combination side, the
    corrected gap is the larger one — so "the uncorrected gap is larger" is a
    claim that has to be measured, and it is measured here rather than in the
    report's prose.
    """
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    row = corr.iloc[1]
    single, combo = corr.iloc[0], corr.iloc[1]
    assert row["gain_apparent"] == pytest.approx(
        combo["J_apparent"] - single["J_apparent"], abs=1e-9)
    assert row["correction_effect"] in {"widens", "narrows", "unchanged"}
    expected = ("widens" if row["gain_corrected"] > row["gain_apparent"]
                else "narrows" if row["gain_corrected"] < row["gain_apparent"]
                else "unchanged")
    assert row["correction_effect"] == expected


def test_a_widening_correction_is_labelled_widening():
    """The real cohort's case, isolated: the single side costs more to choose."""
    assert mp._correction_effect(0.119, 0.134) == "widens"
    assert mp._correction_effect(0.134, 0.119) == "narrows"
    assert mp._correction_effect(0.100, 0.100) == "unchanged"
    assert mp._correction_effect(np.nan, 0.1) == ""


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


def second_artifact() -> dict:
    """A model needing a third sign, so its complete cases are a smaller set."""
    return {
        "model_name": "Other model",
        "target": TARGET,
        "coefficients": {"const": -0.5, "sign_2": 1.1},
        "features": [
            {"name": "sign_2", "type": "binary",
             "encoding": {"sign_2": {"true": 1, "false": 0}}},
        ],
        "validation": {"metrics": [
            {"metric": "AUC", "apparent": 0.65, "optimism_corrected": 0.61},
        ]},
    }


def test_every_model_is_scored_on_one_patient_set():
    """The denominator error this section exists to prevent, in its own house.

    Each model's predictors have their own missingness, so left alone the
    models score on different patients and their AUCs sit in one column
    inviting subtraction. Every model is restricted to the patients *all* of
    them can score.
    """
    df = count_frame().copy()
    df.loc[df.index[:20], "sign_2"] = pd.NA      # costs the second model only
    df.loc[df.index[20:30], "sign_0"] = pd.NA    # costs the first model only
    correction = mp.selection_correction(df, COUNT_MARKERS[:2], TARGET, n_boot=40)
    table = mp.model_vs_single(
        df, {"tiny": tiny_artifact(), "other": second_artifact()},
        TARGET, correction,
    )
    assert table["n_scored"].nunique() == 1
    assert int(table["n_scored"].iloc[0]) == len(df) - 30
    assert set(table["denominator"]) == {mp.DENOM_SHARED}
    own = dict(zip(table["model"], table["n_complete_own"]))
    assert own["tiny"] == len(df) - 10
    assert own["other"] == len(df) - 20
    # The single-marker side is still on the whole marker shared set, and the
    # table says so rather than letting the page imply one number for both.
    assert set(table["n_best_single"]) == {len(df)}


def test_the_single_marker_side_is_offered_as_an_auc_not_only_a_youden_j():
    """A J of 0.14 beside an AUC of 0.75 reads as five times worse. It is 0.57.

    For a yes/no rule the two scales are locked together by AUC = (J + 1) / 2,
    so the comparable number exists and simply has to be printed.
    """
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    row = table.iloc[0]
    assert row["best_single_auc_corrected"] == pytest.approx(
        (row["best_single_J_corrected"] + 1) / 2, abs=1e-12)
    assert 0.0 <= row["best_single_auc_corrected"] <= 1.0


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


# --------------------------------------------------------------------------
# Reading views — no machine column names on the page
# --------------------------------------------------------------------------
def test_the_model_view_names_its_columns_for_a_reader():
    """A page that prints ``best_single_J_corrected`` asks the reader to know
    which of two columns is on a 0.5–1 scale. The headings say so instead.
    """
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    view = mp.model_reading_view(table)
    assert list(view.columns) == [
        "Model", "Patients scored", "Model AUC here (apparent)",
        "Model AUC, own patients (corrected)",
        "Model AUC, own patients (apparent)", "Best single sign",
        "Best single AUC (corrected)", "Best single Youden J (corrected)",
        "Note",
    ]
    assert not any("_" in str(c) for c in view.columns)
    assert view.iloc[0]["Model AUC, own patients (corrected)"] == "0.680"


def test_the_model_view_prints_an_em_dash_where_there_is_no_number():
    table = pd.DataFrame([{
        "model": "tiny", "n_scored": 0, "n_complete_own": 0, "denominator": "",
        "auc_shared_apparent": np.nan, "auc_artifact_corrected": np.nan,
        "auc_artifact_apparent": np.nan, "best_single_rule": "",
        "best_single_auc_corrected": np.nan, "best_single_J_corrected": np.nan,
        "note": "not scorable",
    }])
    view = mp.model_reading_view(table)
    assert view.iloc[0]["Model AUC here (apparent)"] == "—"
    assert view.iloc[0]["Patients scored"] == "—"


def test_the_stability_view_shows_rates_as_rates():
    """``value`` holds a draw count and three proportions; 0.4 means 40%."""
    draws = [count_frame(seed=s) for s in (1, 2, 3)]
    out = mp.imputation_stability(draws, COUNT_MARKERS, TARGET, n_boot=20)
    view = mp.stability_reading_view(out)
    assert list(view.columns) == ["What was checked", "Result", "Detail"]
    rows = dict(zip(view["What was checked"], view["Result"]))
    assert rows["Draws"] == "3"
    assert rows["Winning rule reproduced"].endswith("%")


def test_the_reading_views_are_empty_not_broken_with_nothing_to_show():
    assert mp.model_reading_view(pd.DataFrame()).empty
    assert mp.stability_reading_view(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------
def panel_accuracy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": TARGET, "predictor": f"sign_{i}", "kind": "binary"}
        for i in range(3)
    ])


def test_run_marker_panel_writes_every_table_and_figure(tmp_output):
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
    )
    written = sorted(p.name for p in (tmp_output / "panel" / "tables").glob("*.csv"))
    assert written == [
        "01_marker_panel.csv",
        "02_marker_panel_reading_view.csv",
        "03_shared_cohort.csv",
        "05_rule_menu.csv",
        "06_rule_reading_view.csv",
        "07_count_score.csv",
        "08_count_thresholds.csv",
        "09_selection_correction.csv",
        "10_model_vs_single.csv",
        "11_imputation_stability.csv",
        "12_count_headline.csv",
        "13_model_reading_view.csv",
        "14_stability_reading_view.csv",
    ]
    figures = sorted(p.name for p in (tmp_output / "panel" / "figures").glob("*.svg"))
    assert figures == ["count_score.svg", "lr_forest.svg", "rule_space.svg"]
    assert set(tables) >= {"01_marker_panel", "09_selection_correction"}


def test_run_marker_panel_excludes_what_it_is_told_to(tmp_output):
    mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_2"}, n_boot=40,
    )
    panel = pd.read_csv(tmp_output / "panel" / "tables" / "01_marker_panel.csv")
    assert "sign_2" not in set(panel["marker"])


def test_run_marker_panel_survives_a_single_usable_marker(tmp_output):
    """A combination question needs two markers. One must not crash the run."""
    df = count_frame()
    tables = mp.run_marker_panel(
        df, target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_1", "sign_2"}, n_boot=40,
    )
    assert not tables["01_marker_panel"].empty
    assert tables["05_rule_menu"].empty


def _spy_on_stability(monkeypatch) -> dict:
    seen: dict = {}
    original = mp.imputation_stability

    def spy(draws, markers, target, **kwargs):
        seen.update(kwargs)
        return original(draws, markers, target, **kwargs)

    monkeypatch.setattr(mp, "imputation_stability", spy)
    return seen


def test_the_per_draw_bootstrap_has_its_own_budget(tmp_output, monkeypatch):
    """``draw_n_boot`` governs the per-draw bootstrap; ``n_boot`` does not.

    The shared-set correction runs twice; the one inside imputation_stability
    runs once per MICE draw. Forwarding the shared-set budget multiplied a
    four-minute correction by twenty — the notebook cell would have run for
    about an hour and a half for a stability check whose answer is a
    reproduction rate.
    """
    seen = _spy_on_stability(monkeypatch)
    mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40, draw_n_boot=6,
        draws=[count_frame(seed=9)],
    )
    assert seen.get("n_boot") == 6


def test_the_per_draw_budget_defaults_low_even_when_n_boot_is_high(
    tmp_output, monkeypatch,
):
    """A caller who raises only ``n_boot`` must not silently buy 20× that."""
    seen = _spy_on_stability(monkeypatch)
    mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40, draws=[],
    )
    assert seen.get("n_boot") == mp.DEFAULT_DRAW_N_BOOT
    assert mp.DEFAULT_DRAW_N_BOOT < mp.DEFAULT_N_BOOT


# --------------------------------------------------------------------------
# Source links — the published model rows carry a link to the paper they came
# from, so a reader comparing our AUC against theirs can go and read theirs.
# --------------------------------------------------------------------------
PAPER = "https://pubmed.ncbi.nlm.nih.gov/30317276/"


def test_model_vs_single_carries_the_source_link_when_one_is_known():
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction,
                               links={"tiny": PAPER})
    assert table.iloc[0]["source_link"] == PAPER


def test_model_vs_single_leaves_the_link_blank_for_our_own_models():
    """The experimental variants are ours. Inventing a citation for them would
    be worse than leaving the cell empty."""
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction,
                               links={"someone_else": PAPER})
    assert table.iloc[0]["source_link"] == ""


def test_model_vs_single_has_a_link_column_even_with_no_links_argument():
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    assert "source_link" in table.columns
    assert table.iloc[0]["source_link"] == ""


def test_model_reading_view_puts_the_link_in_the_note():
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction,
                               links={"tiny": PAPER})
    view = mp.model_reading_view(table)
    assert PAPER in str(view.iloc[0]["Note"])


def test_model_reading_view_keeps_an_existing_note_beside_the_link():
    """A row can be both unscorable and cited; the note must not lose either."""
    table = pd.DataFrame([{
        "model": "tiny", "n_scored": 0, "n_complete_own": 0, "denominator": "",
        "auc_shared_apparent": float("nan"),
        "auc_artifact_corrected": float("nan"),
        "auc_artifact_apparent": float("nan"),
        "best_single_rule": "", "n_best_single": 0,
        "best_single_auc_corrected": float("nan"),
        "best_single_J_corrected": float("nan"),
        "note": "not scorable on this set — one outcome class only",
        "source_link": PAPER,
    }])
    note = str(mp.model_reading_view(table).iloc[0]["Note"])
    assert "one outcome class only" in note
    assert PAPER in note


# --------------------------------------------------------------------------
# Self-discovery: artifacts, draws and links
# --------------------------------------------------------------------------
def _write_artifact(art_dir, stem: str, *, target: str = TARGET) -> None:
    """Smallest artifact `load_model_artifact` will accept.

    ``type``/``encoding`` match what the real inferential stage writes for a
    binary feature (see ``model_calculator.py:509-511``) — not ``kind``,
    which is the diagnostic-accuracy-table column name, so scoring
    (``predict_from_artifact``) can actually read this artifact rather than
    only ``load_model_artifact`` parsing it.

    ``target`` defaults to the module ``TARGET`` but can be overridden to
    write an artifact belonging to a different outcome, for testing that
    ``load_panel_artifacts`` filters on it.
    """
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / f"{stem}.json").write_text(json.dumps({
        "model_name": stem,
        "target": target,
        "coefficients": {"const": -1.0, "sign_0": 0.5},
        "features": [{"name": "sign_0", "type": "binary",
                      "encoding": {"sign_0": {"true": 1, "false": 0}}}],
    }), encoding="utf-8")


def test_panel_key_maps_an_artifact_stem_and_a_variant_id_onto_one_key():
    """An artifact filename and the variant id that produced it must agree."""
    assert mp.panel_key("high_grade_yao_et_al_2022_model", TARGET) == "yao_et_al_2022"
    assert mp.panel_key("yao_et_al_2022", TARGET) == "yao_et_al_2022"
    assert mp.panel_key("high_grade_experimental_model_1_model", TARGET) == "experimental_model_1"
    assert mp.panel_key("experimental_model_1", TARGET) == "experimental_model_1"


def test_panel_key_strips_model_only_as_a_suffix():
    """The regression this replaced: `.replace()` stripped every occurrence.

    `experimental_model_1` used to collapse to `experimental_1`, losing part
    of the real id. It matched anyway only because the artifact stem was
    mangled identically — an id with `_model` anywhere else would not have
    been so lucky.
    """
    assert mp.panel_key("high_grade_model_free_zone_model", TARGET) == "model_free_zone"
    assert mp.panel_key("model_free_zone", TARGET) == "model_free_zone"


def test_panel_key_falls_back_to_the_target_for_a_single_model_artifact():
    """``{target}_model.json`` has no model id — only the target — to strip out.

    ``_artifact_model_id`` returns ``""`` for that shape, which would give a
    reading-view row a blank ``Model`` cell. ``panel_key`` has to fall back to
    the target so the artifact stem and the variant id (also just the target,
    for a single-model cohort) still agree on one key.
    """
    assert mp.panel_key(f"{TARGET}_model", TARGET) == TARGET
    assert mp.panel_key(TARGET, TARGET) == TARGET


def test_load_panel_artifacts_reads_the_model_artifact_directory(tmp_output):
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    artifacts = mp.load_panel_artifacts(tmp_output, TARGET)
    assert set(artifacts) == {"yao_et_al_2022"}
    assert artifacts["yao_et_al_2022"]["coefficients"]["const"] == -1.0


def test_load_panel_artifacts_is_empty_when_nothing_has_been_fitted(tmp_output):
    """A panel run before the inferential stage is not an error."""
    assert mp.load_panel_artifacts(tmp_output, TARGET) == {}


def test_load_panel_artifacts_skips_a_model_fitted_for_another_outcome(tmp_output):
    """A foreign model must not enter this target's panel.

    ``model_vs_single`` intersects every loaded model's complete-case mask
    into ONE shared denominator, so a ``brain_invasion`` artifact sitting in
    the same ``model_artifacts/`` directory would silently shrink
    ``n_scored`` for every row this panel publishes — even though it has
    nothing to do with ``high_grade``. Only the artifact whose own
    ``target`` field matches should come back.
    """
    art_dir = tmp_output / "inferential" / "model_artifacts"
    _write_artifact(art_dir, f"{TARGET}_yao_et_al_2022_model", target=TARGET)
    _write_artifact(art_dir, "brain_invasion_other_model", target="brain_invasion")

    artifacts = mp.load_panel_artifacts(tmp_output, TARGET)
    assert set(artifacts) == {"yao_et_al_2022"}


def test_model_links_from_variants_keeps_papers_and_drops_our_own():
    """Experimental variants carry an empty link and must not get a citation."""
    variants = [
        ("yao_et_al_2022", "Yao et al. 2022", "https://example.org/yao",
         TARGET, ["sign_0"]),
        ("experimental_model_1", "model 1", "", TARGET, ["sign_1"]),
    ]
    links = mp.model_links_from_variants(variants, TARGET)
    assert links == {"yao_et_al_2022": "https://example.org/yao"}


def test_panel_draws_is_empty_when_there_is_no_mice_directory_at_all(tmp_output):
    """Simple imputation never writes ``missingness/mice/``. That is one
    missing table, not an error — the cohort just was not run through MICE.
    """
    assert mp._panel_draws(tmp_output) == []


def test_panel_draws_raises_on_a_mice_directory_without_a_manifest(tmp_output):
    """A half-written MICE run must fail loudly, not vanish into an empty table.

    Guarding on the directory rather than the manifest inside it means a run
    that crashed after writing its parquet files (or crashed before writing
    anything) but left the directory behind surfaces
    ``load_imputed_frames``'s own ``FileNotFoundError`` instead of being
    read as "no MICE was done here" and quietly losing the stability check.
    """
    (tmp_output / "missingness" / "mice").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        mp._panel_draws(tmp_output)


def test_run_marker_panel_finds_its_own_artifacts_and_links(tmp_output):
    """Passing `variants` replaces the notebook's two dict comprehensions."""
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
        variants=[("yao_et_al_2022", "Yao et al. 2022",
                   "https://example.org/yao", TARGET, ["sign_0"])],
    )
    models = tables["10_model_vs_single"]
    assert set(models["model"]) == {"yao_et_al_2022"}
    assert models["source_link"].iloc[0] == "https://example.org/yao"


def test_run_marker_panel_still_honours_artifacts_passed_in_explicitly(tmp_output):
    """An explicit empty dict means empty, not 'go and look'."""
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40, artifacts={},
    )
    assert tables["10_model_vs_single"].empty


def test_run_marker_panel_labels_an_experimental_variant_for_the_report(tmp_output):
    """The whole visible payoff of this branch: the model comparison table
    reads ``Experimental model 1``, not ``Experimental 1``.

    The key that ties the artifact to the variant (``experimental_model_1``)
    is a machine id and stays that way in ``10_model_vs_single``. The reading
    view is where it becomes the words a reader — or the research report —
    actually sees, so that is the assertion that matters here.
    """
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_experimental_model_1_model")
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
        variants=[("experimental_model_1", "Experimental model 1", "",
                   TARGET, ["sign_0"])],
    )
    assert "experimental_model_1" in set(tables["10_model_vs_single"]["model"])
    assert "Experimental model 1" in set(tables["13_model_reading_view"]["Model"])


def test_run_marker_panel_survives_a_cohort_with_no_mice_draws(tmp_output):
    """Simple imputation leaves no MICE directory. That is not a crash — and
    the stability table says exactly why it is empty rather than just
    happening to have zero rows for some other reason.

    ``imputation_stability([], ...)`` returns one row — item ``"Draws"``,
    value ``0`` — not an empty frame, so the meaningful check is on that row,
    not on frame emptiness.
    """
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
    )
    stability = tables["11_imputation_stability"]
    assert len(stability) == 1
    row = stability.iloc[0]
    assert row["item"] == "Draws"
    assert row["value"] == 0
