"""Multi-cut rules: Kleene logic, AND/OR scoring, count score, benchmarks."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import combinations as cb
from thresholds import Metric

TARGET = "high_grade"
A = Metric("a", "Metric A", "u", "higher")
B = Metric("b", "Metric B", "u", "higher")
C = Metric("c", "Metric C", "u", "lower")


def tiny_frame() -> pd.DataFrame:
    """Hand-checkable: eight patients, two flags, one missing value each way."""
    return pd.DataFrame({
        "a": pd.array([10.0, 10.0, 0.0, 0.0, 10.0, 0.0, None, 10.0], dtype="Float64"),
        "b": pd.array([10.0, 0.0, 10.0, 0.0, None, None, 10.0, 10.0], dtype="Float64"),
        TARGET: pd.array([True, True, False, False, True, False, True, True],
                         dtype="boolean"),
    })


def two_signal_frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n) + y * 1.2,
        "b": rng.normal(0, 1, n) + y * 1.0,
        "c": -(rng.normal(0, 1, n) + y * 0.8),
        TARGET: pd.array(y, dtype="boolean"),
    })


CPS = [cb.CutPoint(A, 5.0), cb.CutPoint(B, 5.0)]


# --------------------------------------------------------------------------
# CutPoint
# --------------------------------------------------------------------------
def test_cutpoint_label_carries_the_operator():
    assert cb.CutPoint(A, 5.0).label == "Metric A ≥ 5"
    assert cb.CutPoint(C, 0.7).label == "Metric C ≤ 0.7"


def test_cutpoints_for_rule_returns_one_per_metric():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    assert [cp.col for cp in cps] == ["a", "b", "c"]
    assert all(np.isfinite(cp.cutoff) for cp in cps)


def test_cutpoints_from_literature_skips_unknown_columns():
    cps = cb.cutpoints_from_literature(
        [A], {"a": [(3.0, "Author 2020")], "zzz": [(1.0, "Other 2019")]},
    )
    assert len(cps) == 1
    assert cps[0].source == "Author 2020"
    assert cps[0].rule == "literature"


# --------------------------------------------------------------------------
# Kleene logic — the part that decides who gets scored
# --------------------------------------------------------------------------
def test_or_resolves_when_one_flag_is_true_despite_a_missing_partner():
    df = tiny_frame()
    flags = cb.flag_frame(df, CPS)
    combined = cb.combine_flags(flags, ["a", "b"], cb.OR)
    # Row 4: a is above the cut, b is missing → OR is already settled.
    assert combined.iloc[4] is True or combined.iloc[4] == True  # noqa: E712
    # Row 5: a is below, b is missing → still unknown.
    assert pd.isna(combined.iloc[5])


def test_and_resolves_when_one_flag_is_false_despite_a_missing_partner():
    df = tiny_frame()
    flags = cb.flag_frame(df, CPS)
    combined = cb.combine_flags(flags, ["a", "b"], cb.AND)
    assert combined.iloc[5] == False  # noqa: E712  a is False → AND settled
    assert pd.isna(combined.iloc[4])  # a True, b missing → unknown


def test_and_is_never_more_sensitive_than_or():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    menu = cb.pair_rule_table(df, cps, TARGET)
    and_row = menu[menu["kind"] == "and"].iloc[0]
    or_row = menu[menu["kind"] == "or"].iloc[0]
    assert and_row["sensitivity"] <= or_row["sensitivity"] + 1e-12
    assert and_row["specificity"] >= or_row["specificity"] - 1e-12


def test_combine_flags_rejects_an_unknown_logic():
    df = tiny_frame()
    flags = cb.flag_frame(df, CPS)
    with pytest.raises(ValueError, match="logic"):
        cb.combine_flags(flags, ["a", "b"], "XOR")


# --------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------
def test_pair_table_covers_every_pair_and_both_logics():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    menu = cb.pair_rule_table(df, cps, TARGET)
    assert len(menu) == 3 * 2  # three pairs × {AND, OR}
    assert set(menu["kind"]) == {"and", "or"}
    assert (menu["n_criteria"] == 2).all()


def test_triples_are_available_when_asked_for():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    menu = cb.pair_rule_table(df, cps, TARGET, max_size=3)
    assert (menu["n_criteria"] == 3).sum() == 2  # one triple × two logics


def test_single_rule_table_matches_a_hand_count():
    df = tiny_frame()
    out = cb.single_rule_table(df, CPS, TARGET)
    row = out[out["rule_label"] == "Metric A ≥ 5"].iloc[0]
    # a ≥ 5 for rows 0,1,4,7 — all high grade; row 6 has a missing.
    assert row["TP"] == 4
    assert row["FP"] == 0
    assert row["n_used"] == 7  # row 6 dropped: flag unknown


def test_full_menu_contains_all_three_families():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    menu = cb.full_rule_menu(df, cps, TARGET)
    assert set(menu["kind"]) == {"single", "and", "or", "count"}
    assert menu.columns[0] == "rule_label"


def test_youden_j_is_consistent_within_the_menu():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    menu = cb.full_rule_menu(df, cps, TARGET)
    expected = menu["sensitivity"] + menu["specificity"] - 1.0
    assert np.allclose(menu["youden_J"], expected, equal_nan=True)


# --------------------------------------------------------------------------
# Count score
# --------------------------------------------------------------------------
def test_count_table_has_a_row_per_possible_count():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    counts = cb.count_score_table(df, cps, TARGET)
    assert list(counts["n_criteria_met"]) == [0, 1, 2, 3]
    assert counts["n"].sum() == counts.attrs["n_scored"]


def test_count_score_rises_with_the_number_of_criteria():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    counts = cb.count_score_table(df, cps, TARGET)
    usable = counts[counts["n"] >= 10]
    assert usable["risk"].iloc[0] < usable["risk"].iloc[-1]


def test_complete_only_drops_patients_with_a_missing_flag():
    df = tiny_frame()
    strict = cb.count_score_table(df, CPS, TARGET, complete_only=True)
    loose = cb.count_score_table(df, CPS, TARGET, complete_only=False)
    assert strict.attrs["n_scored"] == 5  # rows 4, 5, 6 have a missing flag
    assert loose.attrs["n_scored"] == 8


def test_count_threshold_rules_are_monotone_in_sensitivity():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    rules = cb.count_threshold_table(df, cps, TARGET)
    assert list(rules["rule_label"]) == [
        "≥ 1 of 3 criteria", "≥ 2 of 3 criteria", "≥ 3 of 3 criteria",
    ]
    assert rules["sensitivity"].is_monotonic_decreasing
    assert rules["specificity"].is_monotonic_increasing


def test_count_of_one_equals_the_or_rule():
    """'≥1 of 2' and 'A OR B' are the same test, so they must score identically."""
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    counts = cb.count_threshold_table(df, cps, TARGET)
    pairs = cb.pair_rule_table(df, cps, TARGET)
    or_row = pairs[pairs["kind"] == "or"].iloc[0]
    one_row = counts.iloc[0]
    assert one_row["sensitivity"] == pytest.approx(or_row["sensitivity"])
    assert one_row["specificity"] == pytest.approx(or_row["specificity"])


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------
def test_continuous_benchmark_beats_chance_on_signal():
    df = two_signal_frame()
    out = cb.continuous_model_benchmark(df, [A, B, C], TARGET, n_boot=40)
    assert out["AUC_apparent"] > 0.6
    assert out["AUC_corrected"] <= out["AUC_apparent"]
    assert out["n_used"] == len(df)


def test_continuous_benchmark_returns_blanks_on_a_tiny_frame():
    df = two_signal_frame(n=20)
    out = cb.continuous_model_benchmark(df, [A, B], TARGET, n_boot=5)
    assert np.isnan(out["AUC_apparent"])


def test_best_rule_optimism_is_positive_on_noise():
    """Picking the winner of a dozen rules on noise always looks good here."""
    rng = np.random.default_rng(8)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        TARGET: pd.array(rng.binomial(1, 0.3, n).astype(bool), dtype="boolean"),
    })
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=40, seed=3)
    assert out["optimism"] > 0
    assert out["J_corrected"] < out["J_apparent"]
    assert 0.0 <= out["winner_stability"] <= 1.0


def test_best_rule_is_reproducible():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    kw = dict(n_boot=25, seed=17)
    a = cb.bootstrap_best_rule(df, cps, TARGET, **kw)
    b = cb.bootstrap_best_rule(df, cps, TARGET, **kw)
    assert a["best_rule"] == b["best_rule"]
    assert a["optimism"] == pytest.approx(b["optimism"])


# --------------------------------------------------------------------------
# Views and figures
# --------------------------------------------------------------------------
def test_reading_view_is_sorted_by_j():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    view = cb.combination_reading_view(cb.full_rule_menu(df, cps, TARGET))
    assert view["J"].is_monotonic_decreasing
    assert "Sens (95% CI)" in view.columns


def test_reading_view_respects_top():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    view = cb.combination_reading_view(cb.full_rule_menu(df, cps, TARGET), top=3)
    assert len(view) == 3


def test_combination_figure_builds_with_and_without_a_benchmark():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    menu = cb.full_rule_menu(df, cps, TARGET)
    plt.close(cb.combination_figure(menu))
    bench = cb.continuous_model_benchmark(df, [A, B], TARGET, n_boot=20)
    fig = cb.combination_figure(menu, benchmark=bench)
    assert len(fig.get_axes()) == 2
    plt.close(fig)


def test_count_score_figure_builds():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    counts = cb.count_score_table(df, cps, TARGET)
    fig = cb.count_score_figure(counts, cutpoints=cps, prevalence=0.3)
    assert fig.get_axes()
    plt.close(fig)
