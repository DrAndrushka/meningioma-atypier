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


def test_shared_cohort_keeps_only_fully_measured_patients():
    df = tiny_frame()
    cps = [cb.CutPoint(A, 5.0), cb.CutPoint(B, 5.0)]
    shared = cb.shared_cohort(df, cps, TARGET)
    # Rows 4, 5 and 6 each miss one of the two measurements.
    assert list(shared.index) == [0, 1, 2, 3, 7]
    assert shared[["a", "b"]].notna().all().all()


def test_shared_cohort_gives_every_rule_one_denominator():
    df = two_signal_frame(n=300, seed=9).copy()
    df.loc[:29, "a"] = np.nan          # missing in a, present in b
    df.loc[30:59, "b"] = np.nan        # and the other way round
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")

    full = cb.full_rule_menu(df, cps, TARGET)
    assert full["n_used"].nunique() > 1     # the problem being fixed

    shared = cb.shared_cohort(df, cps, TARGET)
    menu = cb.full_rule_menu(shared, cps, TARGET)
    assert menu["n_used"].nunique() == 1
    assert int(menu["n_used"].iloc[0]) == len(shared)


def test_shared_cohort_drops_a_missing_outcome():
    df = tiny_frame()
    df.loc[0, TARGET] = pd.NA
    cps = [cb.CutPoint(A, 5.0), cb.CutPoint(B, 5.0)]
    assert 0 not in cb.shared_cohort(df, cps, TARGET).index


def test_shared_cohort_does_not_move_the_cutpoints():
    """Only the patient set changes — otherwise the comparison is not like-for-like."""
    df = two_signal_frame(n=300, seed=13).copy()
    df.loc[:19, "a"] = np.nan
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    shared = cb.shared_cohort(df, cps, TARGET)
    menu = cb.full_rule_menu(shared, cps, TARGET)
    for cp in cps:
        assert any(f"{cp.cutoff:.3g}" in label for label in menu["rule_label"])


def test_kinds_restricts_the_menu_the_winner_is_chosen_from():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    singles = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=30, seed=4,
                                     kinds=("single",))
    everything = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=30, seed=4)

    labels = set(cb.single_rule_table(df, cps, TARGET)["rule_label"])
    assert singles["best_rule"] in labels
    # The unrestricted menu contains the singles, so it can never do worse.
    assert everything["J_apparent"] >= singles["J_apparent"]


def test_restricted_selection_still_carries_its_own_optimism():
    """The point of P1.1: 'best of four singles' is a selection too."""
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
        TARGET: pd.array(rng.binomial(1, 0.3, n).astype(bool), dtype="boolean"),
    })
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=40, seed=3, kinds=("single",))
    assert out["optimism"] > 0
    assert out["J_corrected"] < out["J_apparent"]


def test_unknown_kind_degrades_rather_than_raising():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=5, seed=1, kinds=("nope",))
    assert out["best_rule"] == ""
    assert np.isnan(out["optimism"])


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


# --------------------------------------------------------------------------
# Characterization: these numbers came out of the pandas implementation on
# 2026-08-04, before the numpy scorer replaced it. They are not a claim about
# what is statistically right — they are a tripwire. If the scorer is not
# bit-for-bit equivalent, one of them moves.
# --------------------------------------------------------------------------
GOLDEN_BOOT = {
    None: {
        "optimism": 0.0046525807268937265,
        "n_bootstrap": 50,
        "best_rule": "≥ 2 of 3 criteria",
        "J_apparent": 0.6254960317460316,
        "J_corrected": 0.620843451019138,
        "winner_stability": 1.0,
    },
    ("single",): {
        "optimism": 0.038835126255743285,
        "n_bootstrap": 50,
        "best_rule": "Metric B ≥ 0.397",
        "J_apparent": 0.4618055555555556,
        "J_corrected": 0.4229704292998123,
        "winner_stability": 0.58,
    },
    ("and", "or", "count"): {
        "optimism": 0.0046525807268937265,
        "n_bootstrap": 50,
        "best_rule": "≥ 2 of 3 criteria",
        "J_apparent": 0.6254960317460316,
        "J_corrected": 0.620843451019138,
        "winner_stability": 1.0,
    },
}


@pytest.mark.parametrize("kinds", list(GOLDEN_BOOT))
def test_bootstrap_reproduces_the_recorded_numbers(kinds):
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    assert [cp.label for cp in cps] == [
        "Metric A ≥ 0.933", "Metric B ≥ 0.397", "Metric C ≤ -0.498",
    ]
    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=50, seed=20260801,
                                 kinds=kinds)
    want = GOLDEN_BOOT[kinds]
    assert out["best_rule"] == want["best_rule"]
    assert out["n_bootstrap"] == want["n_bootstrap"]
    for key in ("optimism", "J_apparent", "J_corrected", "winner_stability"):
        assert out[key] == pytest.approx(want[key], rel=0, abs=1e-12)


def test_bootstrap_matches_the_menu_on_a_frame_with_missing_outcomes():
    """n is len(df): rows with no outcome are resampled, then dropped by the score."""
    rng = np.random.default_rng(21)
    n = 300
    y = rng.binomial(1, 0.35, n).astype(bool)
    df = pd.DataFrame({
        "a": rng.normal(size=n) + y * 1.1,
        "b": rng.normal(size=n) + y * 0.9,
        TARGET: pd.array(y, dtype="boolean"),
    })
    df.loc[rng.random(n) < 0.2, "a"] = np.nan
    df.loc[rng.random(n) < 0.1, TARGET] = pd.NA
    cps = cb.cutpoints_for_rule(df.dropna(subset=[TARGET]), [A, B], TARGET, "youden")

    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=30, seed=5)
    menu = cb.full_rule_menu(df, cps, TARGET)
    apparent = menu.loc[menu["youden_J"].idxmax()]
    assert out["best_rule"] == apparent["rule_label"]
    assert out["J_apparent"] == pytest.approx(float(apparent["youden_J"]))


def test_bootstrap_is_fast_enough_to_run_in_a_notebook():
    """Twelve cut-points, 300 resamples. The pandas loop took about a minute."""
    import time

    rng = np.random.default_rng(3)
    n = 300
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {f"m{i}": rng.normal(size=n) + y * 0.6 for i in range(12)}
    df = pd.DataFrame({**cols, TARGET: pd.array(y, dtype="boolean")})
    metrics = [Metric(f"m{i}", f"Metric {i}", "u", "higher") for i in range(12)]
    cps = cb.cutpoints_for_rule(df, metrics, TARGET, "youden")

    start = time.perf_counter()
    cb.bootstrap_best_rule(df, cps, TARGET, n_boot=300, seed=1)
    assert time.perf_counter() - start < 10.0


def test_bootstrap_best_rules_equals_running_each_side_separately():
    """One shared resample loop, two answers, identical to two separate runs."""
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    kw = dict(n_boot=60, seed=20260801)

    together = cb.bootstrap_best_rules(
        df, cps, TARGET,
        sides={"single": ("single",), "combo": ("and", "or", "count")}, **kw)
    apart = {
        "single": cb.bootstrap_best_rule(df, cps, TARGET, kinds=("single",), **kw),
        "combo": cb.bootstrap_best_rule(df, cps, TARGET,
                                        kinds=("and", "or", "count"), **kw),
    }

    assert set(together) == set(apart)
    for name in apart:
        assert together[name]["best_rule"] == apart[name]["best_rule"]
        assert together[name]["n_bootstrap"] == apart[name]["n_bootstrap"]
        for key in ("optimism", "J_apparent", "J_corrected", "winner_stability"):
            assert together[name][key] == pytest.approx(apart[name][key],
                                                        rel=0, abs=1e-12)


def test_bootstrap_best_rules_degrades_on_an_unknown_kind():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    out = cb.bootstrap_best_rules(df, cps, TARGET,
                                  sides={"nope": ("nope",)}, n_boot=5, seed=1)
    assert out["nope"]["best_rule"] == ""
    assert np.isnan(out["nope"]["optimism"])
