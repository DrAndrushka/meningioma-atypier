"""Step 3 — the AUC is right, its interval is right, and the ranking is honest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import measurements as ms
import separation as sep
from intervals import proportion_row, wilson_ci


# --- Wilson ----------------------------------------------------------------
def test_wilson_never_leaves_the_unit_interval_at_the_ends():
    lo, hi = wilson_ci(20, 20)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == 1.0 or hi < 1.0000001


def test_wilson_matches_the_published_value():
    """Wilson 95% CI for 9/10 is 0.5958-0.9821 (Newcombe 1998, Table 1)."""
    lo, hi = wilson_ci(9, 10)
    assert lo == pytest.approx(0.5958, abs=5e-4)
    assert hi == pytest.approx(0.9821, abs=5e-4)


def test_wilson_on_an_empty_denominator_is_blank_not_an_error():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_proportion_row_carries_the_counts_it_came_from():
    row = proportion_row(3, 10)
    assert row["n"] == 10 and row["events"] == 3
    assert row["estimate"] == pytest.approx(0.3)
    assert row["lo"] < 0.3 < row["hi"]


# --- the AUC itself --------------------------------------------------------
def _scores(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = rng.normal(y * 0.8, 1.0)
    return y, score


def test_auc_matches_sklearn():
    y, score = _scores()
    assert sep.fast_delong(y, score).auc == pytest.approx(
        roc_auc_score(y, score), abs=1e-12)


def test_auc_matches_sklearn_with_heavy_ties():
    """A third of this cohort shares the value zero — ties must not drift."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    score = np.where(rng.random(300) < 0.4, 0.0, rng.integers(0, 5, 300))
    assert sep.fast_delong(y, score).auc == pytest.approx(
        roc_auc_score(y, score), abs=1e-12)


def test_a_constant_score_is_a_coin_flip():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert sep.fast_delong(y, np.ones(6)).auc == pytest.approx(0.5)


def test_direction_is_applied_not_discovered():
    """Declaring the wrong direction must produce an AUC below 0.5, not above."""
    y, score = _scores()
    higher = sep.auc_with_ci(y, score, "higher")["auc"]
    lower = sep.auc_with_ci(y, score, "lower")["auc"]
    assert higher > 0.5 > lower
    assert higher + lower == pytest.approx(1.0)


# --- the variance and the interval ----------------------------------------
def test_delong_variance_agrees_with_a_bootstrap():
    y, score = _scores(seed=7, n=400)
    result = sep.fast_delong(y, score)
    rng = np.random.default_rng(11)
    draws = []
    for _ in range(600):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(y):
            draws.append(roc_auc_score(y[idx], score[idx]))
    assert np.sqrt(result.var) == pytest.approx(np.std(draws, ddof=1), rel=0.15)


def test_interval_brackets_the_estimate_and_stays_inside_zero_one():
    y, score = _scores(seed=5, n=60)
    out = sep.auc_with_ci(y, score, "higher")
    assert 0.0 < out["auc_lo"] < out["auc"] < out["auc_hi"] < 1.0


def test_a_tiny_sample_does_not_produce_an_impossible_upper_bound():
    """The reason the interval is built on the log-odds scale."""
    y = np.array([0, 0, 0, 1, 1, 1])
    out = sep.auc_with_ci(y, np.array([1.0, 2, 3, 8, 9, 10]), "higher")
    assert out["auc"] == pytest.approx(1.0)
    assert out["auc_hi"] <= 1.0


def test_counts_are_reported_alongside_the_estimate():
    y, score = _scores(n=100)
    out = sep.auc_with_ci(y, score, "higher")
    assert out["n"] == 100
    assert out["n_high"] + out["n_low"] == 100


# --- refusals --------------------------------------------------------------
def test_one_grade_only_is_refused():
    with pytest.raises(sep.AucError, match="Need both grades"):
        sep.fast_delong(np.ones(5, dtype=int), np.arange(5.0))


def test_missing_scores_are_refused_rather_than_dropped():
    """Silently dropping would change the denominator without saying so."""
    with pytest.raises(sep.AucError, match="missing values"):
        sep.fast_delong(np.array([0, 1, 0]), np.array([1.0, np.nan, 3.0]))


def test_mismatched_lengths_are_refused():
    with pytest.raises(sep.AucError, match="different lengths"):
        sep.fast_delong(np.array([0, 1]), np.array([1.0, 2.0, 3.0]))


# --- the table -------------------------------------------------------------
def _cohort(n: int = 240, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    edema = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.1 * y, 0.15),
        "tumor_volume": rng.gamma(2, 8, n) + 3 * y,
        "edema_volume_cm3": edema,
        "edema_index": edema / (rng.gamma(2, 8, n) + 1),
        "max_diameter_cm": rng.normal(3.8 + 0.3 * y, 1.2),
    })


def test_table_has_a_row_per_measurement_per_stratum():
    table = sep.separation_table(_cohort())
    assert len(table) == 7           # 5 measurements + 2 'present' strata
    assert (table["stratum"] == ms.STRATUM_PRESENT).sum() == 2


def test_table_is_ranked_best_first():
    aucs = sep.separation_table(_cohort())["auc"].dropna().to_numpy()
    assert (np.diff(aucs) <= 1e-12).all()


def test_the_present_stratum_has_a_smaller_denominator():
    table = sep.separation_table(_cohort()).set_index("measurement")
    assert (table.loc["Edema volume (where present)", "n"]
            < table.loc["Edema volume", "n"])


def test_a_stratum_with_one_grade_only_is_blank_not_fatal():
    df = _cohort()
    df.loc[df["edema_volume_cm3"] > 0, "high_grade"] = 1
    table = sep.separation_table(df).set_index("measurement")
    assert np.isnan(table.loc["Edema volume (where present)", "auc"])
    assert "Need both grades" in table.loc[
        "Edema volume (where present)", "note"]


# --- presence vs absence ---------------------------------------------------
def test_presence_effect_reports_both_arms_with_intervals():
    out = sep.presence_effect(_cohort(), ms.MEASUREMENTS_BY_COL["edema_volume_cm3"])
    assert out["n_absent"] + out["n_present"] > 0
    assert out["rate_absent_lo"] <= out["rate_absent"] <= out["rate_absent_hi"]
    assert out["rate_difference"] == pytest.approx(
        out["rate_present"] - out["rate_absent"])


def test_presence_effect_refuses_a_measurement_with_no_zero_pile():
    with pytest.raises(sep.AucError, match="not declared zero-inflated"):
        sep.presence_effect(_cohort(), ms.MEASUREMENTS_BY_COL["adc_value"])


def test_presence_table_covers_only_the_zero_inflated_pair():
    assert list(sep.presence_table(_cohort())["measurement"]) == [
        "Edema volume", "Edema index"]


# --- the summary line ------------------------------------------------------
def test_describe_names_the_best_and_counts_what_clears_a_coin_flip():
    line = sep.describe_separation(sep.separation_table(_cohort()))
    assert line.startswith("Best separation:")
    assert "Clearing 0.50" in line or "None is shown" in line


def test_describe_says_plainly_when_nothing_separates():
    rng = np.random.default_rng(2)
    df = _cohort()
    for m in ms.MEASUREMENTS:          # pure noise, unrelated to the outcome
        df[m.col] = rng.normal(size=len(df))
    line = sep.describe_separation(sep.separation_table(df))
    assert "none is shown to separate the grades" in line
