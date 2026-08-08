"""Calibration slope/intercept, Brier, net benefit and decision curves.

The tests that matter here are the ones with a hand-checkable answer: net
benefit has a closed form, so it is checked against arithmetic rather than
against itself, and calibration is checked on data where the right answer is
known by construction (a perfectly calibrated predictor must give slope 1 and
intercept 0; a deliberately over-extreme one must give slope below 1).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import calibration as cal
from thresholds import Metric

TARGET = "high_grade"
A = Metric("a", "Metric A", "u", "higher")
B = Metric("b", "Metric B", "u", "higher", log_x=True)


def calibrated_frame(n: int = 4000, seed: int = 3):
    """Outcomes drawn from a known probability — calibration is 1.0 by construction."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.9, n)
    y = rng.binomial(1, p)
    return y, p


def signal_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n) + y * 1.1,
        "b": np.abs(rng.normal(5, 2, n) + y * 3.0),
        TARGET: pd.array(y, dtype="boolean"),
    })


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def test_a_perfectly_calibrated_predictor_gives_slope_one_intercept_zero():
    y, p = calibrated_frame()
    slope, intercept = cal.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.12)
    assert intercept == pytest.approx(0.0, abs=0.12)


def test_over_extreme_predictions_give_a_slope_below_one():
    """Push the log-odds away from zero: the slope must fall."""
    y, p = calibrated_frame()
    stretched = 1.0 / (1.0 + np.exp(-2.0 * cal.logit(p)))
    slope, _ = cal.calibration_slope_intercept(y, stretched)
    assert slope < 0.75


def test_systematically_inflated_predictions_give_a_negative_intercept():
    y, p = calibrated_frame()
    inflated = 1.0 / (1.0 + np.exp(-(cal.logit(p) + 1.0)))
    _, intercept = cal.calibration_slope_intercept(y, inflated)
    assert intercept < -0.5


def test_logit_survives_predictions_of_exactly_zero_and_one():
    assert np.all(np.isfinite(cal.logit(np.array([0.0, 0.5, 1.0]))))


def test_calibration_degrades_on_a_single_class():
    slope, intercept = cal.calibration_slope_intercept(np.ones(50), np.full(50, 0.7))
    assert np.isnan(slope) and np.isnan(intercept)


def test_calibration_bins_are_equal_count_and_carry_intervals():
    y, p = calibrated_frame(n=500)
    bins = cal.calibration_bins(y, p, n_bins=10)
    assert len(bins) == 10
    assert bins["n"].min() >= 10
    assert (bins["lo"] <= bins["observed"]).all()
    assert (bins["observed"] <= bins["hi"]).all()
    assert int(bins["n"].sum()) == 500


def test_brier_is_zero_for_a_perfect_prediction_and_one_for_the_worst():
    y = np.array([1, 0, 1, 0])
    assert cal.brier_score(y, y.astype(float)) == 0.0
    assert cal.brier_score(y, 1.0 - y) == 1.0


# --------------------------------------------------------------------------
# The uncut model
# --------------------------------------------------------------------------
def test_uncut_design_applies_log1p_only_to_log_metrics():
    df = signal_frame()
    X, y, cols = cal.uncut_design(df, [A, B], TARGET)
    assert cols == ["a", "b"]
    assert X.shape == (len(df), 3)               # constant + two predictors
    assert X[:, 1] == pytest.approx(df["a"].to_numpy())
    assert X[:, 2] == pytest.approx(np.log1p(df["b"].to_numpy()))


def test_apparent_slope_is_one_by_construction():
    """A model scored on its own fitted values cannot be miscalibrated."""
    df = signal_frame()
    out = cal.uncut_model_calibration(df, [A, B], TARGET, n_boot=40, seed=5)
    assert out["slope_apparent"] == pytest.approx(1.0, abs=1e-6)
    assert out["intercept_apparent"] == pytest.approx(0.0, abs=1e-6)
    assert out["n_bootstrap"] > 0


def test_correction_bites_when_the_model_is_overfitted():
    """Eight pure-noise predictors on 120 patients: the slope must fall well below 1.

    Two well-behaved predictors on 400 patients barely overfit at all, so the
    correction there is a rounding error and asserting a strict drop would be
    a flaky test rather than a real one.
    """
    rng = np.random.default_rng(19)
    n, k = 120, 8
    y = rng.binomial(1, 0.3, n).astype(bool)
    noise = {f"z{i}": rng.normal(size=n) for i in range(k)}
    df = pd.DataFrame({**noise, TARGET: pd.array(y, dtype="boolean")})
    metrics = [Metric(f"z{i}", f"Z{i}", "u", "higher") for i in range(k)]

    out = cal.uncut_model_calibration(df, metrics, TARGET, n_boot=80, seed=4)
    assert out["slope_corrected"] < 0.8
    assert out["brier_corrected"] > out["brier_apparent"]


def test_uncut_calibration_degrades_on_a_tiny_cohort():
    df = signal_frame(n=20)
    out = cal.uncut_model_calibration(df, [A, B], TARGET, n_boot=5, seed=1)
    assert np.isnan(out["slope_corrected"])
    assert out["n_bootstrap"] == 0


# --------------------------------------------------------------------------
# Net benefit
# --------------------------------------------------------------------------
def test_net_benefit_matches_the_formula_by_hand():
    # 10 patients, 4 events. Flag 5: 3 true positives, 2 false.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    flagged = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    t = 0.2
    expected = 3 / 10 - (2 / 10) * (t / (1 - t))
    assert cal.net_benefit(y, flagged, t) == pytest.approx(expected)


def test_treat_all_net_benefit_equals_prevalence_at_a_zero_threshold():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    nb = cal.net_benefit(y, np.ones_like(y, dtype=bool), 0.001)
    assert nb == pytest.approx(0.2, abs=0.001)


def test_treat_all_is_worthless_above_a_threshold_it_cannot_justify():
    y = np.array([1, 0, 0, 0])          # 25% prevalence
    assert cal.net_benefit(y, np.ones_like(y, dtype=bool), 0.5) < 0


def test_net_benefit_rejects_an_impossible_threshold():
    y = np.array([1, 0, 1, 0])
    assert np.isnan(cal.net_benefit(y, np.ones_like(y, dtype=bool), 0.0))
    assert np.isnan(cal.net_benefit(y, np.ones_like(y, dtype=bool), 1.0))


def test_decision_curve_always_carries_both_references():
    y = np.array([1, 1, 0, 0, 0, 0])
    curve = cal.decision_curve(y, {"rule": np.array([1, 0, 1, 0, 0, 0], dtype=bool)},
                               thresholds=[0.1, 0.3, 0.5])
    assert set(curve["strategy"]) == {cal.TREAT_ALL, cal.TREAT_NONE, "rule"}
    assert (curve.loc[curve["strategy"] == cal.TREAT_NONE, "net_benefit"] == 0).all()
    assert len(curve) == 3 * 3


def test_a_probability_strategy_is_rethresholded_at_every_t():
    """A boolean rule is fixed; a model is re-applied at each threshold."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.4, 0.1])
    curve = cal.decision_curve(y, {"model": p}, thresholds=[0.2, 0.5])
    model = curve[curve["strategy"] == "model"].set_index("threshold")["net_benefit"]
    # At t=0.2 three patients are flagged; at t=0.5 only one.
    assert model.loc[0.2] == pytest.approx(2 / 4 - (1 / 4) * (0.2 / 0.8))
    assert model.loc[0.5] == pytest.approx(1 / 4 - 0.0)


def test_summary_finds_where_a_strategy_beats_the_references():
    y = np.array([1] * 30 + [0] * 70)
    perfect = np.array([True] * 30 + [False] * 70)
    curve = cal.decision_curve(y, {"perfect": perfect}, thresholds=[0.1, 0.3, 0.5])
    row = cal.decision_curve_summary(curve).set_index("strategy").loc["perfect"]
    assert not row["is_reference"]
    assert row["pct_of_range_best_available"] == pytest.approx(100.0)
    assert row["max_net_benefit"] == pytest.approx(0.30)


def test_summary_marks_a_useless_strategy_as_never_beating_the_references():
    y = np.array([1] * 30 + [0] * 70)
    backwards = np.array([False] * 30 + [True] * 70)   # flags only the benign
    curve = cal.decision_curve(y, {"backwards": backwards}, thresholds=[0.2, 0.4])
    row = cal.decision_curve_summary(curve).set_index("strategy").loc["backwards"]
    assert row["pct_of_range_beating_references"] == 0.0
    assert np.isnan(row["beats_references_from"])


# --------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------
def _write_model(folder, stem, *, auc, slope, n_pred=3):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_model.json").write_text(json.dumps({
        "model_name": "High-grade meningioma risk calculator",
        "n": 352, "events": 105,
        "coefficients": {"const": -1.0, **{f"x{i}": 0.2 for i in range(n_pred)}},
        "validation": {
            "successful_bootstraps": 1000,
            "metrics": [
                {"metric": "AUC", "apparent": auc + 0.02, "optimism_corrected": auc},
                {"metric": "Brier score", "apparent": 0.18, "optimism_corrected": 0.19},
                {"metric": "Calibration slope", "apparent": 1.0,
                 "optimism_corrected": slope},
            ],
            "calibration": {
                "slope_apparent": 1.0, "slope_corrected": slope,
                "intercept_apparent": 0.0,
                "bins": [{"predicted": 0.1, "observed": 0.12, "events": 4, "n": 35},
                         {"predicted": 0.5, "observed": 0.48, "events": 17, "n": 35}],
            },
        },
    }))


def test_artifacts_are_told_apart_by_file_stem_not_by_model_name(tmp_path):
    """Every artifact carries the same product name in model_name."""
    folder = tmp_path / "inferential" / "model_artifacts"
    _write_model(folder, "high_grade_experimental_1", auc=0.73, slope=0.87)
    _write_model(folder, "high_grade_experimental_2", auc=0.66, slope=0.77, n_pred=10)

    table = cal.multivariable_calibration(cal.load_model_artifacts(tmp_path))
    assert set(table["model"]) == {"experimental 1", "experimental 2"}
    assert table["n_bootstrap"].tolist() == [1000, 1000]


def test_missing_corrected_intercept_stays_missing(tmp_path):
    """The modelling artifacts carry no corrected intercept — do not invent one."""
    folder = tmp_path / "inferential" / "model_artifacts"
    _write_model(folder, "high_grade_experimental_1", auc=0.73, slope=0.87)
    row = cal.multivariable_calibration(cal.load_model_artifacts(tmp_path)).iloc[0]
    assert row["intercept_apparent"] == 0.0
    assert pd.isna(row["intercept_corrected"])


def test_best_model_bins_picks_the_best_corrected_auc(tmp_path):
    folder = tmp_path / "inferential" / "model_artifacts"
    _write_model(folder, "high_grade_weak", auc=0.60, slope=0.70)
    _write_model(folder, "high_grade_strong", auc=0.75, slope=0.90)
    bins, stats = cal.best_model_calibration_bins(tmp_path)
    assert stats["model"] == "strong"
    assert len(bins) == 2
    assert bins["lo"].notna().all()


def test_no_model_artifacts_is_not_an_error(tmp_path):
    assert cal.load_model_artifacts(tmp_path) == []
    assert cal.best_model_calibration_bins(tmp_path) is None
    assert cal.multivariable_calibration([]).empty


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------