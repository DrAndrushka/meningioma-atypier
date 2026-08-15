"""Step 9 — the price of replacing a measurement with a yes/no answer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import dichotomy as di
import measurements as ms
import separation as sep


def _paired(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(y * 0.9, 1.0)


# --- paired AUC comparison -------------------------------------------------
def test_a_score_compared_with_itself_has_zero_difference():
    y, x = _paired()
    out = sep.delong_compare(y, x, x)
    assert out["difference"] == pytest.approx(0.0)
    assert out["difference_lo"] == pytest.approx(0.0)
    assert out["difference_hi"] == pytest.approx(0.0)


def test_the_difference_is_the_gap_between_the_two_aucs():
    y, x = _paired()
    noise = np.random.default_rng(1).normal(size=len(y))
    out = sep.delong_compare(y, x, noise)
    assert out["difference"] == pytest.approx(out["auc_a"] - out["auc_b"])


def test_a_clearly_better_score_wins_significantly():
    y, x = _paired()
    noise = np.random.default_rng(2).normal(size=len(y))
    assert sep.delong_compare(y, x, noise)["p"] < 0.01


def test_correlated_scores_get_a_narrower_interval_than_independence_implies():
    """Two scores on one cohort agree; ignoring that overstates the difference."""
    y, x = _paired()
    almost = x + np.random.default_rng(3).normal(scale=0.01, size=len(y))
    paired = sep.delong_compare(y, x, almost)
    a = sep.auc_with_ci(y, x, "higher")
    b = sep.auc_with_ci(y, almost, "higher")
    naive = float(np.sqrt(a["auc_var"] + b["auc_var"])) * 1.96
    assert (paired["difference_hi"] - paired["difference"]) < naive


def test_the_comparison_is_antisymmetric():
    y, x = _paired()
    noise = np.random.default_rng(4).normal(size=len(y))
    forward = sep.delong_compare(y, x, noise)
    backward = sep.delong_compare(y, noise, x)
    assert forward["difference"] == pytest.approx(-backward["difference"])
    assert forward["p"] == pytest.approx(backward["p"])


# --- standardisation -------------------------------------------------------
def test_standardising_gives_unit_spread():
    x = np.random.default_rng(5).gamma(2, 8, 300)
    assert np.std(di.standardise(x, log_x=False), ddof=1) == pytest.approx(1.0)
    assert np.mean(di.standardise(x, log_x=False)) == pytest.approx(0.0, abs=1e-9)


def test_log1p_is_used_so_a_legitimate_zero_survives():
    """log(0) would drop exactly the patients whose absent edema is the finding."""
    x = np.array([0.0, 1.0, 5.0, 20.0])
    assert np.isfinite(di.standardise(x, log_x=True)).all()


def test_a_constant_measurement_does_not_divide_by_zero():
    assert (di.standardise(np.full(20, 3.0), log_x=False) == 0).all()


# --- odds ratios -----------------------------------------------------------
def test_a_null_predictor_has_an_odds_ratio_near_one():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400)
    out = di.odds_ratio(y, rng.normal(size=400))
    assert out["or_lo"] < 1.0 < out["or_hi"]


def test_a_real_predictor_has_an_interval_clear_of_one():
    y, x = _paired()
    out = di.odds_ratio(y, di.standardise(x, log_x=False))
    assert out["or_lo"] > 1.0 and out["p"] < 0.001


def test_a_well_behaved_wald_interval_is_not_flagged_asymmetric():
    y, x = _paired()
    assert not di.odds_ratio(y, di.standardise(x, log_x=False))["asymmetric"]


def test_too_few_patients_returns_blanks_rather_than_raising():
    out = di.odds_ratio(np.array([0, 1, 0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(out["or"])


# --- the comparison --------------------------------------------------------
def _cohort(seed: int = 7, n: int = 340) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tumor = rng.gamma(2, 8, n) + 4 * y
    edema = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.12 * y, 0.15),
        "tumor_volume": tumor,
        "edema_volume_cm3": edema,
        "edema_index": edema / tumor,
        "max_diameter_cm": rng.normal(3.8 + 0.4 * y, 1.2),
    })


def _eligible(df):
    import bend_location as bl
    import eligibility as el
    import nonlinearity as nl
    fits = nl.fit_all(df)
    return el.eligible(el.carry_forward(sep.separation_table(df),
                                        bl.bend_table(df, fits=fits)))


def test_a_binary_auc_is_the_average_of_sensitivity_and_specificity():
    """The identity that makes this comparison exact rather than approximate."""
    from accuracy import accuracy_at
    df = _cohort()
    m = ms.MEASUREMENTS_BY_COL["tumor_volume"]
    out = di.dichotomy_for(df, m, "all", 20.0)
    perf = accuracy_at(df["high_grade"].to_numpy(),
                       df["tumor_volume"].to_numpy(), 20.0, m.direction)
    assert out["auc_binary"] == pytest.approx(
        (perf["sensitivity"] + perf["specificity"]) / 2)


def test_information_retained_is_the_share_of_lift_above_a_coin_flip():
    df = _cohort()
    out = di.dichotomy_for(df, ms.MEASUREMENTS_BY_COL["tumor_volume"], "all", 20.0)
    assert out["information_retained"] == pytest.approx(
        (out["auc_binary"] - 0.5) / (out["auc_continuous"] - 0.5))


def test_dichotomising_can_beat_the_raw_number_when_the_signal_is_a_step():
    """A measurement whose only signal is 'any at all' loses nothing to a cut."""
    rng = np.random.default_rng(8)
    n = 400
    present = rng.random(n) < 0.6
    y = rng.binomial(1, np.where(present, 0.45, 0.15))
    x = np.where(present, rng.gamma(2, 5, n), 0.0)
    df = pd.DataFrame({"high_grade": y, "edema_volume_cm3": x})
    m = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    out = di.dichotomy_for(df, m, "all", 0.01)
    assert out["information_retained"] > 0.9


def test_the_loss_interval_brackets_the_loss():
    df = _cohort()
    out = di.dichotomy_for(df, ms.MEASUREMENTS_BY_COL["tumor_volume"], "all", 20.0)
    assert out["auc_loss_lo"] <= out["auc_loss"] <= out["auc_loss_hi"]


def test_both_odds_ratios_are_reported_for_the_same_patients():
    df = _cohort()
    out = di.dichotomy_for(df, ms.MEASUREMENTS_BY_COL["tumor_volume"], "all", 20.0)
    assert np.isfinite(out["or_per_sd"]) and np.isfinite(out["or_binary"])
    assert out["n"] == int(df["tumor_volume"].notna().sum())


def test_the_log_transform_declaration_travels_into_the_table():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df),
                               {"tumor_volume": 20.0, "adc_value": 0.8})
    by_col = table.set_index("col")["log_transformed"]
    assert by_col["tumor_volume"] and not by_col["adc_value"]


def test_a_measurement_without_a_cutpoint_is_skipped():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df), {"tumor_volume": 20.0})
    assert list(table["col"]) == ["tumor_volume"]


def test_the_table_is_ranked_by_what_survives_the_cut():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df),
                               {"tumor_volume": 20.0, "adc_value": 0.8,
                                "max_diameter_cm": 3.8})
    retained = table["information_retained"].to_numpy()
    assert (np.diff(retained) <= 1e-12).all()


# --- the summary lines -----------------------------------------------------
def test_describe_names_the_worst_loss():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df),
                               {"tumor_volume": 20.0, "adc_value": 0.8})
    assert "Dichotomising costs most in" in di.describe_dichotomy(table)


def test_describe_says_plainly_when_no_loss_is_significant():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df), {"adc_value": 0.8})
    table["auc_loss_p"] = 0.9
    assert "No loss reaches significance" in di.describe_dichotomy(table)


def test_asymmetry_notes_are_empty_on_well_behaved_fits():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df), {"tumor_volume": 20.0})
    assert di.asymmetry_notes(table) == []


def test_asymmetry_notes_name_firth_when_wald_strains():
    df = _cohort()
    table = di.dichotomy_table(df, _eligible(df), {"tumor_volume": 20.0})
    table["or_per_sd_asymmetric"] = True
    assert "Firth" in di.asymmetry_notes(table)[0]


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_the_binary_odds_ratios_match_the_published_forest(real_cohort):
    """These five appear in the existing univariate OR forest figure."""
    expected = {"adc_value": 4.12, "max_diameter_cm": 3.26,
                "edema_volume_cm3": 3.10, "tumor_volume": 3.10,
                "edema_index": 2.65}
    table = di.dichotomy_table(real_cohort, _eligible(real_cohort), _FROZEN)
    got = table.set_index("col")["or_binary"]
    for col, value in expected.items():
        assert got[col] == pytest.approx(value, abs=0.01), col


def test_no_odds_ratio_needs_a_penalised_fit(real_cohort):
    table = di.dichotomy_table(real_cohort, _eligible(real_cohort), _FROZEN)
    assert di.asymmetry_notes(table) == []
