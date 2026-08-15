"""Step 10b — which measurements move together, values and flags alike."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import collinearity as co
import measurements as ms


def _cohort(seed: int = 11, n: int = 320) -> pd.DataFrame:
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
        # Diameter built from volume on purpose: these two must come out
        # correlated, because in the real cohort they are.
        "max_diameter_cm": np.cbrt(tumor) + rng.normal(0, 0.15, n),
    })


_CUTS = {"adc_value": 0.8, "max_diameter_cm": 2.5, "tumor_volume": 20.0,
         "edema_volume_cm3": 4.0, "edema_index": 0.2}


# --- the matrix ------------------------------------------------------------
def test_the_matrix_is_square_and_symmetric():
    rho, _ = co.spearman_matrix(_cohort())
    assert rho.shape == (5, 5)
    assert np.allclose(rho.to_numpy(), rho.to_numpy().T, equal_nan=True)


def test_a_measurement_correlates_perfectly_with_itself():
    rho, _ = co.spearman_matrix(_cohort())
    assert np.allclose(np.diag(rho.to_numpy()), 1.0)


def test_a_monotone_transform_gives_a_correlation_of_one():
    """The reason ranks are used: cube root is monotone, so rho must be 1."""
    df = _cohort()
    df["max_diameter_cm"] = np.cbrt(df["tumor_volume"])
    rho, _ = co.spearman_matrix(df)
    assert rho.loc["Tumor volume", "Max diameter"] == pytest.approx(1.0)


def test_ranks_are_not_moved_by_one_extreme_patient():
    """Pearson would swing here; Spearman must barely notice."""
    df = _cohort()
    before, _ = co.spearman_matrix(df)
    df.loc[0, "tumor_volume"] = 100000.0
    after, _ = co.spearman_matrix(df)
    assert abs(after.loc["Tumor volume", "Max diameter"]
               - before.loc["Tumor volume", "Max diameter"]) < 0.05


def test_each_pair_reports_the_patients_it_actually_used():
    df = _cohort()
    df.loc[:40, "adc_value"] = np.nan
    rho, counts = co.spearman_matrix(df)
    assert counts.loc["ADC (mean)", "Tumor volume"] < counts.loc[
        "Tumor volume", "Max diameter"]


def test_pairwise_counts_are_symmetric():
    _, counts = co.spearman_matrix(_cohort())
    assert (counts.to_numpy() == counts.to_numpy().T).all()


def test_a_pair_with_no_shared_patients_is_blank_not_an_error():
    df = _cohort()
    df.loc[:159, "adc_value"] = np.nan
    df.loc[160:, "tumor_volume"] = np.nan
    rho, counts = co.spearman_matrix(df)
    assert counts.loc["ADC (mean)", "Tumor volume"] == 0
    assert np.isnan(rho.loc["ADC (mean)", "Tumor volume"])


# --- the pair table --------------------------------------------------------
def test_every_pair_is_listed_not_only_the_correlated_ones():
    """A survey completed reads differently from a list of problems found."""
    rho, counts = co.spearman_matrix(_cohort())
    assert len(co.correlated_pairs(rho, counts)) == 10   # 5 choose 2


def test_pairs_are_ranked_by_strength_regardless_of_sign():
    rho, counts = co.spearman_matrix(_cohort())
    strengths = co.correlated_pairs(rho, counts)["abs_spearman"].dropna()
    assert (np.diff(strengths.to_numpy()) <= 1e-12).all()


def test_a_strong_negative_correlation_counts_as_moving_together():
    rho, counts = co.spearman_matrix(_cohort())
    rho.loc["ADC (mean)", "Tumor volume"] = -0.95
    rho.loc["Tumor volume", "ADC (mean)"] = -0.95
    pairs = co.correlated_pairs(rho, counts).set_index(["a", "b"])
    assert pairs.loc[("ADC (mean)", "Tumor volume"), "moves_together"]


def test_the_threshold_is_applied_as_documented():
    rho, counts = co.spearman_matrix(_cohort())
    pairs = co.correlated_pairs(rho, counts, threshold=0.4)
    assert (pairs["moves_together"] == (pairs["abs_spearman"] >= 0.4)).all()


# --- flag agreement --------------------------------------------------------
def test_flags_are_compared_on_patients_with_both_measurements():
    df = _cohort()
    df.loc[:30, "adc_value"] = np.nan
    flags = co.flag_agreement(df, _CUTS).set_index(["col_a", "col_b"])
    assert flags.loc[("adc_value", "tumor_volume"), "n_both"] == int(
        (df["adc_value"].notna() & df["tumor_volume"].notna()).sum())


def test_a_rule_compared_with_itself_agrees_completely():
    df = _cohort()
    df["copy_of_tumor"] = df["tumor_volume"]
    flags = co.flag_agreement(df, _CUTS)
    same = flags[flags["agreement"] == 1.0]
    assert len(flags) == 10 and same.empty      # no duplicated column declared


def test_agreement_and_phi_move_in_the_same_direction():
    flags = co.flag_agreement(_cohort(), _CUTS).dropna(subset=["phi"])
    assert flags["agreement"].corr(flags["phi"]) > 0.5


def test_a_rule_flagging_nobody_gives_no_phi_rather_than_a_crash():
    df = _cohort()
    cuts = dict(_CUTS, adc_value=-99.0)          # flags no patient at all
    flags = co.flag_agreement(df, cuts).set_index(["col_a", "col_b"])
    assert np.isnan(flags.loc[("adc_value", "tumor_volume"), "phi"])


def test_only_measurements_with_a_cutpoint_are_paired():
    flags = co.flag_agreement(_cohort(), {"tumor_volume": 20.0,
                                          "adc_value": 0.8})
    assert len(flags) == 1


def test_values_and_flags_can_disagree():
    """Correlated measurements whose cut-points sit in different places."""
    flags = co.flag_agreement(_cohort(), _CUTS)
    assert flags["agreement"].max() > flags["agreement"].min()


# --- the summary line ------------------------------------------------------
def test_describe_warns_what_collinearity_costs_the_model():
    rho, counts = co.spearman_matrix(_cohort())
    line = co.describe_collinearity(co.correlated_pairs(rho, counts))
    assert "split their effect" in line or "can separate them" in line


def test_describe_says_plainly_when_nothing_moves_together():
    rho, counts = co.spearman_matrix(_cohort())
    pairs = co.correlated_pairs(rho, counts, threshold=0.999)
    assert "A model can separate them." in co.describe_collinearity(pairs)


def test_describe_handles_an_empty_table():
    assert co.describe_collinearity(pd.DataFrame()) == "No pair could be correlated."


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_size_and_edema_each_form_one_collinear_pair(real_cohort):
    rho, counts = co.spearman_matrix(real_cohort)
    pairs = co.correlated_pairs(rho, counts)
    together = {frozenset((r["a"], r["b"]))
                for _, r in pairs[pairs["moves_together"]].iterrows()}
    assert together == {frozenset(("Tumor volume", "Max diameter")),
                        frozenset(("Edema volume", "Edema index"))}


def test_adc_is_independent_of_every_other_measurement(real_cohort):
    """ADC measures something the size and edema measurements do not."""
    rho, _ = co.spearman_matrix(real_cohort)
    others = rho.loc["ADC (mean)"].drop("ADC (mean)")
    assert others.abs().max() < 0.30
