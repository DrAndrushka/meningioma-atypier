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
def test_the_matrix_is_square_symmetric_and_rank_based():
    """The reason ranks are used: cube root is monotone, so rho must be 1, and
    one extreme patient must barely move it where Pearson would swing."""
    rho, counts = co.spearman_matrix(_cohort())
    assert rho.shape == (5, 5)
    assert np.allclose(rho.to_numpy(), rho.to_numpy().T, equal_nan=True)
    assert np.allclose(np.diag(rho.to_numpy()), 1.0)
    assert (counts.to_numpy() == counts.to_numpy().T).all()

    df = _cohort()
    df["max_diameter_cm"] = np.cbrt(df["tumor_volume"])
    rho, _ = co.spearman_matrix(df)
    assert rho.loc["Tumor volume", "Max diameter"] == pytest.approx(1.0)

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

    df = _cohort()
    df.loc[:159, "adc_value"] = np.nan
    df.loc[160:, "tumor_volume"] = np.nan
    rho, counts = co.spearman_matrix(df)
    assert counts.loc["ADC (mean)", "Tumor volume"] == 0
    assert np.isnan(rho.loc["ADC (mean)", "Tumor volume"])


# --- the pair table --------------------------------------------------------
def test_every_pair_is_listed_ranked_by_strength_regardless_of_sign():
    """A survey completed reads differently from a list of problems found."""
    rho, counts = co.spearman_matrix(_cohort())
    pairs = co.correlated_pairs(rho, counts)
    assert len(pairs) == 10   # 5 choose 2
    strengths = pairs["abs_spearman"].dropna()
    assert (np.diff(strengths.to_numpy()) <= 1e-12).all()

    strong_negative = rho.copy()
    strong_negative.loc["ADC (mean)", "Tumor volume"] = -0.95
    strong_negative.loc["Tumor volume", "ADC (mean)"] = -0.95
    by_pair = co.correlated_pairs(strong_negative, counts).set_index(["a", "b"])
    assert by_pair.loc[("ADC (mean)", "Tumor volume"), "moves_together"]

    pairs = co.correlated_pairs(rho, counts, threshold=0.4)
    assert (pairs["moves_together"] == (pairs["abs_spearman"] >= 0.4)).all()


# --- flag agreement --------------------------------------------------------
def test_flags_are_compared_on_patients_with_both_measurements():
    df = _cohort()
    df.loc[:30, "adc_value"] = np.nan
    flags = co.flag_agreement(df, _CUTS).set_index(["col_a", "col_b"])
    assert flags.loc[("adc_value", "tumor_volume"), "n_both"] == int(
        (df["adc_value"].notna() & df["tumor_volume"].notna()).sum())

    df = _cohort()
    df["copy_of_tumor"] = df["tumor_volume"]
    flags = co.flag_agreement(df, _CUTS)
    assert len(flags) == 10                     # no duplicated column declared
    assert flags[flags["agreement"] == 1.0].empty
    # Correlated measurements whose cut-points sit in different places.
    assert flags["agreement"].max() > flags["agreement"].min()
    assert flags.dropna(subset=["phi"])["agreement"].corr(
        flags.dropna(subset=["phi"])["phi"]) > 0.5

    # A rule flagging nobody gives no phi rather than a crash.
    cuts = dict(_CUTS, adc_value=-99.0)
    flags = co.flag_agreement(_cohort(), cuts).set_index(["col_a", "col_b"])
    assert np.isnan(flags.loc[("adc_value", "tumor_volume"), "phi"])

    # Only measurements with a cut-point are paired at all.
    assert len(co.flag_agreement(_cohort(), {"tumor_volume": 20.0,
                                             "adc_value": 0.8})) == 1


# --- the summary line ------------------------------------------------------
def test_describe_warns_what_collinearity_costs_the_model():
    rho, counts = co.spearman_matrix(_cohort())
    line = co.describe_collinearity(co.correlated_pairs(rho, counts))
    assert "split their effect" in line or "can separate them" in line

    pairs = co.correlated_pairs(rho, counts, threshold=0.999)
    assert "A model can separate them." in co.describe_collinearity(pairs)

    assert co.describe_collinearity(pd.DataFrame()) == "No pair could be correlated."


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_size_and_edema_each_form_one_collinear_pair(real_cohort):
    """ADC measures something the size and edema measurements do not."""
    rho, counts = co.spearman_matrix(real_cohort)
    pairs = co.correlated_pairs(rho, counts)
    together = {frozenset((r["a"], r["b"]))
                for _, r in pairs[pairs["moves_together"]].iterrows()}
    assert together == {frozenset(("Tumor volume", "Max diameter")),
                        frozenset(("Edema volume", "Edema index"))}

    others = rho.loc["ADC (mean)"].drop("ADC (mean)")
    assert others.abs().max() < 0.30
