"""Step 11 — the predictor sets head to head, with VIF and optimism correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import models as mo


_CUTS = {"adc_value": 0.8, "max_diameter_cm": 3.8, "tumor_volume": 20.0,
         "edema_volume_cm3": 4.0, "edema_index": 0.2}


def _cohort(seed: int = 12, n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tumor = rng.gamma(2, 8, n) + 5 * y
    edema = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.15 * y, 0.15),
        "tumor_volume": tumor,
        "edema_volume_cm3": edema,
        "edema_index": edema / tumor,
        "max_diameter_cm": np.cbrt(tumor) + rng.normal(0, 0.1, n),
    })


# --- one shared denominator ------------------------------------------------
def test_every_model_is_fitted_on_the_same_patients():
    """Otherwise the smaller model wins on sample size rather than on merit."""
    df = _cohort()
    df.loc[:30, "tumor_volume"] = np.nan
    df.loc[40:60, "edema_index"] = np.nan
    summary, _ = mo.compare_sets(df, cutpoints=_CUTS, n_boot=40)
    assert summary["n"].nunique() == 1

    df = _cohort()
    df.loc[:20, "adc_value"] = np.nan
    mask = mo.common_patients(df)
    assert int(mask.sum()) == int(df[list(mo.ALL_FIVE)].notna().all(axis=1).sum())


# --- the design matrix -----------------------------------------------------
def test_the_design_matrix_standardises_numbers_and_names_its_rules():
    df = _cohort()
    X = mo.design_matrix(df, mo.ALL_FIVE, mo.FORM_NUMBERS)
    assert np.allclose(X.std(ddof=1).to_numpy(), 1.0, atol=1e-9)

    X = mo.design_matrix(df, mo.ALL_FIVE, mo.FORM_CUTPOINTS, _CUTS)
    assert set(np.unique(X.to_numpy())) <= {0.0, 1.0}

    X = mo.design_matrix(df, ("adc_value",), mo.FORM_CUTPOINTS, _CUTS)
    assert list(X.columns) == ["ADC (mean) ≤ 0.80"]

    with pytest.raises(KeyError, match="No cut-point declared"):
        mo.design_matrix(df, mo.ALL_FIVE, mo.FORM_CUTPOINTS, {"adc_value": 0.8})


# --- VIF -------------------------------------------------------------------
def test_vif_catches_redundancy_that_no_pair_of_columns_shows():
    """variance_inflation_factor on one column divides by a zero residual, so a
    lone predictor is 1.0 by definition."""
    assert mo.vif_table(pd.DataFrame({"a": np.arange(50.0)}))["vif"].tolist() == [1.0]

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    assert mo.vif_table(X)["vif"].max() < 1.3

    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "almost_a": a + rng.normal(0, 0.05, 200),
                      "c": rng.normal(size=200)})
    assert mo.vif_table(X).set_index("predictor").loc["a", "vif"] > mo.VIF_SEVERE

    # c = a + b: every pairwise correlation is modest, the trio is redundant.
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=300), rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(0, 0.05, 300)})
    pairwise = np.array(X.corr().abs().to_numpy(), copy=True)
    np.fill_diagonal(pairwise, 0)
    assert pairwise.max() < 0.80          # no pair looks alarming
    assert mo.vif_table(X)["vif"].max() > mo.VIF_SEVERE


# --- optimism --------------------------------------------------------------
def test_more_predictors_carry_more_optimism_and_correction_lowers_the_auc():
    """The reason the comparison would be rigged without correction."""
    df = _cohort()
    y = df["high_grade"].to_numpy()

    out = mo.optimism_corrected_auc(
        mo.design_matrix(df, mo.ALL_FIVE, mo.FORM_NUMBERS), y, n_boot=120)
    assert out["optimism"] > 0
    assert out["auc_corrected"] < out["auc_apparent"]

    big = mo.optimism_corrected_auc(
        mo.design_matrix(df, mo.ALL_FIVE, mo.FORM_NUMBERS), y, n_boot=200)
    small = mo.optimism_corrected_auc(
        mo.design_matrix(df, mo.REPRESENTATIVES, mo.FORM_NUMBERS), y, n_boot=200)
    assert big["optimism"] > small["optimism"]

    X = mo.design_matrix(df, mo.REPRESENTATIVES, mo.FORM_NUMBERS)
    a = mo.optimism_corrected_auc(X, y, n_boot=80, seed=5)
    b = mo.optimism_corrected_auc(X, y, n_boot=80, seed=5)
    assert a["auc_corrected"] == pytest.approx(b["auc_corrected"])


# --- the comparison --------------------------------------------------------
def test_every_predictor_set_is_fitted_ranked_and_flagged():
    df = _cohort()
    summary, coefs = mo.compare_sets(df, cutpoints=_CUTS, n_boot=40)

    assert len(summary) == len(mo.PREDICTOR_SETS)

    values = summary["auc_corrected"].dropna().to_numpy()
    assert (np.diff(values) <= 1e-12).all()

    row = summary.set_index("model").loc["Five numbers"]
    assert row["epv"] == pytest.approx(row["n_high"] / 5)

    assert coefs["vif"].notna().all()
    assert (coefs.loc[coefs["vif"] >= mo.VIF_SEVERE, "vif_flag"]
            == "uninterpretable").all()
    assert (coefs.loc[coefs["vif"] < mo.VIF_NOTABLE, "vif_flag"] == "").all()

    line = mo.describe_models(summary, coefs)
    assert "Best after optimism correction" in line
    assert "collinearity" in line or "No predictor's interval is inflated" in line

    assert mo.describe_models(pd.DataFrame(), pd.DataFrame()) == (
        "No predictor set could be fitted.")


def test_representatives_are_one_per_dimension():
    assert set(mo.REPRESENTATIVES) == {"max_diameter_cm", "edema_volume_cm3",
                                       "adc_value"}
    assert "tumor_volume" not in mo.REPRESENTATIVES
    assert "edema_index" not in mo.REPRESENTATIVES


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_the_shared_denominator_is_three_hundred_and_four(real_cohort):
    assert int(mo.common_patients(real_cohort).sum()) == 304


def test_dropping_the_redundant_twin_rescues_max_diameter(real_cohort):
    """The whole point of step 10b: collinearity hides a real effect."""
    summary, coefs = mo.compare_sets(real_cohort, cutpoints=_FROZEN, n_boot=100)
    five = coefs[(coefs["model"] == "Five numbers")
                 & coefs["predictor"].str.startswith("Max diameter")].iloc[0]
    three = coefs[(coefs["model"] == "Three numbers")
                  & coefs["predictor"].str.startswith("Max diameter")].iloc[0]
    assert five["p"] > 0.05 and five["vif"] >= mo.VIF_NOTABLE
    assert three["p"] < 0.05 and three["vif"] < mo.VIF_NOTABLE

    # And every three-predictor model is clean.
    three_sets = coefs[coefs["model"].str.startswith("Three")]
    assert three_sets["vif"].max() < mo.VIF_NOTABLE
