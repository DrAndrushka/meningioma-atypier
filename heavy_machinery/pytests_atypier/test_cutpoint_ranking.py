"""Step 10 — all ten ranked, and the pre-specified pairs tested properly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import ranking as rk
import separation as sep


def _cohort(seed: int = 9, n: int = 340) -> pd.DataFrame:
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


_CUTS = {"adc_value": 0.8, "max_diameter_cm": 3.8, "tumor_volume": 20.0,
         "edema_volume_cm3": 4.0, "edema_index": 0.2}


# --- the ranking -----------------------------------------------------------
def test_each_measurement_appears_in_both_forms():
    df = _cohort()
    table = rk.ranked_table(df, _eligible(df), _CUTS)
    counts = table.groupby("col")["form"].nunique()
    assert (counts == 2).all()


def test_a_measurement_without_a_cutpoint_appears_only_as_a_number():
    df = _cohort()
    table = rk.ranked_table(df, _eligible(df), {"adc_value": 0.8})
    forms = table[table["col"] == "tumor_volume"]["form"].tolist()
    assert forms == [rk.FORM_CONTINUOUS]


def test_the_table_is_ranked_best_first():
    df = _cohort()
    table = rk.ranked_table(df, _eligible(df), _CUTS)
    assert (np.diff(table["auc"].to_numpy()) <= 1e-12).all()
    assert list(table["rank"]) == list(range(1, len(table) + 1))


def test_every_row_carries_its_own_denominator():
    """Rows come from different numbers of patients; a rank without n misleads."""
    df = _cohort()
    df.loc[:40, "adc_value"] = np.nan
    table = rk.ranked_table(df, _eligible(df), _CUTS)
    by_col = table.set_index("col")["n"]
    assert by_col["adc_value"].max() < by_col["max_diameter_cm"].max()


def test_the_binary_row_is_labelled_with_the_rule_it_applies():
    df = _cohort()
    table = rk.ranked_table(df, _eligible(df), _CUTS)
    label = table[(table["col"] == "adc_value")
                  & (table["form"] == rk.FORM_BINARY)]["label"].iloc[0]
    assert label == "ADC (mean) ≤ 0.80"


def test_the_claim_from_step_five_travels_into_the_ranking():
    df = _cohort()
    table = rk.ranked_table(df, _eligible(df), _CUTS)
    assert table["claim"].str.len().gt(0).all()


# --- the paired comparison -------------------------------------------------
def test_the_pair_is_scored_on_patients_who_have_both():
    """Scoring each on its own patients would contrast two different cohorts."""
    df = _cohort()
    df.loc[:30, "tumor_volume"] = np.nan
    df.loc[300:, "max_diameter_cm"] = np.nan
    out = rk.compare_pair(df, "tumor_volume", "max_diameter_cm")
    both = (df["tumor_volume"].notna() & df["max_diameter_cm"].notna()).sum()
    assert out["n_both"] == both


def test_a_measurement_compared_with_itself_shows_no_difference():
    df = _cohort()
    out = rk.compare_pair(df, "tumor_volume", "tumor_volume")
    assert out["difference"] == pytest.approx(0.0)


def test_the_difference_interval_brackets_the_difference():
    df = _cohort()
    out = rk.compare_pair(df, "tumor_volume", "max_diameter_cm")
    assert out["difference_lo"] <= out["difference"] <= out["difference_hi"]


def test_direction_is_applied_before_comparing():
    """ADC points down; comparing it unoriented would invert its AUC."""
    df = _cohort()
    out = rk.compare_pair(df, "adc_value", "max_diameter_cm")
    assert out["auc_a"] > 0.5


def test_the_binary_form_needs_both_cutpoints():
    df = _cohort()
    out = rk.compare_pair(df, "tumor_volume", "max_diameter_cm",
                          form=rk.FORM_BINARY,
                          cutpoints={"tumor_volume": 20.0})
    assert np.isnan(out["difference"])


def test_too_few_patients_returns_blanks_rather_than_raising():
    df = _cohort().head(12)
    out = rk.compare_pair(df, "tumor_volume", "max_diameter_cm")
    assert np.isnan(out["difference"])


# --- multiplicity ----------------------------------------------------------
def test_only_the_pre_specified_pairs_are_tested():
    """Comparing all forty-five pairs would guarantee a false positive."""
    assert len(rk.PRE_SPECIFIED_PAIRS) == 2
    assert {p[0] for p in rk.PRE_SPECIFIED_PAIRS} == {"tumor_volume",
                                                      "edema_volume_cm3"}


def test_every_pair_is_tested_in_both_forms():
    df = _cohort()
    table = rk.pairwise_table(df, cutpoints=_CUTS)
    assert len(table) == 2 * len(rk.PRE_SPECIFIED_PAIRS)


def test_correction_is_applied_across_the_whole_family():
    df = _cohort()
    table = rk.pairwise_table(df, cutpoints=_CUTS)
    assert (table["p_fdr"] >= table["p"] - 1e-12).all()


def test_the_correction_is_the_same_one_the_eda_tables_use():
    from eda import benjamini_hochberg
    df = _cohort()
    table = rk.pairwise_table(df, cutpoints=_CUTS)
    assert table["p_fdr"].to_numpy() == pytest.approx(
        benjamini_hochberg(table["p"]).to_numpy(), nan_ok=True)


def test_distinguishable_is_judged_on_the_corrected_p():
    df = _cohort()
    table = rk.pairwise_table(df, cutpoints=_CUTS)
    assert (table["distinguishable"] == (table["p_fdr"] < 0.05)).all()


def test_each_pair_records_why_it_was_chosen():
    df = _cohort()
    table = rk.pairwise_table(df, cutpoints=_CUTS)
    assert table["because"].str.contains("both measure").all()


# --- the summary line ------------------------------------------------------
def test_describe_names_the_leader_and_the_indistinguishable_pairs():
    df = _cohort()
    elig = _eligible(df)
    line = rk.describe_ranking(rk.ranked_table(df, elig, _CUTS),
                               rk.pairwise_table(df, cutpoints=_CUTS))
    assert line.startswith("Best discrimination:")
    assert "does not need both" in line or "Genuinely different" in line


def test_describe_handles_an_empty_ranking():
    assert rk.describe_ranking(pd.DataFrame(), pd.DataFrame()) == "Nothing to rank."


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_ten_rows_on_the_real_cohort(real_cohort):
    table = rk.ranked_table(real_cohort, _eligible(real_cohort), _FROZEN)
    assert len(table) == 10
    assert (table["form"] == rk.FORM_CONTINUOUS).sum() == 5


def test_size_measured_two_ways_is_not_distinguishable(real_cohort):
    """Tumour volume and max diameter carry the same information here."""
    table = rk.pairwise_table(real_cohort, cutpoints=_FROZEN)
    size = table[(table["col_a"] == "tumor_volume")
                 & (table["form"] == rk.FORM_CONTINUOUS)].iloc[0]
    assert not size["distinguishable"]
    assert size["n_both"] == 329
