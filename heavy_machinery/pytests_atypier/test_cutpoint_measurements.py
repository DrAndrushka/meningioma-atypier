"""Step 2 — the five measurements are declared correctly and summarised honestly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import measurements as ms


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "adc_value": [0.9, 0.6, 0.8, None],
        "tumor_volume": [10.0, 20.0, 30.0, 40.0],
        "edema_volume_cm3": [1.0, 2.0, 3.0, 4.0],
        "edema_index": [0.1, 0.2, 0.3, 0.4],
        "max_diameter_cm": [3.0, 4.0, 5.0, 6.0],
    })


# --- the declarations themselves ------------------------------------------
def test_all_five_measurements_are_declared():
    assert len(ms.MEASUREMENTS) == 5
    assert {m.col for m in ms.MEASUREMENTS} == {
        "adc_value", "tumor_volume", "edema_volume_cm3", "edema_index",
        "max_diameter_cm"}


def test_adc_is_the_only_one_pointing_downwards():
    lower = [m.col for m in ms.MEASUREMENTS if m.direction == ms.LOWER]
    assert lower == ["adc_value"]


def test_the_three_skewed_measurements_are_log_transformed():
    assert {m.col for m in ms.MEASUREMENTS if m.log_x} == {
        "tumor_volume", "edema_volume_cm3", "edema_index"}


def test_rule_text_uses_the_declared_direction_and_precision():
    adc = ms.MEASUREMENTS_BY_COL["adc_value"]
    diameter = ms.MEASUREMENTS_BY_COL["max_diameter_cm"]
    assert adc.rule_text(0.7241) == "ADC (mean) ≤ 0.72"
    assert diameter.rule_text(3.814) == "Max diameter ≥ 3.81"


def test_every_measurement_prints_a_plain_unit_without_mathtext():
    """Tables and sentences must never carry raw ``$...$``."""
    assert all("$" not in m.unit for m in ms.MEASUREMENTS)


# --- validation ------------------------------------------------------------
def test_declared_measurements_validate_against_the_cohort():
    ms.validate(ms.MEASUREMENTS, _frame())


def test_a_bad_direction_is_rejected():
    bad = ms.Measurement("adc_value", "ADC", "", "", "down", decimals=2)
    with pytest.raises(ms.MeasurementError, match="direction"):
        ms.validate([bad], _frame())


def test_an_absent_column_is_rejected():
    with pytest.raises(ms.MeasurementError, match="not a column"):
        ms.validate(ms.MEASUREMENTS, _frame().drop(columns=["edema_index"]))


# --- the spread table ------------------------------------------------------
def test_spread_table_has_one_row_per_measurement():
    table = ms.spread_table(_frame())
    assert len(table) == 5


def test_spread_table_counts_missing_separately_from_observed():
    row = ms.spread_table(_frame()).set_index("measurement").loc["ADC (mean)"]
    assert row["n_observed"] == 3
    assert row["n_missing"] == 1
    assert row["pct_missing"] == 25.0


def test_spread_table_reports_median_and_iqr_at_declared_precision():
    row = ms.spread_table(_frame()).set_index("measurement").loc["Tumor volume"]
    assert row["median"] == 25.0
    assert row["iqr_low"] == 17.5
    assert row["iqr_high"] == 32.5


def test_spread_table_states_which_way_each_measurement_points():
    table = ms.spread_table(_frame()).set_index("measurement")
    assert table.loc["ADC (mean)", "direction"].startswith("≤")
    assert table.loc["Tumor volume", "direction"].startswith("≥")


def test_spread_table_survives_an_all_missing_measurement():
    df = _frame()
    df["edema_index"] = np.nan
    row = ms.spread_table(df).set_index("measurement").loc["Edema index"]
    assert row["n_observed"] == 0
    assert np.isnan(row["median"])


# --- the summary line ------------------------------------------------------
def test_describe_missing_names_the_widest_gap_first():
    line = ms.describe_missing(ms.spread_table(_frame()))
    assert line.startswith("ADC (mean) is missing for 1 patient (25.0%)")
    assert "Complete for every patient: Tumor volume" in line


def test_describe_missing_ranks_the_rest_behind_the_widest_gap():
    df = _frame()
    df.loc[0, "tumor_volume"] = None
    df.loc[[0, 1], "edema_index"] = None
    line = ms.describe_missing(ms.spread_table(df))
    assert line.startswith("Edema index is missing for 2 patients")
    assert "; then ADC (mean) 1, Tumor volume 1." in line


def test_spread_table_counts_the_pile_at_exactly_zero():
    df = _frame()
    df["edema_volume_cm3"] = [0.0, 0.0, 3.0, 4.0]
    row = ms.spread_table(df).set_index("measurement").loc["Edema volume"]
    assert row["n_zero"] == 2
    assert row["pct_zero"] == 50.0


def test_zeros_are_counted_against_observed_not_the_whole_cohort():
    """A missing value is not a zero — mixing them would understate the pile."""
    df = _frame()
    df["edema_volume_cm3"] = [0.0, None, 3.0, 4.0]
    row = ms.spread_table(df).set_index("measurement").loc["Edema volume"]
    assert row["n_observed"] == 3
    assert row["pct_zero"] == pytest.approx(33.3)


# --- the zero-pile line ----------------------------------------------------
def test_describe_zeros_names_a_large_pile():
    df = _frame()
    df["edema_volume_cm3"] = [0.0, 0.0, 3.0, 4.0]
    line = ms.describe_zeros(ms.spread_table(df))
    assert "Edema volume 2/4 (50.0%)" in line
    assert "amount alone" in line


# --- the two strata --------------------------------------------------------
def test_only_the_edema_measurements_are_declared_zero_inflated():
    assert {m.col for m in ms.MEASUREMENTS if m.zero_inflated} == {
        "edema_volume_cm3", "edema_index"}


def test_zero_inflated_measurements_get_two_strata():
    edema = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    adc = ms.MEASUREMENTS_BY_COL["adc_value"]
    assert edema.strata == (ms.STRATUM_ALL, ms.STRATUM_PRESENT)
    assert adc.strata == (ms.STRATUM_ALL,)


def test_stratum_label_says_which_question_is_being_asked():
    edema = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    assert edema.stratum_label(ms.STRATUM_ALL) == "Edema volume"
    assert edema.stratum_label(ms.STRATUM_PRESENT) == "Edema volume (where present)"


def test_present_stratum_drops_the_zeros():
    df = _frame()
    df["edema_volume_cm3"] = [0.0, 0.0, 3.0, 4.0]
    edema = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    assert ms.stratum_mask(df, edema, ms.STRATUM_ALL).sum() == 4
    assert ms.stratum_mask(df, edema, ms.STRATUM_PRESENT).sum() == 2


def test_a_missing_value_is_in_neither_stratum():
    """Unmeasured is not the same claim as measured-and-absent."""
    df = _frame()
    df["edema_volume_cm3"] = [0.0, None, 3.0, 4.0]
    edema = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    assert ms.stratum_mask(df, edema, ms.STRATUM_ALL).sum() == 3
    assert ms.stratum_mask(df, edema, ms.STRATUM_PRESENT).sum() == 2


def test_unknown_stratum_is_rejected():
    edema = ms.MEASUREMENTS_BY_COL["edema_volume_cm3"]
    with pytest.raises(ms.MeasurementError, match="Unknown stratum"):
        ms.stratum_mask(_frame(), edema, "nonzero")


def test_presence_table_covers_only_zero_inflated_measurements():
    df = _frame()
    df["edema_volume_cm3"] = [0.0, 0.0, 3.0, 4.0]
    table = ms.presence_table(df)
    assert list(table["measurement"]) == ["Edema volume", "Edema index"]
    row = table.set_index("measurement").loc["Edema volume"]
    assert row["n_absent"] == 2 and row["n_present"] == 2
    assert row["pct_present"] == 50.0


# --- keeping the declaration honest ---------------------------------------
def test_an_undeclared_pile_is_reported():
    df = _frame()
    df["tumor_volume"] = [0.0, 0.0, 30.0, 40.0]
    notes = ms.check_zero_declarations(df)
    assert any("Tumor volume" in n and "Declare it" in n for n in notes)


def test_a_declaration_with_no_pile_is_reported():
    df = _frame()          # edema columns have no zeros at all
    notes = ms.check_zero_declarations(df)
    assert any("Edema volume" in n and "costs sample size" in n for n in notes)


def test_a_matching_declaration_is_silent():
    df = _frame()
    df["edema_volume_cm3"] = [0.0, 0.0, 3.0, 4.0]
    df["edema_index"] = [0.0, 0.0, 0.3, 0.4]
    assert ms.check_zero_declarations(df) == []


def test_describe_zeros_stays_quiet_below_the_reporting_floor():
    line = ms.describe_zeros(ms.spread_table(_frame()))
    assert line == "No measurement has more than 10% of patients at zero."


def test_describe_missing_says_so_when_nothing_is_missing():
    df = _frame()
    df["adc_value"] = [0.9, 0.6, 0.8, 0.7]
    assert ms.describe_missing(ms.spread_table(df)) == (
        "Every measurement is complete for every patient.")
