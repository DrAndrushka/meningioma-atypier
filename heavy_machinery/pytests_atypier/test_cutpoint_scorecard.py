"""The evidence table — every criterion traces to a number, and nothing is graded."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import scorecard as sc


# --- the criteria themselves ----------------------------------------------
def test_every_criterion_is_numbered_uniquely_and_in_order():
    numbers = [c.number for c in sc.CRITERIA]
    assert numbers == sorted(numbers)
    assert len(set(numbers)) == len(numbers)


def test_the_numbering_keeps_its_gaps_so_citations_stay_stable():
    """1, 2, 3, 4 and 7 were removed; the survivors keep their original numbers."""
    assert [c.number for c in sc.CRITERIA] == [5, 6, 8, 9, 10, 11]


def test_the_removed_criteria_are_gone_from_the_module():
    removed = {"separates", "number_works", "not_a_copy", "survives_adjustment",
               "bend_interior"}
    assert removed & {c.key for c in sc.CRITERIA} == set()


def test_every_criterion_states_a_formula_a_grading_rule_and_its_step():
    for c in sc.CRITERIA:
        assert c.question.endswith("?")
        assert c.name and not c.name.endswith("?")
        assert c.formula and c.yes_when and c.no_when and c.step


def test_the_grading_rules_are_two_sides_of_one_test():
    """Every Yes must have a stated No, or the reader cannot check the call."""
    for c in sc.CRITERIA:
        assert c.yes_when != c.no_when


def test_every_criterion_belongs_to_a_named_family():
    assert {c.family for c in sc.CRITERIA} == {sc.FAMILY_THRESHOLD,
                                               sc.FAMILY_CUTPOINT}


def test_every_criterion_is_now_about_the_cutpoint_not_the_measurement():
    """The table's scope narrowed; the footnote has to say so."""
    assert "read together with those analyses" in sc.footnote()


def test_every_criterion_has_an_evaluator():
    from scorecard import _evaluators
    assert set(_evaluators()) == {c.key for c in sc.CRITERIA}


# --- the tables it is built from ------------------------------------------
def _tables(**over):
    base = {
        "eligible": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                   "measurement": "ADC (mean)"}]),
        "separation": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                     "auc": 0.63, "auc_lo": 0.56}]),
        "bend": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                               "bend_is_real": True, "lr_p": 0.009,
                               "scales_agree": True, "knee_at_boundary": False,
                               "knee_percentile": 7.1}]),
        "agreement": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                    "cutoff_min": 0.72, "cutoff_max": 0.79,
                                    "spread_vs_iqr": 0.41}]),
        "wobble": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                 "ci_lo": 0.69, "ci_hi": 0.85,
                                 "stability_ratio": 0.93}]),
        "imputation": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                     "diverges": False, "draw_min": 0.72,
                                     "draw_max": 0.73}]),
        "dichotomy": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                    "or_per_sd": 0.62, "or_per_sd_lo": 0.47,
                                    "or_per_sd_hi": 0.83,
                                    "information_retained": 0.93}]),
        "pairs": pd.DataFrame([{"a": "ADC (mean)", "b": "Max diameter",
                                "spearman": -0.26, "abs_spearman": 0.26,
                                "moves_together": False}]),
        "coefficients": pd.DataFrame([{"model": "Five cut-points",
                                       "predictor": "ADC (mean) ≤ 0.72",
                                       "or": 3.67, "p": 0.0001, "vif": 1.05}]),
    }
    base.update(over)
    eligible = base.pop("eligible")
    return eligible, base


def _long(**over):
    eligible, tables = _tables(**over)
    return sc.scorecard_long(eligible, **tables)


def _met(key: str, **over) -> bool:
    long = _long(**over)
    number = sc.CRITERIA_BY_KEY[key].number
    return bool(long.loc[long["criterion_number"] == number, "met"].iloc[0])


# --- each criterion decides on the number it names ------------------------
def test_a_disagreeing_scale_fails_the_scale_free_criterion():
    assert not _met("scale_free", bend=pd.DataFrame(
        [{"col": "adc_value", "stratum": "all", "bend_is_real": True,
          "lr_p": 0.02, "scales_agree": False, "knee_at_boundary": False,
          "knee_percentile": 47.0}]))


def test_a_wide_bootstrap_interval_fails_the_resampling_criterion():
    assert not _met("survives_resampling", wobble=pd.DataFrame(
        [{"col": "adc_value", "stratum": "all", "ci_lo": 6.8, "ci_hi": 44.8,
          "stability_ratio": 1.21}]))


def test_a_divergent_cutpoint_fails_the_missingness_criterion():
    assert not _met("survives_missingness", imputation=pd.DataFrame(
        [{"col": "adc_value", "stratum": "all", "diverges": True,
          "draw_min": 1.0, "draw_max": 2.0}]))


def test_an_expensive_cut_fails_the_cost_criterion():
    assert not _met("cut_costs_little", dichotomy=pd.DataFrame(
        [{"col": "adc_value", "stratum": "all", "or_per_sd": 1.96,
          "or_per_sd_lo": 1.51, "or_per_sd_hi": 2.54,
          "information_retained": 0.77}]))


# --- every cell shows its working -----------------------------------------
def test_every_row_carries_the_value_that_decided_it():
    long = _long()
    assert long["evidence"].str.len().gt(0).all()
    assert not long["evidence"].str.contains("nan").any()


def test_the_evidence_quotes_the_number_not_the_rule():
    long = _long().set_index("criterion_number")
    assert "0.009" in long.loc[5, "evidence"]
    assert "0.93" in long.loc[9, "evidence"]


def test_every_row_carries_its_formula_and_grading_rule():
    long = _long()
    assert long["formula"].str.len().gt(0).all()
    assert long["graded"].str.startswith("Yes, ").all()


def test_a_missing_input_table_fails_the_criterion_rather_than_crashing():
    long = _long(wobble=pd.DataFrame())
    number = sc.CRITERIA_BY_KEY["survives_resampling"].number
    row = long[long["criterion_number"] == number].iloc[0]
    assert not row["met"] and row["evidence"] == "not scored"


# --- the wide table --------------------------------------------------------
def test_the_wide_table_has_criteria_as_rows_and_measurements_as_columns():
    wide = sc.scorecard_wide(_long())
    assert list(wide.columns) == ["ADC (mean)"]
    assert len(wide.index) == len(sc.CRITERIA) + 1     # + the count row


def test_every_row_is_labelled_with_its_number_and_scientific_name():
    wide = sc.scorecard_wide(_long())
    assert wide.index[0] == "5. Nonlinearity"
    assert "11. Dichotomisation cost" in list(wide.index)


def test_the_count_row_sits_at_the_bottom_and_matches_the_ticks():
    long = _long()
    wide = sc.scorecard_wide(long)
    assert wide.index[-1] == "Criteria met"
    met = int(long["met"].sum())
    assert wide.loc["Criteria met", "ADC (mean)"] == f"{met} of {len(sc.CRITERIA)}"


def test_the_count_row_follows_a_failing_criterion_down():
    long = _long(wobble=pd.DataFrame(
        [{"col": "adc_value", "stratum": "all", "ci_lo": 6.8, "ci_hi": 44.8,
          "stability_ratio": 1.21}]))
    wide = sc.scorecard_wide(long)
    assert wide.loc["9. Sampling stability", "ADC (mean)"] == "No"
    assert wide.loc["Criteria met", "ADC (mean)"] == f"{len(sc.CRITERIA) - 1} of {len(sc.CRITERIA)}"


def test_columns_are_ordered_by_criteria_met():
    long = pd.concat([
        _long(),
        _long().assign(measurement="Fewer", col="fewer", met=False),
    ], ignore_index=True)
    assert list(sc.scorecard_wide(long).columns)[0] == "ADC (mean)"


def test_the_wide_table_says_yes_and_no_rather_than_grading():
    """No 'strong' or 'fragile' — those words outrun the numbers behind them."""
    wide = sc.scorecard_wide(_long()).drop(index="Criteria met")
    assert set(wide.to_numpy().ravel()) <= {"Yes", "No"}


def test_no_verdict_word_appears_anywhere_in_the_output():
    long = _long()
    text = " ".join(long.astype(str).to_numpy().ravel()).lower()
    for word in ("strong", "fragile", "weak", "robust", "moderate"):
        assert word not in text


def test_an_empty_input_gives_an_empty_table_not_an_error():
    assert sc.scorecard_wide(pd.DataFrame()).empty


# --- the footnote ----------------------------------------------------------
def test_the_footnote_names_and_numbers_every_row():
    note = sc.footnote()
    for c in sc.CRITERIA:
        assert f"{c.number}, {c.name}:" in note


def test_the_footnote_gives_the_formula_for_every_criterion():
    note = sc.footnote()
    for c in sc.CRITERIA:
        assert c.formula in note


def test_the_footnote_states_both_sides_of_every_grading_rule():
    note = sc.footnote()
    for c in sc.CRITERIA:
        assert f"Yes, {c.yes_when}; No, {c.no_when}" in note


def test_the_footnote_explains_the_gaps_in_the_numbering():
    assert "Numbering is not consecutive" in sc.footnote()


def test_the_footnote_still_warns_about_the_dichotomisation_trap():
    """Nothing in the table catches it now, so the note must point elsewhere."""
    note = sc.footnote()
    assert "read together with those analyses" in note
    assert "cutting something uninformative costs nothing" in note


def test_the_footnote_opens_in_the_journal_pattern():
    assert sc.footnote().startswith("Note:—")


# --- the summary line ------------------------------------------------------
def test_describe_names_the_leader_and_what_it_failed():
    line = sc.describe_scorecard(_long())
    assert "Most criteria met: ADC (mean)" in line


def test_describe_says_so_when_a_measurement_meets_everything():
    long = _long()
    long["met"] = True
    assert "meets every criterion" in sc.describe_scorecard(long)


def test_describe_handles_an_empty_table():
    assert sc.describe_scorecard(pd.DataFrame()) == "No measurement could be scored."


# --- against the real cohort ----------------------------------------------
def test_the_narrowed_table_no_longer_separates_adc_from_the_edema_index(real_cohort):
    """A recorded consequence, not an endorsement.

    With 1-4 and 7 removed, nothing left in this table asks whether a
    measurement carries signal, duplicates another, or survives adjustment. The
    edema index — null as a number, 0.91 correlated with edema volume, and
    non-significant when adjusted — now scores at or above ADC. The table cannot
    be read alone, and this test exists so that stops being a surprise.
    """
    import bend_location as bl
    import collinearity as co
    import criteria as cr
    import dichotomy as di
    import eligibility as el
    import imputation as imp
    import models as mo
    import nonlinearity as nl
    import separation as sep
    import wobble as wb

    frozen = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
              "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    fits = nl.fit_all(real_cohort)
    separation = sep.separation_table(real_cohort)
    bend = bl.bend_table(real_cohort, fits=fits)
    eligible = el.eligible(el.carry_forward(separation, bend))
    wobble, _ = wb.wobble_table(real_cohort, eligible, n_boot=100, frozen=frozen)
    agreement = cr.agreement(cr.criteria_table(real_cohort, eligible), real_cohort)
    dichotomy = di.dichotomy_table(real_cohort, eligible, frozen)
    imputation = imp.per_draw_cutpoints(imp.load_draws("output"), eligible,
                                        frozen=frozen)
    rho, counts = co.spearman_matrix(real_cohort)
    _, coefficients = mo.compare_sets(real_cohort, cutpoints=frozen, n_boot=60)

    long = sc.scorecard_long(
        eligible, separation=separation, bend=bend, agreement=agreement,
        wobble=wobble, imputation=imputation, dichotomy=dichotomy,
        pairs=co.correlated_pairs(rho, counts), coefficients=coefficients)
    counts_met = long.groupby("measurement")["met"].sum()
    assert counts_met["Edema index"] >= counts_met["ADC (mean)"]
