"""Step 5c — separation and bend fail independently, and say different things."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import eligibility as el


def _rows(**over):
    """One separation row and one bend row for the same measurement."""
    sep = {"measurement": "Thing", "col": "thing", "stratum": "all",
           "n": 300, "auc": 0.68, "auc_lo": 0.61}
    bend = {"measurement": "Thing", "col": "thing", "stratum": "all",
            "bend_is_real": True, "quotable": True, "scales_agree": True}
    sep.update({k: v for k, v in over.items() if k in sep})
    bend.update({k: v for k, v in over.items() if k in bend})
    return pd.DataFrame([sep]), pd.DataFrame([bend])


def _claim(**over) -> str:
    return el.carry_forward(*_rows(**over)).loc[0, "claim"]


# --- the two failures are independent -------------------------------------
def test_a_real_bend_inside_the_data_is_a_threshold():
    assert _claim() == el.CLAIM_THRESHOLD


def test_separation_without_a_bend_is_an_operating_point():
    """Max diameter: second-best AUC, no bend. It keeps a cut-point."""
    assert _claim(bend_is_real=False, quotable=False) == el.CLAIM_OPERATING_POINT


def test_a_bend_only_in_clinical_units_carries_its_caveat():
    assert _claim(scales_agree=False) == el.CLAIM_THRESHOLD_SCALED


def test_a_bend_at_the_edge_of_the_data_is_only_an_operating_point():
    assert _claim(quotable=False) == el.CLAIM_OPERATING_POINT


def test_no_separation_means_no_claim_at_all():
    assert _claim(auc_lo=0.47) == ""


def test_a_bend_cannot_rescue_a_measurement_that_does_not_separate():
    """Nothing to cut, however convincing the curve looks."""
    table = el.carry_forward(*_rows(auc_lo=0.47, bend_is_real=True))
    assert not table.loc[0, "carries_forward"]


# --- the separation floor is judged on the interval -----------------------
def test_the_floor_is_the_lower_bound_not_the_point_estimate():
    """AUC 0.58 from an interval starting at 0.49 is a small study, not a signal."""
    assert not el.carry_forward(*_rows(auc=0.58, auc_lo=0.49)).loc[0, "carries_forward"]


def test_an_auc_exactly_at_the_floor_does_not_clear_it():
    assert not el.carry_forward(*_rows(auc_lo=0.50)).loc[0, "carries_forward"]


def test_a_missing_auc_does_not_carry_forward():
    assert not el.carry_forward(*_rows(auc_lo=np.nan)).loc[0, "carries_forward"]


# --- nothing is silently omitted ------------------------------------------
def test_dropped_rows_stay_in_the_table():
    """A measurement tested and dropped is a result, not an omission."""
    table = el.carry_forward(*_rows(auc_lo=0.47))
    assert len(table) == 1
    assert not table.loc[0, "carries_forward"]
    assert "no separation shown" in table.loc[0, "reason"]


def test_eligible_returns_only_what_carries_forward():
    sep, bend = _rows()
    sep2, bend2 = _rows(col="dead", measurement="Dead", auc_lo=0.44)
    table = el.carry_forward(pd.concat([sep, sep2], ignore_index=True),
                             pd.concat([bend, bend2], ignore_index=True))
    assert len(table) == 2
    assert list(el.eligible(table)["col"]) == ["thing"]


def test_every_row_carries_a_reason():
    sep, bend = _rows()
    sep2, bend2 = _rows(col="flat", measurement="Flat", bend_is_real=False,
                        quotable=False)
    table = el.carry_forward(pd.concat([sep, sep2], ignore_index=True),
                             pd.concat([bend, bend2], ignore_index=True))
    assert table["reason"].str.len().gt(0).all()


# --- the summary line ------------------------------------------------------
def test_describe_groups_by_claim_and_names_what_stopped():
    sep, bend = _rows()
    sep2, bend2 = _rows(col="flat", measurement="Flat", bend_is_real=False,
                        quotable=False)
    sep3, bend3 = _rows(col="dead", measurement="Dead", auc_lo=0.44)
    table = el.carry_forward(
        pd.concat([sep, sep2, sep3], ignore_index=True),
        pd.concat([bend, bend2, bend3], ignore_index=True))
    line = el.describe_eligibility(table)
    assert "threshold: Thing" in line
    assert "operating point: Flat" in line
    assert "Stops here: Dead" in line


def test_describe_says_plainly_when_nothing_survives():
    table = el.carry_forward(*_rows(auc_lo=0.4))
    assert el.describe_eligibility(table).startswith("Nothing carries forward")


# --- against the real cohort ----------------------------------------------
def test_the_real_cohort_drops_exactly_the_two_present_strata(real_cohort):
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep_mod

    fits = nl.fit_all(real_cohort)
    table = el.carry_forward(sep_mod.separation_table(real_cohort),
                             bl.bend_table(real_cohort, fits=fits))
    dropped = set(table.loc[~table["carries_forward"], "measurement"])
    assert dropped == {"Edema volume (where present)",
                       "Edema index (where present)"}


def test_max_diameter_survives_as_an_operating_point(real_cohort):
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep_mod

    fits = nl.fit_all(real_cohort)
    table = el.carry_forward(sep_mod.separation_table(real_cohort),
                             bl.bend_table(real_cohort, fits=fits)
                             ).set_index("measurement")
    assert table.loc["Max diameter", "claim"] == el.CLAIM_OPERATING_POINT


def test_adc_survives_as_a_full_threshold(real_cohort):
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep_mod

    fits = nl.fit_all(real_cohort)
    table = el.carry_forward(sep_mod.separation_table(real_cohort),
                             bl.bend_table(real_cohort, fits=fits)
                             ).set_index("measurement")
    assert table.loc["ADC (mean)", "claim"] == el.CLAIM_THRESHOLD
