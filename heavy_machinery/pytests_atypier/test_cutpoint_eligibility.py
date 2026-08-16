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
def test_the_claim_follows_separation_and_bend_independently():
    """Max diameter is the second case: second-best AUC, no bend, and it still
    keeps a cut-point as an operating point."""
    assert _claim() == el.CLAIM_THRESHOLD
    assert _claim(bend_is_real=False, quotable=False) == el.CLAIM_OPERATING_POINT
    assert _claim(scales_agree=False) == el.CLAIM_THRESHOLD_SCALED
    assert _claim(quotable=False) == el.CLAIM_OPERATING_POINT
    assert _claim(auc_lo=0.47) == ""

    # Nothing to cut, however convincing the curve looks.
    table = el.carry_forward(*_rows(auc_lo=0.47, bend_is_real=True))
    assert not table.loc[0, "carries_forward"]


# --- the separation floor is judged on the interval -----------------------
def test_the_floor_is_the_lower_bound_not_the_point_estimate():
    """AUC 0.58 from an interval starting at 0.49 is a small study, not a signal."""
    assert not el.carry_forward(*_rows(auc=0.58, auc_lo=0.49)).loc[0, "carries_forward"]
    assert not el.carry_forward(*_rows(auc_lo=0.50)).loc[0, "carries_forward"]
    assert not el.carry_forward(*_rows(auc_lo=np.nan)).loc[0, "carries_forward"]


# --- nothing is silently omitted ------------------------------------------
def test_a_dropped_measurement_stays_in_the_table_with_a_reason():
    """A measurement tested and dropped is a result, not an omission."""
    table = el.carry_forward(*_rows(auc_lo=0.47))
    assert len(table) == 1
    assert not table.loc[0, "carries_forward"]
    assert "no separation shown" in table.loc[0, "reason"]

    sep, bend = _rows()
    flat = _rows(col="flat", measurement="Flat", bend_is_real=False, quotable=False)
    dead = _rows(col="dead", measurement="Dead", auc_lo=0.44)
    table = el.carry_forward(
        pd.concat([sep, flat[0], dead[0]], ignore_index=True),
        pd.concat([bend, flat[1], dead[1]], ignore_index=True))
    assert len(table) == 3
    assert table["reason"].str.len().gt(0).all()
    assert list(el.eligible(table)["col"]) == ["thing", "flat"]

    line = el.describe_eligibility(table)
    assert "threshold: Thing" in line
    assert "operating point: Flat" in line
    assert "Stops here: Dead" in line

    assert el.describe_eligibility(
        el.carry_forward(*_rows(auc_lo=0.4))).startswith("Nothing carries forward")


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

    by_measurement = table.set_index("measurement")
    assert by_measurement.loc["Max diameter", "claim"] == el.CLAIM_OPERATING_POINT
    assert by_measurement.loc["ADC (mean)", "claim"] == el.CLAIM_THRESHOLD
