"""Step 4 — the spline basis, the likelihood-ratio test, and the scale check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import measurements as ms
import nonlinearity as nl


# --- knots -----------------------------------------------------------------
def test_knots_sit_at_harrells_quantiles_and_degrade_cleanly():
    assert nl.DEFAULT_N_KNOTS == 3

    x = np.arange(1001, dtype=float)
    knots = nl.default_knots(x, 3)
    assert knots.size == 3
    assert knots == pytest.approx(np.quantile(x, (0.10, 0.50, 0.90)))

    # A third of this cohort sits at zero, which ties the lower quantiles.
    piled = np.concatenate([np.zeros(60), np.arange(1, 41, dtype=float)])
    assert nl.default_knots(piled, 3).size < 3

    assert nl.default_knots(np.ones(50), 3).size == 0


# --- the basis -------------------------------------------------------------
def test_the_basis_nests_the_linear_term_and_stays_linear_in_the_tails():
    """The nesting is what makes the likelihood-ratio test valid, and
    'restricted' means the tails cannot flap."""
    x = np.linspace(0, 10, 50)
    basis = nl.rcs_basis(x, nl.default_knots(x, 3))
    assert basis[:, 0] == pytest.approx(x)

    x = np.linspace(0, 10, 60)
    assert nl.rcs_basis(x, nl.default_knots(x, 3)).shape[1] == 2
    assert nl.rcs_basis(np.linspace(0, 10, 30), np.array([1.0, 2.0])).shape[1] == 1

    x = np.linspace(0, 100, 400)
    knots = nl.default_knots(x, 3)
    basis = nl.rcs_basis(x, knots)
    tail = x > knots[-1]
    combined = basis[tail, 0] + basis[tail, 1]
    assert np.ptp(np.diff(np.diff(combined))) == pytest.approx(0.0, abs=1e-8)


# --- the test itself -------------------------------------------------------
def _bent(n: int = 400, seed: int = 0):
    """Risk flat then rising — a real bend."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    logit = np.where(x < 5, -2.0, -2.0 + 1.4 * (x - 5))
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def _straight(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 0.5 * x))))


def test_a_real_bend_is_detected_and_a_straight_climb_is_not():
    x, y = _bent()
    fit = nl.fit_spline(x, y)
    assert fit.bent
    assert fit.lr_df == 1     # 3 knots -> 2 columns -> 1 extra

    assert not nl.fit_spline(*_straight()).bent


def test_the_fitted_curve_comes_back_banded_and_trimmed_to_the_patients():
    x, y = _bent()
    fit = nl.fit_spline(x, y, log_fit=True)
    assert fit.log_fitted
    assert fit.grid.min() >= x.min() and fit.grid.max() <= x.max()

    fit = nl.fit_spline(x, y)
    assert (fit.risk_lo <= fit.risk).all() and (fit.risk <= fit.risk_hi).all()
    assert fit.grid.min() >= np.quantile(x, 0.025) - 1e-9
    assert fit.grid.max() <= np.quantile(x, 0.975) + 1e-9


# --- graceful failure ------------------------------------------------------
def test_too_few_patients_or_events_is_blank_not_fatal():
    fit = nl.fit_spline(np.arange(10.0), np.array([0, 1] * 5))
    assert not fit.spline_fitted
    assert "too few patients" in fit.note
    assert np.isnan(fit.lr_p)
    assert not fit.bent

    y = np.zeros(60, dtype=int)
    y[:2] = 1
    assert "too few patients or events" in nl.fit_spline(
        np.linspace(0, 10, 60), y).note


# --- parity with the implementation this replaces -------------------------
# Frozen from the previous threshold_phase.risk_curves fit on the real cohort,
# verified equal to four decimals before that module was retired. These are the
# numbers the Methods reproducibility sentence refers to; if one moves, the
# claim in the manuscript has stopped being true.
_FROZEN_LR_P = {
    ("adc_value", False): 0.0093, ("adc_value", True): 0.0153,
    ("tumor_volume", False): 0.0210, ("tumor_volume", True): 0.9738,
    ("edema_volume_cm3", False): 0.0074, ("edema_volume_cm3", True): 0.1587,
    ("edema_index", False): 0.0473, ("edema_index", True): 0.0019,
    ("max_diameter_cm", False): 0.5028, ("max_diameter_cm", True): 0.7730,
}


@pytest.mark.parametrize("key,expected", sorted(_FROZEN_LR_P.items()))
def test_reproduces_the_retired_implementation(key, expected, real_cohort):
    col, log_fit = key
    x = pd.to_numeric(real_cohort[col], errors="coerce").to_numpy()
    y = pd.to_numeric(real_cohort["high_grade"], errors="coerce").to_numpy()
    fit = nl.fit_spline(x, y, column=col, log_fit=log_fit)
    assert fit.lr_p == pytest.approx(expected, abs=5e-5)


# --- the table -------------------------------------------------------------
def _cohort(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, y = _bent(n, seed)
    edema = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.1 * y, 0.15),
        "tumor_volume": x,
        "edema_volume_cm3": edema,
        "edema_index": edema / (rng.gamma(2, 8, n) + 1),
        "max_diameter_cm": rng.normal(3.8 + 0.3 * y, 1.2),
    })


def test_the_table_carries_one_row_per_stratum_on_both_scales():
    df = _cohort()
    table = nl.nonlinearity_table(df)
    assert len(table) == 7
    assert {"lr_p", "lr_p_log", "bent_clinical", "bent_log",
            "scales_agree"} <= set(table.columns)

    mismatched = table["bent_clinical"] != table["bent_log"]
    assert (~table.loc[mismatched, "scales_agree"]).all()

    fits = nl.fit_all(df)
    assert len(fits) == 14                      # 7 rows x 2 scales
    assert {log for _, _, log in fits} == {False, True}


# --- the summary line ------------------------------------------------------
def test_describe_names_what_bent_and_whether_the_scales_agreed():
    table = nl.nonlinearity_table(_cohort())
    line = nl.describe_nonlinearity(table)
    assert "Bend detected" in line or "No measurement shows a bend" in line

    scale_dependent = table.copy()
    scale_dependent.loc[0, ["bent_clinical", "bent_log", "scales_agree"]] = [
        True, False, False]
    assert "no claim about a bend is scale-free" in nl.describe_nonlinearity(
        scale_dependent)

    agreeing = table.copy()
    agreeing["scales_agree"] = True
    assert "Both scales agree everywhere." in nl.describe_nonlinearity(agreeing)
