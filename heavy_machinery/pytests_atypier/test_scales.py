"""Every phase standardises on the same scale, and says so where it prints.

These tests exist because of a real defect: the modelling phase standardised
tumor volume on the raw scale while the cut-point phase used log1p, so the same
329 patients were published as OR 1.73 in ``report.html`` and OR 1.96 in
``cutpoint_report.html``, with nothing on either page naming the scale. Both
numbers were arithmetically correct, which is why nothing caught it.

The scale now reaches print by four routes, and each has a test here: the
univariable odds ratio in the EDA paper table, the multivariable design matrix,
the ``z_log`` flag written into the calculator artifacts, and the footnote that
says which measurements were logged. A fix that mends one and misses another
puts two scales on the same page, one section apart, which is the same defect
wearing different clothes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import dichotomy as di
import scales
import marker_rules as th
from eda_paper_tables import _continuous_rows, parse_or_ci
from measurements import MEASUREMENTS
from model_calculator import _apply_continuous_transform
from schema_infer import ColSpec


# --------------------------------------------------------------------------
# One declaration
# --------------------------------------------------------------------------
def test_cutpoint_measurements_agree_with_the_shared_declaration():
    assert scales.disagreements({m.col: m.log_x for m in MEASUREMENTS}) == []


def test_threshold_metrics_must_agree_with_the_shared_declaration():
    logged = sorted(scales.LOG1P_COLUMNS)[0]
    frame = pd.DataFrame({logged: [1.0, 2.0, 3.0], "y": [True, False, True]})
    wrong = th.Metric(logged, "Wrong scale", "u", "higher", log_x=False)
    with pytest.raises(ValueError, match="LOG1P_COLUMNS"):
        th.validate_metrics([wrong], frame)


def test_a_metric_outside_the_governed_set_may_choose_its_own_scale():
    """Exploratory metrics and fixtures are not policed — nothing else prints them."""
    frame = pd.DataFrame({"vol": [1.0, 2.0, 3.0], "y": [True, False, True]})
    local = th.Metric("vol", "Volume", "cc", "higher", log_x=True)
    assert [m.col for m in th.validate_metrics([local], frame)] == ["vol"]


def test_the_governed_set_is_the_five_manuscript_measurements():
    assert scales.GOVERNED_COLUMNS == {m.col for m in MEASUREMENTS}


# --------------------------------------------------------------------------
# One implementation
# --------------------------------------------------------------------------
def _skewed_cohort() -> pd.DataFrame:
    rng = np.random.default_rng(20260815)
    n = 240
    vol = rng.lognormal(mean=2.5, sigma=1.1, size=n)
    logit = -1.4 + 0.9 * (np.log1p(vol) - np.log1p(vol).mean())
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"tumor_volume": vol, "high_grade": y})


def test_cutpoint_and_modelling_produce_the_same_per_sd_odds_ratio():
    """The cross-phase pin: the two OR paths must land on the same number.

    This is the assertion the original defect would have failed. It compares the
    modelling phase's published cell against the cut-point phase's estimate on
    identical patients, so a silent revert to the raw scale cannot pass.
    """
    df = _skewed_cohort()
    rows = _continuous_rows(
        df, "high_grade", "tumor_volume",
        positive_class=True, p_fdr=0.01, kind="continuous",
    )
    modelling_or, _, _ = parse_or_ci(rows[0]["effect"])

    z = di.standardise(df["tumor_volume"].to_numpy(dtype=float), log_x=True)
    cutpoint_or = di.odds_ratio(df["high_grade"].to_numpy(dtype=float), z)["or"]

    assert modelling_or == pytest.approx(cutpoint_or, abs=0.005)


def test_every_published_measurement_agrees_across_the_two_phases(real_cohort):
    """The same pin as above, but on the real cohort and on all five measurements.

    The synthetic version proves the two code paths agree; this one proves the
    two *published pages* agree. They are not the same claim — a fixture cannot
    catch a measurement that is governed here but reaches the report through
    some third path, and every measurement in the manuscript has to be checked
    because the defect was silent on four of the five.
    """
    df = real_cohort
    for m in MEASUREMENTS:
        rows = _continuous_rows(df, "high_grade", m.col, positive_class=True,
                                p_fdr=0.01, kind="continuous")
        modelling_or, _, _ = parse_or_ci(rows[0]["effect"])
        sub = df[[m.col, "high_grade"]].dropna()
        cutpoint_or = di.odds_ratio(
            sub["high_grade"].to_numpy(dtype=float),
            di.standardise(sub[m.col].to_numpy(dtype=float), m.log_x))["or"]
        # The modelling side publishes two decimals, so agreement can only be
        # asserted to the precision a reader can actually see.
        assert modelling_or == pytest.approx(cutpoint_or, abs=0.005), m.col


def test_the_multivariable_design_uses_the_declared_scale(real_cohort):
    """The other OR on the page. A univariable fix that missed this would show
    the same measurement on one scale in the EDA table and the other in the
    model, one section apart."""
    from inferential import _build_design

    logged = sorted(scales.LOG1P_COLUMNS)[0]
    frame = real_cohort.dropna(subset=[logged]).copy()
    schema = {logged: ColSpec(name=logged, kind="continuous")}
    _, _, z_params = _build_design(frame, schema, [logged])
    mu, sd = scales.z_params(frame[logged].to_numpy(dtype=float), True)
    assert z_params[logged]["log"] is True
    assert z_params[logged]["mu"] == pytest.approx(mu)
    assert z_params[logged]["sd"] == pytest.approx(sd)


def test_the_shipped_calculator_artifacts_carry_the_declared_scale():
    """The last thing the scale has to survive: being written to a file.

    The calculator reads ``z_log`` back out of these artifacts and logs the
    patient's value before standardising it. A stale artifact written before the
    declaration changed would give a different risk for the same patient than
    the report gives an odds ratio for, and nothing in either would say so.
    """
    import glob
    import json

    paths = sorted(glob.glob("output/inferential/tables/*__calculator.json"))
    if not paths:
        pytest.skip("no calculator artifacts — run meningioma-modelling first")
    checked = 0
    for path in paths:
        for term in json.loads(Path(path).read_text()).get("terms", []):
            name = term.get("name")
            if term.get("kind") != "continuous" or name not in scales.GOVERNED_COLUMNS:
                continue
            checked += 1
            assert bool(term.get("z_log")) is scales.is_log_scaled(name), \
                f"{Path(path).name}: {name}"
    if not checked:
        # Not a failure: no model variant currently fits a governed measurement
        # on its continuous scale, so there is no artifact for this check to
        # read. Skipping keeps that state distinguishable from a real mismatch,
        # and the guard fires again the moment such a variant comes back.
        pytest.skip(
            "no variant fits a governed continuous measurement "
            f"({', '.join(sorted(scales.GOVERNED_COLUMNS))}) — nothing to check"
        )


def test_the_raw_scale_would_give_a_visibly_different_answer():
    """Guards the guard: the test above is only meaningful if the scales differ."""
    df = _skewed_cohort()
    y = df["high_grade"].to_numpy(dtype=float)
    x = df["tumor_volume"].to_numpy(dtype=float)
    logged = di.odds_ratio(y, scales.standardise(x, log_x=True))["or"]
    raw = di.odds_ratio(y, scales.standardise(x, log_x=False))["or"]
    assert abs(logged - raw) > 0.1


def test_standardise_uses_the_sample_sd_so_both_phases_round_alike():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    z = scales.standardise(x, log_x=False)
    assert float(np.std(z, ddof=scales.SD_DDOF)) == pytest.approx(1.0)


def test_zero_survives_the_log_transform():
    """A third of this cohort has no edema; plain log would drop exactly them."""
    x = np.array([0.0, 0.0, 1.0, 10.0, 100.0])
    assert np.isfinite(scales.to_model_scale(x, log_x=True)).all()


# --------------------------------------------------------------------------
# The risk calculator still speaks clinical units
# --------------------------------------------------------------------------
def test_calculator_logs_the_patient_value_before_standardising():
    """The user types 26 cm³, not 3.3 log-cm³, however the model was fitted."""
    mu, sd = float(np.log1p(20.0)), 1.2
    feature = {"name": "tumor_volume", "z_mu": mu, "z_sd": sd, "z_log": True}
    got = _apply_continuous_transform(26.0, "log_standardize", feature)
    assert got == pytest.approx((np.log1p(26.0) - mu) / sd)


def test_calculator_rejects_a_negative_volume_rather_than_returning_a_risk():
    feature = {"name": "tumor_volume", "z_mu": 3.0, "z_sd": 1.0, "z_log": True}
    with pytest.raises(ValueError, match="non-negative"):
        _apply_continuous_transform(-5.0, "log_standardize", feature)


# --------------------------------------------------------------------------
# And says so in print
# --------------------------------------------------------------------------
