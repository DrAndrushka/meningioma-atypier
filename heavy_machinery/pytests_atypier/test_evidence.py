"""The graded evidence hierarchy and the multiple-testing family.

The point of these tests is that the grade is *derived*, not asserted. Each
one builds a risk-curve row that fails exactly one criterion and checks that
the grade drops the way the pre-specified rules say it should — so a future
change to the rules cannot quietly re-grade a published verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evidence as ev


def risk_row(**overrides) -> dict:
    """A row that passes all five criteria, before overrides."""
    row = {
        "metric": "Volume", "column": "vol", "AUC": 0.70,
        "nonlinearity_p": 0.001,
        "nonlinear": True,
        "nonlinearity_p_log_scale": 0.002,
        "steepest_x": 10.0,
        "steepest_lo": 8.0,
        "steepest_hi": 12.0,
        "steepest_pct_of_patients": 45.0,
        "risk_50_x": 40.0,
        "risk_50_lo": 35.0,
        "risk_50_hi": 50.0,
        "risk_50_found_frac": 0.9,
    }
    row.update(overrides)
    return row


def evidence_for(knee_rate: float = 1.0, **overrides) -> pd.Series:
    risk = pd.DataFrame([risk_row(**overrides)])
    stab = pd.DataFrame([{"column": "vol", "knee_rate": knee_rate}])
    return ev.threshold_evidence(risk, stab).iloc[0]


# --------------------------------------------------------------------------
# Holm and Bonferroni
# --------------------------------------------------------------------------
def test_holm_is_step_down_and_monotone():
    p = [0.01, 0.02, 0.03, 0.04]
    adj = ev.holm_adjust(p)
    # 0.01*4, then max(previous, 0.02*3), then max(previous, 0.03*2), ...
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert np.all(np.diff(adj) >= 0)


def test_holm_never_exceeds_one_and_beats_bonferroni():
    p = [0.4, 0.5, 0.6]
    holm = ev.holm_adjust(p)
    bonf = ev.bonferroni_adjust(p)
    assert holm.max() <= 1.0 and bonf.max() <= 1.0
    assert np.all(holm <= bonf + 1e-12)


def test_holm_passes_nan_through():
    adj = ev.holm_adjust([0.01, np.nan, 0.02])
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_multiplicity_table_flags_the_disagreement():
    risk = pd.DataFrame([
        risk_row(metric="a", column="a", nonlinearity_p=0.007),
        risk_row(metric="b", column="b", nonlinearity_p=0.047),
    ])
    tab = ev.multiplicity_table(risk)
    assert list(tab["survives_holm"]) == [True, True]
    # 0.047 * 2 = 0.094 — Bonferroni drops it, Holm does not.
    assert list(tab["survives_bonferroni"]) == [True, False]


# --------------------------------------------------------------------------
# The grade mapping
# --------------------------------------------------------------------------
def test_all_criteria_met_is_strong():
    assert evidence_for()["verdict"] == ev.GRADE_STRONG


def test_one_robustness_failure_is_moderate():
    row = evidence_for(nonlinearity_p_log_scale=0.9)
    assert row["verdict"] == ev.GRADE_MODERATE
    assert row["limiting_criterion"] == "Scale robustness"


def test_both_robustness_failures_are_fragile():
    risk = pd.DataFrame([risk_row(nonlinearity_p_log_scale=0.9)])
    stab = pd.DataFrame([{"column": "vol", "knee_rate": 0.2}])
    assert ev.threshold_evidence(risk, stab).iloc[0]["verdict"] == ev.GRADE_FRAGILE


def test_one_necessary_failure_is_fragile():
    # Knee at the 2nd percentile — interior test fails, curvature still holds.
    row = evidence_for(steepest_pct_of_patients=2.0)
    assert row["verdict"] == ev.GRADE_FRAGILE
    assert row["limiting_criterion"] == "Knee interiority"


def test_failed_curvature_is_weak_whatever_else_passes():
    row = evidence_for(nonlinearity_p=0.4, nonlinear=False)
    assert row["verdict"] == ev.GRADE_WEAK


def test_two_necessary_failures_are_weak():
    row = evidence_for(steepest_pct_of_patients=2.0, steepest_x=40.0)
    assert row["verdict"] == ev.GRADE_WEAK


# --------------------------------------------------------------------------
# The individual criteria
# --------------------------------------------------------------------------
def test_knee_inside_the_50pc_interval_fails_that_criterion():
    row = evidence_for(steepest_x=40.0)   # sits inside 35–50
    assert not row["pass_distinct_from_50"]
    assert "inside it" in row["detail_distinct_from_50"]


def test_unreached_50pc_crossing_cannot_be_restated():
    row = evidence_for(risk_50_x=np.nan, risk_50_lo=np.nan, risk_50_hi=np.nan)
    assert row["pass_distinct_from_50"]


def test_rarely_located_crossing_is_not_used_as_a_yardstick():
    row = evidence_for(steepest_x=40.0, risk_50_found_frac=0.05)
    assert row["pass_distinct_from_50"]
    assert "5% of resamples" in row["detail_distinct_from_50"]


def test_mice_cut_is_the_pre_specified_one():
    risk = pd.DataFrame([risk_row()])
    just_under = ev.threshold_evidence(
        risk, pd.DataFrame([{"column": "vol",
                             "knee_rate": ev.MICE_REPRODUCIBLE_CUT - 0.01}]))
    just_over = ev.threshold_evidence(
        risk, pd.DataFrame([{"column": "vol", "knee_rate": ev.MICE_REPRODUCIBLE_CUT}]))
    assert not just_under.iloc[0]["pass_mice_reproducible"]
    assert just_over.iloc[0]["pass_mice_reproducible"]


def test_missing_stability_table_fails_the_mice_criterion_rather_than_passing_it():
    row = ev.threshold_evidence(pd.DataFrame([risk_row()]), None).iloc[0]
    assert not row["pass_mice_reproducible"]
    assert row["verdict"] == ev.GRADE_MODERATE


def test_verdict_note_names_every_failing_criterion():
    risk = pd.DataFrame([risk_row(nonlinearity_p_log_scale=0.9)])
    stab = pd.DataFrame([{"column": "vol", "knee_rate": 0.2}])
    note = ev.threshold_evidence(risk, stab).iloc[0]["verdict_note"]
    assert "Scale robustness fails" in note
    assert "MICE reproducibility fails" in note


def test_precision_columns_are_reported_but_do_not_score():
    wide = evidence_for(steepest_lo=1.0, steepest_hi=100.0)
    assert wide["knee_ci_ratio"] == pytest.approx(100.0)
    assert wide["verdict"] == ev.GRADE_STRONG   # precision is context, not a gate


def test_empty_input_gives_empty_output():
    assert ev.threshold_evidence(pd.DataFrame()).empty
    assert ev.multiplicity_table(pd.DataFrame()).empty


def test_reading_view_has_one_column_per_criterion():
    view = ev.evidence_reading_view(ev.threshold_evidence(
        pd.DataFrame([risk_row()]), pd.DataFrame([{"column": "vol", "knee_rate": 1.0}])))
    for criterion in ev.CRITERIA:
        assert criterion.name in view.columns
    assert view.loc[0, "Criteria met"] == "5 of 5"
