"""Tests for variable_selection.py — AUC ranking and the two guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import variable_selection as vs


@pytest.fixture
def toy():
    rng = np.random.RandomState(7)
    n = 300
    y = rng.binomial(1, 0.4, n)
    size = y * 1.2 + rng.normal(size=n)
    return pd.DataFrame({
        "size": size,
        "size_copy": size + rng.normal(scale=0.05, size=n),   # rho ~ 0.99
        "size_ge0": (size >= 0).astype(float),                 # cut-point child
        "edema": y * 0.6 + rng.normal(size=n),
        "protective": -(y * 0.9) + rng.normal(size=n),
        "noise": rng.normal(size=n),
    }), y


def test_discrimination_flips_a_protective_auc():
    assert vs.discrimination(0.37) == pytest.approx(0.63)
    assert vs.discrimination(0.68) == pytest.approx(0.68)


def test_protective_variable_is_ranked_on_discrimination_not_raw_auc(toy):
    df, y = toy
    ranked = vs.rank_candidates(df, y, ["protective", "noise"])
    assert ranked[0][0] == "protective"
    assert ranked[0][1] < 0.5          # raw AUC really is below 0.5


def test_collinear_candidate_is_dropped_and_the_next_one_taken(toy):
    df, y = toy
    picked, audit = vs.select_variables(
        df, y, ["size", "size_copy", "edema", "noise"], k=2, rho_max=0.8)
    assert picked == ["size", "edema"]
    dropped = {r["variable"]: r["reason"] for r in audit if not r["kept"]}
    assert "size_copy" in dropped
    assert "rho=" in dropped["size_copy"] and "size" in dropped["size_copy"]


def test_cutpoint_child_is_dropped_when_its_parent_is_a_candidate(toy):
    df, y = toy
    picked, audit = vs.select_variables(
        df, y, ["size", "size_ge0", "edema"], k=3, rho_max=0.99,
        cutpoint_parent={"size_ge0": "size"})
    assert "size_ge0" not in picked
    reason = next(r["reason"] for r in audit if r["variable"] == "size_ge0")
    assert "cut-point of size" in reason


def test_audit_records_every_candidate_considered(toy):
    df, y = toy
    picked, audit = vs.select_variables(df, y, list(df.columns), k=2, rho_max=0.8)
    assert len(picked) == 2
    assert {r["variable"] for r in audit} <= set(df.columns)
    assert all(set(r) == {"variable", "auc", "discrimination", "kept", "reason"}
               for r in audit)
    # A candidate ranked below the k-th pick is never reached by the walk, and
    # "not reached" must mean "not audited" — otherwise the audit trail claims
    # to have considered variables it never actually looked at.
    rank_order = [c for c, _ in vs.rank_candidates(df, y, list(df.columns))]
    audited = {r["variable"] for r in audit}
    last_reached = max(rank_order.index(v) for v in audited)
    never_reached = rank_order[last_reached + 1:]
    assert never_reached, "toy fixture must leave something unreached for this to mean anything"
    assert audited.isdisjoint(never_reached)


def test_categorical_candidate_is_skipped_and_recorded_as_not_numeric(toy):
    df, y = toy
    df = df.assign(tumor_location=(["skull_base", "convexity", "parasagittal"] * 100))
    picked, audit = vs.select_variables(
        df, y, ["size", "tumor_location", "noise"], k=2, rho_max=0.8)
    assert "tumor_location" not in picked
    row = next(r for r in audit if r["variable"] == "tumor_location")
    assert row["kept"] is False
    assert "not numeric" in row["reason"]
    assert np.isnan(row["auc"])
    assert np.isnan(row["discrimination"])


def test_absent_and_constant_candidates_are_skipped_without_crashing(toy):
    df, y = toy
    df = df.assign(constant_col=1.0)
    picked, audit = vs.select_variables(
        df, y, ["size", "edema", "missing_column", "constant_col"],
        k=2, rho_max=0.8)
    assert picked == ["size", "edema"]
    audited_vars = {r["variable"] for r in audit}
    assert "missing_column" not in audited_vars
    assert "constant_col" not in audited_vars


def test_column_vector_does_not_transform_a_log_scaled_column():
    """``_column_vector`` used to re-apply ``log1p(clip(x, 0, None))`` to any
    column named in ``scales.LOG1P_COLUMNS`` (e.g. ``tumor_volume``),
    regardless of what scale the values already carried. Both of its
    consumers, ``roc_auc_score`` in ``rank_candidates`` and the Spearman
    correlation in the collinearity guard, are rank-based, so a monotone
    transform like log1p is invisible to them on genuinely raw input — it was
    always a no-op there, never a needed correction.

    But a caller building a design matrix (``inferential._build_design``) has
    ALREADY log1p'd and z-scored these columns before fitting. Applying
    log1p a second time to an already mean-centred column means clipping
    every below-mean (negative z-score) patient to a tied 0.0, corrupting the
    ranking — measured at 180/352 patients (51%) for ``tumor_volume`` in the
    real cohort (Task 13 review round 3, Finding 4). ``_column_vector`` must
    return the column exactly as given, on any scale.
    """
    raw = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    df = pd.DataFrame({"tumor_volume": raw})
    assert vs._column_vector(df, "tumor_volume") == pytest.approx(raw)

    # A model-scale (already log1p'd and z-scored, i.e. mean ~0) version of
    # the same kind of column must also come back unchanged — not clipped.
    already_z_scored = np.array([-1.4, -0.6, -0.1, 0.4, 1.7])
    df2 = pd.DataFrame({"tumor_volume": already_z_scored})
    assert vs._column_vector(df2, "tumor_volume") == pytest.approx(already_z_scored)


def test_rank_candidates_gives_the_same_auc_whether_a_log_scaled_column_is_raw_or_already_model_scaled():
    """The regression test for Finding 4. ``rank_candidates`` (and therefore
    the collinearity guard, which reads from the same ``_column_vector``)
    must agree on a log-scaled column's AUC whether it is handed the raw
    measurement or that same measurement already log1p'd-and-z-scored, the
    way a modelling-phase design matrix carries it — because AUC and Spearman
    rho are both rank-based and log1p is a strictly monotone transform, so it
    can never change either. That invariant is exactly what broke when
    ``_column_vector`` re-applied log1p (via ``clip(x, 0, None)``) to input
    that was already on the model scale.
    """
    rng = np.random.RandomState(11)
    n = 250
    y = rng.binomial(1, 0.4, n)
    # A real tumor_volume is strictly positive, as scales.to_model_scale requires.
    raw = y * 3.0 + rng.normal(scale=2.0, size=n) + 10.0
    assert (raw > -1.0).all()

    noise = rng.normal(size=n)
    df_raw = pd.DataFrame({"tumor_volume": raw, "noise": noise})

    log_scaled = np.log1p(raw)
    z = (log_scaled - log_scaled.mean()) / log_scaled.std(ddof=1)
    df_model_scale = pd.DataFrame({"tumor_volume": z, "noise": noise})

    auc_raw = dict(vs.rank_candidates(df_raw, y, ["tumor_volume", "noise"]))["tumor_volume"]
    auc_model_scale = dict(
        vs.rank_candidates(df_model_scale, y, ["tumor_volume", "noise"])
    )["tumor_volume"]
    assert auc_raw == pytest.approx(auc_model_scale)
    # Sanity: a real, non-degenerate signal either way, not a coincidental tie.
    assert 0.5 < auc_raw < 1.0


def test_assert_reference_passes_when_the_declared_variable_is_the_top_pick():
    vs.assert_reference(["tumor_volume", "adc_value", "edema_volume_cm3"])


def test_assert_reference_raises_when_something_else_is_picked_first():
    with pytest.raises(ValueError, match="reference variable"):
        vs.assert_reference(["max_diameter_cm", "tumor_volume"])


def test_assert_reference_raises_on_an_empty_pick():
    with pytest.raises(ValueError, match="reference variable"):
        vs.assert_reference([])


def test_assert_reference_without_audit_skips_the_discrimination_drift_check():
    """Backward compatible: no audit means no drift check, only the top-pick
    check above -- existing call sites that never pass one keep working."""
    vs.assert_reference(["tumor_volume", "adc_value"], None)


def test_assert_reference_passes_when_discrimination_matches_the_declared_value():
    """Item 6, final whole-branch review: analysis.REFERENCE_VARIABLE_DISCRIMINATION
    (0.679) is wired into assert_reference as a drift check rather than sitting
    unread. An audit row within tolerance of the declared value must not raise."""
    from heavy_machinery.config import load
    declared = load("analysis").REFERENCE_VARIABLE_DISCRIMINATION
    audit = [{"variable": "tumor_volume", "discrimination": declared, "kept": True}]
    vs.assert_reference(["tumor_volume", "adc_value"], audit)


def test_assert_reference_raises_when_reference_discrimination_drifts():
    """A real change in the reference's underlying strength (a data fix, a
    derivation bug) must not pass silently just because it is still ranked
    first -- the declared 0.679 has to mean something once it is checked."""
    from heavy_machinery.config import load
    declared = load("analysis").REFERENCE_VARIABLE_DISCRIMINATION
    audit = [{"variable": "tumor_volume", "discrimination": declared + 0.05,
              "kept": True}]
    with pytest.raises(ValueError, match="REFERENCE_VARIABLE_DISCRIMINATION"):
        vs.assert_reference(["tumor_volume", "adc_value"], audit)


def test_assert_reference_skips_the_drift_check_when_the_reference_has_no_audit_row():
    """An audit that never scored the reference at all (e.g. a hand-built
    partial audit) must not raise on a check it has no data for -- that is a
    silent no-op, not a false positive."""
    audit = [{"variable": "adc_value", "discrimination": 0.630, "kept": True}]
    vs.assert_reference(["tumor_volume", "adc_value"], audit)
