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


def test_log_scaled_column_is_log1p_transformed_before_scoring():
    rng = np.random.RandomState(11)
    n = 250
    y = rng.binomial(1, 0.4, n)
    # includes plenty of negative values so clip(...,0,None) + log1p actually
    # diverges from the raw scale (a strictly monotonic transform alone would
    # leave both AUC and Spearman rho unchanged, since both are rank-based)
    raw = y * 3 + rng.normal(scale=2.0, size=n) - 1.0
    df = pd.DataFrame({"tumor_volume": raw, "noise": rng.normal(size=n)})

    expected_vector = np.log1p(np.clip(raw, 0.0, None))
    assert vs._column_vector(df, "tumor_volume") == pytest.approx(expected_vector)

    expected_auc = roc_auc_score(y, expected_vector)
    raw_auc = roc_auc_score(y, raw)
    assert expected_auc != pytest.approx(raw_auc)  # sanity: the clip really changes ranks

    ranked = vs.rank_candidates(df, y, ["tumor_volume", "noise"])
    assert dict(ranked)["tumor_volume"] == pytest.approx(expected_auc)


def test_log_scaled_column_feeds_its_transformed_vector_to_the_collinearity_guard():
    rng = np.random.RandomState(11)
    n = 250
    y = rng.binomial(1, 0.4, n)
    raw = y * 3 + rng.normal(scale=2.0, size=n) - 1.0
    transformed = np.log1p(np.clip(raw, 0.0, None))
    # near-identical to the TRANSFORMED vector, not the raw one — "clone" is
    # not itself a log-scaled column, so it is compared as-is. If the
    # collinearity guard ever compared raw tumor_volume against this clone
    # instead of its log1p'd vector, the reported rho would be measurably
    # different (0.91 vs 0.95 at this seed), not just numerically identical.
    clone = transformed + rng.normal(scale=1e-3, size=n)
    df = pd.DataFrame({"tumor_volume": raw, "tumor_volume_clone": clone})

    picked, audit = vs.select_variables(
        df, y, ["tumor_volume", "tumor_volume_clone"], k=2, rho_max=0.8)
    assert len(picked) == 1
    dropped = next(r for r in audit if not r["kept"])
    reported_rho = float(dropped["reason"].split("rho=")[1].split(" ")[0])

    rho_if_transformed = abs(pd.Series(transformed).corr(pd.Series(clone), method="spearman"))
    rho_if_raw = abs(pd.Series(raw).corr(pd.Series(clone), method="spearman"))
    assert round(rho_if_transformed, 2) != round(rho_if_raw, 2), (
        "fixture must make the two candidate vectors distinguishable for this "
        "test to mean anything"
    )
    assert reported_rho == pytest.approx(round(rho_if_transformed, 2))
    assert reported_rho != pytest.approx(round(rho_if_raw, 2))
