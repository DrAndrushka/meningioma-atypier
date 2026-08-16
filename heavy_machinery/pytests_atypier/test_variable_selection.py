"""Tests for variable_selection.py — AUC ranking and the two guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
