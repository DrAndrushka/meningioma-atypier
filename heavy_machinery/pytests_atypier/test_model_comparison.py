"""Tests for model_comparison.py — paired delta-AUC and D2 pooling."""
from __future__ import annotations

import numpy as np
import pytest

import model_comparison as mc


def test_paired_delta_auc_is_the_mean_of_within_resample_differences():
    a = [0.80, 0.82, 0.78, 0.81]
    b = [0.70, 0.75, 0.71, 0.72]
    out = mc.paired_delta_auc(a, b)
    assert out["delta"] == pytest.approx(np.mean(np.array(a) - np.array(b)))
    assert out["n_resamples"] == 4
    assert out["ci_lo"] <= out["delta"] <= out["ci_hi"]


def test_paired_delta_auc_ci_excludes_zero_when_one_model_always_wins():
    rng = np.random.RandomState(0)
    b = rng.uniform(0.60, 0.70, 500)
    a = b + 0.08
    out = mc.paired_delta_auc(a, b)
    assert out["ci_lo"] > 0


def test_paired_delta_auc_rejects_unpaired_vectors():
    with pytest.raises(ValueError, match="same number of resamples"):
        mc.paired_delta_auc([0.7, 0.8], [0.7])


def test_d2_pool_returns_a_p_value_between_zero_and_one():
    out = mc.d2_pool([8.0, 9.0, 7.5, 8.5], k=2)
    assert 0.0 <= out["p"] <= 1.0
    assert out["df_num"] == 2


def test_d2_pool_is_significant_for_large_consistent_chi_squares():
    out = mc.d2_pool([30.0] * 20, k=1)
    assert out["p"] < 0.001


def test_d2_pool_is_not_significant_for_small_chi_squares():
    out = mc.d2_pool([0.1] * 20, k=1)
    assert out["p"] > 0.5


def test_d2_pool_rejects_an_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        mc.d2_pool([], k=1)
