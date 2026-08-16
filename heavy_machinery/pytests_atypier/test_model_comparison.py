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
