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


def test_d2_pool_matches_the_r_reference_implementation():
    """Pins the formula against miceadds::micombine.chisquare (R), which
    implements Li, Meng, Raghunathan & Rubin 1991 directly. These numbers
    are external ground truth — if this test fails, the formula changed,
    not the reference."""
    out = mc.d2_pool([5.0, 12.0, 3.0, 20.0, 7.0], k=3)
    assert out["statistic"] == pytest.approx(0.4324380610, rel=1e-6)
    assert out["df_den"] == pytest.approx(6.089180, rel=1e-6)
    assert out["p"] == pytest.approx(0.7374592858, rel=1e-6)


def test_fit_single_predictors_returns_one_entry_per_predictor(tiny_schema):
    import numpy as np, pandas as pd
    from schema_infer import ColSpec
    rng = np.random.RandomState(11)
    n = 200
    y = rng.binomial(1, 0.35, n)
    df = pd.DataFrame({
        "event": y.astype(bool),
        "a": y * 1.1 + rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    schema = {"event": ColSpec("event", "binary"),
              "a": ColSpec("a", "continuous"), "b": ColSpec("b", "continuous")}
    out = mc.fit_single_predictors(df, schema, ["a", "b"], "event", n_bootstrap=30)
    assert set(out) == {"a", "b"}
    assert out["a"]["auc_corrected"] > out["b"]["auc_corrected"]
    assert len(out["a"]["resample_aucs"]) == 30
    assert out["a"]["n"] == n


def test_bootstrap_auc_vector_varies_and_is_deterministic():
    """A fixed model scored on resampled patients: the AUC must move, because
    the patients moved. This is what the optimism vector cannot give us for a
    single predictor, whose refit AUC is constant."""
    rng = np.random.RandomState(4)
    n = 200
    y = rng.binomial(1, 0.4, n)
    pred = y * 0.5 + rng.uniform(size=n)
    v1 = mc.bootstrap_auc_vector(y, pred, n_bootstrap=50)
    v2 = mc.bootstrap_auc_vector(y, pred, n_bootstrap=50)
    assert len(v1) == 50
    assert v1 == v2                      # same seed, same answer
    assert len(set(v1)) > 10             # genuinely varies


def test_bootstrap_auc_vector_is_paired_across_two_models():
    """Two models scored over the same index matrix, so element i of each
    vector is the same set of patients — which is what makes the difference
    paired."""
    import model_validation as mv
    rng = np.random.RandomState(5)
    n = 150
    y = rng.binomial(1, 0.4, n)
    good = y * 1.0 + rng.uniform(size=n)
    bad = rng.uniform(size=n)
    idx = mv._resample_indices(n, 20)
    va = mc.bootstrap_auc_vector(y, good, n_bootstrap=20)
    vb = mc.bootstrap_auc_vector(y, bad, n_bootstrap=20)
    from sklearn.metrics import roc_auc_score
    assert va[0] == pytest.approx(roc_auc_score(y[idx[0]], good[idx[0]]), rel=1e-9)
    assert vb[0] == pytest.approx(roc_auc_score(y[idx[0]], bad[idx[0]]), rel=1e-9)
    d = mc.paired_delta_auc(va, vb)
    assert d["ci_lo"] > 0                # the good model really is better


def test_bootstrap_auc_vector_skips_a_single_class_resample():
    """A resample can contain only one outcome class; AUC is undefined there.
    Those resamples are dropped, not recorded as 0.5."""
    y = np.array([1] + [0] * 39)
    pred = np.arange(40, dtype=float)
    v = mc.bootstrap_auc_vector(y, pred, n_bootstrap=200)
    assert len(v) < 200
    assert all(np.isfinite(x) for x in v)


def test_fit_single_predictors_returns_predictions_for_the_ci():
    import pandas as pd
    from schema_infer import ColSpec
    rng = np.random.RandomState(12)
    n = 120
    y = rng.binomial(1, 0.4, n)
    df = pd.DataFrame({"event": y.astype(bool), "a": y * 1.1 + rng.normal(size=n)})
    schema = {"event": ColSpec("event", "binary"), "a": ColSpec("a", "continuous")}
    out = mc.fit_single_predictors(df, schema, ["a"], "event", n_bootstrap=20)
    assert len(out["a"]["pred"]) == n
    assert len(out["a"]["y"]) == n
    assert set(out["a"]["y"]) <= {0, 1}
