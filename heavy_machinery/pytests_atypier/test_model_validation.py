"""Tests for model_validation.py — bootstrap shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_calculator import calculator_meta_to_streamlit_artifact
import model_validation as mv
from model_validation import (
    bootstrap_internal_validation,
    enrich_streamlit_artifact,
)


@pytest.fixture
def tiny_model_df() -> tuple[pd.DataFrame, list[str], dict]:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "flag": rng.integers(0, 2, n),
    })
    design_cols = ["age", "flag"]
    coefficients = {"const": -0.5, "age": 0.05, "flag": 0.4}
    return df, design_cols, coefficients


@pytest.fixture
def separable_model_df() -> tuple[pd.DataFrame, list[str], dict]:
    """A frame where the predictor genuinely carries signal, unlike pure noise."""
    rng = np.random.default_rng(7)
    n = 300
    score = rng.normal(0, 1, n)
    event = (rng.uniform(size=n) < 1 / (1 + np.exp(-(score * 1.5 - 0.8)))).astype(int)
    df = pd.DataFrame({"event": event, "score": score})
    return df, ["score"], {"const": -0.8, "score": 1.5}


def test_calibration_and_decision_curve_on_a_model_with_real_signal(
    separable_model_df,
):
    """LOWESS robustness iterations flatten a binary outcome to zero — it=0 — and
    correcting only the slope reports half the calibration."""
    df, design_cols, coefficients = separable_model_df
    out = bootstrap_internal_validation(
        df, "event", design_cols, coefficients, n_bootstrap=60,
    )

    cal = out["calibration"]
    assert sum(b["n"] for b in cal["bins"]) == len(df)
    assert all(0.0 <= b["observed"] <= 1.0 for b in cal["bins"])
    assert all(b["events"] <= b["n"] for b in cal["bins"])
    assert cal["slope_apparent"] == 1.0
    assert "slope_corrected" in cal and "intercept_apparent" in cal

    smooth = cal["smooth"]
    assert smooth["predicted"], "no smoothed calibration curve"
    observed = np.asarray(smooth["observed"], dtype=float)
    # A model with real signal must not smooth to a flat line at zero.
    assert observed.max() > 0.4
    assert observed[-1] > observed[0]

    assert "intercept_corrected" in cal
    assert np.isfinite(cal["intercept_corrected"])
    # Apparent calibration-in-the-large on the development sample is 0 by
    # construction; the corrected value is what the model would do elsewhere.
    assert cal["intercept_apparent"] == pytest.approx(0.0, abs=0.01)

    row = next(m for m in out["metrics"] if m["metric"] == "Calibration intercept")
    assert row["optimism_corrected"] == cal["intercept_corrected"]
    assert row["apparent"] == cal["intercept_apparent"]

    dca = out["decision_curve"]
    assert len(dca["thresholds"]) == len(dca["model"]) == len(dca["treat_all"])
    prevalence = float(df["event"].mean())
    assert dca["prevalence"] == pytest.approx(prevalence, abs=5e-4)
    # Treat-all is prevalence − (1 − prevalence)·odds(t), by definition.
    t = dca["thresholds"][9]
    expected = prevalence - (1 - prevalence) * (t / (1 - t))
    assert dca["treat_all"][9] == pytest.approx(expected, abs=1e-3)
    # At a threshold below every predicted risk, the model equals treat-all.
    assert dca["model"][0] == pytest.approx(dca["treat_all"][0], abs=0.02)


def test_bootstrap_internal_validation(tiny_model_df):
    df, design_cols, coefficients = tiny_model_df
    out = bootstrap_internal_validation(
        df, "event", design_cols, coefficients, n_bootstrap=30,
    )
    assert out["method"] == "bootstrap internal validation"
    assert len(out["metrics"]) == 4
    assert [m["metric"] for m in out["metrics"]] == [
        "AUC", "Brier score", "Calibration slope", "Calibration intercept"]
    assert "roc_curves" in out
    assert out["roc_curves"]["curves"][0]["fpr"]


def test_enrich_streamlit_artifact_shrinks_every_kind_of_coefficient(
    tiny_model_df,
):
    rng = np.random.default_rng(0)
    n = 80
    age_bins = rng.integers(0, 5, size=n).astype(float)
    event = ((age_bins + rng.normal(0, 1.5, size=n)) > 2).astype(int)
    ordinal_df = pd.DataFrame({"event": event, "age_bins": age_bins})
    meta = {
        "target": "event",
        "intercept": -0.3,
        "terms": [
            {
                "name": "age_bins",
                "kind": "ordinal",
                "coef": 0.2,
                "levels": ["<50", "50-59", "60-69", "70-79", "80+"],
            },
        ],
    }
    artifact = calculator_meta_to_streamlit_artifact(
        meta, n=len(ordinal_df), events=int(ordinal_df["event"].sum()))
    enriched = enrich_streamlit_artifact(artifact, ordinal_df, ["age_bins"],
                                         n_bootstrap=20)
    shrinkage = enriched["coefficient_processing"]["shrinkage_factor"]
    assert enriched["coefficients"]["age_bins"] == pytest.approx(0.2 * shrinkage,
                                                                abs=0.0001)

    df, design_cols, coefficients = tiny_model_df
    meta = {
        "target": "event",
        "intercept": coefficients["const"],
        "terms": [
            {"name": "age", "kind": "continuous", "coef": 0.05, "z_mu": 60.0, "z_sd": 10.0},
            {"name": "flag", "kind": "binary", "coef": 0.4},
        ],
    }
    artifact = calculator_meta_to_streamlit_artifact(meta, n=len(df), events=int(df["event"].sum()))
    enriched = enrich_streamlit_artifact(artifact, df, design_cols, n_bootstrap=30)
    assert "validation" in enriched
    assert "coefficient_processing" in enriched
    assert enriched["coefficient_processing"]["shrinkage_applied"] is True
    assert "missing_data_policy" in enriched


def test_an_overfitted_model_loses_both_slope_and_intercept():
    """Eight noise predictors on 90 patients: the slope must fall well short of 1."""
    rng = np.random.default_rng(23)
    n, k = 90, 8
    noise = {f"z{i}": rng.normal(size=n) for i in range(k)}
    df = pd.DataFrame({**noise, "event": rng.integers(0, 2, n)})
    out = bootstrap_internal_validation(
        df, "event", list(noise), {}, n_bootstrap=60,
    )
    assert out["calibration"]["slope_corrected"] < 0.8
    assert np.isfinite(out["calibration"]["intercept_corrected"])


def test_bootstrap_default_comes_from_shared_config():
    import inspect
    import analysis  # heavy_machinery/config resolves bare per pytest.ini
    for fn in (bootstrap_internal_validation, enrich_streamlit_artifact):
        sig = inspect.signature(fn)
        assert sig.parameters["n_bootstrap"].default == analysis.BOOTSTRAP_RESAMPLES


# ---------------------------------------------------------------------------
# Master bootstrap seed and per-resample AUC vector
# ---------------------------------------------------------------------------
# Later phases difference two models' AUCs resample-by-resample, which is only
# a paired difference if both models were refit on the same index sets.

def test_resample_indices_is_a_deterministic_pure_function_of_shape():
    """``_resample_indices`` takes only ``(n_rows, n_bootstrap)`` — nothing
    else can influence it. This is the primitive the pairing guarantee (below)
    rests on, so it gets its own direct test rather than being inferred from
    a downstream AUC."""
    import model_validation as mv
    m1 = mv._resample_indices(157, 40)
    m2 = mv._resample_indices(157, 40)
    assert np.array_equal(m1, m2)
    assert m1.shape == (40, 157)
    assert m1.min() >= 0
    assert m1.max() < 157
    # A different shape must not silently collide with the one above.
    m3 = mv._resample_indices(157, 41)
    assert m3.shape == (41, 157)


def test_resample_aucs_are_returned_and_paired_across_models():
    """Two models validated on the same frame must use the same resample
    indices, or their AUC difference is not a paired difference."""
    import model_validation as mv
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "y": rng.binomial(1, 0.3, n).astype(float),
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    df["y"] = (df["a"] * 0.9 + rng.normal(size=n) > 0).astype(float)
    out_a = mv.bootstrap_internal_validation(
        df, "y", ["a"], {"const": 0.0, "a": 1.0},
        n_bootstrap=50, return_resample_aucs=True)
    out_b = mv.bootstrap_internal_validation(
        df, "y", ["a", "b"], {"const": 0.0, "a": 1.0, "b": 0.0},
        n_bootstrap=50, return_resample_aucs=True)
    assert len(out_a["resample_aucs"]) == len(out_b["resample_aucs"]) == 50
    # The pairing claim, checked directly rather than inferred from lengths:
    # both models were fit on the same ``df`` (same n_rows = len(df)) with the
    # same n_bootstrap, and ``_resample_indices`` has no other parameter it
    # could depend on — not design_cols, not call order. So the exact matrix
    # each model's loop indexed with is reproducible and provably identical.
    idx_for_a = mv._resample_indices(len(df), 50)
    idx_for_b = mv._resample_indices(len(df), 50)
    assert np.array_equal(idx_for_a, idx_for_b)
    # Same seed -> same index sets -> a rerun reproduces exactly.
    again = mv.bootstrap_internal_validation(
        df, "y", ["a"], {"const": 0.0, "a": 1.0},
        n_bootstrap=50, return_resample_aucs=True)
    assert out_a["resample_aucs"] == again["resample_aucs"]


def test_two_models_actually_call_resample_indices_with_the_same_arguments(
    monkeypatch,
):
    """Closes the gap the previous test leaves open: it's not enough that
    ``_resample_indices(same shape)`` returns the same matrix in isolation —
    ``bootstrap_internal_validation`` must actually call it that way for both
    models. Spies on the real call site so a future change that seeded by
    design_cols, coefficients, or call order (instead of n_rows/n_bootstrap
    alone) would fail this test even though it left ``_resample_indices``
    itself untouched."""
    import model_validation as mv

    real_resample_indices = mv._resample_indices
    calls: list[tuple[int, int, np.ndarray]] = []

    def spy(n_rows, n_bootstrap):
        matrix = real_resample_indices(n_rows, n_bootstrap)
        calls.append((n_rows, n_bootstrap, matrix.copy()))
        return matrix

    monkeypatch.setattr(mv, "_resample_indices", spy)

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "y": rng.binomial(1, 0.3, n).astype(float),
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    df["y"] = (df["a"] * 0.9 + rng.normal(size=n) > 0).astype(float)
    mv.bootstrap_internal_validation(
        df, "y", ["a"], {"const": 0.0, "a": 1.0}, n_bootstrap=50)
    mv.bootstrap_internal_validation(
        df, "y", ["a", "b"], {"const": 0.0, "a": 1.0, "b": 0.0}, n_bootstrap=50)

    assert len(calls) == 2
    (n_rows_a, n_boot_a, matrix_a), (n_rows_b, n_boot_b, matrix_b) = calls
    assert (n_rows_a, n_boot_a) == (n_rows_b, n_boot_b) == (200, 50)
    assert np.array_equal(matrix_a, matrix_b)


def test_bootstrap_seed_is_the_pipeline_seed():
    import model_validation as mv
    assert mv.BOOTSTRAP_SEED == 20260801


def test_select_hook_accepts_a_callable_that_always_agrees():
    """Task 6 wires ``select`` up: a selector that always returns the same
    column must reproduce the fixed-column result, and every resample counts
    toward that one variable."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
    })
    out = bootstrap_internal_validation(
        df, "event", ["age"], {"const": -0.5, "age": 0.05},
        n_bootstrap=20, select=lambda frame, y: ["age"],
    )
    assert out["method"] == "bootstrap internal validation"
    assert out["selection_counts"] == {"age": 20}


def test_select_hook_reselects_per_resample_and_counts_choices():
    """A model whose predictors are chosen from the data must have that choosing
    re-run inside the bootstrap, or its optimism correction misses the part that
    matters most."""
    import model_validation as mv
    rng = np.random.RandomState(3)
    n = 250
    y = rng.binomial(1, 0.4, n)
    df = pd.DataFrame({
        "y": y.astype(float),
        "strong": y * 1.3 + rng.normal(size=n),
        "weak": y * 0.15 + rng.normal(size=n),
        "noise": rng.normal(size=n),
    })

    def select(frame, y_arr):
        import variable_selection as vs
        picked, _ = vs.select_variables(
            frame, y_arr, ["strong", "weak", "noise"], k=1, rho_max=0.8)
        return picked

    out = mv.bootstrap_internal_validation(
        df, "y", ["strong"], {"const": 0.0, "strong": 1.0},
        n_bootstrap=40, return_resample_aucs=True, select=select)
    counts = out["selection_counts"]
    assert sum(counts.values()) == 40
    assert counts["strong"] > counts.get("noise", 0)
    assert len(out["resample_aucs"]) == 40


def test_resample_aucs_absent_unless_requested():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
    })
    out = bootstrap_internal_validation(
        df, "event", ["age"], {"const": -0.5, "age": 0.05}, n_bootstrap=20,
    )
    assert "resample_aucs" not in out


def test_a_resample_that_fails_to_fit_is_skipped_not_padded(monkeypatch):
    """One bad bootstrap draw must shrink ``resample_aucs`` and
    ``successful_bootstraps`` together, in lockstep — never padded with a NaN
    or 0.0 placeholder, which would silently corrupt a later paired
    comparison against another model's (differently-shrunk) vector.

    Forcing a genuine statsmodels convergence failure with synthetic data
    isn't reliable, so this monkeypatches ``sm.Logit`` to raise on exactly
    one resample's fit. The design matrix here has 3 columns (const + age +
    flag), a shape no calibration sub-fit ever produces (those always fit on
    1 or 2 columns), so counting occurrences of that shape safely identifies
    the apparent fit (call 1, must succeed) and each resample's boot fit
    (call 2 onward) without needing to track every internal statsmodels call.
    """
    import model_validation as mv

    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "event": rng.integers(0, 2, n),
        "age": rng.normal(60, 10, n),
        "flag": rng.integers(0, 2, n),
    })
    design_cols = ["age", "flag"]
    design_width = len(design_cols) + 1  # +1 for the prepended constant

    real_logit = mv.sm.Logit
    seen = {"design_shape_fits": 0}

    class BoomOnFirstResample(real_logit):
        def fit(self, *args, **kwargs):
            if self.exog.shape[1] == design_width:
                seen["design_shape_fits"] += 1
                if seen["design_shape_fits"] == 2:  # 1st is the apparent fit
                    raise RuntimeError("synthetic fit failure for testing")
            return super().fit(*args, **kwargs)

    monkeypatch.setattr(mv.sm, "Logit", BoomOnFirstResample)

    n_bootstrap = 15
    out = mv.bootstrap_internal_validation(
        df, "event", design_cols, {"const": -0.5, "age": 0.05, "flag": 0.4},
        n_bootstrap=n_bootstrap, return_resample_aucs=True,
    )

    assert out["successful_bootstraps"] == n_bootstrap - 1
    assert len(out["resample_aucs"]) == n_bootstrap - 1
    assert len(out["resample_aucs"]) == out["successful_bootstraps"]
    # No NaN and no 0.0 placeholder standing in for the skipped resample.
    assert all(np.isfinite(v) for v in out["resample_aucs"])
    assert all(v > 0.0 for v in out["resample_aucs"])


# ---------------------------------------------------------------------------
# Parallel validation policy
# ---------------------------------------------------------------------------
# Running the model validations side by side must stay a wall-clock change and
# nothing else, and it must stay polite on a laptop.

def test_the_worker_count_is_capped_polite_and_overridable(monkeypatch):
    """Unplugged, spinning up every core is exactly what empties the battery."""
    monkeypatch.delenv(mv.WORKERS_ENV, raising=False)
    monkeypatch.setattr(mv, "_on_battery", lambda: False)
    assert mv.validation_workers(1) == 1
    assert mv.validation_workers(0) == 1

    monkeypatch.setattr(mv, "_on_battery", lambda: True)
    assert mv.validation_workers(7) == 1

    monkeypatch.setattr(mv, "_on_battery", lambda: False)
    monkeypatch.setattr(mv.os, "cpu_count", lambda: 64)
    assert mv.validation_workers(50) == mv.MAX_VALIDATION_WORKERS
    assert mv.validation_workers(2) == 2      # never more than there are models

    monkeypatch.setattr(mv.os, "cpu_count", lambda: 4)
    assert mv.validation_workers(50) == 4 - mv.RESERVED_CORES

    monkeypatch.setattr(mv.os, "cpu_count", lambda: 2)
    assert mv.validation_workers(50) == 1  # never zero or negative

    # The environment override wins even over the battery check.
    monkeypatch.setattr(mv, "_on_battery", lambda: True)
    monkeypatch.setenv(mv.WORKERS_ENV, "3")
    assert mv.validation_workers(7) == 3
    monkeypatch.setenv(mv.WORKERS_ENV, "1")
    assert mv.validation_workers(7) == 1

    monkeypatch.setenv(mv.WORKERS_ENV, "lots")
    with pytest.raises(ValueError, match="whole number of workers"):
        mv.validation_workers(7)


def test_parallel_and_sequential_validation_agree(tiny_model_df, monkeypatch):
    """The published numbers must not depend on how many processes ran."""
    df, design_cols, _ = tiny_model_df
    meta = {
        "target": "event",
        "intercept": -0.5,
        "terms": [{"name": c, "kind": "binary", "coef": 0.3} for c in design_cols],
    }
    art = calculator_meta_to_streamlit_artifact(meta, n=len(df), events=None)
    jobs = [(art, df, design_cols), (art, df, design_cols)]

    monkeypatch.setenv(mv.WORKERS_ENV, "1")
    seq = mv.enrich_streamlit_artifacts(jobs, n_bootstrap=40)
    monkeypatch.setenv(mv.WORKERS_ENV, "2")
    par = mv.enrich_streamlit_artifacts(jobs, n_bootstrap=40)

    assert len(seq) == len(par) == 2
    assert [s["validation"] for s in seq] == [p["validation"] for p in par]


def test_a_model_that_cannot_be_validated_keeps_its_plain_artifact(monkeypatch):
    """One unfittable model must not take the other six down with it."""
    monkeypatch.setenv(mv.WORKERS_ENV, "1")

    def boom(*args, **kwargs):
        raise RuntimeError("Bootstrap internal validation failed for all resamples")

    monkeypatch.setattr(mv, "enrich_streamlit_artifact", boom)
    art = {"target": "event", "coefficients": {}}
    df = pd.DataFrame({"event": [1, 0], "x": [1.0, 2.0]})
    assert mv.enrich_streamlit_artifacts([(art, df, ["x"])], n_bootstrap=5) == [art]

    assert mv.enrich_streamlit_artifacts([]) == []
