"""Tests for inferential.py — Rubin pooling, variants, artifact cleanup."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import inferential as inf
from inferential import (
    _empty_inferential_df,
    _safe_z_denominator,
    artifact_base,
    fit_multivariable_logistic,
    model_key,
    normalize_inferential_variants,
    run_inferential,
)
from schema_infer import ColSpec


def test_pool_df_for_display():
    assert inf._pool_df_for_display(float("inf")) == "∞"
    assert inf._pool_df_for_display(12.0) == 12


def test_format_inferential_table():
    df = pd.DataFrame({"df": [5.0, float("inf")], "p": [0.01, 0.05]})
    out = inf._format_inferential_table(df)
    assert "df" in out.columns


def test_safe_z_denominator():
    assert _safe_z_denominator(0.0) == 1.0


def test_ensure_dirs(tmp_output):
    figs, tabs = inf._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()


def test_build_design(tiny_df, tiny_schema):
    X, mapping, z_params = inf._build_design(tiny_df, tiny_schema, ["age", "sex"])
    assert "age" in mapping
    assert not X.empty
    assert "age" in z_params
    assert "mu" in z_params["age"] and "sd" in z_params["age"]


def test_prune_by_vif(tiny_df, tiny_schema):
    X, _, _ = inf._build_design(tiny_df, tiny_schema, ["age"])
    pruned, vif_df = inf._prune_by_vif(X, threshold=5.0)
    assert list(pruned.columns) == list(X.columns)
    assert "vif" in vif_df.columns
    assert vif_df["vif"].notna().all()


def test_prune_by_vif_with_nan_rows(tiny_schema):
    """VIF must use complete cases when design matrix has NaN (e.g. unimputed binary)."""
    X = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan],
        "b": [1.0, 0.0, 1.0, 1.0],
    })
    _, vif_df = inf._prune_by_vif(X, threshold=5.0)
    assert vif_df["vif"].notna().all()
    assert (vif_df["vif"] >= 1.0).all()


def test_rubin_pool():
    thetas = np.array([0.1, 0.2, 0.15])
    ses = np.array([0.05, 0.05, 0.05])
    pooled = inf._rubin_pool(thetas, ses)
    assert np.isfinite(pooled["coef"])
    assert "or" in pooled


def test_target_is_binary():
    y = pd.Series([True, False, True], dtype="boolean")
    assert inf._target_is_binary(y, ColSpec("event", "binary"))


def test_encode_target():
    y = pd.Series([True, False, True])
    enc, pos = inf._encode_target(y, True)
    assert pos is True
    assert enc.iloc[0] == 1.0


def _make_imputed(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    schema = {k: v for k, v in tiny_schema.items() if k in df.columns}
    return [df], schema


def test_fit_logit_robust(tiny_df):
    sub = tiny_df.dropna(subset=["age", "event"])
    Xc = sm.add_constant(sub[["age"]].astype(float), has_constant="add")
    model = inf._fit_logit_robust(sub["event"].astype(float), Xc)
    assert model is not None
    assert inf._logit_converged(model)


def test_fit_logit_robust_retries_after_stalled_newton():
    y = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    Xc = sm.add_constant(pd.DataFrame({"x": [10.0, 10.0, 10.0, -10.0, -10.0, -10.0]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = inf._fit_logit_robust(y, Xc)
    assert model is not None
    assert not any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )


def test_fit_multivariable_logistic(tiny_df, tiny_schema):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    pooled, vif = fit_multivariable_logistic(
        frames, schema, "event", ["age"],
        positive_class=True,
    )
    assert "or" in pooled.columns
    assert "vif" in vif.columns
    assert np.isfinite(pooled.loc[0, "z_mu"])
    assert np.isfinite(pooled.loc[0, "z_sd"])


def test_forest_plot(tmp_path):
    pooled = pd.DataFrame({
        "predictor_col": ["age"],
        "or": [1.5],
        "or_ci_lo": [0.8],
        "or_ci_hi": [2.5],
    })
    figs = tmp_path / "figs"
    figs.mkdir()
    inf._forest_plot(pooled, "event", figs)
    assert (figs / "event__forest.svg").exists()


def test_empty_inferential_df():
    df = _empty_inferential_df()
    assert list(df.columns)


def test_summarize_multivariable_cases(tiny_df, tiny_schema):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    summary = inf.summarize_multivariable_cases(
        frames[0], schema, targets=["event"], predictors=["age"],
        positive_class={"event": True},
    )
    assert summary.iloc[0]["n_complete_cases"] == len(frames[0])
    assert summary.iloc[0]["n_outcome_events"] >= 1


def test_summarize_multivariable_cases_experimental_flag(tiny_df, tiny_schema):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    variants = [
        inf.InferentialModelVariant(
            "lit", "Literature", "", "event", ("age",), experimental=False,
        ),
        inf.InferentialModelVariant(
            "try_hard_model", "Try hard", "", "event", ("sex",), experimental=True,
        ),
    ]
    summary = inf.summarize_multivariable_cases(
        frames[0], schema, targets=["event"], variants=variants,
        positive_class={"event": True},
    )
    by_id = dict(zip(summary["model_id"], summary["experimental"]))
    assert by_id["lit"] is False
    assert by_id["try_hard_model"] is True


def test_normalize_inferential_variants_tuple():
    vars_ = normalize_inferential_variants(
        variants=[("bondo_et_al", "Bondo et al.", "https://example.com/bondo", "high_grade", ["age", "sex"])],
    )
    assert vars_[0].model_id == "bondo_et_al"
    assert vars_[0].title == "Bondo et al."
    assert vars_[0].target == "high_grade"
    assert vars_[0].link == "https://example.com/bondo"


def test_normalize_inferential_variants_legacy_tuple():
    vars_ = normalize_inferential_variants(
        variants=[("bondo_et_al", "Bondo et al.", ["age", "sex"])],
        default_target="event",
    )
    assert vars_[0].target == "event"


def test_run_inferential_variant_target(tiny_df, tiny_schema, tmp_output):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    out = run_inferential(
        frames, schema,
        targets=["event", "grade"],
        variants=[("sex_only", "Sex only", "", "event", ["sex"])],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert set(out["target"].unique()) == {"event"}
    assert (tmp_output / "inferential" / "tables" / "event__sex_only__multivariable.csv").exists()
    assert not (tmp_output / "inferential" / "tables" / "grade__sex_only__multivariable.csv").exists()


def test_run_inferential_variants(tiny_df, tiny_schema, tmp_output):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    out = run_inferential(
        frames, schema,
        targets=["event"],
        variants=[
            ("full", "Full model", "", "event", ["age"]),
            ("sex_only", "Sex only", "", "event", ["sex"]),
        ],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert set(out["model_id"].unique()) == {"full", "sex_only"}
    assert (tmp_output / "inferential" / "tables" / "event__full__multivariable.csv").exists()
    assert (tmp_output / "inferential" / "tables" / "event__sex_only__multivariable.csv").exists()
    cases = pd.read_csv(tmp_output / "inferential" / "tables" / "multivariable_cases.csv")
    assert len(cases) == 2
    assert model_key("event", "full") == "event::full"
    assert artifact_base("event", "full") == "event__full"


def test_run_inferential(tiny_df, tiny_schema, tmp_output):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    out = run_inferential(
        frames, schema,
        targets=["event"], predictors=["age"],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert "target" in out.columns
    cases_path = tmp_output / "inferential" / "tables" / "multivariable_cases.csv"
    assert cases_path.exists()
    cases = pd.read_csv(cases_path)
    assert "epv" in cases.columns
    meta_path = tmp_output / "inferential" / "tables" / "event__calculator.json"
    assert meta_path.exists()


def test_run_inferential_clears_stale_artifacts(tiny_df, tiny_schema, tmp_output):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    tabs = tmp_output / "inferential" / "tables"
    figs = tmp_output / "inferential" / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    stale_table = tabs / "event__old_model__multivariable.csv"
    stale_table.write_text("stale")
    stale_fig = figs / "event__old_model__forest.svg"
    stale_fig.write_text("<svg></svg>")
    model_dir = tmp_output / "inferential" / "model_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    stale_model = model_dir / "event__old_model_model.json"
    stale_model.write_text("{}")

    run_inferential(
        frames, schema,
        targets=["event"],
        variants=[("new_model", "New model", "", "event", ["age"])],
        positive_class={"event": True},
        output_root=tmp_output,
    )

    assert not stale_table.exists()
    assert not stale_fig.exists()
    assert not stale_model.exists()
    assert (tabs / "event__new_model__multivariable.csv").exists()
    assert (model_dir / "event_new_model_model.json").exists()


def test_run_inferential_loads_from_disk(tiny_df, tiny_schema, tmp_output):
    import missingness_resolution as mr

    frames, schema = _make_imputed(tiny_df, tiny_schema)
    mr.save_imputed_frames(frames, tmp_output, source_df=frames[0])
    out = run_inferential(
        None, schema,
        targets=["event"], predictors=["age"],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert "target" in out.columns
