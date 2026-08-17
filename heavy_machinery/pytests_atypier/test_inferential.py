"""Tests for inferential.py — Rubin pooling, variants, artifact cleanup."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
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


def test_small_helpers(tmp_output):
    """Rubin's pooled df blows up when between-draw variance is tiny.

    Left raw it prints as 1e+300 in the CSV, so it is displayed as ``∞``;
    a real df stays a plain integer.
    """
    assert inf._pool_df_for_display(float("inf")) == "∞"
    assert inf._pool_df_for_display(12.0) == 12
    df = pd.DataFrame({"df": [5.0, float("inf")], "p": [0.01, 0.05]})
    assert list(inf._format_inferential_table(df)["df"]) == [5, "∞"]

    assert _safe_z_denominator(0.0) == 1.0

    figs, tabs = inf._ensure_dirs(tmp_output)
    assert figs.is_dir() and tabs.is_dir()

    pooled = inf._rubin_pool(np.array([0.1, 0.2, 0.15]), np.array([0.05, 0.05, 0.05]))
    assert np.isfinite(pooled["coef"])
    assert "or" in pooled

    assert inf._artifact_model_id("high_grade_yao_et_al_2022_model", "high_grade") == (
        "yao_et_al_2022"
    )
    assert inf._artifact_model_id("high_grade_model", "high_grade") == ""


def test_build_design_and_prune_by_vif(tiny_df, tiny_schema):
    X, mapping, z_params = inf._build_design(tiny_df, tiny_schema, ["age", "sex"])
    assert "age" in mapping
    assert not X.empty
    assert "age" in z_params
    assert "mu" in z_params["age"] and "sd" in z_params["age"]

    X, _, _ = inf._build_design(tiny_df, tiny_schema, ["age"])
    pruned, vif_df = inf._prune_by_vif(X, threshold=5.0)
    assert list(pruned.columns) == list(X.columns)
    assert "vif" in vif_df.columns
    assert vif_df["vif"].notna().all()

    # VIF must use complete cases when the design matrix has NaN (e.g. an
    # unimputed binary).
    holey = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan],
        "b": [1.0, 0.0, 1.0, 1.0],
    })
    _, vif_df = inf._prune_by_vif(holey, threshold=5.0)
    assert vif_df["vif"].notna().all()
    assert (vif_df["vif"] >= 1.0).all()


def test_target_encoding():
    y = pd.Series([True, False, True], dtype="boolean")
    assert inf._target_is_binary(y, ColSpec("event", "binary"))

    enc, pos = inf._encode_target(pd.Series([True, False, True]), True)
    assert pos is True
    assert enc.iloc[0] == 1.0

    # With nothing declared, the rarer class becomes the event.
    y = pd.Series(["benign", "benign", "benign", "atypical", "benign"])
    enc, pos = inf._encode_target(y, None)
    assert pos == "atypical"
    assert enc.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_epv_uses_minority_count_when_positive_class_is_majority():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 80, 20
    y = pd.Series([True] * n_pos + [False] * n_neg)
    _, pos = inf._encode_target(y, None)
    assert pos is True

    df = pd.DataFrame({
        "event": y,
        "age": rng.normal(60.0, 10.0, size=n_pos + n_neg),
    })
    schema = {
        "event": ColSpec("event", "binary"),
        "age": ColSpec("age", "continuous"),
    }
    summary = inf.summarize_multivariable_cases(
        df, schema, targets=["event"], predictors=["age"],
    )
    assert summary.iloc[0]["n_outcome_events"] == n_neg
    assert summary.iloc[0]["epv"] == round(n_neg / summary.iloc[0]["n_design_columns"], 1)


def _make_imputed(tiny_df, tiny_schema):
    df = tiny_df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    schema = {k: v for k, v in tiny_schema.items() if k in df.columns}
    return [df], schema


def test_fit_logit_robust_converges_and_retries_a_stalled_newton(tiny_df):
    sub = tiny_df.dropna(subset=["age", "event"])
    Xc = sm.add_constant(sub[["age"]].astype(float), has_constant="add")
    model = inf._fit_logit_robust(sub["event"].astype(float), Xc)
    assert model is not None
    assert inf._logit_converged(model)

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


def test_summarize_multivariable_cases(tiny_df, tiny_schema):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    summary = inf.summarize_multivariable_cases(
        frames[0], schema, targets=["event"], predictors=["age"],
        positive_class={"event": True},
    )
    assert summary.iloc[0]["n_complete_cases"] == len(frames[0])
    assert summary.iloc[0]["n_outcome_events"] >= 1

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


def test_normalize_inferential_variants():
    vars_ = normalize_inferential_variants(
        variants=[("bondo_et_al", "Bondo et al.", "https://example.com/bondo", "high_grade", ["age", "sex"])],
    )
    assert vars_[0].model_id == "bondo_et_al"
    assert vars_[0].title == "Bondo et al."
    assert vars_[0].target == "high_grade"
    assert vars_[0].link == "https://example.com/bondo"

    # The legacy three-tuple takes its target from the default.
    vars_ = normalize_inferential_variants(
        variants=[("bondo_et_al", "Bondo et al.", ["age", "sex"])],
        default_target="event",
    )
    assert vars_[0].target == "event"


def test_run_inferential_variants_write_one_artifact_set_per_variant(
    tiny_df, tiny_schema, tmp_output,
):
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    tables = tmp_output / "inferential" / "tables"

    # A variant declares its own target, so the other target is not modelled.
    out = run_inferential(
        frames, schema,
        targets=["event", "grade"],
        variants=[("sex_only", "Sex only", "", "event", ["sex"])],
        positive_class={"event": True},
        output_root=tmp_output,
    )
    assert set(out["target"].unique()) == {"event"}
    assert (tables / "event__sex_only__multivariable.csv").exists()
    assert not (tables / "grade__sex_only__multivariable.csv").exists()

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
    assert (tables / "event__full__multivariable.csv").exists()
    assert (tables / "event__sex_only__multivariable.csv").exists()
    cases = pd.read_csv(tables / "multivariable_cases.csv")
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


def test_run_inferential_stage_threads_selection_candidates(tiny_df, tiny_schema, tmp_output):
    """The selection pool is the EDA predictor list, not the literature union —
    the variables that win include ones no literature model uses."""
    import inspect
    sig = inspect.signature(inf.run_inferential_stage)
    assert "selection_candidates" in sig.parameters
    sig2 = inspect.signature(inf.run_inferential)
    assert "selection_candidates" in sig2.parameters


def test_run_inferential_warns_when_selection_candidates_omitted(tiny_df, tiny_schema, tmp_output):
    """Skipping the comparison stage must be loud, not silent — a caller that
    forgets ``selection_candidates`` (e.g. the notebook before Task 13 wires
    it through) should see a warning, not just three missing files."""
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    with pytest.warns(UserWarning, match="selection_candidates"):
        run_inferential(
            frames, schema,
            targets=["event"], predictors=["age"],
            positive_class={"event": True},
            output_root=tmp_output,
        )


def test_run_inferential_clears_stale_comparison_tables_when_stage_is_skipped(
    tiny_df, tiny_schema, tmp_output,
):
    """A run without ``selection_candidates`` writes no new comparison tables,
    but it must not leave a PREVIOUS run's comparison tables sitting on disk —
    the report would render them as current numbers for a stage that did not
    run this time."""
    frames, schema = _make_imputed(tiny_df, tiny_schema)
    tabs = tmp_output / "inferential" / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    stale = tabs / "model_vs_single_auc.csv"
    stale.write_text("model_id,single\nold_model,old_predictor\n")
    other_stale = tabs / "single_predictor_reference.csv"
    other_stale.write_text("predictor\nold_predictor\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run_inferential(
            frames, schema,
            targets=["event"], predictors=["age"],
            positive_class={"event": True},
            output_root=tmp_output,
        )

    assert not stale.exists()
    assert not other_stale.exists()


def test_run_inferential_loads_from_disk(tiny_df, tiny_schema, tmp_output):
    import missingness_resolution as mr

    frames, schema = _make_imputed(tiny_df, tiny_schema)
    mr.save_imputed_frames(frames, tmp_output, source_df=frames[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with pytest.warns(UserWarning, match="Rubin pooling on a sensitivity-analysis imputation"):
            out = run_inferential(
                None, schema,
                targets=["event"], predictors=["age"],
                positive_class={"event": True},
                output_root=tmp_output,
            )
    assert "target" in out.columns


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def _pooled(**overrides) -> pd.DataFrame:
    base = pd.DataFrame([
        {"predictor_col": "hyperostosis", "or": 2.31,
         "or_ci_lo": 1.38, "or_ci_hi": 3.89, "z_sd": np.nan},
        {"predictor_col": "adc_value", "or": 0.61,
         "or_ci_lo": 0.46, "or_ci_hi": 0.82, "z_sd": 0.17},
        {"predictor_col": "mass_effect", "or": 1.05,
         "or_ci_lo": 0.60, "or_ci_hi": 1.84, "z_sd": np.nan},
    ])
    for key, val in overrides.items():
        base[key] = val
    return base


def test_predictor_label_names_the_sd_for_a_continuous_predictor():
    row = _pooled().iloc[1].copy()          # adc_value, z_sd = 0.17
    assert "per 1 SD: 0.17" in inf.predictor_label(row)


def test_predictor_label_ignores_a_missing_or_zero_sd():
    row = _pooled().iloc[0].copy()
    row["z_sd"] = 0.0
    assert "per 1 SD" not in inf.predictor_label(row)


def test_forest_plot_writes_the_report_png_and_the_ajnr_tif_on_request(
    tmp_path, monkeypatch,
):
    simple = pd.DataFrame({
        "predictor_col": ["age"],
        "or": [1.5],
        "or_ci_lo": [0.8],
        "or_ci_hi": [2.5],
    })
    figs = tmp_path / "figs"
    figs.mkdir()
    inf._forest_plot(simple, "event", figs)
    assert (figs / "event__forest.png").exists()

    inf._forest_plot(_pooled(), "event", tmp_path, model_id="m1")
    assert (tmp_path / "event__m1__forest.png").exists()
    assert not (tmp_path / "event__m1__forest.eps").exists()

    # An empty model draws nothing at all.
    inf._forest_plot(_pooled().assign(**{"or": np.nan}), "event", tmp_path,
                     model_id="m2")
    assert not list(tmp_path.glob("event__m2__forest*"))

    monkeypatch.setenv("ATYPIER_FIGURES", "submission")
    inf._forest_plot(_pooled(), "event", tmp_path, model_id="m3")
    assert (tmp_path / "event__m3__forest.png").exists()
    assert (tmp_path / "event__m3__forest.tif").exists()
    assert not (tmp_path / "event__m3__forest.eps").exists()
