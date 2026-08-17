"""Tests for model_calculator.py — JSON artifact and Streamlit schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_calculator import (
    CALCULATOR_MODEL_ID,
    build_auc_comparison_figure,
    build_encoded_features,
    build_roc_validation_figure,
    calculator_meta_to_streamlit_artifact,
    load_model_artifact,
    predict_from_artifact,
    resolve_streamlit_artifact_path,
    write_streamlit_artifacts,
)

ARTIFACT_PATH = Path(__file__).resolve().parent / "fixtures" / "high_grade_model.json"


@pytest.fixture
def artifact() -> dict:
    return load_model_artifact(ARTIFACT_PATH)


def test_load_model_artifact(artifact: dict):
    assert artifact["target"] == "high_grade"
    assert "const" in artifact["coefficients"]
    assert len(artifact["features"]) == 4


def test_reference_case_probability(artifact: dict):
    user_inputs = {
        "tumor_location": "non_skull_base",
        "tumor_volume": 40,
        "perifocal_edema": True,
        "hyperostosis": False,
    }
    p = predict_from_artifact(user_inputs, artifact)
    assert p == pytest.approx(0.451, abs=0.02)


def test_build_encoded_features_keys(artifact: dict):
    encoded = build_encoded_features(
        {
            "tumor_location": "skull_base",
            "tumor_volume": 10.0,
            "perifocal_edema": False,
            "hyperostosis": True,
        },
        artifact,
    )
    assert encoded["tumor_location_skull_base"] == 1.0
    vol_feat = next(f for f in artifact["features"] if f["name"] == "tumor_volume")
    if vol_feat.get("transform") == "standardize":
        z_vol = (10.0 - vol_feat["z_mu"]) / vol_feat["z_sd"]
        assert encoded["tumor_volume"] == pytest.approx(z_vol)
    else:
        assert encoded["tumor_volume"] == pytest.approx(10.0)
    assert encoded["perifocal_edema"] == 0.0
    assert encoded["hyperostosis"] == 1.0


def test_auc_validation_figures(artifact: dict):
    validation = artifact["validation"]
    auc_fig = build_auc_comparison_figure(validation)
    roc_fig = build_roc_validation_figure(validation)
    assert auc_fig is not None
    assert roc_fig is not None
    import matplotlib.pyplot as plt

    plt.close(auc_fig)
    plt.close(roc_fig)


def test_calculator_meta_to_streamlit_artifact():
    meta = {
        "target": "high_grade",
        "intercept": -0.5,
        "terms": [
            {
                "name": "tumor_volume",
                "kind": "continuous",
                "coef": 0.4,
                "z_mu": 20.0,
                "z_sd": 10.0,
            },
            {"name": "hyperostosis", "kind": "binary", "coef": 0.6},
        ],
    }
    out = calculator_meta_to_streamlit_artifact(meta, n=100, events=30)
    assert out["target"] == "high_grade"
    assert out["coefficients"]["const"] == -0.5
    assert out["features"][0]["transform"] == "standardize"
    assert out["n"] == 100


def test_calculator_meta_ordinal_age_bins():
    meta = {
        "target": "high_grade",
        "intercept": -0.2,
        "terms": [
            {
                "name": "age_bins",
                "kind": "ordinal",
                "coef": 0.15,
                "levels": ["<50", "50-59", "60-69", "70-79", "80+"],
            },
        ],
    }
    out = calculator_meta_to_streamlit_artifact(meta, n=50, events=10)
    assert out["coefficients"]["age_bins"] == 0.15
    feat = out["features"][0]
    assert feat["type"] == "ordinal"
    encoded = build_encoded_features({"age_bins": "60-69"}, out)
    assert encoded["age_bins"] == 2.0


def test_write_streamlit_artifacts(tmp_path: Path):
    tabs = tmp_path / "inferential" / "tables"
    tabs.mkdir(parents=True)
    (tabs / "event__calculator.json").write_text(
        json.dumps({
            "target": "event",
            "intercept": 0.1,
            "terms": [{"name": "age", "kind": "binary", "coef": 0.2}],
        }),
        encoding="utf-8",
    )
    paths = write_streamlit_artifacts(tmp_path)
    art_dir = tmp_path / "inferential" / "model_artifacts"
    assert paths == [art_dir / "event_model.json"]
    assert json.loads(paths[0].read_text())["target"] == "event"


def test_selected_model_auc_agrees_between_comparison_and_calculator_paths(tmp_path: Path):
    """Blocker 1 (final whole-branch review): a data-selected model (e.g.
    ``top_6_variables``) is validated on two independent paths --
    ``model_comparison.run_comparison_stage`` (writes ``model_vs_single_auc.csv``,
    which the comparison figure and comparison table read) and
    ``model_calculator.write_streamlit_artifacts`` (writes the calculator JSON,
    which the ROC figure annotation and the Streamlit calculator read). Both
    claim to report "optimism-corrected AUC" for the same model on the same
    cohort. Before this fix, only the comparison path re-ran variable
    selection inside its bootstrap; the calculator path validated the fixed,
    already-picked columns and published a different, uncorrected number for
    the same fold. This pins the two paths to the same number.
    """
    import numpy as np
    import pandas as pd
    from schema_infer import ColSpec

    import model_comparison as mc
    import variable_selection as vs

    rng = np.random.RandomState(3)
    n = 220
    y = rng.binomial(1, 0.4, n)
    df = pd.DataFrame({
        "event": y.astype(bool),
        "a": y * 1.2 + rng.normal(size=n),
        "b": y * 0.9 + rng.normal(size=n),
        "c": y * 0.6 + rng.normal(size=n),
        "d": rng.normal(size=n),
    })
    schema = {"event": ColSpec("event", "binary"),
              **{k: ColSpec(k, "continuous") for k in ("a", "b", "c", "d")}}
    candidates = ["a", "b", "c", "d"]
    n_bootstrap = 25

    # Full-cohort selection picks the top 2 -- whatever they are -- and the
    # multivariable model is fitted on exactly those, same as the notebook
    # fits top_6_variables on vs.select_variables's own pick.
    picked, _ = vs.select_variables(df, y, candidates, k=2, rho_max=0.8)
    assert len(picked) == 2
    variants = [{"model_id": "top_6_variables", "predictors": picked}]

    # Path 1: the combined-vs-single comparison stage.
    comparison_dir = tmp_path / "comparison_tables"
    mc.run_comparison_stage(
        df, schema, variants, "event", comparison_dir,
        n_bootstrap=n_bootstrap, candidates=candidates, k_top=2,
        assert_reference=False, selected_model_ids={"top_6_variables"})
    vs_tbl = pd.read_csv(comparison_dir / "model_vs_single_auc.csv")
    comparison_auc = float(vs_tbl["auc_model_corrected"].iloc[0])

    # Path 2: the Streamlit calculator export, built the way
    # inferential.run_inferential actually calls it -- a calculator.json meta
    # file plus the same cohort/schema/candidates, asked to re-select inside
    # its own bootstrap for the same model_id.
    output_root = tmp_path / "output"
    tabs_dir = output_root / "inferential" / "tables"
    tabs_dir.mkdir(parents=True)
    meta = {
        "target": "event",
        "model_id": "top_6_variables",
        "intercept": 0.0,
        "terms": [
            {"name": p, "kind": "continuous", "coef": 0.1, "z_mu": 0.0, "z_sd": 1.0}
            for p in picked
        ],
    }
    (tabs_dir / "event__top_6_variables__calculator.json").write_text(
        json.dumps(meta), encoding="utf-8")
    write_streamlit_artifacts(
        output_root,
        cohort_df=df,
        schema=schema,
        n_bootstrap=n_bootstrap,
        selection_candidates=candidates,
        selected_model_ids={"top_6_variables"},
        k_top=2,
    )
    art_path = (
        output_root / "inferential" / "model_artifacts"
        / "event_top_6_variables_model.json"
    )
    artifact = json.loads(art_path.read_text())
    auc_row = next(
        m for m in artifact["validation"]["metrics"] if m["metric"] == "AUC")
    calculator_auc = float(auc_row["optimism_corrected"])

    assert calculator_auc == pytest.approx(comparison_auc, abs=1e-9)


def test_resolve_streamlit_artifact_path_prefers_experimental_model_1(tmp_path: Path):
    art_dir = tmp_path / "inferential" / "model_artifacts"
    art_dir.mkdir(parents=True)
    other = art_dir / "high_grade_yao_et_al_2022_model.json"
    preferred = art_dir / f"high_grade_{CALCULATOR_MODEL_ID}_model.json"
    other.write_text("{}", encoding="utf-8")
    preferred.write_text("{}", encoding="utf-8")
    # Newer literature file must not win — calculator is pinned to experimental_model_1.
    other.touch()
    got = resolve_streamlit_artifact_path(output_root=tmp_path)
    assert got == preferred.resolve()


def test_resolve_streamlit_artifact_path_requires_experimental_model_1(tmp_path: Path):
    art_dir = tmp_path / "inferential" / "model_artifacts"
    art_dir.mkdir(parents=True)
    (art_dir / "high_grade_experimental_model_2_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=CALCULATOR_MODEL_ID):
        resolve_streamlit_artifact_path(output_root=tmp_path)


def test_missing_coefficient_raises(tmp_path: Path):
    bad = {
        "model_name": "bad",
        "target": "y",
        "coefficients": {"const": 0.0},
        "features": [
            {
                "name": "x",
                "type": "continuous",
                "input_widget": "number_input",
                "transform": "raw",
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    art = load_model_artifact(path)
    with pytest.raises(KeyError, match="Missing user input"):
        predict_from_artifact({}, art)
