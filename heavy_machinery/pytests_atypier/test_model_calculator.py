"""Tests for model_calculator.py — JSON artifact and Streamlit schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_calculator import (
    build_auc_comparison_figure,
    build_encoded_features,
    build_roc_validation_figure,
    calculator_meta_to_streamlit_artifact,
    load_model_artifact,
    predict_from_artifact,
    write_streamlit_artifacts,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_CANDIDATES = (
    _REPO_ROOT / "output" / "inferential" / "model_artifacts" / "high_grade_model.json",
)
ARTIFACT_PATH = next((p for p in _ARTIFACT_CANDIDATES if p.is_file()), _ARTIFACT_CANDIDATES[0])


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
    z_vol = (10.0 - vol_feat["z_mu"]) / vol_feat["z_sd"]
    assert encoded["tumor_volume"] == pytest.approx(z_vol)
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
