"""Tests for artifact-driven model_calculator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_calculator import (
    build_auc_comparison_figure,
    build_encoded_features,
    build_roc_validation_figure,
    load_model_artifact,
    predict_from_artifact,
)

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "model_artifacts" / "high_grade_model.json"


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
    assert p == pytest.approx(0.434, abs=0.001)


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
    assert encoded["tumor_volume"] == 10.0
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
