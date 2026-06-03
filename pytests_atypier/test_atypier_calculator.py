"""Tests for atypier_calculator.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atypier_calculator import (
    find_model_csv,
    load_latest_model,
    load_model_from_csv,
    predict_probability,
    regenerate_calculator_meta,
    risk_category,
)


@pytest.fixture
def model_csv(tmp_path: Path) -> Path:
    path = tmp_path / "high_grade__multivariable.csv"
    path.write_text(
        "predictor_col,coef,se,df,p,ci_lo,ci_hi,or,or_ci_lo,or_ci_hi,n_models,"
        "intercept_coef,intercept_or,z_mu,z_sd,target\n"
        "side_midline,-1.0,0.5,inf,0.1,-2,0,0.37,0.14,1,1,-0.5,0.61,,,high_grade\n"
        "side_right,0.5,0.3,inf,0.1,0,1,1.65,1,2.7,1,-0.5,0.61,,,high_grade\n"
        "tumor_volume,0.4,0.1,inf,0.01,0.2,0.6,1.49,1.22,1.82,1,-0.5,0.61,20,10,high_grade\n"
        "cystic_component,0.6,0.2,inf,0.05,0.2,1,1.82,1.22,2.7,1,-0.5,0.61,,,high_grade\n",
        encoding="utf-8",
    )
    meta = {
        "target": "high_grade",
        "intercept": -0.5,
        "terms": [
            {
                "name": "side",
                "kind": "categorical",
                "reference": "left",
                "levels": ["left", "midline", "right"],
                "dummies": {"midline": -1.0, "right": 0.5},
            },
            {
                "name": "tumor_volume",
                "kind": "continuous",
                "coef": 0.4,
                "z_mu": 20.0,
                "z_sd": 10.0,
            },
            {
                "name": "cystic_component",
                "kind": "binary",
                "coef": 0.6,
            },
        ],
    }
    path.with_name("high_grade__calculator.json").write_text(
        json.dumps(meta), encoding="utf-8",
    )
    return path


def test_load_model_from_csv(model_csv: Path):
    model = load_model_from_csv(model_csv)
    assert model.target == "high_grade"
    assert model.intercept == pytest.approx(-0.5)
    assert set(model.continuous) == {"tumor_volume"}
    assert set(model.binary) == {"cystic_component"}
    assert set(model.categorical) == {"side"}
    assert model.categorical["side"]["reference"] == "left"


def test_predict_reference_patient(model_csv: Path):
    model = load_model_from_csv(model_csv)
    p = predict_probability({
        "side": "left",
        "tumor_volume": 20.0,
        "cystic_component": False,
    }, model)
    assert 0.0 < p < 1.0
    assert p == pytest.approx(1 / (1 + 2.718281828 ** 0.5), rel=1e-3)


def test_predict_with_cystic_raises_probability(model_csv: Path):
    model = load_model_from_csv(model_csv)
    base = predict_probability({
        "side": "left", "tumor_volume": 20.0, "cystic_component": False,
    }, model)
    with_cystic = predict_probability({
        "side": "left", "tumor_volume": 20.0, "cystic_component": True,
    }, model)
    assert with_cystic > base


def test_risk_category():
    assert "Low" in risk_category(0.1)
    assert "Intermediate" in risk_category(0.3)
    assert "High" in risk_category(0.7)


def test_find_model_csv_real_output():
    path = find_model_csv()
    assert path.name == "high_grade__multivariable.csv"
    assert path.exists()


def test_load_latest_model_real_output():
    model = load_latest_model()
    assert model.target == "high_grade"
    assert len(model.raw_table) >= 1
    assert set(model.categorical) >= {"side", "tumor_location", "tumor_episode"}


def test_regenerate_calculator_meta_real_output():
    csv_path = find_model_csv()
    meta_path = regenerate_calculator_meta(csv_path)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["target"] == "high_grade"
    assert any(t["name"] == "side" for t in meta["terms"])
