"""Tests for config/07_analysis — variant resolve helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "heavy_machinery" / "config" / "07_analysis.py"
_spec = importlib.util.spec_from_file_location("analysis_07", _CONFIG_PATH)
assert _spec and _spec.loader
_c07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c07)


@pytest.fixture
def grade_df() -> pd.DataFrame:
    return pd.DataFrame({
        "high_grade": [True, False, True, False],
        "sex": ["M", "F", "M", "F"],
        "age_bins": ["40-50", "50-60", "40-50", "50-60"],
        "adc_value": [900.0, 1100.0, 950.0, 1200.0],
    })


def test_resolve_inferential_variants_tags_by_list(grade_df):
    literature = [
        ("yao_et_al_2022", "Yao et al. 2022", "https://example.com", "high_grade", ["sex"]),
    ]
    experimental = [
        ("try_hard_model", "try_hard | high grade", "", "high_grade", ["age_bins", "adc_value"]),
    ]
    resolved = _c07.resolve_inferential_variants(
        grade_df,
        literature,
        experimental,
    )
    by_id = {var.model_id: var for var in resolved}
    assert by_id["yao_et_al_2022"].experimental is False
    assert by_id["try_hard_model"].experimental is True
    assert [var.model_id for var in resolved] == [
        "yao_et_al_2022",
        "try_hard_model",
    ]


def test_resolve_inferential_variants_experimental_id_not_required(grade_df):
    experimental = [
        ("my_custom_model", "custom", "", "high_grade", ["sex"]),
    ]
    resolved = _c07.resolve_inferential_variants(grade_df, [], experimental)
    assert len(resolved) == 1
    assert resolved[0].model_id == "my_custom_model"
    assert resolved[0].experimental is True
