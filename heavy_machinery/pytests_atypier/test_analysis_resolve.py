"""resolve_eda / resolve_inferential_variants honor hidden_parent_columns.csv."""

from __future__ import annotations

import pandas as pd
from heavy_machinery.config import load


def test_resolve_eda_drops_hidden_parents(tmp_path):
    _analysis = load("analysis")
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False,
    )
    df = pd.DataFrame({"high_grade": [0, 1], "sex": ["m", "f"], "age": [40, 50], "male_sex": [1, 0]})
    targets, preds = _analysis.resolve_eda(
        df, ["high_grade"], ["sex", "age", "male_sex"], output_root=tmp_path,
    )
    assert targets == ["high_grade"]
    assert preds == ["age", "male_sex"]


def test_resolve_inferential_variants_drops_hidden_parents(tmp_path):
    _analysis = load("analysis")
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False,
    )
    df = pd.DataFrame({
        "high_grade": [0, 1, 0, 1],
        "sex": ["m", "f", "m", "f"],
        "age": [40, 50, 60, 70],
        "male_sex": [1, 0, 1, 0],
    })
    variants = _analysis.resolve_inferential_variants(
        df,
        [("m1", "Demo", "", "high_grade", ["sex", "age", "male_sex"])],
        output_root=tmp_path,
    )
    assert len(variants) == 1
    assert list(variants[0].predictors) == ["age", "male_sex"]
