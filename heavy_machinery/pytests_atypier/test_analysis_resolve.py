"""resolve_eda / resolve_inferential_variants honor hidden_parent_columns.csv."""

from __future__ import annotations

import pandas as pd
import pytest

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


def test_a_predictor_that_matches_nothing_is_refused(tmp_path):
    """Two published models were fitted with five predictors instead of seven.

    A rename nobody followed through used to vanish in silence: the model kept
    its title and its citation, printed its own EPV and AUC, and read as a
    replication of the paper it named. Dropping a hidden parent is deliberate;
    dropping a name that matches nothing is a mistake.
    """
    df = pd.DataFrame({"high_grade": [True, False], "male": [True, False]})
    analysis = load("analysis")
    variants = [("m", "M", "", "high_grade", ["male", "irregular_margin"])]
    with pytest.raises(KeyError, match="not the model it says it is"):
        analysis.resolve_inferential_variants(df, variants)


def test_a_hidden_parent_is_dropped_but_said_out_loud(tmp_path):
    """Deliberate, so a warning — but the flag replacing it has another name."""
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["sex"]}).to_csv(
        cleaning / "hidden_parent_columns.csv", index=False)
    df = pd.DataFrame({"high_grade": [True, False], "male": [True, False],
                       "sex": ["male", "female"]})
    analysis = load("analysis")
    variants = [("m", "M", "", "high_grade", ["male", "sex"])]
    with pytest.warns(UserWarning, match="hidden parent"):
        out = analysis.resolve_inferential_variants(
            df, variants, output_root=tmp_path)
    assert out[0].predictors == ("male",)
