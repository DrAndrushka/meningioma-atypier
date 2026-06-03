"""Tests for config/06_derivations.py."""

from __future__ import annotations

import pandas as pd
from schema_infer import ColSpec

from config import load

_c06 = load("06_derivations")


def test_apply_derivations_bin_and_is_in():
    df = pd.DataFrame({"age": [45, 55, 65], "who_grade": ["1", "2", "3"]})
    schema = {
        "age": ColSpec("age", "continuous"),
        "who_grade": ColSpec("who_grade", "ordinal", ordered_levels=["1", "2", "3"]),
    }
    derivations = [
        _c06.BinNumeric(
            name="age_bins",
            source="age",
            bins=[0, 50, 60, 100],
            labels=["<50", "50-59", "60+"],
            reason="test bins",
        ),
        _c06.IsIn(
            name="high_grade",
            source="who_grade",
            values=["2", "3"],
            reason="test flag",
        ),
    ]
    out, out_schema, log = _c06.apply_derivations(
        df, schema, derivations, preview=False,
    )
    assert out["age_bins"].iloc[1] == "50-59"
    assert out["high_grade"].tolist() == [False, True, True]
    assert out["high_grade"].dtype == "boolean"
    assert out_schema["high_grade"].kind == "binary"
    assert log.iloc[1]["matched_n"] == 2
    assert "derivation" in log.columns


def test_apply_derivations_custom_apply():
    df = pd.DataFrame({"x": [1, 2, 3]})
    schema = {"x": ColSpec("x", "continuous")}
    derivations = [
        _c06.Apply(
            name="x_doubled",
            source="x",
            fn=lambda s: s * 2,
            kind="continuous",
            reason="double it",
        ),
    ]
    out, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert out["x_doubled"].tolist() == [2, 4, 6]
    assert log.iloc[0]["schema_action"].startswith("added ColSpec")


def test_skipped_missing_source():
    df = pd.DataFrame({"a": [1]})
    schema = {"a": ColSpec("a", "continuous")}
    derivations = [_c06.IsIn(name="flag", source="missing", values=[1])]
    out, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert "flag" not in out.columns
    assert "source missing" in log.iloc[0]["schema_action"]


def test_skipped_inactive():
    df = pd.DataFrame({"x": [1]})
    schema = {"x": ColSpec("x", "continuous")}
    derivations = [
        _c06.Apply(
            name="y",
            source="x",
            fn=lambda s: s,
            active=False,
        ),
    ]
    out, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert "y" not in out.columns
    assert log.iloc[0]["schema_action"] == "skipped (inactive)"


def test_skipped_already_exists():
    df = pd.DataFrame({"x": [1], "y": [9]})
    schema = {"x": ColSpec("x", "continuous"), "y": ColSpec("y", "continuous")}
    derivations = [
        _c06.Apply(name="y", source="x", fn=lambda s: s * 2, overwrite=False),
    ]
    out, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert out["y"].iloc[0] == 9
    assert "already exists" in log.iloc[0]["schema_action"]


def test_is_in_zero_match_warning():
    df = pd.DataFrame({"g": ["1", "1"]})
    schema = {"g": ColSpec("g", "ordinal")}
    derivations = [_c06.IsIn(name="flag", source="g", values=[99])]
    _, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert log.iloc[0]["matched_n"] == 0
    assert "check dtype" in log.iloc[0]["warning"]
