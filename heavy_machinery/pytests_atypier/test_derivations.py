"""Tests for config/derivations.py — binning and derived columns."""

from __future__ import annotations

import pandas as pd
from schema_infer import ColSpec

from config import load

_derivations = load("derivations")


def test_apply_derivations_bin_and_flag():
    df = pd.DataFrame({"age": [45, 55, 65], "who_grade": ["1", "2", "3"]})
    schema = {
        "age": ColSpec("age", "continuous"),
        "who_grade": ColSpec("who_grade", "ordinal", ordered_levels=["1", "2", "3"]),
    }
    derivations = [
        _derivations.BinNumeric(
            name="age_bins",
            source="age",
            bins=[0, 50, 60, 100],
            labels=["<50", "50-59", "60+"],
            reason="test bins",
        ),
        _derivations.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s.isin(["2", "3"]).astype("boolean"),
            kind="binary",
            reason="test flag",
        ),
    ]
    out, out_schema, log = _derivations.apply_derivations(
        df, schema, derivations, preview=False,
    )
    assert out["age_bins"].iloc[1] == "50-59"
    assert out["high_grade"].tolist() == [False, True, True]
    assert out["high_grade"].dtype == "boolean"
    assert out_schema["high_grade"].kind == "binary"
    assert "derivation" in log.columns


def test_apply_derivations_custom_apply():
    df = pd.DataFrame({"x": [1, 2, 3]})
    schema = {"x": ColSpec("x", "continuous")}
    derivations = [
        _derivations.Apply(
            name="x_doubled",
            source="x",
            fn=lambda s: s * 2,
            kind="continuous",
            reason="double it",
        ),
    ]
    out, _, log = _derivations.apply_derivations(df, schema, derivations, preview=False)
    assert out["x_doubled"].tolist() == [2, 4, 6]
    assert log.iloc[0]["schema_action"].startswith("added ColSpec")


def test_skipped_missing_source():
    df = pd.DataFrame({"a": [1]})
    schema = {"a": ColSpec("a", "continuous")}
    derivations = [_derivations.Apply(name="flag", source="missing", fn=lambda s: s)]
    out, _, log = _derivations.apply_derivations(df, schema, derivations, preview=False)
    assert "flag" not in out.columns
    assert "source missing" in log.iloc[0]["schema_action"]


def test_apply_derivations_writes_derived_summary_row(tmp_path):
    cleaning_dir = tmp_path / "cleaning"
    cleaning_dir.mkdir()
    pd.DataFrame([
        {"step": "raw_data", "detail": "rows", "n_rows": 3, "n_columns": 1, "criterion": ""},
        {"step": "final", "detail": "final", "n_rows": 3, "n_columns": 1, "criterion": ""},
    ]).to_csv(cleaning_dir / "cleaning_summary.csv", index=False)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    schema = {
        "x": ColSpec("x", "continuous"),
        "y": ColSpec("y", "continuous"),
    }
    derivations = [
        _derivations.Apply(
            name="x2", source="x", fn=lambda s: s * 2, kind="continuous",
            reason="double x for demo",
        ),
        _derivations.Apply(
            name="y_half", source="y", fn=lambda s: s / 2, kind="continuous",
            reason="half y for demo",
        ),
    ]
    # Run twice to confirm derived rows are replaced, not duplicated.
    for _ in range(2):
        _derivations.apply_derivations(
            df, schema, derivations,
            output_root=tmp_path, write_csv=True, preview=False,
        )
    summary = pd.read_csv(cleaning_dir / "cleaning_summary.csv")
    derived = summary[summary["step"] == "derived"]
    assert len(derived) == 2
    assert derived.iloc[0]["detail"] == "added x2 ← x"
    assert derived.iloc[0]["criterion"] == "double x for demo"
    assert derived.iloc[0]["n_columns"] == 3
    assert derived.iloc[1]["detail"] == "added y_half ← y"
    assert derived.iloc[1]["criterion"] == "half y for demo"
    assert derived.iloc[1]["n_columns"] == 4
    assert summary.loc[summary["step"] == "final", "n_columns"].iloc[0] == 4
    steps = summary["step"].tolist()
    assert steps.index("derived") < steps.index("final")


def test_skipped_inactive():
    df = pd.DataFrame({"x": [1]})
    schema = {"x": ColSpec("x", "continuous")}
    derivations = [
        _derivations.Apply(
            name="y",
            source="x",
            fn=lambda s: s,
            active=False,
        ),
    ]
    out, _, log = _derivations.apply_derivations(df, schema, derivations, preview=False)
    assert "y" not in out.columns
    assert log.iloc[0]["schema_action"] == "skipped (inactive)"


def test_skipped_already_exists():
    df = pd.DataFrame({"x": [1], "y": [9]})
    schema = {"x": ColSpec("x", "continuous"), "y": ColSpec("y", "continuous")}
    derivations = [
        _derivations.Apply(name="y", source="x", fn=lambda s: s * 2, overwrite=False),
    ]
    out, _, log = _derivations.apply_derivations(df, schema, derivations, preview=False)
    assert out["y"].iloc[0] == 9
    assert "already exists" in log.iloc[0]["schema_action"]
