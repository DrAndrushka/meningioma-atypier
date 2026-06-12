"""Tests for config/06_derivations.py — binning and derived columns."""

from __future__ import annotations

import pandas as pd
from schema_infer import ColSpec

from config import load

_c06 = load("06_derivations")


def test_apply_derivations_bin_and_flag():
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
        _c06.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s.isin(["2", "3"]).astype("boolean"),
            kind="binary",
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
    derivations = [_c06.Apply(name="flag", source="missing", fn=lambda s: s)]
    out, _, log = _c06.apply_derivations(df, schema, derivations, preview=False)
    assert "flag" not in out.columns
    assert "source missing" in log.iloc[0]["schema_action"]


def test_apply_derivations_writes_derived_summary_row(tmp_path):
    cleaning_dir = tmp_path / "cleaning"
    cleaning_dir.mkdir()
    pd.DataFrame([
        {"step": "raw_data", "detail": "rows", "n_rows": 3, "n_dropped": 0},
        {"step": "final", "detail": "final", "n_rows": 3, "n_dropped": 0},
    ]).to_csv(cleaning_dir / "cleaning_summary.csv", index=False)

    df = pd.DataFrame({"x": [1, 2, 3]})
    schema = {"x": ColSpec("x", "continuous")}
    derivations = [
        _c06.Apply(name="x2", source="x", fn=lambda s: s * 2, kind="continuous"),
    ]
    # Run twice to confirm the derived row is replaced, not duplicated.
    for _ in range(2):
        _c06.apply_derivations(
            df, schema, derivations,
            output_root=tmp_path, write_csv=True, preview=False,
        )
    summary = pd.read_csv(cleaning_dir / "cleaning_summary.csv")
    derived = summary[summary["step"] == "derived"]
    assert len(derived) == 1
    assert "x2" in derived.iloc[0]["detail"]
    steps = summary["step"].tolist()
    assert steps.index("derived") < steps.index("final")


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
