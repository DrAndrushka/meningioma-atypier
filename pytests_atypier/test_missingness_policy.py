"""Tests for config/05_missingness.py — structural/MNAR rules."""

from __future__ import annotations

import pandas as pd
from schema_infer import ColSpec

from config import load

_c05 = load("05_missingness")


def test_apply_missingness_policy_empty():
    df = pd.DataFrame({"a": [1, 2]})
    schema = {"a": ColSpec("a", "continuous")}
    out, out_schema, log = _c05.apply_missingness_policy(df, schema, [], [])
    assert len(out) == 2
    assert out_schema["a"].kind == "continuous"
    assert log.empty


def test_apply_structural_group():
    df = pd.DataFrame({
        "slot_1": [1.0, None, 3.0],
        "slot_2": [2.0, None, None],
    })
    schema = {
        "slot_1": ColSpec("slot_1", "continuous"),
        "slot_2": ColSpec("slot_2", "continuous"),
    }
    groups = [
        _c05.StructuralGroup(
            name="lesion_slots",
            cols=["slot_1", "slot_2", "slot_3"],
            derive_count_col="n_slots",
            derive_max_col="max_slot",
            skip_raw=True,
            reason="Blank slots mean lesion does not exist.",
        ),
    ]
    out, out_schema, log = _c05.apply_missingness_policy(df, schema, groups, [])

    assert out["n_slots"].tolist() == [2, 0, 1]
    assert out["max_slot"].iloc[0] == 2.0
    assert pd.isna(out["max_slot"].iloc[1])
    assert out["max_slot"].iloc[2] == 3.0
    assert out_schema["slot_1"].kind == "skip"
    assert out_schema["slot_2"].kind == "skip"
    assert log.iloc[0]["status"] == "applied"
    assert "slot_3" in log.iloc[0]["missing_cols"]


def test_apply_mnar_column():
    df = pd.DataFrame({"ki67_pct": [1.0, None, 3.0]})
    schema = {"ki67_pct": ColSpec("ki67_pct", "text")}
    mnar = [
        _c05.MnarColumn(
            col="ki67_pct",
            reason="Ki-67 may be absent when not measured.",
        ),
    ]
    out, out_schema, log = _c05.apply_missingness_policy(df, schema, [], mnar)

    assert out["ki67_pct_missing"].tolist() == [0, 1, 0]
    assert out["ki67_pct_missing"].dtype == "int8"
    assert out_schema["ki67_pct_missing"].kind == "binary"
    assert log.iloc[0]["status"] == "applied"


def test_skipped_missing_column_does_not_crash():
    df = pd.DataFrame({"a": [1]})
    schema = {"a": ColSpec("a", "continuous")}
    mnar = [_c05.MnarColumn(col="missing_col", reason="test")]
    out, _, log = _c05.apply_missingness_policy(df, schema, [], mnar)
    assert len(out) == 1
    assert log.iloc[0]["status"] == "skipped_col_missing"
