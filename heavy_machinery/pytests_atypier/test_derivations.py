"""Tests for config/derivations.py — binning and derived columns."""

from __future__ import annotations

import pandas as pd
import pytest
from schema_infer import ColSpec

from config import load

_derivations = load("derivations")


def test_apply_derivations_bins_flags_and_carries_the_positive_class():
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
            positive_class="60+",
        ),
        _derivations.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s.isin(["2", "3"]).astype("boolean"),
            kind="binary",
            reason="test flag",
            positive_class=True,
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
    assert out_schema["age_bins"].positive_class == "60+"
    assert out_schema["high_grade"].positive_class is True


def test_a_derivation_is_applied_skipped_or_replaces_an_existing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    schema = {"x": ColSpec("x", "continuous")}
    out, _, log = _derivations.apply_derivations(
        df, schema,
        [_derivations.Apply(name="x_doubled", source="x", fn=lambda s: s * 2,
                            kind="continuous", reason="double it")],
        preview=False)
    assert out["x_doubled"].tolist() == [2, 4, 6]
    assert log.iloc[0]["schema_action"].startswith("added ColSpec")

    out, _, log = _derivations.apply_derivations(
        pd.DataFrame({"a": [1]}), {"a": ColSpec("a", "continuous")},
        [_derivations.Apply(name="flag", source="missing", fn=lambda s: s)],
        preview=False)
    assert "flag" not in out.columns
    assert "source missing" in log.iloc[0]["schema_action"]

    out, _, log = _derivations.apply_derivations(
        pd.DataFrame({"x": [1]}), {"x": ColSpec("x", "continuous")},
        [_derivations.Apply(name="y", source="x", fn=lambda s: s, active=False)],
        preview=False)
    assert "y" not in out.columns
    assert log.iloc[0]["schema_action"] == "skipped (inactive)"

    out, _, log = _derivations.apply_derivations(
        pd.DataFrame({"x": [1], "y": [9]}),
        {"x": ColSpec("x", "continuous"), "y": ColSpec("y", "continuous")},
        [_derivations.Apply(name="y", source="x", fn=lambda s: s * 2)],
        preview=False)
    assert out["y"].iloc[0] == 2
    assert "updated" in log.iloc[0]["schema_action"] or "added" in log.iloc[0]["schema_action"]


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


def test_the_dda_derived_listing_comes_from_the_specs_and_reaches_disk(tmp_path):
    derivations = [
        _derivations.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s,
            kind="binary",
            dda_in_derived=False,
        ),
        _derivations.Apply(
            name="male_sex",
            source="sex",
            fn=lambda s: s,
            kind="binary",
            dda_in_derived=True,
        ),
        _derivations.Compute(
            name="edema_index",
            fn=lambda df: df["x"],
            sources=["x"],
            dda_in_derived=True,
        ),
    ]
    assert _derivations.dda_in_derived_columns(derivations) == frozenset(
        {"male_sex", "edema_index"}
    )
    path = _derivations.write_dda_in_derived_columns(tmp_path, derivations)
    assert pd.read_csv(path)["column"].tolist() == ["edema_index", "male_sex"]

    run_root = tmp_path / "run"
    _derivations.apply_derivations(
        pd.DataFrame({"sex": ["male", "female"]}),
        {"sex": ColSpec("sex", "nominal")},
        [_derivations.Apply(
            name="male_sex", source="sex",
            fn=lambda s: (s == "male").astype("boolean"),
            kind="binary", dda_in_derived=True)],
        output_root=run_root, write_csv=True, preview=False,
    )
    path = run_root / "cleaning" / "dda_derived_columns.csv"
    assert path.exists()
    assert pd.read_csv(path)["column"].tolist() == ["male_sex"]


def test_eda_in_derived_none_writes_excluded_csv(tmp_path):
    derivations = [
        _derivations.Apply(
            name="male_sex",
            source="sex",
            fn=lambda s: s,
            kind="binary",
            eda_in_derived=True,
        ),
        _derivations.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s,
            kind="binary",
            eda_in_derived=None,
        ),
        _derivations.Compute(
            name="age_years",
            fn=lambda df: df["age"],
            sources=["age"],
            eda_in_derived=False,
        ),
    ]
    assert _derivations.eda_in_derived_columns(derivations) == frozenset({"male_sex"})
    assert _derivations.eda_excluded_columns(derivations) == frozenset({"high_grade"})
    derived_path, excluded_path = _derivations.write_eda_in_derived_columns(
        tmp_path, derivations,
    )
    assert pd.read_csv(derived_path)["column"].tolist() == ["male_sex"]
    assert pd.read_csv(excluded_path)["column"].tolist() == ["high_grade"]


def test_derived_dependencies_from_skips_inactive_and_inplace():
    derivations = [
        _derivations.Apply(
            name="high_grade", source="who_grade", fn=lambda s: s, kind="binary",
        ),
        _derivations.Apply(
            name="ki67_mid", source="ki67_pct", fn=lambda s: s, active=False,
        ),
        _derivations.Compute(
            name="edema_volume_cm3",
            sources=["perifocal_edema", "edema_volume_cm3"],
            fn=lambda df: df["edema_volume_cm3"],
        ),
        _derivations.Compute(
            name="edema_index",
            sources=["edema_volume_cm3", "tumor_volume"],
            fn=lambda df: df["edema_volume_cm3"] / df["tumor_volume"],
        ),
    ]
    assert _derivations.derived_dependencies_from(derivations) == {
        "high_grade": ["who_grade"],
        "edema_index": ["edema_volume_cm3", "tumor_volume"],
    }


def test_the_hidden_parent_listing_comes_from_the_specs_and_reaches_disk(tmp_path):
    derivations = [
        _derivations.Apply(
            name="male_sex",
            source="sex",
            fn=lambda s: s,
            kind="binary",
            hide_parent=True,
        ),
        _derivations.Apply(
            name="high_grade",
            source="who_grade",
            fn=lambda s: s,
            kind="binary",
            hide_parent=False,
        ),
        _derivations.Apply(
            name="ki67_mid",
            source="ki67_pct",
            fn=lambda s: s,
            hide_parent=True,
            active=False,
        ),
        _derivations.Compute(
            name="edema_volume_cm3",
            sources=["perifocal_edema", "edema_volume_cm3"],
            fn=lambda df: df["edema_volume_cm3"],
            hide_parent=True,
        ),
        _derivations.BinNumeric(
            name="age_bins",
            source="age",
            bins=[-float("inf"), 50, float("inf")],
            labels=["lt50", "ge50"],
            hide_parent=True,
        ),
    ]
    assert _derivations.hidden_parent_columns(derivations) == frozenset(
        {"sex", "perifocal_edema", "age"}
    )

    df = pd.DataFrame({"sex": ["male", "female"], "who_grade": [1, 2]})
    schema = {
        "sex": ColSpec("sex", "nominal"),
        "who_grade": ColSpec("who_grade", "ordinal"),
    }
    out, _, _ = _derivations.apply_derivations(
        df, schema,
        [_derivations.Apply(
            name="male_sex", source="sex",
            fn=lambda s: (s == "male").astype("boolean"),
            kind="binary", hide_parent=True)],
        output_root=tmp_path, write_csv=True, preview=False,
    )
    path = tmp_path / "cleaning" / "hidden_parent_columns.csv"
    assert path.exists()
    assert pd.read_csv(path)["column"].tolist() == ["sex"]
    assert "sex" in out.columns
    assert "male_sex" in out.columns


# --- eda_in_derived defaults from hide_parent ------------------------------
def _apply(**kwargs):
    return _derivations.Apply(
        name="flag", source="parent", fn=lambda s: s, kind="binary", **kwargs)


def test_eda_in_derived_defaults_from_hide_parent():
    """Hiding the parent makes the flag native — it replaced that column, so
    nothing in the table is left to restate. Leaving the parent in place makes
    it derived: a cut-point and the measurement it was cut from are one thing
    twice. Saying it outright beats the default, and ``None`` was already spoken
    for (omit from the EDA entirely), which is why the default needed a sentinel.
    """
    assert _apply(hide_parent=True).eda_in_derived is False
    assert _apply(hide_parent=False).eda_in_derived is True
    assert _apply(hide_parent=False, eda_in_derived=False).eda_in_derived is False
    assert _apply(hide_parent=True, eda_in_derived=True).eda_in_derived is True
    assert _apply(hide_parent=True, eda_in_derived=None).eda_in_derived is None


def test_the_sentinel_is_resolved_at_declaration_and_refuses_to_be_a_boolean():
    """`if spec.eda_in_derived` on an unresolved AUTO would silently mean False,
    and a late default lets the two handoff CSVs disagree about one column."""
    with pytest.raises(TypeError, match="not a truth value"):
        bool(_derivations.AUTO)

    spec = _apply(hide_parent=True)
    assert not isinstance(spec.eda_in_derived, type(_derivations.AUTO))


def test_no_derivation_can_be_both_hidden_parent_and_eda_derived():
    """The pair that produced the blank-q defect: in neither family.

    A column whose parent is hidden has nothing to restate, so putting it in the
    Derived family leaves it corrected alone while its parent is gone — which is
    how multiple_meningiomas ended up with no q value at all.
    """
    derivations = [
        _derivations.Apply(name="male", source="sex", fn=lambda s: s,
                           kind="binary", hide_parent=True),
        _derivations.Apply(name="adc_le", source="adc", fn=lambda s: s,
                           kind="binary"),
    ]
    derived = _derivations.eda_in_derived_columns(derivations)
    hidden_flags = {d.name for d in derivations if d.hide_parent}
    assert derived & hidden_flags == set()
    assert derived == {"adc_le"}
