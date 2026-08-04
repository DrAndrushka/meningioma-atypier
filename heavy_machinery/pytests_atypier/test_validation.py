"""Pandera validation of the cleaning → modelling handoff."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from validation import validate_unimputed_handoff


def _write_schema(output_root) -> pa.DataFrameSchema:
    schema = pa.DataFrameSchema({
        "age": pa.Column("float64", nullable=True),
        "grade": pa.Column("int64", nullable=True),
    }, strict=True)
    path = output_root / "cleaning" / "schema_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema.to_json(path, indent=4)
    return schema


def test_validate_unimputed_handoff_loads_validates_and_returns_the_schema(
    tmp_output, capsys,
):
    """One call replaces the notebook's path-build + validate + print.

    The schema comes back because the modelling notebook validates the MICE
    draws against the same object later on.
    """
    _write_schema(tmp_output)
    df = pd.DataFrame({"age": [40.0, 50.0], "grade": [1, 2]})

    returned = validate_unimputed_handoff(df, tmp_output)

    assert isinstance(returned, pa.DataFrameSchema)
    assert set(returned.columns) == {"age", "grade"}
    assert "✅" in capsys.readouterr().out


def test_validate_unimputed_handoff_rejects_a_frame_that_breaks_the_schema(tmp_output):
    """A stray column is exactly what a strict handoff schema exists to catch."""
    _write_schema(tmp_output)
    df = pd.DataFrame({"age": [40.0], "grade": [1], "surprise": ["x"]})

    with pytest.raises(pa.errors.SchemaErrors):
        validate_unimputed_handoff(df, tmp_output)


def test_validate_unimputed_handoff_raises_when_the_json_is_missing(tmp_output):
    """A missing schema artifact means cleaning never finished. Say so."""
    with pytest.raises(FileNotFoundError):
        validate_unimputed_handoff(pd.DataFrame({"age": [1.0]}), tmp_output)
