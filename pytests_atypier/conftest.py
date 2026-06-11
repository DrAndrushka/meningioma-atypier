"""tiny df + schema fixtures for pytest"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from schema_infer import ColSpec


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "age": [45.0, 55.0, 65.0, np.nan],
        "sex": ["M", "F", "M", "F"],
        "grade": [1, 2, 1, 2],
        "event": [True, False, True, False],
        "entry_year": pd.to_datetime(["2018-01-01", "2019-06-01", "2020-03-01", "2021-01-01"]),
        "note": ["x", "y", "z", "w"],
    })


@pytest.fixture
def tiny_schema(tiny_df) -> dict[str, ColSpec]:
    return {
        "id": ColSpec("id", "id", keep=False),
        "age": ColSpec("age", "continuous"),
        "sex": ColSpec("sex", "nominal"),
        "grade": ColSpec("grade", "ordinal", ordered_levels=[1, 2, 3]),
        "event": ColSpec("event", "binary"),
        "entry_year": ColSpec("entry_year", "datetime"),
        "note": ColSpec("note", "text"),
    }


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out
