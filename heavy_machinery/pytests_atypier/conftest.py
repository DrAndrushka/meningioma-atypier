"""Shared pytest fixtures for ``heavy_machinery/pytests_atypier/``.

Tiny dataframe + schema for cleaning/modelling tests; ``tmp_output`` as a stand-in
for repo-root ``output/``.
"""

from __future__ import annotations

import os

# joblib/loky workers on Windows default to cp1252 stdout; MICE progress prints use emoji.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

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


_REAL_COHORT = (Path(__file__).resolve().parents[2]
                / "output" / "datasets" / "unimputed_df.parquet")


@pytest.fixture(scope="session")
def real_cohort() -> pd.DataFrame:
    """The actual cleaned cohort, for tests that freeze a published number.

    Skipped rather than failed when ``output/`` has not been built. A fresh
    clone has no cohort to read, and a test that cannot run is not a test that
    failed — but on a machine that *has* run the cleaning notebook, a frozen
    number that moves must be loud.
    """
    if not _REAL_COHORT.exists():
        pytest.skip("no output/datasets/unimputed_df.parquet — run cleaning first")
    return pd.read_parquet(_REAL_COHORT)
