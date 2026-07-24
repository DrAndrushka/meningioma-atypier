"""Tests for config/cohort.py — load + year filter logging."""

from __future__ import annotations

import pandas as pd
import pytest

from config import load

_cohort = load("cohort")


def test_apply_cohort_year_filter_subset_logs_drops():
    df = pd.DataFrame({"entry_year": [2024, 2025, 2025, 2026], "id": [1, 2, 3, 4]})
    out, log = _cohort.apply_cohort_year_filter(df, "entry_year", [2025])
    assert len(out) == 2
    assert list(out["id"]) == [2, 3]
    assert len(log) == 1
    assert log.iloc[0]["name"] == "cohort year"
    assert log.iloc[0]["rows_before"] == 4
    assert log.iloc[0]["rows_after"] == 2
    assert log.iloc[0]["rows_removed"] == 2
    assert "entry_year ∈" in log.iloc[0]["note"]


def test_apply_cohort_year_filter_all_years_zero_drops():
    df = pd.DataFrame({"entry_year": [2024, 2025], "id": [1, 2]})
    out, log = _cohort.apply_cohort_year_filter(df, "entry_year", None)
    assert len(out) == 2
    assert log.iloc[0]["rows_removed"] == 0
    assert "all years" in log.iloc[0]["note"]


def test_apply_cohort_year_filter_empty_list_raises():
    df = pd.DataFrame({"entry_year": [2025], "id": [1]})
    with pytest.raises(ValueError, match="ANALYSIS_YEARS is"):
        _cohort.apply_cohort_year_filter(df, "entry_year", [])


def test_filter_cohort_still_returns_frame_only():
    df = pd.DataFrame({"entry_year": [2024, 2025], "id": [1, 2]})
    out = _cohort.filter_cohort(df, "entry_year", [2025])
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1
