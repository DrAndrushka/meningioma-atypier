"""Tests for config/row_filters.py — cohort inclusion masks."""

from __future__ import annotations

import pandas as pd

from config import load

_row_filters = load("row_filters")


def test_apply_row_filters_active_and_inactive():
    df = pd.DataFrame({"who_grade": [1, None, 2], "sex": ["male", "female", "unknown"]})
    filters = [
        _row_filters.RowFilter(
            name="who_grade present",
            keep=lambda d: d["who_grade"].notna(),
            note="Keep rows where who_grade is not missing",
            active=True,
        ),
        _row_filters.RowFilter(
            name="sex known",
            keep=lambda d: d["sex"] != "unknown",
            note="Keep rows where sex is known",
            active=False,
        ),
    ]
    out, log = _row_filters.apply_row_filters(df, filters)
    assert len(out) == 2
    assert log.iloc[0]["rows_before"] == 3
    assert log.iloc[0]["rows_after"] == 2
    assert log.iloc[0]["rows_removed"] == 1
    assert log.iloc[1]["active"] == False
    assert log.iloc[1]["rows_removed"] == 0


def test_drop_log_for_export():
    log = pd.DataFrame([{
        "name": "who_grade present",
        "active": True,
        "rows_before": 10,
        "rows_after": 8,
        "rows_removed": 2,
        "note": "Keep rows where who_grade is not missing",
    }])
    drop_log = _row_filters._drop_log_for_export(log)
    assert drop_log[0]["n_before"] == 10
    assert drop_log[0]["n_remaining"] == 8
    assert drop_log[0]["n_dropped"] == 2

