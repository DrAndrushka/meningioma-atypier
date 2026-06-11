"""pytest for config/04_row_filters.py"""

from __future__ import annotations

import pandas as pd

from config import load

_c04 = load("04_row_filters")


def test_apply_row_filters_active_and_inactive():
    df = pd.DataFrame({"who_grade": [1, None, 2], "sex": ["male", "female", "unknown"]})
    filters = [
        _c04.RowFilter(
            name="who_grade present",
            keep=lambda d: d["who_grade"].notna(),
            note="Keep rows where who_grade is not missing",
            active=True,
        ),
        _c04.RowFilter(
            name="sex known",
            keep=lambda d: d["sex"] != "unknown",
            note="Keep rows where sex is known",
            active=False,
        ),
    ]
    out, log = _c04.apply_row_filters(df, filters)
    assert len(out) == 2
    assert log.iloc[0]["rows_before"] == 3
    assert log.iloc[0]["rows_after"] == 2
    assert log.iloc[0]["rows_removed"] == 1
    assert log.iloc[1]["active"] is False
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
    drop_log = _c04._drop_log_for_export(log)
    assert drop_log[0]["n_before"] == 10
    assert drop_log[0]["n_remaining"] == 8
    assert drop_log[0]["n_dropped"] == 2


def test_keep_brain_meningioma_on_raw_strings():
    df = pd.DataFrame({
        "side": ["1", _c04.SPINAL_MENINGIOMA_LABEL, "2"],
        "mri_date": ["03.01.2025.", "03.01.2025.", _c04.SPINAL_MENINGIOMA_LABEL],
    })
    out, log = _c04.apply_row_filters(df, [_c04.brain_meningioma_row_filter()])
    assert len(out) == 1
    assert out.iloc[0]["side"] == "1"
    assert log.iloc[0]["rows_removed"] == 2


def test_combine_row_filter_logs():
    pre = pd.DataFrame([{"name": "pre", "rows_removed": 4}])
    post = pd.DataFrame([{"name": "post", "rows_removed": 1}])
    combined = _c04.combine_row_filter_logs(pre, post)
    assert list(combined["name"]) == ["pre", "post"]
