"""Step 1 — the cohort loads, and a cohort that would break the bootstrap does not."""
from __future__ import annotations

import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import loading as ld


def _frame(**overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "high_grade": [0, 1, 0, 1],
        "adc_value": [0.9, 0.6, 0.8, 0.5],
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_clean_cohort_passes():
    ld.check_outcome_complete(_frame())


def test_repeated_patient_is_not_this_phases_business():
    """Deduplication is the cleaning notebook's §04 audit, not a re-check here."""
    ld.check_outcome_complete(_frame(id=[1, 1, 3, 4]))


def test_missing_outcome_is_rejected():
    with pytest.raises(ld.CohortError, match="no high_grade"):
        ld.check_outcome_complete(_frame(high_grade=[0, 1, None, 1]))


def test_absent_outcome_column_is_rejected():
    with pytest.raises(ld.CohortError, match="cannot score"):
        ld.check_outcome_complete(_frame().drop(columns=["high_grade"]))


def test_facts_count_both_arms():
    facts = ld.cohort_facts(_frame())
    assert facts == {"patients": 4, "high_grade": 2, "low_grade": 2,
                     "prevalence": 0.5}


def test_describe_names_both_grades():
    line = ld.describe(ld.cohort_facts(_frame()))
    assert "4 patients" in line and "2 high grade" in line and "50.0%" in line
    assert "one row each" not in line


def test_missing_parquet_names_the_cleaning_notebook(tmp_path):
    with pytest.raises(ld.CohortError, match="meningioma-cleaning"):
        ld.load_cohort(tmp_path)
