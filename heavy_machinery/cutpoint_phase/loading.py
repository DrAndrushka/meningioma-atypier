"""Step 1 — load the cohort and confirm every patient can be scored.

Duplicate patients are not checked here. The cleaning notebook audits them at
§04 against ``["id", "patient_code", "entry_year"]`` and the audit came back
empty, so re-testing it in this phase would only restate a result that has
already been read.

What is checked is the outcome, because a missing ``high_grade`` is not a
missing *predictor* — it is a patient who cannot be scored at all. Dropping
such rows quietly at step 3 would change the denominator between one table and
the next, and a phase whose whole output is proportions cannot afford that.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTCOME = "high_grade"


class CohortError(Exception):
    """The cohort cannot support the proportions this phase reports."""


def load_cohort(output_root: Path | str = "output") -> pd.DataFrame:
    """Read the unimputed cohort written by the cleaning notebook.

    Unimputed on purpose: steps 2-9 answer their questions on values that were
    actually measured, and the 20 MICE draws come back in step 8 to show what
    the missing scans cost. Loading an imputed frame here would quietly fold
    that cost into every earlier number.
    """
    path = Path(output_root) / "datasets" / "unimputed_df.parquet"
    if not path.exists():
        raise CohortError(
            f"No cohort at {path} — run meningioma-cleaning.ipynb first.")
    return pd.read_parquet(path)


def check_outcome_complete(df: pd.DataFrame) -> None:
    """Raise unless every patient has a known grade."""
    if OUTCOME not in df.columns:
        raise CohortError(f"No {OUTCOME!r} column — cannot score any cut-point.")

    unknown = int(df[OUTCOME].isna().sum())
    if unknown:
        raise CohortError(
            f"{unknown} patient(s) have no {OUTCOME}. Exclude them in cleaning "
            "so one denominator holds across the whole phase.")


def cohort_facts(df: pd.DataFrame) -> dict[str, object]:
    """The four numbers every table in this phase is a denominator of."""
    n = len(df)
    events = int(df[OUTCOME].sum())
    return {
        "patients": n,
        "high_grade": events,
        "low_grade": n - events,
        "prevalence": round(events / n, 4) if n else float("nan"),
    }


def describe(facts: dict[str, object]) -> str:
    """One line for the notebook, in the words the manuscript will use."""
    return (
        f"{facts['patients']} patients — "
        f"{facts['high_grade']} high grade (WHO 2-3), "
        f"{facts['low_grade']} low grade (WHO 1), "
        f"{float(facts['prevalence']):.1%} high grade.")
