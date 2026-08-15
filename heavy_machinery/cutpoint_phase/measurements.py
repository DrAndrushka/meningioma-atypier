"""Step 2 — the five measurements: what they are, how they spread, what is missing.

One :class:`Measurement` per continuous MRI number, declaring four things the
rest of the notebook reads rather than infers:

``direction``  which way the measurement points at high grade. Declared, never
               detected. Detecting it from the data guarantees every AUC comes
               out at or above 0.5, which turns a genuine null result into a
               weak-looking positive one.
``log_x``      right-skewed, so it is log-transformed before anything that
               assumes a straight line. Volumes here span three orders of
               magnitude; on the raw scale a model spends all its flexibility
               on a handful of very large tumours.
``decimals``   the precision the manuscript prints. Fixed here so a number
               cannot be rounded one way in a table and another in a figure.
``unit``       written twice, because the same string cannot serve both. Tables
               and sentences need plain text; matplotlib axes need mathtext for
               the superscripts in the ADC unit.
``zero_inflated``
               a large share of patients sit at exactly zero, so the
               measurement is asked twice — see below.

A third of this cohort has no edema at all. Asked of everyone, "does edema
volume predict high grade?" quietly bundles two different claims: that *having*
edema matters, and that *how much* edema matters. They can point in opposite
directions, and a single cut-point cannot tell them apart.

So a zero-inflated measurement is analysed in two strata:

``all``      every patient with the measurement — presence and amount together
``present``  only patients above zero — amount alone, presence held constant

Whichever stratum the manuscript quotes, it has to name it. "Edema volume
≥ 4.76 cm³" read on the whole cohort is mostly a statement about having edema;
read on the present-only stratum it is a statement about how much.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from scales import disagreements

HIGHER = "higher"
LOWER = "lower"
_DIRECTIONS = (HIGHER, LOWER)

STRATUM_ALL = "all"
STRATUM_PRESENT = "present"

# Above this share of observed values at exactly zero, a measurement has to be
# declared zero-inflated. A reporting threshold, not a statistical one: below
# roughly one patient in ten the pile does not change how a cut-point reads.
ZERO_PILE_FLOOR_PCT = 10.0


class Measurement(NamedTuple):
    """One continuous MRI measurement and how it points at high grade."""

    col: str
    label: str
    unit: str
    axis_unit: str
    direction: str
    decimals: int
    log_x: bool = False
    zero_inflated: bool = False
    short: str = ""

    @property
    def short_label(self) -> str:
        """Name for a figure legend, where a panel is barely two inches wide."""
        return self.short or self.label

    @property
    def op(self) -> str:
        """The comparison that flags a patient as suspicious at a cut-point."""
        return "≤" if self.direction == LOWER else "≥"

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.axis_unit})" if self.axis_unit else self.label

    def rule_text(self, cutoff: float) -> str:
        """Human-readable rule, e.g. ``ADC (mean) ≤ 0.72``."""
        return f"{self.label} {self.op} {cutoff:.{self.decimals}f}"

    def round(self, value: float) -> float:
        """Round to the precision the manuscript prints."""
        return float(np.round(float(value), self.decimals))

    @property
    def strata(self) -> tuple[str, ...]:
        """The stratum names this measurement is analysed in."""
        return ((STRATUM_ALL, STRATUM_PRESENT) if self.zero_inflated
                else (STRATUM_ALL,))

    def stratum_label(self, stratum: str) -> str:
        """How a stratum is named in a table or a figure caption."""
        if stratum == STRATUM_PRESENT:
            return f"{self.label} (where present)"
        return self.label


# Directions and log flags carried over unchanged from the previous phase — the
# science did not change, only where it is written down.
MEASUREMENTS: tuple[Measurement, ...] = (
    # ``unit`` is plain text for tables and sentences; ``axis_unit`` is mathtext
    # for anything drawn by matplotlib. Arial has no superscript-minus glyph, so
    # the plain form of the ADC unit renders as an empty box in a figure —
    # figures must use ``axis_unit``, never ``unit``.
    Measurement("adc_value", "ADC (mean)", "×10⁻³ mm²/s",
                r"$\times 10^{-3}$ mm$^2$/s", LOWER, decimals=2, short="ADC"),
    Measurement("tumor_volume", "Tumor volume", "cm³",
                "cm$^3$", HIGHER, decimals=1, log_x=True, short="Tumor vol."),
    Measurement("edema_volume_cm3", "Edema volume", "cm³",
                "cm$^3$", HIGHER, decimals=2, log_x=True, zero_inflated=True,
                short="Edema vol."),
    Measurement("edema_index", "Edema index", "",
                "edema ÷ tumor", HIGHER, decimals=4, log_x=True,
                zero_inflated=True, short="Edema idx"),
    Measurement("max_diameter_cm", "Max diameter", "cm",
                "cm", HIGHER, decimals=2, short="Max diam."),
)

MEASUREMENTS_BY_COL = {m.col: m for m in MEASUREMENTS}


class MeasurementError(Exception):
    """A measurement is declared wrong, or the cohort does not carry it."""


def _check_log_flags_match_shared_declaration() -> None:
    """Fail at import if this phase disagrees with :mod:`scales` about the scale.

    These flags used to be restated here and in the modelling phase, and they
    drifted apart: the same cohort published OR 1.96 for tumor volume on one
    page and 1.73 on another. Restating them is still convenient — the docstring
    above explains them where they are read — so the copy is checked rather than
    trusted.
    """
    mismatched = disagreements({m.col: m.log_x for m in MEASUREMENTS})
    if mismatched:
        raise MeasurementError(
            f"log_x disagrees with scales.LOG1P_COLUMNS for: "
            f"{', '.join(mismatched)} — change scales.LOG1P_COLUMNS, which "
            f"every phase reads, rather than one of the copies."
        )


_check_log_flags_match_shared_declaration()


def validate(measurements: Sequence[Measurement], df: pd.DataFrame) -> None:
    """Raise on a bad direction or an absent column.

    Both failures are silent otherwise: a mistyped direction sails through as a
    reversed cut-point, and an absent column would simply drop a row from every
    table without ever saying which one.
    """
    for m in measurements:
        if m.direction not in _DIRECTIONS:
            raise MeasurementError(
                f"{m.col}: direction {m.direction!r} is not "
                f"{HIGHER!r} or {LOWER!r}.")
        if m.col not in df.columns:
            raise MeasurementError(f"{m.col}: not a column in the cohort.")


def check_zero_declarations(df: pd.DataFrame,
                            measurements: Sequence[Measurement] = MEASUREMENTS
                            ) -> list[str]:
    """Warn where the data and the ``zero_inflated`` declaration disagree.

    Returned rather than raised. A pile that grows past the floor as the cohort
    accrues is a finding to act on deliberately, not a reason to halt a run —
    but it must not pass unremarked, because an undeclared pile silently turns
    an "amount" claim into a "presence" one.
    """
    notes = []
    for m in measurements:
        observed = pd.to_numeric(df[m.col], errors="coerce").dropna()
        if not len(observed):
            continue
        pct = 100 * float((observed == 0).mean())
        if pct >= ZERO_PILE_FLOOR_PCT and not m.zero_inflated:
            notes.append(
                f"{m.label}: {pct:.1f}% of observed values are zero but "
                "zero_inflated=False — a cut-point here is measuring presence "
                "as much as amount. Declare it, or say why not.")
        elif pct < ZERO_PILE_FLOOR_PCT and m.zero_inflated:
            notes.append(
                f"{m.label}: only {pct:.1f}% of observed values are zero, yet "
                "zero_inflated=True — the 'where present' stratum now costs "
                "sample size for nothing.")
    return notes


def stratum_mask(df: pd.DataFrame, m: Measurement, stratum: str) -> pd.Series:
    """Which patients belong to a stratum, as a boolean over ``df``'s index.

    Missing values are excluded from both strata rather than treated as zero.
    An unmeasured edema volume is not the same claim as a measured absence of
    edema, and collapsing them would inflate the absent group with patients
    nobody looked at.
    """
    values = pd.to_numeric(df[m.col], errors="coerce")
    observed = values.notna()
    if stratum == STRATUM_PRESENT:
        return observed & (values > 0)
    if stratum == STRATUM_ALL:
        return observed
    raise MeasurementError(
        f"Unknown stratum {stratum!r} — expected {STRATUM_ALL!r} or "
        f"{STRATUM_PRESENT!r}.")


def presence_table(df: pd.DataFrame,
                   measurements: Sequence[Measurement] = MEASUREMENTS
                   ) -> pd.DataFrame:
    """For each zero-inflated measurement, how big the two strata are.

    No outcome here — this is the denominator each later table will be read
    against. Whether presence itself predicts high grade is step 3's question.
    """
    rows = []
    for m in measurements:
        if not m.zero_inflated:
            continue
        observed = int(stratum_mask(df, m, STRATUM_ALL).sum())
        present = int(stratum_mask(df, m, STRATUM_PRESENT).sum())
        rows.append({
            "measurement": m.label,
            "n_observed": observed,
            "n_absent": observed - present,
            "n_present": present,
            "pct_present": (round(100 * present / observed, 1) if observed
                            else np.nan),
        })
    return pd.DataFrame(rows)


def spread_table(df: pd.DataFrame,
                 measurements: Sequence[Measurement] = MEASUREMENTS) -> pd.DataFrame:
    """One row per measurement: how many we have, and how they are spread.

    Median and IQR rather than mean and SD, because three of the five are
    right-skewed and a mean would sit above most of the cohort.

    ``pct_zero`` earns its column because two of the five measurements pile up
    at exactly zero — a patient with no edema at all. Where that pile is large,
    a cut-point is not really measuring *how much* edema there is; it is mostly
    separating patients who have some from patients who have none, and the
    manuscript has to say which of the two it means.
    """
    rows = []
    for m in measurements:
        values = pd.to_numeric(df[m.col], errors="coerce")
        observed = values.dropna()
        n_missing = int(values.isna().sum())
        n_zero = int((observed == 0).sum())
        q1, q3 = (observed.quantile([0.25, 0.75]) if len(observed)
                  else (np.nan, np.nan))
        rows.append({
            "measurement": m.label,
            "unit": m.unit,
            "direction": f"{m.op} cut-point flags high grade",
            "n_observed": len(observed),
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / len(df), 1) if len(df) else np.nan,
            "n_zero": n_zero,
            "pct_zero": (round(100 * n_zero / len(observed), 1) if len(observed)
                         else np.nan),
            "median": m.round(observed.median()) if len(observed) else np.nan,
            "iqr_low": m.round(q1) if len(observed) else np.nan,
            "iqr_high": m.round(q3) if len(observed) else np.nan,
            "min": m.round(observed.min()) if len(observed) else np.nan,
            "max": m.round(observed.max()) if len(observed) else np.nan,
            "log_transformed": m.log_x,
        })
    return pd.DataFrame(rows)


def describe_missing(table: pd.DataFrame) -> str:
    """One line naming the measurement that will cost us the most.

    Missingness is not a footnote here: it decides which patients each cut-point
    is estimated from, and it is why step 8 exists at all.
    """
    if table.empty:
        return "No measurements to summarise."
    complete = table[table["n_missing"] == 0]["measurement"].tolist()
    incomplete = table[table["n_missing"] > 0].sort_values(
        "n_missing", ascending=False)
    if incomplete.empty:
        return "Every measurement is complete for every patient."
    worst = incomplete.iloc[0]
    n_worst = int(worst["n_missing"])
    line = (f"{worst['measurement']} is missing for {n_worst} "
            f"{'patient' if n_worst == 1 else 'patients'} "
            f"({worst['pct_missing']}%) — the widest gap")
    if len(incomplete) > 1:
        others = ", ".join(
            f"{r['measurement']} {int(r['n_missing'])}"
            for _, r in incomplete.iloc[1:].iterrows())
        line += f"; then {others}"
    line += "."
    if complete:
        line += f" Complete for every patient: {', '.join(complete)}."
    return line


def describe_zeros(table: pd.DataFrame, *, floor: float = 10.0) -> str:
    """Name any measurement with a large pile of patients at exactly zero.

    ``floor`` is a reporting threshold, not a statistical one: below roughly one
    patient in ten the pile does not change how a cut-point should be read.
    """
    if table.empty or "pct_zero" not in table.columns:
        return "No measurements to summarise."
    piled = table[table["pct_zero"] >= floor].sort_values(
        "pct_zero", ascending=False)
    if piled.empty:
        return f"No measurement has more than {floor:.0f}% of patients at zero."
    named = "; ".join(
        f"{r['measurement']} {int(r['n_zero'])}/{int(r['n_observed'])} "
        f"({r['pct_zero']}%)" for _, r in piled.iterrows())
    return (f"Patients sitting at exactly zero — {named}. These are analysed "
            "twice: across all patients (presence and amount together) and "
            "among those above zero (amount alone).")
