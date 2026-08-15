"""Step 8 — do the scans we never measured move the cut-point?

ADC is missing for 43 patients, one in eight. Steps 3-7 simply set them aside,
which quietly assumes the missing scans would have looked like the ones we have.
That assumption is usually wrong in a way that matters: a scan is more often
absent because the tumour was small, or the study was old, or the sequence
failed — none of which is unrelated to grade.

Multiple imputation answers it by filling the gaps twenty different plausible
ways, from the other columns, and running everything twenty times. Where the
twenty answers agree, the missing scans were not deciding anything. Where they
scatter, they were.

Two different questions get two different answers here, and conflating them is
the mistake this module exists to prevent:

**Agreement across imputations** — derive the cut-point separately in each of
the twenty filled-in cohorts and look at the spread. This asks *"how much does
the answer depend on how we filled the gaps?"* It says nothing about sampling.

**The joint interval** — resample patients *and* pick an imputation at random in
the same replicate. This asks *"how much would the number move at another
hospital, allowing for both the patients being different and the gaps being
filled differently?"* It is the wider, honest interval, and it is the one to
quote.

A tight joint interval with wide across-imputation spread is a missing-data
story. A wide joint interval with tight agreement is a sample-size story. The
paper needs to know which it has.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from criteria import YOUDEN, select, sweep
from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import auc_with_ci
from wobble import MIN_PER_ARM, N_BOOTSTRAP, SEED

OUTCOME = "high_grade"


class ImputationError(Exception):
    """The imputed draws are absent or inconsistent with the cohort."""


def load_draws(output_root: Path | str = "output") -> list[pd.DataFrame]:
    """The m MICE draws written by the cleaning notebook.

    Read straight from disk rather than regenerated: imputation is the slowest
    step in the whole project and it belongs to the cleaning phase, not this one.
    """
    folder = Path(output_root) / "missingness" / "mice"
    paths = sorted(folder.glob("imputed_*.parquet"))
    if not paths:
        raise ImputationError(
            f"No imputed draws under {folder} — run the cleaning notebook's "
            "MICE step, or report this phase as complete-case only.")
    return [pd.read_parquet(p) for p in paths]


def check_derived_consistency(draws: Sequence[pd.DataFrame], *,
                              tolerance: float = 1e-6) -> list[str]:
    """Confirm the edema index was rebuilt per draw, not imputed on its own.

    A ratio has to be recomputed from its two imputed parents inside each draw.
    Imputing it directly lets a draw carry an index that contradicts the volumes
    sitting beside it in the same row — and nothing downstream would notice,
    because each column is individually plausible.
    """
    notes = []
    for i, draw in enumerate(draws, start=1):
        needed = {"edema_index", "edema_volume_cm3", "tumor_volume"}
        if not needed <= set(draw.columns):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rebuilt = draw["edema_volume_cm3"] / draw["tumor_volume"]
        both = np.isfinite(rebuilt) & np.isfinite(draw["edema_index"])
        if both.any() and not np.allclose(draw.loc[both, "edema_index"],
                                          rebuilt[both], rtol=tolerance,
                                          atol=1e-9):
            notes.append(
                f"draw {i}: edema_index does not equal edema_volume / "
                "tumor_volume — it was imputed directly rather than rebuilt "
                "from its parents, so a row can contradict itself.")
    return notes


def _cutpoint_in(frame: pd.DataFrame, m: Measurement, stratum: str,
                 criterion: str) -> float:
    sub = frame.loc[stratum_mask(frame, m, stratum)]
    x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
    y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    if y.sum() < MIN_PER_ARM or (y.size - y.sum()) < MIN_PER_ARM:
        return float("nan")
    return select(sweep(y, x, m.direction), criterion,
                  auc=auc_with_ci(y, x, m.direction)["auc"])


def per_draw_cutpoints(draws: Sequence[pd.DataFrame], eligible: pd.DataFrame, *,
                       criterion: str = YOUDEN,
                       frozen: dict[str, float] | None = None) -> pd.DataFrame:
    """The cut-point derived independently in each draw, summarised.

    Rubin's rules do not apply here. They average an estimate and its variance;
    a cut-point is the position of a maximum, and averaging positions across
    draws can land somewhere no draw actually chose. The median with the full
    range is the honest summary.

    ``diverges`` is the gate: a frozen cut-point that falls outside the range
    every draw produced is an artefact of the complete-case derivation that no
    imputation reproduces. That is a finding, not a formatting problem, and it
    has to surface before a table is written rather than after review.
    """
    rows = []
    for _, row in eligible.iterrows():
        m = MEASUREMENTS_BY_COL[row["col"]]
        values = np.array([_cutpoint_in(d, m, row["stratum"], criterion)
                           for d in draws], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        lo, hi = m.round(values.min()), m.round(values.max())
        median = m.round(np.median(values))
        entry = {
            "measurement": row["measurement"], "col": row["col"],
            "stratum": row["stratum"], "claim": row.get("claim", ""),
            "n_draws": int(values.size),
            "draw_median": median, "draw_min": lo, "draw_max": hi,
            "draw_spread": float(hi - lo),
        }
        # Always written, even with no frozen value to check against, so the
        # column stays boolean. A column that is False for some rows and absent
        # for others cannot be filtered on without special-casing every caller.
        observed = (frozen or {}).get(row["col"], np.nan)
        entry["observed_cutpoint"] = observed
        # Compared at the precision the table prints, so a median that rounds
        # just outside a very tight range is not called divergent.
        entry["diverges"] = bool(np.isfinite(observed)
                                 and (observed < lo or observed > hi))
        entry["shift_from_observed"] = (float(median - observed)
                                        if np.isfinite(observed) else np.nan)
        rows.append(entry)
    return pd.DataFrame(rows)


def joint_bootstrap(draws: Sequence[pd.DataFrame], m: Measurement,
                    stratum: str = "all", *, criterion: str = YOUDEN,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED,
                    alpha: float = 0.05) -> dict[str, object]:
    """Resample patients *and* the imputation together, in the same replicate.

    Drawing an imputation index alongside the patient sample propagates both
    sources of uncertainty into one interval. Picking a single "representative"
    draw and bootstrapping inside it would discard the between-imputation
    variance entirely — and it is exactly the quantity this step exists to
    measure, so the resulting interval would be too narrow on the one number the
    argument rests on.
    """
    prepared = []
    for frame in draws:
        sub = frame.loc[stratum_mask(frame, m, stratum)]
        x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        prepared.append((y[ok].astype(int), x[ok]))
    prepared = [(y, x) for y, x in prepared if y.size]
    if not prepared:
        return {"ci_lo": np.nan, "ci_hi": np.nan, "n_valid": 0,
                "n_skipped": int(n_boot), "draws": np.array([], dtype=float)}

    rng = np.random.default_rng(seed)
    picks, skipped = [], 0
    for _ in range(int(n_boot)):
        y_all, x_all = prepared[rng.integers(0, len(prepared))]
        n = y_all.size
        idx = rng.integers(0, n, n)
        y_b, x_b = y_all[idx], x_all[idx]
        if y_b.sum() < MIN_PER_ARM or (n - y_b.sum()) < MIN_PER_ARM:
            skipped += 1
            continue
        try:
            c_b = select(sweep(y_b, x_b, m.direction), criterion,
                         auc=auc_with_ci(y_b, x_b, m.direction)["auc"])
        except Exception:
            skipped += 1
            continue
        if np.isfinite(c_b):
            picks.append(float(c_b))
        else:
            skipped += 1
    if not picks:
        return {"ci_lo": np.nan, "ci_hi": np.nan, "n_valid": 0,
                "n_skipped": skipped, "draws": np.array([], dtype=float)}
    arr = np.asarray(picks, dtype=float)
    return {"ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_valid": int(arr.size), "n_skipped": int(skipped), "draws": arr}


def imputation_table(draws: Sequence[pd.DataFrame], eligible: pd.DataFrame,
                     wobble: pd.DataFrame, *, criterion: str = YOUDEN,
                     n_boot: int = N_BOOTSTRAP, seed: int = SEED,
                     frozen: dict[str, float] | None = None) -> pd.DataFrame:
    """Everything step 8 reports, next to step 7's patients-only interval."""
    per_draw = per_draw_cutpoints(draws, eligible, criterion=criterion,
                                  frozen=frozen)
    patients_only = wobble.set_index("col")[["ci_lo", "ci_hi"]]
    rows = []
    for _, row in per_draw.iterrows():
        m = MEASUREMENTS_BY_COL[row["col"]]
        joint = joint_bootstrap(draws, m, row["stratum"], criterion=criterion,
                                n_boot=n_boot, seed=seed)
        entry = dict(row)
        entry["joint_ci_lo"] = m.round(joint["ci_lo"])
        entry["joint_ci_hi"] = m.round(joint["ci_hi"])
        entry["joint_width"] = joint["ci_hi"] - joint["ci_lo"]
        if row["col"] in patients_only.index:
            po = patients_only.loc[row["col"]]
            entry["patients_ci_lo"] = po["ci_lo"]
            entry["patients_ci_hi"] = po["ci_hi"]
            entry["patients_width"] = po["ci_hi"] - po["ci_lo"]
            entry["widening"] = (entry["joint_width"] / entry["patients_width"]
                                 if entry["patients_width"] else np.nan)
        rows.append(entry)
    return pd.DataFrame(rows)


def describe_imputation(table: pd.DataFrame) -> str:
    """One line: what the missing scans cost, and whether anything diverged."""
    if table.empty:
        return "No cut-point could be derived in the imputed draws."
    parts = []
    if "diverges" in table.columns and table["diverges"].any():
        names = ", ".join(table.loc[table["diverges"], "measurement"])
        parts.append(
            f"No imputation reproduces the published cut-point for {names} — "
            "the number is an artefact of the complete-case derivation.")
    if "widening" in table.columns:
        worst = table.sort_values("widening", ascending=False).iloc[0]
        parts.append(
            f"Allowing for the missing scans widens the interval most for "
            f"{worst['measurement']} ({worst['patients_width']:.3g} to "
            f"{worst['joint_width']:.3g}, {worst['widening']:.1f}x).")
    spread = table.sort_values("draw_spread", ascending=False).iloc[0]
    parts.append(
        f"Widest disagreement between imputations: {spread['measurement']} "
        f"{spread['draw_min']:g}-{spread['draw_max']:g}.")
    return " ".join(parts)
