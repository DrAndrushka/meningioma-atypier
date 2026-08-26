"""Step 7 — how much would this cut-point move if we had different patients?

A cut-point derived by trying every candidate and keeping the best is
optimistically biased. It was chosen *because* it looked good on these 352
patients, and some of how good it looked was luck. Two consequences follow, and
this step measures both.

**The cut-point itself has a confidence interval.** Not its sensitivity — the
*number*. Resample the cohort with replacement, re-derive the cut-point from
scratch inside each resample, and the spread of those values is the interval.
Re-deriving is the whole point: evaluating one fixed cut-point across resamples
measures how stable its performance is, not how stable the number is, and it is
the number the manuscript prints.

**Performance shrinks when the cut-point meets new patients.** Each resample
leaves some patients out of the bag. Deriving the cut-point on the patients who
were drawn and scoring it on those who were not gives the drop honestly:
``optimism = J on the patients used − J on the patients held out``. Subtracting
it gives what to expect at another hospital.

The interval is read against the measurement's own spread. A cut-point whose
interval is narrower than the middle half of the distribution is a landmark. One
whose interval spans that whole middle half is a coin flip dressed as a number,
and the manuscript has to say so.

The five headline cut-points are frozen: this step produces the interval only,
never a new point estimate. Enforced by assertion, not by convention.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from criteria import YOUDEN, select, sweep
from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import auc_with_ci

OUTCOME = "high_grade"

# Shared with every other phase in this project so the resampling lines up.
SEED = 20260801
N_BOOTSTRAP = 1000

# A resample with too few held-out patients, or too few of either grade, cannot
# score a cut-point honestly. Skipped rather than counted, and the skips are
# reported so a run that quietly lost half its replicates is visible.
MIN_HELD_OUT = 20
MIN_PER_ARM = 5


class FrozenCutpointError(Exception):
    """The bootstrap moved a point estimate it was only allowed to bracket."""


def _youden_at(y: np.ndarray, x: np.ndarray, cutoff: float,
               direction: str) -> float:
    """Youden J of one fixed cut-point on one set of patients."""
    from accuracy import confusion, flag
    if not np.isfinite(cutoff) or y.size == 0:
        return np.nan
    k = confusion(y, flag(x, cutoff, direction))
    n_pos, n_neg = k["tp"] + k["fn"], k["fp"] + k["tn"]
    if not n_pos or not n_neg:
        return np.nan
    return k["tp"] / n_pos + k["tn"] / n_neg - 1.0


def bootstrap_cutpoint(y: np.ndarray, x: np.ndarray, direction: str, *,
                       criterion: str = YOUDEN, n_boot: int = N_BOOTSTRAP,
                       seed: int = SEED, alpha: float = 0.05
                       ) -> dict[str, object]:
    """Re-derive the cut-point in ``n_boot`` resampled cohorts.

    Returns the percentile interval, the optimism, and the draws themselves —
    the draws are kept because their histogram says more than any summary of
    them: a bimodal spread means the criterion is flipping between two distant
    candidates, which no interval width can convey.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    n = y.size

    blank = {"cutpoint": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
             "iqr_lo": np.nan, "iqr_hi": np.nan, "optimism": np.nan,
             "j_apparent": np.nan, "j_corrected": np.nan,
             "n_valid": 0, "n_skipped": int(n_boot),
             "draws": np.array([], dtype=float)}
    if n == 0 or y.sum() < MIN_PER_ARM or (n - y.sum()) < MIN_PER_ARM:
        return blank

    full = sweep(y, x, direction)
    auc_full = auc_with_ci(y, x, direction)["auc"]
    observed = select(full, criterion, auc=auc_full)
    if not np.isfinite(observed):
        return blank

    rng = np.random.default_rng(seed)
    draws, gaps = [], []
    skipped = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        y_in, x_in = y[idx], x[idx]
        # Out-of-bag mask instead of setdiff1d(arange, unique(idx)): the drawn
        # indices are 0..n-1, so membership is a boolean array and the two
        # sorts np.unique/np.setdiff1d run per resample are not needed. Same
        # values, same ascending order.
        in_bag = np.zeros(n, dtype=bool)
        in_bag[idx] = True
        held = np.flatnonzero(~in_bag)
        if (held.size < MIN_HELD_OUT or y_in.sum() < MIN_PER_ARM
                or (n - y_in.sum()) < MIN_PER_ARM
                or y[held].sum() < 1 or y[held].size - y[held].sum() < 1):
            skipped += 1
            continue
        try:
            # The AUC is re-derived too. Index-of-union is defined against it,
            # so reusing the whole-cohort AUC would leak the full sample into a
            # replicate that is supposed to know nothing about it.
            c_b = select(sweep(y_in, x_in, direction), criterion,
                         auc=auc_with_ci(y_in, x_in, direction)["auc"])
        except Exception:
            skipped += 1
            continue
        if not np.isfinite(c_b):
            skipped += 1
            continue
        draws.append(float(c_b))
        j_in = _youden_at(y_in, x_in, c_b, direction)
        j_out = _youden_at(y[held], x[held], c_b, direction)
        if np.isfinite(j_in) and np.isfinite(j_out):
            gaps.append(j_in - j_out)

    if not draws:
        return blank
    draws_arr = np.asarray(draws, dtype=float)
    j_apparent = _youden_at(y, x, observed, direction)
    optimism = float(np.mean(gaps)) if gaps else np.nan
    return {
        "cutpoint": float(observed),
        "ci_lo": float(np.percentile(draws_arr, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(draws_arr, 100 * (1 - alpha / 2))),
        "iqr_lo": float(np.percentile(draws_arr, 25)),
        "iqr_hi": float(np.percentile(draws_arr, 75)),
        "optimism": optimism,
        "j_apparent": float(j_apparent),
        "j_corrected": (float(j_apparent - optimism)
                        if np.isfinite(optimism) else np.nan),
        "n_valid": int(draws_arr.size),
        "n_skipped": int(skipped),
        "draws": draws_arr,
    }


def wobble_table(df: pd.DataFrame, eligible: pd.DataFrame, *,
                 criterion: str = YOUDEN, n_boot: int = N_BOOTSTRAP,
                 seed: int = SEED, frozen: dict[str, float] | None = None
                 ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Bootstrap every eligible measurement. Returns the table and the raw draws.

    ``frozen`` maps a column to the cut-point the manuscript already prints. Any
    disagreement raises: this step is allowed to bracket a published number, not
    to move it.
    """
    rows, draws = [], {}
    for _, row in eligible.iterrows():
        m: Measurement = MEASUREMENTS_BY_COL[row["col"]]
        sub = df.loc[stratum_mask(df, m, row["stratum"])]
        out = bootstrap_cutpoint(
            pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy(),
            pd.to_numeric(sub[m.col], errors="coerce").to_numpy(),
            m.direction, criterion=criterion, n_boot=n_boot, seed=seed)
        draws[row["col"]] = out.pop("draws")

        if frozen and row["col"] in frozen and np.isfinite(out["cutpoint"]):
            expected = frozen[row["col"]]
            got = m.round(out["cutpoint"])
            if not np.isclose(got, expected):
                raise FrozenCutpointError(
                    f"{m.label}: bootstrap re-derived {got} but the manuscript "
                    f"prints {expected}. The interval may move; the point "
                    "estimate may not.")

        values = pd.to_numeric(sub[m.col], errors="coerce").dropna()
        iqr = (float(values.quantile(0.75) - values.quantile(0.25))
               if len(values) else np.nan)
        width = out["ci_hi"] - out["ci_lo"]
        rows.append({
            "measurement": row["measurement"],
            "col": row["col"], "stratum": row["stratum"],
            "claim": row.get("claim", ""),
            "cutpoint": m.round(out["cutpoint"]),
            "ci_lo": m.round(out["ci_lo"]), "ci_hi": m.round(out["ci_hi"]),
            "iqr_lo": m.round(out["iqr_lo"]), "iqr_hi": m.round(out["iqr_hi"]),
            "ci_width": width,
            "measurement_iqr": iqr,
            "stability_ratio": width / iqr if iqr else np.nan,
            "j_apparent": out["j_apparent"],
            "optimism": out["optimism"],
            "j_corrected": out["j_corrected"],
            "n_valid": out["n_valid"], "n_skipped": out["n_skipped"],
        })
    return pd.DataFrame(rows), draws


# A cut-point whose interval is wider than the middle half of the measurement's
# own distribution is not locating anything the data can distinguish.
STABILITY_LIMIT = 1.0


def describe_wobble(table: pd.DataFrame) -> str:
    """One line: which cut-points survived resampling, and which dissolved."""
    if table.empty:
        return "No cut-point could be resampled."
    firm = table[table["stability_ratio"] <= STABILITY_LIMIT]
    loose = table[table["stability_ratio"] > STABILITY_LIMIT]
    parts = []
    if not firm.empty:
        named = "; ".join(
            f"{r['measurement']} {r['cutpoint']:g} ({r['ci_lo']:g}-{r['ci_hi']:g})"
            for _, r in firm.iterrows())
        parts.append(f"Holds up under resampling — {named}.")
    if not loose.empty:
        named = "; ".join(
            f"{r['measurement']} {r['cutpoint']:g} "
            f"({r['ci_lo']:g}-{r['ci_hi']:g}, {r['stability_ratio']:.1f}x its IQR)"
            for _, r in loose.iterrows())
        parts.append(f"Interval wider than the measurement's own middle half, "
                     f"so the number is not a landmark — {named}.")
    return " ".join(parts)
