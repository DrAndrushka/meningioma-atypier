"""Step 3 — does each measurement separate grade 1 from grade 2-3 at all?

The answer is the **AUC**: line the patients up by one measurement, pick a
high-grade and a low-grade patient at random, and ask how often the high-grade
one is on the suspicious side. 0.50 is a coin flip — the measurement carries no
information. 1.00 is perfect separation.

The interval comes from **DeLong**, computed exactly rather than by resampling.
The estimator itself lives in :mod:`delong`, because ``report.html`` publishes
the same univariate AUCs in its EDA table and the two pages have to print one
interval per AUC. What stays here is how this phase applies it.

*Direction is applied, not discovered.* ADC is negated before scoring, because
low ADC is the suspicious side. Letting the code choose the orientation would
floor every AUC at 0.50 and quietly convert "this measurement does nothing"
into "this measurement does a little". The EDA screen on the other page has no
direction declared for it to read and so must orient by the data — that is
:func:`delong.auc_ci_auto`, and it is the only difference between the pages.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Imported rather than defined: the modelling phase publishes the same AUCs in
# its EDA table, so one implementation serves both pages — the arrangement
# :mod:`scales` already uses for the log scale. Re-exported under their old
# names because six modules in this phase (criteria, decision_curve, dichotomy,
# figures, imputation, ranking, wobble) and the tests import them from here.
from delong import (AucError, DelongResult, auc_with_ci,  # noqa: F401
                    delong_compare, fast_delong, logit_ci, oriented_score)
from measurements import (MEASUREMENTS, Measurement, STRATUM_ALL,
                          STRATUM_PRESENT, stratum_mask)
from intervals import wilson_ci

OUTCOME = "high_grade"


def separation_table(df: pd.DataFrame,
                     measurements: Sequence[Measurement] = MEASUREMENTS,
                     *, alpha: float = 0.05) -> pd.DataFrame:
    """One row per measurement per stratum, ranked best-separating first.

    The zero-inflated measurements contribute two rows each, so the ranking
    shows directly whether the edema signal comes from having any edema or from
    how much — the two rows sit next to each other with different denominators.
    """
    rows = []
    for m in measurements:
        for stratum in m.strata:
            mask = stratum_mask(df, m, stratum)
            sub = df.loc[mask]
            y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
            x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
            row = {
                "measurement": m.stratum_label(stratum),
                "col": m.col,
                "stratum": stratum,
                "direction": m.direction,
            }
            try:
                row.update(auc_with_ci(y, x, m.direction, alpha=alpha))
            except AucError as exc:
                row.update({"auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                            "n": int(mask.sum()), "n_high": np.nan,
                            "n_low": np.nan, "auc_var": np.nan,
                            "note": str(exc)})
            rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("auc", ascending=False,
                             na_position="last").reset_index(drop=True)


def presence_effect(df: pd.DataFrame, m: Measurement, *,
                    alpha: float = 0.05) -> dict[str, float]:
    """Does *having* the finding at all predict high grade?

    The first half of the zero-inflation question, and the one an AUC cannot
    answer: a yes/no predictor has no ranking to measure. So this is two
    high-grade rates with Wilson intervals, and the gap between them.
    """
    if not m.zero_inflated:
        raise AucError(f"{m.col} is not declared zero-inflated.")
    observed = stratum_mask(df, m, STRATUM_ALL)
    present = stratum_mask(df, m, STRATUM_PRESENT)
    absent = observed & ~present

    out: dict[str, float] = {"measurement": m.label}
    for name, mask in (("absent", absent), ("present", present)):
        y = pd.to_numeric(df.loc[mask, OUTCOME], errors="coerce")
        n, k = int(len(y)), int(y.sum())
        lo, hi = wilson_ci(k, n, alpha=alpha)
        out[f"n_{name}"] = n
        out[f"high_grade_{name}"] = k
        out[f"rate_{name}"] = k / n if n else np.nan
        out[f"rate_{name}_lo"] = lo
        out[f"rate_{name}_hi"] = hi
    out["rate_difference"] = out["rate_present"] - out["rate_absent"]
    return out


def presence_table(df: pd.DataFrame,
                   measurements: Sequence[Measurement] = MEASUREMENTS,
                   *, alpha: float = 0.05) -> pd.DataFrame:
    """`presence_effect` for every zero-inflated measurement."""
    rows = [presence_effect(df, m, alpha=alpha)
            for m in measurements if m.zero_inflated]
    return pd.DataFrame(rows)


def describe_separation(table: pd.DataFrame) -> str:
    """One line: the best measurement, and whether anything clears a coin flip.

    "Clears a coin flip" means the lower end of the interval sits above 0.50. A
    measurement whose interval includes 0.50 has not been shown to carry any
    information at all, and no cut-point drawn on it can be defended.
    """
    scored = table.dropna(subset=["auc"])
    if scored.empty:
        return "No measurement could be scored."
    best = scored.iloc[0]
    informative = scored[scored["auc_lo"] > 0.5]
    line = (f"Best separation: {best['measurement']}, AUC "
            f"{best['auc']:.2f} (95% CI {best['auc_lo']:.2f}-{best['auc_hi']:.2f}), "
            f"n={int(best['n'])}.")
    if informative.empty:
        return line + (" No measurement's interval clears 0.50 — none is shown "
                       "to separate the grades at all.")
    names = ", ".join(informative["measurement"])
    return line + (f" Clearing 0.50 at the lower bound: {names}"
                   f" ({len(informative)} of {len(scored)}).")
