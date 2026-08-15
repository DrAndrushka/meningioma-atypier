"""Step 10b — do the measurements move together?

Step 10 asked whether two measurements carry the same *information about grade*.
This asks something more basic: do they move together at all, grade aside?

It matters for step 11. When two predictors rise and fall together, a model
cannot tell which of them is doing the work, so it splits the effect between
them — and **both** come out looking weak. That is a wrong conclusion, not a
null one, and it is the single most common way a multivariable table misleads.
A pair flagged here explains a pair of disappointing rows there.

**Spearman, not Pearson.** Spearman correlates the *ranks* — patient 3 is the
seventh largest tumour and the fifth largest diameter — rather than the values.
Three of these measurements are heavily right-skewed and two have a third of the
cohort piled at zero, and on such data Pearson mostly reports the influence of a
handful of extreme patients. Ranks are unaffected by that.

**Pairwise denominators.** Every pair is computed on the patients who have both
measurements, and that count is reported. Different measurements are missing in
different patients, so a matrix with one *n* in the caption would be wrong in
most of its cells.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from accuracy import flag
from measurements import MEASUREMENTS, Measurement

# Above this, two measurements are close enough that a model fitted on both will
# struggle to separate them. Not a law — a reporting threshold, chosen because
# it is where variance inflation starts to bite at this sample size.
TOGETHER = 0.70

# Two yes/no rules agreeing on more than this share of patients are, in
# practice, selecting the same patients whatever their definitions say.
FLAG_AGREEMENT = 0.85


def spearman_matrix(df: pd.DataFrame,
                    measurements: Sequence[Measurement] = MEASUREMENTS
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank correlation between every pair, and the patients each rests on."""
    labels = [m.label for m in measurements]
    rho = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for i, a in enumerate(measurements):
        x_a = pd.to_numeric(df[a.col], errors="coerce")
        counts.iloc[i, i] = int(x_a.notna().sum())
        for j, b in enumerate(measurements):
            if j <= i:
                continue
            x_b = pd.to_numeric(df[b.col], errors="coerce")
            both = x_a.notna() & x_b.notna()
            n = int(both.sum())
            counts.iloc[i, j] = counts.iloc[j, i] = n
            if n < 3:
                value = np.nan
            else:
                value = float(spearmanr(x_a[both], x_b[both]).statistic)
            rho.iloc[i, j] = rho.iloc[j, i] = value
    return rho, counts


def correlated_pairs(rho: pd.DataFrame, counts: pd.DataFrame, *,
                     threshold: float = TOGETHER) -> pd.DataFrame:
    """Every pair as one row, strongest first, with the flag applied.

    Returns all pairs, not only the correlated ones. A reader needs to see that
    the others were checked and came back low, otherwise the table reads as a
    list of problems found rather than a survey completed.
    """
    labels = list(rho.index)
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            value = float(rho.loc[a, b])
            rows.append({
                "a": a, "b": b, "n_both": int(counts.loc[a, b]),
                "spearman": value,
                "abs_spearman": abs(value) if np.isfinite(value) else np.nan,
                "moves_together": bool(np.isfinite(value)
                                       and abs(value) >= threshold),
            })
    table = pd.DataFrame(rows)
    return table.sort_values("abs_spearman", ascending=False,
                             na_position="last").reset_index(drop=True)


def flag_agreement(df: pd.DataFrame, cutpoints: dict[str, float],
                   measurements: Sequence[Measurement] = MEASUREMENTS
                   ) -> pd.DataFrame:
    """Do two yes/no rules select the same patients?

    Reported alongside the rank correlation because they can disagree. Two
    measurements can correlate strongly across their whole range while their
    cut-points sit in different places and flag different people — and it is the
    flags, not the raw values, that a count-based rule would combine.

    ``phi`` is the correlation between the two yes/no columns; ``agreement`` is
    the plainer number — the share of patients both rules treat the same way.
    """
    usable = [m for m in measurements if m.col in cutpoints]
    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            x_a = pd.to_numeric(df[a.col], errors="coerce")
            x_b = pd.to_numeric(df[b.col], errors="coerce")
            both = x_a.notna() & x_b.notna()
            n = int(both.sum())
            if n < 3:
                continue
            f_a = flag(x_a[both].to_numpy(), cutpoints[a.col], a.direction)
            f_b = flag(x_b[both].to_numpy(), cutpoints[b.col], b.direction)
            agree = float(np.mean(f_a == f_b))
            if f_a.std() == 0 or f_b.std() == 0:
                phi = np.nan
            else:
                phi = float(np.corrcoef(f_a.astype(float),
                                        f_b.astype(float))[0, 1])
            rows.append({
                "a": a.rule_text(cutpoints[a.col]),
                "b": b.rule_text(cutpoints[b.col]),
                "col_a": a.col, "col_b": b.col, "n_both": n,
                "agreement": agree, "phi": phi,
                "both_flagged": int(np.sum(f_a & f_b)),
                "selects_the_same_patients": bool(agree >= FLAG_AGREEMENT),
            })
    table = pd.DataFrame(rows)
    return (table.sort_values("agreement", ascending=False).reset_index(drop=True)
            if not table.empty else table)


def describe_collinearity(pairs: pd.DataFrame,
                          flags: pd.DataFrame | None = None) -> str:
    """One line: which measurements move together, and what that costs step 11."""
    if pairs.empty:
        return "No pair could be correlated."
    together = pairs[pairs["moves_together"]]
    if together.empty:
        strongest = pairs.iloc[0]
        line = (f"No pair moves together beyond {TOGETHER:.2f}; the strongest is "
                f"{strongest['a']} and {strongest['b']} at "
                f"{strongest['spearman']:.2f}. A model can separate them.")
    else:
        named = "; ".join(f"{r['a']} and {r['b']} ({r['spearman']:.2f}, "
                          f"n={int(r['n_both'])})" for _, r in together.iterrows())
        line = (f"Moves together — {named}. A model fitted on both will split "
                "their effect and make each look weaker than it is.")
    if flags is not None and not flags.empty:
        same = flags[flags["selects_the_same_patients"]]
        if not same.empty:
            named = "; ".join(f"{r['a']} and {r['b']} "
                              f"({r['agreement']:.0%})" for _, r in same.iterrows())
            line += f" Cut-points selecting essentially the same patients — {named}."
    return line
