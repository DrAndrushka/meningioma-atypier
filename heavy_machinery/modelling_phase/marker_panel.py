"""Which MRI markers, and do they combine? — the two study aims, in one place.

The report's EDA section already scores every marker on its own, and the
threshold phase already knows how to compare a combined rule against a single
one. What is missing is a place where those two answers sit side by side on one
patient set, in the report that actually gets read.

Nothing here re-implements an estimator that already exists. The combination
machinery is :mod:`combinations` from the threshold phase, reached through a
nine-line adapter; the model scoring is :mod:`model_calculator`. The only new
statistic is the positive likelihood ratio, which is what turns "most specific"
into a question with a defensible answer — a sign that is never seen is
perfectly specific and perfectly useless.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

import plot_style as ps

_Z95 = 1.959963984540054

BINARY_KINDS = ("binary", "derived_binary")


def likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int) -> dict:
    """How much more likely this sign is in a high-grade tumor than a benign one.

    ``LR+ = sensitivity / (1 - specificity)``. An LR+ of 10 means seeing the
    sign makes high grade ten times more likely; an LR+ of 1 means it says
    nothing. The interval is Katz's, computed on the log scale because a ratio
    bounded below by zero and unbounded above is not symmetric.

    A zero in the flagged column makes the ratio infinite and its interval
    undefined. Half a patient is added to every cell in that case
    (Haldane-Anscombe) and ``continuity_corrected`` says so, because the
    resulting interval is honestly enormous and the reader should see why.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
           "chance_overlap": False, "continuity_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = tp == 0 or fp == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    sens = a / (a + c)
    fpr = b / (b + d)
    if fpr <= 0 or sens <= 0:
        return nan

    lr = sens / fpr
    se = math.sqrt(1.0 / a - 1.0 / (a + c) + 1.0 / b - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_pos": float(lr),
        "lr_pos_lo": float(lo),
        "lr_pos_hi": float(hi),
        "chance_overlap": bool(lo <= 1.0 <= hi),
        "continuity_corrected": bool(corrected),
    }


class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a ``CutPoint`` so ``combinations`` accepts it.

    :mod:`combinations` touches exactly four members of a cut-point — ``col``,
    ``label``, ``short_label`` and ``flag`` — so supplying those is enough to
    run its whole rule machinery on markers that were never continuous. This is
    why ``combinations.py`` needs no change: the threshold phase's estimators
    and this section's are the same code, and cannot drift apart.
    """

    col: str
    label: str

    @property
    def short_label(self) -> str:
        """No cut-point to name, so the short form is the label itself."""
        return self.label

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].astype("boolean")


def markers_from_diagnostic_accuracy(
    table: pd.DataFrame,
    *,
    target: str,
    exclude: Collection[str] = (),
) -> list[BinaryMarker]:
    """The marker panel, read from the EDA table rather than hard-coded.

    Whatever is activated or dropped in the cleaning notebook's ``DERIVATIONS``
    flows through here without an edit, which is the point: the panel cannot
    silently fall out of step with the cohort it describes.

    ``exclude`` is the caller's, and belongs in the notebook. The accuracy
    table carries non-imaging predictors too — ``sex_male`` is
    ``derived_binary`` and would otherwise walk into a section about MRI signs.
    """
    if table is None or table.empty:
        return []
    drop = set(exclude) | {target}
    rows = table[
        (table["target"].astype(str) == str(target))
        & (table["kind"].astype(str).isin(BINARY_KINDS))
        & (~table["predictor"].astype(str).isin(drop))
    ]
    return [
        BinaryMarker(str(r["predictor"]), ps.prettify_label(str(r["predictor"])))
        for _, r in rows.iterrows()
    ]
