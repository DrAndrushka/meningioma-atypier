"""The rule menu as boolean matrices — the same J, without the pandas.

:func:`combinations.full_rule_menu` is the right tool for a table a human will
read: it carries Wilson intervals, a χ², an odds ratio and a label for every
rule. The selection bootstrap reads none of that. It rebuilds the menu on each
of hundreds of resamples and looks at one column, ``youden_J``, which costs
about 830 µs per rule to arrive at and nothing at all to use.

This module computes that one column. A cohort becomes two n×k boolean
matrices — ``present`` ("this flag is True") and ``observed`` — and a resample
becomes an integer index into their rows. Every rule is then a couple of
boolean operations and four ``count_nonzero`` calls, in the same order and
under the same three-valued logic as the pandas version, which is what makes
the two agree to the last bit rather than to three decimals.

The labels are static — they depend on the cut-points, never on the data — so
the bootstrap matches winners by position and builds no strings at all.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

import combinations as cb


class RuleMatrix(NamedTuple):
    """A cohort's flags, ready to be resampled by row index.

    ``present`` and ``observed`` are the two halves of a three-valued flag: the
    value is True where ``present``, False where ``observed & ~present``, and
    missing where ``~observed``. ``known`` marks patients with a recorded
    outcome, ``positive`` the high-grade ones among them. Every row of the
    original frame is kept — including patients with no outcome — because the
    bootstrap resamples the frame it was handed, and dropping rows here would
    quietly shrink the resample.
    """

    present: np.ndarray      # (n, k) bool
    observed: np.ndarray     # (n, k) bool
    positive: np.ndarray     # (n,)   bool — outcome known and positive
    known: np.ndarray        # (n,)   bool — outcome recorded
    labels: list[str]
    kinds: list[str]
    k: int
    n: int


def rule_labels(cutpoints: Sequence, *, max_size: int = 2) -> tuple[list[str], list[str]]:
    """``(labels, kinds)`` in exactly ``full_rule_menu``'s row order.

    The AND row precedes the OR row for each combination, because
    ``pair_rule_table`` iterates ``combinations.LOGICS``. If that constant is
    ever reordered, :func:`youden_j` must be reordered with it.
    """
    k = len(cutpoints)
    labels = [cp.label for cp in cutpoints]
    kinds = ["single"] * k
    for size in range(2, int(max_size) + 1):
        for combo in itertools.combinations(range(k), size):
            for logic in cb.LOGICS:
                labels.append(f" {logic} ".join(cutpoints[i].label for i in combo))
                kinds.append(logic.lower())
    for cut in range(1, k + 1):
        labels.append(f"≥ {cut} of {k} criteria")
        kinds.append("count")
    return labels, kinds


def rule_matrix(
    df: pd.DataFrame,
    cutpoints: Sequence,
    target: str,
    *,
    max_size: int = 2,
) -> RuleMatrix:
    """Flags and outcome as boolean arrays, plus the static menu labels."""
    flags = cb.flag_frame(df, cutpoints)
    observed = flags.notna().to_numpy(dtype=bool)
    present = flags.fillna(False).to_numpy(dtype=bool) & observed

    y = df[target].astype("boolean")
    known = y.notna().to_numpy(dtype=bool)
    positive = y.fillna(False).to_numpy(dtype=bool) & known

    labels, kinds = rule_labels(cutpoints, max_size=max_size)
    return RuleMatrix(present, observed, positive, known,
                      labels, kinds, len(cutpoints), len(df))


def _youden(true_: np.ndarray, false_: np.ndarray,
            pos: np.ndarray, neg: np.ndarray) -> float:
    """``sensitivity + specificity - 1``, NaN where a rate has no denominator."""
    tp = np.count_nonzero(true_ & pos)
    fp = np.count_nonzero(true_ & neg)
    fn = np.count_nonzero(false_ & pos)
    tn = np.count_nonzero(false_ & neg)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens + spec - 1.0


def youden_j(
    matrix: RuleMatrix,
    rows: np.ndarray | None = None,
    *,
    max_size: int = 2,
) -> np.ndarray:
    """Youden J for every rule in the menu, in ``full_rule_menu`` order.

    ``rows`` is an integer index — a bootstrap resample — or ``None`` for the
    cohort as it stands. Patients with an unknown outcome ride along in the
    matrix but are excluded from every count here, exactly as the pandas
    version drops them.
    """
    present, observed = matrix.present, matrix.observed
    positive, known = matrix.positive, matrix.known
    if rows is not None:
        present, observed = present[rows], observed[rows]
        positive, known = positive[rows], known[rows]

    absent = observed & ~present          # observed and False
    pos = positive                        # known and high grade
    neg = known & ~positive               # known and not high grade

    k = matrix.k
    out = np.empty(len(matrix.labels), dtype=float)
    i = 0

    for j in range(k):
        out[i] = _youden(present[:, j], absent[:, j], pos, neg)
        i += 1

    for size in range(2, int(max_size) + 1):
        for combo in itertools.combinations(range(k), size):
            members = list(combo)
            # Kleene: AND is True only if all are True, False if any is False.
            out[i] = _youden(present[:, members].all(axis=1),
                             absent[:, members].any(axis=1), pos, neg)
            # OR is the mirror: True if any is True, False only if all are.
            out[i + 1] = _youden(present[:, members].any(axis=1),
                                 absent[:, members].all(axis=1), pos, neg)
            i += 2

    # Count rules score complete cases only: "2 of 4" means something else when
    # one of the four was never measured.
    valid = observed.all(axis=1)
    counts = present.sum(axis=1)
    for cut in range(1, k + 1):
        hit = valid & (counts >= cut)
        out[i] = _youden(hit, valid & ~hit, pos, neg)
        i += 1

    return out
