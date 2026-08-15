"""The rule menu as boolean matrices — the same J, without the pandas.

:func:`combinations.full_rule_menu` is the right tool for a table a human will
read: it carries Wilson intervals, a χ², an odds ratio and a label for every
rule. The selection bootstrap reads none of that. It rebuilds the menu on each
of hundreds of resamples and looks at one column, ``youden_J``, which costs
about 830 µs per rule to arrive at and nothing at all to use.

This module computes that one column. A cohort becomes two n×k boolean
matrices — ``present`` ("this flag is True") and ``observed`` — and a resample
becomes an integer index into their rows.

Every rule is a row-wise function of those flags, so the whole menu is applied
to every patient **once** (:func:`expand_rule_menu`) and a resample is then four
column sums over the selected rows. Scoring a draw costs no combination
enumeration at all. The counts are the same integers the per-rule version
produced, under the same three-valued logic and in the same order as the pandas
version, which is what makes the two agree to the last bit rather than to three
decimals.

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
    # The menu already applied to every patient: column r of ``rule_present``
    # is "rule r fires for this patient", ``rule_absent`` is "rule r is settled
    # False for them". Both are (n, n_rules) and depend only on the cut-points,
    # so the bootstrap builds them once and then only picks rows.
    rule_present: np.ndarray = None  # type: ignore[assignment]
    rule_absent: np.ndarray = None   # type: ignore[assignment]


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
    rule_present, rule_absent = expand_rule_menu(
        present, observed, len(cutpoints), max_size=max_size,
    )
    return RuleMatrix(present, observed, positive, known,
                      labels, kinds, len(cutpoints), len(df),
                      rule_present, rule_absent)


def expand_rule_menu(
    present: np.ndarray,
    observed: np.ndarray,
    k: int,
    *,
    max_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply every rule in the menu to every patient, in ``rule_labels`` order.

    Returns ``(rule_present, rule_absent)``, both (n, n_rules) bool. A rule is
    three-valued like the flags it combines: neither array set means the rule
    is missing for that patient, and both are needed to count correctly.

    Every rule here is a row-wise function of the flags, which is what lets the
    bootstrap evaluate a resample by selecting rows of these arrays instead of
    re-deriving hundreds of combinations per draw.
    """
    absent = observed & ~present
    cols_present: list[np.ndarray] = []
    cols_absent: list[np.ndarray] = []

    for j in range(k):
        cols_present.append(present[:, j])
        cols_absent.append(absent[:, j])

    for size in range(2, int(max_size) + 1):
        for combo in itertools.combinations(range(k), size):
            members = list(combo)
            # Kleene: AND is True only if all are True, False if any is False.
            cols_present.append(present[:, members].all(axis=1))
            cols_absent.append(absent[:, members].any(axis=1))
            # OR is the mirror: True if any is True, False only if all are.
            cols_present.append(present[:, members].any(axis=1))
            cols_absent.append(absent[:, members].all(axis=1))

    # Count rules score complete cases only: "2 of 4" means something else when
    # one of the four was never measured.
    valid = observed.all(axis=1)
    counts = present.sum(axis=1)
    for cut in range(1, k + 1):
        hit = valid & (counts >= cut)
        cols_present.append(hit)
        cols_absent.append(valid & ~hit)

    empty = np.empty((present.shape[0], 0), dtype=bool)
    stack = lambda cols: np.column_stack(cols) if cols else empty  # noqa: E731
    return stack(cols_present), stack(cols_absent)


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
    rule_present, rule_absent = matrix.rule_present, matrix.rule_absent
    if rule_present is None:  # RuleMatrix built before the menu was expanded
        rule_present, rule_absent = expand_rule_menu(
            matrix.present, matrix.observed, matrix.k, max_size=max_size,
        )
    positive, known = matrix.positive, matrix.known
    if rows is not None:
        rule_present, rule_absent = rule_present[rows], rule_absent[rows]
        positive, known = positive[rows], known[rows]

    pos = positive                        # known and high grade
    neg = known & ~positive               # known and not high grade

    # One pass per cell of the 2x2, for all rules at once. These are the same
    # integer counts the per-rule count_nonzero calls produced, so the J values
    # are bit-for-bit what the scalar version returned.
    tp = rule_present[pos].sum(axis=0)
    fp = rule_present[neg].sum(axis=0)
    fn = rule_absent[pos].sum(axis=0)
    tn = rule_absent[neg].sum(axis=0)

    sens = _rate(tp, tp + fn)
    spec = _rate(tn, tn + fp)
    return sens + spec - 1.0


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, NaN where the rate has no denominator."""
    out = np.full(numerator.shape, np.nan, dtype=float)
    nonzero = denominator != 0
    np.divide(numerator, denominator, out=out, where=nonzero)
    return out
