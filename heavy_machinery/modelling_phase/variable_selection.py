"""Pick the k best predictors by discrimination, with two guards.

``top_1_variable`` and ``top_6_variables`` are chosen from the same cohort they
are then fitted on. That is defensible only if the rule is written down, applied
mechanically, and auditable — which is what this module is.

Ranking is by *discrimination*, ``max(auc, 1 - auc)``, not raw AUC. A protective
predictor scores below 0.5 by construction: ADC is 0.370 in this cohort, which
is 0.630 the other way round and the second-strongest variable available.
Ranking on raw AUC would silently discard every protective finding.

Two guards, in order:

1. **Parent/child.** A derived cut-point is skipped when its continuous parent is
   also a candidate. The parent carries more information, and the child stacks
   the cut-point's own optimism on top of the model's.
2. **Collinearity.** A candidate correlated above ``rho_max`` with something
   already picked is skipped, and the next candidate that clears is taken. Six
   variables chosen on discrimination alone are four tumour-size measurements in
   different costumes.

Every skip is recorded with its reason. A selection nobody can see is a
selection nobody can check.

A candidate that cannot be coerced to a single numeric vector — a categorical
column such as ``tumor_location`` or ``sex`` — is set aside before ranking even
starts, rather than crashing the whole selection. This is a real modelling
limit, not just a type mismatch: a multi-level nominal variable has no single
AUC to rank on, and giving it one would mean one-hot encoding it first, which is
out of scope here. It still gets its own audit row, because a reader who knows
the column exists should be able to see why it never competed, not just that it
is absent from the results.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc as _sk_auc, roc_curve as _sk_roc_curve


def _roc_auc(y_true, y_score) -> float:
    """``sklearn.metrics.roc_auc_score`` without its input validation.

    roc_auc_score spends most of its time in ``validate_params`` and
    ``type_of_target``, which run ``np.unique`` over both arrays on every
    call. Inside a bootstrap that validation costs more than the metric. These
    are the same two functions roc_auc_score itself calls for the binary case,
    in the same order, so the float is bit-for-bit the one it returned before.
    """
    fpr, tpr, _ = _sk_roc_curve(y_true, y_score)
    return float(_sk_auc(fpr, tpr))


class _NotNumeric(Exception):
    """Internal signal: a candidate column could not be coerced to float.

    Caught wherever it matters — ``rank_candidates`` treats it as a reason to
    leave the column out of the ranking, and ``select_variables`` treats it as
    a reason to write an audit row explaining why.
    """


def discrimination(auc: float) -> float:
    """How well a variable separates, regardless of direction."""
    return float(max(auc, 1.0 - auc))


def _column_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    # No log1p here, even for columns scales.is_log_scaled() flags. This
    # function's only two consumers — roc_auc_score in rank_candidates, and
    # the Spearman correlation in select_variables' collinearity guard — are
    # both rank-based, and log1p is a strictly monotone transform, so on
    # genuinely raw input it can never change either one; applying it was
    # always a no-op there. But a caller may hand this a column that is
    # ALREADY on the model scale (inferential._build_design log1p's and then
    # z-scores these columns before fitting), and re-applying log1p to an
    # already mean-centred column means clipping every below-mean patient's
    # negative z-score to a tied 0.0 — silently corrupting the ranking for
    # roughly half the cohort. Return the column exactly as given.
    try:
        x = df[col].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise _NotNumeric(col) from exc
    return x


def _fewer_than_two_distinct(x: np.ndarray) -> bool:
    """``np.unique(x).size < 2`` without building the unique array.

    np.unique sorts the whole column; inside a bootstrap this ran once per
    candidate per resample and cost more than the AUC it guards. NaN is
    treated the way np.unique treats it — every NaN collapses to one value.
    """
    if x.size < 2:
        return True
    first = x[0]
    if first != first:                      # NaN
        return bool(np.isnan(x).all())
    return bool((x == first).all())


def rank_candidates(
    df: pd.DataFrame,
    y: Sequence[int],
    candidates: Sequence[str],
) -> list[tuple[str, float]]:
    """``(column, raw_auc)`` sorted by discrimination, best first."""
    y_arr = np.asarray(y, dtype=int)
    # Loop-invariant: the outcome is the same for every candidate, so the
    # single-class check belongs outside the loop, not once per column.
    if np.unique(y_arr).size < 2:
        return []
    scored: list[tuple[str, float]] = []
    for col in candidates:
        if col not in df.columns:
            continue
        try:
            x = _column_vector(df, col)
        except _NotNumeric:
            # Categorical/text candidates get their own audit row from
            # select_variables; here they are simply left out of the ranking,
            # the same way an absent or constant column already is.
            continue
        if _fewer_than_two_distinct(x):
            continue
        scored.append((col, float(_roc_auc(y_arr, x))))
    return sorted(scored, key=lambda t: -discrimination(t[1]))


def select_variables(
    df: pd.DataFrame,
    y: Sequence[int],
    candidates: Sequence[str],
    *,
    k: int,
    rho_max: float = 0.8,
    cutpoint_parent: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick ``k`` variables by discrimination, applying both guards.

    Returns the picked columns and one audit row per candidate considered.
    Candidates after the k-th pick are not examined and do not appear. A
    candidate that cannot be coerced to numeric is a separate case: it never
    entered the ranking at all, but it is still audited, unconditionally,
    because it is a real modelling limit rather than a competitor that simply
    lost.
    """
    cutpoint_parent = cutpoint_parent or {}
    ranked = rank_candidates(df, y, candidates)
    candidate_set = {c for c, _ in ranked}
    vectors = {c: _column_vector(df, c) for c, _ in ranked}

    picked: list[str] = []
    audit: list[dict[str, Any]] = []
    for col, auc in ranked:
        if len(picked) == k:
            break
        parent = cutpoint_parent.get(col)
        if parent is not None and parent in candidate_set:
            audit.append({"variable": col, "auc": auc,
                          "discrimination": discrimination(auc),
                          "kept": False, "reason": f"cut-point of {parent}"})
            continue
        clash = ""
        for p in picked:
            rho = abs(float(pd.Series(vectors[col]).corr(
                pd.Series(vectors[p]), method="spearman")))
            if rho > rho_max:
                clash = f"rho={rho:.2f} with {p}"
                break
        if clash:
            audit.append({"variable": col, "auc": auc,
                          "discrimination": discrimination(auc),
                          "kept": False, "reason": clash})
            continue
        picked.append(col)
        audit.append({"variable": col, "auc": auc,
                      "discrimination": discrimination(auc),
                      "kept": True, "reason": ""})

    for col in candidates:
        if col not in df.columns:
            continue
        try:
            _column_vector(df, col)
        except _NotNumeric:
            audit.append({
                "variable": col, "auc": float("nan"),
                "discrimination": float("nan"), "kept": False,
                "reason": (
                    "not numeric — selection ranks one AUC per variable; a "
                    "multi-level categorical has no single AUC without "
                    "one-hot encoding, which is out of scope here"
                ),
            })
    return picked, audit


def bootstrap_reselect(
    frame: pd.DataFrame,
    y_boot: Sequence[int],
    *,
    candidates: Sequence[str],
    k: int,
    rho_max: float,
    cutpoint_parent: Mapping[str, str] | None,
) -> list[str]:
    """``select=`` hook for ``model_validation.bootstrap_internal_validation``.

    Re-runs :func:`select_variables` on one bootstrap resample and returns
    just the picked columns. Module-level and built with
    ``functools.partial`` over plain data (a list of names, two numbers, and
    a dict) rather than as a closure, because the data-selected model's
    validation is sometimes shipped to a ``joblib`` worker process (see
    ``model_calculator.enrich_streamlit_artifacts``), and a closure cannot be
    pickled across a process boundary — only this function's *arguments*
    need to be.

    Both callers that re-run selection inside a bootstrap — the
    combined-vs-single comparison table and the Streamlit calculator export
    — call this same function with the same arguments, so the same
    resample produces the same pick in both places and their
    optimism-corrected AUCs cannot drift apart from each other.
    """
    sub, _ = select_variables(
        frame, y_boot, [c for c in candidates if c in frame.columns],
        k=k, rho_max=rho_max, cutpoint_parent=dict(cutpoint_parent or {}))
    return sub


def assert_reference(
    picked: Sequence[str],
    audit: Sequence[dict[str, Any]] | None = None,
    *,
    discrimination_tol: float = 0.01,
) -> None:
    """Raise unless the declared reference variable is the first kept pick.

    Checks the list *after* both guards, not the raw ranking. A derived
    cut-point can top the raw ranking and still be dropped by guard 1, and the
    reference has to be a variable the pipeline would actually fit.

    ``audit`` — the second return value of :func:`select_variables`, from the
    SAME call that produced ``picked`` — is optional, but when given this
    also raises if the reference's discrimination in THIS run has drifted
    from the declared ``analysis.REFERENCE_VARIABLE_DISCRIMINATION`` by more
    than ``discrimination_tol``. Without this, a real change in the
    reference's underlying strength (a data fix, a schema change, a bug in a
    derived column) would pass silently as long as the reference was still
    ranked first — the declared discrimination existed only as a comment
    justifying the choice, never actually checked against a re-run.
    """
    from heavy_machinery.config import load

    declared = load("analysis").REFERENCE_VARIABLE
    if not picked:
        raise ValueError(
            "No variable survived selection, so there is no reference variable."
        )
    if picked[0] != declared:
        raise ValueError(
            f"The declared reference variable {declared!r} is no longer the top "
            f"pick: {picked[0]!r} now leads. Every delta-AUC in the report is "
            f"measured against the reference, so update "
            f"analysis.REFERENCE_VARIABLE deliberately rather than letting the "
            f"denominator move on its own."
        )
    if audit is not None:
        declared_disc = load("analysis").REFERENCE_VARIABLE_DISCRIMINATION
        row = next((r for r in audit if r.get("variable") == declared), None)
        actual = row.get("discrimination") if row is not None else None
        if actual is not None and abs(float(actual) - declared_disc) > discrimination_tol:
            raise ValueError(
                f"analysis.REFERENCE_VARIABLE_DISCRIMINATION is declared as "
                f"{declared_disc}, but {declared!r} scored "
                f"{float(actual):.3f} discrimination in this run — a drift "
                f"of {abs(float(actual) - declared_disc):.3f}, more than the "
                f"{discrimination_tol} tolerance. Update the declared value "
                "deliberately if this is a real, intended change, rather "
                "than letting a data or derivation change move the "
                "reference's justification silently."
            )
