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
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scales import is_log_scaled


def discrimination(auc: float) -> float:
    """How well a variable separates, regardless of direction."""
    return float(max(auc, 1.0 - auc))


def _column_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    x = df[col].astype(float).to_numpy()
    return np.log1p(np.clip(x, 0.0, None)) if is_log_scaled(col) else x


def rank_candidates(
    df: pd.DataFrame,
    y: Sequence[int],
    candidates: Sequence[str],
) -> list[tuple[str, float]]:
    """``(column, raw_auc)`` sorted by discrimination, best first."""
    y_arr = np.asarray(y, dtype=int)
    scored: list[tuple[str, float]] = []
    for col in candidates:
        if col not in df.columns:
            continue
        x = _column_vector(df, col)
        if np.unique(x).size < 2 or np.unique(y_arr).size < 2:
            continue
        scored.append((col, float(roc_auc_score(y_arr, x))))
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
    Candidates after the k-th pick are not examined and do not appear.
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
    return picked, audit
