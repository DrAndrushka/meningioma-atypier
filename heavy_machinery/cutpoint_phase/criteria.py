"""Step 6 — five ways to choose a cut-point, and whether they agree.

There is no single correct cut-point. There are only rules for picking one, and
each rule optimises something different:

``youden``        max ``Se + Sp - 1``. Treats a missed high-grade tumour and a
                  false alarm as equally costly, which is a value judgement
                  rather than a fact. The traditional default.
``closest_01``    the point nearest the perfect corner of the ROC curve. Almost
                  always lands near Youden; disagreement between the two is
                  itself informative.
``equal``         where sensitivity and specificity cross. Youden's ``Se + Sp``
                  is often nearly flat across a wide range, so its maximum
                  wanders from one resample to the next; a crossing of two
                  curves running in opposite directions has one well-defined
                  location. It usually reproduces better — but it optimises
                  nothing clinical, it simply declares the two error types
                  equally tolerable *in rate*, which is a value judgement.
``fixed_sp90``    the cut-point that catches the most high-grade tumours while
                  keeping specificity at 90% or better. **Rule-in**: few false
                  alarms, at the cost of missing cases.
``fixed_se90``    the mirror image — catch 90% of high-grade tumours, accept the
                  false alarms. **Rule-out**.
``index_union``   the point where sensitivity and specificity both sit closest
                  to the overall AUC, so neither is bought at the other's
                  expense.

The comparison is the output, not any single row. If five rules built on
different principles land within a narrow band, that is real evidence the
cut-point is a property of the data rather than of the rule. If they scatter,
the honest report is that the data do not pin a cut-point down — and that is
worth saying plainly.

**The fixed-specificity rules are the ones the manuscript leads with.** They can
be stated before seeing the data, they do not depend on how common high grade is
in the cohort, and they transfer to another hospital. Youden cannot claim any of
that: it is chosen *because* it looked best on these patients, which is why step
7 exists to measure how much of it was luck.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from accuracy import accuracy_at, confusion, flag, metrics_from_counts
from measurements import LOWER, MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import auc_with_ci

OUTCOME = "high_grade"

YOUDEN = "youden"
CLOSEST_01 = "closest_01"
EQUAL = "equal"
FIXED_SP90 = "fixed_sp90"
FIXED_SP80 = "fixed_sp80"
FIXED_SE90 = "fixed_se90"
INDEX_UNION = "index_union"

CRITERION_LABELS = {
    YOUDEN: "Youden (max Se + Sp − 1)",
    CLOSEST_01: "Closest to perfect",
    EQUAL: "Se = Sp crossing",
    FIXED_SP90: "Specificity ≥ 90%",
    FIXED_SP80: "Specificity ≥ 80%",
    FIXED_SE90: "Sensitivity ≥ 90%",
    INDEX_UNION: "Index of union",
}

# The criteria that can be stated before seeing the data. Everything else is
# chosen by looking, and carries optimism that step 7 has to measure.
PRE_SPECIFIED = (FIXED_SP90, FIXED_SP80, FIXED_SE90)

# The rules that are all trying to answer the *same* question — "where is the
# best single dividing line?" — by different arithmetic. Only these belong in an
# agreement measure.
#
# The fixed-Se and fixed-Sp rules are deliberately standing at opposite ends of
# the curve; they are answering "where do I want to stand?", not "where is the
# line?". Including them in a spread guarantees a wide band whatever the data
# say, and would hide genuine agreement among the rules that are comparable.
OPTIMUM_SEEKING = (YOUDEN, CLOSEST_01, EQUAL, INDEX_UNION)


def sweep(y: np.ndarray, x: np.ndarray, direction: str) -> pd.DataFrame:
    """Every observed value as a candidate cut-point, with its performance.

    Only observed values are candidates. A cut-point falling between two
    measurements no patient has is not a rule anyone can apply, and including
    such points would let a criterion pick a number the cohort cannot support.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)
    columns = ["cutoff", "sensitivity", "specificity", "youden_j",
               "distance_01", "tp", "fp", "fn", "tn"]
    if not x.size or y.sum() == 0 or y.sum() == y.size:
        return pd.DataFrame(columns=columns)

    # Counted by binary search rather than by testing every patient against
    # every candidate. The bootstrap in step 7 calls this thousands of times,
    # and the loop version turns a few seconds of work into several minutes.
    candidates = np.unique(x)
    pos = np.sort(x[y == 1])
    neg = np.sort(x[y == 0])
    n_pos, n_neg = pos.size, neg.size
    if direction == LOWER:                      # flagged when x <= cutoff
        tp = np.searchsorted(pos, candidates, side="right")
        fp = np.searchsorted(neg, candidates, side="right")
    else:                                       # flagged when x >= cutoff
        tp = n_pos - np.searchsorted(pos, candidates, side="left")
        fp = n_neg - np.searchsorted(neg, candidates, side="left")
    fn, tn = n_pos - tp, n_neg - fp

    se = tp / n_pos if n_pos else np.full(candidates.shape, np.nan)
    sp = tn / n_neg if n_neg else np.full(candidates.shape, np.nan)
    return pd.DataFrame({
        "cutoff": candidates.astype(float),
        "sensitivity": se,
        "specificity": sp,
        "youden_j": se + sp - 1.0,
        "distance_01": np.hypot(1 - se, 1 - sp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }, columns=columns)


def _pick(table: pd.DataFrame, score: np.ndarray, *,
          largest: bool = True) -> int | None:
    """Index of the best candidate, ties broken deterministically.

    Ties are common — many candidates share a sensitivity and specificity — so
    the tie-break must not depend on row order or floating-point noise. Highest
    Youden J first, then the smallest cut-point value, which is reproducible
    across runs and across machines.
    """
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).any():
        return None
    best = np.nanmax(score) if largest else np.nanmin(score)
    tied = np.flatnonzero(np.isclose(score, best, equal_nan=False))
    if tied.size == 1:
        return int(tied[0])
    sub = table.iloc[tied]
    order = sub.sort_values(["youden_j", "cutoff"],
                            ascending=[False, True]).index[0]
    return int(table.index.get_loc(order))


def _pick_constrained(table: pd.DataFrame, constrained: str, floor: float,
                      maximise: str) -> int | None:
    """Best ``maximise`` among candidates meeting ``constrained >= floor``.

    Phrased as a constraint rather than as "the smallest cut-point", because
    which end of the scale is *smallest* depends on the measurement's direction
    — and a rule written in terms of the axis silently inverts for ADC.
    """
    eligible = table[table[constrained] >= floor]
    if eligible.empty:
        return None
    idx = _pick(eligible, eligible[maximise].to_numpy(), largest=True)
    return None if idx is None else int(table.index.get_loc(eligible.index[idx]))


def select(table: pd.DataFrame, criterion: str, *,
           auc: float | None = None) -> float:
    """The cut-point one criterion picks, or ``nan`` if none can satisfy it."""
    if table.empty:
        return float("nan")
    pickers: dict[str, Callable[[], int | None]] = {
        YOUDEN: lambda: _pick(table, table["youden_j"].to_numpy()),
        CLOSEST_01: lambda: _pick(table, table["distance_01"].to_numpy(),
                                  largest=False),
        EQUAL: lambda: _pick(
            table,
            (table["sensitivity"] - table["specificity"]).abs().to_numpy(),
            largest=False),
        FIXED_SP90: lambda: _pick_constrained(table, "specificity", 0.90,
                                              "sensitivity"),
        FIXED_SP80: lambda: _pick_constrained(table, "specificity", 0.80,
                                              "sensitivity"),
        FIXED_SE90: lambda: _pick_constrained(table, "sensitivity", 0.90,
                                              "specificity"),
        INDEX_UNION: lambda: (
            None if auc is None or not np.isfinite(auc) else
            _pick(table,
                  -((table["sensitivity"] - auc).abs()
                    + (table["specificity"] - auc).abs()).to_numpy())),
    }
    if criterion not in pickers:
        raise KeyError(f"Unknown criterion {criterion!r}; "
                       f"expected one of {sorted(pickers)}.")
    idx = pickers[criterion]()
    return float("nan") if idx is None else float(table.iloc[idx]["cutoff"])


def usability(row: dict) -> tuple[bool, str]:
    """Is this cut-point usable at all, and if not, why not.

    A criterion can be *satisfied* and still produce something nobody should
    apply. Two failures show up in this cohort:

    *Degenerate* — the rule flags every patient or none. On a measurement where
    a third of the cohort sits at exactly zero, "sensitivity at least 90%" is
    met by a cut-point of 0, which flags everyone and has zero specificity. The
    constraint is satisfied; the rule is empty.

    *Anti-predictive* — Youden J at or below zero, or LR+ below 1, meaning a
    positive result makes high grade *less* likely. Reporting such a row as an
    operating point would be worse than reporting nothing.
    """
    se, sp = row.get("sensitivity"), row.get("specificity")
    if not (np.isfinite(se) and np.isfinite(sp)):
        return False, "no cut-point satisfies this criterion"
    if sp <= 0 or se <= 0:
        return False, ("flags every patient" if sp <= 0 else "flags no patient")
    lr_pos, j = row.get("lr_pos", np.nan), row.get("youden_j", np.nan)
    if np.isfinite(j) and j <= 0:
        return False, "Youden J at or below zero — no better than a coin flip"
    if np.isfinite(lr_pos) and lr_pos < 1.0:
        return False, "LR+ below 1 — a positive result argues against high grade"
    return True, ""


def criteria_for(df: pd.DataFrame, m: Measurement, stratum: str, *,
                 criteria: Sequence[str] = tuple(CRITERION_LABELS),
                 alpha: float = 0.05) -> pd.DataFrame:
    """Every criterion's cut-point for one measurement, with full performance."""
    sub = df.loc[stratum_mask(df, m, stratum)]
    x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
    y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
    table = sweep(y, x, m.direction)
    try:
        auc = auc_with_ci(y[np.isfinite(x) & np.isfinite(y)],
                          x[np.isfinite(x) & np.isfinite(y)],
                          m.direction)["auc"]
    except Exception:
        auc = float("nan")

    rows = []
    for criterion in criteria:
        cutoff = select(table, criterion, auc=auc)
        row = {"measurement": m.stratum_label(stratum), "col": m.col,
               "stratum": stratum, "criterion": criterion,
               "criterion_label": CRITERION_LABELS[criterion],
               "pre_specified": criterion in PRE_SPECIFIED,
               "cutoff": cutoff}
        if np.isfinite(cutoff):
            row["cutoff_rounded"] = m.round(cutoff)
            row["rule"] = m.rule_text(cutoff)
            row.update(accuracy_at(y, x, cutoff, m.direction, alpha=alpha))
        else:
            row["cutoff_rounded"] = np.nan
            row["rule"] = f"no cut-point satisfies {CRITERION_LABELS[criterion]}"
            row.update({k: np.nan for k in
                        metrics_from_counts(0, 0, 0, 0)})
        row["usable"], row["unusable_because"] = usability(row)
        rows.append(row)
    return pd.DataFrame(rows)


def criteria_table(df: pd.DataFrame, eligible: pd.DataFrame, *,
                   criteria: Sequence[str] = tuple(CRITERION_LABELS),
                   alpha: float = 0.05) -> pd.DataFrame:
    """The criteria comparison for every measurement that carried forward."""
    frames = [criteria_for(df, MEASUREMENTS_BY_COL[row["col"]], row["stratum"],
                           criteria=criteria, alpha=alpha)
              for _, row in eligible.iterrows()]
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    claims = eligible.set_index(["col", "stratum"])["claim"]
    table["claim"] = [claims.get((c, s), "")
                      for c, s in zip(table["col"], table["stratum"])]
    return table


def agreement(table: pd.DataFrame, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """How tightly the optimum-seeking rules land, per measurement.

    Only :data:`OPTIMUM_SEEKING` counts — several different pieces of arithmetic
    aimed at the same question. If they land in a narrow band, that is evidence
    the cut-point is a property of the data rather than of the rule.

    ``spread_vs_iqr`` divides that band by the middle half of the measurement's
    own distribution, which makes it comparable across measurements in different
    units. Below about 0.5 the rules are picking essentially the same place;
    above 1 they disagree by more than the spread of a typical patient, and no
    single number deserves to be called *the* cut-point.
    """
    rows = []
    for (col, stratum), grp in table.groupby(["col", "stratum"], sort=False):
        m = MEASUREMENTS_BY_COL[col]
        seeking = grp[grp["criterion"].isin(OPTIMUM_SEEKING)]
        # A degenerate cut-point would otherwise stretch the band to the full
        # range and hide real agreement among the rules that worked.
        usable = seeking[seeking["usable"]] if "usable" in seeking else seeking
        chosen = usable["cutoff"].dropna()
        if chosen.empty:
            continue
        counts = grp["n"].dropna()
        n = int(counts.iloc[0]) if len(counts) else 0
        spread = float(chosen.max() - chosen.min())
        rows.append({
            "measurement": grp["measurement"].iloc[0],
            "col": col, "stratum": stratum, "n": n,
            "claim": grp["claim"].iloc[0] if "claim" in grp else "",
            "n_criteria": int(len(chosen)),
            "n_unusable": int((~grp["usable"]).sum()) if "usable" in grp else 0,
            "cutoff_min": m.round(chosen.min()),
            "cutoff_median": m.round(chosen.median()),
            "cutoff_max": m.round(chosen.max()),
            "spread": spread,
            "iqr": _iqr(df, m, stratum),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["spread_vs_iqr"] = (table["spread"] / table["iqr"]).where(
            table["iqr"] > 0)
    return table


def _iqr(df: pd.DataFrame | None, m: Measurement, stratum: str) -> float:
    """Middle half of the measurement, in the stratum the rules were fitted on."""
    if df is None:
        return float("nan")
    values = pd.to_numeric(df.loc[stratum_mask(df, m, stratum), m.col],
                           errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.75) - values.quantile(0.25))


SCATTER_RATIO = 1.0


def describe_agreement(table: pd.DataFrame) -> str:
    """One line: where the optimum-seeking rules concentrate, and where they scatter."""
    if table.empty:
        return "No criterion could be applied."
    tight, loose = [], []
    for _, r in table.iterrows():
        ratio = r.get("spread_vs_iqr", np.nan)
        band = f"{r['measurement']} {r['cutoff_min']:g}-{r['cutoff_max']:g}"
        (loose if (np.isfinite(ratio) and ratio > SCATTER_RATIO)
         else tight).append(band)
    parts = []
    if tight:
        parts.append(f"Rules land in a narrow band — {'; '.join(tight)}.")
    if loose:
        parts.append("Rules disagree by more than the spread of a typical "
                     f"patient, so no single number is 'the' cut-point — "
                     f"{'; '.join(loose)}.")
    return " ".join(parts)
