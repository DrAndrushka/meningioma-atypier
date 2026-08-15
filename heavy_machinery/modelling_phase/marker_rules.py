"""Binary markers and the rules that choose their cut-points.

This is the machinery the marker panel needs and nothing else: a
:class:`Metric` (one continuous predictor plus the direction it points in), the
ROC table of every cut-point the data can distinguish, and five named rules for
picking one row out of that table.

It is deliberately small. Deciding whether a threshold *exists* — whether risk
bends at a value, whether the bend survives a change of scale, how far the
cut-point moves under resampling — is the cut-point phase's job, and lives in
``cutpoint_phase/``. What remains here is the opposite question, and the only
one the marker panel asks: *given* a cut-point, how does the resulting yes/no
marker behave, alone and in combination with others?

Everything downstream reads ``direction``, so a metric declared the wrong way
round shows up as an AUC below 0.5 rather than as silently inverted cut-points.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from scales import disagreements

HIGHER = "higher"
LOWER = "lower"
_DIRECTIONS = (HIGHER, LOWER)


class Metric(NamedTuple):
    """One continuous predictor and how it points at the outcome.

    ``direction="lower"`` means low values are suspicious for high grade (ADC);
    ``"higher"`` means high values are (the volumes).

    ``log_x`` marks a right-skewed metric — volumes here span three orders of
    magnitude. It switches a cut-point axis to a log scale, and it is checked
    against ``scales.LOG1P_COLUMNS`` so that one notebook cannot quietly analyse
    a column on a different scale from the rest of the pipeline.
    """

    col: str
    label: str
    unit: str
    direction: str
    log_x: bool = False
    unit_plain: str | None = None

    @property
    def op(self) -> str:
        """Comparison that flags a case as suspicious at a given cut-point."""
        return "≤" if self.direction == LOWER else "≥"

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label

    @property
    def prose_unit(self) -> str:
        """Unit for running text — may legitimately be empty.

        ``unit`` is written for an *axis*, where matplotlib mathtext is fine
        (the SciencePlots font has no superscript-minus glyph, so ADC needs
        it). Dropped into a sentence or a CSV cell the same string renders as
        raw ``$\\times 10^{-3}$``, and a descriptive pseudo-unit like
        "edema ÷ tumor" reads as nonsense mid-sentence.

        So ``unit_plain`` has three states: ``None`` (the default) derives the
        prose unit from ``unit``, blanking it if it is mathtext; ``""`` says
        this quantity has no unit worth naming in prose; any other string is
        used verbatim.
        """
        if self.unit_plain is not None:
            return self.unit_plain
        return "" if "$" in self.unit else self.unit

    def rule_text(self, cutoff: float) -> str:
        """Human-readable rule, e.g. ``ADC (mean) ≤ 0.72``."""
        return f"{self.label} {self.op} {cutoff:.3g}"

    def flag(self, values: pd.Series, cutoff: float) -> pd.Series:
        """Boolean 'test positive' series at ``cutoff``, preserving missingness."""
        v = pd.Series(values).astype("Float64")
        flagged = (v <= cutoff) if self.direction == LOWER else (v >= cutoff)
        return flagged.astype("boolean")


def validate_metrics(metrics: Sequence[Metric], df: pd.DataFrame) -> list[Metric]:
    """Reject bad directions early and drop metrics the frame does not carry.

    A typo in ``direction`` is the one error in this module that produces
    plausible-looking numbers, so it is raised rather than warned about.
    """
    bad = [m.col for m in metrics if m.direction not in _DIRECTIONS]
    if bad:
        raise ValueError(
            f"direction must be one of {_DIRECTIONS} — bad for: {', '.join(bad)}"
        )
    # ``log_x`` is declared per metric, but which measurements are logged is a
    # pipeline-wide fact: a metric analysed on the log scale here and the raw
    # scale elsewhere yields two different odds ratios for the same patients,
    # with nothing on either page to say which is which.
    mismatched = disagreements({m.col: m.log_x for m in metrics})
    if mismatched:
        raise ValueError(
            f"log_x disagrees with scales.LOG1P_COLUMNS for: "
            f"{', '.join(mismatched)} — change scales.LOG1P_COLUMNS, which "
            f"every phase reads, rather than one notebook's declaration."
        )
    return [m for m in metrics if m.col in df.columns]


# --------------------------------------------------------------------------
# Complete-case extraction
# --------------------------------------------------------------------------
def metric_arrays(
    frame: pd.DataFrame,
    metric: Metric,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete-case ``(values, outcome)`` arrays for one metric.

    Each metric is analysed on its own complete cases, so ``n`` differs between
    metrics. That is deliberate — the alternative (one shared complete-case set)
    throws away patients who are missing an unrelated measurement.
    """
    pair = pd.concat(
        [frame[metric.col].astype("Float64"), frame[target].astype("boolean")],
        axis=1,
    ).dropna()
    x = pair[metric.col].astype(float).to_numpy()
    y = pair[target].astype(bool).to_numpy().astype(int)
    return x, y


def oriented_score(x: np.ndarray, direction: str) -> np.ndarray:
    """Flip so larger always means 'more suspicious' — the ROC needs one convention."""
    return x if direction == HIGHER else -x


# --------------------------------------------------------------------------
# ROC table and selection rules
# --------------------------------------------------------------------------
def roc_table(x: np.ndarray, y: np.ndarray, direction: str) -> pd.DataFrame:
    """Every cut-point the data can distinguish, with its operating characteristics.

    ``sklearn.roc_curve`` works on the oriented score; cut-offs are mapped back
    onto the metric's own scale so a row reads as "ADC ≤ 0.72".

    Cut-points that flag everyone or no one are dropped: they are not tests, and
    they make the 2×2 table degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or x.size == 0:
        return _empty_roc_table()

    fpr, tpr, thr = roc_curve(y, oriented_score(x, direction))
    tab = pd.DataFrame({
        "cutoff": thr if direction == HIGHER else -thr,
        "sensitivity": tpr,
        "specificity": 1.0 - fpr,
        "fpr": fpr,
    })
    tab["youden_j"] = tab["sensitivity"] + tab["specificity"] - 1.0
    tab["dist_01"] = np.hypot(tab["fpr"], 1.0 - tab["sensitivity"])

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tab["n_flagged"] = (tab["sensitivity"] * n_pos + tab["fpr"] * n_neg).round()
    usable = (
        np.isfinite(tab["cutoff"])
        & (tab["n_flagged"] > 0)
        & (tab["n_flagged"] < n_pos + n_neg)
    )
    return tab[usable].reset_index(drop=True)


def _empty_roc_table() -> pd.DataFrame:
    cols = ["cutoff", "sensitivity", "specificity", "fpr",
            "youden_j", "dist_01", "n_flagged"]
    return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})


def _pick_youden(tab: pd.DataFrame) -> int:
    return int(tab["youden_j"].idxmax())


def _pick_closest_01(tab: pd.DataFrame) -> int:
    return int(tab["dist_01"].idxmin())


def _pick_equal(tab: pd.DataFrame) -> int:
    return int((tab["sensitivity"] - tab["specificity"]).abs().idxmin())


def _pick_spec_ge(tab: pd.DataFrame, floor: float = 0.90) -> int | None:
    ok = tab[tab["specificity"] >= floor]
    return int(ok["sensitivity"].idxmax()) if len(ok) else None


def _pick_sens_ge(tab: pd.DataFrame, floor: float = 0.90) -> int | None:
    ok = tab[tab["sensitivity"] >= floor]
    return int(ok["specificity"].idxmax()) if len(ok) else None


# name → (picker, rationale). A picker returns a row index into the ROC table,
# or None when the constraint cannot be met on this cohort.
RULES: dict[str, tuple[Callable[[pd.DataFrame], int | None], str]] = {
    "youden": (_pick_youden, "max(sens + spec − 1) — symmetric, prevalence-free"),
    "closest_01": (_pick_closest_01, "nearest ROC point to the perfect corner"),
    "equal_sens_spec": (_pick_equal, "where the sens and spec curves cross"),
    "spec_ge_90": (_pick_spec_ge, "best sens while spec ≥ 90% — rule-in"),
    "sens_ge_90": (_pick_sens_ge, "best spec while sens ≥ 90% — rule-out"),
}

LITERATURE_RULE = "literature"


def select_cutoff(tab: pd.DataFrame, rule: str) -> int | None:
    """Row index of the cut-point ``rule`` picks, or None if it cannot be met."""
    if rule not in RULES:
        raise KeyError(f"unknown rule {rule!r} — known: {', '.join(RULES)}")
    if tab.empty:
        return None
    return RULES[rule][0](tab)


def chosen_cutoff(x: np.ndarray, y: np.ndarray, direction: str, rule: str) -> float:
    """The cut-point ``rule`` picks on this sample, or NaN if it cannot be met."""
    tab = roc_table(x, y, direction)
    idx = select_cutoff(tab, rule)
    return float("nan") if idx is None else float(tab.loc[idx, "cutoff"])


# --------------------------------------------------------------------------
# Table formatting
# --------------------------------------------------------------------------
def format_pct_ci(row: pd.Series | dict, stem: str) -> str:
    """``62% (52–71)`` from ``stem``, ``stem_lo``, ``stem_hi``."""
    get = row.get
    v, lo, hi = get(stem), get(f"{stem}_lo"), get(f"{stem}_hi")
    if v is None or pd.isna(v):
        return ""
    if lo is None or pd.isna(lo):
        return f"{v * 100:.0f}%"
    return f"{v * 100:.0f}% ({lo * 100:.0f}–{hi * 100:.0f})"


def format_or_ci(row: pd.Series | dict) -> str:
    """``3.01 (1.9–4.8)`` — the odds ratio for being above the cut-point.

    Two decimals below 10 and one above: an OR of 12.4 with two decimals is
    false precision on cells this size.
    """
    get = row.get
    value, lo, hi = get("OR"), get("OR_lo"), get("OR_hi")
    if value is None or pd.isna(value):
        return ""
    fmt = (lambda v: f"{v:.2f}") if float(value) < 10 else (lambda v: f"{v:.1f}")
    if lo is None or pd.isna(lo):
        return fmt(float(value))
    return f"{fmt(float(value))} ({fmt(float(lo))}–{fmt(float(hi))})"
