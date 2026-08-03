"""Single-metric cut-points: ROC table, selection rules, bootstrap stability.

The unit of work is a :class:`Metric` — one continuous predictor plus the
direction in which it points at the outcome. Everything downstream reads
``direction``, so a metric declared the wrong way round shows up as an AUC
below 0.5 rather than as silently inverted cut-points.

A cut-point chosen by maximising a criterion over every candidate in the data
is optimistically biased: it was picked *because* it looked good on these
patients. ``bootstrap_rule`` measures that bias (Harrell's optimism
correction) and gives a percentile interval for the cut-point itself.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

from diagnostic_accuracy import binary_diagnostic_metrics
import plot_style as ps

HIGHER = "higher"
LOWER = "lower"
_DIRECTIONS = (HIGHER, LOWER)


class Metric(NamedTuple):
    """One continuous predictor and how it points at the outcome.

    ``direction="lower"`` means low values are suspicious for high grade (ADC);
    ``"higher"`` means high values are (the volumes).

    ``log_x`` marks a right-skewed metric. It switches the cut-point axis to a
    log scale *and* tells :mod:`risk_curves` to fit the spline on ``log1p(x)``:
    volumes here span three orders of magnitude, and a spline fitted on the raw
    scale spends all its flexibility on the handful of huge tumors.
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


def metric_auc(x: np.ndarray, y: np.ndarray, direction: str) -> float:
    """AUC with the metric oriented so it cannot come out below 0.5 by convention alone."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, oriented_score(x, direction)))


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

# What a constrained rule was asked for, in words, so the "it could not be done"
# message can name the constraint instead of leaving a blank cell.
RULE_CONSTRAINTS: dict[str, str] = {
    "spec_ge_90": "≥ 90% specificity with above-chance sensitivity",
    "sens_ge_90": "≥ 90% sensitivity with above-chance specificity",
}


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
# Applying a fixed cut-point
# --------------------------------------------------------------------------
def operating_point(
    x: np.ndarray, y: np.ndarray, direction: str, cutoff: float,
) -> tuple[float, float]:
    """``(sensitivity, specificity)`` of a fixed cut-point on a given sample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if not np.isfinite(cutoff):
        return float("nan"), float("nan")
    flag = (x <= cutoff) if direction == LOWER else (x >= cutoff)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan"), float("nan")
    return float(flag[pos].mean()), float((~flag[neg]).mean())


def youden_at(x: np.ndarray, y: np.ndarray, direction: str, cutoff: float) -> float:
    """Youden J of a fixed cut-point on a given sample."""
    sens, spec = operating_point(x, y, direction, cutoff)
    return sens + spec - 1.0


# --------------------------------------------------------------------------
# Bootstrap: how much does the cut-point move, and how optimistic is its J?
# --------------------------------------------------------------------------
def bootstrap_rule(
    x: np.ndarray,
    y: np.ndarray,
    direction: str,
    rule: str,
    *,
    n_boot: int = 2000,
    seed: int = 20260801,
    keep_draws: bool = False,
) -> dict:
    """Percentile CI for the chosen cut-point plus optimism-corrected Youden J.

    Two different things come out of the same resampling loop:

    * the spread of the cut-point across resamples — sampling noise in the
      number itself, usually much wider than people expect;
    * Harrell's optimism — pick the cut-point on the resample, score it back on
      the original cohort, average the gap. Sensitivity and specificity quoted
      at a cut-point that was chosen to maximise them on the same data are
      biased upward, and this is the size of that bias.
    """
    apparent = roc_table(x, y, direction)
    idx = select_cutoff(apparent, rule)
    empty = {
        "cutoff_lo": np.nan, "cutoff_hi": np.nan,
        "J_apparent": np.nan, "J_corrected": np.nan,
        "optimism": np.nan, "n_boot_ok": 0,
    }
    if idx is None:
        return empty
    j_apparent = float(apparent.loc[idx, "youden_j"])
    empty["J_apparent"] = j_apparent

    rng = np.random.default_rng(seed)
    n = len(y)
    cutoffs: list[float] = []
    optimisms: list[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        yb, xb = y[take], x[take]
        if yb.sum() < 2 or (1 - yb).sum() < 2:
            continue  # a resample with one class has no ROC
        tab_b = roc_table(xb, yb, direction)
        i_b = select_cutoff(tab_b, rule)
        if i_b is None:
            continue
        cut_b = float(tab_b.loc[i_b, "cutoff"])
        j_orig = youden_at(x, y, direction, cut_b)
        if not np.isfinite(j_orig):
            continue
        cutoffs.append(cut_b)
        optimisms.append(float(tab_b.loc[i_b, "youden_j"]) - j_orig)

    if not cutoffs:
        return empty

    optimism = float(np.mean(optimisms))
    out = {
        "cutoff_lo": float(np.percentile(cutoffs, 2.5)),
        "cutoff_hi": float(np.percentile(cutoffs, 97.5)),
        "J_apparent": j_apparent,
        "J_corrected": j_apparent - optimism,
        "optimism": optimism,
        "n_boot_ok": len(cutoffs),
    }
    if keep_draws:
        out["cutoffs"] = np.asarray(cutoffs)
    return out


# --------------------------------------------------------------------------
# Scoring and summary tables
# --------------------------------------------------------------------------
def score_cutoff(
    df: pd.DataFrame,
    metric: Metric,
    cutoff: float,
    target: str,
    *,
    rule: str,
    source: str = "",
) -> dict:
    """Full 2×2 diagnostic metrics for one metric at one cut-point.

    Delegates to the pipeline's own ``binary_diagnostic_metrics`` so these rows
    are directly comparable with the modelling notebook's accuracy table
    (same Wilson intervals, same Fisher/χ² choice).
    """
    flag = metric.flag(df[metric.col], cutoff)
    row = binary_diagnostic_metrics(
        df, target, f"{metric.col} {metric.op} {cutoff:g}", predictor_series=flag,
    )
    row.update({
        "metric": metric.label,
        "column": metric.col,
        "rule": rule,
        "rule_note": RULES[rule][1] if rule in RULES else source,
        "operator": metric.op,
        "cutoff": float(cutoff),
        "youden_J": row["sensitivity"] + row["specificity"] - 1.0,
        "source": source,
    })
    return row


SUMMARY_COLUMN_ORDER = [
    "metric", "column", "rule", "operator", "cutoff",
    "cutoff_boot_lo", "cutoff_boot_hi", "n_used",
    "TP", "FP", "FN", "TN",
    "sensitivity", "sensitivity_lo", "sensitivity_hi",
    "specificity", "specificity_lo", "specificity_hi",
    "PPV", "PPV_lo", "PPV_hi", "NPV", "NPV_lo", "NPV_hi",
    "OR", "OR_lo", "OR_hi",
    "youden_J", "youden_J_corrected", "optimism", "accuracy", "p", "test",
    "n_bootstrap", "rule_note", "source", "note",
]


def order_summary_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Put the columns a reader scans first, keep everything else after."""
    front = [c for c in SUMMARY_COLUMN_ORDER if c in table.columns]
    return table[front + [c for c in table.columns if c not in front]]


def threshold_summary(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    literature_cutoffs: dict[str, list[tuple[float, str]]] | None = None,
    rules: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int = 20260801,
) -> pd.DataFrame:
    """One row per metric × rule, plus published cut-points scored on this cohort."""
    rules = list(RULES) if rules is None else list(rules)
    literature_cutoffs = literature_cutoffs or {}
    rows: list[dict] = []

    for metric in metrics:
        x, y = metric_arrays(df, metric, target)
        tab = roc_table(x, y, metric.direction)
        for rule in rules:
            idx = select_cutoff(tab, rule)
            if idx is None:
                rows.append({
                    "metric": metric.label, "column": metric.col, "rule": rule,
                    "rule_note": RULES[rule][1], "operator": metric.op,
                    "cutoff": np.nan,
                    "note": "rule not attainable on this cohort",
                })
                continue
            cutoff = float(tab.loc[idx, "cutoff"])
            boot = bootstrap_rule(
                x, y, metric.direction, rule, n_boot=n_boot, seed=seed,
            )
            row = score_cutoff(df, metric, cutoff, target, rule=rule)
            row.update({
                "cutoff_boot_lo": boot["cutoff_lo"],
                "cutoff_boot_hi": boot["cutoff_hi"],
                "youden_J_corrected": boot["J_corrected"],
                "optimism": boot["optimism"],
                "n_bootstrap": boot["n_boot_ok"],
            })
            rows.append(row)

        for cutoff, source in literature_cutoffs.get(metric.col, []):
            # No bootstrap interval: a published cut-point was not estimated
            # here, so it carries no optimism from this cohort. That is exactly
            # why validating an external cut-point is the stronger design.
            rows.append(score_cutoff(
                df, metric, float(cutoff), target,
                rule=LITERATURE_RULE, source=source,
            ))

    return order_summary_columns(pd.DataFrame(rows))


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


def rule_reading(row: pd.Series | dict) -> str:
    """Why this row is not a usable rule, or "" when it is one.

    Two failures used to reach the page as blank cells or as a negative J
    printed like any other number:

    * the constraint could not be met at **any** cut-point — an empty row that
      reads as a missing value rather than as the finding it is;
    * the constraint was met, but only by a cut-point performing **worse than
      chance** (J ≤ 0). Printing "≥3.89, J −0.03" in a table of rules invites
      someone to use it.
    """
    get = row.get
    rule = str(get("rule", ""))
    constraint = RULE_CONSTRAINTS.get(rule)
    if pd.isna(get("cutoff")):
        if constraint:
            return f"not attainable — no cut-point on this cohort reaches {constraint}"
        return "not attainable on this cohort"

    j = get("youden_J")
    if j is not None and not pd.isna(j) and float(j) <= 0.0:
        if constraint:
            return f"no cut-point attains {constraint}"
        return "worse than chance — not a usable rule"
    return ""


def reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """The columns a clinician actually reads, with CIs folded into the estimate.

    A row that is not a usable rule prints the reason instead of the numbers:
    a blank cell and a negative J both read as data when they are findings.
    """
    boot_lo = table.get("cutoff_boot_lo", pd.Series(np.nan, index=table.index))
    boot_hi = table.get("cutoff_boot_hi", pd.Series(np.nan, index=table.index))
    corrected = table.get("youden_J_corrected", pd.Series(np.nan, index=table.index))
    readings = [rule_reading(r) for _, r in table.iterrows()]
    usable = [not bool(r) for r in readings]

    def _blank_unusable(values: Sequence) -> list:
        return [v if ok else "" for v, ok in zip(values, usable)]

    return pd.DataFrame({
        "Metric": table["metric"],
        "Rule": table["rule"],
        # Blanked on an unusable row too: a worse-than-chance rule still has a
        # cut-point, and printing it invites someone to use it.
        "Cut-point": _blank_unusable([
            "" if pd.isna(c) else f"{op}{c:.3g}"
            for op, c in zip(table["operator"], table["cutoff"])
        ]),
        "Cut-point 95% CI": _blank_unusable([
            "" if pd.isna(lo) or pd.isna(hi) else f"{lo:.3g}–{hi:.3g}"
            for lo, hi in zip(boot_lo, boot_hi)
        ]),
        "n": table["n_used"],
        "Sens (95% CI)": _blank_unusable(
            [format_pct_ci(r, "sensitivity") for _, r in table.iterrows()]),
        "Spec (95% CI)": _blank_unusable(
            [format_pct_ci(r, "specificity") for _, r in table.iterrows()]),
        "PPV (95% CI)": _blank_unusable(
            [format_pct_ci(r, "PPV") for _, r in table.iterrows()]),
        "NPV (95% CI)": _blank_unusable(
            [format_pct_ci(r, "NPV") for _, r in table.iterrows()]),
        # The effect size the literature quotes for a dichotomised feature, so a
        # cut-point here reads next to a published one (Magill 2018: OR 1.69 at
        # > 3 cm). An interval crossing 1 says the split separates nothing.
        "OR (95% CI)": _blank_unusable(
            [format_or_ci(r) for _, r in table.iterrows()]),
        "J": _blank_unusable(table["youden_J"].round(2)),
        "J (corr.)": _blank_unusable(corrected.round(2)),
        "Reading": readings,
    })


def cohort_summary(
    df: pd.DataFrame, target: str, *, year_col: str = "entry_year",
) -> pd.DataFrame:
    """Cohort-level counts — the denominators every downstream table is read against.

    Exported on its own because the per-metric table cannot give them: each
    metric drops its own missing rows, so no row of it knows how many patients
    or events the whole cohort holds.

    The accrual window comes along when ``year_col`` is present. A methods
    section has to state it, and deriving it here means the report never has
    to be told what it is.
    """
    y = df[target].astype("boolean")
    n = int(len(df))
    events = int(y.sum())
    row = {
        "n_patients": n,
        "n_high_grade": events,
        "n_benign": int((~y.fillna(False)).sum()) - int(y.isna().sum()),
        "n_outcome_missing": int(y.isna().sum()),
        "prevalence": float(y.mean()) if n else np.nan,
    }
    if year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        if len(years):
            row.update({
                "accrual_first_year": int(years.min()),
                "accrual_last_year": int(years.max()),
                "accrual_n_years": int(years.nunique()),
                "n_year_known": int(len(years)),
            })
    return pd.DataFrame([row])


def metric_cohort_table(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    negative_label: str = "benign",
    positive_label: str = "high grade",
) -> pd.DataFrame:
    """Complete-case count and class balance per metric — the methods-section table."""
    rows = []
    y_all = df[target].astype("boolean")
    for metric in metrics:
        pair = pd.concat([df[metric.col].astype("Float64"), y_all], axis=1).dropna()
        y = pair[target].astype(bool)
        rows.append({
            "metric": metric.label,
            "column": metric.col,
            "direction": metric.direction,
            "n_analysed": len(pair),
            "n_missing": int(len(df) - len(pair)),
            "n_high_grade": int(y.sum()),
            "n_benign": int((~y).sum()),
            "prevalence": float(y.mean()) if len(pair) else np.nan,
            f"median_{negative_label}": pair.loc[~y.to_numpy(), metric.col].median(),
            f"median_{positive_label}": pair.loc[y.to_numpy(), metric.col].median(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def distribution_figure(
    df: pd.DataFrame,
    metric: Metric,
    target: str,
    *,
    negative_label: str = "Benign (WHO 1)",
    positive_label: str = "High grade (WHO 2–3)",
) -> plt.Figure:
    """Raincloud by grade — shows the overlap a single cut-point has to live with."""
    x, y = metric_arrays(df, metric, target)
    benign, high = x[y == 0], x[y == 1]
    auc = metric_auc(x, y, metric.direction)
    p = mannwhitneyu(high, benign, alternative="two-sided").pvalue if (
        benign.size and high.size
    ) else np.nan

    fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_SINGLE, aspect=0.85))
    ps.raincloud(
        ax, [benign, high],
        colors=[ps.PALETTE["low_grade"], ps.PALETTE["high_grade"]],
        rng=ps.deterministic_rng("thresholder", metric.col),
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([negative_label, positive_label])
    ax.set_ylabel(metric.axis_label)
    ps.set_titles(
        ax, metric.label,
        ps.n_subtitle(len(y), extra=f"AUC {auc:.2f} · Mann–Whitney p {ps.format_p(p)}"),
    )
    return fig


def threshold_figure(
    df: pd.DataFrame,
    metric: Metric,
    target: str,
    *,
    n_boot: int = 2000,
    seed: int = 20260801,
    positive_label: str = "High grade (WHO 2–3)",
    negative_label: str = "Benign (WHO 1)",
) -> plt.Figure:
    """Left: ROC with the Youden point. Right: sens/spec/J against the cut-point.

    Both panels are the same ROC table. The right one is not a rule but a
    *display* — it shows how flat the peak is, which is the thing the ROC hides
    and the thing that decides whether quoting a cut-point to three decimals is
    honest.
    """
    x, y = metric_arrays(df, metric, target)
    tab = roc_table(x, y, metric.direction)
    auc = metric_auc(x, y, metric.direction)

    i_youden = select_cutoff(tab, "youden")
    i_cross = select_cutoff(tab, "equal_sens_spec")
    i_spec90 = select_cutoff(tab, "spec_ge_90")
    boot = bootstrap_rule(x, y, metric.direction, "youden", n_boot=n_boot, seed=seed)

    fig, (ax_roc, ax_cut) = plt.subplots(
        1, 2, figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.46),
    )

    ax_roc.plot(tab["fpr"], tab["sensitivity"], color=ps.PALETTE["primary"],
                linewidth=1.6, zorder=3, label=f"ROC (AUC {auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=ps.PALETTE["neutral"],
                linewidth=1.0, zorder=1, label="No discrimination")

    yx, yy = float(tab.loc[i_youden, "fpr"]), float(tab.loc[i_youden, "sensitivity"])
    # J is exactly this vertical gap between the ROC point and the diagonal.
    ax_roc.vlines(yx, yx, yy, color=ps.PALETTE["accent"], linewidth=1.2, zorder=4)
    ax_roc.scatter([yx], [yy], s=26, color=ps.PALETTE["accent"], zorder=5,
                   label=f"Youden J = {tab.loc[i_youden, 'youden_j']:.2f}")
    ax_roc.annotate(
        f"{metric.op}{tab.loc[i_youden, 'cutoff']:.3g}",
        xy=(yx, yy), xytext=(6, -9), textcoords="offset points",
        fontsize=plt.rcParams["font.size"] * 0.8, color=ps.PALETTE["accent"],
    )
    cx, cy = float(tab.loc[i_cross, "fpr"]), float(tab.loc[i_cross, "sensitivity"])
    ax_roc.scatter([cx], [cy], s=22, facecolors="none",
                   edgecolors=ps.PALETTE["good"], linewidths=1.1, zorder=5,
                   label=f"sens = spec ({metric.op}{tab.loc[i_cross, 'cutoff']:.3g})")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_aspect("equal")
    ax_roc.set_xlabel("False-positive rate (1 − specificity)")
    ax_roc.set_ylabel("Sensitivity")
    ps.place_legend(ax_roc, loc="lower right", scale=0.72)
    ps.set_titles(ax_roc, "ROC — one point per cut-point")

    lo, hi = np.nanpercentile(x, [1, 99])
    view = tab[(tab["cutoff"] >= lo) & (tab["cutoff"] <= hi)].sort_values("cutoff")
    if metric.log_x:
        # Volumes span three orders of magnitude; on a linear axis every
        # interesting cut-point piles up against the left spine.
        view = view[view["cutoff"] > 0]
        ax_cut.set_xscale("log")
    ax_cut.plot(view["cutoff"], view["sensitivity"], color=ps.PALETTE["high_grade"],
                linewidth=1.5, label="Sensitivity")
    ax_cut.plot(view["cutoff"], view["specificity"], color=ps.PALETTE["low_grade"],
                linewidth=1.5, label="Specificity")
    ax_cut.plot(view["cutoff"], view["youden_j"], color=ps.PALETTE["neutral"],
                linewidth=1.1, linestyle=":", label="Youden J")

    if np.isfinite(boot["cutoff_lo"]) and np.isfinite(boot["cutoff_hi"]):
        ax_cut.axvspan(boot["cutoff_lo"], boot["cutoff_hi"],
                       color=ps.PALETTE["accent"], alpha=0.10, zorder=0,
                       label="Youden cut-point 95% boot")
    ax_cut.axvline(float(tab.loc[i_youden, "cutoff"]), color=ps.PALETTE["accent"],
                   linewidth=1.2, zorder=4, label="Youden cut-point")
    ax_cut.axvline(float(tab.loc[i_cross, "cutoff"]), color=ps.PALETTE["good"],
                   linewidth=1.0, linestyle="--", zorder=4, label="sens = spec")
    ax_cut.axhline(0.90, color=ps.PALETTE["neutral"], linewidth=0.8,
                   linestyle="-.", zorder=1, label="90% constraint")
    if i_spec90 is not None:
        ax_cut.scatter([float(tab.loc[i_spec90, "cutoff"])], [0.90], s=20,
                       color=ps.PALETTE["neutral"], zorder=5)

    ax_cut.set_ylim(0, 1)
    ax_cut.set_xlabel(f"Cut-point: flag as high grade if {metric.label.lower()} {metric.op} x")
    ax_cut.set_ylabel("Sensitivity / specificity / J")
    # Outside: the curves fill this panel, and which corner is free flips with
    # the metric's direction, so no in-axes placement is safe for all metrics.
    ps.place_legend(ax_cut, outside=True, scale=0.62)
    ps.set_titles(ax_cut, "Trade-off at every cut-point")

    fig.suptitle(
        f"{metric.axis_label} — {positive_label} vs {negative_label}",
        fontsize=plt.rcParams["font.size"] * 1.05,
    )
    return fig


def combined_roc_figure(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
) -> plt.Figure:
    """All metrics on one axes — discrimination, independent of any cut-point choice."""
    fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.9))
    for metric, color in zip(metrics, ps.categorical_palette(len(metrics))):
        x, y = metric_arrays(df, metric, target)
        tab = roc_table(x, y, metric.direction)
        auc = metric_auc(x, y, metric.direction)
        ax.plot(tab["fpr"], tab["sensitivity"], color=color, linewidth=1.5,
                zorder=3, label=f"{metric.label} (AUC {auc:.2f}, n = {len(y)})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=ps.PALETTE["neutral"],
            linewidth=1.0, zorder=1, label="No discrimination")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("Sensitivity")
    ps.place_legend(ax, loc="lower right", scale=0.72)
    ps.set_titles(ax, "Discrimination for high-grade meningioma",
                  "Univariable — each metric on its own complete cases")
    return fig
