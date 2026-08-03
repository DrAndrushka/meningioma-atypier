"""Do two or more cut-points beat one?

The design decision that makes this defensible: **the cut-points are fixed
before they are combined.** They come from the single-metric analysis or from
the literature, and only the way of joining them is varied here. Searching over
cut-points *and* combination rules at once would be thousands of comparisons on
105 events, and every "improvement" found that way is noise.

Three families of combined rule are scored:

``AND``    both flags positive — buys specificity, spends sensitivity ("rule in")
``OR``     either flag positive — buys sensitivity, spends specificity ("rule out")
``count``  how many of the k flags are positive, used as an ordinal score

and compared against three benchmarks that any honest claim of improvement has
to clear:

1. the best **single** cut-point — the comparison actually asked for;
2. a logistic model on the **uncut, continuous** metrics — the strong one.
   Dichotomising throws away information, so a combined rule that cannot beat
   this is evidence that cutting the variables up was the wrong move;
3. bootstrap **optimism** for "best of the menu" — the winner of a dozen
   candidate rules always looks better on its own data than it will elsewhere.

Missing values follow Kleene logic (pandas nullable booleans): ``False AND
missing`` is ``False`` because one negative flag already settles an AND rule,
while ``True AND missing`` stays missing. Patients whose combined flag is still
missing are dropped by the scorer, and every row reports its own ``n_used``.
"""
from __future__ import annotations

from itertools import combinations as _iter_combinations
from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from diagnostic_accuracy import binary_diagnostic_metrics
import plot_style as ps
from thresholds import (
    LITERATURE_RULE,
    Metric,
    RULES,
    metric_arrays,
    roc_table,
    select_cutoff,
)

AND = "AND"
OR = "OR"
LOGICS = (AND, OR)


class CutPoint(NamedTuple):
    """A metric frozen at one cut-point, ready to be combined with others."""

    metric: Metric
    cutoff: float
    rule: str = "youden"
    source: str = ""

    @property
    def col(self) -> str:
        return self.metric.col

    @property
    def label(self) -> str:
        return f"{self.metric.label} {self.metric.op} {self.cutoff:.3g}"

    @property
    def short_label(self) -> str:
        return f"{self.metric.label} {self.metric.op}{self.cutoff:.3g}"

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return self.metric.flag(df[self.metric.col], self.cutoff)


def cutpoints_for_rule(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    rule: str = "youden",
) -> list[CutPoint]:
    """Freeze each metric at the cut-point ``rule`` selects on this cohort."""
    out: list[CutPoint] = []
    for metric in metrics:
        x, y = metric_arrays(df, metric, target)
        tab = roc_table(x, y, metric.direction)
        idx = select_cutoff(tab, rule)
        if idx is None:
            continue
        out.append(CutPoint(metric, float(tab.loc[idx, "cutoff"]), rule=rule))
    return out


def cutpoints_from_literature(
    metrics: Sequence[Metric],
    literature_cutoffs: dict[str, list[tuple[float, str]]],
) -> list[CutPoint]:
    """Published cut-points as combinable objects — the pre-specified menu.

    Combining externally derived cut-points is the strongest version of this
    analysis: nothing about the rule was chosen on these patients, so the
    accuracy it reports needs no optimism correction at all.
    """
    by_col = {m.col: m for m in metrics}
    out: list[CutPoint] = []
    for col, entries in literature_cutoffs.items():
        metric = by_col.get(col)
        if metric is None:
            continue
        for cutoff, source in entries:
            out.append(CutPoint(metric, float(cutoff),
                                rule=LITERATURE_RULE, source=source))
    return out


# --------------------------------------------------------------------------
# Flags and their combinations
# --------------------------------------------------------------------------
def flag_frame(df: pd.DataFrame, cutpoints: Sequence[CutPoint]) -> pd.DataFrame:
    """One nullable-boolean column per cut-point, indexed like ``df``."""
    return pd.DataFrame(
        {cp.col: cp.flag(df) for cp in cutpoints}, index=df.index,
    )


def combine_flags(flags: pd.DataFrame, cols: Sequence[str], logic: str) -> pd.Series:
    """AND/OR across ``cols`` under Kleene logic (missing only where it matters)."""
    if logic not in LOGICS:
        raise ValueError(f"logic must be one of {LOGICS}, got {logic!r}")
    out = flags[cols[0]].astype("boolean")
    for col in cols[1:]:
        other = flags[col].astype("boolean")
        out = (out & other) if logic == AND else (out | other)
    return out


def _score(
    df: pd.DataFrame,
    target: str,
    series: pd.Series,
    *,
    label: str,
    kind: str,
    members: Sequence[str],
    note: str = "",
) -> dict:
    row = binary_diagnostic_metrics(df, target, label, predictor_series=series)
    row.update({
        "rule_label": label,
        "kind": kind,
        "n_criteria": len(members),
        "members": " + ".join(members),
        "youden_J": row["sensitivity"] + row["specificity"] - 1.0,
        "note": note,
    })
    return row


def single_rule_table(
    df: pd.DataFrame, cutpoints: Sequence[CutPoint], target: str,
) -> pd.DataFrame:
    """Each cut-point on its own — the benchmark the combinations must beat."""
    flags = flag_frame(df, cutpoints)
    return pd.DataFrame([
        _score(df, target, flags[cp.col], label=cp.label,
               kind="single", members=[cp.short_label])
        for cp in cutpoints
    ])


def pair_rule_table(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    max_size: int = 2,
) -> pd.DataFrame:
    """Every AND and OR combination up to ``max_size`` cut-points.

    ``max_size=2`` by default. Triples are available but the cells thin out
    fast: an AND of three flags on ~105 events routinely lands in single
    figures, and a sensitivity computed from eight patients is not a number
    worth putting on a poster.
    """
    flags = flag_frame(df, cutpoints)
    by_col = {cp.col: cp for cp in cutpoints}
    rows: list[dict] = []
    for size in range(2, int(max_size) + 1):
        for combo in _iter_combinations([cp.col for cp in cutpoints], size):
            members = [by_col[c].short_label for c in combo]
            for logic in LOGICS:
                series = combine_flags(flags, combo, logic)
                label = f" {logic} ".join(by_col[c].label for c in combo)
                rows.append(_score(
                    df, target, series, label=label,
                    kind=logic.lower(), members=members,
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Count score — "how many criteria are met"
# --------------------------------------------------------------------------
def count_series(flags: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``(n_positive, n_observed)`` per patient across all flag columns."""
    numeric = flags.astype("Float64")
    return numeric.sum(axis=1, skipna=True), numeric.notna().sum(axis=1)


def count_score_table(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    complete_only: bool = True,
) -> pd.DataFrame:
    """Observed high-grade rate at each count of met criteria, with Wilson CIs.

    Restricted to patients with **every** flag observed by default: a count of
    "2 of 4" means something different when one of the four was never measured.
    The MICE-draw version in :mod:`stability` recovers the dropped patients.
    """
    flags = flag_frame(df, cutpoints)
    counts, observed = count_series(flags)
    k = len(cutpoints)

    keep = (observed == k) if complete_only else (observed > 0)
    y = df[target].astype("boolean")
    keep = keep & y.notna()

    frame = pd.DataFrame({
        "count": counts[keep].astype(int),
        "y": y[keep].astype(bool),
    })
    rows = []
    for count in range(k + 1):
        grp = frame[frame["count"] == count]
        n, events = len(grp), int(grp["y"].sum())
        lo, hi = ps.wilson_ci(events, n) if n else (np.nan, np.nan)
        rows.append({
            "n_criteria_met": count,
            "n": n,
            "n_high_grade": events,
            "risk": (events / n) if n else np.nan,
            "risk_lo": float(lo) if n else np.nan,
            "risk_hi": float(hi) if n else np.nan,
        })
    out = pd.DataFrame(rows)
    out.attrs["n_scored"] = int(keep.sum())
    out.attrs["k"] = k
    return out


def count_threshold_table(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    complete_only: bool = True,
) -> pd.DataFrame:
    """The count score used as a test: "≥1 criterion", "≥2 criteria", …

    This is the clinically usable form of a multi-threshold rule — no logic
    gates to remember, just how many of the boxes a tumor ticks.
    """
    flags = flag_frame(df, cutpoints)
    counts, observed = count_series(flags)
    k = len(cutpoints)
    valid = (observed == k) if complete_only else (observed > 0)

    rows = []
    for cut in range(1, k + 1):
        series = pd.Series(pd.NA, index=df.index, dtype="boolean")
        series[valid] = (counts[valid] >= cut).astype("boolean")
        rows.append(_score(
            df, target, series,
            label=f"≥ {cut} of {k} criteria",
            kind="count", members=[cp.short_label for cp in cutpoints],
            note="all flags observed" if complete_only else "≥1 flag observed",
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Benchmark: the uncut, continuous model
# --------------------------------------------------------------------------
def continuous_model_benchmark(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    n_boot: int = 500,
    seed: int = 20260801,
) -> dict:
    """Logistic model on the continuous metrics — the AUC a cut rule must beat.

    Fitted on the patients who have *all* the metrics, so it is scored on the
    same complete-case set as the count rules. The optimism correction is
    Harrell's: refit on each resample, score on the original cohort, average
    the gap.
    """
    cols = [m.col for m in metrics if m.col in df.columns]
    frame = df[cols + [target]].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if next(m for m in metrics if m.col == col).log_x:
            frame[col] = np.log1p(frame[col])
    frame[target] = frame[target].astype("boolean")
    frame = frame.dropna()

    n = len(frame)
    y = frame[target].astype(int).to_numpy()
    X = sm.add_constant(frame[cols].astype(float).to_numpy(), has_constant="add")
    blank = {
        "kind": "continuous_model", "rule_label": "Logistic on continuous metrics",
        "n_used": n, "AUC_apparent": np.nan, "AUC_corrected": np.nan,
        "optimism": np.nan, "n_bootstrap": 0, "members": " + ".join(cols),
    }
    if n < 30 or len(np.unique(y)) < 2:
        return blank

    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception:
        return blank
    apparent = float(roc_auc_score(y, fit.predict(X)))

    rng = np.random.default_rng(seed)
    optimisms: list[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        yb, Xb = y[take], X[take]
        if len(np.unique(yb)) < 2:
            continue
        try:
            fb = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        try:
            boot_auc = float(roc_auc_score(yb, fb.predict(Xb)))
            orig_auc = float(roc_auc_score(y, fb.predict(X)))
        except ValueError:
            continue
        optimisms.append(boot_auc - orig_auc)

    optimism = float(np.mean(optimisms)) if optimisms else np.nan
    blank.update({
        "AUC_apparent": apparent,
        "AUC_corrected": apparent - optimism if optimisms else np.nan,
        "optimism": optimism,
        "n_bootstrap": len(optimisms),
    })
    return blank


# --------------------------------------------------------------------------
# Optimism of "best rule off the menu"
# --------------------------------------------------------------------------
def bootstrap_best_rule(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    criterion: str = "youden_J",
    n_boot: int = 500,
    seed: int = 20260801,
    max_size: int = 2,
    kinds: Sequence[str] | None = None,
) -> dict:
    """How much of the winning rule's advantage is having been picked here?

    On each resample, the whole menu (singles + AND + OR + count) is rebuilt,
    the best rule by ``criterion`` is selected, and that *same* rule is then
    scored on the original cohort. The average gap is the optimism of the
    selection itself — the part of the winner's performance that will not
    survive being applied to new patients.

    ``kinds`` restricts the menu the winner is chosen from, and it is what
    makes the head-to-head comparison fair. ``kinds=("single",)`` measures the
    optimism of picking the best of the four single criteria, which is exactly
    the same kind of selection the combined rule is corrected for. Comparing a
    corrected combination against an *uncorrected* single flatters the
    combination by the whole of the single's own selection optimism.

    Note this is a different correction from the per-metric one in
    :mod:`thresholds`: that one pays for choosing a cut-point on a fixed
    metric, this one pays for choosing which metric (or rule) to report.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    gaps: list[float] = []
    winners: list[str] = []

    def _menu(frame: pd.DataFrame) -> pd.DataFrame:
        table = full_rule_menu(frame, cutpoints, target, max_size=max_size)
        if kinds is not None and not table.empty:
            table = table[table["kind"].isin(list(kinds))]
        return table

    apparent = _menu(df)
    if apparent.empty or apparent[criterion].isna().all():
        return {"optimism": np.nan, "n_bootstrap": 0,
                "best_rule": "", "J_apparent": np.nan, "J_corrected": np.nan,
                "winner_stability": np.nan}
    best_idx = apparent[criterion].idxmax()
    best_label = str(apparent.loc[best_idx, "rule_label"])
    j_apparent = float(apparent.loc[best_idx, criterion])

    by_label = apparent.set_index("rule_label")[criterion]
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        boot = df.iloc[take].reset_index(drop=True)
        if boot[target].astype("boolean").sum() < 5:
            continue
        menu_b = _menu(boot)
        if menu_b.empty or menu_b[criterion].isna().all():
            continue
        i_b = menu_b[criterion].idxmax()
        label_b = str(menu_b.loc[i_b, "rule_label"])
        if label_b not in by_label.index:
            continue
        on_original = float(by_label.loc[label_b])
        if not np.isfinite(on_original):
            continue
        gaps.append(float(menu_b.loc[i_b, criterion]) - on_original)
        winners.append(label_b)

    optimism = float(np.mean(gaps)) if gaps else np.nan
    return {
        "optimism": optimism,
        "n_bootstrap": len(gaps),
        "best_rule": best_label,
        "J_apparent": j_apparent,
        "J_corrected": j_apparent - optimism if gaps else np.nan,
        # How often the same rule wins. Below ~0.5 the "best combination" is a
        # coin toss between near-equivalent rules, not a finding.
        "winner_stability": (
            float(np.mean([w == best_label for w in winners])) if winners else np.nan
        ),
    }


def full_rule_menu(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    max_size: int = 2,
) -> pd.DataFrame:
    """Singles + AND/OR pairs + count rules in one table, in comparison order."""
    parts = [
        single_rule_table(df, cutpoints, target),
        pair_rule_table(df, cutpoints, target, max_size=max_size),
        count_threshold_table(df, cutpoints, target),
    ]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    front = [
        "rule_label", "kind", "n_criteria", "members", "n_used",
        "TP", "FP", "FN", "TN",
        "sensitivity", "sensitivity_lo", "sensitivity_hi",
        "specificity", "specificity_lo", "specificity_hi",
        "PPV", "PPV_lo", "PPV_hi", "NPV", "NPV_lo", "NPV_hi",
        "youden_J", "accuracy", "p", "test", "note",
    ]
    cols = [c for c in front if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def combination_reading_view(menu: pd.DataFrame, *, top: int | None = None) -> pd.DataFrame:
    """Rules ranked by Youden J, in the columns a clinician reads."""
    from thresholds import format_pct_ci

    ranked = menu.sort_values("youden_J", ascending=False)
    if top is not None:
        ranked = ranked.head(int(top))
    return pd.DataFrame({
        "Rule": ranked["rule_label"],
        "Type": ranked["kind"],
        "n": ranked["n_used"],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in ranked.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in ranked.iterrows()],
        "PPV (95% CI)": [format_pct_ci(r, "PPV") for _, r in ranked.iterrows()],
        "NPV (95% CI)": [format_pct_ci(r, "NPV") for _, r in ranked.iterrows()],
        "TP/FP/FN/TN": [
            f"{r['TP']:.0f}/{r['FP']:.0f}/{r['FN']:.0f}/{r['TN']:.0f}"
            if pd.notna(r["TP"]) else ""
            for _, r in ranked.iterrows()
        ],
        "J": ranked["youden_J"].round(2),
    })


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
_KIND_COLOR = {
    "single": ps.PALETTE["neutral"],
    "and": ps.PALETTE["primary"],
    "or": ps.PALETTE["accent"],
    "count": ps.PALETTE["good"],
}


def combination_figure(
    menu: pd.DataFrame,
    *,
    benchmark: dict | None = None,
    top: int = 12,
) -> plt.Figure:
    """Every rule in sensitivity–specificity space, singles marked apart.

    A combined rule is only interesting if it sits **up and to the right** of
    every single cut-point. Rules that merely trade one for the other have
    moved along the same trade-off, not improved on it — which is what AND and
    OR mostly do, and is the honest headline of this section when it happens.
    """
    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2, figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.46),
    )

    for kind, grp in menu.groupby("kind", sort=False):
        ax_scatter.scatter(
            grp["specificity"], grp["sensitivity"],
            s=30 if kind == "single" else 22,
            color=_KIND_COLOR.get(kind, ps.PALETTE["neutral"]),
            marker="D" if kind == "single" else "o",
            edgecolors="none", alpha=0.85, zorder=4, label=kind,
        )
    # Iso-J lines: everything on one of these is equally good under Youden.
    for j in (0.1, 0.2, 0.3, 0.4, 0.5):
        spec = np.linspace(0, 1, 50)
        ax_scatter.plot(spec, j + 1 - spec, color=ps.PALETTE["neutral"],
                        linewidth=0.5, linestyle=":", zorder=1)
    ax_scatter.set_xlim(0, 1.02)
    ax_scatter.set_ylim(0, 1.02)
    ax_scatter.set_xlabel("Specificity")
    ax_scatter.set_ylabel("Sensitivity")
    ps.place_legend(ax_scatter, loc="lower left", scale=0.68)
    ps.set_titles(ax_scatter, "Every rule in one space",
                  "Dotted lines join equally good rules (equal J)")

    ranked = menu.sort_values("youden_J", ascending=False).head(int(top)).iloc[::-1]
    pos = np.arange(len(ranked))
    ax_bar.barh(pos, ranked["youden_J"],
                color=[_KIND_COLOR.get(k, ps.PALETTE["neutral"]) for k in ranked["kind"]],
                height=0.7, zorder=3)
    ax_bar.set_yticks(pos)
    ax_bar.set_yticklabels(
        [ps._wrap_label(str(lbl), 34) for lbl in ranked["rule_label"]],
        fontsize=plt.rcParams["font.size"] * 0.62,
    )
    ax_bar.set_xlabel("Youden J")

    right = max(float(ranked["youden_J"].max()), 0.1)
    if benchmark and np.isfinite(benchmark.get("AUC_corrected", np.nan)):
        # A logistic AUC is not a J, but 2·(AUC − 0.5) is the J of the best
        # single split of that model's own risk score — the fairest one-line
        # comparison available without re-deriving a cut-point.
        equiv = 2.0 * (benchmark["AUC_corrected"] - 0.5)
        ax_bar.axvline(equiv, color=ps.PALETTE["bad"], linewidth=1.1,
                       linestyle="--", zorder=4,
                       label=f"Continuous model (AUC {benchmark['AUC_corrected']:.2f})")
        ps.place_legend(ax_bar, loc="lower right", scale=0.62)
        right = max(right, equiv)
    # Headroom so the benchmark line never lands on the spine.
    ax_bar.set_xlim(0, right * 1.35)

    ps.set_titles(ax_bar, f"Top {len(ranked)} rules by J",
                  "Uncorrected — see the optimism estimate below")
    return fig


def count_score_figure(
    counts: pd.DataFrame,
    *,
    cutpoints: Sequence[CutPoint] = (),
    prevalence: float | None = None,
) -> plt.Figure:
    """Observed risk against the number of criteria met — the poster figure.

    No logic gates, no ROC: a radiologist counts how many of the boxes the
    tumor ticks and reads the risk off the bar. The Wilson intervals are there
    because the end bars usually rest on very few patients.
    """
    fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.62))
    usable = counts[counts["n"] > 0]

    ps.proportion_bars(
        ax, usable["n_high_grade"].to_numpy(), usable["n"].to_numpy(),
        positions=usable["n_criteria_met"].to_numpy(dtype=float),
        color=ps.PALETTE["high_grade"], width=0.62, annotate=True,
    )
    if prevalence is not None and np.isfinite(prevalence):
        ax.axhline(prevalence * 100, color=ps.PALETTE["neutral"], linewidth=0.9,
                   linestyle="-.", zorder=2,
                   label=f"Cohort rate ({prevalence * 100:.0f}%)")
        ps.place_legend(ax, loc="upper left", scale=0.7)

    ax.set_xticks(usable["n_criteria_met"].to_numpy(dtype=float))
    ax.set_xlabel(f"Criteria met (of {counts.attrs.get('k', len(cutpoints))})")
    ax.set_ylabel("High grade (%)")
    ax.set_ylim(0, 100)

    criteria = " · ".join(cp.short_label for cp in cutpoints)
    ps.set_titles(
        ax, "Risk by number of criteria met",
        ps.n_subtitle(int(counts.attrs.get("n_scored", usable["n"].sum())),
                      extra=criteria) if criteria else None,
    )
    return fig
