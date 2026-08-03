"""How strong is each threshold claim? A graded verdict, not a p < 0.05 gate.

A bare "4 of 4 measurements have a threshold" is both weaker and more
suspicious than a graded answer. It rests entirely on one likelihood-ratio
test clearing 0.05, and it hides the fact that the four claims are not
remotely equally supported: one knee sits at the 7th percentile of patients,
one disappears if the same test is run on the log scale, one is essentially
the 50%-risk crossing under another name.

So the gate is replaced by five criteria, **fixed before the verdicts were
read off**, split into two families:

*Necessary* — if one of these fails, the threshold claim as stated is not
supportable, whatever the others say:

``nonlinearity``       the likelihood-ratio test for curvature, p < 0.05.
                       Without curvature there is no threshold to locate.
``interiority``        at least ``BOUNDARY_PERCENTILE`` of patients on each
                       side of the steepest point. A knee at the edge is an
                       artefact of where recruitment stopped.
``distinct_from_50``   the steepest point lies outside the bootstrap interval
                       of the 50%-risk crossing. Every S-shaped probability
                       curve has its steepest point at the 50% mark; a "knee"
                       that lands there restates the crossing under a new
                       name and carries no extra information.

*Robustness* — the claim survives choices that were not forced on us:

``scale_robust``       non-linear on the clinical scale **and** on the log
                       scale. "Is the risk bent?" is not scale-free, and a
                       finding that flips with the transform is a finding
                       about the transform.
``mice_reproducible``  the knee is found again in at least
                       ``MICE_REPRODUCIBLE_CUT`` of the imputed datasets.

Four grades come out of those:

``strong``    all three necessary criteria pass, and both robustness ones.
``moderate``  all three necessary pass, exactly one robustness one passes.
``fragile``   all three necessary pass but neither robustness one does, **or**
              exactly one necessary criterion other than curvature fails.
``weak``      the curvature test itself fails, or two or more necessary
              criteria fail.

Two further numbers are reported next to the grade but deliberately **do not
score**: the width of the knee's bootstrap interval and the metric's AUC.
They say how *precisely* the knee is located and how much signal the metric
carries at all, which the five binary criteria cannot see. See the module's
test suite for the worked cases.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

import plot_style as ps
from risk_curves import BOUNDARY_PERCENTILE

# --- the pre-specified constants ------------------------------------------
ALPHA = 0.05
# A knee counts as reproducible when this share of the MICE draws finds it
# again. 80% is a round number chosen before any knee_rate was looked at; it
# is stated here rather than in prose so it cannot drift.
MICE_REPRODUCIBLE_CUT = 0.80
# The risk level a knee must not merely restate.
RESTATED_LEVEL = 0.50
# Below this share of resamples, the crossing's own interval rests on too few
# draws to test against, and ``distinct_from_50`` passes with that recorded.
CROSSING_ESTABLISHED_FRAC = 0.50

NECESSARY = "necessary"
ROBUSTNESS = "robustness"

GRADE_STRONG = "strong"
GRADE_MODERATE = "moderate"
GRADE_FRAGILE = "fragile"
GRADE_WEAK = "weak"
GRADES = (GRADE_STRONG, GRADE_MODERATE, GRADE_FRAGILE, GRADE_WEAK)


class Criterion(NamedTuple):
    """One pre-specified test, with where its number comes from."""

    key: str
    name: str
    source: str
    rule: str
    family: str


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "nonlinearity", "Non-linearity",
        "Likelihood-ratio test, spline vs straight line (section 3)",
        f"p &lt; {ALPHA}", NECESSARY),
    Criterion(
        "interiority", "Knee interiority",
        "Percentile of patients below the steepest point (section 3)",
        f"≥ {BOUNDARY_PERCENTILE:.0f}% of patients on each side", NECESSARY),
    Criterion(
        "distinct_from_50", "Knee ≠ 50%-risk point",
        f"Bootstrap interval of the {RESTATED_LEVEL:.0%} risk crossing (section 3)",
        "steepest point falls outside that interval", NECESSARY),
    Criterion(
        "scale_robust", "Scale robustness",
        "<code>nonlinearity_p_log_scale</code> (section 3)",
        "non-linear on the clinical scale <b>and</b> on the log scale", ROBUSTNESS),
    Criterion(
        "mice_reproducible", "MICE reproducibility",
        "<code>knee_rate</code> across the imputed datasets (section 6)",
        f"knee found again in ≥ {MICE_REPRODUCIBLE_CUT:.0%} of draws", ROBUSTNESS),
)

CRITERIA_BY_KEY = {c.key: c for c in CRITERIA}


def criteria_table() -> pd.DataFrame:
    """The hierarchy itself, for printing in the methods paragraph."""
    return pd.DataFrame([
        {"Criterion": c.name, "Family": c.family, "Source": c.source, "Rule": c.rule}
        for c in CRITERIA
    ])


# --------------------------------------------------------------------------
# Multiple testing across the pre-specified family
# --------------------------------------------------------------------------
def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values, NaNs passed through.

    Holm controls the same family-wise error rate as Bonferroni but is
    uniformly more powerful, so a result that survives Bonferroni always
    survives Holm and never the other way round. Both are reported because
    the pair is the honest answer to "did you correct?": Holm is the correct
    correction, Bonferroni is the one a reviewer will have in mind.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out

    order = finite[np.argsort(p[finite], kind="stable")]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        # Step-down: each p is scaled by how many hypotheses remain, then made
        # monotone so an adjusted value can never fall below an earlier one.
        running = max(running, float(p[idx]) * (m - rank))
        out[idx] = min(running, 1.0)
    return out


def bonferroni_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Single-step Bonferroni, capped at 1."""
    p = np.asarray(p_values, dtype=float)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    return np.minimum(p * m, 1.0)


def multiplicity_table(risk_table: pd.DataFrame, *, alpha: float = ALPHA) -> pd.DataFrame:
    """The primary family — one non-linearity LRT per metric — raw and adjusted.

    Pre-specified as *the* family for this analysis. The cut-point selection
    rules and the combination menu are not in it: those are selections, not
    hypothesis tests, and they are handled by optimism correction instead.
    """
    if risk_table.empty or "nonlinearity_p" not in risk_table.columns:
        return pd.DataFrame()

    p = pd.to_numeric(risk_table["nonlinearity_p"], errors="coerce").to_numpy(dtype=float)
    holm = holm_adjust(p)
    bonf = bonferroni_adjust(p)
    return pd.DataFrame({
        "metric": risk_table["metric"],
        "column": risk_table["column"],
        "family": "non-linearity LRT (primary)",
        "n_tests": int(np.isfinite(p).sum()),
        "p_raw": p,
        "p_holm": holm,
        "p_bonferroni": bonf,
        "survives_raw": p < alpha,
        "survives_holm": holm < alpha,
        "survives_bonferroni": bonf < alpha,
    })


def multiplicity_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """Raw next to Holm next to Bonferroni, in the words the report uses."""
    if table.empty:
        return pd.DataFrame()
    mark = {True: "survives", False: "does not survive"}
    return pd.DataFrame({
        "Metric": table["metric"],
        "Non-linearity p": [ps.format_p(v) for v in table["p_raw"]],
        "Holm-adjusted p": [ps.format_p(v) for v in table["p_holm"]],
        "Holm": [mark.get(bool(v), "—") for v in table["survives_holm"]],
        "Bonferroni-adjusted p": [ps.format_p(v) for v in table["p_bonferroni"]],
        "Bonferroni": [mark.get(bool(v), "—") for v in table["survives_bonferroni"]],
    })


# --------------------------------------------------------------------------
# The five criteria, one metric at a time
# --------------------------------------------------------------------------
def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _truthy(value) -> bool:
    """CSV round-tripping turns booleans into 'True'/'False' strings."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return bool(value)


def _score_nonlinearity(row: pd.Series, alpha: float) -> tuple[bool, str]:
    p = _f(row.get("nonlinearity_p"))
    if not np.isfinite(p):
        return False, "no curvature test available"
    ok = p < alpha
    return ok, f"p {ps.format_p(p)} on the clinical scale"


def _score_interiority(row: pd.Series, floor: float) -> tuple[bool, str]:
    pct = _f(row.get("steepest_pct_of_patients"))
    if not np.isfinite(pct):
        return False, "steepest point not located"
    ok = bool(floor <= pct <= 100.0 - floor)
    return ok, f"{pct:.0f}% of patients below the knee"


def _score_distinct_from_50(row: pd.Series, level: float) -> tuple[bool, str]:
    tag = f"risk_{int(round(level * 100))}"
    knee = _f(row.get("steepest_x"))
    lo, hi = _f(row.get(f"{tag}_lo")), _f(row.get(f"{tag}_hi"))
    found = _f(row.get(f"{tag}_found_frac"))

    if not np.isfinite(knee):
        return False, "steepest point not located"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True, f"{level:.0%} risk never reached, so the knee cannot restate it"
    if np.isfinite(found) and found < CROSSING_ESTABLISHED_FRAC:
        return True, (f"{level:.0%} risk located in only {found:.0%} of resamples, "
                      f"so there is no crossing for the knee to restate")
    inside = bool(lo <= knee <= hi)
    where = f"knee {knee:.3g} vs {level:.0%} crossing CI {lo:.3g}–{hi:.3g}"
    return (not inside), (f"{where} — inside it" if inside else f"{where} — outside it")


def _score_scale_robust(row: pd.Series, alpha: float) -> tuple[bool, str]:
    raw_ok = _truthy(row.get("nonlinear"))
    p_log = _f(row.get("nonlinearity_p_log_scale"))
    if not np.isfinite(p_log):
        return False, "log-scale test unavailable"
    log_ok = p_log < alpha
    ok = bool(raw_ok and log_ok)
    return ok, f"log-scale p {ps.format_p(p_log)}"


def _score_mice(row: pd.Series, knee_rate: float, cut: float) -> tuple[bool, str]:
    if not np.isfinite(knee_rate):
        return False, "not checked across imputed datasets"
    return bool(knee_rate >= cut), f"knee found in {knee_rate:.0%} of draws"


def grade(passes: dict[str, bool]) -> str:
    """Map the five pass/fail flags onto the four-level verdict."""
    necessary = [c.key for c in CRITERIA if c.family == NECESSARY]
    robustness = [c.key for c in CRITERIA if c.family == ROBUSTNESS]

    if not passes.get("nonlinearity", False):
        return GRADE_WEAK
    n_failed = sum(1 for k in necessary if not passes.get(k, False))
    if n_failed >= 2:
        return GRADE_WEAK
    if n_failed == 1:
        return GRADE_FRAGILE

    n_robust = sum(1 for k in robustness if passes.get(k, False))
    if n_robust == len(robustness):
        return GRADE_STRONG
    if n_robust >= 1:
        return GRADE_MODERATE
    return GRADE_FRAGILE


def threshold_evidence(
    risk_table: pd.DataFrame,
    risk_stability: pd.DataFrame | None = None,
    *,
    alpha: float = ALPHA,
    mice_cut: float = MICE_REPRODUCIBLE_CUT,
    interiority_floor: float = BOUNDARY_PERCENTILE,
    restated_level: float = RESTATED_LEVEL,
) -> pd.DataFrame:
    """One row per metric: the five criteria, the grade, and what limits it.

    Every input column already exists in ``03_risk_curves.csv`` and
    ``21_risk_curve_stability.csv``; nothing is refitted here, so the grade
    and the tables can never disagree.
    """
    if risk_table.empty:
        return pd.DataFrame()

    knee_rates: dict[str, float] = {}
    if risk_stability is not None and not risk_stability.empty:
        if {"column", "knee_rate"} <= set(risk_stability.columns):
            knee_rates = {str(c): _f(v) for c, v
                          in zip(risk_stability["column"], risk_stability["knee_rate"])}

    rows: list[dict] = []
    for _, row in risk_table.iterrows():
        col = str(row.get("column", ""))
        scored = {
            "nonlinearity": _score_nonlinearity(row, alpha),
            "interiority": _score_interiority(row, interiority_floor),
            "distinct_from_50": _score_distinct_from_50(row, restated_level),
            "scale_robust": _score_scale_robust(row, alpha),
            "mice_reproducible": _score_mice(
                row, knee_rates.get(col, float("nan")), mice_cut),
        }
        passes = {k: ok for k, (ok, _) in scored.items()}
        verdict = grade(passes)
        failed = [CRITERIA_BY_KEY[c.key].name for c in CRITERIA if not passes[c.key]]

        # Reported next to the grade, not scored by it: how precisely the knee
        # is pinned down, and whether the metric discriminates at all.
        lo, hi = _f(row.get("steepest_lo")), _f(row.get("steepest_hi"))
        ci_ratio = (hi / lo) if (np.isfinite(lo) and np.isfinite(hi) and lo > 0) else np.nan

        entry = {
            "metric": row.get("metric", ""),
            "column": col,
            "verdict": verdict,
            "n_criteria_passed": int(sum(passes.values())),
            "n_criteria": len(CRITERIA),
            "limiting_criterion": failed[0] if failed else "",
            "failed_criteria": "; ".join(failed),
            "knee_ci_ratio": ci_ratio,
            "AUC": _f(row.get("AUC")),
        }
        for key, (ok, detail) in scored.items():
            entry[f"pass_{key}"] = bool(ok)
            entry[f"detail_{key}"] = detail
        entry["verdict_note"] = _verdict_note(verdict, failed, scored)
        rows.append(entry)

    return pd.DataFrame(rows)


def _verdict_note(verdict: str, failed: list[str], scored: dict) -> str:
    """The grade with its binding constraint named — what goes in the cell."""
    if not failed:
        return f"{verdict} — all {len(CRITERIA)} criteria met"
    details = {CRITERIA_BY_KEY[k].name: d for k, (_, d) in scored.items()}
    named = "; ".join(f"{name} fails ({details.get(name, '')})" for name in failed)
    return f"{verdict} — {named}"


def evidence_reading_view(table: pd.DataFrame) -> pd.DataFrame:
    """The graded verdict in the columns a clinician reads."""
    if table.empty:
        return pd.DataFrame()
    tick = {True: "pass", False: "fail"}
    out = {"Metric": table["metric"], "Evidence": table["verdict"]}
    for c in CRITERIA:
        out[c.name] = [
            f"{tick[bool(ok)]} — {detail}"
            for ok, detail in zip(table[f"pass_{c.key}"], table[f"detail_{c.key}"])
        ]
    out["Criteria met"] = (table["n_criteria_passed"].astype(str)
                           + " of " + table["n_criteria"].astype(str))
    out["What limits it"] = table["limiting_criterion"].replace("", "—")
    return pd.DataFrame(out)
