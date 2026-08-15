"""Step 9 — what does a cut-point cost, compared with the number it replaces?

Turning a measurement into a yes/no answer throws information away. Two patients
with ADC 0.71 and 0.30 become the same patient; two with 0.71 and 0.73 land on
opposite sides of a line neither of them is near. The question is how much that
costs, and whether the convenience is worth it.

Three ways of asking, reported side by side:

**Discrimination.** The AUC of the raw number against the AUC of the yes/no
flag. For a yes/no test the AUC is just ``(Se + Sp) / 2``, so the comparison is
exact and needs no extra machinery. Compared with DeLong on the same patients,
because the two scores are not independent — a patient who is easy to classify
is easy for both, and treating them as separate studies would overstate the
difference.

**Information retained.** ``(AUC_binary − 0.5) / (AUC_continuous − 0.5)`` — the
share of the measurement's discriminating power that survives the cut. A value
of 0.75 means a quarter of what the measurement knew was discarded to gain a
yes/no answer.

**Effect size.** The odds ratio per 1 SD of the raw number against the odds
ratio for being on the wrong side of the cut-point. These are not on the same
scale and must never be compared as numbers — the point of showing them together
is that the dichotomised OR usually *looks* bigger while describing less.

Odds ratios use Wald intervals. With 105 events, one continuous predictor at a
time and no separation, the likelihood is near-quadratic and profile likelihood
would agree to two decimals; a visibly lopsided interval is the signal that this
has stopped being true, and ``asymmetric`` flags it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from accuracy import flag
from measurements import (MEASUREMENTS, MEASUREMENTS_BY_COL, Measurement,
                          stratum_mask)
from scales import standardise as _standardise
from separation import auc_with_ci, delong_compare, oriented_score

OUTCOME = "high_grade"

# Above this ratio between the two halves of a log-odds interval, Wald has
# stopped approximating the likelihood and the estimate needs a penalised fit.
ASYMMETRY_LIMIT = 1.25


def standardise(x: np.ndarray, log_x: bool) -> np.ndarray:
    """Centre and scale so one unit is one SD, log-transforming where declared.

    Kept as a name here because this module's readers look for it here, but the
    implementation lives in :mod:`scales` so the modelling phase cannot drift
    onto a different scale and print a different odds ratio for the same column.
    """
    return _standardise(x, log_x)


def odds_ratio(y: np.ndarray, x: np.ndarray, *, alpha: float = 0.05
               ) -> dict[str, float]:
    """Univariable logistic odds ratio for one predictor, with a Wald interval."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    y, x = y[ok].astype(int), x[ok]
    blank = {"or": np.nan, "or_lo": np.nan, "or_hi": np.nan, "p": np.nan,
             "n": int(y.size), "asymmetric": False}
    if y.size < 10 or y.sum() == 0 or y.sum() == y.size:
        return blank
    try:
        fit = sm.Logit(y, sm.add_constant(x.reshape(-1, 1),
                                          has_constant="add")).fit(disp=0)
    except Exception:
        return blank
    beta = float(fit.params[1])
    lo, hi = (float(v) for v in fit.conf_int(alpha=alpha)[1])
    # Asymmetry is judged on the log-odds scale, where a well-behaved Wald
    # interval is symmetric by construction. On the odds-ratio scale every
    # interval is lopsided and the check would fire on all of them.
    left, right = beta - lo, hi - beta
    ratio = max(left, right) / min(left, right) if min(left, right) > 0 else np.inf
    return {"or": float(np.exp(beta)), "or_lo": float(np.exp(lo)),
            "or_hi": float(np.exp(hi)), "p": float(fit.pvalues[1]),
            "n": int(y.size), "asymmetric": bool(ratio > ASYMMETRY_LIMIT)}


def dichotomy_for(df: pd.DataFrame, m: Measurement, stratum: str,
                  cutoff: float, *, alpha: float = 0.05) -> dict[str, object]:
    """One measurement: raw number against the same measurement cut in two."""
    sub = df.loc[stratum_mask(df, m, stratum)]
    x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
    y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)

    binary = flag(x, cutoff, m.direction).astype(float)
    continuous = auc_with_ci(y, x, m.direction, alpha=alpha)
    dichotomised = auc_with_ci(y, binary, "higher", alpha=alpha)
    comparison = delong_compare(y, oriented_score(x, m.direction), binary,
                                alpha=alpha)

    lift_c = continuous["auc"] - 0.5
    retained = ((dichotomised["auc"] - 0.5) / lift_c
                if lift_c > 0 else np.nan)
    or_continuous = odds_ratio(y, standardise(x, m.log_x), alpha=alpha)
    or_binary = odds_ratio(y, binary, alpha=alpha)

    return {
        "measurement": m.stratum_label(stratum), "col": m.col,
        "stratum": stratum, "cutoff": m.round(cutoff), "n": int(y.size),
        "auc_continuous": continuous["auc"],
        "auc_continuous_lo": continuous["auc_lo"],
        "auc_continuous_hi": continuous["auc_hi"],
        "auc_binary": dichotomised["auc"],
        "auc_binary_lo": dichotomised["auc_lo"],
        "auc_binary_hi": dichotomised["auc_hi"],
        "auc_loss": comparison["difference"],
        "auc_loss_lo": comparison["difference_lo"],
        "auc_loss_hi": comparison["difference_hi"],
        "auc_loss_p": comparison["p"],
        "information_retained": retained,
        "or_per_sd": or_continuous["or"],
        "or_per_sd_lo": or_continuous["or_lo"],
        "or_per_sd_hi": or_continuous["or_hi"],
        "or_per_sd_p": or_continuous["p"],
        "or_per_sd_asymmetric": or_continuous["asymmetric"],
        "or_binary": or_binary["or"],
        "or_binary_lo": or_binary["or_lo"],
        "or_binary_hi": or_binary["or_hi"],
        "or_binary_p": or_binary["p"],
        "log_transformed": m.log_x,
    }


def presence_versus_amount(df: pd.DataFrame, m: Measurement, *,
                           alpha: float = 0.05) -> dict[str, object]:
    """Split a zero-inflated measurement into the two claims it conflates.

    A third of this cohort has no edema at all. Asked of everyone, "does edema
    predict high grade?" bundles two questions that can have different answers:

    *presence* — does having any at all predict high grade? A yes/no predictor,
    scored as an odds ratio against the patients who have none.
    *amount* — among patients who do have it, does having more predict high
    grade? The same odds ratio per 1 SD, on the present-only stratum.

    Reported together because the pair is the finding. A large presence effect
    with a null amount effect means the measurement is a *sign*, not a *dose*,
    and every cut-point quoted in cm³ is really a test for the sign.
    """
    x = pd.to_numeric(df[m.col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok].astype(int)

    present = x > 0
    n_absent, n_present = int((~present).sum()), int(present.sum())
    events_absent = int(y[~present].sum()) if n_absent else 0
    events_present = int(y[present].sum()) if n_present else 0

    presence_or = odds_ratio(y, present.astype(float), alpha=alpha)
    amount_or = (odds_ratio(y[present], standardise(x[present], m.log_x),
                            alpha=alpha) if n_present >= 20
                 else {"or": np.nan, "or_lo": np.nan, "or_hi": np.nan,
                       "p": np.nan, "n": n_present, "asymmetric": False})
    overall_or = odds_ratio(y, standardise(x, m.log_x), alpha=alpha)

    def _excludes_one(entry) -> bool | None:
        lo, hi = entry["or_lo"], entry["or_hi"]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return bool(lo > 1.0 or hi < 1.0)

    return {
        "measurement": m.label, "col": m.col,
        "n": int(y.size), "n_absent": n_absent, "n_present": n_present,
        "pct_zero": 100.0 * n_absent / y.size if y.size else np.nan,
        "rate_absent": events_absent / n_absent if n_absent else np.nan,
        "rate_present": events_present / n_present if n_present else np.nan,
        "events_absent": events_absent, "events_present": events_present,
        "presence_or": presence_or["or"], "presence_lo": presence_or["or_lo"],
        "presence_hi": presence_or["or_hi"], "presence_p": presence_or["p"],
        "amount_or": amount_or["or"], "amount_lo": amount_or["or_lo"],
        "amount_hi": amount_or["or_hi"], "amount_p": amount_or["p"],
        "overall_or": overall_or["or"], "overall_lo": overall_or["or_lo"],
        "overall_hi": overall_or["or_hi"], "overall_p": overall_or["p"],
        "presence_matters": _excludes_one(presence_or),
        "amount_matters": _excludes_one(amount_or),
    }


def presence_amount_table(df: pd.DataFrame,
                          measurements: Sequence[Measurement] = MEASUREMENTS,
                          *, alpha: float = 0.05) -> pd.DataFrame:
    """`presence_versus_amount` for every measurement declared zero-inflated."""
    rows = [presence_versus_amount(df, m, alpha=alpha)
            for m in measurements if m.zero_inflated]
    return pd.DataFrame(rows)


def dichotomy_table(df: pd.DataFrame, eligible: pd.DataFrame,
                    cutpoints: dict[str, float], *,
                    alpha: float = 0.05) -> pd.DataFrame:
    """The price of dichotomising, for every measurement that carried forward."""
    rows = []
    for _, row in eligible.iterrows():
        if row["col"] not in cutpoints:
            continue
        m = MEASUREMENTS_BY_COL[row["col"]]
        entry = dichotomy_for(df, m, row["stratum"], cutpoints[row["col"]],
                              alpha=alpha)
        entry["claim"] = row.get("claim", "")
        rows.append(entry)
    table = pd.DataFrame(rows)
    return (table.sort_values("information_retained", ascending=False)
            .reset_index(drop=True) if not table.empty else table)


def describe_dichotomy(table: pd.DataFrame) -> str:
    """One line: how much the cut-points cost, and whether any loss is real."""
    if table.empty:
        return "No measurement could be compared against its cut-point."
    worst = table.sort_values("information_retained").iloc[0]
    significant = table[table["auc_loss_p"] < 0.05]
    line = (f"Dichotomising costs most in {worst['measurement']}: AUC "
            f"{worst['auc_continuous']:.2f} as a number falls to "
            f"{worst['auc_binary']:.2f} as a yes/no answer, keeping "
            f"{worst['information_retained']:.0%} of its discriminating power.")
    if significant.empty:
        return line + (" No loss reaches significance — with this sample size "
                       "the cut-points are not measurably worse than the "
                       "numbers they replace.")
    names = ", ".join(significant["measurement"])
    return line + (f" The loss is statistically significant for: {names}.")


def asymmetry_notes(table: pd.DataFrame) -> list[str]:
    """Any odds ratio whose interval says Wald has stopped being adequate."""
    if table.empty or "or_per_sd_asymmetric" not in table:
        return []
    return [
        f"{r['measurement']}: the odds ratio interval is lopsided on the "
        "log-odds scale, so the Wald approximation is straining — refit with a "
        "penalised (Firth) likelihood before quoting it."
        for _, r in table.iterrows() if r["or_per_sd_asymmetric"]]
