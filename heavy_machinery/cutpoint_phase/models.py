"""Step 11 — five cut-offs together against five numbers together.

Every earlier step judged one measurement at a time. A radiologist looking at a
scan sees all of them at once, so the question that decides how this work is
used is whether the measurements *combined* do better as numbers or as yes/no
answers — and whether five of them beat three.

Three predictor sets, on the same patients:

``numbers``         all five as raw values, standardised
``cutpoints``       all five as yes/no flags at the published cut-points
``representatives`` one per dimension — step 10b showed the five measurements
                    are really three things

**The same patients, always.** All three models are fitted on the patients who
have every one of the five measurements. Letting the three-predictor model use
the extra patients that its two dropped variables were missing in would make it
win on sample size rather than on merit.

**VIF, alongside every odds ratio.** ``VIF`` is how much wider a predictor's
interval became because of the other predictors — ``sqrt(VIF)`` is the factor
directly. It answers what a correlation matrix cannot: three predictors can each
look innocent in pairs while one is nearly reconstructable from the other two.
Above 5 is worth saying; above 10 the coefficient is not interpretable.

**Optimism-corrected AUC, or the comparison is rigged.** A model with five
predictors fits the patients it was trained on better than one with three,
whether or not the extra predictors carry anything. Each model is therefore
refitted in a thousand resampled cohorts and scored on the original patients;
the average drop is subtracted. Without that, the biggest model always wins.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import auc as _sk_auc, roc_curve as _sk_roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

from accuracy import flag
from dichotomy import standardise
from measurements import MEASUREMENTS, MEASUREMENTS_BY_COL
from wobble import N_BOOTSTRAP, SEED

OUTCOME = "high_grade"

FORM_NUMBERS = "numbers"
FORM_CUTPOINTS = "cut-points"

# Concerning above 5, uninterpretable above 10 — the thresholds the modelling
# phase already prunes on, kept identical so the two phases mean the same thing.
VIF_NOTABLE = 5.0
VIF_SEVERE = 10.0


class PredictorSet(NamedTuple):
    name: str
    columns: tuple[str, ...]
    form: str
    why: str


ALL_FIVE = tuple(m.col for m in MEASUREMENTS)

# Max diameter rather than tumour volume: step 10 found them statistically
# indistinguishable and diameter is measured in every patient. Edema volume
# rather than the index: step 10 found the volume genuinely better.
REPRESENTATIVES = ("max_diameter_cm", "edema_volume_cm3", "adc_value")

PREDICTOR_SETS: tuple[PredictorSet, ...] = (
    PredictorSet("Five numbers", ALL_FIVE, FORM_NUMBERS,
                 "every measurement on its own scale"),
    PredictorSet("Five cut-points", ALL_FIVE, FORM_CUTPOINTS,
                 "every measurement as a yes/no rule"),
    PredictorSet("Three numbers", REPRESENTATIVES, FORM_NUMBERS,
                 "one per dimension — size, edema, diffusion"),
    PredictorSet("Three cut-points", REPRESENTATIVES, FORM_CUTPOINTS,
                 "one per dimension, as yes/no rules"),
)


def common_patients(df: pd.DataFrame,
                    columns: Sequence[str] = ALL_FIVE) -> pd.Series:
    """Patients with every measurement, so all models share one denominator."""
    present = [c for c in columns if c in df.columns]
    return df[present].notna().all(axis=1) & df[OUTCOME].notna()


def design_matrix(df: pd.DataFrame, columns: Sequence[str], form: str,
                  cutpoints: dict[str, float] | None = None) -> pd.DataFrame:
    """Predictors in the requested form, one column each.

    Numbers are standardised so their odds ratios are per 1 SD and therefore
    comparable with each other. Flags are left as 0/1, where the odds ratio is
    already the quantity of interest.
    """
    out = {}
    for col in columns:
        m = MEASUREMENTS_BY_COL[col]
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if form == FORM_CUTPOINTS:
            if not cutpoints or col not in cutpoints:
                raise KeyError(f"No cut-point declared for {col!r}.")
            out[m.rule_text(cutpoints[col])] = flag(
                values, cutpoints[col], m.direction).astype(float)
        else:
            out[f"{m.label} (per 1 SD)"] = standardise(values, m.log_x)
    return pd.DataFrame(out, index=df.index)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation for each predictor, given the others in the model.

    A single predictor has nothing to be inflated by, so its VIF is 1 by
    definition rather than by computation — ``variance_inflation_factor`` on a
    one-column design divides by a zero residual variance.
    """
    cols = list(X.columns)
    if len(cols) < 2:
        return pd.DataFrame({"predictor": cols, "vif": [1.0] * len(cols)})
    design = sm.add_constant(X.dropna(), has_constant="add")
    # A writable copy: pandas can hand out a read-only view, and the regression
    # inside variance_inflation_factor writes into the array it is given.
    values = np.array(design.to_numpy(dtype=float), copy=True)
    rows = []
    for i, name in enumerate(design.columns):
        if name == "const":
            continue
        try:
            value = float(variance_inflation_factor(values, i))
        except Exception:
            value = np.nan
        rows.append({"predictor": name, "vif": value})
    return pd.DataFrame(rows)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    # roc_auc_score's binary path is exactly these two calls. Going straight
    # to them skips its per-call validate_params/type_of_target pass, which
    # runs np.unique over both arrays on every one of the thousand resamples,
    # and returns the same float bit for bit.
    fpr, tpr, _ = _sk_roc_curve(y, p)
    return float(_sk_auc(fpr, tpr))


def optimism_corrected_auc(X: pd.DataFrame, y: np.ndarray, *,
                           n_boot: int = N_BOOTSTRAP, seed: int = SEED
                           ) -> dict[str, float]:
    """Harrell's bootstrap optimism correction for the model's own AUC.

    Refit in each resampled cohort, score that refit both on the resample it
    learned from and on the original patients, and average the gap. Subtracting
    it is what makes a five-predictor model comparable with a three-predictor
    one — without it, the larger model wins on flexibility alone.
    """
    X_arr = sm.add_constant(X.to_numpy(dtype=float), has_constant="add")
    y = np.asarray(y, dtype=int)
    try:
        full = sm.Logit(y, X_arr).fit(disp=0)
    except Exception:
        return {"auc_apparent": np.nan, "optimism": np.nan,
                "auc_corrected": np.nan, "n_valid": 0}
    apparent = _auc(y, full.predict(X_arr))

    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            fit_b = sm.Logit(y[idx], X_arr[idx]).fit(disp=0)
        except Exception:
            continue
        in_bag = _auc(y[idx], fit_b.predict(X_arr[idx]))
        on_original = _auc(y, fit_b.predict(X_arr))
        if np.isfinite(in_bag) and np.isfinite(on_original):
            gaps.append(in_bag - on_original)
    optimism = float(np.mean(gaps)) if gaps else np.nan
    return {"auc_apparent": apparent, "optimism": optimism,
            "auc_corrected": apparent - optimism if np.isfinite(optimism)
            else np.nan, "n_valid": len(gaps)}


def fit_set(df: pd.DataFrame, spec: PredictorSet, *,
            cutpoints: dict[str, float] | None = None,
            mask: pd.Series | None = None, n_boot: int = N_BOOTSTRAP,
            seed: int = SEED, alpha: float = 0.05
            ) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one predictor set. Returns the model summary and its coefficient table."""
    rows = df.loc[mask] if mask is not None else df
    X = design_matrix(rows, spec.columns, spec.form, cutpoints)
    y = pd.to_numeric(rows[OUTCOME], errors="coerce")
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[ok], y.loc[ok].astype(int).to_numpy()

    summary = {"model": spec.name, "form": spec.form, "why": spec.why,
               "n_predictors": len(spec.columns), "n": int(len(y)),
               "n_high": int(y.sum())}
    if len(y) < 20 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        summary.update({"auc_apparent": np.nan, "optimism": np.nan,
                        "auc_corrected": np.nan, "max_vif": np.nan,
                        "epv": np.nan})
        return summary, pd.DataFrame()

    try:
        fit = sm.Logit(y, sm.add_constant(X.to_numpy(dtype=float),
                                          has_constant="add")).fit(disp=0)
    except Exception:
        summary.update({"auc_apparent": np.nan, "optimism": np.nan,
                        "auc_corrected": np.nan, "max_vif": np.nan,
                        "epv": np.nan})
        return summary, pd.DataFrame()

    conf = fit.conf_int(alpha=alpha)
    vifs = vif_table(X).set_index("predictor")["vif"]
    coefficients = pd.DataFrame({
        "model": spec.name,
        "predictor": list(X.columns),
        "or": np.exp(fit.params[1:]),
        "or_lo": np.exp(conf[1:, 0]),
        "or_hi": np.exp(conf[1:, 1]),
        "p": fit.pvalues[1:],
        "vif": [vifs.get(c, np.nan) for c in X.columns],
    })
    coefficients["vif_flag"] = np.where(
        coefficients["vif"] >= VIF_SEVERE, "uninterpretable",
        np.where(coefficients["vif"] >= VIF_NOTABLE, "inflated", ""))

    summary.update(optimism_corrected_auc(X, y, n_boot=n_boot, seed=seed))
    summary["max_vif"] = float(coefficients["vif"].max())
    # Events per variable: below about 10 a logistic model starts to overfit
    # regardless of how the predictors were chosen.
    summary["epv"] = float(y.sum() / len(spec.columns))
    return summary, coefficients


def compare_sets(df: pd.DataFrame, *,
                 sets: Sequence[PredictorSet] = PREDICTOR_SETS,
                 cutpoints: dict[str, float] | None = None,
                 n_boot: int = N_BOOTSTRAP, seed: int = SEED
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every predictor set on one shared set of patients."""
    mask = common_patients(df, ALL_FIVE)
    summaries, coefficients = [], []
    for spec in sets:
        summary, coefs = fit_set(df, spec, cutpoints=cutpoints, mask=mask,
                                 n_boot=n_boot, seed=seed)
        summaries.append(summary)
        if not coefs.empty:
            coefficients.append(coefs)
    table = pd.DataFrame(summaries).sort_values(
        "auc_corrected", ascending=False, na_position="last").reset_index(drop=True)
    return table, (pd.concat(coefficients, ignore_index=True)
                   if coefficients else pd.DataFrame())


def describe_models(summary: pd.DataFrame, coefficients: pd.DataFrame) -> str:
    """One line: which set wins once optimism is removed, and what VIF cost."""
    if summary.empty or "auc_corrected" not in summary.columns:
        return "No predictor set could be fitted."
    scored = summary.dropna(subset=["auc_corrected"])
    if scored.empty:
        return "No predictor set could be fitted."
    best = scored.iloc[0]
    line = (f"Best after optimism correction: {best['model']} "
            f"({best['n_predictors']} predictors), AUC "
            f"{best['auc_corrected']:.3f} corrected from "
            f"{best['auc_apparent']:.3f}, n={int(best['n'])}.")
    if len(scored) > 1:
        worst = scored.iloc[-1]
        line += (f" Worst: {worst['model']} at {worst['auc_corrected']:.3f}. "
                 f"Spread {best['auc_corrected'] - worst['auc_corrected']:.3f}.")
    if not coefficients.empty:
        inflated = coefficients[coefficients["vif_flag"] != ""]
        if inflated.empty:
            line += " No predictor's interval is inflated by the others."
        else:
            named = "; ".join(
                f"{r['predictor']} in {r['model']} (VIF {r['vif']:.1f})"
                for _, r in inflated.iterrows())
            line += (f" Intervals widened by collinearity — {named}.")
    return line
