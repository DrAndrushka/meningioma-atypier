"""Calibration and net benefit — is the probability right, and is it worth acting on?

Everything else in this phase reports **discrimination**: AUC, Youden J,
sensitivity, specificity. All of those are invariant to a monotone squashing of
the predicted probability. A model that says 90% when the truth is 40% has
exactly the same AUC as one that says 40%, and the report presents its outputs
as probabilities — "risk reaches 30% at …", "risk runs 11% → 66% across the
count score". Discrimination cannot check any of that.

Two things are added here:

**Calibration.** A logistic regression of the outcome on the model's own
log-odds. The *slope* says whether the predictions are too extreme (slope < 1
— the usual direction for a model fitted and scored on the same patients) and
the *intercept* whether they are systematically too high or low. Both are
bootstrap-corrected the way the AUCs are: refit on each resample, score on the
original cohort, subtract the average gap.

**Net benefit.** The question the rest of the report keeps circling — "what
does the line cost" — has a direct answer. Fix a threshold probability *t*:
the value above which you would act. Then

    net benefit = TP/n − (FP/n) × t/(1 − t)

is the true positives gained, minus the false positives, with a false positive
weighted by the odds of the threshold you just declared. It is in units of
"true positives per patient", and it can be compared across *any* strategies —
a cut-point, a count score, a model, treating everyone, treating no one — on
one axis. A strategy whose curve sits below treat-all and treat-none over the
plausible range of *t* is not worth using, however good its AUC.

The reference strategies matter: **treat-all** is what you do without any test,
**treat-none** is what you do if you never act. Beating those two is the
minimum bar, and it is a bar plenty of published cut-points do not clear.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import plot_style as ps
from thresholds import Metric

# The range of threshold probabilities a clinician might plausibly declare for
# "act as though this is high grade". Anchored on the cohort's own prevalence:
# below the base rate the sensible action is to treat everyone.
DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)

TREAT_ALL = "Treat all"
TREAT_NONE = "Treat none"

_EPS = 1e-9


def _clip_probability(p: np.ndarray) -> np.ndarray:
    """Keep logit finite — a predicted 0 or 1 would otherwise blow up the slope."""
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probability(p)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """``(slope, intercept)`` of the outcome regressed on the predicted log-odds.

    Two separate fits, because they answer different questions:

    * **slope** — ``y ~ a + b·logit(p)``. ``b`` is the slope. Below 1 means the
      predictions are too extreme: high risks too high, low risks too low.
    * **intercept** — ``y ~ a + offset(logit(p))``, the slope held at 1. ``a``
      is calibration-in-the-large: above 0 means the model under-predicts risk
      across the board.

    Fitting them together and quoting the intercept from the slope model is a
    common error — that intercept is only interpretable at ``logit(p) = 0``.
    """
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    if y.size == 0 or len(np.unique(y)) < 2 or not np.all(np.isfinite(lp)):
        return np.nan, np.nan

    slope = np.nan
    try:
        fit = sm.GLM(y, sm.add_constant(lp, has_constant="add"),
                     family=sm.families.Binomial()).fit()
        slope = float(fit.params[1])
    except Exception:  # separation on a resample — the caller drops it
        pass

    intercept = np.nan
    try:
        fit0 = sm.GLM(y, np.ones((y.size, 1)), family=sm.families.Binomial(),
                      offset=lp).fit()
        intercept = float(fit0.params[0])
    except Exception:
        pass
    return slope, intercept


def calibration_bins(
    y: np.ndarray, p: np.ndarray, *, n_bins: int = 10, min_per_bin: int = 10,
) -> pd.DataFrame:
    """Observed rate against mean predicted risk in equal-count bins, with Wilson CIs.

    The dots on the calibration plot. Equal-count rather than equal-width so
    every point rests on a similar number of patients.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if y.size < 2 * min_per_bin:
        return pd.DataFrame(columns=["predicted", "observed", "lo", "hi", "n", "events"])
    n_bins = min(int(n_bins), max(2, y.size // min_per_bin))
    try:
        bins = pd.qcut(p, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["predicted", "observed", "lo", "hi", "n", "events"])

    frame = pd.DataFrame({"p": p, "y": y, "bin": bins})
    rows = []
    for _, grp in frame.groupby("bin", observed=True):
        n, k = int(len(grp)), int(grp["y"].sum())
        if n < min_per_bin:
            continue
        lo, hi = ps.wilson_ci(k, n)
        rows.append({"predicted": float(grp["p"].mean()), "observed": k / n,
                     "lo": float(lo), "hi": float(hi), "n": n, "events": k})
    return pd.DataFrame(rows)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability — discrimination and calibration together."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if y.size else np.nan


# --------------------------------------------------------------------------
# The uncut four-measurement model
# --------------------------------------------------------------------------
def uncut_design(
    df: pd.DataFrame, metrics: Sequence[Metric], target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(X, y, columns)`` for the continuous model, matching the §6 benchmark.

    Deliberately the same transform as ``combinations.continuous_model_benchmark``
    — ``log1p`` for the metrics marked ``log_x`` — so the AUC quoted there and
    the calibration quoted here belong to the same model.
    """
    cols = [m.col for m in metrics if m.col in df.columns]
    frame = df[cols + [target]].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if next(m for m in metrics if m.col == col).log_x:
            frame[col] = np.log1p(frame[col])
    frame[target] = frame[target].astype("boolean")
    frame = frame.dropna()

    y = frame[target].astype(int).to_numpy()
    X = sm.add_constant(frame[cols].astype(float).to_numpy(), has_constant="add")
    return X, y, cols


def uncut_model_predictions(
    df: pd.DataFrame, metrics: Sequence[Metric], target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(y, predicted risk)`` from the uncut model fitted on these patients."""
    X, y, _ = uncut_design(df, metrics, target)
    if y.size == 0 or len(np.unique(y)) < 2:
        return np.array([]), np.array([])
    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception:
        return np.array([]), np.array([])
    return y, np.asarray(fit.predict(X), dtype=float)


def uncut_model_calibration(
    df: pd.DataFrame,
    metrics: Sequence[Metric],
    target: str,
    *,
    n_boot: int = 500,
    seed: int = 20260801,
) -> dict:
    """Apparent and bootstrap-corrected calibration slope and intercept.

    Harrell's optimism, applied to calibration rather than to discrimination:
    refit on each resample, measure the slope of that model's predictions *on
    the original cohort*, and subtract the average gap. The apparent slope of a
    model scored on its own data is 1 by construction, so the corrected slope
    is essentially the average slope achieved on data the model did not see —
    which is what would happen to it at another hospital.
    """
    X, y, cols = uncut_design(df, metrics, target)
    blank = {
        "model": "Uncut four-measurement model", "n_used": int(y.size),
        "events": int(y.sum()) if y.size else 0, "n_predictors": len(cols),
        "slope_apparent": np.nan, "slope_corrected": np.nan,
        "intercept_apparent": np.nan, "intercept_corrected": np.nan,
        "brier_apparent": np.nan, "brier_corrected": np.nan,
        "n_bootstrap": 0, "source": "threshold phase (fitted here)",
    }
    if y.size < 30 or len(np.unique(y)) < 2:
        return blank
    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    except Exception:
        return blank

    p_apparent = np.asarray(fit.predict(X), dtype=float)
    slope_app, intercept_app = calibration_slope_intercept(y, p_apparent)
    brier_app = brier_score(y, p_apparent)

    rng = np.random.default_rng(seed)
    n = y.size
    gaps_slope: list[float] = []
    gaps_intercept: list[float] = []
    gaps_brier: list[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        yb, Xb = y[take], X[take]
        if len(np.unique(yb)) < 2:
            continue
        try:
            fb = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit()
        except Exception:
            continue
        pb, po = (np.asarray(fb.predict(Xb), dtype=float),
                  np.asarray(fb.predict(X), dtype=float))
        s_b, i_b = calibration_slope_intercept(yb, pb)
        s_o, i_o = calibration_slope_intercept(y, po)
        if np.isfinite(s_b) and np.isfinite(s_o):
            gaps_slope.append(s_b - s_o)
        if np.isfinite(i_b) and np.isfinite(i_o):
            gaps_intercept.append(i_b - i_o)
        gaps_brier.append(brier_score(yb, pb) - brier_score(y, po))

    blank.update({
        "slope_apparent": slope_app,
        "slope_corrected": (slope_app - float(np.mean(gaps_slope))
                            if gaps_slope else np.nan),
        "intercept_apparent": intercept_app,
        "intercept_corrected": (intercept_app - float(np.mean(gaps_intercept))
                                if gaps_intercept else np.nan),
        "brier_apparent": brier_app,
        "brier_corrected": (brier_app - float(np.mean(gaps_brier))
                            if gaps_brier else np.nan),
        "n_bootstrap": len(gaps_slope),
    })
    return blank


MODEL_ARTIFACT_GLOB = "*_model.json"


def load_model_artifacts(output_root: Path) -> list[dict]:
    """The modelling phase's own model JSONs, or an empty list if it has not run."""
    folder = Path(output_root) / "inferential" / "model_artifacts"
    if not folder.exists():
        return []
    payloads = []
    for path in sorted(folder.glob(MODEL_ARTIFACT_GLOB)):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Every artifact carries the same product name in `model_name` ("High-grade
        # meningioma risk calculator"), so the file stem is the only thing that
        # tells the model variants apart.
        payload["_stem"] = path.stem.replace("high_grade_", "").replace("_model", "")
        payloads.append(payload)
    return payloads


def _corrected_auc(payload: Mapping) -> float:
    for m in payload.get("validation", {}).get("metrics", []):
        if m.get("metric") == "AUC":
            try:
                return float(m.get("optimism_corrected"))
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def best_model_calibration_bins(
    output_root: Path,
) -> tuple[pd.DataFrame, dict] | None:
    """Calibration bins of the best-discriminating multivariable model, if any.

    Read straight from the artifact so the panel next to the uncut model shows
    the modelling phase's own validated calibration, not a re-derivation.
    """
    payloads = [p for p in load_model_artifacts(output_root)
                if np.isfinite(_corrected_auc(p))]
    if not payloads:
        return None
    payload = max(payloads, key=_corrected_auc)
    cal = payload.get("validation", {}).get("calibration", {}) or {}
    bins = cal.get("bins") or []
    if not bins:
        return None

    rows = []
    for b in bins:
        n, k = int(b.get("n", 0)), int(b.get("events", 0))
        lo, hi = ps.wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append({"predicted": b.get("predicted"), "observed": b.get("observed"),
                     "lo": float(lo), "hi": float(hi), "n": n, "events": k})
    stats = multivariable_calibration([payload]).iloc[0].to_dict()
    return pd.DataFrame(rows), stats


def multivariable_calibration(models: Sequence[Mapping]) -> pd.DataFrame:
    """Calibration of the full multivariable models, read from their own artifacts.

    Not refitted here. Those models are Rubin-pooled across the MICE draws in
    the modelling phase, and re-deriving their calibration from the coefficients
    alone would produce an apparent number dressed as a validated one. What the
    modelling phase itself validated is what gets quoted.

    ⚠️ Those artifacts carry a bootstrap-corrected *slope* but only an apparent
    *intercept*; the gap is reported as such rather than filled in.
    """
    rows: list[dict] = []
    for payload in models:
        validation = payload.get("validation", {}) or {}
        cal = validation.get("calibration", {}) or {}
        metrics = {m.get("metric"): m for m in validation.get("metrics", [])}
        slope = metrics.get("Calibration slope", {})
        brier = metrics.get("Brier score", {})
        coefficients = payload.get("coefficients", {}) or {}
        rows.append({
            "model": str(payload.get("_stem")
                         or payload.get("model_name", "")).replace("_", " "),
            "n_used": payload.get("n"),
            "events": payload.get("events"),
            "n_predictors": max(len(coefficients) - 1, 0),
            "slope_apparent": cal.get("slope_apparent", slope.get("apparent")),
            "slope_corrected": cal.get("slope_corrected",
                                       slope.get("optimism_corrected")),
            "intercept_apparent": cal.get("intercept_apparent"),
            "intercept_corrected": cal.get("intercept_corrected", np.nan),
            "brier_apparent": brier.get("apparent"),
            "brier_corrected": brier.get("optimism_corrected"),
            "n_bootstrap": validation.get("successful_bootstraps",
                                          validation.get("bootstrap_resamples")),
            "source": "modelling phase artifact",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Net benefit
# --------------------------------------------------------------------------
def net_benefit(y: np.ndarray, flagged: np.ndarray, threshold: float) -> float:
    """``TP/n − (FP/n)·t/(1−t)`` — true positives per patient, false ones discounted.

    The weight ``t/(1−t)`` is the exchange rate the clinician sets by declaring
    the threshold: at t = 0.3 you are saying one missed high-grade tumour is
    worth about 2.3 unnecessary alarms, so each false positive costs 1/2.3 of a
    true one.
    """
    y = np.asarray(y, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    n = y.size
    if n == 0 or not (0.0 < threshold < 1.0):
        return np.nan
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def decision_curve(
    y: np.ndarray,
    strategies: Mapping[str, np.ndarray],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Net benefit of every strategy across the threshold range, plus the two references.

    A strategy is either a boolean array (a fixed rule — flagged or not, the
    same at every threshold) or a float array of predicted probabilities, in
    which case it is thresholded at each *t* the way it would be used.

    ``Treat all`` and ``Treat none`` are added automatically: they are the
    comparison that decides whether a strategy is worth having at all.
    """
    y = np.asarray(y, dtype=int)
    rows: list[dict] = []
    for t in thresholds:
        t = float(t)
        rows.append({"strategy": TREAT_ALL, "threshold": t,
                     "net_benefit": net_benefit(y, np.ones_like(y, dtype=bool), t),
                     "kind": "reference"})
        rows.append({"strategy": TREAT_NONE, "threshold": t,
                     "net_benefit": 0.0, "kind": "reference"})
        for name, values in strategies.items():
            values = np.asarray(values)
            if values.dtype == bool:
                flagged, kind = values, "rule"
            else:
                flagged, kind = (values.astype(float) >= t), "model"
            rows.append({"strategy": name, "threshold": t,
                         "net_benefit": net_benefit(y, flagged, t), "kind": kind})
    return pd.DataFrame(rows)


def decision_curve_summary(
    curve: pd.DataFrame, *, prevalence: float | None = None,
) -> pd.DataFrame:
    """Per strategy: where it is the best available, and where it is worthless.

    The two sentences a decision curve is read for — "over what range of
    thresholds would I actually use this" and "is it ever better than doing
    nothing or treating everyone".
    """
    if curve.empty:
        return pd.DataFrame()
    wide = curve.pivot_table(index="threshold", columns="strategy",
                             values="net_benefit")
    best = wide.idxmax(axis=1)
    reference = wide[[c for c in (TREAT_ALL, TREAT_NONE) if c in wide.columns]].max(axis=1)

    rows = []
    for name in wide.columns:
        col = wide[name]
        beats = col > reference + 1e-12
        wins = best == name
        rows.append({
            "strategy": name,
            "is_reference": name in (TREAT_ALL, TREAT_NONE),
            "max_net_benefit": float(col.max()),
            "threshold_at_max": float(col.idxmax()),
            "beats_references_from": float(col.index[beats].min()) if beats.any() else np.nan,
            "beats_references_to": float(col.index[beats].max()) if beats.any() else np.nan,
            "pct_of_range_beating_references": 100.0 * float(beats.mean()),
            "pct_of_range_best_available": 100.0 * float(wins.mean()),
            "prevalence": prevalence if prevalence is not None else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["is_reference", "pct_of_range_best_available"], ascending=[True, False],
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def calibration_figure(
    panels: Sequence[tuple[str, pd.DataFrame, dict]],
) -> plt.Figure:
    """One panel per model: binned observed vs predicted, with the diagonal.

    ``panels`` is ``(title, bins_frame, stats_dict)``. Perfect calibration is
    the diagonal; points below it mean the model promises more risk than the
    patients delivered.
    """
    n = max(len(panels), 1)
    fig, axes = plt.subplots(
        1, n, squeeze=False,
        figsize=ps.figure_size(ps.FIG_WIDTH_DOUBLE, aspect=0.46 / max(n / 2, 1)),
    )
    for ax, (title, bins, stats) in zip(axes.ravel(), panels):
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0,
                color=ps.PALETTE["neutral"], zorder=1, label="Perfect calibration")
        if len(bins):
            ax.errorbar(
                bins["predicted"], bins["observed"],
                yerr=ps.errorbar_lengths(bins["observed"], bins["lo"], bins["hi"]),
                fmt="o", markersize=3.6, color=ps.PALETTE["primary"],
                ecolor=ps.PALETTE["primary"], elinewidth=0.8, capsize=1.8,
                linestyle="none", zorder=4, label="Observed (equal-count bins)")
            top = float(max(bins["predicted"].max(), bins["observed"].max())) * 1.15
        else:
            top = 1.0
        limit = min(1.0, max(0.2, top))
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed high-grade rate")

        slope = stats.get("slope_corrected", np.nan)
        intercept = stats.get("intercept_corrected", stats.get("intercept_apparent"))
        subtitle_bits = []
        if np.isfinite(slope):
            subtitle_bits.append(f"slope {slope:.2f}")
        if intercept is not None and np.isfinite(float(intercept)):
            subtitle_bits.append(f"intercept {float(intercept):+.2f}")
        ps.set_titles(ax, title, " · ".join(subtitle_bits) or None)
        ps.place_legend(ax, loc="upper left", scale=0.62)

    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.suptitle("Calibration — is the probability the right size?",
                 fontsize=plt.rcParams["font.size"] * 1.05)
    return fig


def decision_curve_figure(
    curve: pd.DataFrame,
    *,
    prevalence: float | None = None,
    y_floor: float | None = None,
) -> plt.Figure:
    """Net benefit against threshold probability, all strategies on one axis."""
    fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.62))

    names = [n for n in curve["strategy"].unique() if n not in (TREAT_ALL, TREAT_NONE)]
    colours = dict(zip(names, ps.categorical_palette(max(len(names), 1))))

    for name in (TREAT_ALL, TREAT_NONE):
        sub = curve[curve["strategy"] == name].sort_values("threshold")
        if sub.empty:
            continue
        ax.plot(sub["threshold"], sub["net_benefit"],
                color=ps.PALETTE["neutral"], linewidth=1.0,
                linestyle="--" if name == TREAT_ALL else ":", zorder=2, label=name)

    for name in names:
        sub = curve[curve["strategy"] == name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["net_benefit"], color=colours[name],
                linewidth=1.5, zorder=3, label=ps._wrap_label(str(name), 34))

    if prevalence is not None and np.isfinite(prevalence):
        ax.axvline(prevalence, color=ps.PALETTE["neutral"], linewidth=0.8,
                   linestyle="-.", zorder=1,
                   label=f"Cohort rate ({prevalence * 100:.0f}%)")

    finite = curve["net_benefit"].replace([np.inf, -np.inf], np.nan).dropna()
    floor = y_floor if y_floor is not None else -0.05
    ax.set_ylim(floor, float(finite.max()) * 1.1 if len(finite) else 0.3)
    ax.set_xlabel("Threshold probability — the risk above which you would act")
    ax.set_ylabel("Net benefit (true positives per patient)")
    ps.place_legend(ax, outside=True, scale=0.6)
    ps.set_titles(ax, "Decision curve — what is each strategy worth?",
                  "Higher is better; below both dashed lines is worse than no test")
    return fig
