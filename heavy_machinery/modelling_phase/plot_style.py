"""One plotting pipeline for DDA / EDA / forest / missingness / report captions.

Shared concerns only (do not duplicate these in phase modules):

1. SciencePlots session style (``science`` + ``nature`` + ``no-latex``)
2. Okabe–Ito palette, publication figure geometry, title/subtitle blocks
3. Statistical drawing primitives (Wilson CIs, KDE, LOWESS, raincloud)
4. SVG save / bytes export
5. Human-readable labels for axes, category levels, and report captions

Everything here is column-agnostic: label maps are data, not logic, so phase
modules never branch on a study-specific column name.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Pipeline default — Nature typography without requiring a LaTeX install.
SCIENCE_STYLE_DEFAULT: tuple[str, ...] = ("science", "nature", "no-latex")

# Okabe–Ito (colorblind-friendlier than Set2 pastels).
CATEGORICAL_COLORS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

# Semantic roles mapped onto the same family.
PALETTE = {
    "primary": CATEGORICAL_COLORS[0],   # blue
    "accent": CATEGORICAL_COLORS[1],    # vermillion
    "good": CATEGORICAL_COLORS[2],     # green
    "bad": CATEGORICAL_COLORS[1],      # vermillion
    "neutral": "#666666",
    "low_grade": CATEGORICAL_COLORS[0],
    "high_grade": CATEGORICAL_COLORS[1],
    "significant": CATEGORICAL_COLORS[1],
    "nonsignificant": "#7A7A7A",
}

# Tokens that should keep a fixed casing/spelling when prettifying.
_ACRONYMS = {
    "who": "WHO", "mri": "MRI", "ct": "CT", "dwi": "DWI", "adc": "ADC",
    "flair": "FLAIR", "iv": "IV", "t1": "T1", "t2": "T2",
    "ppv": "PPV", "npv": "NPV", "auc": "AUC", "roc": "ROC", "vif": "VIF",
    "epv": "EPV", "fdr": "FDR", "ci": "CI", "or": "OR", "sd": "SD",
    "iqr": "IQR", "cv": "CV", "id": "ID", "fpr": "FPR", "tpr": "TPR",
}

# Comparison suffixes/tokens used by threshold derivations and their levels.
_COMPARATORS = {"ge": "≥", "gt": ">", "le": "≤", "lt": "<", "eq": "="}

# Exact, clinician-friendly names for known columns (overrides token logic).
COLUMN_LABELS = {
    "id": "ID",
    "patient_code": "Patient code",
    "entry_year": "Entry year",
    "age": "Age",
    "age_bins": "Age group",
    "sex": "Sex",
    "histology_available": "Histology available",
    "who_grade": "WHO grade",
    "high_grade": "High-grade",
    "progesterone_pos": "Progesterone positive",
    "ki67_pct": "Ki-67 (%)",
    "ki67_mid": "Ki-67 midpoint (%)",
    "ki67_group": "Ki-67 group",
    "brain_invasion": "Brain invasion",
    "hist_necrosis": "Histological necrosis",
    "mri_date": "MRI date",
    "side": "Side",
    "tumor_location": "Tumor location",
    "meningioma_count": "Meningioma count",
    "multiple_meningiomas": "Multiple meningiomas",
    "max_diameter_cm": "Max diameter (cm)",
    "tumor_volume": "Tumor volume (cm³)",
    "base_modality": "Base modality",
    "iv_contrast": "IV contrast",
    "tumor_episode": "Tumor episode",
    "tumor_margin": "Tumor margin",
    "dural_tail": "Dural tail",
    "capsular_enhancement": "Capsular enhancement",
    "heterogeneous_enhancement": "Heterogeneous enhancement",
    "perifocal_edema": "Perifocal edema",
    "edema_volume_cm3": "Edema volume (cm³)",
    "mass_effect": "Mass effect",
    "calcification": "Calcification",
    "cystic_component": "Cystic component",
    "mri_necrosis": "MRI necrosis",
    "hemorrhage": "Hemorrhage",
    "hyperostosis": "Hyperostosis",
    "cortical_destruction": "Cortical destruction",
    "dwi_hyperintensity": "DWI hyperintensity",
    "t2_hyperintensity": "T2 hyperintensity",
    "t1_hypointensity": "T1 hypointensity",
    "sinus_invasion": "Sinus invasion",
    "transfalcine_extension": "Transfalcine extension",
    "adc_value": "ADC value",
}

# Display names for the two states of a boolean column: {column: (False, True)}.
# Purely a label map — plotting code never branches on the column name itself.
BOOL_LEVEL_LABELS: dict[str, tuple[str, str]] = {
    "high_grade": ("Low grade", "High grade"),
}

# Display names for individual levels of a categorical column: {column: {level: label}}.
LEVEL_LABELS: dict[str, dict] = {}

# Figure-type suffixes in file stems → descriptive phrases.
_FIG_SUFFIX = {
    "hist": "distribution",
    "distribution": "distribution",
    "box": "box plot",
    "bar": "distribution",
    "timeline": "records over time",
    "forest": "forest plot",
    "vif": "collinearity (VIF)",
    "multivariable": "multivariable model",
    "roc": "ROC curve",
    "calibration": "calibration",
    "decision_curve": "decision curve",
    "model_comparison": "model comparison",
}

# Whole-stem overrides for standalone figures.
_STEM_OVERRIDES = {
    "missing_per_column": "Missing values per column",
    "co_missingness_heatmap": "Co-missingness overlap (Jaccard)",
    "association_heatmap": "Target × predictor association heatmap",
    "chain_diagnostics": "MICE chain diagnostics",
}


def prettify_label(name: str) -> str:
    """Turn a single machine name (``who_grade``) into a clean label.

    Uses the explicit ``COLUMN_LABELS`` map first, then falls back to
    sentence-case with acronym fixes (``adc_value`` → ``ADC value``).
    """
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    if raw in COLUMN_LABELS:
        return COLUMN_LABELS[raw]

    # Threshold-flag derivations ("edema_volume_ge3.64") read as the comparison
    # they encode rather than as a machine suffix.
    threshold = re.fullmatch(
        r"(?P<base>.+?)_(?P<op>ge|gt|le|lt|eq)_?(?P<num>\d+(?:\.\d+)?)", raw,
    )
    if threshold:
        # Three significant figures, the same as the threshold phase prints on
        # its own axes and in its own prose. Formatting the number here rather
        # than echoing the column name is what stops the same cut-point from
        # reading differently in two sections of the same manuscript.
        return (
            f"{prettify_label(threshold.group('base'))} "
            f"{_COMPARATORS[threshold.group('op')]} "
            f"{float(threshold.group('num')):.3g}"
        )

    # Binned derivations ("age_bins_10") are groups of their base variable; the
    # bin edges are already spelled out on the category axis.
    binned = re.fullmatch(r"(?P<base>.+?)_bins?(?:_\d+)?", raw)
    if binned:
        return f"{prettify_label(binned.group('base'))} group"

    tokens = raw.replace("-", "_").split("_")
    out: list[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _ACRONYMS:
            out.append(_ACRONYMS[low])
        elif low == "pct":
            out.append("(%)")
        elif tok.isdigit():
            out.append(tok)
        elif tok.isupper() or (len(tok) <= 2 and tok.isalpha()):
            out.append(tok.upper())  # keep category codes like M / L / R
        elif i == 0:
            out.append(tok[:1].upper() + tok[1:].lower())
        else:
            out.append(tok.lower())
    return " ".join(t for t in out if t)


def prettify_level(value, column: str | None = None) -> str:
    """Human-readable label for one *level* of a categorical column.

    Resolution order: explicit ``LEVEL_LABELS[column][value]`` →
    ``BOOL_LEVEL_LABELS[column]`` for booleans → generic ``Yes``/``No`` →
    sentence-cased tokens with acronym fixes (``non_skull_base`` → ``Non skull
    base``). Values that already carry their own formatting (``≥ 3.6``,
    ``60–69``) pass through untouched.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""

    col = str(column) if column else ""
    overrides = LEVEL_LABELS.get(col) if col else None
    if overrides:
        try:
            if value in overrides:
                return str(overrides[value])
        except TypeError:  # unhashable level
            pass

    bool_pair = BOOL_LEVEL_LABELS.get(col) if col else None
    is_bool = isinstance(value, (bool, np.bool_))
    if bool_pair is not None and (is_bool or value in (0, 1)):
        return bool_pair[1] if bool(value) else bool_pair[0]
    if is_bool:
        return "Yes" if bool(value) else "No"

    text = str(value).strip()
    if not text:
        return ""
    # Only reshape machine-style tokens; leave hand-written labels alone.
    # Hyphens are never treated as separators — bin labels ("60-69") and
    # threshold labels must survive verbatim.
    if not re.fullmatch(r"[A-Za-z0-9.\-]+([_\s][A-Za-z0-9.\-]+)*", text):
        return text
    if text.isupper() and len(text) <= 4:
        return text  # category codes (L / R / NOS)

    tokens = re.split(r"[_\s]+", text)
    out: list[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _ACRONYMS:
            out.append(_ACRONYMS[low])
        elif low in _COMPARATORS:
            out.append(_COMPARATORS[low])
        elif tok.isdigit() or not tok.isalpha():
            out.append(tok)
        elif i == 0:
            out.append(tok[:1].upper() + tok[1:].lower())
        else:
            out.append(tok.lower())
    return " ".join(_join_numeric_ranges(out))


def _join_numeric_ranges(tokens: list[str]) -> list[str]:
    """``["Intermediate", "5", "9"]`` → ``["Intermediate", "5–9"]``.

    Level names generated from bin edges arrive as separate numeric tokens;
    joining them keeps the range readable as a range.
    """
    merged: list[str] = []
    for tok in tokens:
        if (
            merged
            and _is_number(tok)
            and _is_number(merged[-1].split("–")[-1])
            and "–" not in merged[-1]
        ):
            merged[-1] = f"{merged[-1]}–{tok}"
        else:
            merged.append(tok)
    return [t for t in merged if t]


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def level_tick_labels(
    levels: Sequence,
    counts: Sequence[int] | None = None,
    *,
    column: str | None = None,
    wrap: int = 14,
) -> list[str]:
    """Tick labels for category levels, optionally with the denominator below.

    Denominators belong on the axis, not in a caption: a 50% bar over n=4 and
    one over n=200 must not look alike.
    """
    labels = [prettify_level(lv, column) for lv in levels]
    labels = [_wrap_label(lab, wrap) for lab in labels]
    if counts is None:
        return labels
    return [f"{lab}\n(n={int(c)})" for lab, c in zip(labels, counts)]


def _wrap_label(label: str, width: int) -> str:
    """Soft-wrap a tick label on spaces so long level names stay legible."""
    if len(label) <= width or " " not in label:
        return label
    words, lines, current = label.split(" "), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def prettify_caption(stem: str) -> str:
    """Turn a figure file stem into a descriptive caption (no ``__``)."""
    if stem is None:
        return ""
    s = str(stem)
    if s in _STEM_OVERRIDES:
        return _STEM_OVERRIDES[s]

    segments = [seg for seg in s.split("__") if seg]
    if not segments:
        return prettify_label(s)

    parts: list[str] = []
    for idx, seg in enumerate(segments):
        is_last = idx == len(segments) - 1
        if is_last and seg.lower() in _FIG_SUFFIX:
            phrase = _FIG_SUFFIX[seg.lower()]
            parts.append(phrase[:1].upper() + phrase[1:])  # keep inner acronyms
        else:
            parts.append(prettify_label(seg))
    return " — ".join(p for p in parts if p)


def normalize_science_styles(
    styles: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Coerce a style name / list to a non-empty SciencePlots style list."""
    if styles is None:
        return list(SCIENCE_STYLE_DEFAULT)
    if isinstance(styles, str):
        styles = [styles]
    out = [str(s).strip() for s in styles if str(s).strip()]
    return out or list(SCIENCE_STYLE_DEFAULT)


def categorical_palette(n: int) -> list[str]:
    """Cycle Okabe–Ito colours to length ``n``."""
    n = max(int(n), 1)
    base = list(CATEGORICAL_COLORS)
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


# ---------------------------------------------------------------------------
# Publication figure geometry
# ---------------------------------------------------------------------------

# Journal column widths in inches; every figure size is derived from these
# rather than from per-plot magic numbers.
FIG_WIDTH_SINGLE = 3.6
FIG_WIDTH_MEDIUM = 5.2
FIG_WIDTH_DOUBLE = 7.2
_GOLDEN = 0.618


def figure_size(
    width: float,
    *,
    aspect: float = _GOLDEN,
    height: float | None = None,
) -> tuple[float, float]:
    """Clamp a requested width to printable limits and derive the height."""
    w = float(np.clip(width, 2.6, 13.0))
    h = float(height) if height is not None else w * aspect
    return w, float(np.clip(h, 2.0, 11.0))


def width_for_levels(
    n_levels: int,
    *,
    per_level: float = 0.62,
    base: float = 1.8,
    minimum: float = FIG_WIDTH_SINGLE,
    maximum: float = FIG_WIDTH_DOUBLE,
) -> float:
    """Figure width that grows with the number of plotted category levels."""
    raw = base + per_level * max(int(n_levels), 1)
    return float(np.clip(raw, minimum, maximum))


def set_titles(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    *,
    pad: float = 6.0,
) -> None:
    """Title plus an optional smaller subtitle line above the axes.

    Sample size and test results live here rather than in a floating badge:
    text above the axes can never occlude the data it describes.
    """
    if subtitle:
        ax.set_title(title, pad=pad + 11.0)
        ax.text(
            0.5, 1.012, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=plt.rcParams["font.size"] * 0.82, color="#444444",
        )
    else:
        ax.set_title(title, pad=pad)


def n_subtitle(n: int, *, extra: str | None = None) -> str:
    """``n = 352 · median 64 [IQR 54–72]`` style subtitle text."""
    parts = [f"n = {int(n)}"]
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def place_legend(
    ax: plt.Axes,
    *,
    title: str | None = None,
    loc: str = "best",
    ncol: int = 1,
    scale: float = 0.85,
    outside: bool = False,
    **kwargs,
) -> None:
    """Compact framed legend with consistent typography.

    ``outside=True`` parks the legend to the right of the axes, which is the
    only placement guaranteed not to cover annotated bars regardless of how
    tall they turn out to be.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if outside:
        kwargs.setdefault("bbox_to_anchor", (1.02, 1.0))
        kwargs.setdefault("borderaxespad", 0.0)
        loc = "upper left"
    ax.legend(
        handles, labels, title=title, loc=loc, ncol=ncol,
        frameon=not outside, framealpha=0.92, fancybox=False,
        fontsize=plt.rcParams["font.size"] * scale,
        title_fontsize=plt.rcParams["font.size"] * scale,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Statistical drawing primitives
# ---------------------------------------------------------------------------

def deterministic_rng(*keys) -> np.random.Generator:
    """Seeded generator so jittered figures are byte-reproducible across runs.

    ``hash()`` is salted per interpreter process, which silently makes jittered
    point clouds differ between runs of the same pipeline.
    """
    material = "|".join(str(k) for k in keys).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")
    return np.random.default_rng(seed)


def wilson_ci(k, n, *, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion (scalar or array).

    Preferred over the Wald interval: it stays inside [0, 1] and behaves at the
    small denominators that per-level subgroup plots routinely produce.
    """
    from scipy.stats import norm

    k_a = np.asarray(k, dtype=float)
    n_a = np.asarray(n, dtype=float)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_a > 0, k_a / n_a, np.nan)
        denom = 1.0 + z**2 / n_a
        centre = (p + z**2 / (2.0 * n_a)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n_a + z**2 / (4.0 * n_a**2)) / denom
    lo = np.clip(centre - half, 0.0, 1.0)
    hi = np.clip(centre + half, 0.0, 1.0)
    lo = np.where(n_a > 0, lo, np.nan)
    hi = np.where(n_a > 0, hi, np.nan)
    if np.ndim(k) == 0 and np.ndim(n) == 0:
        return float(lo), float(hi)
    return lo, hi


def errorbar_lengths(values, lo, hi) -> np.ndarray:
    """Non-negative ``yerr``/``xerr`` pairs from point estimates and CI bounds."""
    v = np.asarray(values, dtype=float)
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    return np.vstack([
        np.nan_to_num(np.maximum(0.0, v - lo_a)),
        np.nan_to_num(np.maximum(0.0, hi_a - v)),
    ])


def freedman_diaconis_bins(
    values, *, minimum: int = 6, maximum: int = 40,
) -> int:
    """Bin count from the Freedman–Diaconis rule, capped for print legibility.

    Never exceeds the number of distinct values, so discrete/count variables do
    not get empty bins between their support points.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    n_unique = int(np.unique(v).size)
    if n < 2 or n_unique < 2:
        return max(int(minimum), 1)

    spread = float(v.max() - v.min())
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 0 and spread > 0:
        width = 2.0 * iqr / np.cbrt(n)
        bins = int(np.ceil(spread / width)) if width > 0 else int(np.ceil(np.sqrt(n)))
    else:
        bins = int(np.ceil(np.sqrt(n)))

    bins = min(bins, n_unique, int(maximum))
    return int(max(bins, min(int(minimum), n_unique)))


def histogram_bin_edges(values, *, max_discrete_levels: int = 25) -> np.ndarray:
    """Bin edges for a histogram, integer-aware.

    Counts and other integer variables get unit-wide bins centred on their
    values. Freedman–Diaconis edges land between integers, which shifts every
    bar half a unit off the value it represents and merges adjacent counts.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.asarray([0.0, 1.0])
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.asarray([lo - 0.5, lo + 0.5])

    n_unique = int(np.unique(v).size)
    integral = bool(np.all(np.isclose(v, np.round(v))))
    if integral and n_unique <= int(max_discrete_levels):
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    return np.linspace(lo, hi, freedman_diaconis_bins(v) + 1)


def kde_curve(
    values,
    *,
    clip: tuple[float, float] | None = None,
    grid: int = 256,
):
    """KDE evaluated only inside the observed support.

    Returns ``None`` when a density is not estimable (n < 3 or zero variance).
    Clipping matters scientifically: an unclipped Gaussian KDE puts mass on
    impossible values (negative volumes, ages below the youngest patient).
    """
    from scipy.stats import gaussian_kde

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or float(np.std(v)) <= 0:
        return None
    lo, hi = clip if clip is not None else (float(v.min()), float(v.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    try:
        density = gaussian_kde(v)
    except (ValueError, np.linalg.LinAlgError):
        return None
    xs = np.linspace(lo, hi, int(grid))
    return xs, np.asarray(density(xs), dtype=float)


def lowess_curve(x, y, *, frac: float = 0.6, min_points: int = 8):
    """Non-parametric LOESS trend, or ``None`` when the fit is not meaningful.

    Used instead of an OLS line wherever the accompanying test is rank-based:
    drawing a straight-line fit next to a Spearman statistic implies a linear
    model that was never assumed or checked.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < int(min_points) or float(np.std(xs)) <= 0:
        return None
    smoothed = lowess(ys, xs, frac=float(np.clip(frac, 0.15, 1.0)), return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def spearman_summary(x, y) -> dict | None:
    """Spearman ρ with a Fisher-z / Bonett–Wright 95% CI and p-value."""
    from scipy.stats import norm, spearmanr

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    n = int(xs.size)
    if n < 4 or float(np.std(xs)) <= 0 or float(np.std(ys)) <= 0:
        return None
    rho, p = spearmanr(xs, ys)
    rho = float(rho)
    if not np.isfinite(rho):
        return None
    out = {"rho": rho, "p": float(p), "n": n, "ci_lo": np.nan, "ci_hi": np.nan}
    if n > 4 and abs(rho) < 1.0:
        # Bonett & Wright (2000) standard error for the Fisher z of Spearman ρ.
        se = np.sqrt((1.0 + rho**2 / 2.0) / (n - 3))
        z = np.arctanh(rho)
        crit = float(norm.ppf(0.975))
        out["ci_lo"] = float(np.tanh(z - crit * se))
        out["ci_hi"] = float(np.tanh(z + crit * se))
    return out


def format_p(p: float | None, *, digits: int = 3) -> str:
    """APA-style p-value text (``p < 0.001`` rather than ``p = 0.0000``)."""
    if p is None:
        return ""
    try:
        val = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    floor = 10.0 ** (-digits)
    if val < floor:
        return f"< {floor:g}"
    return f"= {val:.{digits}f}"


def describe_continuous(values, *, unit: str = "") -> str:
    """``median 64 [IQR 54–72]`` summary text for a subtitle line."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ""
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    digits = 0 if float(np.nanmax(np.abs(v))) >= 100 else (1 if med % 1 else 0)
    suffix = f" {unit}" if unit else ""
    return (
        f"median {med:.{digits}f}{suffix} "
        f"[IQR {q1:.{digits}f}–{q3:.{digits}f}]"
    )


def proportion_bars(
    ax: plt.Axes,
    counts,
    totals,
    *,
    positions=None,
    color: str = PALETTE["primary"],
    width: float = 0.7,
    orient: str = "v",
    label: str | None = None,
    annotate: bool = True,
    alpha: float = 0.05,
    annotate_fontsize: float | None = None,
) -> np.ndarray:
    """Percentage bars with Wilson score CIs (the honest categorical bar chart).

    A bare proportion bar hides both its denominator and its uncertainty; this
    draws the interval and, optionally, the ``k/n`` that produced it. Returns
    the percentages so callers can size the axis.
    """
    k = np.asarray(counts, dtype=float)
    n = np.asarray(totals, dtype=float)
    pos = (
        np.arange(k.size, dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.where(n > 0, k / n, np.nan)
    lo, hi = wilson_ci(k, n, alpha=alpha)
    pct, pct_lo, pct_hi = prop * 100.0, np.asarray(lo) * 100.0, np.asarray(hi) * 100.0
    err = errorbar_lengths(pct, pct_lo, pct_hi)
    vertical = str(orient).lower().startswith("v")
    heights = np.nan_to_num(pct)

    if vertical:
        ax.bar(
            pos, heights, width=width, color=color, edgecolor="white",
            linewidth=0.6, label=label, zorder=2,
        )
        ax.errorbar(
            pos, heights, yerr=err, fmt="none", ecolor="#333333",
            elinewidth=0.9, capsize=2.5, zorder=3,
        )
    else:
        ax.barh(
            pos, heights, height=width, color=color, edgecolor="white",
            linewidth=0.6, label=label, zorder=2,
        )
        ax.errorbar(
            heights, pos, xerr=err, fmt="none", ecolor="#333333",
            elinewidth=0.9, capsize=2.5, zorder=3,
        )

    if annotate:
        fs = annotate_fontsize or plt.rcParams["font.size"] * 0.8
        for p_i, val, top, k_i, n_i in zip(pos, heights, pct_hi, k, n):
            if not np.isfinite(val):
                continue
            text = f"{val:.0f}% ({int(k_i)}/{int(n_i)})"
            edge = top if np.isfinite(top) else val
            if vertical:
                ax.annotate(
                    text, xy=(p_i, edge), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=fs, color="#333333",
                )
            else:
                ax.annotate(
                    text, xy=(edge, p_i), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=fs, color="#333333",
                )
    return pct


def boxplot_orientation(vertical: bool) -> dict:
    """``Axes.boxplot`` orientation kwarg, tolerant of the ``vert`` deprecation."""
    import inspect

    params = inspect.signature(plt.Axes.boxplot).parameters
    if "orientation" in params:
        return {"orientation": "vertical" if vertical else "horizontal"}
    return {"vert": bool(vertical)}


def raincloud(
    ax: plt.Axes,
    groups: Sequence[Sequence[float]],
    *,
    positions: Sequence[float] | None = None,
    colors: Sequence[str] | None = None,
    orient: str = "v",
    width: float = 0.8,
    rng: np.random.Generator | None = None,
    point_size: float = 5.0,
    max_points: int = 400,
) -> None:
    """Raincloud (half-violin + box + jittered raw points) for each group.

    Three non-overlapping lanes per slot — density, summary, raw data — so no
    element hides another. Groups too small for a density estimate still get
    their box and points instead of being silently dropped.
    """
    vertical = str(orient).lower().startswith("v")
    n_groups = len(groups)
    if n_groups == 0:
        return
    pos = (
        np.arange(n_groups, dtype=float)
        if positions is None
        else np.asarray(positions, dtype=float)
    )
    palette = list(colors) if colors else categorical_palette(n_groups)
    rng = rng if rng is not None else deterministic_rng("raincloud", n_groups)

    slot = float(width)
    cloud_base = 0.06 * slot
    cloud_span = 0.36 * slot
    box_offset = -0.08 * slot
    box_width = 0.14 * slot
    rain_offset = -0.28 * slot
    rain_spread = 0.09 * slot

    box_orientation = boxplot_orientation(vertical)

    def _place(along, across):
        """Map (category position, value) onto (x, y) for the chosen orient."""
        return (along, across) if vertical else (across, along)

    for idx, raw in enumerate(groups):
        vals = np.asarray(list(raw), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        centre = float(pos[idx])
        color = palette[idx % len(palette)]

        curve = kde_curve(vals, clip=(float(vals.min()), float(vals.max())))
        if curve is not None:
            grid, dens = curve
            peak = float(dens.max())
            if peak > 0:
                scaled = centre + cloud_base + (dens / peak) * cloud_span
                lower = np.full_like(scaled, centre + cloud_base)
                if vertical:
                    ax.fill_betweenx(
                        grid, lower, scaled, color=color, alpha=0.35,
                        linewidth=0.0, zorder=2,
                    )
                    ax.plot(scaled, grid, color=color, linewidth=0.9, zorder=3)
                else:
                    ax.fill_between(
                        grid, lower, scaled, color=color, alpha=0.35,
                        linewidth=0.0, zorder=2,
                    )
                    ax.plot(grid, scaled, color=color, linewidth=0.9, zorder=3)

        ax.boxplot(
            [vals],
            positions=[centre + box_offset],
            widths=box_width,
            showfliers=False,
            **box_orientation,
            patch_artist=True,
            medianprops={"color": "#1a1a1a", "linewidth": 1.3},
            whiskerprops={"color": "#4d4d4d", "linewidth": 0.9},
            capprops={"color": "#4d4d4d", "linewidth": 0.9},
            boxprops={
                "facecolor": "white", "edgecolor": color,
                "linewidth": 1.0, "alpha": 0.95,
            },
            zorder=4,
        )

        shown = vals
        if vals.size > int(max_points):
            shown = rng.choice(vals, size=int(max_points), replace=False)
        jitter = rng.uniform(-rain_spread, rain_spread, size=shown.size)
        along = np.full(shown.size, centre + rain_offset) + jitter
        px, py = _place(along, shown)
        ax.scatter(
            px, py, s=point_size, color=color, alpha=0.45,
            edgecolors="none", zorder=3, rasterized=False,
        )


@contextmanager
def science_style_context(
    styles: str | list[str] | tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Temporary SciencePlots context (for overrides / tests)."""
    import scienceplots  # noqa: F401

    with plt.style.context(normalize_science_styles(styles)):
        _apply_export_overrides()
        yield


def maybe_science_style(
    styles: str | list[str] | tuple[str, ...] | None = None,
):
    """No-op if ``styles`` is None (global ``apply_plot_style`` already active)."""
    if styles is None:
        return nullcontext()
    return science_style_context(styles)


def _apply_export_overrides() -> None:
    """SVG-friendly overrides that SciencePlots does not set for this pipeline."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.titlepad": 12,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "axes.axisbelow": True,
        # Radiology house font. SciencePlots' `nature` style sets a Times-like
        # serif; journals in this field want Arial/Helvetica, and every figure
        # in a submission has to match, so it is set here rather than per
        # figure. Mathtext follows, or exponents would stay serif.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })


def apply_plot_style(
    styles: str | list[str] | tuple[str, ...] | None = None,
) -> None:
    """Apply SciencePlots + export overrides for the whole plotting session."""
    import scienceplots  # noqa: F401

    plt.style.use(normalize_science_styles(styles))
    _apply_export_overrides()


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    *,
    close: bool = True,
    pad_inches: float | None = None,
    tight_layout: bool = True,
) -> Path:
    """Single SVG export path for every phase module.

    ``tight_layout=False`` preserves hand-tuned panel spacing (shared-axis
    figures) while still trimming the outer margin via ``bbox_inches``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    kwargs: dict = {"format": "svg", "bbox_inches": "tight"}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(out, **kwargs)
    if close:
        plt.close(fig)
    return out


def figure_to_svg_bytes(
    fig: plt.Figure,
    *,
    close: bool = True,
    pad_inches: float | None = None,
    tight_layout: bool = True,
) -> bytes:
    """SVG bytes (association heatmap / in-memory exports)."""
    if tight_layout:
        fig.tight_layout()
    buf = BytesIO()
    kwargs: dict = {"format": "svg", "bbox_inches": "tight"}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(buf, **kwargs)
    if close:
        plt.close(fig)
    return buf.getvalue()
