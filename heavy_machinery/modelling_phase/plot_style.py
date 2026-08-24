"""One plotting pipeline for DDA / EDA / forest / missingness / report captions.

Shared concerns only (do not duplicate these in phase modules):

1. AJNR house style (Okabe–Ito, Arial/Helvetica, journal vs conference)
2. Publication figure geometry, title/subtitle blocks
3. Statistical drawing primitives (Wilson CIs, KDE, LOWESS, raincloud)
4. Forest / ROC / calibration / decision-curve builders
5. TIF / PNG save and PNG bytes export
6. Human-readable labels for axes, category levels, and report captions

Everything here is column-agnostic: label maps are data, not logic, so phase
modules never branch on a study-specific column name.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

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

# Okabe–Ito named swatches (colour-blind safe).
OKABE = {
    "black":     "#000000",
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "grey":      "#7F7F7F",
    "lightgrey": "#D9D9D9",
}
G1 = OKABE["blue"]        # WHO CNS grade 1
HG = OKABE["vermilion"]   # WHO CNS grade 2-3
NS = OKABE["grey"]        # non-significant / CI crosses the null

# Semantic roles mapped onto the same family.
PALETTE = {
    "primary": CATEGORICAL_COLORS[0],   # blue
    "accent": CATEGORICAL_COLORS[1],    # vermillion
    "good": CATEGORICAL_COLORS[2],     # green
    "bad": CATEGORICAL_COLORS[1],      # vermillion
    "neutral": NS,
    "low_grade": G1,
    "high_grade": HG,
    "significant": CATEGORICAL_COLORS[1],
    "nonsignificant": NS,
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
        # Three significant figures, the same as the cut-point phase prints on
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
    """Record a figure's title and its subtitle — as text, not as pixels.

    Nothing is drawn. Both are attached to the figure and :func:`save_figure`
    writes them to the ``.legend.json`` sidecar, from which the report prints
    the title above the image and the subtitle below it. See
    :func:`set_figure_legend` for why the words stay out of the image.

    A figure whose panels each call this — a facet grid, a two-panel
    comparison — accumulates them in drawing order rather than letting the last
    panel silently overwrite the first.

    ``pad`` is accepted and ignored. It described the gap the drawn title used
    to need above the axes, and callers still pass it.
    """
    del pad
    fig = ax.figure
    legend = dict(getattr(fig, _LEGEND_ATTR, None)
                  or {k: "" for k in _LEGEND_FIELDS})

    def _add(field: str, text: str | None) -> None:
        text = (text or "").strip()
        have = legend.get(field, "")
        if not text or text in have.split(" · "):
            return
        legend[field] = f"{have} · {text}" if have else text

    _add("title", title)
    _add("note", subtitle)
    setattr(fig, _LEGEND_ATTR, legend)


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
            text = f"{val:.1f}% ({int(k_i)}/{int(n_i)})"
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


_BASE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

_AJNR = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
}

_CONF = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "axes.linewidth": 1.2,
}

STYLE = "ajnr"
_SAVE_DPI = {"line": 1200, "combo": 600, "halftone": 300}

# ---------------------------------------------------------------------------
# Figure export profile
# ---------------------------------------------------------------------------
# Two audiences want different files out of the same figure, and writing for
# both on every run is what made the pipeline image-bound: rasterising and
# LZW-compressing a 1200-dpi TIF is ~0.3 s per figure, and report.html inlines
# its PNGs as base64, so a 24-megapixel PNG shown at 36 rem is ~86x the pixels
# the browser can use.
#
#   "report"      (default) only the PNGs report.html embeds, at screen dpi.
#   "submission"  byte-for-byte what the pipeline produced before this switch
#                 existed: 1200-dpi TIF + PNG for journal upload.
#
# Select with the ATYPIER_FIGURES environment variable, e.g.
#   ATYPIER_FIGURES=submission jupyter nbconvert --execute meningioma-modelling.ipynb
FIGURE_PROFILES: dict[str, tuple[str, ...]] = {
    "report": ("png",),
    "submission": ("tif", "png"),
}
# Suffixes save_figure() may strip. Anything else after a dot is part of the
# name — cut-points routinely put one there.
_IMAGE_SUFFIXES = frozenset({"png", "tif", "tiff", "jpg", "jpeg",
                            "pdf", "svg", "eps"})

_FIGURE_PROFILE_ENV = "ATYPIER_FIGURES"
_DEFAULT_FIGURE_PROFILE = "report"

# Report cards cap figures at 36 rem (576 px); 200 dpi keeps a 7.2-inch figure
# at 1440 px, which is still oversampled on a 2x display.
REPORT_PNG_DPI = 200


def figure_profile() -> str:
    """Active export profile — read fresh so a notebook can flip it mid-session."""
    name = os.environ.get(_FIGURE_PROFILE_ENV, _DEFAULT_FIGURE_PROFILE).strip().lower()
    if name not in FIGURE_PROFILES:
        raise ValueError(
            f"{_FIGURE_PROFILE_ENV}={name!r} is not a figure profile; "
            f"expected one of {sorted(FIGURE_PROFILES)}"
        )
    return name


def figure_formats() -> tuple[str, ...]:
    """File formats written by :func:`save_figure` under the active profile."""
    return FIGURE_PROFILES[figure_profile()]


def _dpi_for(fmt: str, kind: str) -> int:
    """Export dpi. PNG is a screen artifact; everything else is print-bound."""
    if fmt == "png" and figure_profile() == "report":
        return min(REPORT_PNG_DPI, _SAVE_DPI[kind])
    return _SAVE_DPI[kind]


def use_style(profile: str = "ajnr") -> None:
    """Set global style. profile in {'ajnr', 'conference'}."""
    global STYLE
    if profile not in ("ajnr", "conference"):
        raise ValueError("profile must be 'ajnr' or 'conference'")
    STYLE = profile
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(_BASE)
    mpl.rcParams.update(_AJNR if profile == "ajnr" else _CONF)


def _title(ax, text):
    """On-figure titles are for posters only. Journal titles live in the legend."""
    if STYLE == "conference" and text:
        ax.set_title(text, loc="left", fontweight="bold", pad=10)


def footnote(fig, text, y=-0.02):
    """Poster-only footnote block. For AJNR this text belongs in the figure legend."""
    if STYLE == "conference" and text:
        fig.text(0.0, y, text, ha="left", va="top", fontsize=10,
                 color="#333333", wrap=True)


def panel_label(ax, letter, dx=-0.10, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=mpl.rcParams["axes.titlesize"] + 1, va="bottom", ha="left")


@contextmanager
def science_style_context(
    styles: str | list[str] | tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Temporary AJNR rc context (``styles`` is a legacy no-op)."""
    del styles
    with mpl.rc_context():
        use_style(STYLE)
        yield


def maybe_science_style(
    styles: str | list[str] | tuple[str, ...] | None = None,
):
    """No-op if ``styles`` is None (global ``apply_plot_style`` already active)."""
    if styles is None:
        return nullcontext()
    return science_style_context(styles)


def apply_plot_style(
    styles: str | list[str] | tuple[str, ...] | None = None,
) -> None:
    """Apply AJNR house style for the whole plotting session.

    ``styles`` is accepted for call-site compatibility and ignored.
    """
    del styles
    use_style("ajnr")


_LEGEND_ATTR = "_atypier_legend"


_LEGEND_FIELDS = ("title", "plain", "note")


def set_figure_legend(fig: plt.Figure, *, title: str = "", plain: str = "",
                      note: str = "") -> None:
    """Hand a figure's title and its two explanations to the report as *text*.

    Nothing is drawn. :func:`save_figure` writes the three to a ``.legend.json``
    sidecar beside the image, and the report prints the title above the figure
    with ``plain`` then ``note`` below it.

    The two explanations answer different questions and are not shortened
    versions of each other:

    - ``plain`` — how to read the picture, in one or two sentences of ordinary
      words. What the marks are, which direction is better. No title, no
      statistics vocabulary, nothing a reader has to already know.
    - ``note`` — the ``Note:—`` block as the journal sets it: definitions,
      abbreviations, cohort, and the caveats a reviewer needs.

    Words belong in the page, not in the pixels: kept out of the image they stay
    selectable, searchable and re-wrappable at any width, and the exported TIF
    reaches the journal without a legend burnt into it — which is how a figure is
    supposed to be submitted. Panel letters (A, B, C) are the exception and stay
    drawn, because they are positional: they label a place in the image and mean
    nothing detached from it.
    """
    values = {"title": title, "plain": plain, "note": note}
    setattr(fig, _LEGEND_ATTR,
            {k: (values[k] or "").strip() for k in _LEGEND_FIELDS})


def figure_legend(fig: plt.Figure) -> dict[str, str]:
    """The title/plain/note recorded on a figure, before it has been saved."""
    return dict(getattr(fig, _LEGEND_ATTR, None)
                or {k: "" for k in _LEGEND_FIELDS})


def figure_legend_path(image_path: Path | str) -> Path:
    """The sidecar carrying one figure's title and note."""
    p = Path(image_path)
    stem = (p.with_suffix("")
            if p.suffix.lower().lstrip(".") in _IMAGE_SUFFIXES else p)
    return Path(f"{stem}.legend.json")


def write_figure_legend(image_path: Path | str, *, title: str = "",
                        plain: str = "", note: str = "") -> Path | None:
    """Write a legend sidecar for an image saved outside :func:`save_figure`.

    A figure rendered straight to bytes never passes through ``save_figure`` and
    so never gets its legend written; this is the escape hatch for those. An
    all-empty legend removes any sidecar left from an earlier run rather than
    leaving stale text on the page.
    """
    values = {"title": title, "plain": plain, "note": note}
    legend = {k: (values[k] or "").strip() for k in _LEGEND_FIELDS}
    path = figure_legend_path(image_path)
    if not any(legend.values()):
        if path.is_file():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legend, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def read_figure_legend(image_path: Path | str) -> dict[str, str]:
    """``{"title": ..., "plain": ..., "note": ...}``; empty when it has none.

    A missing or unreadable sidecar is not an error — the figure simply has no
    legend of its own and the caller falls back to the name-derived caption.
    """
    path = figure_legend_path(image_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {k: str(data.get(k) or "").strip() for k in _LEGEND_FIELDS}
    return out if any(out.values()) else {}


def prune_embedded_figures(
    report_path: Path | str,
    *,
    roots: Sequence[Path | str],
    dry_run: bool = False,
) -> tuple[int, int, list[Path]]:
    """Delete figure files whose bytes are provably inside the report.

    The report embeds every figure as base64, so once it is written the PNG on
    disk is a second copy of the same pixels. This reclaims that space.

    It deletes a file **only** when the report demonstrably contains those exact
    bytes. A figure that silently failed to render or embed must not be deleted
    merely for sitting in a figures directory — otherwise one bad render turns
    into quiet data loss, which is the one outcome worse than wasted disk.

    TIFFs and vector exports are never touched: those are the journal
    deliverable, not a duplicate of anything on the page. A pruned figure's
    ``.legend.json`` goes with it, since the legend has already been baked into
    the report and describes a file that no longer exists.

    Returns ``(files_deleted, bytes_reclaimed, kept)`` where ``kept`` lists the
    PNGs left behind because the report does not contain them.
    """
    html = Path(report_path).read_text(encoding="utf-8")
    embedded: set[str] = set()
    for chunk in re.findall(r"base64,([A-Za-z0-9+/=]{200,})", html):
        try:
            embedded.add(hashlib.md5(base64.b64decode(chunk)).hexdigest())
        except Exception:            # a truncated or non-image payload
            continue

    deleted = reclaimed = 0
    kept: list[Path] = []
    for root in roots:
        for png in sorted(Path(root).rglob("*.png")):
            if hashlib.md5(png.read_bytes()).hexdigest() not in embedded:
                kept.append(png)
                continue
            size = png.stat().st_size
            sidecar = figure_legend_path(png)
            if not dry_run:
                png.unlink()
                if sidecar.is_file():
                    sidecar.unlink()
            deleted += 1
            reclaimed += size
    return deleted, reclaimed, kept


def save_figure(
    fig: plt.Figure,
    path: Path | str,
    *,
    close: bool = True,
    pad_inches: float | None = None,
    tight_layout: bool = True,
    kind: str = "line",
    formats: tuple[str, ...] | None = None,
    min_width_in: float = 4.0,
) -> Path:
    """AJNR export. Returns the PNG path for report embedding.

    ``path`` may carry any suffix; it is stripped to a stem. ``kind`` selects
    print dpi (line 1200, combo 600, halftone 300). ``tight_layout=False``
    preserves hand-tuned panel spacing.

    ``formats=None`` follows the active :func:`figure_profile` — PNG only for a
    normal run, TIF + PNG under ``ATYPIER_FIGURES=submission``. Pass an explicit
    tuple to override the profile for one figure.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    w, h = fig.get_size_inches()
    if w < min_width_in:
        fig.set_size_inches(min_width_in, h * min_width_in / w)
    if formats is None:
        formats = figure_formats()
    extra: dict = {}
    if pad_inches is not None:
        extra["pad_inches"] = pad_inches
    # Only strip a *real* image extension. A cut-point in the name —
    # "high_grade__adc_value_le0.72" — is not a suffix, and with_suffix() ate it,
    # so the file landed as adc_value_le0.png and every caption derived from that
    # stem read "ADC value ≤ 0": an impossible threshold, on a plot whose own
    # axis label was correct.
    stem = (out.with_suffix("")
            if out.suffix.lower().lstrip(".") in _IMAGE_SUFFIXES else out)
    written: list[Path] = []
    for fmt in formats:
        pth = Path(f"{stem}.{fmt}")
        dpi = _dpi_for(fmt, kind)
        if fmt == "tif":
            fig.savefig(pth, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"}, **extra)
        elif fmt in ("pdf", "svg"):
            fig.savefig(pth, **extra)
        else:
            fig.savefig(pth, dpi=dpi, **extra)
        written.append(pth)
    # The sidecar is written next to the image and refreshed on every run, so a
    # legend can never outlive the figure it describes.
    legend = getattr(fig, _LEGEND_ATTR, None)
    sidecar = figure_legend_path(stem)
    if legend and any(legend.get(k) for k in _LEGEND_FIELDS):
        sidecar.write_text(json.dumps(legend, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    elif sidecar.is_file():
        sidecar.unlink()
    if close:
        plt.close(fig)
    png = next((p for p in written if p.suffix.lower() == ".png"), written[0])
    return png


def figure_to_png_bytes(
    fig: plt.Figure,
    *,
    close: bool = True,
    pad_inches: float | None = None,
    tight_layout: bool = True,
    kind: str = "halftone",
) -> bytes:
    """PNG bytes (association heatmap / in-memory exports)."""
    if tight_layout:
        fig.tight_layout()
    buf = BytesIO()
    kwargs: dict = {"format": "png", "dpi": _SAVE_DPI[kind], "bbox_inches": "tight"}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(buf, **kwargs)
    if close:
        plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Figure builders (AJNR)
# ---------------------------------------------------------------------------

def roc_auc(y_true, y_score) -> float:
    """ROC AUC for a binary label, via the Mann-Whitney statistic.

    Equivalent to ``sklearn.metrics.roc_auc_score`` — tied scores take the
    mid-rank, which is exactly the half-credit the trapezoidal ROC gives them —
    but without sklearn's per-call input validation. That validation dominates
    the cost inside a bootstrap loop, where this is called thousands of times
    on arrays that are already clean.

    Returns NaN when one class is absent, where AUC is undefined.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score, kind="mergesort")
    ordered = y_score[order]
    n = ordered.size
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    bounds = np.r_[starts, n]
    # Average rank shared by each run of equal scores, 1-based.
    group_rank = (bounds[:-1] + bounds[1:] + 1) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(group_rank, np.diff(bounds))

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ci(y, score, n_boot=2000, seed=42, alpha=0.05):
    """Stratified bootstrap percentile 95% CI for AUC. Returns (auc, lo, hi)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    point = roc_auc(y, score)
    rng = np.random.default_rng(seed)
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = np.concatenate([rng.choice(idx_pos, idx_pos.size, replace=True),
                            rng.choice(idx_neg, idx_neg.size, replace=True)])
        boots[b] = roc_auc(y[s], score[s])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def youden_point(y, score):
    """Return (fpr, tpr, threshold) maximising the Youden index."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, score)
    j = np.argmax(tpr - fpr)
    return fpr[j], tpr[j], thr[j]


def _lowess(y, x, frac=0.75):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _sm
        out = _sm(y, x, frac=frac, return_sorted=True)
        return out[:, 0], out[:, 1]
    except ImportError:
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        n = xs.size
        w = max(int(frac * n), 3)
        fit = np.empty(n)
        for i in range(n):
            d = np.abs(xs - xs[i])
            h = np.sort(d)[w - 1] or 1e-9
            u = np.clip(d / h, 0, 1)
            wt = (1 - u ** 3) ** 3
            X = np.vstack([np.ones(n), xs - xs[i]]).T
            W = wt[:, None]
            beta = np.linalg.lstsq((X * W).T @ X, (X * W).T @ ys, rcond=None)[0]
            fit[i] = beta[0]
        return xs, fit


def calibration_metrics(y, p, eps=1e-6):
    """Calibration intercept (ideal 0), slope (ideal 1), and Brier score."""
    from scipy.optimize import brentq
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)\
        .fit(lp.reshape(-1, 1), y).coef_[0][0]

    def score(a):
        return np.sum(y - 1.0 / (1.0 + np.exp(-(a + lp))))
    intercept = brentq(score, -20, 20)
    return dict(intercept=float(intercept), slope=float(slope),
                brier=float(brier_score_loss(y, p)))


def forest_row_order(est, lo=None, hi=None, *, ref=1.0) -> np.ndarray:
    """Row order for a forest: all rows together, estimate descending."""
    del lo, hi, ref
    return np.argsort(-np.asarray(est, dtype=float))


def forest_lr(labels, est, lo, hi, *, n_hg=None, n_g1=None, ref=1.0,
              xlabel="Positive likelihood ratio (95% CI)", value_header="LR+ (95% CI)",
              title=None, width=7.0, row_h=0.30, log=True, ax=None, ns=None,
              order=None, open_marker=False, show_labels=True):
    """Forest plot for LR+ or OR. Rows whose CI crosses ``ref`` are drawn grey.
    All rows share one ranking by the estimate, descending.

    ``ns`` optionally marks extra rows as non-significant (e.g. an FDR-p above
    alpha). It only ever adds grey: a row whose interval crosses ``ref`` stays
    grey whatever ``ns`` says, so a full-ink row never straddles the null.

    ``order`` pins the row order instead of ranking by this panel's own
    estimate. Two panels side by side have to agree on which row is which, and
    a panel that re-sorted itself would put a variable's name against another
    variable's estimate. ``show_labels=False`` drops the y tick labels for the
    right-hand panel of such a pair, which reads off the left one's.

    ``open_marker`` draws hollow squares. Filled against hollow is the one
    pairing that survives greyscale and photocopying, which colour does not.
    """
    del n_hg, n_g1
    order = (forest_row_order(est, lo, hi, ref=ref) if order is None
             else np.asarray(order, dtype=int))
    labels = [labels[i] for i in order]
    est = np.asarray(est, dtype=float)[order]
    lo = np.asarray(lo, dtype=float)[order]
    hi = np.asarray(hi, dtype=float)[order]
    ns = None if ns is None else np.asarray(ns, dtype=bool)[order]
    k = len(labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, max(2.2, row_h * k + 1.0)))
    else:
        fig = ax.figure
    y = np.arange(k)[::-1]
    crosses = (lo <= ref) & (hi >= ref)
    if ns is not None:
        crosses = crosses | ns

    ms = mpl.rcParams["lines.markersize"]
    for i, yy in enumerate(y):
        c = NS if crosses[i] else OKABE["black"]
        ax.plot([lo[i], hi[i]], [yy, yy], color=c, lw=1.1,
                solid_capstyle="butt", zorder=2)
        if open_marker:
            ax.plot([est[i]], [yy], marker="s", ms=ms, mfc="white", mec=c,
                    mew=1.1, ls="none", zorder=3)
        else:
            ax.plot([est[i]], [yy], marker="s", ms=ms, color=c, zorder=3)
        if crosses[i]:
            ax.axhspan(yy - 0.5, yy + 0.5, color=OKABE["lightgrey"], alpha=0.35, zorder=0)

    ax.axvline(ref, color=OKABE["black"], lw=0.8, ls="--", zorder=1)
    if log:
        ax.set_xscale("log")
        ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_labels else [""] * k)
    ax.set_ylim(-0.7, k - 0.3)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for i, yy in enumerate(y):
        ax.text(1.02, yy, f"{est[i]:.2f} ({lo[i]:.2f}\u2013{hi[i]:.2f})",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=mpl.rcParams["ytick.labelsize"],
                color=NS if crosses[i] else OKABE["black"], clip_on=False)
    ax.text(1.02, k - 0.15, value_header, transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontweight="bold",
            fontsize=mpl.rcParams["ytick.labelsize"], clip_on=False)
    _title(ax, title)
    return fig, ax


def roc_panel(curves, *, title=None, mark_youden=True, ax=None, figsize=(3.6, 3.6),
              n_boot=1000):
    """ROC panel. Each curve is ``name`` + ``color`` and either ``y``/``score``
    or precomputed ``fpr``/``tpr`` (optional ``auc``, ``auc_lo``, ``auc_hi``, ``n``).
    """
    from sklearn.metrics import roc_curve

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color=OKABE["grey"], zorder=0)

    for c in curves:
        color = c.get("color", OKABE["blue"])
        name = c.get("name", "Model")
        if "fpr" in c and "tpr" in c:
            fpr = np.asarray(c["fpr"], dtype=float)
            tpr = np.asarray(c["tpr"], dtype=float)
            a = c.get("auc")
            lo, hi = c.get("auc_lo"), c.get("auc_hi")
            n = c.get("n")
            if a is None:
                label = name
            elif lo is not None and hi is not None:
                n_bit = f"; n = {int(n)}" if n is not None else ""
                label = f"{name}\nAUC {float(a):.2f} (95% CI {float(lo):.2f}\u2013{float(hi):.2f}){n_bit}"
            else:
                label = f"{name}\nAUC {float(a):.3f}"
        else:
            y, s = np.asarray(c["y"]), np.asarray(c["score"], dtype=float)
            m = ~np.isnan(s)
            y, s = y[m], s[m]
            fpr, tpr, _ = roc_curve(y, s)
            a, lo, hi = auc_ci(y, s, n_boot=n_boot)
            label = f"{name}\nAUC {a:.2f} (95% CI {lo:.2f}\u2013{hi:.2f}); n = {y.size}"
            n = y.size
        ax.plot(fpr, tpr, color=color, label=label)
        if mark_youden and len(fpr):
            j = int(np.argmax(np.asarray(tpr) - np.asarray(fpr)))
            ax.plot([fpr[j]], [tpr[j]], "o", ms=mpl.rcParams["lines.markersize"],
                    mfc="white", mec=color, mew=1.4, zorder=4)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", handlelength=1.4, labelspacing=0.8, borderpad=0.2)
    _title(ax, title)
    return fig, ax


def calibration_plot(y=None, p=None, *, n_bins=10, title=None, curve_color=None,
                     show_hist=True, figsize=(3.6, 4.0), frac=0.75, label=None,
                     bins=None, smooth=None, metrics=None, ax=None):
    """Flexible calibration curve. Raw ``y``/``p`` compute LOWESS + bins +
    metrics; otherwise draw stored ``bins`` / ``smooth`` / ``metrics``.
    """
    curve_color = curve_color or OKABE["blue"]
    y_arr = None if y is None else np.asarray(y).astype(int)
    p_arr = None if p is None else np.asarray(p, dtype=float)
    can_hist = show_hist and ax is None and y_arr is not None and p_arr is not None

    if ax is not None:
        fig = ax.figure
        axh = None
    elif can_hist:
        fig, (ax, axh) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw=dict(height_ratios=[4, 1], hspace=0.06))
    else:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))
        axh = None

    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color=OKABE["grey"], zorder=0,
            label="Ideal")

    if y_arr is not None and p_arr is not None:
        xs, fit = _lowess(y_arr.astype(float), p_arr, frac=frac)
        ax.plot(xs, np.clip(fit, 0, 1), color=curve_color,
                lw=mpl.rcParams["lines.linewidth"], zorder=3,
                label=label or "Flexible calibration")
        rng = np.random.default_rng(7)
        grid = np.linspace(p_arr.min(), p_arr.max(), 100)
        boots = []
        for _ in range(300):
            idx = rng.integers(0, y_arr.size, y_arr.size)
            bx, bf = _lowess(y_arr[idx].astype(float), p_arr[idx], frac=frac)
            boots.append(np.interp(grid, bx, bf))
        lo, hi = np.percentile(np.vstack(boots), [2.5, 97.5], axis=0)
        ax.fill_between(grid, np.clip(lo, 0, 1), np.clip(hi, 0, 1),
                        color=curve_color, alpha=0.18, lw=0, zorder=2)
        edges = np.quantile(p_arr, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        bi = np.digitize(p_arr, edges[1:-1])
        mx = np.array([p_arr[bi == b].mean() for b in range(n_bins)])
        my = np.array([y_arr[bi == b].mean() for b in range(n_bins)])
        ax.plot(mx, my, "s", ms=mpl.rcParams["lines.markersize"], color=OKABE["black"],
                zorder=4, label=f"Observed, {n_bins} equal-size groups")
        m = metrics or calibration_metrics(y_arr, p_arr)
    else:
        if smooth and smooth.get("predicted"):
            ax.plot(smooth["predicted"], smooth["observed"], color=curve_color,
                    lw=mpl.rcParams["lines.linewidth"], zorder=3,
                    label=label or "Flexible calibration")
        if bins:
            mx = np.array([float(b["predicted"]) for b in bins])
            my = np.array([float(b["observed"]) for b in bins])
            ax.plot(mx, my, "s", ms=mpl.rcParams["lines.markersize"],
                    color=OKABE["black"], zorder=4,
                    label=f"Observed, {len(bins)} equal-size groups")
        m = metrics or {}

    if m:
        intercept = m.get("intercept", m.get("intercept_corrected",
                                             m.get("intercept_apparent")))
        slope = m.get("slope", m.get("slope_corrected", m.get("slope_apparent")))
        brier = m.get("brier")
        lines = []
        if intercept is not None and np.isfinite(float(intercept)):
            lines.append(f"Calibration intercept {float(intercept):.2f}")
        if slope is not None and np.isfinite(float(slope)):
            lines.append(f"Calibration slope {float(slope):.2f}")
        if brier is not None and np.isfinite(float(brier)):
            lines.append(f"Brier score {float(brier):.3f}")
        if lines:
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                    va="top", ha="left", fontsize=mpl.rcParams["xtick.labelsize"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed proportion, WHO CNS grade 2\u20133")
    ax.legend(loc="lower right", handlelength=1.4, borderpad=0.2)
    _title(ax, title)

    if axh is not None:
        axh.hist(p_arr[y_arr == 0], bins=np.linspace(0, 1, 41), color=G1, alpha=0.65, lw=0)
        axh.hist(p_arr[y_arr == 1], bins=np.linspace(0, 1, 41), color=HG, alpha=0.65, lw=0)
        axh.set_yticks([])
        axh.spines["left"].set_visible(False)
        axh.set_xlabel("Predicted probability of WHO CNS grade 2\u20133")
    else:
        ax.set_xlabel("Predicted probability of WHO CNS grade 2\u20133")
    return fig, ax


def decision_curve(y=None, models=None, *, thresholds=None, title=None, ax=None,
                   figsize=(4.0, 3.4), series=None, prevalence=None):
    """Net benefit vs threshold. Raw ``y`` + ``models`` dict, or stored
    ``series`` mapping name → (thresholds, net_benefit).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    palette = [OKABE["blue"], OKABE["vermilion"], OKABE["green"], OKABE["purple"]]

    if series is not None:
        model_i = 0
        for name, (t, nb) in series.items():
            t = np.asarray(t, dtype=float)
            nb = np.asarray(nb, dtype=float)
            if name == "Treat all":
                ax.plot(t, nb, color=OKABE["grey"], lw=1.0, ls="--", label="Treat all")
            elif name == "Treat none":
                ax.axhline(0, color=OKABE["black"], lw=0.8, label="Treat none")
            else:
                ax.plot(t, nb, color=palette[model_i % len(palette)], label=name)
                model_i += 1
        nb_vals = np.concatenate([np.asarray(v[1], dtype=float) for v in series.values()])
        prev = 0.3 if prevalence is None else float(prevalence)
        finite = nb_vals[np.isfinite(nb_vals)]
        top = float(np.nanmax(finite)) if finite.size else prev
        ax.set_ylim(min(-0.02, prev * -0.15), max(prev * 1.05, top * 1.05))
    else:
        y = np.asarray(y).astype(int)
        n = y.size
        thresholds = np.linspace(0.01, 0.60, 120) if thresholds is None else thresholds
        prev = y.mean()
        nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
        ax.plot(thresholds, nb_all, color=OKABE["grey"], lw=1.0, ls="--", label="Treat all")
        ax.axhline(0, color=OKABE["black"], lw=0.8, label="Treat none")
        for (name, p), c in zip((models or {}).items(), palette):
            p = np.asarray(p, dtype=float)
            nb = []
            for t in thresholds:
                pred = p >= t
                tp = np.sum(pred & (y == 1)) / n
                fp = np.sum(pred & (y == 0)) / n
                nb.append(tp - fp * t / (1 - t))
            ax.plot(thresholds, nb, color=c, label=name)
        ax.set_ylim(min(-0.02, prev * -0.15), prev * 1.05)

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", handlelength=1.4, borderpad=0.2)
    _title(ax, title)
    return fig, ax
