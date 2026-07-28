"""One plotting pipeline for DDA / EDA / forest / missingness / report captions.

Shared concerns only (do not duplicate these in phase modules):

1. SciencePlots session style (``science`` + ``nature`` + ``no-latex``)
2. Okabe–Ito palette + ``n=`` badge
3. SVG save / bytes export
4. Human-readable labels for axes and report captions
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from io import BytesIO
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt

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

# Figure-type suffixes in file stems → descriptive phrases.
_FIG_SUFFIX = {
    "hist": "distribution",
    "box": "box plot",
    "bar": "counts",
    "timeline": "records over time",
    "forest": "forest plot",
    "vif": "collinearity (VIF)",
    "multivariable": "multivariable model",
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


def annotate_total_n(ax: plt.Axes, n: int) -> None:
    """Corner badge with total non-missing n used in the plot."""
    ax.text(
        0.98, 0.98, f"n={int(n)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
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
) -> Path:
    """Single SVG export path for every phase module."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
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
