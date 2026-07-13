"""Shared plotting style + human-readable labels for figures and the report.

Two jobs:

1. ``apply_plot_style()`` — one consistent, publication-quality matplotlib
   look (readable fonts, light grid, no top/right spines, tight SVG export)
   so exported figures don't clip or overlap.
2. ``prettify_label`` / ``prettify_caption`` — turn machine column names and
   figure file stems (``high_grade__experimental_model_1__forest``) into clean
   captions (``High-grade — Experimental model 1 — Forest plot``). No more
   double-underscores in the report or figure titles.
"""

from __future__ import annotations

# Consistent palette (kept close to the colors already used across modules).
PALETTE = {
    "primary": "#3b7ddd",
    "accent": "#e76f51",
    "good": "#2a9d8f",
    "bad": "#e76f51",
    "neutral": "#888888",
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


def apply_plot_style() -> None:
    """Apply a consistent, high-quality matplotlib style for SVG export."""
    import matplotlib as mpl

    mpl.rcParams.update({
        # Crisp export; SVG keeps text as text, dpi guards any raster insets.
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        # (modules call tight_layout()/bbox_inches="tight" explicitly, so we
        # leave autolayout off to avoid double-layout warnings.)
        # Readable typography.
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        # Clean frame + light grid.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })
