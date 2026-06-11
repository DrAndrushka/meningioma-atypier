"""Step 07 — eda/inferential targets, predictors, and INFERENTIAL_MODEL_VARIANTS.

resolve_analysis drops missing cols and sets default positive class for binaries.
resolve_inferential_variants filters the model list to what's actually in df.
"""
from __future__ import annotations

import pandas as pd

from inferential import InferentialModelVariant, normalize_inferential_variants

_BINARY_POSITIVE_TARGETS = frozenset(
    {
        "histology_available",
        "progesterone_pos",
        "brain_invasion",
        "hist_necrosis",
        "iv_contrast",
        "tumor_margin",
        "dural_tail",
        "capsular_enhancement",
        "heterogeneous_enhancement",
        "perifocal_edema",
        "mass_effect",
        "calcification",
        "cystic_component",
        "necrosis",
        "hemorrhage",
        "hyperostosis",
        "cortical_destruction",
        "dwi_hyperintensity",
        "t2_hyperintensity",
        "t1_hypointensity",
        "transfalcine_extension",
    }
)


def resolve_analysis(
    df: pd.DataFrame,
    eda_targets: list,
    eda_predictors: list,
    inferential_targets: list,
    inferential_predictors: list,
):
    eda_predictors = [c for c in eda_predictors if c in df.columns]
    inferential_predictors = [c for c in inferential_predictors if c in df.columns]
    eda_positive_class = {t: True for t in eda_targets if t in _BINARY_POSITIVE_TARGETS}
    inferential_positive_class = {
        t: True for t in inferential_targets if t in _BINARY_POSITIVE_TARGETS
    }
    return (
        eda_targets,
        eda_predictors,
        inferential_targets,
        inferential_predictors,
        eda_positive_class,
        inferential_positive_class,
    )


def resolve_inferential_variants(
    df: pd.DataFrame,
    variants: list,
    *,
    default_target: str = "",
) -> list[InferentialModelVariant]:
    """Keep only targets/predictors present in ``df`` for each multivariable model variant."""
    resolved: list[InferentialModelVariant] = []
    for var in normalize_inferential_variants(variants=variants, default_target=default_target):
        if var.target and var.target not in df.columns:
            continue
        preds = tuple(c for c in var.predictors if c in df.columns)
        if not preds:
            continue
        resolved.append(InferentialModelVariant(
            var.model_id, var.title, var.link, var.target, preds,
        ))
    return resolved
