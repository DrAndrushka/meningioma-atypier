"""EDA config, literature/experimental model variants, resolve helpers.

``resolve_eda`` filters EDA predictors to columns present in ``df``.
``resolve_inferential_variants`` keeps only targets/predictors present in ``df`` and tags
variants as literature or experimental from the list they came from.
``print_copy_pasteable_columns`` prints a Python list of column names for §03.
``resolve_inferential_targets`` collects outcome columns from merged variants.
"""
from __future__ import annotations

import pandas as pd

from inferential import InferentialModelVariant, normalize_inferential_variants

# One bootstrap-resample count for the whole pipeline (spec 5.3): the
# modelling phase and the threshold phase must validate with the same
# number, printed once per report in the Methods block.
BOOTSTRAP_RESAMPLES: int = 1000


def print_copy_pasteable_columns(
    df: pd.DataFrame,
    *,
    label: str = "COLUMNS",
) -> None:
    """Print a Python list of ``df`` column names for pasting into notebook config cells."""
    cols = list(df.columns)
    print(f"# Copy-paste into EDA_PREDICTORS / model variant lists ({len(cols)} columns)")
    print(f"{label} = [")
    for col in cols:
        print(f"    {col!r},")
    print("]")


def resolve_eda(
    df: pd.DataFrame,
    eda_targets: list,
    eda_predictors: list,
) -> tuple[list, list]:
    """Filter EDA predictors to ``df`` columns."""
    eda_predictors = [c for c in eda_predictors if c in df.columns]
    return eda_targets, eda_predictors


def inferential_targets_from_variants(
    variants: list[InferentialModelVariant],
) -> list[str]:
    """Unique outcome columns referenced by multivariable model variants (stable order)."""
    seen: list[str] = []
    for var in variants:
        if var.target and var.target not in seen:
            seen.append(var.target)
    return seen


def resolve_inferential_targets(
    df: pd.DataFrame,
    variants: list[InferentialModelVariant],
) -> list[str]:
    """Outcome columns from resolved model variants, filtered to ``df`` columns."""
    return [t for t in inferential_targets_from_variants(variants) if t in df.columns]


def resolve_inferential_variants(
    df: pd.DataFrame,
    literature: list,
    experimental: list | None = None,
    *,
    default_target: str = "",
) -> list[InferentialModelVariant]:
    """Keep only targets/predictors present in ``df`` for each multivariable model variant.

    Literature vs experimental grouping follows which list each variant came from,
    not the model id string.
    """
    resolved: list[InferentialModelVariant] = []

    def _resolve_one(variants: list, *, is_experimental: bool) -> None:
        for var in normalize_inferential_variants(variants=variants, default_target=default_target):
            if var.target and var.target not in df.columns:
                continue
            preds = tuple(c for c in var.predictors if c in df.columns)
            if not preds:
                continue
            resolved.append(InferentialModelVariant(
                var.model_id, var.title, var.link, var.target, preds,
                experimental=is_experimental,
            ))

    _resolve_one(literature, is_experimental=False)
    _resolve_one(experimental or [], is_experimental=True)
    resolved.sort(
        key=lambda var: (1 if var.experimental else 0, var.model_id),
    )
    return resolved
