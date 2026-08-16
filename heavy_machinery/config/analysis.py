"""EDA config, literature/experimental model variants, resolve helpers.

``resolve_eda`` filters EDA predictors to columns present in ``df`` and drops
``hide_parent`` sources when ``output_root`` has ``hidden_parent_columns.csv``.
``resolve_inferential_variants`` keeps only targets/predictors present in ``df`` and tags
variants as literature or experimental from the list they came from.
``print_copy_pasteable_columns`` prints a Python list of column names for §03.
``resolve_inferential_targets`` collects outcome columns from merged variants.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from inferential import InferentialModelVariant, normalize_inferential_variants

# One bootstrap-resample count for the whole pipeline (spec 5.3): the
# modelling phase and the cut-point phase must validate with the same
# number, printed once per report in the Methods block.
BOOTSTRAP_RESAMPLES: int = 1000

# Derived cut-point column -> the continuous column it dichotomises. Selection
# skips the child whenever the parent is also a candidate: the parent carries
# more information, and the child would stack the cut-point search's own
# optimism on top of the model's.
CUTPOINT_PARENT: dict[str, str] = {
    "adc_value_le0.72": "adc_value",
    "edema_index_ge0.0617": "edema_index",
    "edema_volume_ge4.76": "edema_volume_cm3",
    "max_diameter_cm_ge3.81": "max_diameter_cm",
    "tumor_volume_ge15.1": "tumor_volume",
}


def _load_hidden_parent_columns(output_root: Path | str | None) -> frozenset[str]:
    """Load ``cleaning/hidden_parent_columns.csv``; empty if missing."""
    if output_root is None:
        return frozenset()
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    if not path.exists():
        return frozenset()
    try:
        tbl = pd.read_csv(path)
    except Exception:
        return frozenset()
    if "column" not in tbl.columns:
        return frozenset()
    return frozenset(str(c) for c in tbl["column"].dropna().tolist())


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
    *,
    output_root: Path | str | None = None,
) -> tuple[list, list]:
    """Filter EDA predictors to ``df`` columns; drop hidden parents when present."""
    hidden = _load_hidden_parent_columns(output_root)
    eda_predictors = [
        c for c in eda_predictors
        if c in df.columns and c not in hidden
    ]
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
    output_root: Path | str | None = None,
) -> list[InferentialModelVariant]:
    """Keep only targets/predictors present in ``df`` for each multivariable model variant.

    Literature vs experimental grouping follows which list each variant came from,
    not the model id string. Hidden parents (``hide_parent=True``) are dropped.
    """
    hidden = _load_hidden_parent_columns(output_root)
    resolved: list[InferentialModelVariant] = []

    def _resolve_one(variants: list, *, is_experimental: bool) -> None:
        if not variants:
            return
        for var in normalize_inferential_variants(variants=variants, default_target=default_target):
            if var.target and var.target not in df.columns:
                continue
            # Dropping a hidden parent is the point of this function. Dropping a
            # name that matches nothing at all is a typo or a rename nobody
            # followed through, and it used to happen in silence: two published
            # models were fitted with five predictors instead of seven, printed
            # their own EPV and AUC for the smaller model, and read as
            # replications of the papers they named.
            unknown = [
                c for c in var.predictors
                if c not in df.columns and c not in hidden
            ]
            if unknown:
                raise KeyError(
                    f"Model {var.model_id!r} names predictor(s) that are not in "
                    f"the cohort and are not hidden parents: {', '.join(unknown)}. "
                    "Rename them or remove them — a model quietly fitted on "
                    "fewer predictors than it declares is not the model it says "
                    "it is."
                )
            # Hiding a parent is deliberate, so this is a warning rather than an
            # error — but it still removes a predictor the model asked for, and
            # the flag that replaced it usually has a different name. Saying
            # nothing is how a model loses a variable without anyone noticing.
            dropped = [c for c in var.predictors if c in hidden]
            if dropped:
                warnings.warn(
                    f"Model {var.model_id!r} names hidden parent(s) "
                    f"{', '.join(dropped)}; they were dropped. Name the derived "
                    "flag that replaced each one if you meant to keep it.",
                    stacklevel=2,
                )
            preds = tuple(
                c for c in var.predictors
                if c in df.columns and c not in hidden
            )
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
