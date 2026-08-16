"""Published multivariable models, transcribed from their source papers.

Reference data only — nothing here is fitted, resampled, or derived from this
cohort. ``report.render_inferential`` prints it directly above the model this
pipeline fitted, so a reader can put the published odds ratio and ours side by
side and see where the two cohorts disagree.

Keyed by the ``model_id`` used in ``LITERATURE_MODEL_VARIANTS`` (modelling
notebook §02). Every number must be quoted from the paper as printed; leave a
field empty rather than filling it from a related figure, and leave
``performance`` empty when the paper reports no AUC or c-statistic for the
multivariable model. ``column`` is the column in *this* cohort that stands in
for the paper's variable — the honest place to record a mapping that is not
always one-to-one.
"""
from __future__ import annotations

PUBLISHED_MODELS: dict[str, dict] = {
    "radeesri_lekhavat_2023": {
        "citation": (
            "Radeesri K, Lekhavat V. The Role of Pre-Operative MRI for Prediction "
            "of High-Grade Intracranial Meningioma: A Retrospective Study. "
            "Asian Pacific J Cancer Prev 2023;24(3):819–825."
        ),
        "cohort": (
            "210 of 327 consecutive intracranial meningiomas had a pre-operative "
            "MRI available; 50 of the 210 were high-grade. The paper does not state "
            "how many of those entered the multivariable model."
        ),
        "outcome": "High-grade meningioma (WHO grade 2–3, 2021 CNS classification).",
        # Empty on purpose: the paper reports no AUC, c-statistic, sensitivity or
        # specificity for the multivariable model, so there is nothing to compare
        # our validated AUC against.
        "performance": "",
        "terms": [
            {
                "variable": "Necrosis or hemorrhage",
                "meaning": (
                    "A non-enhancing patch inside the tumour (dead tissue), or "
                    "blooming on SWI/GRE showing old bleeding. Scored as one finding."
                ),
                "or": 2.94, "ci_lo": 1.15, "ci_hi": 7.48, "p": 0.024,
                "column": "necrosis_or_hemorrhage",
            },
            {
                "variable": "Hyperostosis",
                "meaning": (
                    "Thickened skull bone next to the tumour. The only protective "
                    "term in the model — it argued for low grade, not high."
                ),
                "or": 0.31, "ci_lo": 0.12, "ci_hi": 0.79, "p": 0.014,
                "column": "hyperostosis",
            },
            {
                "variable": "Brain edema",
                "meaning": (
                    "Peritumoral swelling: bright signal on T2 or FLAIR in the brain "
                    "immediately around the tumour."
                ),
                "or": 2.33, "ci_lo": 1.13, "ci_hi": 4.81, "p": 0.022,
                "column": "perifocal_edema",
            },
        ],
    },
}


def published_model(model_id: str) -> dict | None:
    """The published model for ``model_id``, or None when none is recorded."""
    return PUBLISHED_MODELS.get(str(model_id or ""))
