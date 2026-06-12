"""Heavy machinery — meningioma-atypier analysis pipeline (PSKUS cohort).

Two notebooks (run from ``heavy_machinery/``):

1. ``meningioma-cleaning.ipynb`` — cohort cleaning, schema, MICE → ``output/datasets/``
2. ``meningioma-modelling.ipynb`` — DDA, EDA, multivariable models, HTML report

Also: ``streamlit run app.py`` (repo root), ``python -m pytest`` (repo root).

Pipeline tables/figures → ``output/``. Calculator JSON → ``../model_artifacts/``.
Research tool — not for clinical use without external validation.
"""
