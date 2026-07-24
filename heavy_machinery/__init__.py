"""Heavy machinery — meningioma-atypier analysis pipeline (PSKUS cohort).

Notebooks, ``output/``, and ``app.py`` live at the **repo root**. Library code lives here:

- ``cleaning_phase/`` — schema, cleaning, DDA, MICE, Pandera validation, dataset handoff
- ``modelling_phase/`` — EDA, inferential, bootstrap validation, report, Streamlit calculator
- ``config/`` — pipeline config modules (``load("name")``)
- ``scripts/run_mice.R`` — formal mixed-type MICE engine (R subprocess)
- ``pytests_atypier/`` — pytest suite (``python -m pytest`` from repo root or here)

Generated artifacts (tables, figures, ``output/inferential/model_artifacts/*.json``) are
written under repo-root ``output/``, not inside this package.
"""
