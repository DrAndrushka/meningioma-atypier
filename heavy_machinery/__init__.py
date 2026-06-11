"""Heavy machinery — the meningioma analysis pipeline.

High-grade (WHO 2–3) from MRI + clinical stuff, PSKUS cohort. I run
meningioma.ipynb top to bottom with flat imports (from cleaning import ...).

jupyter notebook meningioma.ipynb   # from this folder
streamlit run app.py                # repo root
python -m pytest                    # repo root

Tables and figures → output/. Calculator models → ../model_artifacts/.
Research tool — don't use clinically without external validation.
"""
