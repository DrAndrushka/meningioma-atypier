"""Streamlit calculator entry point for meningioma-atypier.

Loads ``model_artifacts/high_grade_model.json`` by default via
``render_model_calculator``. Run from repo root: ``streamlit run app.py``.
"""

from __future__ import annotations

import streamlit as st

from heavy_machinery.model_calculator import render_model_calculator

st.set_page_config(page_title="Meningioma Atypier", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background: #0b0f14;}
    [data-testid="stAppViewContainer"] > .main {background: transparent;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    [data-testid="stAppViewContainer"]::before {
        content: "🧠";
        position: fixed;
        top: 55%;
        left: 50%;
        right: 18%;
        transform: translate(-50%, -50%);
        font-size: min(90vw, 90vh);
        opacity: 0.4;
        filter: grayscale(0.5) brightness(0.3) blur(9px);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stAppViewContainer"] > .main * {position: relative; z-index: 1;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐍 Meningioma Atypier")
st.write(
    "MRI-based prediction tool for high-grade meningioma (WHO grade 2–3), "
    "using a multivariable logistic model defined in a portable JSON artifact."
)

render_model_calculator()
