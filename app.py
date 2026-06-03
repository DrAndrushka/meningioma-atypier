"""Streamlit calculator — auto-built from the latest inferential model CSV."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from heavy_machinery.atypier_calculator import (
    BinaryInput,
    CategoricalInput,
    ContinuousInput,
    find_model_csv,
    load_model_from_csv,
    model_to_dict,
    predict_probability,
    risk_category,
)

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
    "using the latest multivariable logistic model from the analysis pipeline."
)


@st.cache_data(show_spinner="Loading model…")
def _load_model(model_path: str):
    return load_model_from_csv(Path(model_path))


try:
    model_path = find_model_csv()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

model = _load_model(str(model_path))
inputs = model.ui_inputs()

st.divider()
st.header("🩻 High-grade meningioma probability calculator")
st.caption(
    f"Target: **{model.target}** · "
    f"Loaded from `{model_path.name}` · "
    "Exploratory model — not clinically validated."
)

patient: dict = {}
mid = (len(inputs) + 1) // 2
left_fields = inputs[:mid]
right_fields = inputs[mid:]

left_col, right_col = st.columns(2)

for col, field_list in ((left_col, left_fields), (right_col, right_fields)):
    with col:
        for field in field_list:
            if isinstance(field, ContinuousInput):
                patient[field.name] = st.number_input(
                    field.label,
                    min_value=field.min_value,
                    max_value=field.max_value,
                    value=field.default,
                    step=field.step,
                )
            elif isinstance(field, CategoricalInput):
                labels = {
                    level: level.replace("_", " ")
                    for level in field.options
                }
                choice = st.selectbox(
                    field.label,
                    options=list(field.options),
                    index=list(field.options).index(field.default),
                    format_func=lambda lv, labels=labels: labels.get(lv, lv),
                )
                patient[field.name] = choice
                st.caption(f"Reference category in model: {field.reference.replace('_', ' ')}")
            elif isinstance(field, BinaryInput):
                patient[field.name] = st.checkbox(field.label, value=field.default)

probability = predict_probability(patient, model)
percent = probability * 100

st.divider()
metric_col, risk_col = st.columns([1, 2])

with metric_col:
    st.metric("Estimated high-grade probability", f"{percent:.1f}%")

with risk_col:
    st.subheader(risk_category(probability))
    st.progress(min(max(probability, 0.0), 1.0))

st.warning(
    "Research prototype only. This calculator is exploratory and has "
    "not been externally validated."
)

with st.expander("Model details"):
    meta_path = model_path.with_name("high_grade__calculator.json")
    st.write(f"**Model CSV:** `{model.source_path}`")
    if meta_path.is_file():
        st.write(f"**Calculator meta:** `{meta_path}`")
    st.write(f"**Intercept (log-odds):** {model.intercept:.3f}")
    st.write(f"**Intercept OR:** {model.raw_table['intercept_or'].iloc[0]}")
    st.dataframe(model.summary_rows(), use_container_width=True, hide_index=True)
    st.json(model_to_dict(model))
