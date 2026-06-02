#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                            ⚙️ SETUP for Greatness ⚙️
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
import streamlit as st
import numpy as np


#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                            🎨 Background Design 🎨
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
st.set_page_config(layout="wide")
st.markdown("""
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
        z-index: 0;}

    [data-testid="stAppViewContainer"] > .main * {position: relative; z-index: 1;}
    </style>
    """, unsafe_allow_html=True)

#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                           Starting intro text
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
starting_title = "🐍 Meningioma Atypier"
starting_text = "> MRI-based prediction tool for identifying atypical meningioma patterns using radiological and histopathological data"

st.title(starting_title)
st.write(starting_text)

#🟦🟦🟦🟦🟦🟦🟦
# High-grade calculator
#🟦🟦🟦🟦🟦🟦🟦

MODEL_PARAMS = {
    "intercept_coef": -0.41,
    "max_diameter_cm": {"coef": 0.31, "z_mu": 4.04, "z_sd": 1.49},
    "capsular_enhancement": {"coef": -0.01, "z_mu": None, "z_sd": None},
    "cystic_component": {"coef": 0.92, "z_mu": None, "z_sd": None},
    "cortical_destruction": {"coef": 0.81, "z_mu": None, "z_sd": None},
    "t1_hypointensity": {"coef": -0.80, "z_mu": None, "z_sd": None},
    "adc_value": {"coef": -0.28, "z_mu": 0.84, "z_sd": 0.13},
}

def _safe_z_denominator(sd: float) -> float:
    if sd is None or not np.isfinite(sd) or sd == 0:
        return 1.0
    return float(sd)

def high_grade_probability(patient: dict, params: dict = MODEL_PARAMS) -> float:
    logit = params["intercept_coef"]

    for predictor, spec in params.items():
        if predictor == "intercept_coef":
            continue

        if predictor not in patient:
            raise KeyError(f"Missing patient value: {predictor}")

        x = patient[predictor]

        if spec["z_mu"] is not None:
            x = (float(x) - spec["z_mu"]) / _safe_z_denominator(spec["z_sd"])
        else:
            x = int(bool(x))

        logit += spec["coef"] * x

    return float(1 / (1 + np.exp(-logit)))

def risk_category(probability: float) -> str:
    if probability < 0.20:
        return "Low estimated probability"
    if probability < 0.50:
        return "Intermediate estimated probability"
    return "High estimated probability"


st.divider()

st.header("🩻 High-grade meningioma probability calculator")
st.caption("Frozen exploratory multivariable logistic model. Not clinically validated.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Tumor measurements")

    max_diameter_cm = st.number_input(
        "Maximum tumor diameter, cm",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1,
    )

    adc_value = st.number_input(
        "ADC value",
        min_value=0.0,
        max_value=3.0,
        value=0.84,
        step=0.01,
    )

with right:
    st.subheader("MRI features")

    capsular_enhancement = st.checkbox("Capsular enhancement")
    cystic_component = st.checkbox("Cystic component")
    cortical_destruction = st.checkbox("Cortical destruction")
    t1_hypointensity = st.checkbox("T1 hypointensity")

patient = {
    "max_diameter_cm": max_diameter_cm,
    "capsular_enhancement": capsular_enhancement,
    "cystic_component": cystic_component,
    "cortical_destruction": cortical_destruction,
    "t1_hypointensity": t1_hypointensity,
    "adc_value": adc_value,
}

probability = high_grade_probability(patient)
percent = probability * 100

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="Estimated high-grade probability",
        value=f"{percent:.1f}%",
    )

with col2:
    st.subheader(risk_category(probability))
    st.progress(min(max(probability, 0.0), 1.0))

st.warning(
    "Research prototype only. This calculator is exploratory and has not been externally validated."
)

with st.expander("Show model details"):
    st.write("Model: high_grade_model_v1")
    st.write("Continuous predictors are z-scored using frozen training-set mean and SD.")
    st.json(MODEL_PARAMS)