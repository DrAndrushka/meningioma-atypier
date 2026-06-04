"""Artifact-driven logistic calculator for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_REQUIRED_ARTIFACT_KEYS = (
    "model_name",
    "target",
    "coefficients",
    "features",
)


def _resolve_artifact_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    candidate = _PROJECT_ROOT / p
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Model artifact not found: {path} (also tried {candidate})")


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON model artifact."""
    resolved = _resolve_artifact_path(path)
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model artifact {resolved}: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ValueError(f"Model artifact must be a JSON object: {resolved}")

    missing = [k for k in _REQUIRED_ARTIFACT_KEYS if k not in artifact]
    if missing:
        raise ValueError(
            f"Model artifact {resolved} is missing required keys: {', '.join(missing)}"
        )

    coefficients = artifact["coefficients"]
    if not isinstance(coefficients, dict) or "const" not in coefficients:
        raise ValueError(
            f"Model artifact {resolved} must include coefficients.const"
        )

    features = artifact["features"]
    if not isinstance(features, list) or not features:
        raise ValueError(f"Model artifact {resolved} must include a non-empty features list")

    for i, feature in enumerate(features):
        if not isinstance(feature, dict) or "name" not in feature:
            raise ValueError(f"Feature entry {i} in {resolved} is malformed")

    return artifact


def sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def risk_category(probability: float) -> str:
    """Map predicted probability to a short risk band label for the UI."""
    if probability < 0.20:
        return "Low estimated probability"
    if probability < 0.50:
        return "Intermediate estimated probability"
    return "High estimated probability"


def _apply_continuous_transform(value: Any, transform: str, feature_name: str) -> float:
    if transform == "raw":
        return float(value)
    if transform == "standardize":
        raise NotImplementedError(
            f"Transform 'standardize' is not implemented yet for {feature_name!r}"
        )
    raise ValueError(f"Unknown transform {transform!r} for feature {feature_name!r}")


def encode_feature_value(feature: dict[str, Any], raw_value: Any) -> dict[str, float]:
    """Map one UI feature value to encoded model variable(s)."""
    ftype = feature.get("type")
    name = feature["name"]

    if ftype == "continuous":
        transform = feature.get("transform", "raw")
        encoded_val = _apply_continuous_transform(raw_value, transform, name)
        return {name: encoded_val}

    if ftype == "binary":
        encoding = feature.get("encoding")
        if not isinstance(encoding, dict):
            raise ValueError(f"Binary feature {name!r} is missing encoding")
        result: dict[str, float] = {}
        level_key = "true" if bool(raw_value) else "false"
        for var_name, level_map in encoding.items():
            if not isinstance(level_map, dict):
                raise ValueError(f"Invalid encoding for binary feature {name!r}")
            if level_key not in level_map:
                raise ValueError(
                    f"Binary feature {name!r} encoding missing level {level_key!r}"
                )
            result[var_name] = float(level_map[level_key])
        return result

    if ftype == "categorical":
        encoding = feature.get("encoding")
        if not isinstance(encoding, dict):
            raise ValueError(f"Categorical feature {name!r} is missing encoding")
        choice = str(raw_value)
        result = {}
        for var_name, level_map in encoding.items():
            if not isinstance(level_map, dict):
                raise ValueError(f"Invalid encoding for categorical feature {name!r}")
            if choice not in level_map:
                raise ValueError(
                    f"Unknown level {choice!r} for {name!r}. "
                    f"Expected one of: {list(level_map.keys())}"
                )
            result[var_name] = float(level_map[choice])
        return result

    raise ValueError(f"Unsupported feature type {ftype!r} for {name!r}")


def build_encoded_features(
    user_inputs: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, float]:
    """Encode all user inputs into model variable names."""
    encoded: dict[str, float] = {}
    coefficients = artifact["coefficients"]

    for feature in artifact["features"]:
        name = feature["name"]
        if name not in user_inputs:
            raise KeyError(f"Missing user input for feature: {name}")
        partial = encode_feature_value(feature, user_inputs[name])
        for var_name, value in partial.items():
            if var_name not in coefficients:
                continue
            encoded[var_name] = value

    return encoded


def predict_from_artifact(
    user_inputs: dict[str, Any],
    artifact: dict[str, Any],
) -> float:
    """Return predicted probability from artifact coefficients and encoded inputs."""
    coefficients = artifact["coefficients"]
    if "const" not in coefficients:
        raise ValueError("Artifact coefficients must include 'const' intercept")

    encoded = build_encoded_features(user_inputs, artifact)
    logit = float(coefficients["const"])

    for var_name, value in encoded.items():
        if var_name == "const":
            continue
        if var_name not in coefficients:
            raise ValueError(
                f"Encoded variable {var_name!r} has no matching coefficient in artifact"
            )
        logit += float(coefficients[var_name]) * float(value)

    return sigmoid(logit)


def render_model_inputs(artifact: dict[str, Any]) -> dict[str, Any]:
    """Render Streamlit inputs from artifact feature definitions."""
    import streamlit as st

    user_inputs: dict[str, Any] = {}
    features = artifact["features"]

    mid = (len(features) + 1) // 2
    left_features = features[:mid]
    right_features = features[mid:]
    left_col, right_col = st.columns(2)

    def _render_field(feature: dict[str, Any]) -> None:
        name = feature["name"]
        label = feature.get("label", name)
        widget = feature.get("input_widget")

        if widget == "selectbox":
            choices = feature.get("choices")
            if not choices:
                raise ValueError(f"selectbox feature {name!r} requires choices")
            labels = {c: c.replace("_", " ") for c in choices}
            user_inputs[name] = st.selectbox(
                label,
                options=list(choices),
                format_func=lambda lv, labels=labels: labels.get(lv, lv),
            )
        elif widget == "number_input":
            unit = feature.get("unit")
            display_label = f"{label} ({unit})" if unit else label
            kwargs: dict[str, Any] = {
                "label": display_label,
                "min_value": float(feature.get("min_value", 0.0)),
                "value": float(feature.get("default", 0.0)),
                "step": float(feature.get("step", 1.0)),
            }
            if "max_value" in feature:
                kwargs["max_value"] = float(feature["max_value"])
            user_inputs[name] = st.number_input(**kwargs)
        elif widget == "checkbox":
            user_inputs[name] = st.checkbox(label, value=bool(feature.get("default", False)))
        else:
            raise ValueError(f"Unsupported input_widget {widget!r} for feature {name!r}")

    for col, field_list in ((left_col, left_features), (right_col, right_features)):
        with col:
            for feature in field_list:
                _render_field(feature)

    return user_inputs


def _find_auc_metric(validation: dict[str, Any]) -> dict[str, Any] | None:
    for metric_row in validation.get("metrics", []):
        if str(metric_row.get("metric", "")).strip().upper() == "AUC":
            return metric_row
    return None


_COMPACT_CHART_FIGSIZE = (3.2, 3.2)
_DEFAULT_AUC_FIGSIZE = (6, 3.5)
_DEFAULT_ROC_FIGSIZE = (6, 6)


def build_auc_comparison_figure(
    validation: dict[str, Any],
    *,
    compact: bool = False,
) -> plt.Figure | None:
    """Bar chart: apparent vs optimism-corrected AUC from artifact validation metrics."""
    auc_row = _find_auc_metric(validation)
    if auc_row is None:
        return None

    chart_cfg = validation.get("auc_chart", {})
    series_cfg = chart_cfg.get("series")
    if series_cfg:
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for entry in series_cfg:
            field = entry.get("value_field")
            if field not in auc_row:
                raise ValueError(
                    f"AUC chart references missing field {field!r} in validation metrics"
                )
            labels.append(str(entry.get("label", field)))
            values.append(float(auc_row[field]))
            colors.append(str(entry.get("color", "#555555")))
    else:
        labels = ["Apparent", "Optimism-corrected"]
        values = [float(auc_row["apparent"]), float(auc_row["optimism_corrected"])]
        colors = ["#2E86AB", "#E94F37"]

    title = chart_cfg.get("title", "AUC before and after internal validation")
    figsize = _COMPACT_CHART_FIGSIZE if compact else _DEFAULT_AUC_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_pos, labels, rotation=25 if compact else 0, ha="right" if compact else "center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC", fontsize=8 if compact else 10)
    ax.set_title(title, fontsize=8 if compact else 11)
    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1)
    ax.grid(axis="y", alpha=0.25)
    if compact:
        ax.tick_params(labelsize=7)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7 if compact else 10,
        )
    fig.tight_layout()
    return fig


def build_roc_validation_figure(
    validation: dict[str, Any],
    *,
    compact: bool = False,
) -> plt.Figure | None:
    """ROC plot from artifact curve coordinates; legend AUC from curve or metrics."""
    roc_cfg = validation.get("roc_curves")
    if not roc_cfg:
        return None

    curves = roc_cfg.get("curves")
    if not curves:
        return None

    auc_row = _find_auc_metric(validation)
    figsize = _COMPACT_CHART_FIGSIZE if compact else _DEFAULT_ROC_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize)
    plotted = False

    for curve in curves:
        fpr = curve.get("fpr")
        tpr = curve.get("tpr")
        if not fpr or not tpr:
            continue
        if len(fpr) != len(tpr):
            raise ValueError(
                f"ROC curve {curve.get('series', '?')!r} has mismatched fpr/tpr lengths"
            )
        label = str(curve.get("label", curve.get("series", "Model")))
        if "auc" in curve:
            auc_val = float(curve["auc"])
        elif curve.get("auc_from_metrics") and auc_row:
            auc_val = float(auc_row.get("optimism_corrected", np.nan))
            label = f"{label} (AUC = {auc_val:.3f})"
        else:
            auc_val = None
        if auc_val is not None and "AUC" not in label:
            label = f"{label} (AUC = {auc_val:.3f})"
        color = curve.get("color", None)
        linewidth = 1.5 if compact else 2
        ax.plot(fpr, tpr, color=color, linewidth=linewidth, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Coinflip")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR", fontsize=8 if compact else 10)
    ax.set_ylabel("TPR", fontsize=8 if compact else 10)
    ax.set_title(roc_cfg.get("title", "ROC curve"), fontsize=8 if compact else 11)
    legend_font = 6 if compact else 9
    ax.legend(loc="lower right", fontsize=legend_font)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    if compact:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def render_auc_validation_charts(validation: dict[str, Any]) -> None:
    """Show validation charts in two equal columns with a narrow spacer between."""
    import streamlit as st

    auc_fig = build_auc_comparison_figure(validation, compact=True)
    roc_fig = build_roc_validation_figure(validation, compact=True)
    if auc_fig is None and roc_fig is None:
        return

    _, col_auc, _col_gap, col_roc,__ = st.columns([0.5, 5, 0.8, 5, 0.5], gap="small")
    if auc_fig is not None:
        with col_auc:
            st.pyplot(auc_fig, clear_figure=True, use_container_width=True)
        plt.close(auc_fig)
    if roc_fig is not None:
        with col_roc:
            st.pyplot(roc_fig, clear_figure=True, use_container_width=True)
        plt.close(roc_fig)


def render_validation_expander(artifact: dict[str, Any]) -> None:
    """Methodology, validation metrics, and coefficients (expander content only)."""
    import streamlit as st

    st.markdown(f"**Model:** {artifact['model_name']}")
    st.markdown(f"**Target:** `{artifact['target']}`")
    st.markdown(f"**Model type:** {artifact.get('model_type', '—')}")

    if "n" in artifact:
        st.markdown(f"**Development cohort n:** {artifact['n']}")
    if "events" in artifact:
        st.markdown(f"**Events:** {artifact['events']}")
    if "outcome_definition" in artifact:
        st.markdown(f"**Outcome definition:** {artifact['outcome_definition']}")

    st.markdown("**Predictors / features**")
    feature_rows = [
        {
            "name": f["name"],
            "label": f.get("label", f["name"]),
            "type": f.get("type", ""),
            "widget": f.get("input_widget", ""),
        }
        for f in artifact["features"]
    ]
    st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

    if "missing_data_policy" in artifact:
        st.markdown("**Missing data policy**")
        st.write(artifact["missing_data_policy"])

    processing = artifact.get("coefficient_processing")
    if processing:
        st.markdown("**Coefficient processing**")
        if processing.get("shrinkage_applied"):
            factor = processing.get("shrinkage_factor")
            st.write(
                f"Shrinkage applied"
                + (f" (factor {factor})" if factor is not None else "")
                + "."
            )
        if processing.get("intercept_recalibrated"):
            st.write("Intercept recalibrated to match cohort prevalence.")
        if processing.get("notes"):
            st.write(processing["notes"])

    validation = artifact.get("validation")
    if validation:
        st.markdown("**Internal validation**")
        if validation.get("method"):
            st.write(validation["method"])
        if validation.get("bootstrap_resamples"):
            st.caption(f"Bootstrap resamples: {validation['bootstrap_resamples']}")

        metrics = validation.get("metrics")
        if metrics:
            st.dataframe(
                pd.DataFrame(metrics),
                use_container_width=True,
                hide_index=True,
            )

        if _find_auc_metric(validation) is not None:
            st.markdown("**Discrimination (AUC)**")
            render_auc_validation_charts(validation)

        if "baseline_brier" in validation:
            st.markdown(f"**Baseline Brier score (prevalence-only):** {validation['baseline_brier']}")
        if validation.get("interpretation"):
            st.markdown("**Validation interpretation**")
            st.write(validation["interpretation"])

    st.markdown("**Model coefficients**")
    coef_rows = [
        {"coefficient": name, "value": value}
        for name, value in sorted(artifact["coefficients"].items())
    ]
    st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)

    if "clinical_note" in artifact:
        st.markdown("**Clinical note**")
        st.info(artifact["clinical_note"])


def render_model_calculator(artifact_path: str) -> None:
    """Load artifact, render inputs, show predicted risk, and methodology expander."""
    import streamlit as st

    try:
        artifact = load_model_artifact(artifact_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.divider()
    st.header(artifact["model_name"])

    user_inputs = render_model_inputs(artifact)
    probability = predict_from_artifact(user_inputs, artifact)

    st.divider()
    metric_col, risk_col = st.columns([1, 2])

    with metric_col:
        st.metric(
            f"Predicted P({artifact['target']})",
            f"{probability * 100:.1f}%",
        )

    with risk_col:
        st.subheader(risk_category(probability))
        st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("How this calculator was built and validated", expanded=False):
        render_validation_expander(artifact)
