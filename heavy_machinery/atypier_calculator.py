"""High-grade meningioma probability calculator — built from inferential output."""

from __future__ import annotations

import ast
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Notebook runs with cwd=heavy_machinery; Streamlit runs from meningioma-atypier/.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from schema_infer import ColSpec

TARGET = "high_grade"
MODEL_FILENAME = f"{TARGET}__multivariable.csv"
META_FILENAME = f"{TARGET}__calculator.json"

_FIELD_LABELS: dict[str, str] = {
    "side": "Tumor side",
    "tumor_location": "Tumor location",
    "tumor_volume": "Tumor volume (cm³)",
    "tumor_episode": "Tumor episode",
    "edema_volume_cm3": "Perifocal edema volume (cm³)",
    "cystic_component": "Cystic component",
    "adc_value": "ADC value",
    "max_diameter_cm": "Maximum tumor diameter (cm)",
    "age_bins": "Age group",
    "cortical_destruction": "Cortical destruction",
}


def _safe_z_denominator(sd: float | None) -> float:
    if sd is None or not np.isfinite(sd) or sd == 0:
        return 1.0
    return float(sd)


def _parse_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s in {"∞", "inf", "nan", "None"}:
            return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _parse_levels_value(val: Any) -> list | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None


def _humanize(name: str) -> str:
    return _FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def _humanize_level(level: str) -> str:
    return level.replace("_", " ")


@dataclass(frozen=True)
class ContinuousInput:
    name: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float


@dataclass(frozen=True)
class BinaryInput:
    name: str
    label: str
    default: bool = False


@dataclass(frozen=True)
class CategoricalInput:
    name: str
    label: str
    options: tuple[str, ...]
    reference: str
    default: str


UIInput = ContinuousInput | BinaryInput | CategoricalInput


@dataclass
class LogisticModel:
    target: str
    intercept: float
    source_path: Path
    continuous: dict[str, dict[str, float]] = field(default_factory=dict)
    binary: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def model_id(self) -> str:
        return f"{self.target}__{self.source_path.stat().st_mtime_ns}"

    def ui_inputs(self) -> list[UIInput]:
        fields: list[UIInput] = []
        for name, spec in sorted(self.continuous.items()):
            mu = spec["z_mu"]
            sd = max(spec["z_sd"], 1.0)
            step = 0.1 if sd < 5 else 1.0
            fields.append(ContinuousInput(
                name=name,
                label=_humanize(name),
                default=round(mu, 2),
                min_value=round(max(0.0, mu - 3 * sd), 2),
                max_value=round(mu + 3 * sd, 2),
                step=step,
            ))
        for name, levels in sorted(self.categorical.items()):
            ref = levels["reference"]
            opts = tuple(levels["levels"])
            fields.append(CategoricalInput(
                name=name,
                label=_humanize(name),
                options=opts,
                reference=ref,
                default=ref,
            ))
        for name in sorted(self.binary):
            fields.append(BinaryInput(name=name, label=_humanize(name)))
        return fields

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, spec in sorted(self.continuous.items()):
            rows.append({
                "predictor": name,
                "kind": "continuous (z-scored)",
                "coef": spec["coef"],
                "or": round(math.exp(spec["coef"]), 2),
                "z_mu": spec["z_mu"],
                "z_sd": spec["z_sd"],
            })
        for name, spec in sorted(self.categorical.items()):
            ref = spec["reference"]
            rows.append({
                "predictor": name,
                "kind": "categorical",
                "reference": ref,
                "levels": ", ".join(spec["levels"]),
            })
            for level, coef in sorted(spec["dummies"].items()):
                rows.append({
                    "predictor": f"  → {_humanize_level(level)} vs {ref}",
                    "kind": "dummy",
                    "coef": coef,
                    "or": round(math.exp(coef), 2),
                })
        for name, coef in sorted(self.binary.items()):
            rows.append({
                "predictor": name,
                "kind": "binary",
                "coef": coef,
                "or": round(math.exp(coef), 2),
            })
        return rows


def _meta_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(META_FILENAME)


def _output_roots(output_root: Path | str | None) -> list[Path]:
    roots: list[Path] = []
    if output_root is not None:
        roots.append(Path(output_root))
    pkg = Path(__file__).resolve().parent
    roots.extend([
        pkg / "output",
        pkg.parent / "output",
        Path.cwd() / "heavy_machinery" / "output",
        Path.cwd() / "output",
    ])
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def find_model_csv(
    *,
    output_root: Path | str | None = None,
    filename: str = MODEL_FILENAME,
) -> Path:
    """Locate the newest multivariable model CSV under known output roots."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in _output_roots(output_root):
        path = (root / "inferential" / "tables" / filename).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            candidates.append(path)

    if not candidates:
        searched = ", ".join(
            str(r / "inferential" / "tables" / filename) for r in _output_roots(output_root)
        )
        raise FileNotFoundError(
            f"No {filename} found. Run §11 inferential in the notebook first. "
            f"Searched: {searched}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _schema_from_summary(path: Path) -> dict[str, ColSpec]:
    df = pd.read_csv(path)
    schema: dict[str, ColSpec] = {}
    for row in df.itertuples(index=False):
        col = str(row.column)
        kind = str(row.kind)
        levels = _parse_levels_value(getattr(row, "levels", None))
        schema[col] = ColSpec(
            name=col,
            kind=kind,  # type: ignore[arg-type]
            ordered_levels=levels if kind == "ordinal" else None,
        )
    return schema


def _predictors_from_csv(
    df: pd.DataFrame,
    schema: dict[str, ColSpec] | None = None,
) -> list[str]:
    """Recover original predictor names from design-matrix column names."""
    names = [str(x) for x in df["predictor_col"]]
    name_set = set(names)
    schema_cols = set(schema) if schema else None
    predictors: set[str] = set()
    assigned: set[str] = set()

    for name in names:
        z_mu = df.loc[df["predictor_col"] == name, "z_mu"].iloc[0]
        if _parse_float(z_mu) is not None:
            predictors.add(name)
            assigned.add(name)

    for name in names:
        if name in assigned:
            continue
        parts = name.split("_")
        for i in range(len(parts) - 1, 0, -1):
            base = "_".join(parts[:i])
            if schema_cols is not None and base not in schema_cols:
                continue
            cluster = [
                n for n in name_set
                if n.startswith(f"{base}_")
                and "_".join(n.split("_")[:i]) == base
            ]
            if name not in cluster:
                continue
            if len(cluster) >= 2 or len(cluster) == 1:
                predictors.add(base)
                assigned.update(cluster)
                break

    for name in names:
        if name not in assigned:
            predictors.add(name)
    return sorted(predictors)


def regenerate_calculator_meta(
    csv_path: Path | str,
    *,
    output_root: Path | str | None = None,
) -> Path:
    """Rebuild calculator JSON from cleaned data + schema summary + model CSV."""
    from inferential import export_calculator_meta

    csv_path = Path(csv_path)
    root = csv_path.parents[2] if csv_path.parent.name == "tables" else Path(output_root or csv_path.parent)
    cleaned = root / "cleaning" / "cleaned.csv"
    schema_path = root / "schema" / "schema_summary.csv"
    if not cleaned.is_file():
        raise FileNotFoundError(f"Missing cleaned cohort: {cleaned}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing schema summary: {schema_path}")

    df = pd.read_csv(cleaned)
    schema = _schema_from_summary(schema_path)
    model_df = pd.read_csv(csv_path)
    predictors = _predictors_from_csv(model_df, schema)
    meta = export_calculator_meta([df], schema, predictors, model_df)
    meta_path = _meta_path_for_csv(csv_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def _model_from_meta(meta: dict[str, Any], source_path: Path) -> LogisticModel:
    continuous: dict[str, dict[str, float]] = {}
    binary: dict[str, float] = {}
    categorical: dict[str, dict[str, Any]] = {}

    for term in meta.get("terms", []):
        kind = term["kind"]
        name = term["name"]
        if kind == "continuous":
            continuous[name] = {
                "coef": float(term["coef"]),
                "z_mu": float(term["z_mu"]),
                "z_sd": float(term["z_sd"]),
            }
        elif kind == "binary":
            binary[name] = float(term["coef"])
        elif kind == "categorical":
            categorical[name] = {
                "reference": str(term["reference"]),
                "levels": [str(x) for x in term["levels"]],
                "dummies": {str(k): float(v) for k, v in term["dummies"].items()},
            }

    raw_table = pd.read_csv(source_path) if source_path.is_file() else pd.DataFrame()
    return LogisticModel(
        target=str(meta.get("target", TARGET)),
        intercept=float(meta["intercept"]),
        source_path=source_path.resolve(),
        continuous=continuous,
        binary=binary,
        categorical=categorical,
        raw_table=raw_table,
    )


def load_model_from_csv(path: Path | str) -> LogisticModel:
    """Load model from multivariable CSV, using/regenerating companion JSON metadata."""
    csv_path = Path(path)
    meta_path = _meta_path_for_csv(csv_path)

    if not meta_path.is_file() or meta_path.stat().st_mtime < csv_path.stat().st_mtime:
        try:
            regenerate_calculator_meta(csv_path)
        except FileNotFoundError:
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"Missing {meta_path.name} and could not regenerate it. "
                    "Re-run §11 inferential in the notebook."
                ) from None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _model_from_meta(meta, csv_path)


def load_latest_model(output_root: Path | str | None = None) -> LogisticModel:
    """Load the most recently written high-grade multivariable model."""
    return load_model_from_csv(find_model_csv(output_root=output_root))


def predict_probability(patient: dict[str, Any], model: LogisticModel) -> float:
    """Return P(high_grade) on the logit scale used in inferential.py."""
    logit = model.intercept

    for name, spec in model.continuous.items():
        if name not in patient:
            raise KeyError(f"Missing value for continuous predictor: {name}")
        raw = float(patient[name])
        z = (raw - spec["z_mu"]) / _safe_z_denominator(spec["z_sd"])
        logit += spec["coef"] * z

    for name, coef in model.binary.items():
        if name not in patient:
            raise KeyError(f"Missing value for binary predictor: {name}")
        logit += coef * (1.0 if patient[name] else 0.0)

    for name, spec in model.categorical.items():
        if name not in patient:
            raise KeyError(f"Missing value for categorical predictor: {name}")
        level = str(patient[name])
        if level == spec["reference"]:
            continue
        dummy_coef = spec["dummies"].get(level)
        if dummy_coef is None:
            raise ValueError(
                f"Unknown level {level!r} for {name}. "
                f"Expected one of: {list(spec['levels'])}"
            )
        logit += dummy_coef

    return float(1.0 / (1.0 + math.exp(-logit)))


def high_grade_probability(
    patient: dict[str, Any],
    model: LogisticModel | None = None,
) -> float:
    if model is None:
        model = load_latest_model()
    return predict_probability(patient, model)


def risk_category(probability: float) -> str:
    if probability < 0.20:
        return "Low estimated probability"
    if probability < 0.50:
        return "Intermediate estimated probability"
    return "High estimated probability"


def model_to_dict(model: LogisticModel) -> dict[str, Any]:
    """JSON-serialisable snapshot for debugging / expander display."""
    return {
        "target": model.target,
        "intercept": model.intercept,
        "source": str(model.source_path),
        "continuous": model.continuous,
        "binary": model.binary,
        "categorical": model.categorical,
    }
