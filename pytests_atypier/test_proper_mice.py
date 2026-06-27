"""Tests for formal mixed-type MICE (proper_mice_impute) and the RF relabel.

The Python unit tests never require R: the R subprocess is replaced by a fake
runner that fills missing cells deterministically, so the full Python
orchestration (restore, reattach, callback, validation, diagnostics, manifest)
is exercised offline. One real-R integration smoke test runs only when Rscript
and the ``mice`` package are available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import missingness_resolution as mr
from missingness_resolution import (
    MICE_ROW_ID,
    mice_method_for_kind,
    proper_mice_impute,
    rf_chained_impute,
)
from schema_infer import ColSpec


# ---------------------------------------------------------------------------
# Fixtures: a small mixed-type cohort with derived columns + analysis outcome
# ---------------------------------------------------------------------------

def _make_cohort() -> tuple[pd.DataFrame, dict[str, ColSpec], dict, object]:
    df = pd.DataFrame({
        "id": ["p0", "p1", "p2", "p3", "p4", "p5"],
        "age": pd.array([45.0, 55.0, np.nan, 65.0, np.nan, 50.0], dtype="Float64"),
        "sex": pd.Categorical(
            ["male", "female", None, "male", "female", None],
            categories=["female", "male"], ordered=False,
        ),
        "grade": pd.Categorical(
            ["1", "2", "1", None, "3", "2"],
            categories=["1", "2", "3"], ordered=True,
        ),
        "event": pd.array([True, False, None, True, None, False], dtype="boolean"),
        "who_grade": pd.Categorical(
            ["1", "2", "3", "1", "2", "3"],
            categories=["1", "2", "3"], ordered=True,
        ),
        "high_grade": pd.array(
            [False, True, True, False, True, True], dtype="boolean"
        ),
        "age_bins": pd.Categorical(
            ["<50", "50+", None, "50+", None, "50+"],
            categories=["<50", "50+"], ordered=True,
        ),
    })
    df.index = [f"row{i}" for i in range(len(df))]

    schema = {
        "id": ColSpec("id", "id", keep=False),
        "age": ColSpec("age", "continuous"),
        "sex": ColSpec("sex", "nominal"),
        "grade": ColSpec("grade", "ordinal", ordered_levels=["1", "2", "3"]),
        "event": ColSpec("event", "binary"),
        "who_grade": ColSpec("who_grade", "ordinal", ordered_levels=["1", "2", "3"]),
        "high_grade": ColSpec("high_grade", "binary"),
        "age_bins": ColSpec("age_bins", "ordinal", ordered_levels=["<50", "50+"]),
    }
    deps = {"high_grade": ["who_grade"], "age_bins": ["age"]}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        age = pd.to_numeric(out["age"], errors="coerce").astype(float)
        bins = pd.cut(age, [-np.inf, 50, np.inf], labels=["<50", "50+"], right=False)
        out["age_bins"] = pd.Categorical(
            bins, categories=["<50", "50+"], ordered=True,
        )
        if "high_grade" not in out.columns:
            wg = pd.to_numeric(out["who_grade"].astype("object"), errors="coerce")
            out["high_grade"] = ((wg == 2) | (wg == 3)).astype("boolean")
        return out

    return df, schema, deps, transform


def _fake_r_run(rscript: str, run_dir: Path) -> None:
    """Stand-in for run_mice.R: fill missing cells, write the same artifacts."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "mice_spec.json").read_text())
    raw = pd.read_csv(run_dir / "input.csv", dtype=str)
    cols = spec["columns"]
    m = spec["m"]
    for d in range(1, m + 1):
        out = raw.copy()
        for col in spec["vars_with_missing"]:
            kind = spec["kinds"][col]
            mask = out[col].isna()
            if not mask.any():
                continue
            if kind in ("continuous", "count"):
                obs = pd.to_numeric(out[col], errors="coerce").dropna()
                out.loc[mask, col] = str(float(obs.mean()) if len(obs) else 0.0)
            elif kind == "binary":
                levels = spec["levels"][col]
                out.loc[mask, col] = levels[d % len(levels)]
            else:
                out.loc[mask, col] = spec["levels"][col][0]
        out.to_csv(run_dir / f"imputed_{d:03d}.csv", index=False)

    pd.DataFrame({
        "column": cols, "method": [spec["methods"][c] for c in cols],
    }).to_csv(run_dir / "methods.csv", index=False)
    pd.DataFrame({"_": []}).to_csv(run_dir / "predictor_matrix.csv", index=False)
    pd.DataFrame(columns=["it", "im", "dep", "meth", "out"]).to_csv(
        run_dir / "logged_events.csv", index=False,
    )
    (run_dir / "r_session.json").write_text(json.dumps({
        "r_version": "fake R", "mice_version": "0.0.0", "logged_events_count": 0,
    }))


@pytest.fixture
def fake_r(monkeypatch):
    monkeypatch.setattr(mr, "_check_r_environment", lambda: "Rscript")
    monkeypatch.setattr(mr, "_run_r_mice", _fake_r_run)


# ---------------------------------------------------------------------------
# 1. Method mapping
# ---------------------------------------------------------------------------

def test_method_mapping_by_kind():
    assert mice_method_for_kind("continuous") == "pmm"
    assert mice_method_for_kind("count") == "pmm"
    assert mice_method_for_kind("binary") == "logreg"
    assert mice_method_for_kind("nominal") == "polyreg"
    assert mice_method_for_kind("ordinal") == "polr"
    with pytest.raises(ValueError):
        mice_method_for_kind("id")


# ---------------------------------------------------------------------------
# 2-5. Column classification + spec serialization (no R needed)
# ---------------------------------------------------------------------------

def test_classification_excludes_derived_and_duplicate_source():
    df, schema, deps, _ = _make_cohort()
    parts = mr._classify_mice_columns(
        df, schema, analysis_outcome="high_grade",
        derived_dependencies=deps, predictor_exclusions=(),
    )
    # non-outcome derived excluded from R matrix
    assert "age_bins" not in parts["r_columns"]
    assert "age_bins" in parts["non_outcome_derived"]
    # duplicate source of the derived analysis outcome excluded
    assert "who_grade" not in parts["r_columns"]
    assert "who_grade" in parts["outcome_source_cols"]
    # structural id excluded
    assert "id" not in parts["r_columns"]
    # analysis outcome retained AND usable as a predictor
    assert "high_grade" in parts["r_columns"]
    assert "high_grade" not in parts["non_predictor_cols"]


def test_spec_serializes_methods_and_dependencies():
    df, schema, deps, _ = _make_cohort()
    parts = mr._classify_mice_columns(
        df, schema, analysis_outcome="high_grade",
        derived_dependencies=deps, predictor_exclusions=(),
    )
    spec = mr._build_mice_spec(
        df, schema, parts, m=3, max_iter=5, random_state=42,
        analysis_outcome="high_grade", derived_dependencies=deps,
        predictor_exclusions=(), input_sha256="deadbeef",
    )
    assert spec["methods"]["age"] == "pmm"
    assert spec["methods"]["sex"] == "polyreg"
    assert spec["methods"]["grade"] == "polr"
    assert spec["methods"]["event"] == "logreg"
    assert spec["methods"]["high_grade"] == ""  # outcome never imputed
    assert spec["derived_dependencies"] == {
        "high_grade": ["who_grade"], "age_bins": ["age"],
    }
    assert spec["analysis_outcome"] == "high_grade"
    assert set(spec["vars_with_missing"]) == {"age", "sex", "grade", "event"}
    # round-trips through JSON
    assert json.loads(json.dumps(spec))["methods"]["age"] == "pmm"


# ---------------------------------------------------------------------------
# 6-13. Full orchestration via the fake R runner
# ---------------------------------------------------------------------------

def test_proper_mice_callback_called_once_per_frame(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    calls = {"n": 0}

    def counting(frame):
        calls["n"] += 1
        return transform(frame)

    frames = proper_mice_impute(
        df, schema, m=3, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=counting,
    )
    assert len(frames) == 3
    assert calls["n"] == 3


def test_proper_mice_observed_unchanged_and_missing_filled(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    frames = proper_mice_impute(
        df, schema, m=2, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    for frame in frames:
        # imputable columns fully filled
        for col in ("age", "sex", "grade", "event"):
            assert frame[col].isna().sum() == 0
        # observed cells preserved
        obs = df["age"].notna()
        assert np.allclose(
            frame.loc[obs, "age"].astype(float),
            df.loc[obs, "age"].astype(float),
        )
        assert frame.loc[df["sex"].notna(), "sex"].astype("object").tolist() == \
            df.loc[df["sex"].notna(), "sex"].astype("object").tolist()


def test_proper_mice_dtype_and_row_identity(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    frames = proper_mice_impute(
        df, schema, m=2, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    frame = frames[0]
    assert list(frame.columns) == list(df.columns)
    assert frame.index.equals(df.index)
    assert isinstance(frame["age"].dtype, pd.Float64Dtype)
    assert isinstance(frame["sex"].dtype, pd.CategoricalDtype)
    assert frame["grade"].dtype.ordered
    assert frame["event"].dtype == "boolean"
    assert frame["age_bins"].dtype.ordered
    # who_grade is reattached (bypasses R as the outcome's source) — its
    # categorical dtype must survive reattachment, not flatten to str.
    assert isinstance(frame["who_grade"].dtype, pd.CategoricalDtype)
    assert frame["who_grade"].dtype.ordered
    assert list(frame["who_grade"].cat.categories) == ["1", "2", "3"]


def test_proper_mice_derived_consistency(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    frames = proper_mice_impute(
        df, schema, m=2, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    for frame in frames:
        age = frame["age"].astype(float)
        expected = pd.cut(age, [-np.inf, 50, np.inf], labels=["<50", "50+"], right=False)
        assert frame["age_bins"].astype("object").tolist() == \
            pd.Series(expected).astype("object").tolist()


def test_proper_mice_cell_variation_only_missing(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    proper_mice_impute(
        df, schema, m=3, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    path = tmp_output / "missingness" / "mice" / "imputed_cell_variation.csv"
    assert path.exists()
    table = pd.read_csv(path)
    expected_cells = sum(
        int(df[c].isna().sum()) for c in ("age", "sex", "grade", "event")
    )
    assert len(table) == expected_cells
    # every listed cell was originally missing
    for _, r in table.iterrows():
        pos = int(r["original_row_id"])
        assert pd.isna(df[r["variable"]].iloc[pos])
    # JSON list-like fields parse
    assert isinstance(json.loads(table.iloc[0]["values_across_draws"]), list)


def test_proper_mice_manifest_fields(fake_r, tmp_output):
    df, schema, deps, transform = _make_cohort()
    proper_mice_impute(
        df, schema, m=2, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    manifest = mr.read_mice_manifest(tmp_output)
    assert manifest["method"] == "mice_fcs_mixed_type_r"
    assert manifest["engine"] == "R mice"
    assert manifest["proper_multiple_imputation"] is True
    assert manifest["rubin_pooling_supported"] is True
    assert manifest["analysis_outcome"] == "high_grade"
    assert manifest["derived_dependencies"]["high_grade"] == ["who_grade"]
    assert manifest["methods_by_column"]["age"] == "pmm"
    assert "input_sha256" in manifest
    mr.assert_proper_multiple_imputation(tmp_output)  # passes


# ---------------------------------------------------------------------------
# RF relabel + manifest gating
# ---------------------------------------------------------------------------

def test_rf_manifest_disables_rubin_pooling(tiny_df, tiny_schema, tmp_output):
    rf_chained_impute(
        tiny_df, tiny_schema, m=2, max_iter=2, random_state=0,
        output_root=tmp_output,
    )
    manifest = mr.read_mice_manifest(tmp_output)
    assert manifest["method"] == "rf_chained_imputation_posthoc_bernoulli"
    assert manifest["proper_multiple_imputation"] is False
    assert manifest["rubin_pooling_supported"] is False
    with pytest.raises(ValueError):
        mr.assert_proper_multiple_imputation(tmp_output)


def test_mice_impute_alias_is_rf():
    assert mr.mice_impute is mr.rf_chained_impute


# ---------------------------------------------------------------------------
# Real-R integration smoke test (skipped if R / mice unavailable)
# ---------------------------------------------------------------------------

def _r_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    try:
        probe = (
            'cat(if (all(c(requireNamespace("mice", quietly=TRUE), '
            'requireNamespace("jsonlite", quietly=TRUE)))) "OK" else "NO")'
        )
        out = subprocess.run([rscript, "-e", probe], capture_output=True, text=True)
        return "OK" in (out.stdout or "")
    except Exception:
        return False


def _make_cohort_large(n: int = 60) -> tuple[pd.DataFrame, dict[str, ColSpec], dict, object]:
    """Realistically-sized cohort so real mice() converges (toy 6-row data can't).

    One incomplete variable per kind (continuous, binary, nominal, ordinal) plus
    a derived column and a derived analysis outcome.
    """
    rng = np.random.default_rng(0)
    who = rng.integers(1, 4, n)

    def _blank(values, frac=0.12):
        out = list(values)
        for i in rng.choice(n, size=int(n * frac), replace=False):
            out[i] = None
        return out

    df = pd.DataFrame({
        "id": [f"p{i:03d}" for i in range(n)],
        "age": pd.array(_blank(rng.normal(60, 10, n).round(1)), dtype="Float64"),
        "sex": pd.Categorical(
            _blank(rng.choice(["male", "female"], n)),
            categories=["female", "male"], ordered=False,
        ),
        "grade": pd.Categorical(
            _blank(rng.integers(1, 4, n).astype(str)),
            categories=["1", "2", "3"], ordered=True,
        ),
        "event": pd.array(
            _blank(rng.choice([True, False], n)), dtype="boolean",
        ),
        "who_grade": pd.Categorical(
            who.astype(str), categories=["1", "2", "3"], ordered=True,
        ),
        "high_grade": pd.array(who >= 2, dtype="boolean"),
        "age_bins": pd.Categorical(
            [None] * n, categories=["<50", "50+"], ordered=True,
        ),
    })
    df.index = [f"row{i}" for i in range(n)]

    schema = {
        "id": ColSpec("id", "id", keep=False),
        "age": ColSpec("age", "continuous"),
        "sex": ColSpec("sex", "nominal"),
        "grade": ColSpec("grade", "ordinal", ordered_levels=["1", "2", "3"]),
        "event": ColSpec("event", "binary"),
        "who_grade": ColSpec("who_grade", "ordinal", ordered_levels=["1", "2", "3"]),
        "high_grade": ColSpec("high_grade", "binary"),
        "age_bins": ColSpec("age_bins", "ordinal", ordered_levels=["<50", "50+"]),
    }
    deps = {"high_grade": ["who_grade"], "age_bins": ["age"]}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        age = pd.to_numeric(out["age"], errors="coerce").astype(float)
        out["age_bins"] = pd.Categorical(
            pd.cut(age, [-np.inf, 50, np.inf], labels=["<50", "50+"], right=False),
            categories=["<50", "50+"], ordered=True,
        )
        if "high_grade" not in out.columns:
            wg = pd.to_numeric(out["who_grade"].astype("object"), errors="coerce")
            out["high_grade"] = ((wg == 2) | (wg == 3)).astype("boolean")
        return out

    return df, schema, deps, transform


@pytest.mark.skipif(
    not _r_available(),
    reason="Formal MICE requires R with the mice and jsonlite packages installed.",
)
def test_proper_mice_real_r_smoke(tmp_output):
    df, schema, deps, transform = _make_cohort_large(60)
    frames = proper_mice_impute(
        df, schema, m=3, max_iter=5, output_root=tmp_output,
        analysis_outcome="high_grade", derived_dependencies=deps,
        post_impute_transform=transform,
    )
    assert len(frames) == 3
    for frame in frames:
        for col in ("age", "sex", "grade", "event"):
            assert frame[col].isna().sum() == 0
        assert frame.index.equals(df.index)
    mice_dir = tmp_output / "missingness" / "mice"
    for artifact in (
        "methods.csv", "predictor_matrix.csv", "logged_events.csv",
        "chain_diagnostics.png", "r_session.json", "imputed_cell_variation.csv",
    ):
        assert (mice_dir / artifact).exists(), artifact
