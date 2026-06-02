"""Regression tests for pipeline correctness (pytest)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaning import apply_schema
from eda import _encode_binary_target
from inferential import _build_design, _encode_target, _safe_z_denominator
from report import _embed_svg_src, svg_grid
from schema_infer import ColSpec, infer_schema


def test_encode_respects_false_positive_class():
    y = pd.Series([False, False, True, True], name="event")
    auto, _ = _encode_binary_target(y, None)
    explicit, _ = _encode_binary_target(y, False)
    np.testing.assert_array_equal(auto.values, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(explicit.values, [1.0, 1.0, 0.0, 0.0])
    assert not np.allclose(auto, explicit)


def test_plot_rate_counts_not_inverted():
    df = pd.DataFrame({"y": [False, False, True, True], "x": ["A", "A", "B", "B"]})
    mask_a = df["x"] == "A"
    mask_b = df["x"] == "B"
    auto, _ = _encode_binary_target(df["y"], None)
    pos_false, _ = _encode_binary_target(df["y"], False)
    assert auto.loc[mask_a].mean() == 0.0
    assert pos_false.loc[mask_a].mean() == 1.0
    assert auto.loc[mask_b].mean() == 1.0
    assert pos_false.loc[mask_b].mean() == 0.0


def test_missing_target_stays_nan():
    y = pd.Series([True, False, pd.NA], dtype="boolean")
    enc, _ = _encode_target(y, True)
    assert np.isnan(enc.iloc[2])
    assert enc.iloc[0] == 1.0
    assert enc.iloc[1] == 0.0


def test_z_denominator_nan_and_zero():
    assert _safe_z_denominator(0.0) == 1.0
    assert _safe_z_denominator(float("nan")) == 1.0
    assert _safe_z_denominator(2.5) == 2.5


def test_dummy_reindex_fill_zero():
    schema = {"g": ColSpec("g", "nominal")}
    f0 = pd.DataFrame({"g": ["a", "a", "b"]})
    f1 = pd.DataFrame({"g": ["b", "b", "b"]})
    X0, _, _ = _build_design(f0, schema, ["g"])
    keep = list(X0.columns)
    X1, _, _ = _build_design(f1, schema, ["g"])
    aligned = X1.reindex(columns=keep, fill_value=0.0)
    assert list(aligned.columns) == keep
    assert (aligned.fillna(-99) >= 0).all().all()


def test_unique_continuous_not_marked_id():
    n = 50
    df = pd.DataFrame({"measurement": np.arange(n, dtype=float)})
    schema = infer_schema(df)
    assert schema["measurement"].kind == "continuous"


def test_ordinal_without_levels_raises():
    df = pd.DataFrame({"grade": [1, 2, 3]})
    schema = {"grade": ColSpec("grade", "ordinal")}
    with pytest.raises(ValueError):
        apply_schema(df, schema)


def test_svg_embedded_as_data_uri(tmp_path: Path):
    fig = tmp_path / "plot.svg"
    fig.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle r="1"/></svg>')
    uri = _embed_svg_src(fig)
    assert uri is not None
    assert uri.startswith("data:image/svg+xml;base64,")
    html = svg_grid([fig])
    assert "data:image/svg+xml;base64," in html
    assert "../" not in html
    assert "file://" not in html
