"""Tests for dataset_handoff.py — parquet roundtrip validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_handoff import (
    assert_dataset_equivalent,
    detect_imputation_method,
    validate_handoff_datasets,
)
from missingness_resolution import (
    MICE_MODELING_DATASET_NAME,
    SIMPLE_MODELING_DATASET_NAME,
    _datasets_dir,
    _save_dataset_parquet,
    prepare_datasets_dir,
    save_imputed_frames,
    stage_unimputed_dataset,
)


def test_assert_dataset_equivalent_passes(tiny_df):
    assert_dataset_equivalent(tiny_df, tiny_df.copy(), context="roundtrip")


def test_assert_dataset_equivalent_shape_mismatch(tiny_df):
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_dataset_equivalent(tiny_df, tiny_df.head(1), context="roundtrip")


def test_detect_imputation_method_mice(tmp_output, tiny_df):
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / MICE_MODELING_DATASET_NAME,
        context=MICE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    assert detect_imputation_method(tmp_output) == "mice"


def test_detect_imputation_method_simple(tmp_output, tiny_df):
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / SIMPLE_MODELING_DATASET_NAME,
        context=SIMPLE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    assert detect_imputation_method(tmp_output) == "simple"


def test_detect_imputation_method_missing(tmp_output):
    with pytest.raises(FileNotFoundError):
        detect_imputation_method(tmp_output)


def test_validate_handoff_unimputed_and_simple(tmp_output, tiny_df):
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    imputed = tiny_df.fillna(tiny_df.median(numeric_only=True))
    _save_dataset_parquet(
        imputed,
        _datasets_dir(tmp_output) / SIMPLE_MODELING_DATASET_NAME,
        context=SIMPLE_MODELING_DATASET_NAME,
        dtype_reference=imputed,
    )
    manifest_path = _datasets_dir(tmp_output) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["modeling_dataset"] = SIMPLE_MODELING_DATASET_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validate_handoff_datasets(
        tmp_output,
        df_unimputed=tiny_df,
        imputation_method="simple",
        imputed_single=imputed,
    )


def test_validate_handoff_mice_draws(tmp_output, tiny_df):
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    frames = [tiny_df.copy(), tiny_df.copy()]
    imputed = frames[0].fillna(frames[0].median(numeric_only=True))
    frames[0] = imputed
    frames[1] = imputed.copy()
    _save_dataset_parquet(
        frames[0],
        _datasets_dir(tmp_output) / MICE_MODELING_DATASET_NAME,
        context=MICE_MODELING_DATASET_NAME,
        dtype_reference=frames[0],
    )
    save_imputed_frames(frames, tmp_output, source_df=frames[0])

    validate_handoff_datasets(
        tmp_output,
        df_unimputed=tiny_df,
        imputation_method="mice",
        imputed_frames=frames,
    )
