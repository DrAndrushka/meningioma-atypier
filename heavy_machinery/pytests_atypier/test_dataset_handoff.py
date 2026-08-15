"""Tests for dataset_handoff.py — parquet roundtrip validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dataset_handoff import (
    assert_dataset_equivalent,
    detect_imputation_method,
    load_modelling_handoff,
    validate_handoff_datasets,
    validate_schema_against_frame,
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
from schema_infer import export_schema_summary


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


def test_validate_handoff_reexports_schema(tmp_output, tiny_df, tiny_schema):
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
        schema=tiny_schema,
    )
    assert (tmp_output / "schema" / "schema_summary.csv").exists()


def test_load_modelling_handoff(tmp_output, tiny_df, tiny_schema):
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    imputed = tiny_df.fillna(tiny_df.median(numeric_only=True))
    _save_dataset_parquet(
        imputed,
        _datasets_dir(tmp_output) / SIMPLE_MODELING_DATASET_NAME,
        context=SIMPLE_MODELING_DATASET_NAME,
        dtype_reference=imputed,
    )
    export_schema_summary(tiny_schema, tmp_output)

    df, schema, method = load_modelling_handoff(tmp_output)
    assert method == "simple"
    assert list(df.columns) == list(tiny_df.columns)
    assert schema["age"].kind == "continuous"
    validate_schema_against_frame(df, schema)


def _stage_simple_handoff(tmp_output, tiny_df, tiny_schema) -> None:
    """A complete simple-imputation handoff on disk, as cleaning §16 leaves it."""
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / SIMPLE_MODELING_DATASET_NAME,
        context=SIMPLE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    export_schema_summary(tiny_schema, tmp_output)


def test_load_modelling_handoff_reports_what_it_loaded(tmp_output, tiny_df, tiny_schema, capsys):
    """The notebook used to print this itself, with an if/else on the method.

    A branch in a notebook cell is a branch nobody tests. It lives here now.
    """
    _stage_simple_handoff(tmp_output, tiny_df, tiny_schema)

    df, schema, method = load_modelling_handoff(tmp_output)
    out = capsys.readouterr().out

    assert method == "simple"
    assert f"{len(df)} rows" in out
    assert f"{len(schema)} schema columns" in out
    assert "one imputed parquet" in out


def test_load_modelling_handoff_reports_the_mice_branch(tmp_output, tiny_df, tiny_schema, capsys):
    """MICE means the inferential stage pools draws. The reader is told which."""
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / MICE_MODELING_DATASET_NAME,
        context=MICE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    export_schema_summary(tiny_schema, tmp_output)

    _, _, method = load_modelling_handoff(tmp_output)

    assert method == "mice"
    assert "MICE" in capsys.readouterr().out


def test_load_modelling_handoff_can_be_quiet(tmp_output, tiny_df, tiny_schema, capsys):
    """Callers that are not a notebook should not have to eat the printing."""
    _stage_simple_handoff(tmp_output, tiny_df, tiny_schema)
    capsys.readouterr()  # Clear setup output

    load_modelling_handoff(tmp_output, verbose=False)

    assert capsys.readouterr().out == ""


def test_midline_flag_means_midline():
    """Renamed to ``midline_positioning`` when the nominal parents became flags.

    The name changed; what it has to mean did not. This test exists because the
    flag once marked the lateralised tumours instead of the midline ones, so it
    follows the rename rather than being retired with it.
    """
    df = pd.read_parquet("output/datasets/unimputed_df.parquet")
    derived = (df["side"] == "midline")
    both = df["midline_positioning"].notna() & df["side"].notna()
    assert (df.loc[both, "midline_positioning"].astype(bool) == derived[both]).all()
    # cohort fact: 35 midline vs 317 lateralised of 352
    assert int(df["midline_positioning"].sum()) == 35
