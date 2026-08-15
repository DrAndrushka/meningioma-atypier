"""The provenance record: does it say what actually happened, and can it be checked?"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import manifest as mf


@pytest.fixture()
def tiny(tmp_path):
    """A minimal frame and a file for it to point at."""
    source = tmp_path / "datasets" / "unimputed_df.parquet"
    source.parent.mkdir(parents=True)
    # Distinctive digits: short values like 0.7 turn up by chance inside a
    # hash or a version string and the disclosure test would fire on noise.
    df = pd.DataFrame({"high_grade": [1, 0, 1, 0],
                       "adc_value": [0.612345, 0.987654, 0.734512, 1.098765]})
    df.to_parquet(source)
    return df, tmp_path


def _record(df, root, written):
    return mf.build(df, cutpoints={"adc_value": 0.72},
                    facts={"patients": 4, "high_grade": 2, "prevalence": 0.5},
                    written=written, output_root=root)


# --- what it records -------------------------------------------------------
def test_the_seed_and_resample_count_come_from_the_modules_that_use_them():
    """A manifest that restates a seed can record one the run never used."""
    import wobble as wb
    settings = mf.settings()
    assert settings["seed"] == wb.SEED
    assert settings["bootstrap_resamples"] == wb.N_BOOTSTRAP


def test_the_log_scale_declaration_is_recorded(tiny):
    from scales import LOG1P_COLUMNS
    assert mf.settings()["log1p_columns"] == sorted(LOG1P_COLUMNS)


def test_the_cohort_is_hashed_rather_than_copied(tiny):
    """A hash proves two runs read the same rows without disclosing any of them."""
    df, root = tiny
    record = _record(df, root, [])
    source = record["cohort"]["source"]
    assert source["exists"] and len(source["sha256"]) == 64
    # Counts and a hash, never values: the manifest travels with the manuscript.
    written = json.dumps(record)
    assert not any(str(v) in written for v in df["adc_value"])


def test_a_dirty_tree_is_recorded_and_not_hidden():
    """A commit hash means nothing if the files did not match it."""
    state = mf.code_state()
    assert set(state) >= {"commit", "dirty", "python", "packages"}
    assert state["dirty"] in (True, False, None)


def test_every_library_that_can_move_a_digit_has_a_version():
    versions = mf.code_state()["packages"]
    assert versions["numpy"] != "not installed"
    assert "python-docx" in versions


def test_a_file_that_was_not_written_is_recorded_as_absent(tmp_path):
    record = mf.file_record(tmp_path / "never_written.docx")
    assert record["exists"] is False and "sha256" not in record


# --- checking it later -----------------------------------------------------
def test_verify_is_silent_when_nothing_has_changed(tiny, tmp_path):
    df, root = tiny
    written = root / "table_1.docx"
    written.write_bytes(b"pretend this is Word")
    path = mf.write(_record(df, root, [written]), tmp_path / "manifest.json")
    assert mf.verify(path) == []


def test_verify_names_a_file_that_changed_after_the_run(tiny, tmp_path):
    df, root = tiny
    written = root / "table_1.docx"
    written.write_bytes(b"first version")
    path = mf.write(_record(df, root, [written]), tmp_path / "manifest.json")
    written.write_bytes(b"someone edited this by hand")
    problems = mf.verify(path)
    assert len(problems) == 1 and "table_1.docx" in problems[0]


def test_verify_notices_the_cohort_being_re_exported(tiny, tmp_path):
    """The commonest silent cause of two runs disagreeing."""
    df, root = tiny
    path = mf.write(_record(df, root, []), tmp_path / "manifest.json")
    source = root / "datasets" / "unimputed_df.parquet"
    df.assign(adc_value=[0.612346, 0.987654, 0.734512, 1.098765]).to_parquet(source)
    assert any("unimputed_df.parquet" in p for p in mf.verify(path))


def test_verify_reports_a_file_that_has_gone_missing(tiny, tmp_path):
    df, root = tiny
    written = root / "fig_1.tif"
    written.write_bytes(b"tiff")
    path = mf.write(_record(df, root, [written]), tmp_path / "manifest.json")
    written.unlink()
    assert any("missing now" in p for p in mf.verify(path))


def test_the_manifest_is_valid_json_and_round_trips(tiny, tmp_path):
    df, root = tiny
    path = mf.write(_record(df, root, []), tmp_path / "manifest.json")
    loaded = json.loads(path.read_text())
    assert loaded["phase"] == "cutpoint"
    assert loaded["frozen_cutpoints"] == {"adc_value": 0.72}


def test_describe_names_the_commit_and_the_cohort_size(tiny, tmp_path):
    df, root = tiny
    line = mf.describe(_record(df, root, []))
    assert "4 patients" in line and "seed" in line
