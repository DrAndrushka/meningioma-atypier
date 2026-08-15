"""The provenance record: does it say what actually happened, and can it be checked?"""
from __future__ import annotations

import json
from pathlib import Path

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


# --- which repository ------------------------------------------------------
def _run(*args, cwd):
    import subprocess
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                          check=True).stdout.strip()


def _repo_with_one_commit(path, filename: str):
    """A git repository with a single commit, and that commit's SHA."""
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text("x\n")
    _run("git", "init", "-q", cwd=path)
    _run("git", "add", "-A", cwd=path)
    _run("git", "-c", "user.email=t@t", "-c", "user.name=T",
         "-c", "commit.gpgsign=false", "commit", "-qm", "one", cwd=path)
    return _run("git", "rev-parse", "HEAD", cwd=path)


def test_the_commit_is_the_project_repo_not_one_nested_inside_it(tmp_path):
    """The real defect: git walks *up* and stops at the first .git it meets.

    This project carries an abandoned repository inside ``heavy_machinery/``
    from an earlier layout. Running git from a module's own folder found that
    one, so every manifest recorded a commit absent from the project and a
    permanent "25 uncommitted files" from modules the dead repo still thinks
    were deleted.
    """
    outer = tmp_path / "project"
    outer_sha = _repo_with_one_commit(outer, "outer.txt")
    inner_sha = _repo_with_one_commit(outer / "library", "inner.txt")
    assert outer_sha != inner_sha

    state = mf.code_state(root=outer)
    assert state["commit"] == outer_sha
    assert state["commit"] != inner_sha
    assert state["repo_warning"] is None


def test_the_real_manifest_records_this_repository(tmp_path):
    """The regression itself, on the actual working tree."""
    here = Path(mf.PROJECT_ROOT)
    expected = _run("git", "rev-parse", "HEAD", cwd=here)
    state = mf.code_state()
    assert state["commit"] == expected
    assert state["repo_warning"] is None
    # And specifically not the nested one, if it is still on disk.
    nested = here / "heavy_machinery"
    if (nested / ".git").exists():
        assert state["commit"] != _run("git", "rev-parse", "HEAD", cwd=nested)


def test_no_commit_is_claimed_when_git_resolves_somewhere_else(tmp_path):
    """Better to admit ignorance than to name the wrong history."""
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    state = mf.code_state(root=outside)
    assert state["commit"] is None
    assert state["dirty"] is None
    assert state["repo_warning"] and "no commit recorded" in state["repo_warning"]


def test_describe_shouts_when_the_repository_is_wrong(tiny, tmp_path):
    """A wrong commit hash looks exactly like a right one — it has to shout."""
    df, root = tiny
    record = _record(df, root, [])
    record["code"]["repo_warning"] = "git resolved /somewhere/else rather than /here"
    assert "⚠️" in mf.describe(record)


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
