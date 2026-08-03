"""Study facts: what the pipeline can answer, and what it must leave open.

The point of this module is that a methods fact is either *read back out of the
cleaning run* or *visibly missing* — never typed twice and never guessed. These
tests are about that boundary.
"""
from __future__ import annotations

import pandas as pd
import pytest

import study


def _cleaning_artifacts(root):
    """A minimal output/cleaning/ — the two files study.py reads."""
    folder = root / "cleaning"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"step": "raw_data", "detail": "rows in source file", "n_rows": 397,
         "criterion": ""},
        # A filter that excluded nobody: real in this cohort (the year filter),
        # and it must not reach the methods sentence.
        {"step": "drop_rows", "detail": "cohort year", "n_rows": 397,
         "criterion": "all years in entry_year: [2018, 2019]"},
        {"step": "drop_rows", "detail": "location", "n_rows": 354,
         "criterion": "Intracranial meningioma: side recorded"},
        {"step": "drop_rows", "detail": "who_grade exists", "n_rows": 352,
         "criterion": "Histologically confirmed WHO grade recorded"},
        {"step": "final", "detail": "rows entering analysis", "n_rows": 352,
         "criterion": ""},
    ]).to_csv(folder / "cleaning_summary.csv", index=False)
    pd.DataFrame([
        {"derivation": "high_grade", "kind": "binary", "source": "who_grade",
         "rule": "who_grade in {2, 3}",
         "reason": "WHO 2021 CNS classification: grade 2/3 = high-grade meningioma."},
        {"derivation": "edema_index", "kind": "continuous",
         "source": "edema_volume_cm3, tumor_volume", "rule": "",
         "reason": "Frati, Armocida et al., Tomography 2022;8(4):1987–1996"},
        {"derivation": "tumor_volume_ge13.95", "kind": "binary",
         "source": "tumor_volume", "rule": "tumor_volume ≥ 13.95 cm³",
         "reason": "Shin, Kim, Cheong et al., PLoS One 2021;16(6):e0252945"},
        {"derivation": "multiple_meningiomas", "kind": "binary",
         "source": "meningioma_count", "rule": "meningioma_count > 1", "reason": ""},
    ]).to_csv(folder / "derivation_log.csv", index=False)
    return root


@pytest.fixture
def output_root(tmp_path):
    return _cleaning_artifacts(tmp_path)


def test_who_edition_comes_from_the_derivation_that_built_the_outcome(output_root):
    """The edition is recorded where the outcome was defined, not retyped here."""
    answers = {r["key"]: r["answer"] for r in study.pipeline_answers(output_root)}
    assert "WHO 2021" in answers["who_edition"]
    assert "who_grade in {2, 3}" in answers["who_edition"]


def test_inclusion_reports_the_flow_and_skips_filters_that_excluded_nobody(output_root):
    answers = {r["key"]: r["answer"] for r in study.pipeline_answers(output_root)}
    inclusion = answers["inclusion"]
    assert "397 records" in inclusion and "352 analysed" in inclusion
    assert "Intracranial meningioma" in inclusion
    assert "all years in entry_year" not in inclusion


def test_published_cutpoints_are_listed_with_their_papers(output_root):
    answers = {r["key"]: r["answer"] for r in study.pipeline_answers(output_root)}
    assert "PLoS One 2021" in answers["published_cutpoints"]
    # A derived flag with no citation is not a published cut-point.
    assert "meningioma_count" not in answers["published_cutpoints"]


def test_missing_cleaning_artifacts_degrade_to_no_answers(tmp_path):
    """A threshold run against an old output folder still works — it just has
    more open questions."""
    assert study.pipeline_answers(tmp_path) == []
    table = study.study_facts_table({}, tmp_path)
    assert len(table) == len(study.QUESTIONS)
    assert (table["status"] == study.STATUS_OPEN).all()


def test_a_blank_answer_stays_an_open_question(output_root):
    table = study.study_facts_table({"dwi_acquisition": "   "}, output_root)
    row = table.set_index("key").loc["dwi_acquisition"]
    assert row["status"] == study.STATUS_OPEN
    assert row["answer"] == study.QUESTIONS_BY_KEY["dwi_acquisition"].prompt


def test_a_filled_answer_replaces_the_prompt(output_root):
    table = study.study_facts_table({"dwi_acquisition": "3 T Siemens, b = 0/1000"},
                                    output_root)
    row = table.set_index("key").loc["dwi_acquisition"]
    assert row["status"] == study.STATUS_ANSWERED
    assert row["answer"] == "3 T Siemens, b = 0/1000"
    assert len(study.open_questions(table)) == len(study.QUESTIONS) - 1


def test_a_typo_in_a_fact_key_is_raised_not_swallowed(output_root):
    """Silently dropping "dwi_aquisition" would leave the question open forever
    while the notebook shows it as answered."""
    with pytest.raises(KeyError, match="dwi_aquisition"):
        study.study_facts_table({"dwi_aquisition": "3 T"}, output_root)


def test_every_question_says_why_it_is_needed():
    """A question without a reason gets skipped; that is how the ten-item list
    this replaced stopped being read."""
    for question in study.QUESTIONS:
        assert question.why and question.why[0].isupper()
        assert question.prompt


def test_literature_sources_pair_each_cutpoint_with_its_link():
    table = study.literature_sources_table(
        {"vol": [(13.95, "Shin 2021")], "adc": [(0.85, "Nobody 2019")]},
        {"Shin 2021": "https://doi.org/10.1371/journal.pone.0252945"},
    )
    assert list(table["column"]) == ["vol", "adc"]
    assert table.set_index("source").loc["Shin 2021", "link"].startswith("https://")
    assert table.set_index("source").loc["Nobody 2019", "link"] == ""


def test_no_literature_cutpoints_is_an_empty_table_not_a_crash():
    assert study.literature_sources_table({}, {}).empty
