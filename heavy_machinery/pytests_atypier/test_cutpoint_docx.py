"""Word tables — the journal's rules checked against the saved file, not intent."""
from __future__ import annotations

import pandas as pd
import pytest
from docx import Document
from docx.shared import Inches

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import docx_tables as dx


def _frame(cols: int = 3, rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [[f"v{r}{c}" for c in range(cols)] for r in range(rows)],
        index=[f"Criterion {r}" for r in range(rows)],
        columns=[f"Measure {c}" for c in range(cols)])


def _write(tmp_path, frame=None, **over):
    spec = dict(frame=frame if frame is not None else _frame(),
                title="Table. Demo", note="A note.", index_header="Criterion")
    spec.update(over)
    return dx.write_tables(tmp_path / "table.docx", [spec])


# --- no borders ------------------------------------------------------------
def test_the_saved_file_prints_no_borders(tmp_path):
    assert dx.audit(_write(tmp_path))["visible_borders"] == 0


def test_a_grid_style_is_counted_as_visible_before_stripping():
    """Borders live in the style, so the table XML alone proves nothing."""
    document = Document()
    table = document.add_table(rows=2, cols=2, style="Table Grid")
    assert "tblBorders" not in table._tbl.xml      # nothing to delete
    assert dx.visible_borders(table) > 0           # yet it would print a grid


def test_stripping_overrides_the_style_rather_than_deleting_nothing():
    document = Document()
    table = document.add_table(rows=2, cols=2, style="Table Grid")
    dx.strip_borders(table)
    assert dx.visible_borders(table) == 0
    assert "tblBorders" in table._tbl.xml          # an explicit all-none block


# --- width -----------------------------------------------------------------
def test_the_saved_width_is_under_the_limit(tmp_path):
    """Measured on disk: EMU rounding once pushed a compliant layout over."""
    assert dx.audit(_write(tmp_path))["max_width_in"] <= dx.MAX_WIDTH_IN


def test_layout_leaves_headroom_for_emu_rounding():
    assert dx.LAYOUT_WIDTH_IN < dx.MAX_WIDTH_IN


def test_an_oversized_column_spec_is_refused(tmp_path):
    with pytest.raises(dx.DocxContractError, match="broadside"):
        _write(tmp_path, column_widths=[2.0, 2.0, 2.0, 2.0])


def test_widths_are_measured_in_the_units_word_actually_stores():
    """Twips, not EMUs. 0.858 in is recorded as 0.858333 — rounded up."""
    assert dx.stored_inches(0.858) == pytest.approx(1236 / 1440)
    assert dx.stored_inches(0.858) > 0.858


def test_laying_out_to_the_hard_limit_would_overshoot_in_the_file():
    """The bug this module was written around: 6.5 in of columns saved as 6.5014."""
    first = dx.MAX_WIDTH_IN * 0.34
    rest = (dx.MAX_WIDTH_IN - first) / 5
    widths = [first] + [rest] * 5              # six columns, as the real table has
    assert sum(widths) == pytest.approx(dx.MAX_WIDTH_IN)
    assert sum(dx.stored_inches(w) for w in widths) > dx.MAX_WIDTH_IN


def test_the_default_layout_stays_under_once_stored(tmp_path):
    assert dx.audit(_write(tmp_path, frame=_frame(cols=5)))["max_width_in"] <= dx.MAX_WIDTH_IN


# --- type size and orientation --------------------------------------------
def test_nothing_is_shrunk_below_body_size(tmp_path):
    assert dx.audit(_write(tmp_path))["font_sizes_pt"] == [dx.BODY_PT]


def test_the_document_stays_portrait(tmp_path):
    assert dx.audit(_write(tmp_path))["landscape"] is False


def test_every_paragraph_is_double_spaced(tmp_path):
    document = Document(str(_write(tmp_path)))
    spacings = {p.paragraph_format.line_spacing for p in document.paragraphs
                if p.text.strip()}
    assert spacings == {dx.LINE_SPACING}


# --- the note --------------------------------------------------------------
def test_the_note_is_written_beneath_the_table(tmp_path):
    document = Document(str(_write(tmp_path)))
    texts = [p.text for p in document.paragraphs]
    note = next(t for t in texts if t.startswith(dx.NOTE_PREFIX))
    assert note.startswith("Note:—A note.")


def test_only_abbreviations_that_appear_are_defined():
    """A note glossing PPV in a table without one was not written for it."""
    used = dx.abbreviations_used("AUC 0.63 with a 95% CI and an IQR")
    assert used == ["AUC", "CI", "IQR"]
    assert "PPV" not in used


def test_abbreviations_are_defined_alphabetically_in_the_journal_pattern():
    note = dx.abbreviation_note("SD and ADC and AUC")
    assert note.startswith("ADC indicates apparent diffusion coefficient; ")
    assert note.endswith("SD, standard deviation.")


def test_a_term_inside_a_longer_word_is_not_matched():
    """'OR' must not fire on 'ORDER' — an uppercase letter after it disqualifies."""
    assert "OR" not in dx.abbreviations_used("Ordered by ORDER of magnitude")
    assert "OR" in dx.abbreviations_used("the OR was 3.6")


def test_a_plural_abbreviation_still_counts():
    assert "AUC" in dx.abbreviations_used("the two AUCs were compared")


def test_a_table_with_no_abbreviations_gets_no_glossary():
    assert dx.abbreviation_note("plain words only") == ""


def test_the_glossary_defines_each_term_once():
    assert len(set(dx.GLOSSARY.values())) == len(dx.GLOSSARY)


# --- structure -------------------------------------------------------------
def test_the_header_row_carries_the_column_names(tmp_path):
    document = Document(str(_write(tmp_path)))
    header = [c.text for c in document.tables[0].rows[0].cells]
    assert header == ["Criterion", "Measure 0", "Measure 1", "Measure 2"]


def test_the_index_becomes_the_first_column(tmp_path):
    document = Document(str(_write(tmp_path)))
    first = [row.cells[0].text for row in document.tables[0].rows[1:]]
    assert first == [f"Criterion {r}" for r in range(4)]


def test_each_table_starts_on_its_own_page(tmp_path):
    path = dx.write_tables(tmp_path / "two.docx", [
        dict(frame=_frame(), title="Table 1", note="", index_header="A"),
        dict(frame=_frame(), title="Table 2", note="", index_header="A")])
    document = Document(str(path))
    breaks = sum(1 for p in document.paragraphs
                 for r in p.runs if "w:br" in r._element.xml)
    assert breaks >= 1
    assert len(document.tables) == 2


def test_values_are_written_as_given_and_never_rounded(tmp_path):
    """A second rounding rule here would be free to disagree with the figures."""
    frame = pd.DataFrame([["0.7241"]], index=["x"], columns=["y"])
    document = Document(str(_write(tmp_path, frame=frame)))
    assert document.tables[0].rows[1].cells[1].text == "0.7241"


# --- against the real scorecard -------------------------------------------
def test_the_real_criteria_table_passes_every_rule(tmp_path, real_cohort):
    import bend_location as bl
    import collinearity as co
    import criteria as cr
    import dichotomy as di
    import eligibility as el
    import imputation as imp
    import models as mo
    import nonlinearity as nl
    import scorecard as sc
    import separation as sep
    import wobble as wb

    frozen = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
              "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    fits = nl.fit_all(real_cohort)
    separation = sep.separation_table(real_cohort)
    bend = bl.bend_table(real_cohort, fits=fits)
    eligible = el.eligible(el.carry_forward(separation, bend))
    wobble, _ = wb.wobble_table(real_cohort, eligible, n_boot=60, frozen=frozen)
    long = sc.scorecard_long(
        eligible, separation=separation, bend=bend,
        agreement=cr.agreement(cr.criteria_table(real_cohort, eligible),
                               real_cohort),
        wobble=wobble,
        imputation=imp.per_draw_cutpoints(imp.load_draws("output"), eligible,
                                          frozen=frozen),
        dichotomy=di.dichotomy_table(real_cohort, eligible, frozen),
        pairs=co.correlated_pairs(*co.spearman_matrix(real_cohort)),
        coefficients=mo.compare_sets(real_cohort, cutpoints=frozen,
                                     n_boot=40)[1])

    path = dx.write_tables(tmp_path / "criteria.docx", [dict(
        frame=sc.scorecard_wide(long),
        title="Table. Criterion-based appraisal of candidate cut-points",
        note=sc.footnote(), index_header="Criterion")])
    report = dx.audit(path)
    assert report["visible_borders"] == 0
    assert report["max_width_in"] <= dx.MAX_WIDTH_IN
    assert report["font_sizes_pt"] == [dx.BODY_PT]
    assert report["landscape"] is False
    assert report["has_note"]
