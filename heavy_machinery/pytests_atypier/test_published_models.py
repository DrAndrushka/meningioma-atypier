"""Tests for config/published_models.py — transcription integrity."""
from __future__ import annotations

from pathlib import Path

from heavy_machinery.config import load

pm = load("published_models")

EXPECTED = {"radeesri_2023", "spille_2020", "zhang_2020", "funari_2023",
            "kawahara_2012", "lin_2014", "peng_2021"}


def test_every_literature_model_has_a_published_record():
    assert EXPECTED <= set(pm.PUBLISHED_MODELS)


def test_surrogate_note_is_set_exactly_on_the_interface_substitutions():
    with_note = {k for k, v in pm.PUBLISHED_MODELS.items() if v.get("surrogate_note")}
    assert with_note == {"kawahara_2012", "lin_2014", "peng_2021"}


def test_kawahara_carries_the_transcribed_multivariable_odds_ratios():
    """From the publisher PDF, Table 3. Exactly two retained terms; capsular
    enhancement and tumoral margin were assessed and dropped, so they must NOT
    appear as model terms."""
    terms = pm.PUBLISHED_MODELS["kawahara_2012"]["terms"]
    assert len(terms) == 2
    by_var = {t["variable"].lower(): t for t in terms}
    tbi = next(v for k, v in by_var.items() if "interface" in k)
    het = next(v for k, v in by_var.items() if "heterogeneous" in k)
    assert (tbi["or"], tbi["ci_lo"], tbi["ci_hi"]) == (42.0, 4.5, 390)
    assert (het["or"], het["ci_lo"], het["ci_hi"]) == (8.3, 1.7, 40.4)


def test_kawahara_calibration_probabilities_pair_with_the_larger_odds_ratio():
    """Interface has the larger aOR (42.0 vs 8.3 for heterogeneity), so it must
    give the larger single-factor probability. Pins the clause order so the
    two conditional probabilities cannot silently swap back."""
    perf = pm.PUBLISHED_MODELS["kawahara_2012"]["performance"]
    assert "85.3% with unclear interface" in perf
    assert "53.3% with heterogeneous enhancement" in perf


def test_kawahara_surrogate_note_quotes_both_published_effects():
    """The caveat only lands if it names the published effect of the variable we
    actually have (margin, 10.3) beside the one we substitute for (71.8)."""
    note = pm.PUBLISHED_MODELS["kawahara_2012"]["surrogate_note"]
    assert "10.3" in note and "71.8" in note


def test_kawahara_surrogate_note_capsular_enhancement_or_is_negative():
    """Final whole-branch review, Blocker 2: OR 19.2 (5.4-69) is published for
    NEGATIVE capsular enhancement -- its absence. Dropping "negative" reverses
    the clinical claim, and it also makes this paper read as agreeing with
    lin_2014 (which fits capsular_enhancement as PRESENT scoring toward high
    grade in this same report) when the two published findings actually point
    opposite directions."""
    note = pm.PUBLISHED_MODELS["kawahara_2012"]["surrogate_note"]
    assert "19.2" in note
    idx = note.index("19.2")
    assert "negative capsular enhancement" in note[:idx]


def test_kawahara_cohort_states_its_who_grading_edition():
    """The grading criteria changed across WHO editions (see radeesri_2023,
    which states 2021); a cohort description without one is not comparable to
    a cohort description that has one."""
    assert "WHO 2000" in pm.PUBLISHED_MODELS["kawahara_2012"]["cohort"]


def test_kawahara_surrogate_note_has_no_unsourced_p_value():
    """"each p<0.001" for the Table 2 univariable capsular-enhancement/margin
    odds ratios could not be confirmed in any reachable source and must not
    appear in a manuscript-facing string."""
    note = pm.PUBLISHED_MODELS["kawahara_2012"]["surrogate_note"]
    assert "p<0.001" not in note


def test_zhang_carries_beta_not_odds_ratios():
    for term in pm.PUBLISHED_MODELS["zhang_2020"]["terms"]:
        assert term.get("beta") is not None
        assert term.get("or") in (None, "")


def test_every_mapped_column_exists_in_the_cohort():
    df_cols = set(__import__("pandas").read_parquet(
        Path("output/datasets/unimputed_df.parquet")).columns)
    for mid in EXPECTED:
        for term in pm.PUBLISHED_MODELS[mid]["terms"]:
            col = term.get("column")
            if col:
                assert col in df_cols, f"{mid}: {col} missing from the cohort"


def test_kawahara_and_lin_each_flag_their_capsular_enhancement_conflict():
    """The two papers point opposite ways on capsular enhancement: Kawahara
    scored its ABSENCE as the high-grade finding, Lin scores its PRESENCE. Both
    transcriptions are correct, so a reader comparing the two tables sees a
    contradiction with no explanation unless each entry says so."""
    for mid, other in (("kawahara_2012", "Lin"), ("lin_2014", "Kawahara")):
        note = pm.PUBLISHED_MODELS[mid].get("cross_reference", "")
        assert other in note, mid
        assert "capsular enhancement" in note.lower(), mid
        assert "ABSENCE" in note and "PRESENCE" in note, mid


def test_only_the_two_conflicting_papers_carry_a_cross_reference():
    with_x = {k for k, v in pm.PUBLISHED_MODELS.items() if v.get("cross_reference")}
    assert with_x == {"kawahara_2012", "lin_2014"}
