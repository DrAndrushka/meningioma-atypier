"""Tests for config/published_models.py — transcription integrity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_kawahara_surrogate_note_quotes_both_published_effects():
    """The caveat only lands if it names the published effect of the variable we
    actually have (margin, 10.3) beside the one we substitute for (71.8)."""
    note = pm.PUBLISHED_MODELS["kawahara_2012"]["surrogate_note"]
    assert "10.3" in note and "71.8" in note


def test_zhang_carries_beta_not_odds_ratios():
    for term in pm.PUBLISHED_MODELS["zhang_2020"]["terms"]:
        assert term.get("beta") is not None
        assert term.get("or") in (None, "")


def test_not_fitted_records_every_excluded_model_with_a_reason():
    assert set(pm.NOT_FITTED) == {
        "azeemuddin_2018", "yao_2022", "amano_2022",
        "duarte_gomes_quintas_neves_2026", "hale_2018",
    }
    assert all(v.strip() for v in pm.NOT_FITTED.values())


def test_every_mapped_column_exists_in_the_cohort():
    df_cols = set(__import__("pandas").read_parquet(
        Path("output/datasets/unimputed_df.parquet")).columns)
    for mid in EXPECTED:
        for term in pm.PUBLISHED_MODELS[mid]["terms"]:
            col = term.get("column")
            if col:
                assert col in df_cols, f"{mid}: {col} missing from the cohort"
