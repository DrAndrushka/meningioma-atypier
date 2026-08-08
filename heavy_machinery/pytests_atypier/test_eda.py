"""EDA screening helpers — effect sign conventions and figure content."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import eda
from schema_infer import ColSpec


def test_mwu_rank_biserial_positive_when_group1_higher():
    g1 = np.array([8.0, 10.0, 12.0, 15.0, 20.0])
    g0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _U, _p, r, _n = eda._mwu_with_effect(g1, g0)
    assert r == 1.0


def test_mwu_rank_biserial_negative_when_group1_lower():
    g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    g0 = np.array([8.0, 10.0, 12.0, 15.0, 20.0])
    _U, _p, r, _n = eda._mwu_with_effect(g1, g0)
    assert r == -1.0


def test_phi_negative_when_present_protective():
    # [[n00, n01], [n10, n11]] — present (row1) less often has y=1
    table = np.array([[3.0, 6.0], [243.0, 99.0]])
    phi = eda._phi_coef(table)
    assert phi < 0
    row = eda._chi2_row(table)
    assert row["effect_label"] == "phi"
    assert row["effect"] < 0


def test_phi_positive_when_present_risk():
    table = np.array([[198.0, 65.0], [49.0, 39.0]])
    phi = eda._phi_coef(table)
    assert phi > 0


def test_ordered_binary_2x2_locks_sign():
    x = pd.Series([True, True, False, False, True, False])
    y = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    table = eda._ordered_binary_2x2(eda._binary_predictor_scores(x), y)
    assert table.shape == (2, 2)
    assert table[1, 1] == 2  # present & y=1
    assert eda._phi_coef(table) > 0


def test_heatmap_signed_effect_keeps_phi_sign():
    row = pd.Series({"effect": -0.13, "effect_label": "phi"})
    assert eda._heatmap_signed_effect(row) == -0.13
    row_v = pd.Series({"effect": 0.18, "effect_label": "cramers_v"})
    assert eda._heatmap_signed_effect(row_v) == 0.18


def test_heatmap_flags_magnitude_only_effects():
    """Unsigned effects are marked so red never silently means 'positive'."""
    assert eda._heatmap_is_unsigned(pd.Series({"effect_label": "cramers_v"}))
    assert eda._heatmap_is_unsigned(pd.Series({"effect_label": "epsilon_sq"}))
    assert not eda._heatmap_is_unsigned(pd.Series({"effect_label": "phi"}))
    assert not eda._heatmap_is_unsigned(pd.Series({"effect_label": "spearman_rho"}))


def test_heatmap_cell_text_keeps_two_decimals():
    """One decimal collapses distinguishable effect sizes onto one label."""
    row = pd.Series({"test": "chi2", "p_fdr": 0.01, "effect": 0.243})
    assert eda._heatmap_cell_text(row, fdr_alpha=0.05) == "0.24*"
    weak = pd.Series({"test": "chi2", "p_fdr": 0.01, "effect": 0.164})
    assert eda._heatmap_cell_text(weak, fdr_alpha=0.05) == "0.16*"
    # Non-significant cells stay colour-only.
    ns = pd.Series({"test": "chi2", "p_fdr": 0.4, "effect": 0.24})
    assert eda._heatmap_cell_text(ns, fdr_alpha=0.05) == ""


# ---------------------------------------------------------------------------
# Figure content
# ---------------------------------------------------------------------------

def test_result_subtitle_reports_test_effect_and_fdr():
    row = {
        "test": "mann_whitney_u", "effect": 0.187,
        "effect_label": "rank_biserial_r", "p": 0.0081, "p_fdr": 0.0412,
    }
    text = eda._result_subtitle(row, 352, fdr_alpha=0.05)
    assert "n = 352" in text
    assert "Mann–Whitney U" in text
    assert "r = 0.19" in text
    assert "p = 0.008" in text
    assert "q = 0.041" in text
    assert "FDR-significant" in text


def test_result_subtitle_omits_the_flag_when_not_significant():
    row = {"test": "chi2", "effect": 0.06, "effect_label": "phi",
           "p": 0.264, "p_fdr": 0.264}
    text = eda._result_subtitle(row, 352, fdr_alpha=0.05)
    assert "q = 0.264" in text
    assert "FDR-significant" not in text


def _tiny_cohort() -> tuple[pd.DataFrame, dict[str, ColSpec]]:
    rng = np.random.default_rng(0)
    n = 60
    high = rng.choice([False, True], n)
    df = pd.DataFrame({
        "high_grade": high,
        "age": rng.normal(65, 10, n),
        "margin": np.where(high, "irregular", "regular"),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "age": ColSpec("age", "continuous"),
        "margin": ColSpec("margin", "nominal"),
    }
    return df, schema


def test_screen_associations_plots_every_tested_pair(tmp_path):
    df, schema = _tiny_cohort()
    out = eda.screen_associations(
        df, schema, targets=["high_grade"], predictors=["age", "margin"],
        output_root=tmp_path,
    )
    figs = {p.stem for p in (tmp_path / "eda" / "figures").glob("*.svg")}
    assert {"high_grade__age", "high_grade__margin"} <= figs
    assert set(out["predictor"]) == {"age", "margin"}


def test_pair_figures_carry_the_fdr_adjusted_q_value(tmp_path):
    """Figures are drawn after FDR adjustment, so they can quote q."""
    df, schema = _tiny_cohort()
    captured: list[dict] = []
    original = eda._plot_pair

    def spy(*args, **kwargs):
        captured.append(kwargs.get("result") or {})
        return original(*args, **kwargs)

    eda._plot_pair = spy
    try:
        eda.screen_associations(
            df, schema, targets=["high_grade"], predictors=["age", "margin"],
            output_root=tmp_path,
        )
    finally:
        eda._plot_pair = original

    assert captured
    assert all("p_fdr" in result for result in captured)


def _fdr_family_fixture():
    rng = np.random.default_rng(7)
    n = 80
    y = pd.Series(rng.integers(0, 2, n).astype(bool), name="high_grade")
    df = pd.DataFrame({
        "high_grade": y,
        "vol": rng.normal(10, 3, n) + y * 2.0,
        "vol_ge10": pd.array(rng.integers(0, 2, n).astype(bool)),
        "adc": rng.normal(1.0, 0.2, n) - y * 0.1,
    })
    schema = {
        "high_grade": ColSpec(name="high_grade", kind="binary"),
        "vol": ColSpec(name="vol", kind="continuous"),
        "vol_ge10": ColSpec(name="vol_ge10", kind="binary"),
        "adc": ColSpec(name="adc", kind="continuous"),
    }
    return df, schema


def test_fdr_family_limits_correction_to_family(tmp_path):
    df, schema = _fdr_family_fixture()
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        fdr_family=["vol", "adc"],
        output_root=tmp_path,
    )
    fam = out[out["in_fdr_family"]]
    extra = out[~out["in_fdr_family"]]
    assert sorted(fam["predictor"]) == ["adc", "vol"]
    assert list(extra["predictor"]) == ["vol_ge10"]
    # excluded row: raw p kept, no corrected p, never flagged significant
    assert extra["p"].notna().all()
    assert extra["p_fdr"].isna().all()
    assert not extra["fdr_significant"].any()
    # family rows: BH computed over exactly the 2 family tests
    expected = eda.benjamini_hochberg(fam["p"])
    assert np.allclose(fam["p_fdr"].values, expected.values)


def test_fdr_family_none_keeps_everything_in_family(tmp_path):
    df, schema = _fdr_family_fixture()
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        output_root=tmp_path,
    )
    assert out["in_fdr_family"].all()
    assert out["p_fdr"].notna().all()
