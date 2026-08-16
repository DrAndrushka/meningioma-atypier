"""EDA screening helpers — effect sign conventions and figure content."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import eda
from schema_infer import ColSpec


def test_mwu_rank_biserial_follows_which_group_is_higher():
    g1 = np.array([8.0, 10.0, 12.0, 15.0, 20.0])
    g0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _U, _p, r, _n = eda._mwu_with_effect(g1, g0)
    assert r == 1.0

    _U, _p, r, _n = eda._mwu_with_effect(g0, g1)
    assert r == -1.0


def test_phi_is_signed_by_whether_present_is_risk_or_protective():
    # [[n00, n01], [n10, n11]] — present (row1) less often has y=1
    table = np.array([[3.0, 6.0], [243.0, 99.0]])
    assert eda._phi_coef(table) < 0
    row = eda._chi2_row(table)
    assert row["effect_label"] == "phi"
    assert row["effect"] < 0

    assert eda._phi_coef(np.array([[198.0, 65.0], [49.0, 39.0]])) > 0

    # The 2×2 builder is what locks that sign down.
    x = pd.Series([True, True, False, False, True, False])
    y = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    table = eda._ordered_binary_2x2(eda._binary_predictor_scores(x), y)
    assert table.shape == (2, 2)
    assert table[1, 1] == 2  # present & y=1
    assert eda._phi_coef(table) > 0


def test_the_heatmap_keeps_signs_flags_unsigned_effects_and_prints_two_decimals():
    """Unsigned effects are marked so red never silently means 'positive', and
    one decimal would collapse distinguishable effect sizes onto one label."""
    assert eda._heatmap_signed_effect(
        pd.Series({"effect": -0.13, "effect_label": "phi"})) == -0.13
    assert eda._heatmap_signed_effect(
        pd.Series({"effect": 0.18, "effect_label": "cramers_v"})) == 0.18

    assert eda._heatmap_is_unsigned(pd.Series({"effect_label": "cramers_v"}))
    assert eda._heatmap_is_unsigned(pd.Series({"effect_label": "epsilon_sq"}))
    assert not eda._heatmap_is_unsigned(pd.Series({"effect_label": "phi"}))
    assert not eda._heatmap_is_unsigned(pd.Series({"effect_label": "spearman_rho"}))

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

def test_level_order_pins_the_positive_class_last_and_keeps_clinical_order():
    s = pd.Series(["left", "right", "left", "midline"])
    spec = ColSpec("side", "nominal", positive_class="right")
    order = eda._level_order(s, spec)
    assert order[-1] == "right"
    assert set(order) == {"left", "right", "midline"}

    s = pd.Series(["<50", "80+", "50-59", "<50"])
    spec = ColSpec(
        "age_bins", "ordinal",
        ordered_levels=["<50", "50-59", "60-69", "70-79", "80+"],
        positive_class="50-59",
    )
    assert eda._level_order(s, spec) == ["<50", "50-59", "80+"]


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


def test_screen_associations_plots_every_tested_pair_and_quotes_q(tmp_path):
    """Figures are drawn after FDR adjustment, so they can quote q."""
    df, schema = _tiny_cohort()
    captured: list[dict] = []
    original = eda._plot_pair

    def spy(*args, **kwargs):
        captured.append(kwargs.get("result") or {})
        return original(*args, **kwargs)

    eda._plot_pair = spy
    try:
        out = eda.screen_associations(
            df, schema, targets=["high_grade"], predictors=["age", "margin"],
            output_root=tmp_path,
        )
    finally:
        eda._plot_pair = original

    figs = {p.stem for p in (tmp_path / "eda" / "figures").glob("*.png")}
    assert {"high_grade__age", "high_grade__margin"} <= figs
    assert set(out["predictor"]) == {"age", "margin"}
    assert captured
    assert all("p_fdr" in result for result in captured)


@pytest.mark.parametrize(
    "listing", ["eda_excluded_columns.csv", "hidden_parent_columns.csv"],
)
def test_screen_associations_skips_the_columns_cleaning_took_out(tmp_path, listing):
    df, schema = _tiny_cohort()
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["margin"]}).to_csv(cleaning / listing, index=False)
    out = eda.screen_associations(
        df, schema, targets=["high_grade"], predictors=["age", "margin"],
        output_root=tmp_path,
    )
    assert set(out["predictor"]) == {"age"}
    figs = {p.stem for p in (tmp_path / "eda" / "figures").glob("*.png")}
    assert "high_grade__age" in figs
    assert "high_grade__margin" not in figs


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

    # With no family named, everything is in it.
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        output_root=tmp_path / "all",
    )
    assert out["in_fdr_family"].all()
    assert out["p_fdr"].notna().all()


def test_derived_columns_get_separate_fdr_and_stay_out_of_native_family(tmp_path):
    df, schema = _fdr_family_fixture()
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["vol_ge10"]}).to_csv(
        cleaning / "eda_derived_columns.csv", index=False,
    )
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        fdr_family=["vol", "vol_ge10", "adc"],
        output_root=tmp_path,
    )
    native = out[out["in_fdr_family"]]
    derived = out[~out["in_fdr_family"]]
    assert sorted(native["predictor"]) == ["adc", "vol"]
    assert list(derived["predictor"]) == ["vol_ge10"]
    assert derived["p_fdr"].notna().all()
    expected_native = eda.benjamini_hochberg(native["p"])
    assert np.allclose(native["p_fdr"].values, expected_native.values)
    # one derived test → q equals raw p
    assert np.allclose(derived["p_fdr"].values, derived["p"].values)


# ---------------------------------------------------------------------------
# Ordinal predictor vs binary target: Cochran–Armitage trend test
# ---------------------------------------------------------------------------

def _unbalanced_ordinal() -> tuple[pd.DataFrame, dict]:
    """Three ordered levels where the rarest level carries the signal.

    Rank scores weight a level by how many patients are in it, so they dilute
    a small top category; equal-spacing scores do not. The two therefore give
    different answers on this cohort, which is what pins the scoring down.
    """
    levels = ["none", "some", "extensive"]
    counts = {"none": (190, 72), "some": (44, 20), "extensive": (13, 13)}
    rows = []
    for lv, (n_low, n_high) in counts.items():
        rows += [{"stage": lv, "high_grade": False}] * n_low
        rows += [{"stage": lv, "high_grade": True}] * n_high
    df = pd.DataFrame(rows)
    df["high_grade"] = df["high_grade"].astype("boolean")
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "stage": ColSpec("stage", "ordinal", ordered_levels=levels),
    }
    return df, schema


def _linear_by_linear_p(codes: np.ndarray, y: np.ndarray) -> float:
    """Reference Cochran–Armitage p: M² = (n − 1)·r², one degree of freedom."""
    from scipy.stats import chi2

    n = len(codes)
    r = np.corrcoef(codes.astype(float), y.astype(float))[0, 1]
    return float(chi2.sf((n - 1) * r * r, 1))


def test_an_ordinal_predictor_uses_cochran_armitage_with_equal_spacing(tmp_path):
    """Equal spacing keeps the rare extreme level at full weight, where rank
    scores would dilute it."""
    from scipy.stats import spearmanr

    df, schema = _unbalanced_ordinal()
    out = eda.screen_associations(
        df, schema, targets=["high_grade"], predictors=["stage"],
        output_root=tmp_path,
    )
    row = out.iloc[0]
    assert row["test"] == "cochran_armitage"

    codes = pd.Categorical(
        df["stage"], categories=["none", "some", "extensive"], ordered=True,
    ).codes
    y = df["high_grade"].astype(float).to_numpy()
    assert row["p"] == pytest.approx(_linear_by_linear_p(codes, y), rel=1e-9)

    rank_p = float(spearmanr(y, codes).pvalue)
    assert row["p"] < rank_p            # equal spacing is the stronger test here
    assert rank_p > 0.05 > row["p"]     # and the two land on opposite verdicts


def test_cochran_armitage_reads_the_declared_level_order(tmp_path):
    """Reversing the clinical order flips the sign, not just the label."""
    df, schema = _unbalanced_ordinal()
    forward = eda.screen_associations(
        df, schema, targets=["high_grade"], predictors=["stage"],
        output_root=tmp_path,
    ).iloc[0]
    schema["stage"] = ColSpec(
        "stage", "ordinal", ordered_levels=["extensive", "some", "none"],
    )
    reverse = eda.screen_associations(
        df, schema, targets=["high_grade"], predictors=["stage"],
        output_root=tmp_path / "rev",
    ).iloc[0]
    assert forward["effect"] > 0 > reverse["effect"]
    assert forward["effect"] == pytest.approx(-reverse["effect"])
    assert forward["p"] == pytest.approx(reverse["p"])


def test_cochran_armitage_handles_rows_it_cannot_score():
    """One level, or one outcome class, leaves the row empty rather than
    crashing; codes are NaN for values outside ``ordered_levels`` and those
    must not poison r either."""
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    one_level = eda._cochran_armitage_row(y, np.zeros(6))
    assert one_level["test"] == "cochran_armitage"
    assert np.isnan(one_level["p"]) and np.isnan(one_level["effect"])
    one_class = eda._cochran_armitage_row(np.ones(6), np.array([0.0, 1, 2, 0, 1, 2]))
    assert np.isnan(one_class["p"])

    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, np.nan])
    x = np.array([0.0, 1.0, 0.0, 1.0, np.nan, 2.0, 1.0])
    row = eda._cochran_armitage_row(y, x)
    keep = np.isfinite(y) & np.isfinite(x)
    assert row["p"] == pytest.approx(_linear_by_linear_p(x[keep], y[keep]))


def test_derived_predictors_still_get_their_own_q():
    """Excluded from the native family, not from correction altogether."""
    out = pd.DataFrame({
        "target": ["y"] * 3,
        "predictor": ["adc_value", "adc_value_le0.72", "tumor_volume_ge15.1"],
        "p": [0.01, 0.02, 0.04],
    })
    got = eda.apply_native_and_derived_fdr(
        out, derived_cols={"adc_value_le0.72", "tumor_volume_ge15.1"},
        fdr_family=["adc_value"],
    )
    assert got["p_fdr"].notna().all()
    assert list(got["in_fdr_family"]) == [True, False, False]
