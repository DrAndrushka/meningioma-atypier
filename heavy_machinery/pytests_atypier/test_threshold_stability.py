"""Across-draw stability and the ``output/thresholds/`` export."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import combinations as cb
import stability as st
from artifacts import ThresholdArtifacts
from thresholds import Metric, threshold_summary

TARGET = "high_grade"
VOL = Metric("vol", "Volume", "cc", "higher")
ADC = Metric("adc", "ADC", "unit", "lower")
METRICS = [VOL, ADC]


def draw(seed: int, n: int = 300, shift: float = 0.0) -> pd.DataFrame:
    """One 'imputed' dataset. ``shift`` mimics a draw that guessed differently.

    The two metrics carry independent noise on purpose: a mirrored pair would
    flag exactly the same patients, so a count of "1 of 2" could never occur
    and the count-score tests would be checking an impossible column.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    vol = rng.normal(10, 3, n) + y * 5.0 + shift
    adc = -(rng.normal(10, 3, n) + y * 4.0 + shift)
    return pd.DataFrame({
        "vol": vol,
        "adc": adc,
        TARGET: pd.array(y, dtype="boolean"),
    })


@pytest.fixture
def frames() -> list[pd.DataFrame]:
    return [draw(seed=i, shift=0.2 * i) for i in range(6)]


@pytest.fixture
def observed() -> pd.DataFrame:
    return draw(seed=100)


# --------------------------------------------------------------------------
# Cut-points across draws
# --------------------------------------------------------------------------
def test_draw_cutoffs_has_a_row_per_draw_metric_rule(frames):
    out = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden", "closest_01"])
    assert len(out) == len(frames) * len(METRICS) * 2
    assert set(out["draw"]) == set(range(1, len(frames) + 1))


def test_stability_table_summarises_each_metric_and_rule(frames, observed):
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    summary = threshold_summary(observed, METRICS, TARGET, rules=["youden"], n_boot=20)
    out = st.imputation_stability(draws, frames, METRICS, TARGET, summary)
    assert len(out) == len(METRICS)
    assert (out["m_draws"] == len(frames)).all()
    assert (out["cutoff_min"] <= out["cutoff_mean"]).all()
    assert (out["cutoff_mean"] <= out["cutoff_max"]).all()


def test_shift_column_is_the_gap_to_the_complete_case_cutpoint(frames, observed):
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    summary = threshold_summary(observed, METRICS, TARGET, rules=["youden"], n_boot=20)
    out = st.imputation_stability(draws, frames, METRICS, TARGET, summary)
    expected = out["cutoff_mean"] - out["cutoff_complete_case"]
    assert np.allclose(out["shift_vs_complete_case"], expected)


def test_sens_at_mean_is_not_better_than_the_per_draw_optimum(frames, observed):
    """Applying one averaged cut-point cannot beat cut-points chosen per draw."""
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    summary = threshold_summary(observed, METRICS, TARGET, rules=["youden"], n_boot=20)
    out = st.imputation_stability(draws, frames, METRICS, TARGET, summary)
    j_at_mean = out["sens_at_mean"] + out["spec_at_mean"] - 1.0
    assert (j_at_mean <= out["J_mean"] + 1e-9).all()


def test_stability_reading_view_is_all_strings(frames, observed):
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    summary = threshold_summary(observed, METRICS, TARGET, rules=["youden"], n_boot=20)
    table = st.imputation_stability(draws, frames, METRICS, TARGET, summary)
    view = st.stability_reading_view(table)
    assert "MICE mean (m=6)" in view.columns
    assert view["Complete-case"].str.startswith(("≥", "≤")).all()


def test_several_published_cutpoints_for_one_metric_do_not_break_the_lookup(
        frames, observed):
    """Three published cut-points for the same measurement (max diameter has
    exactly that) share the key ("column", "literature"). Looking that key up
    returns a frame where a number is expected, which crashed the run."""
    summary = threshold_summary(
        observed, METRICS, TARGET, rules=["youden"], n_boot=20,
        literature_cutoffs={"vol": [(9.0, "A 2018"), (12.0, "B 2021"),
                                    (14.0, "C 2022")]},
    )
    assert (summary["rule"] == "literature").sum() == 3

    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    out = st.imputation_stability(draws, frames, METRICS, TARGET, summary)

    assert len(out) == len(METRICS)
    # The published rows carry no per-draw counterpart, so they must not appear.
    assert set(out["rule"]) == {"youden"}
    assert out["cutoff_complete_case"].notna().all()


def test_stability_survives_an_empty_complete_case_table(frames):
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    out = st.imputation_stability(draws, frames, METRICS, TARGET, pd.DataFrame())
    assert len(out) == len(METRICS)
    assert "cutoff_complete_case" not in out.columns


# --------------------------------------------------------------------------
# Risk curves across draws
# --------------------------------------------------------------------------
def test_draw_risk_curves_and_their_summary(frames):
    draws = st.draw_risk_curves(frames, [VOL], TARGET)
    assert len(draws) == len(frames)
    out = st.risk_curve_stability(draws)
    assert len(out) == 1
    row = out.iloc[0]
    assert 0.0 <= row["knee_rate"] <= 1.0
    assert 0.0 <= row["nonlinear_rate"] <= 1.0
    assert row["m_draws"] == len(frames)


def test_knee_rate_is_zero_when_no_draw_finds_one(frames):
    draws = st.draw_risk_curves(frames, [VOL], TARGET)
    draws["knee_found"] = False
    out = st.risk_curve_stability(draws)
    assert out.iloc[0]["knee_rate"] == 0.0
    assert np.isnan(out.iloc[0]["steepest_median"])


# --------------------------------------------------------------------------
# Combinations across draws
# --------------------------------------------------------------------------
def test_draw_count_scores_average_the_draws(frames):
    cps = cb.cutpoints_for_rule(frames[0], METRICS, TARGET, "youden")
    out = st.draw_count_scores(frames, cps, TARGET)
    assert list(out["n_criteria_met"]) == [0, 1, 2]
    assert out.attrs["m_draws"] == len(frames)
    assert (out["risk_min"] <= out["risk"]).all()
    assert (out["risk"] <= out["risk_max"]).all()


def test_draw_count_rules_report_a_range(frames):
    cps = cb.cutpoints_for_rule(frames[0], METRICS, TARGET, "youden")
    out = st.draw_count_rules(frames, cps, TARGET)
    assert len(out) == len(cps)
    assert (out["sens_min"] <= out["sensitivity"]).all()
    assert (out["sensitivity"] <= out["sens_max"]).all()


def test_flag_missingness_counts_scorable_patients():
    df = pd.DataFrame({
        "vol": pd.array([1.0, 2.0, None, 4.0], dtype="Float64"),
        "adc": pd.array([1.0, None, 3.0, 4.0], dtype="Float64"),
        TARGET: pd.array([True, False, True, False], dtype="boolean"),
    })
    cps = [cb.CutPoint(VOL, 2.0), cb.CutPoint(ADC, 2.0)]
    out = st.flag_missingness(df, cps).iloc[0]
    assert out["n_patients"] == 4
    assert out["n_all_flags_observed"] == 2
    assert out["n_some_flag_missing"] == 2


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def test_stability_figure_builds(frames, observed):
    draws = st.draw_cutoffs(frames, METRICS, TARGET, rules=["youden"])
    summary = threshold_summary(observed, METRICS, TARGET, rules=["youden"], n_boot=20)
    table = st.imputation_stability(draws, frames, METRICS, TARGET, summary)
    fig = st.stability_figure(draws, table, METRICS)
    assert len(fig.get_axes()) == len(METRICS)
    plt.close(fig)


def test_knee_stability_figure_builds_even_with_no_knees(frames):
    draws = st.draw_risk_curves(frames, METRICS, TARGET)
    draws["knee_found"] = False
    fig = st.knee_stability_figure(draws, METRICS)
    assert fig.get_axes()
    plt.close(fig)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def test_artifacts_write_figures_tables_and_manifest(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "thresholds", context={"n": 352})
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    art.figure(fig, "demo", caption="a line")
    art.table(pd.DataFrame({"a": [1, 2]}), "demo", caption="two rows")
    art.note("edema volume has no 50% crossing")
    manifest_path = art.write_manifest()

    assert (tmp_path / "thresholds" / "figures" / "demo.svg").exists()
    assert (tmp_path / "thresholds" / "tables" / "demo.csv").exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["context"]["n"] == 352
    assert payload["figures"][0]["name"] == "demo.svg"
    assert payload["tables"][0]["rows"] == 2
    assert payload["notes"] == ["edema volume has no 50% crossing"]


def test_artifacts_disabled_writes_nothing(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "thresholds", enabled=False)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    art.figure(fig, "demo")
    art.table(pd.DataFrame({"a": [1]}), "demo")
    assert art.write_manifest() is None
    assert not (tmp_path / "thresholds").exists()
    assert art.written().empty


def test_artifacts_do_not_duplicate_the_suffix(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "t")
    art.table(pd.DataFrame({"a": [1]}), "named.csv")
    assert (tmp_path / "t" / "tables" / "named.csv").exists()


def test_artifacts_return_the_untouched_frame(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "t")
    table = pd.DataFrame({"value": [1.23456789]})
    returned = art.table(table, "rounded")
    assert returned is table  # rounding applies to the CSV, not the working frame
    assert returned["value"].iloc[0] == 1.23456789


def test_stale_files_lists_what_this_run_did_not_write(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "t")
    art.table(pd.DataFrame({"a": [1]}), "kept")
    (tmp_path / "t" / "tables" / "leftover.csv").write_text("old\n")
    stale = art.stale_files()
    assert [p.name for p in stale] == ["leftover.csv"]
    assert (tmp_path / "t" / "tables" / "leftover.csv").exists()  # never deleted


def test_written_receipt_lists_both_kinds(tmp_path):
    art = ThresholdArtifacts(root=tmp_path / "t")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    art.figure(fig, "f")
    art.table(pd.DataFrame({"a": [1]}), "t")
    receipt = art.written()
    assert set(receipt["kind"]) == {"figure", "table"}
