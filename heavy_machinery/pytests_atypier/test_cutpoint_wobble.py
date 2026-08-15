"""Step 7 — the cut-point's own interval, and how much of its performance was luck."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import criteria as cr
import measurements as ms
import wobble as wb


def _data(seed: int = 0, n: int = 300, sep: float = 0.9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(y * sep, 1.0)


# --- the settings this phase shares with the rest of the project ----------
def test_the_seed_matches_the_other_phases():
    """A different seed desynchronises this from the modelling-phase bootstraps."""
    assert wb.SEED == 20260801


def test_the_default_is_a_thousand_replicates():
    assert wb.N_BOOTSTRAP == 1000


# --- the interval ----------------------------------------------------------
def test_the_interval_brackets_the_observed_cutpoint():
    y, x = _data()
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=200)
    assert out["ci_lo"] <= out["cutpoint"] <= out["ci_hi"]


def test_the_inner_quartiles_sit_inside_the_interval():
    y, x = _data()
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=200)
    assert out["ci_lo"] <= out["iqr_lo"] <= out["iqr_hi"] <= out["ci_hi"]


def test_the_same_seed_gives_the_same_interval():
    y, x = _data()
    a = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=150, seed=7)
    b = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=150, seed=7)
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_a_different_seed_gives_a_different_interval():
    y, x = _data()
    a = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=150, seed=7)
    b = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=150, seed=8)
    assert (a["ci_lo"], a["ci_hi"]) != (b["ci_lo"], b["ci_hi"])


def test_a_stronger_signal_gives_a_tighter_interval():
    y, x_weak = _data(seed=1, sep=0.3)
    _, x_strong = _data(seed=1, sep=3.0)
    weak = wb.bootstrap_cutpoint(y, x_weak, ms.HIGHER, n_boot=300)
    strong = wb.bootstrap_cutpoint(y, x_strong, ms.HIGHER, n_boot=300)
    assert (strong["ci_hi"] - strong["ci_lo"]) < (weak["ci_hi"] - weak["ci_lo"])


def test_the_cutpoint_is_rederived_in_every_replicate_not_held_fixed():
    """A fixed cut-point would give a zero-width interval — the classic error."""
    y, x = _data()
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=200)
    assert np.unique(out["draws"]).size > 1


def test_the_draws_are_kept_for_the_histogram():
    y, x = _data()
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=120)
    assert out["draws"].size == out["n_valid"]


# --- optimism --------------------------------------------------------------
def test_optimism_is_positive_when_the_criterion_is_chasing_noise():
    y, x = _data(seed=4, sep=0.0)       # pure noise: all apparent J is luck
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=300)
    assert out["optimism"] > 0


def test_the_corrected_j_is_below_the_apparent_j():
    y, x = _data()
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=300)
    assert out["j_corrected"] < out["j_apparent"]


def test_optimism_scores_held_out_patients_not_the_ones_it_learned_from():
    """Scoring in-bag against in-bag would report zero optimism, always."""
    y, x = _data(seed=6, sep=0.2)
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=300)
    assert np.isfinite(out["optimism"]) and out["optimism"] != 0.0


# --- refusals and graceful failure ----------------------------------------
def test_one_grade_only_returns_blanks_rather_than_raising():
    out = wb.bootstrap_cutpoint(np.ones(50, dtype=int), np.arange(50.0),
                                ms.HIGHER, n_boot=50)
    assert np.isnan(out["cutpoint"]) and out["n_valid"] == 0


def test_too_few_events_returns_blanks():
    y = np.zeros(60, dtype=int)
    y[:2] = 1
    out = wb.bootstrap_cutpoint(y, np.linspace(0, 1, 60), ms.HIGHER, n_boot=50)
    assert out["n_valid"] == 0


def test_skipped_replicates_are_counted_not_hidden():
    y, x = _data(n=60)
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=100)
    assert out["n_valid"] + out["n_skipped"] == 100


def test_missing_values_are_dropped_before_resampling():
    y, x = _data(n=200)
    x = x.copy()
    x[:20] = np.nan
    out = wb.bootstrap_cutpoint(y, x, ms.HIGHER, n_boot=100)
    assert np.isfinite(out["cutpoint"])


# --- the frozen point estimates -------------------------------------------
def _cohort(seed: int = 5, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    edema = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.12 * y, 0.15),
        "tumor_volume": rng.gamma(2, 8, n) + 4 * y,
        "edema_volume_cm3": edema,
        "edema_index": edema / (rng.gamma(2, 8, n) + 1),
        "max_diameter_cm": rng.normal(3.8 + 0.4 * y, 1.2),
    })


def _eligible(df):
    import bend_location as bl
    import eligibility as el
    import nonlinearity as nl
    import separation as sep
    fits = nl.fit_all(df)
    return el.eligible(el.carry_forward(sep.separation_table(df),
                                        bl.bend_table(df, fits=fits)))


def test_a_moved_point_estimate_stops_the_run():
    """The bootstrap may bracket a published number; it may not move it."""
    df = _cohort()
    with pytest.raises(wb.FrozenCutpointError, match="may not"):
        wb.wobble_table(df, _eligible(df), n_boot=40,
                        frozen={"adc_value": 0.01})


def test_a_matching_point_estimate_passes_the_freeze():
    df = _cohort()
    table, _ = wb.wobble_table(df, _eligible(df), n_boot=40)
    adc = table.set_index("col").loc["adc_value", "cutpoint"]
    wb.wobble_table(df, _eligible(df), n_boot=40, frozen={"adc_value": adc})


# --- the table -------------------------------------------------------------
def test_the_stability_ratio_is_the_interval_over_the_measurements_iqr():
    df = _cohort()
    table, _ = wb.wobble_table(df, _eligible(df), n_boot=60)
    assert table["stability_ratio"].to_numpy() == pytest.approx(
        (table["ci_width"] / table["measurement_iqr"]).to_numpy())


def test_draws_come_back_keyed_by_measurement():
    df = _cohort()
    table, draws = wb.wobble_table(df, _eligible(df), n_boot=60)
    assert set(draws) == set(table["col"])


def test_the_claim_from_step_five_is_carried_forward():
    df = _cohort()
    table, _ = wb.wobble_table(df, _eligible(df), n_boot=40)
    assert table["claim"].str.len().gt(0).all()


# --- the summary line ------------------------------------------------------
def test_describe_separates_firm_cutpoints_from_dissolved_ones():
    df = _cohort()
    table, _ = wb.wobble_table(df, _eligible(df), n_boot=60)
    table["stability_ratio"] = 3.0
    assert "not a landmark" in wb.describe_wobble(table)
    table["stability_ratio"] = 0.2
    assert "Holds up under resampling" in wb.describe_wobble(table)


def test_describe_handles_an_empty_table():
    assert wb.describe_wobble(pd.DataFrame()) == "No cut-point could be resampled."


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_the_real_cohort_keeps_all_five_published_cutpoints(real_cohort):
    table, _ = wb.wobble_table(real_cohort, _eligible(real_cohort),
                               n_boot=100, frozen=_FROZEN)
    assert len(table) == 5
    assert table["n_skipped"].sum() == 0
