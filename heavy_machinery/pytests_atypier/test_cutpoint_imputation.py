"""Step 8 — what the scans we never measured cost the cut-point."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import imputation as imp
import measurements as ms
import wobble as wb


def _draw(seed: int, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    tumor = rng.gamma(2, 8, n) + 4 * y
    edema = np.where(rng.random(n) < 0.35, 0.0, rng.gamma(2, 5, n) + y)
    return pd.DataFrame({
        "high_grade": y,
        "adc_value": rng.normal(0.9 - 0.12 * y, 0.15),
        "tumor_volume": tumor,
        "edema_volume_cm3": edema,
        "edema_index": edema / tumor,
        "max_diameter_cm": rng.normal(3.8 + 0.4 * y, 1.2),
    })


def _draws(k: int = 6) -> list[pd.DataFrame]:
    return [_draw(seed) for seed in range(k)]


def _eligible(df):
    import bend_location as bl
    import eligibility as el
    import nonlinearity as nl
    import separation as sep
    fits = nl.fit_all(df)
    return el.eligible(el.carry_forward(sep.separation_table(df),
                                        bl.bend_table(df, fits=fits)))


# --- loading ---------------------------------------------------------------
def test_absent_draws_name_the_step_that_makes_them(tmp_path):
    with pytest.raises(imp.ImputationError, match="cleaning notebook"):
        imp.load_draws(tmp_path)


def test_the_real_run_has_twenty_draws():
    assert len(imp.load_draws("output")) == 20


# --- the derived-ratio guard ----------------------------------------------
def test_a_correctly_rebuilt_ratio_is_silent():
    assert imp.check_derived_consistency(_draws()) == []


def test_a_directly_imputed_ratio_is_caught():
    """Imputed on its own, the index can contradict the volumes beside it."""
    draws = _draws(2)
    draws[1] = draws[1].copy()
    draws[1]["edema_index"] = draws[1]["edema_index"] * 1.5 + 0.01
    notes = imp.check_derived_consistency(draws)
    assert len(notes) == 1 and "draw 2" in notes[0]


def test_the_real_draws_rebuild_the_ratio_from_its_parents():
    assert imp.check_derived_consistency(imp.load_draws("output")) == []


# --- per-draw cut-points ---------------------------------------------------
def test_one_row_per_eligible_measurement():
    draws = _draws()
    table = imp.per_draw_cutpoints(draws, _eligible(draws[0]))
    assert len(table) == len(_eligible(draws[0]))


def test_the_range_brackets_the_median():
    draws = _draws()
    table = imp.per_draw_cutpoints(draws, _eligible(draws[0]))
    assert (table["draw_min"] <= table["draw_median"]).all()
    assert (table["draw_median"] <= table["draw_max"]).all()


def test_identical_draws_give_zero_spread():
    """No missing data means nothing for imputation to disagree about."""
    same = [_draw(0) for _ in range(5)]
    table = imp.per_draw_cutpoints(same, _eligible(same[0]))
    assert (table["draw_spread"] == 0).all()


def test_the_median_is_used_not_the_mean():
    """Averaging the position of a maximum can land where no draw chose."""
    draws = _draws()
    elig = _eligible(draws[0])
    table = imp.per_draw_cutpoints(draws, elig).set_index("col")
    for col in table.index:
        m = ms.MEASUREMENTS_BY_COL[col]
        values = [imp._cutpoint_in(d, m, "all", "youden") for d in draws]
        assert table.loc[col, "draw_median"] == m.round(np.median(values))


# --- the divergence gate ---------------------------------------------------
def test_a_cutpoint_inside_the_range_does_not_diverge():
    draws = _draws()
    elig = _eligible(draws[0])
    plain = imp.per_draw_cutpoints(draws, elig).set_index("col")
    frozen = {col: plain.loc[col, "draw_median"] for col in plain.index}
    table = imp.per_draw_cutpoints(draws, elig, frozen=frozen)
    assert not table["diverges"].any()


def test_a_cutpoint_no_draw_reproduces_is_flagged():
    draws = _draws()
    elig = _eligible(draws[0])
    table = imp.per_draw_cutpoints(draws, elig, frozen={"adc_value": 99.0})
    assert table.set_index("col").loc["adc_value", "diverges"]


def test_divergence_is_judged_at_the_printed_precision():
    """A median rounding just outside a very tight range is not divergence."""
    same = [_draw(0) for _ in range(4)]
    elig = _eligible(same[0])
    plain = imp.per_draw_cutpoints(same, elig).set_index("col")
    value = plain.loc["adc_value", "draw_median"]
    table = imp.per_draw_cutpoints(same, elig, frozen={"adc_value": value})
    assert not table.set_index("col").loc["adc_value", "diverges"]


# --- the joint interval ----------------------------------------------------
def test_the_joint_interval_is_reproducible():
    draws = _draws()
    m = ms.MEASUREMENTS_BY_COL["adc_value"]
    a = imp.joint_bootstrap(draws, m, n_boot=120, seed=3)
    b = imp.joint_bootstrap(draws, m, n_boot=120, seed=3)
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_the_joint_interval_draws_an_imputation_per_replicate():
    """Fixing one draw would discard the between-imputation variance."""
    draws = _draws()
    m = ms.MEASUREMENTS_BY_COL["adc_value"]
    joint = imp.joint_bootstrap(draws, m, n_boot=200, seed=1)
    single = wb.bootstrap_cutpoint(
        draws[0]["high_grade"].to_numpy(), draws[0]["adc_value"].to_numpy(),
        m.direction, n_boot=200, seed=1)
    assert (joint["ci_lo"], joint["ci_hi"]) != (single["ci_lo"], single["ci_hi"])


def test_identical_draws_add_no_width_beyond_sampling_noise():
    """With nothing to disagree about, the extra source contributes nothing.

    Not equality: picking an imputation consumes a random number per replicate,
    so the patient samples differ from the single-draw run even at the same
    seed. What must hold is that the widths agree to within Monte Carlo error.
    """
    same = [_draw(0) for _ in range(5)]
    m = ms.MEASUREMENTS_BY_COL["adc_value"]
    joint = imp.joint_bootstrap(same, m, n_boot=600, seed=2)
    single = wb.bootstrap_cutpoint(
        same[0]["high_grade"].to_numpy(), same[0]["adc_value"].to_numpy(),
        m.direction, n_boot=600, seed=2)
    joint_width = joint["ci_hi"] - joint["ci_lo"]
    single_width = single["ci_hi"] - single["ci_lo"]
    assert joint_width == pytest.approx(single_width, rel=0.20)


def test_skipped_replicates_are_counted():
    draws = _draws()
    m = ms.MEASUREMENTS_BY_COL["adc_value"]
    out = imp.joint_bootstrap(draws, m, n_boot=80, seed=4)
    assert out["n_valid"] + out["n_skipped"] == 80


def test_empty_draws_return_blanks_rather_than_raising():
    m = ms.MEASUREMENTS_BY_COL["adc_value"]
    out = imp.joint_bootstrap([], m, n_boot=10)
    assert np.isnan(out["ci_lo"]) and out["n_valid"] == 0


# --- the combined table ----------------------------------------------------
def test_the_table_sets_both_intervals_side_by_side():
    draws = _draws()
    elig = _eligible(draws[0])
    patients, _ = wb.wobble_table(draws[0], elig, n_boot=60)
    table = imp.imputation_table(draws, elig, patients, n_boot=60)
    assert {"patients_ci_lo", "patients_ci_hi", "joint_ci_lo", "joint_ci_hi",
            "widening"} <= set(table.columns)


def test_widening_is_the_ratio_of_the_two_widths():
    draws = _draws()
    elig = _eligible(draws[0])
    patients, _ = wb.wobble_table(draws[0], elig, n_boot=60)
    table = imp.imputation_table(draws, elig, patients, n_boot=60)
    assert table["widening"].to_numpy() == pytest.approx(
        (table["joint_width"] / table["patients_width"]).to_numpy())


# --- the summary line ------------------------------------------------------
def test_describe_leads_with_a_divergence_when_there_is_one():
    draws = _draws()
    elig = _eligible(draws[0])
    patients, _ = wb.wobble_table(draws[0], elig, n_boot=40)
    table = imp.imputation_table(draws, elig, patients, n_boot=40,
                                 frozen={"adc_value": 99.0})
    assert imp.describe_imputation(table).startswith("No imputation reproduces")


def test_describe_handles_an_empty_table():
    assert imp.describe_imputation(pd.DataFrame()).startswith(
        "No cut-point could be derived")


# --- against the real cohort ----------------------------------------------
_FROZEN = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
           "edema_volume_cm3": 4.76, "edema_index": 0.0617}


def test_no_published_cutpoint_diverges_from_the_imputations(real_cohort):
    draws = imp.load_draws("output")
    table = imp.per_draw_cutpoints(draws, _eligible(real_cohort), frozen=_FROZEN)
    diverged = list(table.loc[table["diverges"], "measurement"])
    assert diverged == [], f"no imputation reproduces: {diverged}"


def test_a_complete_measurement_is_untouched_by_imputation(real_cohort):
    """Max diameter has no missing values, so every draw must agree exactly."""
    draws = imp.load_draws("output")
    table = imp.per_draw_cutpoints(draws, _eligible(real_cohort)).set_index("col")
    assert table.loc["max_diameter_cm", "draw_spread"] == 0.0
