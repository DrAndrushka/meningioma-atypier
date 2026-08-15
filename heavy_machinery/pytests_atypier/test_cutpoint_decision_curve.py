"""Step 14 — net benefit: is the rule worth following, and is the number better?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import criteria as cr
import decision_curve as dc
import eligibility as el
import measurements as ms


def _data(seed: int = 0, n: int = 300, sep: float = 0.9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(y * sep, 1.0)


# --- the arithmetic --------------------------------------------------------
def test_net_benefit_matches_the_formula_by_hand():
    """20 true positives, 10 false, 100 patients, threshold 0.2 → 0.2 − 0.1×0.25."""
    got = dc.net_benefit(20, 10, 100, np.array([0.2]))[0]
    assert got == pytest.approx(0.20 - 0.10 * 0.25)


def test_treat_all_starts_at_the_prevalence():
    """At a threshold near zero, flagging everyone costs nothing and finds all."""
    y = np.array([1] * 30 + [0] * 70)
    curve = dc.treat_all_curve(y, np.array([1e-9]))
    assert curve[0] == pytest.approx(0.30, abs=1e-6)


def test_treat_all_is_zero_at_a_threshold_equal_to_the_prevalence():
    """The defining property: at t = prevalence, treating everyone breaks even."""
    y = np.array([1] * 30 + [0] * 70)
    assert dc.treat_all_curve(y, np.array([0.30]))[0] == pytest.approx(0.0,
                                                                      abs=1e-9)


def test_a_perfect_rule_has_net_benefit_equal_to_the_prevalence_everywhere():
    """No false positives means nothing to charge for, at any threshold."""
    y = np.array([1] * 40 + [0] * 60)
    x = np.array([10.0] * 40 + [0.0] * 60)
    curve = dc.rule_curve(y, x, 5.0, ms.HIGHER, dc.threshold_grid())
    assert np.allclose(curve, 0.40)


def test_a_useless_rule_that_flags_everyone_equals_treat_all():
    y, x = _data()
    t = dc.threshold_grid()
    flags_everyone = float(np.min(x) - 1.0)
    assert np.allclose(dc.rule_curve(y, x, flags_everyone, ms.HIGHER, t),
                       dc.treat_all_curve(y, t))


def test_the_grid_never_reaches_a_threshold_of_one():
    """t/(1−t) is undefined at 1 and enormous just below it."""
    t = dc.threshold_grid(0.99)
    assert t.max() < 1.0 and t.min() >= dc.MIN_THRESHOLD


# --- the fitter ------------------------------------------------------------
def test_the_hand_rolled_fit_matches_statsmodels():
    """It exists only for speed, so it has to agree with the reference."""
    y, x = _data(seed=3)
    mine = dc._irls(y.astype(float), x)
    reference = sm.Logit(y, sm.add_constant(x.reshape(-1, 1))).fit(disp=0).params
    assert mine == pytest.approx(tuple(reference), abs=1e-6)


def test_the_fit_declines_rather_than_guessing_on_one_class():
    assert dc._irls(np.ones(50), np.arange(50.0)) is None


def test_a_logged_measurement_is_fitted_on_the_log_scale():
    """Otherwise the same column would carry two different models in one report."""
    x = np.array([0.0, 1.0, np.e - 1.0])
    assert dc._model_scale(x, True) == pytest.approx([0.0, np.log(2.0), 1.0])
    assert dc._model_scale(x, False) == pytest.approx(x)


# --- optimism --------------------------------------------------------------
def test_correction_never_flatters_a_rule_chosen_on_noise():
    """On pure noise the apparent curve is all luck, so the correction bites."""
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 250)
    x = rng.normal(0.0, 1.0, 250)
    out = dc.corrected_curves(y, x, ms.HIGHER, False, dc.threshold_grid(0.4),
                              n_boot=150)
    apparent = dc.at_threshold(out["rule"], dc.threshold_grid(0.4), 0.3)
    corrected = dc.at_threshold(out["rule_corrected"], dc.threshold_grid(0.4), 0.3)
    assert corrected < apparent


def test_the_same_seed_gives_the_same_correction():
    y, x = _data()
    t = dc.threshold_grid(0.4)
    a = dc.corrected_curves(y, x, ms.HIGHER, False, t, n_boot=80, seed=5)
    b = dc.corrected_curves(y, x, ms.HIGHER, False, t, n_boot=80, seed=5)
    assert np.allclose(a["rule_corrected"], b["rule_corrected"])


def test_the_seed_and_count_are_the_ones_step_7_used():
    """Two phases correcting the same cut-point on different resamples is a bug."""
    import wobble as wb
    assert (dc.SEED, dc.N_BOOTSTRAP) == (wb.SEED, wb.N_BOOTSTRAP)


def test_too_few_events_returns_blanks_rather_than_a_number():
    y = np.array([1] + [0] * 60)
    out = dc.corrected_curves(y, np.arange(61.0), ms.HIGHER, False,
                              dc.threshold_grid(), n_boot=10)
    assert np.isnan(out["rule"]).all() and out["n_valid"] == 0


def test_the_curve_scores_the_published_cut_point_not_a_re_derived_one():
    """Table S4 has to describe the same rule as every other table in the phase.

    The criterion's own optimum is an unrounded float; the manuscript prints a
    rounded one. On the edema index those differ by 0.0000155 with one
    high-grade patient between them, which moved the published net benefit by
    0.003 and made S2 and S4 quietly describe two different rules.
    """
    y, x = _data()
    t = dc.threshold_grid(0.4)
    derived = cr.select(cr.sweep(y, x, ms.HIGHER), cr.YOUDEN)
    published = round(float(derived), 2)
    assert published != derived, "fixture must actually exercise the rounding"
    out = dc.corrected_curves(y, x, ms.HIGHER, False, t, cutoff=published,
                              n_boot=40)
    assert out["cutpoint"] == pytest.approx(published)
    assert out["derived_cutpoint"] == pytest.approx(derived)
    assert np.allclose(out["rule"], dc.rule_curve(y, x, published, ms.HIGHER, t))


def test_a_cut_point_the_criterion_does_not_reproduce_is_refused(real_cohort):
    """Rounding may move a published cut-point; nothing else may.

    The check lives here rather than inside the curve because only the caller
    knows the precision each measurement prints at — the same division of labour
    as step 7, and the same exception.
    """
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep

    df = real_cohort
    fits = nl.fit_all(df)
    eligible = el.eligible(el.carry_forward(sep.separation_table(df),
                                            bl.bend_table(df, fits=fits)))
    tampered = {"adc_value": 0.65, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
                "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    with pytest.raises(dc.FrozenCutpointError, match="may not move"):
        dc.decision_table(df, eligible, tampered, n_boot=20)


def test_the_conversion_to_patients_uses_one_threshold_not_two():
    """Reading a gain at .30 and dividing by the odds at .2983 is two rules."""
    t = dc.threshold_grid(0.4)
    curve = np.full(t.shape, 0.05)
    treat_all = np.full(t.shape, 0.01)
    # The grid has no .2983, so the gain is read at .30; the odds must match.
    got = dc.net_reduction_per_100(curve, treat_all, t, 0.2983)
    assert got == pytest.approx(100.0 * 0.04 / (0.30 / 0.70))


# --- reading the curves ----------------------------------------------------
def test_useful_range_takes_the_widest_contiguous_run():
    """A rule useful at 10% and again at 40% but not between is not one rule."""
    t = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    curve = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    zeros = np.zeros_like(t)
    assert dc.useful_range(curve, zeros, t) == (0.3, 0.5)


def test_a_rule_that_never_beats_the_alternatives_has_no_useful_range():
    t = np.array([0.1, 0.2, 0.3])
    assert np.isnan(dc.useful_range(np.zeros(3), np.ones(3), t)).all()


def test_net_reduction_converts_a_gain_into_patients():
    """At t = 0.2 the exchange rate is 1/4, so a gain of 0.01 is 4 per 100."""
    t = np.array([0.2])
    got = dc.net_reduction_per_100(np.array([0.05]), np.array([0.04]), t, 0.2)
    assert got == pytest.approx(4.0)


# --- on the real cohort ----------------------------------------------------
def test_every_carried_measurement_gets_a_curve(real_cohort):
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep

    df = real_cohort
    fits = nl.fit_all(df)
    eligible = el.eligible(el.carry_forward(sep.separation_table(df),
                                            bl.bend_table(df, fits=fits)))
    frozen = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
              "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    table, curves = dc.decision_table(df, eligible, frozen, n_boot=60)
    assert len(table) == len(eligible)
    assert set(curves) == set(table["col"])
    # Every correction has to move the curve down. A resample that made a rule
    # look better on patients it had never seen would mean the loop is scoring
    # the wrong half.
    assert (table["nb_rule"] <= table["nb_rule_apparent"] + 1e-12).all()
    # And the apparent value must be the net benefit of the PUBLISHED rule,
    # recomputed here straight from the 2x2 rather than trusted.
    for _, r in table.iterrows():
        m = ms.MEASUREMENTS_BY_COL[r["col"]]
        sub = df[[m.col, "high_grade"]].dropna()
        xv = sub[m.col].to_numpy(float)
        yv = sub["high_grade"].to_numpy(int)
        flagged = xv <= r["cutpoint"] if m.direction == ms.LOWER else xv >= r["cutpoint"]
        tp = float((flagged & (yv == 1)).sum())
        fp = float((flagged & (yv == 0)).sum())
        t_used = dc._nearest(dc.threshold_grid(), r["prevalence"])
        expected = tp / yv.size - (fp / yv.size) * (t_used / (1 - t_used))
        assert r["nb_rule_apparent"] == pytest.approx(expected, abs=1e-9), m.col


def test_the_decidable_span_is_never_wider_than_the_grid(real_cohort):
    import bend_location as bl
    import nonlinearity as nl
    import separation as sep

    df = real_cohort
    fits = nl.fit_all(df)
    eligible = el.eligible(el.carry_forward(sep.separation_table(df),
                                            bl.bend_table(df, fits=fits)))
    frozen = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
              "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    _, curves = dc.decision_table(df, eligible, frozen, n_boot=40)
    for entry in curves.values():
        mask = np.asarray(entry["decidable"], dtype=bool)
        assert mask.shape == entry["thresholds"].shape
        # Contiguous by construction: one span, not scattered points.
        if mask.any():
            assert mask[mask.argmax():mask.size - mask[::-1].argmax()].all()
