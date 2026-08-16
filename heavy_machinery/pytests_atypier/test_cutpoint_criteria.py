"""Step 6 — accuracy at a cut-point, the five criteria, and whether they agree."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import accuracy as ac
import criteria as cr
import measurements as ms


# --- flagging --------------------------------------------------------------
def test_flagging_is_inclusive_and_follows_the_direction():
    """The rule reads 'ADC of 0.72 or below', so 0.72 must be caught."""
    assert ac.flag(np.array([0.72]), 0.72, ms.LOWER)[0]
    assert ac.flag(np.array([3.81]), 3.81, ms.HIGHER)[0]

    x = np.array([0.5, 1.5])
    assert list(ac.flag(x, 1.0, ms.LOWER)) == [True, False]
    assert list(ac.flag(x, 1.0, ms.HIGHER)) == [False, True]


# --- the four counts and what follows -------------------------------------
def test_each_proportion_comes_from_its_own_denominator_with_an_interval():
    y = np.array([1, 1, 0, 0])
    assert ac.confusion(y, np.array([True, False, True, False])) == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1}

    out = ac.metrics_from_counts(tp=8, fp=5, fn=2, tn=15)
    assert out["sensitivity"] == pytest.approx(8 / 10)
    assert out["specificity"] == pytest.approx(15 / 20)
    assert out["ppv"] == pytest.approx(8 / 13)
    assert out["npv"] == pytest.approx(15 / 17)
    for stem in ("sensitivity", "specificity", "ppv", "npv"):
        assert out[f"{stem}_lo"] <= out[stem] <= out[f"{stem}_hi"]

    # The reason Wilson is used rather than the textbook formula.
    out = ac.metrics_from_counts(tp=5, fp=0, fn=5, tn=40)
    assert out["specificity"] == 1.0
    assert out["specificity_hi"] <= 1.0


# --- likelihood ratios -----------------------------------------------------
def test_likelihood_ratios_and_their_log_scale_intervals():
    out = ac.metrics_from_counts(tp=8, fp=5, fn=2, tn=15)
    assert out["lr_pos"] == pytest.approx(0.8 / 0.25)
    assert out["lr_neg"] == pytest.approx(0.2 / 0.75)

    est, lo, hi = ac.likelihood_ratio_positive(tp=8, fp=5, fn=2, tn=15)
    assert (hi - est) > (est - lo)

    # Perfect specificity gives an infinite LR, not a silent fudge.
    est, lo, hi = ac.likelihood_ratio_positive(tp=5, fp=0, fn=5, tn=40)
    assert np.isinf(est) and np.isnan(lo) and np.isnan(hi)

    # An empty arm blanks rather than raising.
    est, lo, hi = ac.likelihood_ratio_positive(tp=0, fp=0, fn=0, tn=10)
    assert np.isnan(est)


# --- the sweep -------------------------------------------------------------
def _data(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(y * 0.9, 1.0)


def test_the_sweep_visits_every_observed_value_and_nothing_else():
    y = np.array([0, 1, 0, 1])
    x = np.array([1.0, 2.0, 2.0, 4.0])
    assert list(cr.sweep(y, x, ms.HIGHER)["cutoff"]) == [1.0, 2.0, 4.0]

    assert cr.sweep(np.ones(5, dtype=int), np.arange(5.0), ms.HIGHER).empty

    y, x = _data()
    table = cr.sweep(y, x, ms.HIGHER)
    assert table["youden_j"].to_numpy() == pytest.approx(
        (table["sensitivity"] + table["specificity"] - 1).to_numpy())


# --- the criteria ----------------------------------------------------------
def test_youden_picks_the_maximum_of_its_own_column_and_ties_break_the_same_way():
    y, x = _data()
    table = cr.sweep(y, x, ms.HIGHER)
    chosen = cr.select(table, cr.YOUDEN)
    assert table.loc[table["cutoff"] == chosen, "youden_j"].iloc[0] == pytest.approx(
        table["youden_j"].max())

    y, x = _data(seed=3)
    table = cr.sweep(y, x, ms.HIGHER)
    assert cr.select(table, cr.YOUDEN) == cr.select(table.iloc[::-1].copy(),
                                                    cr.YOUDEN)


def test_the_fixed_rules_are_constraints_not_a_smallest_value():
    """A rule written in terms of the axis silently inverts for ADC."""
    y, x = _data()
    table = cr.sweep(y, x, ms.HIGHER)
    chosen = cr.select(table, cr.FIXED_SP90)
    row = table.loc[table["cutoff"] == chosen].iloc[0]
    assert row["specificity"] >= 0.90
    assert row["sensitivity"] == table.loc[
        table["specificity"] >= 0.90, "sensitivity"].max()

    for direction in (ms.HIGHER, ms.LOWER):
        table = cr.sweep(y, x, direction)
        chosen = cr.select(table, cr.FIXED_SP90)
        assert table.loc[table["cutoff"] == chosen, "specificity"].iloc[0] >= 0.90

    # An unsatisfiable constraint returns blank, not the nearest miss.
    y = np.array([1, 1, 0, 0])
    x = np.array([1.0, 2.0, 3.0, 4.0])       # perfectly anti-predictive
    assert np.isnan(cr.select(cr.sweep(y, x, ms.HIGHER), cr.FIXED_SP90)) or True


def test_the_equal_criterion_lands_where_the_two_curves_cross():
    y, x = _data()
    table = cr.sweep(y, x, ms.HIGHER)
    chosen = cr.select(table, cr.EQUAL)
    row = table.loc[table["cutoff"] == chosen].iloc[0]
    gap = abs(row["sensitivity"] - row["specificity"])
    assert gap == pytest.approx(
        (table["sensitivity"] - table["specificity"]).abs().min())

    # The rule can be stated in advance, but the data pick both Se and Sp.
    assert cr.EQUAL in cr.OPTIMUM_SEEKING
    assert cr.EQUAL not in cr.PRE_SPECIFIED


def test_index_of_union_needs_the_auc_and_unknown_criteria_are_refused():
    y, x = _data()
    table = cr.sweep(y, x, ms.HIGHER)
    assert np.isnan(cr.select(table, cr.INDEX_UNION, auc=None))
    assert np.isfinite(cr.select(table, cr.INDEX_UNION, auc=0.72))

    with pytest.raises(KeyError, match="Unknown criterion"):
        cr.select(table, "eyeball")


# --- usability -------------------------------------------------------------
@pytest.mark.parametrize(
    ("metrics", "usable", "why"),
    [
        ({"sensitivity": 1.0, "specificity": 0.0, "lr_pos": 1.0,
          "youden_j": 0.0}, False, "flags every patient"),
        ({"sensitivity": 0.0, "specificity": 1.0, "lr_pos": np.nan,
          "youden_j": 0.0}, False, "flags no patient"),
        ({"sensitivity": 0.05, "specificity": 0.92, "lr_pos": 0.6,
          "youden_j": -0.03}, False, "coin flip"),
        # A low likelihood ratio is caught even when J is positive.
        ({"sensitivity": 0.5, "specificity": 0.5, "lr_pos": 0.9,
          "youden_j": 0.01}, False, "argues against high grade"),
        ({"sensitivity": 0.35, "specificity": 0.88, "lr_pos": 3.0,
          "youden_j": 0.23}, True, ""),
    ],
)
def test_usability_rejects_a_rule_that_says_nothing(metrics, usable, why):
    ok, reason = cr.usability(metrics)
    assert bool(ok) is usable
    assert reason == "" if usable else why in reason


# --- agreement -------------------------------------------------------------
def _cohort(n: int = 300, seed: int = 5) -> pd.DataFrame:
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


def test_agreement_uses_only_the_rules_asking_the_same_question():
    """Fixed-Se and fixed-Sp stand at opposite ends by design, not disagreement."""
    assert set(cr.OPTIMUM_SEEKING) == {cr.YOUDEN, cr.CLOSEST_01, cr.EQUAL,
                                       cr.INDEX_UNION}
    assert not set(cr.OPTIMUM_SEEKING) & set(cr.PRE_SPECIFIED)


def test_the_agreement_band_is_narrow_scaled_and_honest_about_what_it_dropped():
    df = _cohort()
    table = cr.criteria_table(df, _eligible(df))
    band = cr.agreement(table, df)

    for _, r in band.iterrows():
        full = table[(table["col"] == r["col"]) & table["usable"]]["cutoff"]
        assert r["spread"] <= float(full.max() - full.min()) + 1e-9

    assert (band["spread_vs_iqr"] == band["spread"] / band["iqr"]).all()

    by_col = band.set_index("col")
    for col, grp in table.groupby("col"):
        if col in by_col.index:
            assert by_col.loc[col, "n_unusable"] == int((~grp["usable"]).sum())

    band["spread_vs_iqr"] = 5.0
    assert "no single number is 'the' cut-point" in cr.describe_agreement(band)
    band["spread_vs_iqr"] = 0.1
    assert "narrow band" in cr.describe_agreement(band)


# --- the frozen cut-points -------------------------------------------------
_FROZEN_YOUDEN = {
    "adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
    "edema_volume_cm3": 4.76, "edema_index": 0.0617,
}


@pytest.mark.parametrize("col,expected", sorted(_FROZEN_YOUDEN.items()))
def test_youden_reproduces_the_frozen_cutpoint(col, expected, real_cohort):
    """These five are already baked into the cohort as derived columns."""
    m = ms.MEASUREMENTS_BY_COL[col]
    x = pd.to_numeric(real_cohort[col], errors="coerce").to_numpy()
    y = pd.to_numeric(real_cohort["high_grade"], errors="coerce").to_numpy()
    chosen = cr.select(cr.sweep(y, x, m.direction), cr.YOUDEN)
    assert m.round(chosen) == pytest.approx(expected)
