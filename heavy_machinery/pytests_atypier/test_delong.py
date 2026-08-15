"""One AUC, one interval — pinned across the two pages that publish it.

This file exists because the drift shipped: ``report.html`` computed its
intervals with a 400-draw percentile bootstrap and ``cutpoint_report.html`` with
exact DeLong, so max diameter reached print as 0.67 (0.62–0.74) on one page and
0.67 (0.61–0.73) on the other. Both were defensible; neither page named its
method; and no test compared them, which is the actual reason it survived.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import delong
import scales
from measurements import MEASUREMENTS


def _data(seed: int = 0, n: int = 240, sep: float = 0.8):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, rng.normal(y * sep, 1.0)


# --- the estimator ---------------------------------------------------------
def test_the_point_estimate_is_the_ordinary_auc():
    """Whatever the interval, the middle of it has to be the AUC everyone means."""
    y, x = _data()
    assert delong.fast_delong(y, x).auc == pytest.approx(roc_auc_score(y, x))


def test_the_interval_brackets_the_estimate_and_stays_inside_zero_and_one():
    for seed in range(5):
        y, x = _data(seed=seed)
        r = delong.fast_delong(y, x)
        lo, hi = delong.logit_ci(r.auc, r.var)
        assert 0.0 <= lo <= r.auc <= hi <= 1.0


def test_the_logit_scale_keeps_a_wide_interval_from_exceeding_one():
    """A near-chance AUC with a large variance is where a naive interval breaks."""
    lo, hi = delong.logit_ci(0.52, 0.05)
    assert hi < 1.0 and lo > 0.0


def test_the_same_data_always_gives_the_same_interval():
    """The point of a closed formula: no seed, no wobble, no second answer."""
    y, x = _data()
    first = delong.logit_ci(*(lambda r: (r.auc, r.var))(delong.fast_delong(y, x)))
    second = delong.logit_ci(*(lambda r: (r.auc, r.var))(delong.fast_delong(y, x)))
    assert first == second


def test_the_variance_agrees_with_a_large_bootstrap():
    """Validates the part the other tests cannot: the variance itself.

    Everything above checks the AUC or the shape of the interval. Only this
    checks that the closed form gets the *width* right, by asking a method with
    no shared assumptions. 20 000 draws, because the 400-draw bootstrap this
    replaced was itself the defect — at that size the lower bound moved in the
    second decimal with the seed, so a loose agreement here would prove nothing.
    """
    y, x = _data(seed=7, n=400)
    r = delong.fast_delong(y, x)
    lo, hi = delong.logit_ci(r.auc, r.var)

    rng = np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(20_000):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        draws.append(roc_auc_score(y[idx], x[idx]))
    b_lo, b_hi = np.percentile(draws, [2.5, 97.5])
    assert lo == pytest.approx(b_lo, abs=0.015)
    assert hi == pytest.approx(b_hi, abs=0.015)


def test_ties_widen_the_interval_rather_than_being_broken_arbitrarily():
    """A third of this cohort shares edema volume 0.0 exactly."""
    y, x = _data(seed=2, n=300)
    tied = np.where(x < np.median(x), 0.0, x)      # pile a third at one value
    assert delong.fast_delong(y, tied).var > 0
    r = delong.fast_delong(y, tied)
    lo, hi = delong.logit_ci(r.auc, r.var)
    assert lo < r.auc < hi


def test_orienting_by_the_data_cannot_report_a_measurement_below_chance():
    """`auc_ci_auto` is for the EDA screen, which has no declared direction."""
    y, x = _data()
    auc, lo, hi = delong.auc_ci_auto(y, -x)
    assert auc >= 0.5 and lo <= auc <= hi


def test_flipping_the_score_leaves_the_interval_width_unchanged():
    """Var(AUC) = Var(1 − AUC), so orientation cannot change how wide it is."""
    y, x = _data()
    _, lo_a, hi_a = delong.auc_ci_auto(y, x)
    _, lo_b, hi_b = delong.auc_ci_auto(y, -x)
    assert (hi_a - lo_a) == pytest.approx(hi_b - lo_b, abs=1e-9)


# --- the cross-page pin ----------------------------------------------------
def test_both_reports_publish_the_same_interval_for_every_measurement(real_cohort):
    """The assertion the original defect would have failed.

    Compares the modelling phase's published cell against the cut-point phase's
    on identical patients. Checked for all five, because the defect was
    invisible on the one measurement whose two methods happened to round alike.
    """
    import separation as sep
    from eda_paper_tables import _continuous_rows

    df = real_cohort
    published = sep.separation_table(df)
    published = published[published["stratum"] == "all"].set_index("col")

    for m in MEASUREMENTS:
        rows = _continuous_rows(df, "high_grade", m.col, positive_class=True,
                                p_fdr=0.01, kind="continuous")
        cell = rows[0]["auc"]                       # e.g. "0.67 (0.61–0.73)"
        r = published.loc[m.col]
        expected = f"{r['auc']:.2f} ({r['auc_lo']:.2f}–{r['auc_hi']:.2f})"
        assert cell == expected, f"{m.col}: {cell!r} vs {expected!r}"


def test_the_cut_point_phase_still_reaches_delong_through_separation():
    """Seven modules and eleven test files import these names from `separation`."""
    import separation as sep
    for name in ("fast_delong", "logit_ci", "auc_with_ci", "delong_compare",
                 "oriented_score"):
        assert getattr(sep, name) is getattr(delong, name), name


def test_the_estimator_lives_where_the_shared_declarations_live():
    """Same direction of dependency as `scales`: cut-point reads modelling."""
    assert Path(delong.__file__).parent == Path(scales.__file__).parent
