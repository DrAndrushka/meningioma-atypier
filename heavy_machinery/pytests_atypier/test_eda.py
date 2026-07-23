"""EDA screening helpers — sign convention for Mann–Whitney effects."""

from __future__ import annotations

import numpy as np

import eda


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
