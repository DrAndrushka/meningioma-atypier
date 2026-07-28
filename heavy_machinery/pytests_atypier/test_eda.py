"""EDA screening helpers — sign convention for Mann–Whitney / ϕ effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
