"""Marker panel: LR+, the BinaryMarker adapter, rule menus, model re-scoring."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import marker_panel as mp

TARGET = "high_grade"


# --------------------------------------------------------------------------
# Positive likelihood ratio
# --------------------------------------------------------------------------
def test_lr_pos_matches_a_hand_computed_2x2():
    """27 of 105 high-grade flagged, 23 of 247 benign flagged.

    sens = 27/105 = 0.2571, spec = 224/247 = 0.9069, LR+ = sens / (1 - spec).
    Katz log-scale interval: exp(log LR+ ± 1.96 * sqrt(1/TP - 1/(TP+FN) + 1/FP - 1/(FP+TN))).
    """
    out = mp.likelihood_ratio_positive(tp=27, fp=23, fn=78, tn=224)
    assert out["lr_pos"] == pytest.approx(2.7615, abs=1e-4)
    assert out["lr_pos_lo"] == pytest.approx(1.6631, abs=1e-3)
    assert out["lr_pos_hi"] == pytest.approx(4.5854, abs=1e-3)
    assert out["chance_overlap"] is False
    assert out["continuity_corrected"] is False


def test_lr_pos_flags_a_marker_whose_interval_covers_one():
    """A sign that fires equally often in both groups carries no information."""
    out = mp.likelihood_ratio_positive(tp=20, fp=45, fn=85, tn=202)
    assert out["lr_pos_lo"] < 1.0 < out["lr_pos_hi"]
    assert out["chance_overlap"] is True


def test_lr_pos_survives_a_zero_cell_with_a_continuity_correction():
    """brain_invasion-shaped: never seen in a benign tumour, so FP = 0.

    Without a correction LR+ is infinite and its interval undefined. Adding 0.5
    to every cell (Haldane-Anscombe) gives a finite, very wide interval — which
    is the honest answer: a huge point estimate resting on five patients.
    """
    out = mp.likelihood_ratio_positive(tp=5, fp=0, fn=100, tn=247)
    assert np.isfinite(out["lr_pos"])
    assert out["lr_pos"] == pytest.approx(25.7358, abs=1e-3)
    assert out["lr_pos_lo"] == pytest.approx(1.4358, abs=1e-2)
    assert out["lr_pos_hi"] == pytest.approx(461.3, rel=1e-3)
    assert out["continuity_corrected"] is True


def test_lr_pos_returns_nan_when_a_margin_is_empty():
    """No high-grade patients at all: nothing to compute, and no crash."""
    out = mp.likelihood_ratio_positive(tp=0, fp=3, fn=0, tn=40)
    assert np.isnan(out["lr_pos"])
    assert out["chance_overlap"] is False
