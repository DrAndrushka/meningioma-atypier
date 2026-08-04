"""The numpy rule scorer must agree with full_rule_menu, exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd

import combinations as cb
import rule_matrix as rm
from thresholds import Metric

TARGET = "high_grade"
A = Metric("a", "Metric A", "u", "higher")
B = Metric("b", "Metric B", "u", "higher")
C = Metric("c", "Metric C", "u", "lower")
D = Metric("d", "Metric D", "u", "higher")
E = Metric("e", "Metric E", "u", "higher")


def tiny_frame() -> pd.DataFrame:
    """Eight patients, two flags, one missing value each way."""
    return pd.DataFrame({
        "a": pd.array([10.0, 10.0, 0.0, 0.0, 10.0, 0.0, None, 10.0], dtype="Float64"),
        "b": pd.array([10.0, 0.0, 10.0, 0.0, None, None, 10.0, 10.0], dtype="Float64"),
        TARGET: pd.array([True, True, False, False, True, False, True, True],
                         dtype="boolean"),
    })


def holey_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    """Missing predictors AND missing outcomes — the case the matrix must not drop."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    df = pd.DataFrame({
        "a": rng.normal(size=n) + y * 1.2,
        "b": rng.normal(size=n) + y * 1.0,
        "c": -(rng.normal(size=n) + y * 0.8),
        TARGET: pd.array(y, dtype="boolean"),
    })
    df.loc[rng.random(n) < 0.25, "a"] = np.nan
    df.loc[rng.random(n) < 0.30, "b"] = np.nan
    df.loc[rng.random(n) < 0.15, "c"] = np.nan
    df.loc[rng.random(n) < 0.10, TARGET] = pd.NA
    return df


def holey_cutpoints(df: pd.DataFrame):
    return cb.cutpoints_for_rule(df.dropna(subset=[TARGET]), [A, B, C], TARGET, "youden")


def assert_matches_menu(df: pd.DataFrame, cps, max_size: int = 2) -> None:
    menu = cb.full_rule_menu(df, cps, TARGET, max_size=max_size)
    mat = rm.rule_matrix(df, cps, TARGET, max_size=max_size)
    got = rm.youden_j(mat, max_size=max_size)

    assert list(menu["rule_label"]) == list(mat.labels)
    assert list(menu["kind"]) == list(mat.kinds)
    want = menu["youden_J"].to_numpy(dtype=float)
    assert np.array_equal(np.isnan(want), np.isnan(got))
    assert np.nanmax(np.abs(want - got)) == 0.0


def test_matches_menu_on_a_hand_checkable_frame():
    assert_matches_menu(tiny_frame(), [cb.CutPoint(A, 5.0), cb.CutPoint(B, 5.0)])


def test_matches_menu_with_missing_predictors_and_outcomes():
    df = holey_frame()
    assert_matches_menu(df, holey_cutpoints(df))


def test_matches_menu_on_degenerate_markers():
    """An all-missing flag and a never-true flag must give the same NaNs."""
    df = tiny_frame()
    df["d"] = pd.array([None] * 8, dtype="Float64")
    df["e"] = pd.array([0.0] * 8, dtype="Float64")
    assert_matches_menu(
        df, [cb.CutPoint(A, 5.0), cb.CutPoint(D, 5.0), cb.CutPoint(E, 5.0)])


def test_rows_selects_a_resample_like_iloc_does():
    """Scoring rows=take must equal scoring df.iloc[take] through the menu."""
    df = holey_frame()
    cps = holey_cutpoints(df)
    take = np.random.default_rng(11).integers(0, len(df), len(df))

    boot = df.iloc[take].reset_index(drop=True)
    want = cb.full_rule_menu(boot, cps, TARGET)["youden_J"].to_numpy(dtype=float)
    got = rm.youden_j(rm.rule_matrix(df, cps, TARGET), rows=take)

    assert np.array_equal(np.isnan(want), np.isnan(got))
    assert np.nanmax(np.abs(want - got)) == 0.0


def test_matrix_keeps_every_row_including_unknown_outcomes():
    """n must be len(df): the bootstrap resamples the frame it was handed."""
    df = holey_frame()
    mat = rm.rule_matrix(df, holey_cutpoints(df), TARGET)
    assert mat.n == len(df)
    assert mat.known.sum() < len(df)


def test_menu_shape_is_singles_then_pairs_then_counts():
    df = holey_frame()
    mat = rm.rule_matrix(df, holey_cutpoints(df), TARGET)
    assert mat.kinds[:3] == ["single"] * 3
    assert mat.kinds[3:5] == ["and", "or"]
    assert mat.kinds[-3:] == ["count"] * 3
    assert len(mat.labels) == 3 + 2 * 3 + 3


def test_triples_are_supported_when_asked_for():
    df = holey_frame()
    assert_matches_menu(df, holey_cutpoints(df), max_size=3)
