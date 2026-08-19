"""Marker panel: LR+, the BinaryMarker adapter, and combination scoring."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import combinations as cb
import marker_panel as mp
from marker_rules import Metric

TARGET = "high_grade"


# --------------------------------------------------------------------------
# Positive likelihood ratio
# --------------------------------------------------------------------------
def test_likelihood_ratio_positive():
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

    # A sign that fires equally often in both groups carries no information.
    out = mp.likelihood_ratio_positive(tp=20, fp=45, fn=85, tn=202)
    assert out["lr_pos_lo"] < 1.0 < out["lr_pos_hi"]
    assert out["chance_overlap"] is True

    # brain_invasion-shaped: never seen in a benign tumour, so FP = 0. Without a
    # correction LR+ is infinite and its interval undefined. Adding 0.5 to every
    # cell (Haldane-Anscombe) gives a finite, very wide interval — which is the
    # honest answer: a huge point estimate resting on five patients.
    out = mp.likelihood_ratio_positive(tp=5, fp=0, fn=100, tn=247)
    assert np.isfinite(out["lr_pos"])
    assert out["lr_pos"] == pytest.approx(25.7358, abs=1e-3)
    assert out["lr_pos_lo"] == pytest.approx(1.4358, abs=1e-2)
    assert out["lr_pos_hi"] == pytest.approx(461.3, rel=1e-3)
    assert out["continuity_corrected"] is True

    # No high-grade patients at all: nothing to compute, and no crash.
    out = mp.likelihood_ratio_positive(tp=0, fp=3, fn=0, tn=40)
    assert np.isnan(out["lr_pos"])
    assert out["chance_overlap"] is False


def marker_frame() -> pd.DataFrame:
    """Eight patients, three signs, one missing value in each of two signs."""
    return pd.DataFrame({
        "sign_a": pd.array([True, True, False, False, True, False, None, True],
                           dtype="boolean"),
        "sign_b": pd.array([True, False, True, False, None, False, True, True],
                           dtype="boolean"),
        "sign_c": pd.array([False, False, False, False, True, True, True, False],
                           dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, True],
                         dtype="boolean"),
    })


def accuracy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": TARGET, "predictor": TARGET, "kind": "binary"},
        {"target": TARGET, "predictor": "sign_a", "kind": "binary"},
        {"target": TARGET, "predictor": "sign_b", "kind": "derived_binary"},
        {"target": TARGET, "predictor": "sex_male", "kind": "derived_binary"},
        {"target": TARGET, "predictor": "adc_value", "kind": "continuous"},
        {"target": "other", "predictor": "sign_c", "kind": "binary"},
    ])


# --------------------------------------------------------------------------
# BinaryMarker — the adapter that lets combinations.py accept yes/no columns
# --------------------------------------------------------------------------
def test_binary_marker_is_a_drop_in_cutpoint():
    """The reuse claim, verified: a marker and a 0.5 cut-point flag the same rows."""
    df = marker_frame()
    numeric = df.assign(sign_a=df["sign_a"].astype("Float64"))
    marker = mp.BinaryMarker("sign_a", "Sign A")
    cutpoint = cb.CutPoint(Metric("sign_a", "Sign A", "", "higher"), 0.5)

    from_marker = marker.flag(df)
    from_cutpoint = cutpoint.flag(numeric)
    pd.testing.assert_series_equal(
        from_marker.astype("boolean"), from_cutpoint.astype("boolean"),
        check_names=False,
    )

    # No cut-point to print, so the short form is just the name.
    assert marker.label == "Sign A"
    assert marker.short_label == "Sign A"

    # single_rule_table is threshold-phase code, called here on plain columns.
    markers = [marker, mp.BinaryMarker("sign_b", "Sign B")]
    table = cb.single_rule_table(df, markers, TARGET)
    assert list(table["rule_label"]) == ["Sign A", "Sign B"]
    assert table["youden_J"].notna().all()


# --------------------------------------------------------------------------
# Marker selection
# --------------------------------------------------------------------------
def test_markers_are_read_from_the_accuracy_table():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    cols = [m.col for m in markers]
    assert cols == ["sign_a", "sign_b", "sex_male"]
    # The outcome is never treated as a marker, and neither are continuous
    # predictors or the columns screened against a different target.
    assert TARGET not in cols
    assert "adc_value" not in cols
    assert "sign_c" not in cols
    # Labels are prettified for printing.
    assert markers[0].label == "Sign A"

    # sex_male is derived_binary and would otherwise enter a section on MRI signs.
    excluded = mp.markers_from_diagnostic_accuracy(
        accuracy_table(), target=TARGET, exclude={"sex_male"},
    )
    assert [m.col for m in excluded] == ["sign_a", "sign_b"]


# --------------------------------------------------------------------------
# Aim 1 — the marker table
# --------------------------------------------------------------------------
def test_marker_panel_reports_yield_alongside_specificity():
    """The guard against the specificity trap.

    ``rare`` is present in one patient and never in a benign tumor, so its
    specificity is 1.0 — and it catches 1 of 5 high-grade tumors. Both numbers
    must be in the row, or the table crowns a useless sign.
    """
    df = pd.DataFrame({
        "rare": pd.array([True, False, False, False, False, False, False, False],
                         dtype="boolean"),
        "common": pd.array([True, True, True, False, True, True, False, False],
                           dtype="boolean"),
        TARGET: pd.array([True, True, True, True, True, False, False, False],
                         dtype="boolean"),
    })
    markers = [mp.BinaryMarker("rare", "Rare"), mp.BinaryMarker("common", "Common")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    rare = panel[panel["marker"] == "rare"].iloc[0]
    assert rare["specificity"] == 1.0
    assert rare["present_n"] == 1
    assert rare["catches"] == 1
    assert rare["n_high_grade"] == 5


def test_markers_that_cannot_beat_chance_sort_last():
    """A ranked table must not open with a row that says nothing."""
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.3, 300).astype(bool)
    df = pd.DataFrame({
        "informative": pd.array(rng.binomial(1, 0.15 + 0.5 * y).astype(bool),
                                dtype="boolean"),
        "noise": pd.array(rng.binomial(1, 0.4, 300).astype(bool), dtype="boolean"),
        TARGET: pd.array(y, dtype="boolean"),
    })
    markers = [mp.BinaryMarker("noise", "Noise"),
               mp.BinaryMarker("informative", "Informative")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    assert panel.iloc[0]["marker"] == "informative"
    assert bool(panel.iloc[-1]["chance_overlap"]) is True


def test_marker_reading_view_prints_the_estimate_even_when_it_covers_one():
    df = pd.DataFrame({
        "noise": pd.array([True, False, True, False, True, False, True, False],
                          dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, False],
                         dtype="boolean"),
    })
    panel = mp.marker_panel_table(df, [mp.BinaryMarker("noise", "Noise")], TARGET)
    view = mp.marker_panel_reading_view(panel)
    assert list(view.columns) == [
        "Variable", "n/N (%)",
        "Sens (95% CI)", "Spec (95% CI)",
        "PPV (95% CI)", "NPV (95% CI)", "FDR p", "LR+ (95% CI)", "origin",
    ]
    # Predictive values come from the same 2×2 table as sensitivity: of the
    # four scans with the sign, three are high grade.
    assert view.iloc[0]["PPV (95% CI)"] == "75% (30–95)"
    # The interval covers 1; the table still carries the number, and the
    # reader draws the conclusion from the interval.
    assert bool(panel.iloc[0]["chance_overlap"])
    assert view.iloc[0]["LR+ (95% CI)"] == "3.00 (0.50–17.95)"
    assert view.iloc[0]["n/N (%)"] == "4/8 (50%)"

    # With no markers at all the panel is empty, not broken.
    empty = mp.marker_panel_table(
        pd.DataFrame({TARGET: pd.array([True, False], dtype="boolean")}), [], TARGET,
    )
    assert empty.empty
    assert mp.marker_panel_reading_view(empty).empty


# --------------------------------------------------------------------------
# Aim 1 — the figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt


def test_lr_forest_draws_one_row_per_marker_on_a_log_axis():
    df = pd.DataFrame({
        "a": pd.array([True, True, False, False, True, False, False, False],
                      dtype="boolean"),
        "b": pd.array([True, False, True, False, True, True, False, False],
                      dtype="boolean"),
        TARGET: pd.array([True, True, True, True, True, False, False, False],
                         dtype="boolean"),
    })
    markers = [mp.BinaryMarker("a", "Sign A"), mp.BinaryMarker("b", "Sign B")]
    panel = mp.marker_panel_table(df, markers, TARGET)

    fig = mp.lr_forest_figure(panel)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(ax.get_yticklabels()) == 2
    plt.close(fig)

    # An empty panel must not crash the notebook cell that saves figures.
    fig = mp.lr_forest_figure(pd.DataFrame(columns=["label", "lr_pos"]))
    assert fig is not None
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — one denominator
# --------------------------------------------------------------------------
def sparse_frame() -> pd.DataFrame:
    """Six patients; ``sign_b`` is missing for two of them."""
    return pd.DataFrame({
        "sign_a": pd.array([True, False, True, False, True, False], dtype="boolean"),
        "sign_b": pd.array([True, False, None, None, True, True], dtype="boolean"),
        "always_off": pd.array([False] * 6, dtype="boolean"),
        TARGET: pd.array([True, False, True, False, True, False], dtype="boolean"),
    })


def test_shared_cohort_keeps_only_patients_with_every_marker_observed():
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("sign_b", "B")]
    shared = mp.shared_cohort_frame(df, markers, TARGET)
    assert len(shared) == 4
    assert shared["sign_b"].notna().all()

    # The audit records what each marker cost.
    audit = mp.shared_cohort_audit(df, markers, TARGET, dropped=[])
    assert set(audit.columns) == {"item", "value", "note"}
    rows = dict(zip(audit["item"], audit["value"]))
    assert rows["Patients in the shared set"] == 4
    assert rows["sign_b"] == 2  # patients this marker cost

    # When no patient has everything, the result is empty, not broken.
    nothing_shared = pd.DataFrame({
        "a": pd.array([True, None], dtype="boolean"),
        "b": pd.array([None, True], dtype="boolean"),
        TARGET: pd.array([True, False], dtype="boolean"),
    })
    assert mp.shared_cohort_frame(
        nothing_shared,
        [mp.BinaryMarker("a", "A"), mp.BinaryMarker("b", "B")],
        TARGET,
    ).empty


def test_a_marker_that_never_fires_is_dropped_with_a_reason():
    """An all-false column has an undefined likelihood ratio and no rule value."""
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("always_off", "Off")]
    kept, dropped = mp.usable_markers(df, markers, TARGET)
    assert [m.col for m in kept] == ["sign_a"]
    assert dropped[0]["marker"] == "always_off"
    assert "never" in dropped[0]["reason"].lower()


# --------------------------------------------------------------------------
# Aim 2 — the count score
# --------------------------------------------------------------------------
def count_frame(n: int = 240, seed: int = 3) -> pd.DataFrame:
    """Three signs, each independently more common in high-grade tumors."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {
        f"sign_{i}": pd.array(rng.binomial(1, 0.15 + 0.45 * y).astype(bool),
                              dtype="boolean")
        for i in range(3)
    }
    cols[TARGET] = pd.array(y, dtype="boolean")
    return pd.DataFrame(cols)


COUNT_MARKERS = [mp.BinaryMarker(f"sign_{i}", f"Sign {i}") for i in range(3)]


def test_count_score_rows_and_thresholds():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    assert list(counts["n_criteria_met"]) == [0, 1, 2, 3]
    assert counts["n"].sum() == counts.attrs["n_scored"]

    # The literal claim the section makes. If this fails, the claim is wrong.
    risks = counts[counts["n"] >= 10]["risk"].to_numpy(dtype=float)
    assert risks[0] < risks[-1]

    rules = mp.count_thresholds(count_frame(), COUNT_MARKERS, TARGET)
    assert list(rules["rule_label"]) == [
        "≥ 1 of 3 criteria", "≥ 2 of 3 criteria", "≥ 3 of 3 criteria",
    ]
    assert rules["youden_J"].notna().all()


def test_count_score_figure_names_the_count_and_a_short_marker_list():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.count_score_figure(counts, COUNT_MARKERS)
    ax = fig.axes[0]
    assert "3" in ax.get_xlabel()
    # The marker list is recorded for the report, not drawn on the axes.
    import plot_style as ps
    legend = ps.figure_legend(fig)
    assert "Sign 0" in f"{legend['title']} {legend['note']}"
    plt.close(fig)


def wide_frame(k: int = 16, n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {
        f"sign_{i}": pd.array(rng.binomial(1, 0.3 + 0.2 * y).astype(bool),
                              dtype="boolean")
        for i in range(k)
    }
    cols[TARGET] = pd.array(y, dtype="boolean")
    return pd.DataFrame(cols)


def test_a_long_marker_list_is_left_out_of_the_subtitle():
    """Sixteen names is a 200-character line and matplotlib abandons the layout.

    The count is still on the x-axis, and the names are the rows of the aim-1
    table on the same page, so nothing is lost by dropping them here.
    """
    markers = [mp.BinaryMarker(f"sign_{i}", f"Sign {i}") for i in range(16)]
    counts = mp.count_score(wide_frame(), markers, TARGET)
    fig = mp.count_score_figure(counts, markers)
    ax = fig.axes[0]
    text = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
    assert "Sign 15" not in text
    assert "16" in ax.get_xlabel()
    plt.close(fig)


# --------------------------------------------------------------------------
# Aim 2 — which two counts the headline sentence may quote
# --------------------------------------------------------------------------
def _counts(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """``(count, n, events)`` triples in the shape ``count_score`` returns."""
    frame = pd.DataFrame([
        {"n_criteria_met": c, "n": n, "n_high_grade": e,
         "risk": (e / n) if n else np.nan,
         "risk_lo": np.nan, "risk_hi": np.nan}
        for c, n, e in rows
    ])
    frame.attrs["k"] = max(c for c, _, _ in rows)
    return frame


def test_the_headline_picks_two_usable_counts_and_measures_the_direction():
    """The real failure: the top bin held one patient, whose 0% became the claim.

    Occupied is not the same as usable. With the highest bin holding a single
    patient, a sentence built from the first and last occupied rows reported
    two coin flips as a trend. The direction is measured too — a hard-coded
    "rises" is a hope, not a finding.
    """
    head = mp.count_headline(_counts([
        (3, 9, 0), (4, 12, 0), (8, 60, 18), (10, 40, 17), (15, 1, 0),
    ]), min_n=10)
    row = head.iloc[0]
    assert row["low_count"] == 4
    assert row["high_count"] == 10
    assert row["direction"] == "rises"

    assert mp.count_headline(_counts([
        (0, 40, 24), (1, 50, 20), (2, 60, 6),
    ]), min_n=10).iloc[0]["direction"] == "falls"

    assert mp.count_headline(_counts([
        (0, 40, 20), (1, 30, 12), (2, 60, 30),
    ]), min_n=10).iloc[0]["direction"] == "flat"


def test_the_headline_degrades_rather_than_going_silent():
    """A thin honest sentence beats no sentence — but it must say it is thin."""
    head = mp.count_headline(_counts([(0, 4, 0), (1, 5, 3)]), min_n=10)
    row = head.iloc[0]
    assert row["low_count"] == 0 and row["high_count"] == 1
    assert row["min_n"] == 1
    assert "no two counts" in row["note"]

    # One occupied count is no trend at all, and neither is no table.
    assert mp.count_headline(_counts([(2, 30, 9)]), min_n=10).empty
    assert mp.count_headline(pd.DataFrame()).empty

    # Whatever it quotes, it carries the denominators.
    row = mp.count_headline(_counts([(1, 25, 2), (4, 40, 20)]), min_n=10).iloc[0]
    assert row["low_n"] == 25 and row["high_n"] == 40
    assert row["k_markers"] == 4


# --------------------------------------------------------------------------
# Aim 2 — paying for having picked a winner
# --------------------------------------------------------------------------
def test_both_sides_of_the_head_to_head_are_corrected():
    """The CHANGES.md regression.

    A corrected combination scored against an *uncorrected* single flatters the
    combination by the whole of the single's own selection optimism. Picking the
    best of N single markers is a choice made on these patients too, so it costs
    something, and that cost must be non-zero and recorded.

    Whether correction shrinks the gap is data, not doctrine: correction
    subtracts each side's *own* optimism, so when the best-of-N-singles side
    pays more of it the corrected gap is the larger one. That is measured here
    rather than asserted in the report's prose.
    """
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    assert list(corr["side"]) == ["best single", "best combination"]
    assert corr["optimism"].notna().all()
    assert (corr["optimism"] > 0).all()
    assert corr["J_corrected"].notna().all()

    single, combo = corr.iloc[0], corr.iloc[1]
    expected = combo["J_corrected"] - single["J_corrected"]
    assert corr["gain_corrected"].iloc[0] == pytest.approx(expected, abs=1e-9)

    row = corr.iloc[1]
    assert row["gain_apparent"] == pytest.approx(
        combo["J_apparent"] - single["J_apparent"], abs=1e-9)
    assert row["correction_effect"] in {"widens", "narrows", "unchanged"}
    expected = ("widens" if row["gain_corrected"] > row["gain_apparent"]
                else "narrows" if row["gain_corrected"] < row["gain_apparent"]
                else "unchanged")
    assert row["correction_effect"] == expected

    # The real cohort's case, isolated: the single side costs more to choose.
    assert mp._correction_effect(0.119, 0.134) == "widens"
    assert mp._correction_effect(0.134, 0.119) == "narrows"
    assert mp._correction_effect(0.100, 0.100) == "unchanged"
    assert mp._correction_effect(np.nan, 0.1) == ""


def test_selection_correction_is_deterministic_for_a_seed():
    args = (count_frame(), COUNT_MARKERS, TARGET)
    a = mp.selection_correction(*args, n_boot=40, seed=7)
    b = mp.selection_correction(*args, n_boot=40, seed=7)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------
def panel_accuracy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": TARGET, "predictor": f"sign_{i}", "kind": "binary"}
        for i in range(3)
    ])


def test_run_marker_panel_writes_every_table_and_honours_exclusions(tmp_output):
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
    )
    written = sorted(p.name for p in (tmp_output / "panel" / "tables").glob("*.csv"))
    assert written == [
        "01_marker_panel.csv",
        "02_marker_panel_reading_view.csv",
        "03_shared_cohort.csv",
        "07_count_score.csv",
        "08_count_thresholds.csv",
        "09_selection_correction.csv",
        "12_count_headline.csv",
    ]
    figures = sorted(p.name for p in (tmp_output / "panel" / "figures").glob("*.png"))
    assert figures == ["count_score.png", "lr_forest.png", "lr_forest_native.png"]
    assert set(tables) >= {"01_marker_panel", "09_selection_correction"}

    mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_2"}, n_boot=40,
    )
    panel = pd.read_csv(tmp_output / "panel" / "tables" / "01_marker_panel.csv")
    assert "sign_2" not in set(panel["marker"])

    # A combination question needs two markers. One must not crash the run.
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_1", "sign_2"}, n_boot=40,
    )
    assert not tables["01_marker_panel"].empty
    assert tables["07_count_score"].empty


# --- which side of the panel a flag belongs to -----------------------------
def test_classify_origin_follows_whether_the_parent_is_still_shown():
    """A flag whose parent is hidden counts as native — it replaced that column,
    so nothing in the table restates anything. ``male`` replaced ``sex``; there
    is no ``sex`` row left for it to duplicate, so it is corrected with the
    other recorded signs. A cut-point whose measurement is still on the page is
    the opposite case: one thing twice.
    """
    got = mp.classify_origin(
        ["male", "cystic_component"],
        derivation_sources={"male": "sex"},
        hidden_parents={"sex"},
    )
    assert got == {"male": mp.NATIVE, "cystic_component": mp.NATIVE}

    got = mp.classify_origin(
        ["adc_value_le0.72"],
        derivation_sources={"adc_value_le0.72": "adc_value"},
        hidden_parents=set(),
    )
    assert got == {"adc_value_le0.72": mp.DERIVED}


def test_derived_rows_take_no_part_in_the_native_family():
    """The native q must be what BH gives on the native rows alone.

    A q is p × m/rank, so a derived row slipping into the native family would
    inflate every native q by the ratio of the two family sizes — uniform, and
    invisible in any single number on the page.
    """
    from eda import benjamini_hochberg

    rng = np.random.default_rng(0)
    n = 80
    y = rng.integers(0, 2, n).astype(bool)
    df = pd.DataFrame({
        TARGET: pd.array(y, dtype="boolean"),
        "sign_a": pd.array(rng.random(n) < 0.3 + 0.3 * y, dtype="boolean"),
        "sign_b": pd.array(rng.random(n) < 0.5, dtype="boolean"),
        "cut_flag": pd.array(rng.random(n) < 0.2 + 0.5 * y, dtype="boolean"),
    })
    markers = [mp.BinaryMarker(c, c.title())
               for c in ("sign_a", "sign_b", "cut_flag")]
    panel = mp.marker_panel_table(
        df, markers, TARGET, origin_by_marker={"cut_flag": mp.DERIVED},
    )
    assert set(panel["origin"]) == {mp.NATIVE, mp.DERIVED}

    native = panel[panel["origin"] == mp.NATIVE]
    expected = benjamini_hochberg(native["p"]).values
    assert np.allclose(native["p_fdr"].to_numpy(float),
                       np.asarray(expected, dtype=float), equal_nan=True)
    # And the derived row is corrected in a family of its own, not left blank.
    derived = panel[panel["origin"] == mp.DERIVED]
    assert derived["p_fdr"].notna().all()
