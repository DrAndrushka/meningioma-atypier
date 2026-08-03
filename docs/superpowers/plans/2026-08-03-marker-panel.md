# Marker Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a section to `output/report/report.html`, after multivariable modelling, that answers the study's two aims — which MRI markers argue hardest for high-grade meningioma, and whether a combination of markers beats any single one.

**Architecture:** A new compute-only module `modelling_phase/marker_panel.py` writes CSVs and SVGs to `output/panel/`; `report.py` gains a renderer that reads them. All combination estimators are reused from `threshold_phase/combinations.py` unchanged, via a duck-typed `BinaryMarker` adapter. Model re-scoring reuses `model_calculator.predict_from_artifact`. No new estimator is written except the positive likelihood ratio.

**Tech Stack:** Python 3, pandas (nullable `boolean`/`Float64` dtypes), numpy, matplotlib (Agg in tests), scikit-learn (`roc_auc_score`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-03-marker-panel-design.md`

## Global Constraints

- **Never modify `heavy_machinery/threshold_phase/combinations.py`.** The reuse claim depends on it. If something there seems to need changing, stop and ask.
- **`report.py` never computes.** It reads CSV/SVG under `output/` and renders. Any statistic must be computed in `marker_panel.py` and saved first.
- Imports in this repo are flat, not packaged: `import combinations as cb`, `import plot_style as ps`. `pytest.ini` sets `pythonpath = . heavy_machinery heavy_machinery/cleaning_phase heavy_machinery/modelling_phase heavy_machinery/threshold_phase`.
- Tests live in `heavy_machinery/pytests_atypier/` and run from the repo root with `python -m pytest`.
- Default seed: `20260801`. Default bootstrap count for selection correction: `500`.
- Missing values use pandas nullable dtypes (`"boolean"`, `"Float64"`), never numpy `bool`/`float` with sentinel NaN.
- Figures are written with `plot_style.save_figure(fig, path)` and are SVG.
- The test suite is fully green before this work starts. Any failure in an untouched test is a regression you caused.

---

## File Structure

| File | Responsibility |
|---|---|
| `heavy_machinery/modelling_phase/diagnostic_accuracy.py` (modify) | Task 0 only: a degenerate 2×2 returns a missing p-value instead of raising |
| `heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py` (modify) | Task 0's tests |
| `heavy_machinery/modelling_phase/marker_panel.py` (create) | All computation: LR+, marker selection, shared cohort, rule menu, count score, selection correction, model re-scoring, MICE stability, figures, and the `run_marker_panel` orchestrator that writes `output/panel/` |
| `heavy_machinery/pytests_atypier/test_marker_panel.py` (create) | Tests for the above |
| `heavy_machinery/modelling_phase/report.py` (modify) | `Artifacts` fields, `load_artifacts` reading `output/panel/`, `render_marker_panel`, insertion into `build_report` |
| `heavy_machinery/pytests_atypier/test_report.py` (modify) | Section renders, and degrades to a warning box when artifacts are absent |
| `meningioma-modelling.ipynb` (modify) | New §04.5 cell calling `run_marker_panel` |

---

### Task 0: Stop a rule that flags nobody from crashing the run

**This is a prerequisite, not a nicety.** `binary_diagnostic_metrics` raises `ValueError` on a 2×2 whose flagged row is all zeros, and the rule search in Task 7 produces exactly such a table. Verified on the real cohort: `brain_invasion AND mri_necrosis` flags 0 patients, so `chi2_contingency` raises *"The internally computed table of expected frequencies has a zero element"*. Without this fix the panel dies on its first real run.

`_chi2_row` in `eda.py` already guards the wholly-empty table (`table.sum() == 0`) but not an empty row or column. Fix it at the `diagnostic_accuracy` boundary rather than in `eda.py`, because that keeps the change inside the one caller that can produce such a table.

**Files:**
- Modify: `heavy_machinery/modelling_phase/diagnostic_accuracy.py:101-104` (`_contingency_pvalue`)
- Test: `heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_contingency_pvalue` returns `(nan, "not applicable")` for a degenerate table instead of raising. `binary_diagnostic_metrics` gains no new keys.

- [ ] **Step 1: Write the failing test**

Append to `heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py`:

```python
def test_a_predictor_that_flags_nobody_returns_metrics_instead_of_raising():
    """An AND of two rare signs can flag zero patients.

    The 2×2 then has an empty row, chi2_contingency raises, and the whole rule
    search dies. There is no test to run on such a table, so the p-value is
    missing — but sensitivity, specificity and the counts are all still real
    numbers and the caller needs them.
    """
    df = pd.DataFrame({
        "flag": pd.array([False] * 8, dtype="boolean"),
        "high_grade": pd.array([True, True, True, False, False, False, False, False],
                               dtype="boolean"),
    })
    row = binary_diagnostic_metrics(df, "high_grade", "flag")
    assert row["TP"] == 0
    assert row["FP"] == 0
    assert row["specificity"] == 1.0
    assert np.isnan(row["p"])
    assert row["test"] == "not applicable"


def test_a_predictor_that_flags_everybody_also_survives():
    df = pd.DataFrame({
        "flag": pd.array([True] * 8, dtype="boolean"),
        "high_grade": pd.array([True, True, True, False, False, False, False, False],
                               dtype="boolean"),
    })
    row = binary_diagnostic_metrics(df, "high_grade", "flag")
    assert row["sensitivity"] == 1.0
    assert np.isnan(row["p"])
```

Check the imports at the top of `test_diagnostic_accuracy.py` — add `numpy as np`, `pandas as pd` or `binary_diagnostic_metrics` only if they are not already there.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py -k flags_nobody -v`
Expected: FAIL with `ValueError: The internally computed table of expected frequencies has a zero element at (1, 0).`

- [ ] **Step 3: Implement**

Replace `_contingency_pvalue` in `heavy_machinery/modelling_phase/diagnostic_accuracy.py`:

```python
def _contingency_pvalue(tp: int, fp: int, fn: int, tn: int) -> tuple[float, str]:
    """χ²/Fisher for the 2×2, or nothing when there is nothing to test.

    A rule that flags nobody — an AND of two rare signs, say — gives a table
    with an empty row. There is no association to test in that table, and
    ``chi2_contingency`` raises rather than saying so. The accuracy numbers
    around it are still real, so the p-value is reported missing and the row
    survives.
    """
    table = np.array([[tp, fp], [fn, tn]], dtype=float)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return float("nan"), "not applicable"
    row = _chi2_row(table)
    return float(row["p"]), str(row["test"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py -v`
Expected: all pass, including the two new ones

- [ ] **Step 5: Check nothing downstream depended on the crash**

Run: `python -m pytest`
Expected: all pass. `combinations.py` calls this on every rule it scores, so `test_combinations.py` passing here is the signal that the threshold phase is unaffected.

- [ ] **Step 6: Commit**

```bash
git add heavy_machinery/modelling_phase/diagnostic_accuracy.py heavy_machinery/pytests_atypier/test_diagnostic_accuracy.py
git commit -m "Report a missing p-value for a rule that flags nobody, instead of raising"
```

---

### Task 1: Positive likelihood ratio

The one statistic this project does not already compute. `diagnostic_accuracy.binary_diagnostic_metrics` gives sensitivity, specificity and an odds ratio, but not LR+.

**Files:**
- Create: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int) -> dict` with keys `lr_pos`, `lr_pos_lo`, `lr_pos_hi`, `chance_overlap` (bool), `continuity_corrected` (bool).

- [ ] **Step 1: Write the failing tests**

Create `heavy_machinery/pytests_atypier/test_marker_panel.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marker_panel'`

- [ ] **Step 3: Write the module and the estimator**

Create `heavy_machinery/modelling_phase/marker_panel.py`:

```python
"""Which MRI markers, and do they combine? — the two study aims, in one place.

The report's EDA section already scores every marker on its own, and the
threshold phase already knows how to compare a combined rule against a single
one. What is missing is a place where those two answers sit side by side on one
patient set, in the report that actually gets read.

Nothing here re-implements an estimator that already exists. The combination
machinery is :mod:`combinations` from the threshold phase, reached through a
nine-line adapter; the model scoring is :mod:`model_calculator`. The only new
statistic is the positive likelihood ratio, which is what turns "most specific"
into a question with a defensible answer — a sign that is never seen is
perfectly specific and perfectly useless.
"""
from __future__ import annotations

import math

import numpy as np

_Z95 = 1.959963984540054


def likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int) -> dict:
    """How much more likely this sign is in a high-grade tumor than a benign one.

    ``LR+ = sensitivity / (1 - specificity)``. An LR+ of 10 means seeing the
    sign makes high grade ten times more likely; an LR+ of 1 means it says
    nothing. The interval is Katz's, computed on the log scale because a ratio
    bounded below by zero and unbounded above is not symmetric.

    A zero in the flagged column makes the ratio infinite and its interval
    undefined. Half a patient is added to every cell in that case
    (Haldane-Anscombe) and ``continuity_corrected`` says so, because the
    resulting interval is honestly enormous and the reader should see why.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
           "chance_overlap": False, "continuity_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = tp == 0 or fp == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    sens = a / (a + c)
    fpr = b / (b + d)
    if fpr <= 0 or sens <= 0:
        return nan

    lr = sens / fpr
    se = math.sqrt(1.0 / a - 1.0 / (a + c) + 1.0 / b - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_pos": float(lr),
        "lr_pos_lo": float(lo),
        "lr_pos_hi": float(hi),
        "chance_overlap": bool(lo <= 1.0 <= hi),
        "continuity_corrected": bool(corrected),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Add the positive likelihood ratio, the one estimator the panel needs"
```

---

### Task 2: The BinaryMarker adapter and marker selection

The reuse claim in the spec — that `combinations.py` works unchanged on plain yes/no columns — is verified here rather than assumed.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: Task 1's module.
- Produces:
  - `BinaryMarker(col: str, label: str)` — NamedTuple with `.short_label` property and `.flag(df) -> pd.Series`.
  - `markers_from_diagnostic_accuracy(table: pd.DataFrame, *, target: str, exclude: Collection[str] = ()) -> list[BinaryMarker]`

- [ ] **Step 1: Write the failing tests**

Append to `heavy_machinery/pytests_atypier/test_marker_panel.py`:

```python
import combinations as cb
from thresholds import Metric


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
def test_binary_marker_flags_match_an_equivalent_cutpoint():
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


def test_binary_marker_short_label_is_the_label():
    """No cut-point to print, so the short form is just the name."""
    marker = mp.BinaryMarker("sign_a", "Sign A")
    assert marker.label == "Sign A"
    assert marker.short_label == "Sign A"


def test_combinations_accepts_binary_markers_unchanged():
    """single_rule_table is threshold-phase code, called here on plain columns."""
    df = marker_frame()
    markers = [mp.BinaryMarker("sign_a", "Sign A"), mp.BinaryMarker("sign_b", "Sign B")]
    table = cb.single_rule_table(df, markers, TARGET)
    assert list(table["rule_label"]) == ["Sign A", "Sign B"]
    assert table["youden_J"].notna().all()


# --------------------------------------------------------------------------
# Marker selection
# --------------------------------------------------------------------------
def test_markers_are_read_from_the_accuracy_table():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert [m.col for m in markers] == ["sign_a", "sign_b", "sex_male"]


def test_the_outcome_is_never_treated_as_a_marker():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert TARGET not in [m.col for m in markers]


def test_continuous_predictors_and_other_targets_are_left_out():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    cols = [m.col for m in markers]
    assert "adc_value" not in cols
    assert "sign_c" not in cols


def test_the_exclude_list_excludes():
    """sex_male is derived_binary and would otherwise enter a section on MRI signs."""
    markers = mp.markers_from_diagnostic_accuracy(
        accuracy_table(), target=TARGET, exclude={"sex_male"},
    )
    assert [m.col for m in markers] == ["sign_a", "sign_b"]


def test_marker_labels_are_prettified():
    markers = mp.markers_from_diagnostic_accuracy(accuracy_table(), target=TARGET)
    assert markers[0].label == "Sign A"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'BinaryMarker'`

- [ ] **Step 3: Implement**

Add to the imports at the top of `marker_panel.py`:

```python
from collections.abc import Collection, Sequence
from typing import NamedTuple

import pandas as pd

import plot_style as ps

BINARY_KINDS = ("binary", "derived_binary")
```

Then append:

```python
class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a ``CutPoint`` so ``combinations`` accepts it.

    :mod:`combinations` touches exactly four members of a cut-point — ``col``,
    ``label``, ``short_label`` and ``flag`` — so supplying those is enough to
    run its whole rule machinery on markers that were never continuous. This is
    why ``combinations.py`` needs no change: the threshold phase's estimators
    and this section's are the same code, and cannot drift apart.
    """

    col: str
    label: str

    @property
    def short_label(self) -> str:
        """No cut-point to name, so the short form is the label itself."""
        return self.label

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].astype("boolean")


def markers_from_diagnostic_accuracy(
    table: pd.DataFrame,
    *,
    target: str,
    exclude: Collection[str] = (),
) -> list[BinaryMarker]:
    """The marker panel, read from the EDA table rather than hard-coded.

    Whatever is activated or dropped in the cleaning notebook's ``DERIVATIONS``
    flows through here without an edit, which is the point: the panel cannot
    silently fall out of step with the cohort it describes.

    ``exclude`` is the caller's, and belongs in the notebook. The accuracy
    table carries non-imaging predictors too — ``sex_male`` is
    ``derived_binary`` and would otherwise walk into a section about MRI signs.
    """
    if table is None or table.empty:
        return []
    drop = set(exclude) | {target}
    rows = table[
        (table["target"].astype(str) == str(target))
        & (table["kind"].astype(str).isin(BINARY_KINDS))
        & (~table["predictor"].astype(str).isin(drop))
    ]
    return [
        BinaryMarker(str(r["predictor"]), ps.prettify_label(str(r["predictor"])))
        for _, r in rows.iterrows()
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 12 passed

If `test_marker_labels_are_prettified` fails because `ps.prettify_label("sign_a")` does not return `"Sign A"`, read `plot_style.prettify_label` and change the assertion to match what it actually produces. Do not change `prettify_label` — it is shared by every figure in the repo.

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Adapt binary markers to the threshold phase's rule machinery"
```

---

### Task 3: The aim-1 marker table

Per-marker accuracy with LR+, plus the column that stops the specificity trap.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `BinaryMarker`, `likelihood_ratio_positive`.
- Produces:
  - `marker_panel_table(df, markers, target) -> pd.DataFrame` — one row per marker with the columns of `binary_diagnostic_metrics` plus `marker`, `present_n`, `catches`, `n_high_grade`, `lr_pos`, `lr_pos_lo`, `lr_pos_hi`, `chance_overlap`, `continuity_corrected`; sorted by `lr_pos` descending with `chance_overlap` rows last.
  - `marker_panel_reading_view(panel) -> pd.DataFrame` — columns `Marker`, `Present in`, `Catches`, `Sens (95% CI)`, `Spec (95% CI)`, `LR+ (95% CI)`.

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
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


def test_marker_reading_view_says_so_instead_of_printing_a_rank():
    df = pd.DataFrame({
        "noise": pd.array([True, False, True, False, True, False, True, False],
                          dtype="boolean"),
        TARGET: pd.array([True, True, False, False, True, False, True, False],
                         dtype="boolean"),
    })
    panel = mp.marker_panel_table(df, [mp.BinaryMarker("noise", "Noise")], TARGET)
    view = mp.marker_panel_reading_view(panel)
    assert list(view.columns) == [
        "Marker", "Present in", "Catches",
        "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
    ]
    assert "not distinguishable from chance" in view.iloc[0]["LR+ (95% CI)"]


def test_marker_panel_is_empty_not_broken_when_there_are_no_markers():
    df = pd.DataFrame({TARGET: pd.array([True, False], dtype="boolean")})
    panel = mp.marker_panel_table(df, [], TARGET)
    assert panel.empty
    assert mp.marker_panel_reading_view(panel).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'marker_panel_table'`

- [ ] **Step 3: Implement**

Add to imports in `marker_panel.py`:

```python
from diagnostic_accuracy import binary_diagnostic_metrics
from thresholds import format_pct_ci
```

Append:

```python
_PANEL_COLUMNS = [
    "marker", "label", "n_used", "present_n", "catches", "n_high_grade",
    "TP", "FP", "FN", "TN",
    "sensitivity", "sensitivity_lo", "sensitivity_hi",
    "specificity", "specificity_lo", "specificity_hi",
    "PPV", "PPV_lo", "PPV_hi", "NPV", "NPV_lo", "NPV_hi",
    "AUC", "OR", "OR_lo", "OR_hi",
    "lr_pos", "lr_pos_lo", "lr_pos_hi", "chance_overlap", "continuity_corrected",
    "p", "test",
]


def marker_panel_table(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """Every marker on its own, ranked by how hard a positive finding argues.

    Ranked by LR+, but ``catches`` is what keeps the ranking honest: the most
    specific sign in a cohort is usually the one nobody ever sees, and a table
    sorted on specificity alone puts it first. Markers whose interval covers 1
    are sorted to the bottom instead of being given a rank they have not
    earned.
    """
    if not markers:
        return pd.DataFrame(columns=_PANEL_COLUMNS)

    rows = []
    for marker in markers:
        row = binary_diagnostic_metrics(
            df, target, marker.col, predictor_series=marker.flag(df),
        )
        row["marker"] = marker.col
        row["label"] = marker.label
        tp, fp = row.get("TP"), row.get("FP")
        fn, tn = row.get("FN"), row.get("TN")
        if any(pd.isna(v) for v in (tp, fp, fn, tn)):
            row.update({"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
                        "chance_overlap": False, "continuity_corrected": False})
            row.update({"present_n": 0, "catches": 0, "n_high_grade": 0})
        else:
            row.update(likelihood_ratio_positive(tp, fp, fn, tn))
            row["present_n"] = int(tp) + int(fp)
            row["catches"] = int(tp)
            row["n_high_grade"] = int(tp) + int(fn)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["chance_overlap"] = out["chance_overlap"].astype(bool)
    out = out.sort_values(
        ["chance_overlap", "lr_pos"], ascending=[True, False], kind="mergesort",
    ).reset_index(drop=True)
    cols = [c for c in _PANEL_COLUMNS if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def _format_lr(row: pd.Series) -> str:
    """``2.8 (1.7-4.6)``, or a sentence when the interval covers 1."""
    if pd.isna(row.get("lr_pos")):
        return "—"
    if bool(row.get("chance_overlap")):
        return "not distinguishable from chance"
    text = f"{row['lr_pos']:.1f} ({row['lr_pos_lo']:.1f}–{row['lr_pos_hi']:.1f})"
    return text + "*" if bool(row.get("continuity_corrected")) else text


def marker_panel_reading_view(panel: pd.DataFrame) -> pd.DataFrame:
    """The aim-1 table in the columns a clinician reads."""
    if panel is None or panel.empty:
        return pd.DataFrame(columns=[
            "Marker", "Present in", "Catches",
            "Sens (95% CI)", "Spec (95% CI)", "LR+ (95% CI)",
        ])
    return pd.DataFrame({
        "Marker": panel["label"],
        "Present in": [f"{int(r['present_n'])}/{int(r['n_used'])}"
                       for _, r in panel.iterrows()],
        "Catches": [f"{int(r['catches'])} of {int(r['n_high_grade'])}"
                    for _, r in panel.iterrows()],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in panel.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in panel.iterrows()],
        "LR+ (95% CI)": [_format_lr(r) for _, r in panel.iterrows()],
    })
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Rank markers by likelihood ratio, with case yield as the guard"
```

---

### Task 4: The LR+ forest figure

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `marker_panel_table` output.
- Produces: `lr_forest_figure(panel: pd.DataFrame) -> matplotlib.figure.Figure`

- [ ] **Step 1: Write the failing test**

Append to `test_marker_panel.py`:

```python
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Aim 1 — the figure
# --------------------------------------------------------------------------
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


def test_lr_forest_returns_a_figure_even_with_nothing_to_plot():
    """An empty panel must not crash the notebook cell that saves figures."""
    fig = mp.lr_forest_figure(pd.DataFrame(columns=["label", "lr_pos"]))
    assert fig is not None
    plt.close(fig)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k forest -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'lr_forest_figure'`

- [ ] **Step 3: Implement**

Add to imports:

```python
import matplotlib.pyplot as plt
```

Append:

```python
def lr_forest_figure(panel: pd.DataFrame) -> plt.Figure:
    """LR+ per marker with its interval, on a log axis with a line at 1.

    Log scale because a likelihood ratio is a multiplier: 0.5 and 2 are the
    same distance from "says nothing", and a linear axis hides that. Markers
    whose interval crosses the line at 1 are drawn in the neutral colour, so
    the ones carrying no information are visible as a group rather than as a
    ranking.
    """
    usable = panel[panel["lr_pos"].notna()] if len(panel) else panel
    if usable is None or usable.empty:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No marker has an estimable likelihood ratio",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ordered = usable.iloc[::-1]
    y = np.arange(len(ordered), dtype=float)
    values = ordered["lr_pos"].to_numpy(dtype=float)
    xerr = ps.errorbar_lengths(values, ordered["lr_pos_lo"], ordered["lr_pos_hi"])
    colors = [
        ps.PALETTE["neutral"] if bool(flag) else ps.PALETTE["high_grade"]
        for flag in ordered["chance_overlap"]
    ]

    height = max(2.0, 0.32 * len(ordered) + 1.0)
    fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_MEDIUM, height))
    ax.axvline(1.0, color=ps.PALETTE["neutral"], linewidth=0.9, linestyle="-.", zorder=1)
    for i, color in enumerate(colors):
        ax.errorbar(values[i], y[i], xerr=xerr[:, i: i + 1], fmt="o",
                    color=color, ecolor=color, elinewidth=1.1, capsize=2.5,
                    markersize=4, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["label"].astype(str))
    ax.set_xlabel("Positive likelihood ratio (log scale)")
    ps.set_titles(
        ax, "How much a positive finding argues for high grade",
        "A ratio of 1 says nothing; grey intervals cross it",
    )
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 18 passed

If `ps.PALETTE` has no `"high_grade"` key, read `plot_style.PALETTE` and use the key that `combinations.py` uses for the outcome colour — it references `ps.PALETTE["high_grade"]` in `count_score_figure`, so it should exist.

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Add the likelihood-ratio forest figure"
```

---

### Task 5: The shared cohort and its audit trail

One denominator for the head-to-head, and a table saying what it cost.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `BinaryMarker`, `combinations.shared_cohort`.
- Produces:
  - `usable_markers(df, markers, target) -> tuple[list[BinaryMarker], list[dict]]` — markers with at least one positive and one negative in `df`, plus one dict per dropped marker with keys `marker`, `reason`.
  - `shared_cohort_frame(df, markers, target) -> pd.DataFrame`
  - `shared_cohort_audit(df, markers, target, dropped) -> pd.DataFrame` — columns `item`, `value`, `note`.

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
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


def test_a_marker_that_never_fires_is_dropped_with_a_reason():
    """An all-false column has an undefined likelihood ratio and no rule value."""
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("always_off", "Off")]
    kept, dropped = mp.usable_markers(df, markers, TARGET)
    assert [m.col for m in kept] == ["sign_a"]
    assert dropped[0]["marker"] == "always_off"
    assert "never" in dropped[0]["reason"].lower()


def test_shared_cohort_audit_records_what_each_marker_cost():
    df = sparse_frame()
    markers = [mp.BinaryMarker("sign_a", "A"), mp.BinaryMarker("sign_b", "B")]
    audit = mp.shared_cohort_audit(df, markers, TARGET, dropped=[])
    assert set(audit.columns) == {"item", "value", "note"}
    rows = dict(zip(audit["item"], audit["value"]))
    assert rows["Patients in the shared set"] == 4
    assert rows["sign_b"] == 2  # patients this marker cost


def test_shared_cohort_is_empty_not_broken_when_no_patient_has_everything():
    df = pd.DataFrame({
        "a": pd.array([True, None], dtype="boolean"),
        "b": pd.array([None, True], dtype="boolean"),
        TARGET: pd.array([True, False], dtype="boolean"),
    })
    markers = [mp.BinaryMarker("a", "A"), mp.BinaryMarker("b", "B")]
    assert mp.shared_cohort_frame(df, markers, TARGET).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k shared -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'shared_cohort_frame'`

- [ ] **Step 3: Implement**

`combinations.shared_cohort` coerces each marker column with `pd.to_numeric`, which works on a nullable boolean column. Append:

```python
def usable_markers(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> tuple[list[BinaryMarker], list[dict]]:
    """Markers that vary. A column that is always the same answers nothing.

    An all-absent sign has no true positives and an undefined likelihood ratio;
    an always-present one has no true negatives. Both would enter the rule
    search and produce cells of zeros, so they are dropped here with the reason
    recorded rather than silently.
    """
    kept: list[BinaryMarker] = []
    dropped: list[dict] = []
    y = df[target].astype("boolean")
    for marker in markers:
        flags = marker.flag(df)[y.notna()]
        n_true = int((flags == True).sum())   # noqa: E712 - nullable boolean
        n_false = int((flags == False).sum())  # noqa: E712
        if n_true == 0:
            dropped.append({"marker": marker.col,
                            "reason": "never present in this cohort"})
        elif n_false == 0:
            dropped.append({"marker": marker.col,
                            "reason": "always present in this cohort"})
        else:
            kept.append(marker)
    return kept, dropped


def shared_cohort_frame(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The patients every rule is scored on — threshold-phase logic, reused."""
    if not markers:
        return df.iloc[0:0].copy()
    return cb.shared_cohort(df, markers, target)


def shared_cohort_audit(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    dropped: Sequence[dict] = (),
) -> pd.DataFrame:
    """What the shared set cost, per marker, so the denominator is auditable.

    A restriction to complete cases is defensible; a restriction nobody can
    check is not. This is the table that lets a reader see the loss is one or
    two measurements rather than the marker panel as a whole.
    """
    shared = shared_cohort_frame(df, markers, target)
    y = df[target].astype("boolean")
    rows: list[dict] = [
        {"item": "Patients in the cohort", "value": int(len(df)), "note": ""},
        {"item": "Patients in the shared set", "value": int(len(shared)),
         "note": "every marker observed and the outcome known"},
        {"item": "High grade in the shared set",
         "value": int(shared[target].astype("boolean").sum()) if len(shared) else 0,
         "note": ""},
        {"item": "Markers required", "value": int(len(markers)),
         "note": ", ".join(m.col for m in markers)},
    ]
    for marker in markers:
        missing = int((marker.flag(df).isna() & y.notna()).sum())
        rows.append({"item": marker.col, "value": missing,
                     "note": "patients this marker alone was missing for"})
    for entry in dropped:
        rows.append({"item": entry["marker"], "value": 0,
                     "note": f"excluded — {entry['reason']}"})
    return pd.DataFrame(rows)
```

Add to imports: `import combinations as cb`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 22 passed

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Score every rule on one patient set, and record what that cost"
```

---

### Task 6: Count score and its figure

The headline answer to aim 2, and the one that needs no optimism correction because no winner is selected.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `BinaryMarker`, `combinations.count_score_table`, `combinations.count_threshold_table`, `combinations.count_score_figure`.
- Produces:
  - `count_score(df, markers, target) -> pd.DataFrame`
  - `count_thresholds(df, markers, target) -> pd.DataFrame`
  - `count_score_figure(counts, markers, prevalence=None) -> plt.Figure`

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
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


def test_count_score_has_a_row_for_every_possible_count():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    assert list(counts["n_criteria_met"]) == [0, 1, 2, 3]
    assert counts["n"].sum() == counts.attrs["n_scored"]


def test_risk_climbs_with_the_number_of_signs_present():
    """The literal claim the section makes. If this fails, the claim is wrong."""
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    risks = counts[counts["n"] >= 10]["risk"].to_numpy(dtype=float)
    assert risks[0] < risks[-1]


def test_count_thresholds_are_scored_as_tests():
    rules = mp.count_thresholds(count_frame(), COUNT_MARKERS, TARGET)
    assert list(rules["rule_label"]) == [
        "≥ 1 of 3 criteria", "≥ 2 of 3 criteria", "≥ 3 of 3 criteria",
    ]
    assert rules["youden_J"].notna().all()


def test_count_score_figure_labels_the_axis_with_the_marker_count():
    counts = mp.count_score(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.count_score_figure(counts, COUNT_MARKERS)
    ax = fig.axes[0]
    assert "3" in ax.get_xlabel()
    plt.close(fig)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k count -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'count_score'`

- [ ] **Step 3: Implement**

Append:

```python
def count_score(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """Observed high-grade rate at each number of signs present.

    The answer to the study aim that involves no choosing. Every other
    comparison in this section picks a winner and then has to pay for having
    picked it; this one asks a question with no winner in it, so the number it
    produces is the number.
    """
    return cb.count_score_table(df, markers, target, complete_only=True)


def count_thresholds(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The count used as a test: "at least one sign", "at least two", …"""
    return cb.count_threshold_table(df, markers, target, complete_only=True)


def count_score_figure(
    counts: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    *,
    prevalence: float | None = None,
) -> plt.Figure:
    """Threshold-phase figure, drawn on markers instead of cut-points."""
    return cb.count_score_figure(counts, cutpoints=markers, prevalence=prevalence)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 26 passed

If `test_risk_climbs_with_the_number_of_signs_present` fails, the simulated frame is too small for the trend to show — raise `n` to 600 in `count_frame`, not the assertion's tolerance. The test exists to check the claim, so weakening it defeats the purpose.

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Add the count score — the headline answer to the combination aim"
```

---

### Task 7: The rule menu and its selection correction

The task that stops the section overstating the combination's advantage. `CHANGES.md` records this exact error happening once in this project.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `BinaryMarker`, `combinations.full_rule_menu`, `combinations.combination_reading_view`, `combinations.bootstrap_best_rule`, `combinations.combination_figure`.
- Produces:
  - `rule_menu(df, markers, target, *, max_size=2) -> pd.DataFrame`
  - `rule_reading_view(menu, *, top=12) -> pd.DataFrame`
  - `selection_correction(df, markers, target, *, n_boot=500, seed=20260801, max_size=2) -> pd.DataFrame` — two rows, `side` in `{"best single", "best combination"}`, columns `side`, `best_rule`, `J_apparent`, `optimism`, `J_corrected`, `winner_stability`, `n_bootstrap`, plus a `gain_corrected` column repeating the corrected difference on both rows.
  - `rule_space_figure(menu, *, top=12) -> plt.Figure`

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
# --------------------------------------------------------------------------
# Aim 2 — the rule menu, and paying for having picked a winner
# --------------------------------------------------------------------------
def test_rule_menu_holds_singles_and_combinations_together():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    kinds = set(menu["kind"])
    assert {"single", "and", "or", "count"} <= kinds
    assert (menu["n_used"] > 0).all()


def test_both_sides_of_the_head_to_head_are_corrected():
    """The CHANGES.md regression.

    A corrected combination scored against an *uncorrected* single flatters the
    combination by the whole of the single's own selection optimism. Picking the
    best of N single markers is a choice made on these patients too, so it costs
    something, and that cost must be non-zero and recorded.
    """
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    assert list(corr["side"]) == ["best single", "best combination"]
    assert corr["optimism"].notna().all()
    assert (corr["optimism"] > 0).all()
    assert corr["J_corrected"].notna().all()


def test_the_reported_gain_is_corrected_on_both_sides():
    corr = mp.selection_correction(count_frame(), COUNT_MARKERS, TARGET, n_boot=60)
    single = corr[corr["side"] == "best single"].iloc[0]
    combo = corr[corr["side"] == "best combination"].iloc[0]
    expected = combo["J_corrected"] - single["J_corrected"]
    assert corr["gain_corrected"].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_selection_correction_is_deterministic_for_a_seed():
    args = (count_frame(), COUNT_MARKERS, TARGET)
    a = mp.selection_correction(*args, n_boot=40, seed=7)
    b = mp.selection_correction(*args, n_boot=40, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_rule_reading_view_is_ranked_by_youden_j():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    view = mp.rule_reading_view(menu, top=5)
    assert len(view) == 5
    assert list(view["J"]) == sorted(view["J"], reverse=True)


def test_rule_space_figure_draws():
    menu = mp.rule_menu(count_frame(), COUNT_MARKERS, TARGET)
    fig = mp.rule_space_figure(menu)
    assert fig.axes
    plt.close(fig)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k rule -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'rule_menu'`

- [ ] **Step 3: Implement**

Append:

```python
DEFAULT_SEED = 20260801
DEFAULT_N_BOOT = 500


def rule_menu(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    max_size: int = 2,
) -> pd.DataFrame:
    """Singles, AND/OR pairs and count rules, all on one patient set.

    Pairs only. ``max_size=3`` is available but an AND of three signs on this
    many events lands in single figures, and a sensitivity computed from eight
    patients is not a number worth printing.
    """
    return cb.full_rule_menu(df, markers, target, max_size=max_size)


def rule_reading_view(menu: pd.DataFrame, *, top: int | None = 12) -> pd.DataFrame:
    """Rules ranked by Youden J, in clinician-facing columns."""
    return cb.combination_reading_view(menu, top=top)


def selection_correction(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> pd.DataFrame:
    """What the winner's advantage is worth once you pay for having picked it.

    With a dozen markers there are hundreds of candidate rules, and the best of
    hundreds beats the best single marker on the data that chose it even when
    no rule is genuinely better. :func:`combinations.bootstrap_best_rule`
    measures that gap by rebuilding the whole menu on each resample, taking its
    winner, and scoring that same rule back on the original cohort.

    It is run **twice**, with the same budget and seed. Correcting only the
    combination and comparing it against an uncorrected single is the error
    recorded in ``CHANGES.md``: it reported a gain of +0.008 that was really
    +0.050, because choosing the best of the singles costs almost exactly as
    much as choosing the best of the combinations.
    """
    sides = [
        ("best single", ("single",)),
        ("best combination", ("and", "or", "count")),
    ]
    rows: list[dict] = []
    for label, kinds in sides:
        result = cb.bootstrap_best_rule(
            df, markers, target, n_boot=n_boot, seed=seed,
            max_size=max_size, kinds=kinds,
        )
        rows.append({
            "side": label,
            "best_rule": result.get("best_rule", ""),
            "J_apparent": result.get("J_apparent", np.nan),
            "optimism": result.get("optimism", np.nan),
            "J_corrected": result.get("J_corrected", np.nan),
            "winner_stability": result.get("winner_stability", np.nan),
            "n_bootstrap": result.get("n_bootstrap", 0),
        })
    out = pd.DataFrame(rows)
    gain = float(out.loc[1, "J_corrected"] - out.loc[0, "J_corrected"])
    out["gain_corrected"] = gain
    return out


def rule_space_figure(menu: pd.DataFrame, *, top: int = 12) -> plt.Figure:
    """Every rule in sensitivity-specificity space, singles marked apart."""
    return cb.combination_figure(menu, top=top)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 32 passed

If `test_both_sides_of_the_head_to_head_are_corrected` fails with `optimism` not greater than zero, raise `n_boot` to 200 in that test — 60 resamples can average to a very small number. Do not relax the assertion to `>= 0`; a zero optimism would mean the correction is not running.

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Correct both sides of the combination-vs-single comparison"
```

---

### Task 8: Re-score the multivariable models on the shared set

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Background the implementer needs:** each file in `output/inferential/model_artifacts/*_model.json` holds `coefficients` (encoded variable names), `features` (raw column names with their encodings and standardisation constants), and `validation.metrics`, a list of dicts like `{"metric": "AUC", "apparent": 0.729, "optimism_corrected": 0.705}`. `model_calculator.predict_from_artifact(user_inputs, artifact)` turns one patient's raw values into a probability. The raw feature columns (`tumor_location`, `tumor_margin`, `sex`, and the binary signs) exist in the cohort frame with matching level strings, so no new encoding logic is needed.

**Interfaces:**
- Consumes: `model_calculator.predict_from_artifact`, `model_calculator.load_model_artifact`.
- Produces:
  - `score_model_on(df, artifact) -> pd.Series` — predicted probability per row, `NaN` where any feature is missing.
  - `model_vs_single(df, artifacts, target, panel_correction) -> pd.DataFrame` — columns `model`, `n_scored`, `auc_shared_apparent`, `auc_artifact_corrected`, `auc_artifact_apparent`, `best_single_rule`, `best_single_J_corrected`, `note`.

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
# --------------------------------------------------------------------------
# Aim 2 — the multivariable comparison
# --------------------------------------------------------------------------
def tiny_artifact() -> dict:
    """A two-predictor logistic model in the shape the pipeline saves."""
    return {
        "model_name": "Tiny model",
        "target": TARGET,
        "coefficients": {"const": -1.0, "sign_0": 1.5, "sign_1": 0.8},
        "features": [
            {"name": "sign_0", "type": "binary",
             "encoding": {"sign_0": {"true": 1, "false": 0}}},
            {"name": "sign_1", "type": "binary",
             "encoding": {"sign_1": {"true": 1, "false": 0}}},
        ],
        "validation": {"metrics": [
            {"metric": "AUC", "apparent": 0.71, "optimism_corrected": 0.68},
        ]},
    }


def test_scoring_a_model_gives_one_probability_per_patient():
    df = count_frame()
    probs = mp.score_model_on(df, tiny_artifact())
    assert len(probs) == len(df)
    assert probs.between(0, 1).all()


def test_a_patient_missing_a_predictor_scores_nan_not_a_guess():
    """Imputing silently inside a scoring helper would be a lie by omission."""
    df = count_frame().copy()
    df.loc[df.index[0], "sign_0"] = pd.NA
    probs = mp.score_model_on(df, tiny_artifact())
    assert pd.isna(probs.iloc[0])
    assert probs.iloc[1:].notna().all()


def test_model_vs_single_keeps_the_two_aucs_in_separate_labelled_columns():
    """The artifact AUC and the re-scored AUC are different patients.

    Collapsing them into one column is the denominator mistake this section
    exists to avoid, so the table carries both and says which is which.
    """
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)

    row = table.iloc[0]
    assert row["model"] == "tiny"
    assert row["auc_artifact_corrected"] == 0.68
    assert row["auc_artifact_apparent"] == 0.71
    assert 0.0 <= row["auc_shared_apparent"] <= 1.0
    assert row["n_scored"] == len(df)
    assert row["best_single_rule"] == correction.iloc[0]["best_rule"]


def test_a_model_whose_predictors_are_absent_is_reported_not_dropped():
    df = count_frame().drop(columns=["sign_0"])
    correction = mp.selection_correction(df, COUNT_MARKERS[1:], TARGET, n_boot=40)
    table = mp.model_vs_single(df, {"tiny": tiny_artifact()}, TARGET, correction)
    assert pd.isna(table.iloc[0]["auc_shared_apparent"])
    assert "sign_0" in table.iloc[0]["note"]


def test_model_vs_single_is_empty_when_there_are_no_artifacts():
    df = count_frame()
    correction = mp.selection_correction(df, COUNT_MARKERS, TARGET, n_boot=40)
    assert mp.model_vs_single(df, {}, TARGET, correction).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k model -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'score_model_on'`

- [ ] **Step 3: Implement**

Add to imports:

```python
from sklearn.metrics import roc_auc_score

from model_calculator import predict_from_artifact
```

Append:

```python
def _artifact_auc(artifact: dict, key: str) -> float:
    """Read one stored AUC (``apparent`` or ``optimism_corrected``) from a model."""
    metrics = (artifact.get("validation") or {}).get("metrics") or []
    for entry in metrics:
        if str(entry.get("metric")) == "AUC":
            value = entry.get(key)
            return float(value) if value is not None else np.nan
    return np.nan


def score_model_on(df: pd.DataFrame, artifact: dict) -> pd.Series:
    """Apply a saved model's coefficients to this cohort, row by row.

    Re-scoring, not refitting. The coefficients are the ones the modelling
    phase already fitted and shrank; only the patient set changes, which is the
    whole point — a model AUC computed on 352 patients and a marker's Youden J
    computed on 301 are not comparable, and the fix is to move the model, not
    to hope the difference is small.

    A patient missing any predictor scores ``NaN``. Filling one in here would
    be imputation smuggled into a scoring helper, and the section's whole claim
    is that its accuracy numbers describe findings someone actually saw.
    """
    features = artifact.get("features") or []
    names = [str(f["name"]) for f in features]
    missing_cols = [n for n in names if n not in df.columns]
    if missing_cols:
        raise KeyError(f"cohort is missing model predictors: {missing_cols}")

    out: list[float] = []
    for _, row in df.iterrows():
        values = {n: row[n] for n in names}
        if any(pd.isna(v) for v in values.values()):
            out.append(np.nan)
            continue
        inputs = {
            n: (bool(v) if str(f.get("type")) == "binary" else v)
            for (n, v), f in zip(values.items(), features)
        }
        out.append(float(predict_from_artifact(inputs, artifact)))
    return pd.Series(out, index=df.index, dtype="float64")


def model_vs_single(
    df: pd.DataFrame,
    artifacts: dict[str, dict],
    target: str,
    correction: pd.DataFrame,
) -> pd.DataFrame:
    """Each multivariable model against the best single marker, labelled honestly.

    Three AUC columns, deliberately not collapsed into one:

    ``auc_shared_apparent``   the model re-scored on the shared set — the
                              like-for-like comparison, and *apparent*, because
                              correcting it would mean re-running the bootstrap
                              on this set, which is refitting.
    ``auc_artifact_corrected`` the model's own optimism-corrected AUC from its
                              artifact, on its own patients. The gap between
                              this and ``auc_artifact_apparent`` bounds how
                              optimistic the re-scored column is.
    ``best_single_J_corrected`` the single-marker side, corrected, from
                              :func:`selection_correction`.
    """
    if not artifacts:
        return pd.DataFrame(columns=[
            "model", "n_scored", "auc_shared_apparent", "auc_artifact_corrected",
            "auc_artifact_apparent", "best_single_rule", "best_single_J_corrected",
            "note",
        ])

    single = correction[correction["side"] == "best single"]
    best_rule = str(single["best_rule"].iloc[0]) if len(single) else ""
    best_j = float(single["J_corrected"].iloc[0]) if len(single) else np.nan
    y = df[target].astype("boolean")

    rows: list[dict] = []
    for name, artifact in artifacts.items():
        note = ""
        auc = np.nan
        n_scored = 0
        try:
            probs = score_model_on(df, artifact)
            usable = probs.notna() & y.notna()
            n_scored = int(usable.sum())
            truth = y[usable].astype(int)
            if n_scored and truth.nunique() == 2:
                auc = float(roc_auc_score(truth, probs[usable]))
            else:
                note = "not scorable on this set — one outcome class only"
        except KeyError as exc:
            note = f"not scorable on this set — {exc.args[0]}"

        rows.append({
            "model": name,
            "n_scored": n_scored,
            "auc_shared_apparent": auc,
            "auc_artifact_corrected": _artifact_auc(artifact, "optimism_corrected"),
            "auc_artifact_apparent": _artifact_auc(artifact, "apparent"),
            "best_single_rule": best_rule,
            "best_single_J_corrected": best_j,
            "note": note,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 37 passed

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Re-score saved models on the shared set instead of comparing across cohorts"
```

---

### Task 9: MICE stability check

Answers "does filling in the missing scans change the story", in the only currency that survives: reproduction rates, not pooled estimates.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `marker_panel_table`, `rule_menu`, `selection_correction`.
- Produces: `imputation_stability(draws, markers, target, *, n_boot=200, seed=DEFAULT_SEED) -> pd.DataFrame` — columns `item`, `value`, `note`.

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
# --------------------------------------------------------------------------
# Does filling in the missing scans change the story?
# --------------------------------------------------------------------------
def test_imputation_stability_reports_reproduction_rates():
    """Rubin's rules can average an estimate, but not a choice.

    A different rule can win in each draw, so the honest output is "the same
    rule won in X% of draws", not a pooled winner.
    """
    draws = [count_frame(seed=s) for s in (1, 2, 3)]
    out = mp.imputation_stability(draws, COUNT_MARKERS, TARGET, n_boot=30)
    items = dict(zip(out["item"], out["value"]))
    assert items["Draws"] == 3
    assert 0.0 <= items["Top marker reproduced"] <= 1.0
    assert 0.0 <= items["Winning rule reproduced"] <= 1.0
    assert 0.0 <= items["Combination still beat the best single"] <= 1.0


def test_imputation_stability_says_so_when_there_are_no_draws():
    out = mp.imputation_stability([], COUNT_MARKERS, TARGET)
    assert dict(zip(out["item"], out["value"]))["Draws"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k imputation -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'imputation_stability'`

- [ ] **Step 3: Implement**

Append:

```python
def imputation_stability(
    draws: Sequence[pd.DataFrame],
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    n_boot: int = 200,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """How often the observed-data story survives filling in the missing scans.

    Reported as reproduction rates, not pooled estimates. Rubin's rules average
    an *estimate*; they cannot average a *choice*, and both headline answers
    here are choices — which marker ranks first, which rule wins. A pooled
    "winner" would be an average of things that are not on the same scale.

    The modal answer is carried alongside each rate so a low rate is
    interpretable: "the same rule won in 40% of draws, and the runner-up was
    this one" says something; "40%" alone does not.
    """
    if not draws:
        return pd.DataFrame([{"item": "Draws", "value": 0,
                              "note": "no MICE draws were found"}])

    top_markers: list[str] = []
    winners: list[str] = []
    combo_wins = 0
    scored = 0

    for i, draw in enumerate(draws):
        kept, _ = usable_markers(draw, markers, target)
        if len(kept) < 2:
            continue
        panel = marker_panel_table(draw, kept, target)
        if not panel.empty:
            top_markers.append(str(panel.iloc[0]["marker"]))
        corr = selection_correction(draw, kept, target, n_boot=n_boot, seed=seed + i)
        winners.append(str(corr.loc[1, "best_rule"]))
        if float(corr.loc[0, "gain_corrected"]) > 0:
            combo_wins += 1
        scored += 1

    def _rate(values: Sequence[str]) -> tuple[float, str]:
        if not values:
            return (np.nan, "")
        counts = pd.Series(values).value_counts()
        return (float(counts.iloc[0] / len(values)), str(counts.index[0]))

    top_rate, top_mode = _rate(top_markers)
    win_rate, win_mode = _rate(winners)
    return pd.DataFrame([
        {"item": "Draws", "value": int(len(draws)), "note": f"{scored} scorable"},
        {"item": "Top marker reproduced", "value": top_rate,
         "note": f"most often: {top_mode}" if top_mode else ""},
        {"item": "Winning rule reproduced", "value": win_rate,
         "note": f"most often: {win_mode}" if win_mode else ""},
        {"item": "Combination still beat the best single",
         "value": (combo_wins / scored) if scored else np.nan,
         "note": "share of draws with a positive corrected gain"},
    ])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 39 passed

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Check the panel's story against the 20 MICE draws"
```

---

### Task 10: The orchestrator that writes `output/panel/`

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py`
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_marker_panel(df, *, target, accuracy_table, output_root, exclude=(), artifacts=None, draws=(), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, max_size=2) -> dict[str, pd.DataFrame]` — writes `output/panel/tables/*.csv` and `output/panel/figures/*.svg`, returns the tables by filename stem.

- [ ] **Step 1: Write the failing tests**

Append to `test_marker_panel.py`:

```python
# --------------------------------------------------------------------------
# The orchestrator
# --------------------------------------------------------------------------
def panel_accuracy_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"target": TARGET, "predictor": f"sign_{i}", "kind": "binary"}
        for i in range(3)
    ])


def test_run_marker_panel_writes_every_table_and_figure(tmp_output):
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
    )
    written = sorted(p.name for p in (tmp_output / "panel" / "tables").glob("*.csv"))
    assert written == [
        "01_marker_panel.csv",
        "02_marker_panel_reading_view.csv",
        "03_shared_cohort.csv",
        "05_rule_menu.csv",
        "06_rule_reading_view.csv",
        "07_count_score.csv",
        "08_count_thresholds.csv",
        "09_selection_correction.csv",
        "10_model_vs_single.csv",
        "11_imputation_stability.csv",
    ]
    figures = sorted(p.name for p in (tmp_output / "panel" / "figures").glob("*.svg"))
    assert figures == ["count_score.svg", "lr_forest.svg", "rule_space.svg"]
    assert set(tables) >= {"01_marker_panel", "09_selection_correction"}


def test_run_marker_panel_excludes_what_it_is_told_to(tmp_output):
    mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_2"}, n_boot=40,
    )
    panel = pd.read_csv(tmp_output / "panel" / "tables" / "01_marker_panel.csv")
    assert "sign_2" not in set(panel["marker"])


def test_run_marker_panel_survives_a_single_usable_marker(tmp_output):
    """A combination question needs two markers. One must not crash the run."""
    df = count_frame()
    tables = mp.run_marker_panel(
        df, target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, exclude={"sign_1", "sign_2"}, n_boot=40,
    )
    assert not tables["01_marker_panel"].empty
    assert tables["05_rule_menu"].empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k run_marker -v`
Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'run_marker_panel'`

- [ ] **Step 3: Implement**

Add to imports:

```python
from pathlib import Path

from cleaning import format_table_for_csv
```

Append:

```python
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"


def _write_table(tables: dict, root: Path, stem: str, frame: pd.DataFrame) -> None:
    tables[stem] = frame
    (root / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    format_table_for_csv(frame).to_csv(
        root / TABLES_DIRNAME / f"{stem}.csv", index=False,
    )


def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    artifacts: dict[str, dict] | None = None,
    draws: Sequence[pd.DataFrame] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> dict[str, pd.DataFrame]:
    """Compute the whole panel and write it to ``output/panel/``.

    Two patient sets on purpose. The aim-1 marker table uses each marker's own
    complete cases, because it ranks markers and each row stands alone. Every
    aim-2 comparison uses the shared set, because a Youden J compared across
    two different groups of patients is not a comparison.

    Nothing here renders. ``report.py`` reads what this writes.
    """
    root = Path(output_root) / "panel"
    tables: dict[str, pd.DataFrame] = {}

    markers = markers_from_diagnostic_accuracy(
        accuracy_table, target=target, exclude=exclude,
    )
    markers = [m for m in markers if m.col in df.columns]
    kept, dropped = usable_markers(df, markers, target)

    panel = marker_panel_table(df, kept, target)
    _write_table(tables, root, "01_marker_panel", panel)
    _write_table(tables, root, "02_marker_panel_reading_view",
                 marker_panel_reading_view(panel))

    shared = shared_cohort_frame(df, kept, target)
    _write_table(tables, root, "03_shared_cohort",
                 shared_cohort_audit(df, kept, target, dropped))

    empty = pd.DataFrame()
    if len(kept) >= 2 and not shared.empty:
        menu = rule_menu(shared, kept, target, max_size=max_size)
        correction = selection_correction(
            shared, kept, target, n_boot=n_boot, seed=seed, max_size=max_size,
        )
        counts = count_score(shared, kept, target)
        _write_table(tables, root, "05_rule_menu", menu)
        _write_table(tables, root, "06_rule_reading_view", rule_reading_view(menu))
        _write_table(tables, root, "07_count_score", counts)
        _write_table(tables, root, "08_count_thresholds",
                     count_thresholds(shared, kept, target))
        _write_table(tables, root, "09_selection_correction", correction)
        _write_table(tables, root, "10_model_vs_single",
                     model_vs_single(shared, artifacts or {}, target, correction))
        _write_table(tables, root, "11_imputation_stability",
                     imputation_stability(list(draws), kept, target, seed=seed))
    else:
        for stem in ("05_rule_menu", "06_rule_reading_view", "07_count_score",
                     "08_count_thresholds", "09_selection_correction",
                     "10_model_vs_single", "11_imputation_stability"):
            _write_table(tables, root, stem, empty)
        menu, counts = empty, empty

    fig_dir = root / FIGURES_DIRNAME
    ps.save_figure(lr_forest_figure(panel), fig_dir / "lr_forest.svg")
    prevalence = (
        float(shared[target].astype("boolean").mean()) if len(shared) else None
    )
    if len(counts):
        ps.save_figure(count_score_figure(counts, kept, prevalence=prevalence),
                       fig_dir / "count_score.svg")
        ps.save_figure(rule_space_figure(menu), fig_dir / "rule_space.svg")
    else:
        for name in ("count_score.svg", "rule_space.svg"):
            fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM,
                                                          aspect=0.4))
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Not enough markers for a combination",
                    ha="center", va="center", transform=ax.transAxes)
            ps.save_figure(fig, fig_dir / name)

    return tables
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v`
Expected: 42 passed

- [ ] **Step 5: Run the whole suite to check nothing else broke**

Run: `python -m pytest`
Expected: all pass. Any failure outside `test_marker_panel.py` is a regression — fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "Write the marker panel to output/panel/"
```

---

### Task 11: Render the section in report.html

**Files:**
- Modify: `heavy_machinery/modelling_phase/report.py` — `Artifacts` (around line 750), `load_artifacts` (around line 880, just before `return art`), a new `render_marker_panel` (after `render_inferential`, around line 2516), and `build_report` (around line 3260)
- Modify: `heavy_machinery/pytests_atypier/test_report.py`

**Interfaces:**
- Consumes: the CSVs and SVGs from Task 10.
- Produces: `render_marker_panel(cfg: ReportConfig, art: Artifacts) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `heavy_machinery/pytests_atypier/test_report.py`:

```python
# --------------------------------------------------------------------------
# Marker panel section
# --------------------------------------------------------------------------
@pytest.fixture
def panel_output(tmp_output):
    tables = tmp_output / "panel" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([
        {"marker": "cortical_destruction", "label": "Cortical destruction",
         "n_used": 300, "present_n": 50, "catches": 27, "n_high_grade": 105,
         "lr_pos": 2.76, "lr_pos_lo": 1.66, "lr_pos_hi": 4.59,
         "chance_overlap": False, "continuity_corrected": False},
    ]).to_csv(tables / "01_marker_panel.csv", index=False)
    pd.DataFrame([
        {"Marker": "Cortical destruction", "Present in": "50/300",
         "Catches": "27 of 105", "Sens (95% CI)": "26% (19–35)",
         "Spec (95% CI)": "91% (86–94)", "LR+ (95% CI)": "2.8 (1.7–4.6)"},
    ]).to_csv(tables / "02_marker_panel_reading_view.csv", index=False)
    pd.DataFrame([
        {"item": "Patients in the shared set", "value": 301, "note": "every marker observed"},
    ]).to_csv(tables / "03_shared_cohort.csv", index=False)
    pd.DataFrame([
        {"n_criteria_met": 0, "n": 100, "n_high_grade": 11, "risk": 0.11,
         "risk_lo": 0.06, "risk_hi": 0.19},
        {"n_criteria_met": 1, "n": 90, "n_high_grade": 30, "risk": 0.33,
         "risk_lo": 0.24, "risk_hi": 0.43},
    ]).to_csv(tables / "07_count_score.csv", index=False)
    pd.DataFrame([
        {"side": "best single", "best_rule": "Cortical destruction",
         "J_apparent": 0.21, "optimism": 0.04, "J_corrected": 0.17,
         "winner_stability": 0.41, "n_bootstrap": 500, "gain_corrected": 0.06},
        {"side": "best combination", "best_rule": "Cortical destruction OR Edema",
         "J_apparent": 0.27, "optimism": 0.04, "J_corrected": 0.23,
         "winner_stability": 0.33, "n_bootstrap": 500, "gain_corrected": 0.06},
    ]).to_csv(tables / "09_selection_correction.csv", index=False)
    pd.DataFrame([
        {"Rule": "Cortical destruction OR Edema", "Type": "or", "n": 301,
         "Sens (95% CI)": "58% (48–67)", "Spec (95% CI)": "65% (58–71)",
         "PPV (95% CI)": "44% (36–53)", "NPV (95% CI)": "76% (69–82)",
         "TP/FP/FN/TN": "55/70/40/136", "OR (95% CI)": "2.7 (1.6–4.4)", "J": 0.23},
    ]).to_csv(tables / "06_rule_reading_view.csv", index=False)
    pd.DataFrame([
        {"model": "amano_et_al_2021", "n_scored": 301,
         "auc_shared_apparent": 0.74, "auc_artifact_corrected": 0.733,
         "auc_artifact_apparent": 0.749, "best_single_rule": "Cortical destruction",
         "best_single_J_corrected": 0.17, "note": ""},
    ]).to_csv(tables / "10_model_vs_single.csv", index=False)
    pd.DataFrame([
        {"item": "Draws", "value": 20, "note": "20 scorable"},
        {"item": "Winning rule reproduced", "value": 0.4,
         "note": "most often: Cortical destruction OR Edema"},
    ]).to_csv(tables / "11_imputation_stability.csv", index=False)

    figures = tmp_output / "panel" / "figures"
    figures.mkdir(parents=True)
    for name in ("lr_forest.svg", "count_score.svg", "rule_space.svg"):
        (figures / name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
            encoding="utf-8",
        )
    return tmp_output


def test_load_artifacts_finds_the_panel_tables(panel_output):
    art = load_artifacts(ReportConfig(output_root=panel_output, title="T"))
    assert art.panel_marker_reading_view is not None
    assert art.panel_selection_correction is not None
    assert len(art.panel_figures) == 3


def test_marker_panel_section_answers_both_aims(panel_output):
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "Cortical destruction" in html
    assert "2.8 (1.7–4.6)" in html
    assert "301" in html


def test_marker_panel_section_quotes_the_corrected_gain_not_the_apparent_one(
    panel_output,
):
    """0.06 is the corrected gain; 0.27 is the apparent combination J.

    Quoting the apparent number is the CHANGES.md mistake in prose form.
    """
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "0.06" in html


def test_marker_panel_degrades_to_a_warning_when_nothing_was_computed(tmp_output):
    cfg = ReportConfig(output_root=tmp_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "warning" in html.lower()


def test_the_panel_section_sits_between_modelling_and_the_appendix(panel_output):
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = build_report(cfg)
    assert html.index("Multivariable modelling") < html.index("Which MRI markers")
    assert html.index("Which MRI markers") < html.index("📎 Appendix")
```

Add `render_marker_panel` to the `from report import (...)` list at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_report.py -k panel -v`
Expected: FAIL — `ImportError: cannot import name 'render_marker_panel' from 'report'`

- [ ] **Step 3: Add the Artifacts fields**

In `report.py`, in the `Artifacts` dataclass, after the `inferential_figures` line and before the `# Warnings accumulated during load` comment:

```python
    # Marker panel
    panel_marker: pd.DataFrame | None = None
    panel_marker_reading_view: pd.DataFrame | None = None
    panel_shared_cohort: pd.DataFrame | None = None
    panel_rule_reading_view: pd.DataFrame | None = None
    panel_count_score: pd.DataFrame | None = None
    panel_selection_correction: pd.DataFrame | None = None
    panel_model_vs_single: pd.DataFrame | None = None
    panel_imputation_stability: pd.DataFrame | None = None
    panel_figures: list[Path] = field(default_factory=list)
```

- [ ] **Step 4: Add the loader**

In `load_artifacts`, immediately before `return art`:

```python
    # Marker panel
    panel_tab = root / "panel" / "tables"
    art.panel_marker = _maybe_read_csv(panel_tab / "01_marker_panel.csv", art.warnings)
    art.panel_marker_reading_view = _maybe_read_csv(
        panel_tab / "02_marker_panel_reading_view.csv", art.warnings,
    )
    art.panel_shared_cohort = _maybe_read_csv(
        panel_tab / "03_shared_cohort.csv", art.warnings,
    )
    art.panel_rule_reading_view = _maybe_read_csv(
        panel_tab / "06_rule_reading_view.csv", art.warnings,
    )
    art.panel_count_score = _maybe_read_csv(
        panel_tab / "07_count_score.csv", art.warnings,
    )
    art.panel_selection_correction = _maybe_read_csv(
        panel_tab / "09_selection_correction.csv", art.warnings,
    )
    art.panel_model_vs_single = _maybe_read_csv(
        panel_tab / "10_model_vs_single.csv", art.warnings,
    )
    art.panel_imputation_stability = _maybe_read_csv(
        panel_tab / "11_imputation_stability.csv", art.warnings,
    )
    panel_fig = root / "panel" / "figures"
    if panel_fig.exists():
        art.panel_figures = sorted(panel_fig.glob("*.svg"))
```

- [ ] **Step 5: Write the renderer**

In `report.py`, after `render_inferential` ends (before `def _to_int_or_none`):

```python
def _panel_figure(art: Artifacts, stem: str) -> str:
    """One panel SVG by filename stem, or nothing if it was not written."""
    for path in art.panel_figures:
        if path.stem == stem:
            return _figure_img_html(path)
    return ""


def _panel_shared_n(art: Artifacts) -> str:
    """The denominator the head-to-head was run on, quoted from its own table."""
    table = art.panel_shared_cohort
    if table is None or table.empty:
        return "—"
    row = table[table["item"].astype(str) == "Patients in the shared set"]
    return _int(row["value"].iloc[0]) if len(row) else "—"


def _panel_aim_one(art: Artifacts) -> str:
    view = art.panel_marker_reading_view
    if view is None or view.empty:
        return warning_box("No marker table was found.")
    top = art.panel_marker.iloc[0] if art.panel_marker is not None and \
        not art.panel_marker.empty else None
    lead = ""
    if top is not None and not bool(top.get("chance_overlap")):
        lead = _lead(
            f"<strong>{_esc(top['label'])}</strong> argues hardest for high grade: "
            f"seeing it makes high grade {_num(top['lr_pos'], 1)}× more likely "
            f"({_num(top['lr_pos_lo'], 1)}–{_num(top['lr_pos_hi'], 1)}). "
            f"It is present in {_int(top['present_n'])} of "
            f"{_int(top['n_used'])} scans and flags "
            f"{_int(top['catches'])} of the {_int(top['n_high_grade'])} "
            "high-grade tumours."
        )
    return (
        "<h3>Which sign argues hardest for high grade</h3>"
        + lead
        + "<p>Ranked by <strong>positive likelihood ratio</strong> — how many "
          "times more often the sign appears in a high-grade tumour than in a "
          "benign one. <strong>Catches</strong> is there because the most "
          "specific sign in any cohort is usually the one nobody ever sees: a "
          "marker that is almost never present is almost perfectly specific "
          "and almost never useful. A ratio whose interval covers 1 says "
          "nothing, and is labelled rather than ranked.</p>"
        + _table(view)
        + _panel_figure(art, "lr_forest")
    )


def _panel_aim_two(art: Artifacts) -> str:
    counts = art.panel_count_score
    if counts is None or counts.empty:
        return "<h3>Does a combination beat one sign?</h3>" + info_box(
            "A combination needs at least two usable markers on a shared set of "
            "patients; this run did not have them."
        )

    usable = counts[counts["n"] > 0]
    lead = ""
    if len(usable) >= 2:
        first, last = usable.iloc[0], usable.iloc[-1]
        lead = _lead(
            f"Risk rises from {_pct(first['risk'])} with "
            f"{_int(first['n_criteria_met'])} of the signs present to "
            f"{_pct(last['risk'])} with {_int(last['n_criteria_met'])}."
        )

    corr = art.panel_selection_correction
    correction_html = ""
    if corr is not None and not corr.empty:
        single = corr[corr["side"] == "best single"]
        combo = corr[corr["side"] == "best combination"]
        if len(single) and len(combo):
            s, c = single.iloc[0], combo.iloc[0]
            correction_html = (
                "<p>Head-to-head on the same "
                f"{_panel_shared_n(art)} patients, both sides corrected for "
                "having been picked here: the best single sign "
                f"(<em>{_esc(s['best_rule'])}</em>) scores "
                f"{_num(s['J_corrected'])}, the best combination "
                f"(<em>{_esc(c['best_rule'])}</em>) scores "
                f"{_num(c['J_corrected'])} — a gain of "
                f"<strong>{_signed(c['gain_corrected'])}</strong>. "
                "The uncorrected gap is larger, and most of that difference is "
                "the advantage of having chosen the winner on these same "
                "patients.</p>"
            )

    model_html = ""
    if art.panel_model_vs_single is not None and not art.panel_model_vs_single.empty:
        model_html = (
            "<h4>Against the multivariable models</h4>"
            "<p><code>auc_shared_apparent</code> is each model re-scored on the "
            "same patients as the markers — the like-for-like column, and "
            "apparent, so it is optimistic. "
            "<code>auc_artifact_corrected</code> is the model's own "
            "optimism-corrected figure on its own patients; the gap between "
            "the two artifact columns bounds how optimistic the re-scored one "
            "is.</p>"
            + _table(art.panel_model_vs_single)
        )

    stability_html = ""
    if art.panel_imputation_stability is not None and \
            not art.panel_imputation_stability.empty:
        stability_html = details_block(
            "🎲 Does filling in the missing scans change this?",
            "<p>Every headline above is computed on patients whose markers were "
            "actually recorded. Re-running across the MICE draws asks whether "
            "the same answers come back. Reported as reproduction rates rather "
            "than pooled estimates: averaging works for an estimate, not for a "
            "choice, and 'which rule wins' is a choice.</p>"
            + _table(art.panel_imputation_stability),
        )

    rules_html = ""
    if art.panel_rule_reading_view is not None and \
            not art.panel_rule_reading_view.empty:
        rules_html = details_block(
            "📋 Every rule, ranked",
            "<p>Singles, AND/OR pairs and count rules on one patient set, "
            "ranked by Youden J (sensitivity + specificity − 1).</p>"
            + _table(art.panel_rule_reading_view)
            + _panel_figure(art, "rule_space"),
        )

    return (
        "<h3>Does a combination beat one sign?</h3>"
        + lead
        + _panel_figure(art, "count_score")
        + correction_html
        + model_html
        + rules_html
        + stability_html
    )


def render_marker_panel(cfg: ReportConfig, art: Artifacts) -> str:
    """🎯 The two study aims, answered on one cohort.

    Everything here is read from ``output/panel/``. The section is the last
    substantive one because it depends on every section above it: the markers
    come from the EDA screen, the models from multivariable modelling, and the
    cut-points baked into the derived flags from the threshold notebook.
    """
    if art.panel_marker is None and art.panel_count_score is None:
        return section_block(
            "🎯 Which MRI markers, and do they combine?",
            warning_box(
                "No marker panel was found under <code>output/panel/</code>. "
                "Run the marker panel cell in the modelling notebook."
            ),
        )
    return section_block(
        "🎯 Which MRI markers, and do they combine?",
        _panel_aim_one(art) + _panel_aim_two(art),
    )
```

`_lead`, `_table`, `_num`, `_pct`, `_int`, `_signed` are helpers in `threshold_report.py`, **not** in `report.py`. Check whether `report.py` has equivalents before using them: it has `table_to_html`, `_esc`, `_fmt_count`, `warning_box`, `info_box`, `details_block`, `section_block`. Replace each helper above with the `report.py` equivalent, and where none exists add a small private one next to `_panel_figure` — for example:

```python
def _lead(text: str) -> str:
    return f'<p class="lead">{text}</p>'


def _num(value: Any, digits: int = 2) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{v:.{digits}f}"


def _signed(value: Any, digits: int = 3) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{v:+.{digits}f}"


def _pct(value: Any) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{v * 100:.0f}%"


def _int(value: Any) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{int(round(v))}"


def _table(df: pd.DataFrame) -> str:
    return table_to_html(df)
```

Do not import from `threshold_report.py` — the modelling report must not depend on the threshold phase to build.

- [ ] **Step 6: Wire it into `build_report`**

In `build_report`, between `render_inferential(cfg, art),` and `render_appendix(cfg, art),`:

```python
        render_marker_panel(cfg, art),
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_report.py -v`
Expected: all pass, including the five new panel tests

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add heavy_machinery/modelling_phase/report.py heavy_machinery/pytests_atypier/test_report.py
git commit -m "Render the marker panel section after multivariable modelling"
```

---

### Task 12: The notebook cell

**Files:**
- Modify: `meningioma-modelling.ipynb` — new markdown + code cells forming §04.5, between §04 (multivariable modelling) and §05 (build report.html)

**Interfaces:**
- Consumes: `marker_panel.run_marker_panel`.
- Produces: `output/panel/` on every modelling run.

- [ ] **Step 1: Add the markdown cell**

Insert before the `## 05. Build report.html` cell:

```markdown
## 04.5 · Marker panel — the two study aims

🎯 Which MRI signs argue hardest for high grade, and whether a combination of
them beats any single one.

Both answers land in `output/panel/` and are rendered as the last section of
`report.html`. Two things are worth knowing before reading them:

- **`NON_IMAGING`** is the boundary of "a radiological feature". The EDA
  accuracy table also carries predictors that are not read off a scan, and this
  is where they are kept out. Edit it here, not in the module.
- Marker accuracy is computed on **observed** data, not the MICE draws. A
  sensitivity describes how a sign performs when a radiologist looks at the
  scan; imputing the sign would report the accuracy of a finding nobody saw.
  The draws are used as a stability check instead — "did the same rule win".
```

- [ ] **Step 2: Add the code cell**

```python
from pathlib import Path

from marker_panel import run_marker_panel
from missingness_resolution import load_imputed_frames
from model_calculator import load_model_artifact

# Not read off a scan. The accuracy table carries these too, and a section
# about MRI markers should not silently include them.
NON_IMAGING = {
    "sex_male", "hist_necrosis", "progesterone_pos", "multiple_meningiomas",
}

_artifact_dir = Path(OUTPUT_ROOT) / "inferential" / "model_artifacts"
_artifacts = {
    p.stem.replace("_model", "").replace(f"{TARGET}_", ""): load_model_artifact(p)
    for p in sorted(_artifact_dir.glob("*_model.json"))
} if _artifact_dir.exists() else {}

panel_tables = run_marker_panel(
    df_unimputed,
    target=TARGET,
    accuracy_table=diag_acc,
    output_root=OUTPUT_ROOT,
    exclude=NON_IMAGING,
    artifacts=_artifacts,
    draws=load_imputed_frames(OUTPUT_ROOT),
)

display(panel_tables["02_marker_panel_reading_view"])
display(panel_tables["09_selection_correction"])
```

- [ ] **Step 3: Check the variable names against the notebook**

`OUTPUT_ROOT`, `TARGET`, `diag_acc` and `df_unimputed` are used above. Open the notebook and confirm each exists with that exact name in an earlier cell — `diag_acc` is assigned in the §03 EDA cell, and the unimputed frame may be called something else. Fix the names in the cell to match; do not rename anything in the notebook.

- [ ] **Step 4: Run the notebook from §00 to §05**

Expected: `output/panel/tables/` holds ten CSVs, `output/panel/figures/` holds three SVGs, and the two displayed tables are populated.

- [ ] **Step 5: Open the report and read the new section**

Run: `open output/report/report.html`
Check by eye: the section is the last one before the appendix; the marker table is sorted with an informative sign first; no cell reads `nan` or `—` where a number should be; the corrected gain sentence quotes a plausible number.

- [ ] **Step 6: Commit**

```bash
git add meningioma-modelling.ipynb output/panel
git commit -m "Run the marker panel from the modelling notebook"
```

---

## Self-Review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| New section after multivariable modelling | 11 |
| Compute in `marker_panel.py`, render in `report.py` | 1–10, 11 |
| `BinaryMarker` adapter, `combinations.py` unchanged | 2 |
| Marker list read from accuracy table + exclude list | 2, 12 |
| LR+ with Katz interval, chance-overlap labelling | 1, 3 |
| `Catches` column as the specificity guard | 3 |
| LR+ forest figure | 4 |
| Shared cohort, audit table, unusable markers dropped with reason | 5 |
| Count score + figure | 6 |
| Rule menu, reading view, rule-space figure | 7 |
| Selection correction on **both** sides | 7 |
| Model re-scored on shared set, three labelled AUC columns | 8 |
| MICE stability as reproduction rates | 9 |
| All eleven tables and three figures written | 10 |
| Error handling: warning box, info box, missing artifacts | 3, 8, 10, 11 |
| Tests incl. the `CHANGES.md` regression and determinism | 1–11 |
| Out of scope: triples, refitting, EDA table changes | honoured throughout |
| *(not in the spec)* degenerate 2×2 crashes the rule search | 0 — found while writing the plan, verified on the real cohort |

Table `04_single_rules.csv` from the spec is **not** written: `05_rule_menu.csv` already contains every single-marker row (`kind == "single"`), so a separate file would be the same numbers twice with two chances to disagree. The spec's file list is superseded on this one point.

**Placeholder scan:** none — every step carries the code it needs.

**Type consistency:** `BinaryMarker` is used with the same `(col, label)` signature in Tasks 2, 3, 5–10. `selection_correction` returns a two-row frame whose row 0 is `"best single"` and row 1 is `"best combination"`; Tasks 8 and 9 both rely on that order and on the `gain_corrected` column. `run_marker_panel` returns `dict[str, pd.DataFrame]` keyed by filename stem without the `.csv`, which Task 12 indexes as `panel_tables["02_marker_panel_reading_view"]`.
