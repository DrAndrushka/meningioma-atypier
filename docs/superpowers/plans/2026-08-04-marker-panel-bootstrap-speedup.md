# Marker-panel bootstrap speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make §04.5 of `meningioma-modelling.ipynb` run in seconds instead of 102 minutes, producing **byte-identical** output.

**Architecture:** The selection-optimism bootstrap rebuilds the whole 650-rule menu with pandas + scipy on every resample — a Wilson interval, a χ² and an odds ratio per rule — and then reads one column of it, `youden_J`. Add a numpy scorer that computes only that column, run the bootstrap on it, and let both sides of the correction share one resample loop. `full_rule_menu` itself is untouched: it still builds the table that gets printed.

**Tech Stack:** Python 3.12, numpy, pandas (nullable `boolean` dtype), pytest.

---

## The acceptance test

**`output/panel/baseline_2026-08-04/` holds the 13 CSVs the pre-optimisation code produced** on the real cohort at full budget — the run that took 102 minutes. It is the definition of correct for this plan.

```bash
diff -r output/panel/baseline_2026-08-04 /tmp/panel-after/panel/tables
```

**No output = pass.** Anything else is a failure, including a difference in the fourth decimal place. Do not delete, regenerate, or edit that directory.

The two tables the bootstrap decides, which must come back exactly:

`09_selection_correction.csv`

| side | best_rule | J_apparent | optimism | J_corrected | winner_stability | n_bootstrap | gain_apparent | gain_corrected | correction_effect |
|---|---|---|---|---|---|---|---|---|---|
| best single | Max diameter (cm) ≥ 3.81 | 0.303 | 0.049 | 0.254 | 0.450 | 500 | 0.07 | 0.047 | narrows |
| best combination | Max diameter (cm) ≥ 3.81 OR ADC value ≤ 0.72 | 0.374 | 0.073 | 0.301 | 0.228 | 500 | 0.07 | 0.047 | narrows |

`11_imputation_stability.csv`

| item | value | note |
|---|---|---|
| Draws | 20.0 | 20 scorable |
| Top marker reproduced | 0.7 | most often: ADC value ≤ 0.72 |
| Winning rule reproduced | 0.7 | most often: Max diameter (cm) ≥ 3.81 OR ADC value ≤ 0.72 |
| Combination still beat the best single | 1.0 | share of draws with a positive corrected gain |

**There is no equivalent baseline for the threshold phase.** `output/thresholds/` is empty in this checkout, so `meningioma-thresholder.ipynb` has nothing to regress against. That notebook calls `combinations.bootstrap_best_rule` directly, so its guard is the test suite — Task 2 keeps that function's signature, its return keys and its exact numbers, and pins them with a characterization test.

---

## Measured baseline (measured on this repo, 2026-08-04)

| Quantity | Measured |
|---|---|
| Markers kept for the panel | 25 |
| Rules per menu (25 singles + 600 AND/OR pairs + 25 count rules) | 650 |
| `binary_diagnostic_metrics` — one rule | 830 µs |
| `full_rule_menu` — one menu | 614 ms |
| Shared-set correction (2 sides × 500 resamples = 1000 menus) | 9 min |
| Stability check (20 draws × 2 sides × 200 resamples = 8000 menus) | 72 min |
| **Observed wall clock for §04.5** | **102 min** |
| Everything in §04.5 that is *not* the bootstrap | **1.8 s** |

Where the 830 µs goes (cProfile): `scipy.stats.chi2_contingency` 22%, five `statsmodels` Wilson intervals 11%, and the rest pandas `Series` construction, `astype`, `concat`, `dropna` — about 17 million Python calls to produce 1950 rows. The bootstrap uses none of it.

**The replacement is already prototyped and checked** against `full_rule_menu` on four frames: the 8-patient fixture with missing values both ways, a 300-patient frame with 25/30/15% missing predictors *and* missing outcomes, the real 352-patient cohort with all 25 markers, and a degenerate frame with an all-missing marker and a never-true marker. Result on every one: identical labels, identical order, identical NaN pattern, `max |ΔJ| = 0.0`, at **1.54 ms** per menu instead of 614 ms.

Projected: bootstrap ~8 s, §04.5 total **~10 s**.

## Global Constraints

- **No number may change.** Byte-identical CSVs against the baseline. This is the deliverable, not a nice-to-have.
- **Seed stays `20260801`. Budgets stay `n_boot=500` and `draw_n_boot=200`.** Do not raise them and do not lower them. Raising `n_boot` would tighten the optimism estimate and is affordable once this lands, but it moves published numbers, so it is out of scope. Lowering them to make a runtime look good is a fabrication.
- **The rng must be consumed identically.** Today's loop calls `rng.integers(0, n, n)` exactly once per iteration, *including* iterations later skipped by the `< 5 events` guard. Draw first, test second. Get this wrong and every resample after the first skip differs.
- **`n` is `len(df)`, not the number of rows with a known outcome.** `bootstrap_best_rule` resamples the whole frame it is handed, missing outcomes included, and the metric functions drop them afterwards. The matrix must keep every row and apply the known-outcome mask *inside* the scorer.
- **Public signatures do not change.** `bootstrap_best_rule(df, cutpoints, target, *, criterion, n_boot, seed, max_size, kinds)` keeps its parameters and its six return keys: `optimism`, `n_bootstrap`, `best_rule`, `J_apparent`, `J_corrected`, `winner_stability`.
- **Only `criterion="youden_J"` gets the fast path.** Any other criterion falls back to the existing pandas loop.
- Run tests from the repo root: `python -m pytest` (paths come from `pytest.ini`).

## File Structure

- **Create `heavy_machinery/threshold_phase/rule_matrix.py`** — the numpy scorer. Turns a cohort plus cut-points into boolean matrices, knows the static rule labels in `full_rule_menu` order, and computes Youden J for a row selection. One responsibility: score the whole menu fast, on rows I name. Knows nothing about bootstrapping.
- **Create `heavy_machinery/pytests_atypier/test_rule_matrix.py`** — tests it against `full_rule_menu`, which is the definition of correct.
- **Modify `heavy_machinery/threshold_phase/combinations.py`** — `bootstrap_best_rule` reimplemented on the scorer; the old loop kept as a private fallback; new `bootstrap_best_rules` for several `kinds` subsets off one resample loop.
- **Modify `heavy_machinery/modelling_phase/marker_panel.py`** — `selection_correction` calls `bootstrap_best_rules` once instead of `bootstrap_best_rule` twice; the docstring quoting the old wall-clock times gets corrected.
- **Modify `heavy_machinery/pytests_atypier/test_combinations.py`** — characterization tests pinning today's exact bootstrap output, plus a speed guard.
- **Modify `CHANGES.md`** — record that nothing moved, which is the whole claim.
- **Create `scripts/check_panel_against_baseline.py`** — the acceptance test as a committed script, so it can be re-run after any future change to this code.

`heavy_machinery/threshold_phase` is on `pythonpath` (see `pytest.ini`), so modules there import each other by bare name (`import combinations as cb`). Follow that.

---

### Task 1: The numpy rule scorer

**Files:**
- Create: `heavy_machinery/threshold_phase/rule_matrix.py`
- Test: `heavy_machinery/pytests_atypier/test_rule_matrix.py`

**Interfaces:**
- Consumes: `combinations.flag_frame(df, cutpoints) -> pd.DataFrame`, `combinations.CutPoint` (only `.label`, `.short_label`, `.col`, `.flag` are touched), `combinations.LOGICS == ("AND", "OR")`.
- Produces, for Tasks 2 and 3:
  - `RuleMatrix` — a `NamedTuple` with fields `present: np.ndarray` (n×k bool), `observed: np.ndarray` (n×k bool), `positive: np.ndarray` (n, bool), `known: np.ndarray` (n, bool), `labels: list[str]`, `kinds: list[str]`, `k: int`, `n: int`.
  - `rule_labels(cutpoints, *, max_size=2) -> tuple[list[str], list[str]]`
  - `rule_matrix(df, cutpoints, target, *, max_size=2) -> RuleMatrix`
  - `youden_j(matrix, rows=None, *, max_size=2) -> np.ndarray` — one float per rule in `full_rule_menu` order, NaN where a rate is undefined. `rows` is an integer index array (a bootstrap resample); `None` means the cohort as given.

**Background — the three-valued logic.** A flag is `True`, `False`, or *missing*. `combinations.combine_flags` uses Kleene logic: `A AND B` is `True` only if both are `True`, and `False` if *either* is `False` (a `False` settles it even when the other is missing); anything else is missing. `A OR B` is the mirror. Representing each flag by two boolean arrays — `present` ("is `True`") and `absent` ("is observed and `False`") — makes each rule one line, and the rule is "observed" exactly where `present | absent`. Patients whose rule value is missing land in none of TP/FP/FN/TN, which is what dropping them does today.

**Background — the count rules.** `count_threshold_table` uses `complete_only=True`: a patient counts only if **every** flag was observed. So `valid = observed.all(axis=1)`, and "≥ c of k criteria" is `True` for valid patients with at least c present flags, `False` for the other valid patients, missing for everyone else.

**Background — the menu order.** `full_rule_menu` concatenates `single_rule_table` (one row per cut-point, in the given order), then `pair_rule_table` (for each `itertools.combinations` of column indices, an `AND` row then an `OR` row), then `count_threshold_table` (`≥ 1 of k` … `≥ k of k`). Labels are `cp.label`, `" AND ".join(...)` / `" OR ".join(...)`, `f"≥ {cut} of {k} criteria"`. Kinds are `"single"`, `"and"`, `"or"`, `"count"`. Reproduce this exactly — Tasks 2 and 3 match winners by position.

- [ ] **Step 1: Write the failing test**

Create `heavy_machinery/pytests_atypier/test_rule_matrix.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_rule_matrix.py -v`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'rule_matrix'`.

- [ ] **Step 3: Write the implementation**

Create `heavy_machinery/threshold_phase/rule_matrix.py`:

```python
"""The rule menu as boolean matrices — the same J, without the pandas.

:func:`combinations.full_rule_menu` is the right tool for a table a human will
read: it carries Wilson intervals, a χ², an odds ratio and a label for every
rule. The selection bootstrap reads none of that. It rebuilds the menu on each
of hundreds of resamples and looks at one column, ``youden_J``, which costs
about 830 µs per rule to arrive at and nothing at all to use.

This module computes that one column. A cohort becomes two n×k boolean
matrices — ``present`` ("this flag is True") and ``observed`` — and a resample
becomes an integer index into their rows. Every rule is then a couple of
boolean operations and four ``count_nonzero`` calls, in the same order and
under the same three-valued logic as the pandas version, which is what makes
the two agree to the last bit rather than to three decimals.

The labels are static — they depend on the cut-points, never on the data — so
the bootstrap matches winners by position and builds no strings at all.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

import combinations as cb


class RuleMatrix(NamedTuple):
    """A cohort's flags, ready to be resampled by row index.

    ``present`` and ``observed`` are the two halves of a three-valued flag: the
    value is True where ``present``, False where ``observed & ~present``, and
    missing where ``~observed``. ``known`` marks patients with a recorded
    outcome, ``positive`` the high-grade ones among them. Every row of the
    original frame is kept — including patients with no outcome — because the
    bootstrap resamples the frame it was handed, and dropping rows here would
    quietly shrink the resample.
    """

    present: np.ndarray      # (n, k) bool
    observed: np.ndarray     # (n, k) bool
    positive: np.ndarray     # (n,)   bool — outcome known and positive
    known: np.ndarray        # (n,)   bool — outcome recorded
    labels: list[str]
    kinds: list[str]
    k: int
    n: int


def rule_labels(cutpoints: Sequence, *, max_size: int = 2) -> tuple[list[str], list[str]]:
    """``(labels, kinds)`` in exactly ``full_rule_menu``'s row order.

    The AND row precedes the OR row for each combination, because
    ``pair_rule_table`` iterates ``combinations.LOGICS``. If that constant is
    ever reordered, :func:`youden_j` must be reordered with it.
    """
    k = len(cutpoints)
    labels = [cp.label for cp in cutpoints]
    kinds = ["single"] * k
    for size in range(2, int(max_size) + 1):
        for combo in itertools.combinations(range(k), size):
            for logic in cb.LOGICS:
                labels.append(f" {logic} ".join(cutpoints[i].label for i in combo))
                kinds.append(logic.lower())
    for cut in range(1, k + 1):
        labels.append(f"≥ {cut} of {k} criteria")
        kinds.append("count")
    return labels, kinds


def rule_matrix(
    df: pd.DataFrame,
    cutpoints: Sequence,
    target: str,
    *,
    max_size: int = 2,
) -> RuleMatrix:
    """Flags and outcome as boolean arrays, plus the static menu labels."""
    flags = cb.flag_frame(df, cutpoints)
    observed = flags.notna().to_numpy(dtype=bool)
    present = flags.fillna(False).to_numpy(dtype=bool) & observed

    y = df[target].astype("boolean")
    known = y.notna().to_numpy(dtype=bool)
    positive = y.fillna(False).to_numpy(dtype=bool) & known

    labels, kinds = rule_labels(cutpoints, max_size=max_size)
    return RuleMatrix(present, observed, positive, known,
                      labels, kinds, len(cutpoints), len(df))


def _youden(true_: np.ndarray, false_: np.ndarray,
            pos: np.ndarray, neg: np.ndarray) -> float:
    """``sensitivity + specificity - 1``, NaN where a rate has no denominator."""
    tp = np.count_nonzero(true_ & pos)
    fp = np.count_nonzero(true_ & neg)
    fn = np.count_nonzero(false_ & pos)
    tn = np.count_nonzero(false_ & neg)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens + spec - 1.0


def youden_j(
    matrix: RuleMatrix,
    rows: np.ndarray | None = None,
    *,
    max_size: int = 2,
) -> np.ndarray:
    """Youden J for every rule in the menu, in ``full_rule_menu`` order.

    ``rows`` is an integer index — a bootstrap resample — or ``None`` for the
    cohort as it stands. Patients with an unknown outcome ride along in the
    matrix but are excluded from every count here, exactly as the pandas
    version drops them.
    """
    present, observed = matrix.present, matrix.observed
    positive, known = matrix.positive, matrix.known
    if rows is not None:
        present, observed = present[rows], observed[rows]
        positive, known = positive[rows], known[rows]

    absent = observed & ~present          # observed and False
    pos = positive                        # known and high grade
    neg = known & ~positive               # known and not high grade

    k = matrix.k
    out = np.empty(len(matrix.labels), dtype=float)
    i = 0

    for j in range(k):
        out[i] = _youden(present[:, j], absent[:, j], pos, neg)
        i += 1

    for size in range(2, int(max_size) + 1):
        for combo in itertools.combinations(range(k), size):
            members = list(combo)
            # Kleene: AND is True only if all are True, False if any is False.
            out[i] = _youden(present[:, members].all(axis=1),
                             absent[:, members].any(axis=1), pos, neg)
            # OR is the mirror: True if any is True, False only if all are.
            out[i + 1] = _youden(present[:, members].any(axis=1),
                                 absent[:, members].all(axis=1), pos, neg)
            i += 2

    # Count rules score complete cases only: "2 of 4" means something else when
    # one of the four was never measured.
    valid = observed.all(axis=1)
    counts = present.sum(axis=1)
    for cut in range(1, k + 1):
        hit = valid & (counts >= cut)
        out[i] = _youden(hit, valid & ~hit, pos, neg)
        i += 1

    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_rule_matrix.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest`
Expected: PASS. A new module that nothing imports cannot break anything, so any failure here is a pre-existing regression — investigate it before continuing rather than assuming it is noise.

- [ ] **Step 6: Commit**

```bash
git add heavy_machinery/threshold_phase/rule_matrix.py heavy_machinery/pytests_atypier/test_rule_matrix.py
git commit -m "perf: add a numpy rule scorer that reproduces full_rule_menu's Youden J"
```

---

### Task 2: Run the bootstrap on the scorer

**Files:**
- Modify: `heavy_machinery/threshold_phase/combinations.py:392-472` (`bootstrap_best_rule`)
- Test: `heavy_machinery/pytests_atypier/test_combinations.py` (add tests; change nothing existing)

**Interfaces:**
- Consumes: `rule_matrix.rule_matrix`, `rule_matrix.youden_j` from Task 1.
- Produces: `bootstrap_best_rule` with an unchanged signature and unchanged return dict — Task 3 and `meningioma-thresholder.ipynb` both depend on it.

**Why the characterization test comes first.** The claim is "no number moves", and that is only checkable against numbers written down *before* the change. Step 1 pins today's output; if Step 5 changes it, the refactor is wrong. The values below were produced by running today's `bootstrap_best_rule` on the fixture already present in `test_combinations.py`.

- [ ] **Step 1: Write the characterization test (it must pass against today's code)**

Append to `heavy_machinery/pytests_atypier/test_combinations.py`:

```python
# --------------------------------------------------------------------------
# Characterization: these numbers came out of the pandas implementation on
# 2026-08-04, before the numpy scorer replaced it. They are not a claim about
# what is statistically right — they are a tripwire. If the scorer is not
# bit-for-bit equivalent, one of them moves.
# --------------------------------------------------------------------------
GOLDEN_BOOT = {
    None: {
        "optimism": 0.0046525807268937265,
        "n_bootstrap": 50,
        "best_rule": "≥ 2 of 3 criteria",
        "J_apparent": 0.6254960317460316,
        "J_corrected": 0.620843451019138,
        "winner_stability": 1.0,
    },
    ("single",): {
        "optimism": 0.038835126255743285,
        "n_bootstrap": 50,
        "best_rule": "Metric B ≥ 0.397",
        "J_apparent": 0.4618055555555556,
        "J_corrected": 0.4229704292998123,
        "winner_stability": 0.58,
    },
    ("and", "or", "count"): {
        "optimism": 0.0046525807268937265,
        "n_bootstrap": 50,
        "best_rule": "≥ 2 of 3 criteria",
        "J_apparent": 0.6254960317460316,
        "J_corrected": 0.620843451019138,
        "winner_stability": 1.0,
    },
}


@pytest.mark.parametrize("kinds", list(GOLDEN_BOOT))
def test_bootstrap_reproduces_the_recorded_numbers(kinds):
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    assert [cp.label for cp in cps] == [
        "Metric A ≥ 0.933", "Metric B ≥ 0.397", "Metric C ≤ -0.498",
    ]
    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=50, seed=20260801,
                                 kinds=kinds)
    want = GOLDEN_BOOT[kinds]
    assert out["best_rule"] == want["best_rule"]
    assert out["n_bootstrap"] == want["n_bootstrap"]
    for key in ("optimism", "J_apparent", "J_corrected", "winner_stability"):
        assert out[key] == pytest.approx(want[key], rel=0, abs=1e-12)


def test_bootstrap_matches_the_menu_on_a_frame_with_missing_outcomes():
    """n is len(df): rows with no outcome are resampled, then dropped by the score."""
    rng = np.random.default_rng(21)
    n = 300
    y = rng.binomial(1, 0.35, n).astype(bool)
    df = pd.DataFrame({
        "a": rng.normal(size=n) + y * 1.1,
        "b": rng.normal(size=n) + y * 0.9,
        TARGET: pd.array(y, dtype="boolean"),
    })
    df.loc[rng.random(n) < 0.2, "a"] = np.nan
    df.loc[rng.random(n) < 0.1, TARGET] = pd.NA
    cps = cb.cutpoints_for_rule(df.dropna(subset=[TARGET]), [A, B], TARGET, "youden")

    out = cb.bootstrap_best_rule(df, cps, TARGET, n_boot=30, seed=5)
    menu = cb.full_rule_menu(df, cps, TARGET)
    apparent = menu.loc[menu["youden_J"].idxmax()]
    assert out["best_rule"] == apparent["rule_label"]
    assert out["J_apparent"] == pytest.approx(float(apparent["youden_J"]))
```

- [ ] **Step 2: Run it against the current implementation**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_combinations.py -k "recorded_numbers or missing_outcomes" -v`
Expected: PASS, 4 tests. If a golden fails *here*, it was captured on a different environment — stop and re-capture rather than editing the expectations to match.

- [ ] **Step 3: Commit the tripwire before touching anything**

```bash
git add heavy_machinery/pytests_atypier/test_combinations.py
git commit -m "test: pin today's bootstrap_best_rule output before optimising it"
```

- [ ] **Step 4: Move the existing loop to a private fallback**

In `heavy_machinery/threshold_phase/combinations.py`, add the import beside the others at the top of the file:

```python
import rule_matrix as rmx
```

Then, directly above `bootstrap_best_rule`, add today's implementation under a private name. This is a move, not a rewrite — copy the existing body verbatim:

```python
def _bootstrap_best_rule_pandas(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    criterion: str,
    n_boot: int,
    seed: int,
    max_size: int,
    kinds: Sequence[str] | None,
) -> dict:
    """The original menu-rebuilding loop, for criteria the matrix cannot score.

    :mod:`rule_matrix` knows one column. Anything else — an accuracy, a PPV —
    needs the whole table on every resample. Kept rather than deleted so that
    asking for a different criterion still answers.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    gaps: list[float] = []
    winners: list[str] = []

    def _menu(frame: pd.DataFrame) -> pd.DataFrame:
        table = full_rule_menu(frame, cutpoints, target, max_size=max_size)
        if kinds is not None and not table.empty:
            table = table[table["kind"].isin(list(kinds))]
        return table

    apparent = _menu(df)
    if apparent.empty or apparent[criterion].isna().all():
        return {"optimism": np.nan, "n_bootstrap": 0,
                "best_rule": "", "J_apparent": np.nan, "J_corrected": np.nan,
                "winner_stability": np.nan}
    best_idx = apparent[criterion].idxmax()
    best_label = str(apparent.loc[best_idx, "rule_label"])
    j_apparent = float(apparent.loc[best_idx, criterion])

    by_label = apparent.set_index("rule_label")[criterion]
    for _ in range(int(n_boot)):
        take = rng.integers(0, n, n)
        boot = df.iloc[take].reset_index(drop=True)
        if boot[target].astype("boolean").sum() < 5:
            continue
        menu_b = _menu(boot)
        if menu_b.empty or menu_b[criterion].isna().all():
            continue
        i_b = menu_b[criterion].idxmax()
        label_b = str(menu_b.loc[i_b, "rule_label"])
        if label_b not in by_label.index:
            continue
        on_original = float(by_label.loc[label_b])
        if not np.isfinite(on_original):
            continue
        gaps.append(float(menu_b.loc[i_b, criterion]) - on_original)
        winners.append(label_b)

    optimism = float(np.mean(gaps)) if gaps else np.nan
    return {
        "optimism": optimism,
        "n_bootstrap": len(gaps),
        "best_rule": best_label,
        "J_apparent": j_apparent,
        "J_corrected": j_apparent - optimism if gaps else np.nan,
        "winner_stability": (
            float(np.mean([w == best_label for w in winners])) if winners else np.nan
        ),
    }
```

- [ ] **Step 5: Reimplement `bootstrap_best_rule` on the scorer**

Keep the whole existing docstring, and append this paragraph to it before the closing `"""`:

```
    The menu is scored by :mod:`rule_matrix` rather than rebuilt as a table:
    the loop reads only ``youden_J``, and paying for a Wilson interval and a χ²
    on six hundred rules per resample was costing about a hundred minutes a
    run. Any other ``criterion`` still goes the long way round, because only
    the Youden J has a matrix form here.
```

Replace the body — everything from `rng = np.random.default_rng(seed)` to the end of the function — with:

```python
    if criterion != "youden_J":
        return _bootstrap_best_rule_pandas(
            df, cutpoints, target, criterion=criterion, n_boot=n_boot,
            seed=seed, max_size=max_size, kinds=kinds,
        )

    rng = np.random.default_rng(seed)
    n = len(df)
    matrix = rmx.rule_matrix(df, cutpoints, target, max_size=max_size)
    wanted = (np.ones(len(matrix.labels), dtype=bool) if kinds is None
              else np.isin(matrix.kinds, list(kinds)))

    apparent = rmx.youden_j(matrix, max_size=max_size)
    side = np.where(wanted, apparent, np.nan)
    if not wanted.any() or np.isnan(side).all():
        return {"optimism": np.nan, "n_bootstrap": 0,
                "best_rule": "", "J_apparent": np.nan, "J_corrected": np.nan,
                "winner_stability": np.nan}

    best_idx = int(np.nanargmax(side))
    best_label = matrix.labels[best_idx]
    j_apparent = float(side[best_idx])

    gaps: list[float] = []
    winners: list[int] = []
    for _ in range(int(n_boot)):
        # Draw first, test second: the guard must not swallow a draw, or every
        # later resample shifts and the numbers move.
        take = rng.integers(0, n, n)
        if np.count_nonzero(matrix.positive[take]) < 5:
            continue
        boot = np.where(wanted, rmx.youden_j(matrix, take, max_size=max_size), np.nan)
        if np.isnan(boot).all():
            continue
        i_b = int(np.nanargmax(boot))
        on_original = apparent[i_b]
        if not np.isfinite(on_original):
            continue
        gaps.append(float(boot[i_b]) - float(on_original))
        winners.append(i_b)

    optimism = float(np.mean(gaps)) if gaps else np.nan
    return {
        "optimism": optimism,
        "n_bootstrap": len(gaps),
        "best_rule": best_label,
        "J_apparent": j_apparent,
        "J_corrected": j_apparent - optimism if gaps else np.nan,
        # How often the same rule wins. Below ~0.5 the "best combination" is a
        # coin toss between near-equivalent rules, not a finding.
        "winner_stability": (
            float(np.mean([w == best_idx for w in winners])) if winners else np.nan
        ),
    }
```

- [ ] **Step 6: Run the combinations tests**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_combinations.py -v`
Expected: PASS — the four goldens from Step 1 and the pre-existing `test_kinds_restricts_the_menu_the_winner_is_chosen_from`, `test_unknown_kind_degrades_rather_than_raising`, `test_best_rule_is_reproducible`.

If a golden moves, diagnose in this order: (a) the `< 5 events` guard runs before `rng.integers` and is swallowing draws; (b) `rule_matrix` dropped unknown-outcome rows, so `n` is too small; (c) a tie resolves differently, which means the label order does not match — re-run Task 1's `test_menu_shape_is_singles_then_pairs_then_counts`.

- [ ] **Step 7: Check the real table early, while the change is small**

`09_selection_correction.csv` depends only on `bootstrap_best_rule`, so it can be reproduced now — in about a second — rather than waiting for Task 4. Run from the repo root:

```bash
python -c "
import sys; sys.path[:0]=['.','heavy_machinery','heavy_machinery/cleaning_phase','heavy_machinery/modelling_phase','heavy_machinery/threshold_phase']
import pandas as pd, time
from pathlib import Path
from dataset_handoff import load_modelling_handoff
import marker_panel as mp
root=Path('output'); target='high_grade'
df,_,_=load_modelling_handoff(root)
acc=pd.read_csv(root/'eda'/'tables'/'diagnostic_accuracy.csv')
mk=[m for m in mp.markers_from_diagnostic_accuracy(acc,target=target,exclude={'sex_male','hist_necrosis','progesterone_pos'}) if m.col in df.columns]
kept,_=mp.usable_markers(df,mk,target)
shared=mp.shared_cohort_frame(df,kept,target)
t=time.perf_counter()
got=mp.selection_correction(shared,kept,target)
print(f'{time.perf_counter()-t:.1f}s')
want=pd.read_csv('output/panel/baseline_2026-08-04/09_selection_correction.csv')
pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False, rtol=0, atol=5e-4)
print('MATCHES BASELINE')
"
```

Expected: a time of roughly **1–2 seconds** and the line `MATCHES BASELINE`. The tolerance is `5e-4` only because the baseline CSV is written rounded to three decimals; the underlying values must be exact.

- [ ] **Step 8: Add a speed guard**

Append to `heavy_machinery/pytests_atypier/test_combinations.py`:

```python
def test_bootstrap_is_fast_enough_to_run_in_a_notebook():
    """Twelve cut-points, 300 resamples. The pandas loop took about a minute."""
    import time

    rng = np.random.default_rng(3)
    n = 300
    y = rng.binomial(1, 0.3, n).astype(bool)
    cols = {f"m{i}": rng.normal(size=n) + y * 0.6 for i in range(12)}
    df = pd.DataFrame({**cols, TARGET: pd.array(y, dtype="boolean")})
    metrics = [Metric(f"m{i}", f"Metric {i}", "u", "higher") for i in range(12)]
    cps = cb.cutpoints_for_rule(df, metrics, TARGET, "youden")

    start = time.perf_counter()
    cb.bootstrap_best_rule(df, cps, TARGET, n_boot=300, seed=1)
    assert time.perf_counter() - start < 10.0
```

- [ ] **Step 9: Run the whole suite**

Run: `python -m pytest`
Expected: PASS. Watch `test_marker_panel.py` and `test_threshold_stability.py` — they reach `bootstrap_best_rule` through two layers.

- [ ] **Step 10: Commit**

```bash
git add heavy_machinery/threshold_phase/combinations.py heavy_machinery/pytests_atypier/test_combinations.py
git commit -m "perf: score the selection bootstrap with the numpy menu instead of pandas"
```

---

### Task 3: One resample loop for both sides of the correction

**Files:**
- Modify: `heavy_machinery/threshold_phase/combinations.py` (add `bootstrap_best_rules` after `bootstrap_best_rule`)
- Modify: `heavy_machinery/modelling_phase/marker_panel.py:484-533` (`selection_correction`)
- Test: `heavy_machinery/pytests_atypier/test_combinations.py`

**Interfaces:**
- Consumes: `rule_matrix.rule_matrix`, `rule_matrix.youden_j`.
- Produces: `combinations.bootstrap_best_rules(df, cutpoints, target, *, sides, n_boot=500, seed=20260801, max_size=2) -> dict[str, dict]` — `sides` maps a caller's name to a `kinds` tuple; each value is the same dict `bootstrap_best_rule` returns.

**Why this is exactly equivalent, not an approximation.** `selection_correction` calls `bootstrap_best_rule` twice with the **same seed**. `np.random.default_rng(seed)` with the same `n` produces the identical sequence of `take` arrays both times, and the `< 5 events` guard depends only on the resample, not on the side. The two runs already visit the same resamples — the second merely rebuilds the same 650-rule menu to look at a different part of it. Scoring once and taking two arg-maxes gives the same answer for half the work.

The one thing that is **not** side-independent is the `np.isnan(boot).all()` skip: a resample can leave every count rule undefined while the singles are fine. Each side therefore keeps its own `gaps` and `winners` and applies its own skip.

- [ ] **Step 1: Write the failing test**

Append to `heavy_machinery/pytests_atypier/test_combinations.py`:

```python
def test_bootstrap_best_rules_equals_running_each_side_separately():
    """One shared resample loop, two answers, identical to two separate runs."""
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B, C], TARGET, "youden")
    kw = dict(n_boot=60, seed=20260801)

    together = cb.bootstrap_best_rules(
        df, cps, TARGET,
        sides={"single": ("single",), "combo": ("and", "or", "count")}, **kw)
    apart = {
        "single": cb.bootstrap_best_rule(df, cps, TARGET, kinds=("single",), **kw),
        "combo": cb.bootstrap_best_rule(df, cps, TARGET,
                                        kinds=("and", "or", "count"), **kw),
    }

    assert set(together) == set(apart)
    for name in apart:
        assert together[name]["best_rule"] == apart[name]["best_rule"]
        assert together[name]["n_bootstrap"] == apart[name]["n_bootstrap"]
        for key in ("optimism", "J_apparent", "J_corrected", "winner_stability"):
            assert together[name][key] == pytest.approx(apart[name][key],
                                                        rel=0, abs=1e-12)


def test_bootstrap_best_rules_degrades_on_an_unknown_kind():
    df = two_signal_frame()
    cps = cb.cutpoints_for_rule(df, [A, B], TARGET, "youden")
    out = cb.bootstrap_best_rules(df, cps, TARGET,
                                  sides={"nope": ("nope",)}, n_boot=5, seed=1)
    assert out["nope"]["best_rule"] == ""
    assert np.isnan(out["nope"]["optimism"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_combinations.py -k best_rules -v`
Expected: FAIL with `AttributeError: module 'combinations' has no attribute 'bootstrap_best_rules'`.

- [ ] **Step 3: Write the implementation**

Add to `heavy_machinery/threshold_phase/combinations.py`, immediately after `bootstrap_best_rule`:

```python
def bootstrap_best_rules(
    df: pd.DataFrame,
    cutpoints: Sequence[CutPoint],
    target: str,
    *,
    sides: dict[str, Sequence[str] | None],
    n_boot: int = 500,
    seed: int = 20260801,
    max_size: int = 2,
) -> dict[str, dict]:
    """Several restricted selections off one set of resamples.

    Correcting "best single" and "best combination" means running the same
    bootstrap twice on the same seed and taking the arg-max over a different
    slice of the same menu each time — the resamples were always identical.
    Scoring each resample once and taking one arg-max per side is the same
    arithmetic for half the work, and the halving matters: this runs once per
    MICE draw.

    ``sides`` maps a name to a ``kinds`` filter, ``None`` meaning the whole
    menu. Each value of the result has the same keys as
    :func:`bootstrap_best_rule`. Sides keep separate tallies, because a
    resample can leave every count rule undefined while the singles are fine,
    and the two are not obliged to agree on ``n_bootstrap``.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    matrix = rmx.rule_matrix(df, cutpoints, target, max_size=max_size)
    apparent = rmx.youden_j(matrix, max_size=max_size)

    masks, best, gaps, winners = {}, {}, {}, {}
    for name, kinds in sides.items():
        mask = (np.ones(len(matrix.labels), dtype=bool) if kinds is None
                else np.isin(matrix.kinds, list(kinds)))
        masks[name] = mask
        side = np.where(mask, apparent, np.nan)
        best[name] = (None if (not mask.any() or np.isnan(side).all())
                      else int(np.nanargmax(side)))
        gaps[name], winners[name] = [], []

    live = [name for name, idx in best.items() if idx is not None]
    if live:
        for _ in range(int(n_boot)):
            take = rng.integers(0, n, n)
            if np.count_nonzero(matrix.positive[take]) < 5:
                continue
            scored = rmx.youden_j(matrix, take, max_size=max_size)
            for name in live:
                boot = np.where(masks[name], scored, np.nan)
                if np.isnan(boot).all():
                    continue
                i_b = int(np.nanargmax(boot))
                on_original = apparent[i_b]
                if not np.isfinite(on_original):
                    continue
                gaps[name].append(float(boot[i_b]) - float(on_original))
                winners[name].append(i_b)

    out: dict[str, dict] = {}
    for name in sides:
        idx = best[name]
        if idx is None:
            out[name] = {"optimism": np.nan, "n_bootstrap": 0, "best_rule": "",
                         "J_apparent": np.nan, "J_corrected": np.nan,
                         "winner_stability": np.nan}
            continue
        g, w = gaps[name], winners[name]
        optimism = float(np.mean(g)) if g else np.nan
        j_apparent = float(apparent[idx])
        out[name] = {
            "optimism": optimism,
            "n_bootstrap": len(g),
            "best_rule": matrix.labels[idx],
            "J_apparent": j_apparent,
            "J_corrected": j_apparent - optimism if g else np.nan,
            "winner_stability": (float(np.mean([x == idx for x in w])) if w
                                 else np.nan),
        }
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_combinations.py -k best_rules -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Point `selection_correction` at it**

In `heavy_machinery/modelling_phase/marker_panel.py`, replace the `sides = [...]` list and the `for label, kinds in sides:` block with:

```python
    sides = {
        "best single": ("single",),
        "best combination": ("and", "or", "count"),
    }
    results = cb.bootstrap_best_rules(
        df, markers, target, sides=sides, n_boot=n_boot, seed=seed,
        max_size=max_size,
    )
    rows = [
        {
            "side": label,
            "best_rule": results[label].get("best_rule", ""),
            "J_apparent": results[label].get("J_apparent", np.nan),
            "optimism": results[label].get("optimism", np.nan),
            "J_corrected": results[label].get("J_corrected", np.nan),
            "winner_stability": results[label].get("winner_stability", np.nan),
            "n_bootstrap": results[label].get("n_bootstrap", 0),
        }
        for label in sides
    ]
```

`sides` is a plain dict and dicts keep insertion order, so `rows[0]` is still "best single" and `rows[1]` still "best combination" — which matters, because the lines just below index them by position (`out.loc[1, "J_corrected"] - out.loc[0, "J_corrected"]`). Leave those lines alone.

Append to `selection_correction`'s docstring, before the closing `"""`:

```
    Both sides come out of one resample loop. They always used the same seed,
    so they were already visiting the same resamples; the second run was only
    re-scoring the same menu to look at a different part of it.
```

- [ ] **Step 6: Re-run the early baseline check from Task 2 Step 7**

The command is unchanged. Expected: `MATCHES BASELINE`, in about half the time it took after Task 2 — this task halves the work, and must not move the answer.

- [ ] **Step 7: Run the whole suite**

Run: `python -m pytest`
Expected: PASS, including `test_selection_correction_is_deterministic_for_a_seed` and the tests reading `gain_corrected` and `correction_effect`.

- [ ] **Step 8: Commit**

```bash
git add heavy_machinery/threshold_phase/combinations.py heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_combinations.py
git commit -m "perf: run both sides of the selection correction off one resample loop"
```

---

### Task 4: Reproduce the whole panel and prove nothing moved

**Files:**
- Create: `scripts/check_panel_against_baseline.py`
- Modify: `heavy_machinery/modelling_phase/marker_panel.py` (the `run_marker_panel` docstring's timing paragraph)
- Modify: `CHANGES.md`

**Interfaces:** none — this task runs the pipeline and writes prose.

**Why this is a committed script, not a one-off command.** The baseline is worth something only if re-checking against it is easy. Anyone who touches `combinations.py` or `marker_panel.py` later should be able to run one command and find out whether they moved a published number.

- [ ] **Step 1: Write the checker**

Create `scripts/check_panel_against_baseline.py`:

```python
"""Re-run the marker panel and prove it still matches the recorded baseline.

``output/panel/baseline_2026-08-04/`` holds the thirteen CSVs the pre-numpy
code produced on the real cohort at full budget — the run that took 102
minutes. Any change to :mod:`combinations`, :mod:`rule_matrix` or
:mod:`marker_panel` that moves one of those numbers is a change to a published
result, and this script is how you find that out in under a minute.

    python scripts/check_panel_against_baseline.py

Exits 0 and prints the wall time if every table matches, 1 otherwise.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), *(str(REPO / p) for p in (
    "heavy_machinery", "heavy_machinery/cleaning_phase",
    "heavy_machinery/modelling_phase", "heavy_machinery/threshold_phase"))]

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import pandas as pd                    # noqa: E402

from dataset_handoff import load_modelling_handoff        # noqa: E402
from missingness_resolution import load_imputed_frames    # noqa: E402
from marker_panel import run_marker_panel                 # noqa: E402
from model_calculator import load_model_artifact          # noqa: E402

TARGET = "high_grade"
NON_IMAGING = {"sex_male", "hist_necrosis", "progesterone_pos"}
BASELINE = REPO / "output" / "panel" / "baseline_2026-08-04"


def main() -> int:
    root = REPO / "output"
    if not BASELINE.is_dir():
        print(f"❌ no baseline at {BASELINE}")
        return 1

    df, _, _ = load_modelling_handoff(root)
    accuracy = pd.read_csv(root / "eda" / "tables" / "diagnostic_accuracy.csv")

    # Loaded exactly as the notebook's §04.5 cell does. Without them, tables 10
    # and 13 come out empty and the comparison proves nothing.
    art_dir = root / "inferential" / "model_artifacts"
    artifacts = {
        p.stem.replace("_model", "").replace(f"{TARGET}_", ""): load_model_artifact(p)
        for p in sorted(art_dir.glob("*_model.json"))
    } if art_dir.is_dir() else {}
    print(f"{len(artifacts)} model artifacts loaded")

    scratch = Path(tempfile.mkdtemp(prefix="panel-check-"))
    try:
        start = time.perf_counter()
        run_marker_panel(
            df, target=TARGET, accuracy_table=accuracy, output_root=scratch,
            exclude=NON_IMAGING, artifacts=artifacts,
            draws=load_imputed_frames(root),
        )
        elapsed = time.perf_counter() - start
        fresh = scratch / "panel" / "tables"

        failures = []
        expected = sorted(p.name for p in BASELINE.glob("*.csv"))
        for name in expected:
            new = fresh / name
            if not new.is_file():
                failures.append(f"{name}: not produced")
                continue
            if new.read_bytes() != (BASELINE / name).read_bytes():
                failures.append(f"{name}: differs from baseline")

        print(f"\n{len(expected) - len(failures)}/{len(expected)} tables identical")
        print(f"run_marker_panel: {elapsed:.1f}s")
        if failures:
            print("\n❌ " + "\n❌ ".join(failures))
            print(f"\nfresh tables kept for inspection: {fresh}")
            return 1
        print("✅ every table matches the baseline")
        return 0
    finally:
        if not sys.exc_info()[0]:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it — this is the acceptance test**

```bash
python scripts/check_panel_against_baseline.py
```

Expected:

```
7 model artifacts loaded

13/13 tables identical
run_marker_panel: <N>s
✅ every table matches the baseline
```

with `<N>` **under 60 seconds** — the bootstrap projects to ~8 s and everything else in the panel measures 1.8 s. Record the number; Steps 4 and 5 quote it.

If tables differ, work through the diagnosis list in Task 2 Step 6. If `10_model_vs_single.csv` or `13_model_reading_view.csv` are the only ones differing, the artifacts did not load — check the count on the first line.

If the time lands above five minutes, do not touch the budgets — they are fixed by the Global Constraints. Profile `run_marker_panel` with `cProfile` and find the real cause; the bootstrap is no longer a candidate.

- [ ] **Step 3: Refresh the live output directory**

Only once Step 2 prints ✅. The panel tables in `output/panel/tables/` are the 102-minute run's; overwriting them with an identical set is harmless and keeps timestamps honest, but the *baseline* copy must be left alone.

```bash
python scripts/check_panel_against_baseline.py && echo "verified — safe to re-run §04.5 in the notebook"
```

Re-running the notebook cell is now the simplest way to refresh `output/panel/`, and it takes seconds.

- [ ] **Step 4: Correct the timing paragraph that is now wrong**

In `heavy_machinery/modelling_phase/marker_panel.py`, `run_marker_panel`'s docstring says 500 resamples "costs about four minutes" and that forwarding it to the draws "turns a four-minute correction into an eighty-five-minute one". Both are historical. Replace that paragraph with:

```
    Two bootstrap budgets on purpose too. ``n_boot`` buys the two corrections
    on the shared set, run once each; ``draw_n_boot`` buys the one inside
    :func:`imputation_stability`, which runs **per MICE draw** — twenty times
    over. They stayed separate parameters after the menu scoring moved to
    :mod:`rule_matrix`, because they still buy different things: the shared-set
    correction is a number the report prints, and the per-draw one only has to
    settle a reproduction rate.
```

- [ ] **Step 5: Record the change**

Add to `CHANGES.md`, immediately after the opening heading block and before the first `---`. Replace `<N>` with the number from Step 2:

```markdown
## 2026-08-04 — §04.5 marker panel: same numbers, <N> seconds instead of 102 minutes

**No number moved.** The seed (`20260801`), both bootstrap budgets (500 on the
shared set, 200 per draw) and all 20 MICE draws are unchanged. This is a speed
change only.

The selection-optimism bootstrap was rebuilding the entire 650-rule menu on
every resample — a Wilson interval, a χ² and an odds ratio for each rule — and
then reading one column of it, `youden_J`. That column now comes from
`heavy_machinery/threshold_phase/rule_matrix.py`, which scores the same menu in
the same order out of two boolean matrices, and the two sides of the correction
share one resample loop instead of independently drawing the identical
resamples.

Checked, not assumed. The pre-change full-budget run is preserved in
`output/panel/baseline_2026-08-04/`, and `scripts/check_panel_against_baseline.py`
reproduces all thirteen CSVs byte-for-byte. `test_combinations.py` additionally
pins the pre-change bootstrap output to twelve decimal places.

| What | Was | Now |
|---|---|---|
| One 650-rule menu | 614 ms | 1.5 ms |
| Shared-set correction (2 × 500 resamples) | 9.0 min | ~1 s |
| Stability check (20 draws × 2 × 200) | 72 min | ~8 s |
| §04.5 end to end | 102 min | **<N> s** |

---
```

- [ ] **Step 6: Run the whole suite one more time**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_panel_against_baseline.py heavy_machinery/modelling_phase/marker_panel.py CHANGES.md
git commit -m "docs: verify the panel against its baseline and record the speedup"
```

---

## Self-Review

**Spec coverage.** The request: make §04.5 faster to execute, with the statistics as precise as possible or unchanged. Cause measured and named (650 rules × 830 µs of discarded statistics × 9000 resample-menus; observed 102 min). Tasks 1–3 remove it without touching a seed or a budget. Task 4's acceptance test is a byte-for-byte comparison against the real pre-change output, which is the strongest available form of "the statistics did not change" — stronger than the tolerance-based check the tests use. The runtime target is checked in Task 4 Step 2, with an explicit instruction not to meet it by lowering budgets.

**Feedback speed.** The full acceptance test needs Tasks 1–3 done, so Task 2 Step 7 adds an early one-second check of `09_selection_correction.csv` against the baseline — drift is caught at the task that causes it, not at the end.

**Placeholder scan.** Every code step carries its code. The only deliberate blank is `<N>` in Task 4 Steps 4–5, a measurement that cannot exist before Step 2 runs; the step producing it says so. The goldens in Task 2 and the acceptance tables above are measured on this repo, not invented.

**Type consistency.** `rule_matrix` exports `RuleMatrix`, `rule_labels`, `rule_matrix`, `youden_j`; Tasks 2 and 3 use those names and the fields `present`/`observed`/`positive`/`known`/`labels`/`kinds`/`k`/`n`. `bootstrap_best_rule` keeps all six return keys on both the fast and fallback paths. `bootstrap_best_rules` returns those same six per side, which is what `selection_correction`'s row-builder reads.

**Known residual risk.** `np.nanargmax` and `pandas.Series.idxmax` both return the first maximum, so ties resolve identically — but only if the label order matches, which Task 1 tests directly and Task 2's goldens re-check end to end. If `combinations.LOGICS` is ever reordered, `rule_labels` and `youden_j` must change together; both say so in the file.
