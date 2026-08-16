# Literature Model Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit seven published high-grade-meningioma models in our cohort and measure each against the single predictors it is built from, with optimism-corrected paired ΔAUC.

**Architecture:** Two new modules — `variable_selection.py` (rank candidates by discrimination, apply two guards, emit an audit trail) and `model_comparison.py` (fit the 12 single-predictor models, difference AUCs within bootstrap resamples, pool nested tests with D2). `model_validation.bootstrap_internal_validation` gains two options: return its per-resample AUC vector, and accept a `select` callable so a data-selected model's *selection* is re-run inside each resample. Everything else is config and wiring.

**Tech Stack:** Python 3.12, pandas, numpy, statsmodels, scikit-learn, pytest. R `mice` 3.19 upstream (unchanged).

**Spec:** `docs/superpowers/specs/2026-08-16-literature-model-refit-design.md`

## Global Constraints

- Bootstrap: **B = 1000** resamples, master seed **20260801**. `analysis.BOOTSTRAP_RESAMPLES` stays 1000; the cut-point phase is not touched.
- MICE is **unchanged**: m=20, maxit=20, seed=42.
- Collinearity threshold: **ρ = 0.8** (absolute Spearman).
- All 22 models fit on `imputed_frames[0]`, which has no missing values — every model sees all **352** patients, which is what makes paired differencing valid.
- Continuous predictors stay continuous. `log1p` applies only to columns already in `scales.LOG1P_COLUMNS`. **Never** import a foreign cohort's cut-point.
- Published numbers are transcribed verbatim or left empty. Never fill a cell from a related figure.
- Run from repo root. Tests: `python3 -m pytest -q`.
- Every new derived column that exists only to serve one model carries `eda_in_derived=None`.

---

### Task 1: `age_ge75` derived column

**Files:**
- Modify: `meningioma-cleaning.ipynb` (the `DERIVATIONS = [` cell)
- Test: `heavy_machinery/pytests_atypier/test_derivations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: column `age_ge75` (pandas `boolean`) in the cleaned cohort and every MICE draw. Task 9 uses it as a `lin_2014` predictor.

- [ ] **Step 1: Write the failing test**

Add to `heavy_machinery/pytests_atypier/test_derivations.py`:

```python
def test_age_ge75_splits_both_grade_groups_above_the_floor(real_cohort):
    """Lin 2014 dichotomises age at 75. The source spec says fall back to
    continuous age if either grade group has <20 patients in the >=75 stratum.
    It does not: 46 grade-1 and 23 grade-2/3. This test re-raises the question
    if the cohort ever changes."""
    df = real_cohort
    assert "age_ge75" in df.columns
    hg = df["high_grade"].astype("boolean").fillna(False)
    ge75 = df["age_ge75"].astype("boolean").fillna(False)
    assert int((ge75 & ~hg).sum()) >= 20
    assert int((ge75 & hg).sum()) >= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_derivations.py::test_age_ge75_splits_both_grade_groups_above_the_floor -q`
Expected: FAIL — `assert "age_ge75" in df.columns`

- [ ] **Step 3: Add the derivation**

In `meningioma-cleaning.ipynb`, insert into `DERIVATIONS` immediately before the `#🟧… Hide-parent binaries` banner:

```python
    _derivations.Apply(
        name="age_ge75",
        source="age",
        fn=lambda s: s.astype("Float64") >= 75,
        kind="binary",
        active=True,
        hide_parent=False,
        rule="age >= 75",
        # Model-only, exactly like necrosis_or_hemorrhage. It exists to fit one
        # published model; screening it next to continuous age would test the
        # same signal twice and spend FDR budget on the weaker version.
        eda_in_derived=None,
        reason=(
            "Lin BJ, Chou KN, Kao HW et al. Correlation between magnetic resonance "
            "imaging grading and pathological grading in meningioma. J Neurosurg "
            "2014;121(5):1201-1208 — age >= 75 carries 2 of the 12 score points."
        ),
        dda_in_derived=True,
    ),
```

Edit the notebook JSON directly (do not execute it):

```bash
python3 - <<'PY'
import json
p='meningioma-cleaning.ipynb'
nb=json.load(open(p,encoding='utf-8'))
cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'DERIVATIONS = [' in ''.join(c['source']))
src=''.join(cell['source'])
anchor="\n    #\U0001f7e7"*1
i=src.index("    #\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7\U0001f7e7 Hide-parent binaries")
new = open('/tmp/age_ge75_block.py').read()   # the block above, verbatim
src = src[:i] + new + "\n" + src[i:]
cell['source']=src.splitlines(keepends=True)
json.dump(nb, open(p,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
open(p,'a',encoding='utf-8').write('\n')
PY
```

- [ ] **Step 4: Re-run the cleaning notebook so the column reaches the MICE draws**

```bash
ATYPIER_FIGURES=report jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 --output-dir /tmp --output cleaning.out.ipynb \
  meningioma-cleaning.ipynb
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_derivations.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meningioma-cleaning.ipynb heavy_machinery/pytests_atypier/test_derivations.py
git commit -m "feat: age_ge75 derived column for the Lin 2014 refit"
```

---

### Task 2: Master bootstrap seed and per-resample AUC vector

**Files:**
- Modify: `heavy_machinery/modelling_phase/model_validation.py:216-290`
- Test: `heavy_machinery/pytests_atypier/test_model_validation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `bootstrap_internal_validation(model_df, target, design_cols, coefficients, *, n_bootstrap=1000, return_resample_aucs=False, select=None)`. When `return_resample_aucs=True` the returned dict gains `"resample_aucs": list[float]` — the *original-cohort* AUC of each resample's refitted model, in resample order. Tasks 3, 6 and 8 depend on this.
- Module constant `BOOTSTRAP_SEED: int = 20260801`.

- [ ] **Step 1: Write the failing test**

```python
def test_resample_aucs_are_returned_and_paired_across_models():
    """Two models validated on the same frame must use the same resample
    indices, or their AUC difference is not a paired difference."""
    import model_validation as mv
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "y": rng.binomial(1, 0.3, n).astype(float),
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    df["y"] = (df["a"] * 0.9 + rng.normal(size=n) > 0).astype(float)
    out_a = mv.bootstrap_internal_validation(
        df, "y", ["a"], {"const": 0.0, "a": 1.0},
        n_bootstrap=50, return_resample_aucs=True)
    out_b = mv.bootstrap_internal_validation(
        df, "y", ["a", "b"], {"const": 0.0, "a": 1.0, "b": 0.0},
        n_bootstrap=50, return_resample_aucs=True)
    assert len(out_a["resample_aucs"]) == len(out_b["resample_aucs"]) == 50
    # Same seed -> same index sets -> a rerun reproduces exactly.
    again = mv.bootstrap_internal_validation(
        df, "y", ["a"], {"const": 0.0, "a": 1.0},
        n_bootstrap=50, return_resample_aucs=True)
    assert out_a["resample_aucs"] == again["resample_aucs"]


def test_bootstrap_seed_is_the_pipeline_seed():
    import model_validation as mv
    assert mv.BOOTSTRAP_SEED == 20260801
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_validation.py -k "resample_aucs or pipeline_seed" -q`
Expected: FAIL — `TypeError: unexpected keyword argument 'return_resample_aucs'`

- [ ] **Step 3: Implement**

In `model_validation.py`, add near `MAX_VALIDATION_WORKERS`:

```python
# One master seed for every bootstrap in the modelling phase, matching the
# cut-point phase's ``wobble.SEED``. Resample i draws from a child stream, so
# two models validated in the same run see identical index sets and their AUC
# difference is a paired difference.
BOOTSTRAP_SEED: int = 20260801


def _resample_indices(n_rows: int, n_bootstrap: int) -> np.ndarray:
    """``n_bootstrap`` x ``n_rows`` index matrix, fixed by ``BOOTSTRAP_SEED``."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    return rng.choice(n_rows, size=(n_bootstrap, n_rows), replace=True)
```

Change the signature and loop of `bootstrap_internal_validation`:

```python
def bootstrap_internal_validation(
    model_df: pd.DataFrame,
    target: str,
    design_cols: list[str],
    coefficients: dict[str, float],
    *,
    n_bootstrap: int = analysis.BOOTSTRAP_RESAMPLES,
    return_resample_aucs: bool = False,
    select=None,
) -> dict[str, Any]:
```

Replace the `idx = np.random.RandomState(i).choice(...)` line with a precomputed matrix before the loop:

```python
    idx_matrix = _resample_indices(n_rows, n_bootstrap)
    resample_aucs: list[float] = []
```

and inside the loop:

```python
    for i in range(n_bootstrap):
        idx = idx_matrix[i]
        y_boot = y_arr[idx]
        X_boot = X_orig[idx]
```

After `pred_orig` is computed, record it:

```python
            resample_aucs.append(_round_metric(_auc(y_arr, pred_orig), 6))
```

and in the returned dict, before `return`:

```python
    result = { ... existing keys ... }
    if return_resample_aucs:
        result["resample_aucs"] = resample_aucs
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_validation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_validation.py heavy_machinery/pytests_atypier/test_model_validation.py
git commit -m "feat: master bootstrap seed and per-resample AUC vector"
```

---

### Task 3: Paired ΔAUC with percentile CI

**Files:**
- Create: `heavy_machinery/modelling_phase/model_comparison.py`
- Test: `heavy_machinery/pytests_atypier/test_model_comparison.py`

**Interfaces:**
- Consumes: `resample_aucs` from Task 2.
- Produces: `paired_delta_auc(aucs_combined, aucs_single, *, alpha=0.05) -> dict` with keys `delta`, `ci_lo`, `ci_hi`, `n_resamples`. Task 11 writes these into `model_vs_single_auc.csv`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for model_comparison.py — paired delta-AUC and D2 pooling."""
from __future__ import annotations

import numpy as np
import pytest

import model_comparison as mc


def test_paired_delta_auc_is_the_mean_of_within_resample_differences():
    a = [0.80, 0.82, 0.78, 0.81]
    b = [0.70, 0.75, 0.71, 0.72]
    out = mc.paired_delta_auc(a, b)
    assert out["delta"] == pytest.approx(np.mean(np.array(a) - np.array(b)))
    assert out["n_resamples"] == 4
    assert out["ci_lo"] <= out["delta"] <= out["ci_hi"]


def test_paired_delta_auc_ci_excludes_zero_when_one_model_always_wins():
    rng = np.random.RandomState(0)
    b = rng.uniform(0.60, 0.70, 500)
    a = b + 0.08
    out = mc.paired_delta_auc(a, b)
    assert out["ci_lo"] > 0


def test_paired_delta_auc_rejects_unpaired_vectors():
    with pytest.raises(ValueError, match="same number of resamples"):
        mc.paired_delta_auc([0.7, 0.8], [0.7])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_comparison'`

- [ ] **Step 3: Implement**

Create `heavy_machinery/modelling_phase/model_comparison.py`:

```python
"""Combined-versus-single model comparison for the multivariable phase.

Zhang 2020 and Peng 2021 are the only published meningioma-grading papers that
report a combined model against each of its own single predictors. This module
reproduces that comparison on our cohort: every literature model against each
predictor it is built from, on the optimism-corrected scale.

Differences are paired. Two models validated in the same run see the same
bootstrap index sets (``model_validation.BOOTSTRAP_SEED``), so their AUCs can be
differenced within each resample. Comparing two independent CIs by eye instead
overstates how different two models are, because a patient who is easy to
classify is easy for both.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_delta_auc(
    aucs_combined: Sequence[float],
    aucs_single: Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Within-resample AUC difference, with a percentile CI.

    Both vectors must come from the same run, so element *i* of each is the
    same resample. The CI is the percentiles of the difference distribution,
    not the difference of two separate CIs.
    """
    a = np.asarray(aucs_combined, dtype=float)
    b = np.asarray(aucs_single, dtype=float)
    if a.size != b.size:
        raise ValueError(
            "paired_delta_auc needs the same number of resamples for both "
            f"models; got {a.size} and {b.size}."
        )
    if a.size == 0:
        raise ValueError("paired_delta_auc needs at least one resample.")
    diff = a - b
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(np.mean(diff)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_resamples": int(a.size),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_comparison.py heavy_machinery/pytests_atypier/test_model_comparison.py
git commit -m "feat: paired optimism-corrected delta-AUC with percentile CI"
```

---

### Task 4: D2 pooling for the nested test

**Files:**
- Modify: `heavy_machinery/modelling_phase/model_comparison.py`
- Test: `heavy_machinery/pytests_atypier/test_model_comparison.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `d2_pool(chi2_stats, k) -> dict` with keys `statistic`, `df_num`, `df_den`, `p`. Task 11 calls it per (model, single) pair.

- [ ] **Step 1: Write the failing test**

```python
def test_d2_pool_returns_a_p_value_between_zero_and_one():
    out = mc.d2_pool([8.0, 9.0, 7.5, 8.5], k=2)
    assert 0.0 <= out["p"] <= 1.0
    assert out["df_num"] == 2


def test_d2_pool_is_significant_for_large_consistent_chi_squares():
    out = mc.d2_pool([30.0] * 20, k=1)
    assert out["p"] < 0.001


def test_d2_pool_is_not_significant_for_small_chi_squares():
    out = mc.d2_pool([0.1] * 20, k=1)
    assert out["p"] > 0.5


def test_d2_pool_rejects_an_empty_set():
    with pytest.raises(ValueError, match="at least one"):
        mc.d2_pool([], k=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -k d2 -q`
Expected: FAIL — `AttributeError: module 'model_comparison' has no attribute 'd2_pool'`

- [ ] **Step 3: Implement**

Append to `model_comparison.py`:

```python
def d2_pool(chi2_stats: Sequence[float], k: int) -> dict[str, Any]:
    """Pool ``m`` chi-square statistics across imputations — Rubin's D2.

    Li KH, Meng XL, Raghunathan TE, Rubin DB. Significance levels from repeated
    p-values with multiply-imputed data. Statistica Sinica 1991;1:65-92.

    A likelihood-ratio test run separately on each of the 20 MICE draws gives 20
    chi-square statistics that cannot simply be averaged: doing so ignores the
    between-draw variance and reports a test that is too confident. D2 combines
    them with a variance correction. It is the cheap, standard alternative to
    Meng-Rubin D3, which needs the per-draw likelihoods rather than the
    statistics alone.

    ``k`` is the number of parameters being tested — for a combined model
    against one of its own predictors, the count of the extra terms.
    """
    from scipy.stats import f as f_dist

    d = np.asarray(chi2_stats, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("d2_pool needs at least one chi-square statistic.")
    if k < 1:
        raise ValueError("d2_pool needs k >= 1 parameters under test.")
    m = d.size
    d_bar = float(np.mean(d))
    # Relative increase in variance due to nonresponse, on the sqrt scale the
    # D2 derivation works on.
    r = (1.0 + 1.0 / m) * float(np.var(np.sqrt(d), ddof=1)) if m > 1 else 0.0
    stat = (d_bar / k - (m + 1.0) / (m - 1.0) * r) / (1.0 + r) if m > 1 else d_bar / k
    stat = max(stat, 0.0)
    df_den = k ** (-3.0 / m) * (m - 1) * (1.0 + 1.0 / r) ** 2 if (m > 1 and r > 0) else 1e6
    p = float(f_dist.sf(stat, k, df_den))
    return {
        "statistic": float(stat),
        "df_num": int(k),
        "df_den": float(df_den),
        "p": p,
        "m": int(m),
        "method": "D2 (Li, Meng, Raghunathan & Rubin 1991)",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_comparison.py heavy_machinery/pytests_atypier/test_model_comparison.py
git commit -m "feat: D2 pooling for nested tests across MICE draws"
```

---

### Task 5: Variable selection with both guards and an audit trail

**Files:**
- Create: `heavy_machinery/modelling_phase/variable_selection.py`
- Create: `heavy_machinery/pytests_atypier/test_variable_selection.py`
- Modify: `heavy_machinery/config/analysis.py`

**Interfaces:**
- Consumes: `scales.is_log_scaled`.
- Produces:
  - `CUTPOINT_PARENT: dict[str, str]` in `analysis.py` — derived cut-point column → its continuous parent.
  - `discrimination(auc) -> float` — `max(auc, 1 - auc)`.
  - `rank_candidates(df, y, candidates) -> list[tuple[str, float]]` — `(column, raw_auc)` sorted by discrimination, descending.
  - `select_variables(df, y, candidates, *, k, rho_max=0.8) -> tuple[list[str], list[dict]]` — the picked columns and the audit rows. Audit row keys: `variable`, `auc`, `discrimination`, `kept`, `reason`. Tasks 6, 8 and 11 use both.

- [ ] **Step 1: Write the failing test**

Create `heavy_machinery/pytests_atypier/test_variable_selection.py`:

```python
"""Tests for variable_selection.py — AUC ranking and the two guards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import variable_selection as vs


@pytest.fixture
def toy():
    rng = np.random.RandomState(7)
    n = 300
    y = rng.binomial(1, 0.4, n)
    size = y * 1.2 + rng.normal(size=n)
    return pd.DataFrame({
        "size": size,
        "size_copy": size + rng.normal(scale=0.05, size=n),   # rho ~ 0.99
        "size_ge0": (size >= 0).astype(float),                 # cut-point child
        "edema": y * 0.6 + rng.normal(size=n),
        "protective": -(y * 0.9) + rng.normal(size=n),
        "noise": rng.normal(size=n),
    }), y


def test_discrimination_flips_a_protective_auc():
    assert vs.discrimination(0.37) == pytest.approx(0.63)
    assert vs.discrimination(0.68) == pytest.approx(0.68)


def test_protective_variable_is_ranked_on_discrimination_not_raw_auc(toy):
    df, y = toy
    ranked = vs.rank_candidates(df, y, ["protective", "noise"])
    assert ranked[0][0] == "protective"
    assert ranked[0][1] < 0.5          # raw AUC really is below 0.5


def test_collinear_candidate_is_dropped_and_the_next_one_taken(toy):
    df, y = toy
    picked, audit = vs.select_variables(
        df, y, ["size", "size_copy", "edema", "noise"], k=2, rho_max=0.8)
    assert picked == ["size", "edema"]
    dropped = {r["variable"]: r["reason"] for r in audit if not r["kept"]}
    assert "size_copy" in dropped
    assert "rho=" in dropped["size_copy"] and "size" in dropped["size_copy"]


def test_cutpoint_child_is_dropped_when_its_parent_is_a_candidate(toy):
    df, y = toy
    picked, audit = vs.select_variables(
        df, y, ["size", "size_ge0", "edema"], k=3, rho_max=0.99,
        cutpoint_parent={"size_ge0": "size"})
    assert "size_ge0" not in picked
    reason = next(r["reason"] for r in audit if r["variable"] == "size_ge0")
    assert "cut-point of size" in reason


def test_audit_records_every_candidate_considered(toy):
    df, y = toy
    picked, audit = vs.select_variables(df, y, list(df.columns), k=2, rho_max=0.8)
    assert len(picked) == 2
    assert {r["variable"] for r in audit} <= set(df.columns)
    assert all(set(r) == {"variable", "auc", "discrimination", "kept", "reason"}
               for r in audit)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_variable_selection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'variable_selection'`

- [ ] **Step 3: Implement**

Create `heavy_machinery/modelling_phase/variable_selection.py`:

```python
"""Pick the k best predictors by discrimination, with two guards.

``top_1_variable`` and ``top_6_variables`` are chosen from the same cohort they
are then fitted on. That is defensible only if the rule is written down, applied
mechanically, and auditable — which is what this module is.

Ranking is by *discrimination*, ``max(auc, 1 - auc)``, not raw AUC. A protective
predictor scores below 0.5 by construction: ADC is 0.370 in this cohort, which
is 0.630 the other way round and the second-strongest variable available.
Ranking on raw AUC would silently discard every protective finding.

Two guards, in order:

1. **Parent/child.** A derived cut-point is skipped when its continuous parent is
   also a candidate. The parent carries more information, and the child stacks
   the cut-point's own optimism on top of the model's.
2. **Collinearity.** A candidate correlated above ``rho_max`` with something
   already picked is skipped, and the next candidate that clears is taken. Six
   variables chosen on discrimination alone are four tumour-size measurements in
   different costumes.

Every skip is recorded with its reason. A selection nobody can see is a
selection nobody can check.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scales import is_log_scaled


def discrimination(auc: float) -> float:
    """How well a variable separates, regardless of direction."""
    return float(max(auc, 1.0 - auc))


def _column_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    x = df[col].astype(float).to_numpy()
    return np.log1p(np.clip(x, 0.0, None)) if is_log_scaled(col) else x


def rank_candidates(
    df: pd.DataFrame,
    y: Sequence[int],
    candidates: Sequence[str],
) -> list[tuple[str, float]]:
    """``(column, raw_auc)`` sorted by discrimination, best first."""
    y_arr = np.asarray(y, dtype=int)
    scored: list[tuple[str, float]] = []
    for col in candidates:
        if col not in df.columns:
            continue
        x = _column_vector(df, col)
        if np.unique(x).size < 2 or np.unique(y_arr).size < 2:
            continue
        scored.append((col, float(roc_auc_score(y_arr, x))))
    return sorted(scored, key=lambda t: -discrimination(t[1]))


def select_variables(
    df: pd.DataFrame,
    y: Sequence[int],
    candidates: Sequence[str],
    *,
    k: int,
    rho_max: float = 0.8,
    cutpoint_parent: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick ``k`` variables by discrimination, applying both guards.

    Returns the picked columns and one audit row per candidate considered.
    Candidates after the k-th pick are not examined and do not appear.
    """
    cutpoint_parent = cutpoint_parent or {}
    ranked = rank_candidates(df, y, candidates)
    candidate_set = {c for c, _ in ranked}
    vectors = {c: _column_vector(df, c) for c, _ in ranked}

    picked: list[str] = []
    audit: list[dict[str, Any]] = []
    for col, auc in ranked:
        if len(picked) == k:
            break
        parent = cutpoint_parent.get(col)
        if parent is not None and parent in candidate_set:
            audit.append({"variable": col, "auc": auc,
                          "discrimination": discrimination(auc),
                          "kept": False, "reason": f"cut-point of {parent}"})
            continue
        clash = ""
        for p in picked:
            rho = abs(float(pd.Series(vectors[col]).corr(
                pd.Series(vectors[p]), method="spearman")))
            if rho > rho_max:
                clash = f"rho={rho:.2f} with {p}"
                break
        if clash:
            audit.append({"variable": col, "auc": auc,
                          "discrimination": discrimination(auc),
                          "kept": False, "reason": clash})
            continue
        picked.append(col)
        audit.append({"variable": col, "auc": auc,
                      "discrimination": discrimination(auc),
                      "kept": True, "reason": ""})
    return picked, audit
```

Add to `heavy_machinery/config/analysis.py`, below `BOOTSTRAP_RESAMPLES`:

```python
# Derived cut-point column -> the continuous column it dichotomises. Selection
# skips the child whenever the parent is also a candidate: the parent carries
# more information, and the child would stack the cut-point search's own
# optimism on top of the model's.
CUTPOINT_PARENT: dict[str, str] = {
    "adc_value_le0.72": "adc_value",
    "edema_index_ge0.0617": "edema_index",
    "edema_volume_ge4.76": "edema_volume_cm3",
    "max_diameter_cm_ge3.81": "max_diameter_cm",
    "tumor_volume_ge15.1": "tumor_volume",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_variable_selection.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/variable_selection.py \
        heavy_machinery/pytests_atypier/test_variable_selection.py \
        heavy_machinery/config/analysis.py
git commit -m "feat: discrimination-ranked variable selection with parent and collinearity guards"
```

---

### Task 6: Re-run selection inside each bootstrap resample

**Files:**
- Modify: `heavy_machinery/modelling_phase/model_validation.py`
- Test: `heavy_machinery/pytests_atypier/test_model_validation.py`

**Interfaces:**
- Consumes: `select_variables` (Task 5), `_resample_indices` (Task 2).
- Produces: `bootstrap_internal_validation(..., select=callable)`. `select` takes `(model_df, y_array)` and returns `list[str]` of design columns. When supplied, each resample re-selects, refits on its own columns, and scores against the original cohort. The result dict gains `"selection_counts": dict[str, int]` — how many resamples chose each variable.

- [ ] **Step 1: Write the failing test**

```python
def test_select_hook_reselects_per_resample_and_counts_choices():
    """A model whose predictors are chosen from the data must have that choosing
    re-run inside the bootstrap, or its optimism correction misses the part that
    matters most."""
    import model_validation as mv
    rng = np.random.RandomState(3)
    n = 250
    y = rng.binomial(1, 0.4, n)
    df = pd.DataFrame({
        "y": y.astype(float),
        "strong": y * 1.3 + rng.normal(size=n),
        "weak": y * 0.15 + rng.normal(size=n),
        "noise": rng.normal(size=n),
    })

    def select(frame, y_arr):
        import variable_selection as vs
        picked, _ = vs.select_variables(
            frame, y_arr, ["strong", "weak", "noise"], k=1, rho_max=0.8)
        return picked

    out = mv.bootstrap_internal_validation(
        df, "y", ["strong"], {"const": 0.0, "strong": 1.0},
        n_bootstrap=40, return_resample_aucs=True, select=select)
    counts = out["selection_counts"]
    assert sum(counts.values()) == 40
    assert counts["strong"] > counts.get("noise", 0)
    assert len(out["resample_aucs"]) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_validation.py -k select_hook -q`
Expected: FAIL — `KeyError: 'selection_counts'`

- [ ] **Step 3: Implement**

In `bootstrap_internal_validation`, before the loop:

```python
    selection_counts: dict[str, int] = {}
```

Inside the loop, replace the fixed `X_boot = X_orig[idx]` block with:

```python
        idx = idx_matrix[i]
        y_boot = y_arr[idx]
        if select is None:
            X_boot, X_score = X_orig[idx], X_orig
        else:
            # Re-run the *choosing* on this resample, not just the fitting.
            boot_frame = model_df.iloc[idx].reset_index(drop=True)
            cols_i = select(boot_frame, y_boot)
            if not cols_i:
                continue
            for c in cols_i:
                selection_counts[c] = selection_counts.get(c, 0) + 1
            X_boot = np.column_stack(
                [np.ones(n_rows), boot_frame[cols_i].astype(float).to_numpy()])
            X_score = np.column_stack(
                [np.ones(n_rows), model_df[cols_i].astype(float).to_numpy()])
```

and change the two `predict` calls in the `try` block to use `X_score` for the original-cohort prediction:

```python
            pred_boot = np.asarray(boot_result.predict(X_boot), dtype=float)
            pred_orig = np.asarray(boot_result.predict(X_score), dtype=float)
```

Add to the result dict:

```python
    if select is not None:
        result["selection_counts"] = selection_counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_validation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_validation.py heavy_machinery/pytests_atypier/test_model_validation.py
git commit -m "feat: re-run variable selection inside each bootstrap resample"
```

---

### Task 7: Reference-variable declaration and assertion

**Files:**
- Modify: `heavy_machinery/config/analysis.py`
- Modify: `heavy_machinery/modelling_phase/variable_selection.py`
- Test: `heavy_machinery/pytests_atypier/test_variable_selection.py`

**Interfaces:**
- Consumes: `rank_candidates` (Task 5).
- Produces: `analysis.REFERENCE_VARIABLE: str`, `analysis.REFERENCE_VARIABLE_DISCRIMINATION: float`, and `variable_selection.assert_reference(picked) -> None`, which raises `ValueError` when the first **kept** variable is not the declared one. It checks the post-guard pick, not the raw ranking: a cut-point child can top the raw ranking and still be dropped by guard 1, and the reference must be a variable we would actually fit.

- [ ] **Step 1: Write the failing test**

```python
def test_assert_reference_passes_when_the_declared_variable_is_the_top_pick():
    vs.assert_reference(["tumor_volume", "adc_value", "edema_volume_cm3"])


def test_assert_reference_raises_when_something_else_is_picked_first():
    with pytest.raises(ValueError, match="reference variable"):
        vs.assert_reference(["max_diameter_cm", "tumor_volume"])


def test_assert_reference_raises_on_an_empty_pick():
    with pytest.raises(ValueError, match="reference variable"):
        vs.assert_reference([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_variable_selection.py -k reference -q`
Expected: FAIL — `AttributeError: module 'variable_selection' has no attribute 'assert_reference'`

- [ ] **Step 3: Implement**

Add to `analysis.py`:

```python
# The single predictor every delta-AUC is measured against. Declared rather than
# taken silently from whatever wins, because max_diameter_cm scores 0.675 to
# tumor_volume's 0.679 and the two correlate at ~0.92: a trivial data change
# would otherwise flip the denominator under the manuscript with nothing in the
# output saying so. The run recomputes the ranking and raises on disagreement.
REFERENCE_VARIABLE: str = "tumor_volume"
REFERENCE_VARIABLE_DISCRIMINATION: float = 0.679
```

Add to `variable_selection.py`:

```python
def assert_reference(picked: Sequence[str]) -> None:
    """Raise unless the declared reference variable is the first kept pick.

    Checks the list *after* both guards, not the raw ranking. A derived
    cut-point can top the raw ranking and still be dropped by guard 1, and the
    reference has to be a variable the pipeline would actually fit.
    """
    from heavy_machinery.config import load

    declared = load("analysis").REFERENCE_VARIABLE
    if not picked:
        raise ValueError(
            "No variable survived selection, so there is no reference variable."
        )
    if picked[0] != declared:
        raise ValueError(
            f"The declared reference variable {declared!r} is no longer the top "
            f"pick: {picked[0]!r} now leads. Every delta-AUC in the report is "
            f"measured against the reference, so update "
            f"analysis.REFERENCE_VARIABLE deliberately rather than letting the "
            f"denominator move on its own."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_variable_selection.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/config/analysis.py heavy_machinery/modelling_phase/variable_selection.py heavy_machinery/pytests_atypier/test_variable_selection.py
git commit -m "feat: declare the delta-AUC reference variable and assert it still wins"
```

---

### Task 8: Fit and validate the 12 single-predictor models

**Files:**
- Modify: `heavy_machinery/modelling_phase/model_comparison.py`
- Test: `heavy_machinery/pytests_atypier/test_model_comparison.py`

**Interfaces:**
- Consumes: `build_complete_case_frame`, `bootstrap_internal_validation` (Task 2).
- Produces: `fit_single_predictors(cohort_df, schema, predictors, target, *, n_bootstrap) -> dict[str, dict]` — keyed by predictor, each value `{"auc_apparent": float, "auc_corrected": float, "n": int, "events": int, "resample_aucs": list[float]}`. Task 11 consumes it.

- [ ] **Step 1: Write the failing test**

```python
def test_fit_single_predictors_returns_one_entry_per_predictor(tiny_schema):
    import numpy as np, pandas as pd
    from schema_infer import ColSpec
    rng = np.random.RandomState(11)
    n = 200
    y = rng.binomial(1, 0.35, n)
    df = pd.DataFrame({
        "event": y.astype(bool),
        "a": y * 1.1 + rng.normal(size=n),
        "b": rng.normal(size=n),
    })
    schema = {"event": ColSpec("event", "binary"),
              "a": ColSpec("a", "continuous"), "b": ColSpec("b", "continuous")}
    out = mc.fit_single_predictors(df, schema, ["a", "b"], "event", n_bootstrap=30)
    assert set(out) == {"a", "b"}
    assert out["a"]["auc_corrected"] > out["b"]["auc_corrected"]
    assert len(out["a"]["resample_aucs"]) == 30
    assert out["a"]["n"] == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -k single_predictors -q`
Expected: FAIL — `AttributeError: module 'model_comparison' has no attribute 'fit_single_predictors'`

- [ ] **Step 3: Implement**

Append to `model_comparison.py`:

```python
def fit_single_predictors(
    cohort_df,
    schema,
    predictors: Sequence[str],
    target: str,
    *,
    n_bootstrap: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit and bootstrap-validate each predictor on its own.

    These exist only to supply the yardstick a combined model is measured
    against, so they emit AUC and nothing else — no forest plot, no VIF table,
    no calculator artifact, no report fold of their own.
    """
    import statsmodels.api as sm
    from heavy_machinery.config import load
    from model_validation import (
        build_complete_case_frame,
        bootstrap_internal_validation,
    )

    if n_bootstrap is None:
        n_bootstrap = load("analysis").BOOTSTRAP_RESAMPLES

    out: dict[str, dict[str, Any]] = {}
    for pred in predictors:
        if pred not in cohort_df.columns:
            continue
        try:
            model_df, design_cols = build_complete_case_frame(
                cohort_df, schema, [pred], target)
        except (ValueError, RuntimeError):
            continue
        y = model_df[target].astype(int).to_numpy()
        X = sm.add_constant(model_df[design_cols].astype(float), has_constant="add")
        fit = sm.Logit(y, X).fit(disp=False)
        coefs = {"const": float(fit.params.iloc[0])}
        coefs.update({c: float(fit.params[c]) for c in design_cols})
        val = bootstrap_internal_validation(
            model_df, target, design_cols, coefs,
            n_bootstrap=n_bootstrap, return_resample_aucs=True)
        auc_row = next(m for m in val["metrics"] if m["metric"] == "AUC")
        out[pred] = {
            "auc_apparent": float(auc_row["apparent"]),
            "auc_corrected": float(auc_row["optimism_corrected"]),
            "n": int(len(model_df)),
            "events": int(y.sum()),
            "resample_aucs": val["resample_aucs"],
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_comparison.py heavy_machinery/pytests_atypier/test_model_comparison.py
git commit -m "feat: lightweight single-predictor fits for the combined-vs-single comparison"
```

---

### Task 9: The six new literature model variants

**Files:**
- Modify: `meningioma-modelling.ipynb` (the `LITERATURE_MODEL_VARIANTS = [` cell)
- Modify: `heavy_machinery/config/published_models.py`
- Test: `heavy_machinery/pytests_atypier/test_published_models.py` (create)

**Interfaces:**
- Consumes: `age_ge75` (Task 1).
- Produces: seven entries in `LITERATURE_MODEL_VARIANTS` with IDs `radeesri_2023`, `spille_2020`, `zhang_2020`, `funari_2023`, `kawahara_2012`, `lin_2014`, `peng_2021`, each with a matching `PUBLISHED_MODELS` key.

- [ ] **Step 1: Write the failing test**

Create `heavy_machinery/pytests_atypier/test_published_models.py`:

```python
"""Tests for config/published_models.py — transcription integrity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from heavy_machinery.config import load

pm = load("published_models")

EXPECTED = {"radeesri_2023", "spille_2020", "zhang_2020", "funari_2023",
            "kawahara_2012", "lin_2014", "peng_2021"}


def test_every_literature_model_has_a_published_record():
    assert EXPECTED <= set(pm.PUBLISHED_MODELS)


def test_surrogate_note_is_set_exactly_on_the_interface_substitutions():
    with_note = {k for k, v in pm.PUBLISHED_MODELS.items() if v.get("surrogate_note")}
    assert with_note == {"kawahara_2012", "lin_2014", "peng_2021"}


def test_kawahara_has_no_invented_odds_ratios():
    """No open source carries its ORs. Empty is correct; a number is a bug."""
    for term in pm.PUBLISHED_MODELS["kawahara_2012"]["terms"]:
        assert term.get("or") in (None, "")
        assert term.get("ci_lo") in (None, "")


def test_zhang_carries_beta_not_odds_ratios():
    for term in pm.PUBLISHED_MODELS["zhang_2020"]["terms"]:
        assert term.get("beta") is not None
        assert term.get("or") in (None, "")


def test_not_fitted_records_every_excluded_model_with_a_reason():
    assert set(pm.NOT_FITTED) == {
        "azeemuddin_2018", "yao_2022", "amano_2022",
        "duarte_gomes_quintas_neves_2026", "hale_2018",
    }
    assert all(v.strip() for v in pm.NOT_FITTED.values())


def test_every_mapped_column_exists_in_the_cohort():
    df_cols = set(__import__("pandas").read_parquet(
        Path("output/datasets/unimputed_df.parquet")).columns)
    for mid in EXPECTED:
        for term in pm.PUBLISHED_MODELS[mid]["terms"]:
            col = term.get("column")
            if col:
                assert col in df_cols, f"{mid}: {col} missing from the cohort"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_published_models.py -q`
Expected: FAIL — `AssertionError` on the first test (only `radeesri_lekhavat_2023` exists)

- [ ] **Step 3: Rename Radeesri and add the six**

In `meningioma-modelling.ipynb`, replace the `LITERATURE_MODEL_VARIANTS` list body with:

```python
    # Radeesri K, Lekhavat V. Asian Pacific J Cancer Prev 2023;24(3):819-825.
    ("radeesri_2023",
     "Radeesri & Lekhavat 2023 | necrosis / hyperostosis / edema MRI model",
     "https://journal.waocp.org/article_90552.html",
     "high_grade",
     ["necrosis_or_hemorrhage", "hyperostosis", "perifocal_edema"]),

    # Spille DC, Adeli A, Sporns PB et al. Neurosurg Rev 2021;44(2):1109-1117.
    ("spille_2020",
     "Spille et al. 2020 | edema volume + enhancement pattern",
     "https://pubmed.ncbi.nlm.nih.gov/32306190/",
     "high_grade",
     ["edema_volume_cm3", "heterogeneous_enhancement"]),

    # Zhang S, Chiang GC, Knapp JM et al. J Neuroradiol 2020;47(4):272-277.
    # Morphological arm only — the SWI/QSM/ADC quantitative model is not refit.
    ("zhang_2020",
     "Zhang et al. 2020 | morphological MRI model",
     "https://pubmed.ncbi.nlm.nih.gov/31541639/",
     "high_grade",
     ["calcification", "perifocal_edema", "irregular_tumor_margin",
      "skull_base_location"]),

    # Funari A, De la Garza Ramos R, Cezayirli P et al. Neuroradiology 2023;65(3):453-462.
    # tumor_volume stays continuous — their 36.0 cc cut-point is not imported.
    ("funari_2023",
     "Funari et al. 2023 | imaging score components",
     "https://pubmed.ncbi.nlm.nih.gov/36242642/",
     "high_grade",
     ["tumor_volume", "irregular_tumor_margin", "perifocal_edema"]),

    # Kawahara Y, Nakada M, Hayashi Y et al. J Neurooncol 2012;108(1):147-152.
    ("kawahara_2012",
     "Kawahara et al. 2012 | interface + enhancement heterogeneity",
     "https://pubmed.ncbi.nlm.nih.gov/22392126/",
     "high_grade",
     ["irregular_tumor_margin", "heterogeneous_enhancement"]),

    # Lin BJ, Chou KN, Kao HW et al. J Neurosurg 2014;121(5):1201-1208.
    ("lin_2014",
     "Lin et al. 2014 | MRI grading scale components",
     "https://pubmed.ncbi.nlm.nih.gov/25148007/",
     "high_grade",
     ["age_ge75", "irregular_tumor_margin", "capsular_enhancement",
      "heterogeneous_enhancement"]),

    # Peng S, Cheng Z, Guo Z. Transl Cancer Res 2021;10(9):4057-4064.
    ("peng_2021",
     "Peng, Cheng & Guo 2021 | nomogram predictors",
     "https://tcr.amegroups.org/article/view/55552/html",
     "high_grade",
     ["irregular_tumor_margin", "cortical_destruction", "skull_base_location"]),
```

- [ ] **Step 4: Add the six published records**

In `heavy_machinery/config/published_models.py`, rename the `radeesri_lekhavat_2023` key to `radeesri_2023` and append the six. Each follows the existing shape; the three interface substitutions add `surrogate_note`. Kawahara's terms carry no `or`/`ci_lo`/`ci_hi`; Zhang's carry `beta` instead. Then add:

```python
# Published models deliberately NOT refit, and why. These reasons otherwise live
# only in a commit message, and this file is where someone about to re-add one
# would look.
NOT_FITTED: dict[str, str] = {
    "azeemuddin_2018": (
        "No multivariable model exists in the paper — it is an ADC "
        "mean-comparison study. Previously mis-attributed in this pipeline."
    ),
    "yao_2022": (
        "Needs a 3-level shape (regular / lobulated / irregular) and oedema "
        "categorised by maximum diameter in mm. We hold binary shape and volume "
        "only, and the published effect concentrates in the >40 mm stratum. "
        "Cite in Discussion."
    ),
    "amano_2022": (
        "The final model requires symptomatic presentation, which this cohort "
        "does not record. Oedema volume is not a substitute: it is already in "
        "the model as an imaging term and is causally downstream of symptoms. "
        "Cite in Discussion."
    ),
    "duarte_gomes_quintas_neves_2026": (
        "Requires normalised ADC (ratio to contralateral white matter) and "
        "midline shift in mm. We hold absolute ADC and binary mass effect."
    ),
    "hale_2018": (
        "The final multivariable model collapses to a single predictor "
        "(peritumoral edema), so it is a single-predictor benchmark rather than "
        "a multivariable refit."
    ),
}
```

Kawahara's `surrogate_note` must be the strong one:

```python
        "surrogate_note": (
            "Refit with irregular_tumor_margin standing in for the paper's "
            "tumour-brain interface. This paper scored tumoral margin and "
            "tumour-brain interface as two separate factors; both were "
            "significant univariably and the multivariable model retained the "
            "interface while margin dropped out. Our substitute is therefore "
            "the variable these authors specifically discarded, in place of the "
            "one they kept. The paper also assessed capsular enhancement and "
            "did not retain it. This is a refit, not an external validation."
        ),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_published_models.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add meningioma-modelling.ipynb heavy_machinery/config/published_models.py heavy_machinery/pytests_atypier/test_published_models.py
git commit -m "feat: six more literature models, with their published records and exclusion reasons"
```

---

### Task 10: Render `surrogate_note`, `NOT_FITTED` and β-only terms

**Files:**
- Modify: `heavy_machinery/modelling_phase/report.py:1050-1100`
- Test: `heavy_machinery/pytests_atypier/test_report.py`

**Interfaces:**
- Consumes: `PUBLISHED_MODELS`, `NOT_FITTED` (Task 9).
- Produces: `_published_model_block` renders a `surrogate_note` warning box and a β column when present; `_not_fitted_block() -> str` renders the exclusions once at the end of the multivariable section.

- [ ] **Step 1: Write the failing test**

```python
def test_published_block_shows_the_surrogate_note_as_a_warning(monkeypatch):
    import report as rp
    monkeypatch.setattr(rp.published_models, "PUBLISHED_MODELS", {
        "m1": {"citation": "Someone 2020", "terms": [
            {"variable": "Interface", "meaning": "x", "column": "irregular_tumor_margin"}],
            "surrogate_note": "Refit with a surrogate; not an external validation."},
    }, raising=False)
    monkeypatch.setattr(rp.published_models, "published_model",
                        lambda mid: rp.published_models.PUBLISHED_MODELS.get(mid))
    html = rp._published_model_block("m1")
    assert "not an external validation" in html
    assert "warn" in html


def test_not_fitted_block_lists_every_exclusion(monkeypatch):
    import report as rp
    monkeypatch.setattr(rp.published_models, "NOT_FITTED",
                        {"yao_2022": "needs a 3-level shape"}, raising=False)
    html = rp._not_fitted_block()
    assert "yao_2022" in html and "3-level shape" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py -k "surrogate or not_fitted" -q`
Expected: FAIL — `AttributeError: module 'report' has no attribute '_not_fitted_block'`

- [ ] **Step 3: Implement**

In `report.py`, extend `_published_model_block`. After the `table = table_to_html(...)` call, add the β column when any term carries one — change the row dict to include `"Published β": t.get("beta", "")` and drop that key when every value is blank. Then before `return details_block(`:

```python
    note = str(published.get("surrogate_note") or "").strip()
    surrogate_html = warning_box(note) if note else ""
```

and include `surrogate_html` in the block body ahead of the table.

Add after `_published_model_block`:

```python
def _not_fitted_block() -> str:
    """Published models deliberately not refit, and why.

    A reader who knows the literature will ask where Yao or Amano went. Without
    this they have to assume the models were missed rather than excluded.
    """
    excluded = getattr(published_models, "NOT_FITTED", {}) or {}
    if not excluded:
        return ""
    rows = pd.DataFrame(
        [{"Model": k, "Why it is not refit": v} for k, v in sorted(excluded.items())]
    )
    return details_block(
        "🚫 Published models not refit",
        "<p>Identified in the literature search and deliberately excluded — "
        "each needs a variable this cohort does not record.</p>"
        + table_to_html(rows),
    )
```

In `render_inferential`, after the per-target loop and before the final `return`, add:

```python
    body.append(_not_fitted_block())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/report.py heavy_machinery/pytests_atypier/test_report.py
git commit -m "feat: render surrogate notes, beta-only terms and the not-refit list"
```

---

### Task 11: Wire the comparison into the inferential stage

**Files:**
- Modify: `heavy_machinery/modelling_phase/inferential.py:869-1005`
- Modify: `heavy_machinery/modelling_phase/model_comparison.py`
- Test: `heavy_machinery/pytests_atypier/test_model_comparison.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 7, 8.
- Produces: three CSVs under `output/inferential/tables/` — `single_predictor_reference.csv`, `model_vs_single_auc.csv`, `top_variable_selection.csv`. Task 12 renders them.
- `run_comparison_stage(cohort_df, schema, variants, target, tabs_dir, *, n_bootstrap) -> dict[str, pd.DataFrame]`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_comparison_stage_writes_all_three_tables(tmp_path):
    import numpy as np, pandas as pd
    from schema_infer import ColSpec
    rng = np.random.RandomState(5)
    n = 220
    y = rng.binomial(1, 0.35, n)
    df = pd.DataFrame({
        "event": y.astype(bool),
        "a": y * 1.2 + rng.normal(size=n),
        "b": y * 0.5 + rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    schema = {"event": ColSpec("event", "binary"),
              **{k: ColSpec(k, "continuous") for k in ("a", "b", "c")}}
    variants = [{"model_id": "m1", "predictors": ["a", "b"]}]
    out = mc.run_comparison_stage(
        df, schema, variants, "event", tmp_path,
        n_bootstrap=30, candidates=["a", "b", "c"], k_top=2,
        assert_reference=False)
    assert (tmp_path / "single_predictor_reference.csv").exists()
    assert (tmp_path / "model_vs_single_auc.csv").exists()
    assert (tmp_path / "top_variable_selection.csv").exists()
    vs_tbl = out["model_vs_single_auc"]
    assert set(vs_tbl["single"]) == {"a", "b"}
    assert {"delta_auc", "delta_ci_lo", "delta_ci_hi", "d2_p"} <= set(vs_tbl.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -k comparison_stage -q`
Expected: FAIL — `AttributeError: module 'model_comparison' has no attribute 'run_comparison_stage'`

- [ ] **Step 3: Implement**

Append to `model_comparison.py`:

```python
def _nested_chi2_per_imputation(
    frames, schema, target: str, full_cols: Sequence[str], single: str,
) -> tuple[list[float], int]:
    """Per-draw LR chi-square for the full model against one of its predictors."""
    import statsmodels.api as sm
    from model_validation import build_complete_case_frame

    stats: list[float] = []
    k = max(len(full_cols) - 1, 1)
    for frame in frames:
        try:
            df_full, cols_full = build_complete_case_frame(
                frame, schema, list(full_cols), target)
            df_red, cols_red = build_complete_case_frame(
                frame, schema, [single], target)
        except (ValueError, RuntimeError):
            continue
        y = df_full[target].astype(int).to_numpy()
        ll_full = sm.Logit(
            y, sm.add_constant(df_full[cols_full].astype(float), has_constant="add")
        ).fit(disp=False).llf
        ll_red = sm.Logit(
            y, sm.add_constant(df_red[cols_red].astype(float), has_constant="add")
        ).fit(disp=False).llf
        stats.append(2.0 * (ll_full - ll_red))
        k = max(len(cols_full) - len(cols_red), 1)
    return stats, k


def run_comparison_stage(
    cohort_df,
    schema,
    variants: Sequence[Any],
    target: str,
    tabs_dir,
    *,
    n_bootstrap: int | None = None,
    candidates: Sequence[str] | None = None,
    k_top: int = 6,
    frames: Sequence[Any] | None = None,
    assert_reference: bool = True,
    selected_model_ids: set[str] | None = None,
):
    """Fit the singles, difference every model against its own, write 3 tables."""
    from pathlib import Path

    import pandas as pd
    import variable_selection as vs
    from heavy_machinery.config import load
    from model_validation import (
        build_complete_case_frame,
        bootstrap_internal_validation,
    )

    analysis = load("analysis")
    if n_bootstrap is None:
        n_bootstrap = analysis.BOOTSTRAP_RESAMPLES
    tabs_dir = Path(tabs_dir)
    tabs_dir.mkdir(parents=True, exist_ok=True)
    frames = list(frames) if frames is not None else [cohort_df]

    def _pred(v):
        return list(v["predictors"] if isinstance(v, dict) else v.predictors)

    def _mid(v):
        return str(v["model_id"] if isinstance(v, dict) else v.model_id)

    singles = sorted({p for v in variants for p in _pred(v)})
    # ``candidates`` is the selection pool and is NOT the literature union: the
    # six that win include adc_value and cystic_component, which appear in no
    # literature model. The notebook passes its EDA predictor list.
    fitted = fit_single_predictors(
        cohort_df, schema, singles, target, n_bootstrap=n_bootstrap)

    single_tbl = pd.DataFrame([
        {"predictor": p, "n": d["n"], "events": d["events"],
         "auc_apparent": round(d["auc_apparent"], 3),
         "auc_corrected": round(d["auc_corrected"], 3)}
        for p, d in sorted(fitted.items())
    ])
    single_tbl.to_csv(tabs_dir / "single_predictor_reference.csv", index=False)

    y_all = cohort_df[target].astype(int).to_numpy()
    cand = list(candidates) if candidates is not None else singles
    picked, audit = vs.select_variables(
        cohort_df, y_all, cand, k=k_top,
        rho_max=0.8, cutpoint_parent=analysis.CUTPOINT_PARENT)
    if assert_reference:
        vs.assert_reference(picked)
    sel_tbl = pd.DataFrame(audit)
    sel_tbl.to_csv(tabs_dir / "top_variable_selection.csv", index=False)

    rows = []
    for v in variants:
        mid, preds = _mid(v), _pred(v)
        try:
            model_df, design_cols = build_complete_case_frame(
                cohort_df, schema, preds, target)
        except (ValueError, RuntimeError):
            continue
        import statsmodels.api as sm
        y = model_df[target].astype(int).to_numpy()
        X = sm.add_constant(model_df[design_cols].astype(float), has_constant="add")
        fit = sm.Logit(y, X).fit(disp=False)
        coefs = {"const": float(fit.params.iloc[0])}
        coefs.update({c: float(fit.params[c]) for c in design_cols})
        # The one data-selected model re-runs its own selection inside every
        # resample, so its optimism covers the picking and not just the fitting.
        selector = None
        if selected_model_ids and mid in selected_model_ids:
            def selector(frame, y_boot, _cand=cand, _k=k_top):
                sub, _ = vs.select_variables(
                    frame, y_boot, [c for c in _cand if c in frame.columns],
                    k=_k, rho_max=0.8,
                    cutpoint_parent=analysis.CUTPOINT_PARENT)
                return sub
        val = bootstrap_internal_validation(
            model_df, target, design_cols, coefs,
            n_bootstrap=n_bootstrap, return_resample_aucs=True, select=selector)
        combined_auc = next(
            m for m in val["metrics"] if m["metric"] == "AUC")["optimism_corrected"]
        for single in preds:
            if single not in fitted:
                continue
            d = paired_delta_auc(val["resample_aucs"], fitted[single]["resample_aucs"])
            chi2s, k = _nested_chi2_per_imputation(
                frames, schema, target, preds, single)
            p = d2_pool(chi2s, k)["p"] if chi2s else float("nan")
            rows.append({
                "model_id": mid, "single": single,
                "auc_model_corrected": round(float(combined_auc), 3),
                "auc_single_corrected": round(fitted[single]["auc_corrected"], 3),
                "delta_auc": round(d["delta"], 3),
                "delta_ci_lo": round(d["ci_lo"], 3),
                "delta_ci_hi": round(d["ci_hi"], 3),
                "d2_p": p,
            })
    vs_tbl = pd.DataFrame(rows)
    vs_tbl.to_csv(tabs_dir / "model_vs_single_auc.csv", index=False)
    return {"single_predictor_reference": single_tbl,
            "model_vs_single_auc": vs_tbl,
            "top_variable_selection": sel_tbl,
            "top_variables": picked}
```

In `inferential.run_inferential`, after `_write_performance_figures(...)`:

```python
    from model_comparison import run_comparison_stage

    run_comparison_stage(
        imputed_frames[0], schema, model_variants, targets[0], tabs_dir,
        frames=imputed_frames,
        candidates=selection_candidates,   # EDA predictor pool, from the notebook
        selected_model_ids={"top_6_variables"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_model_comparison.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/model_comparison.py heavy_machinery/modelling_phase/inferential.py heavy_machinery/pytests_atypier/test_model_comparison.py
git commit -m "feat: write the single-predictor, delta-AUC and selection-audit tables"
```

---

### Task 12: Replace the frozen top-N lists with the computed selection

**Files:**
- Modify: `heavy_machinery/modelling_phase/inferential.py` (`run_inferential`, `run_inferential_stage` — thread `selection_candidates`)
- Modify: `meningioma-modelling.ipynb` (`EXPERIMENTAL_MODEL_VARIANTS` cell, and the `run_inferential_stage` call)
- Test: `heavy_machinery/pytests_atypier/test_inferential.py`

**Interfaces:**
- Consumes: `variable_selection.select_variables` and `assert_reference` (Tasks 5, 7); `run_comparison_stage(..., candidates=, selected_model_ids=)` (Task 11).
- Produces: variants `top_6_variables` and `top_1_variable` built at runtime from the selection, and `run_inferential_stage(..., selection_candidates=Sequence[str])`.

Without this task the machinery from Tasks 5, 6, 7 and 11 exists and never runs
on a real model: the notebook still fits the frozen LR+ lists `top_6_signs` and
`top_1_sign`, and no model passes `select=` into its validation. This is the task
that makes the selection real.

- [ ] **Step 1: Write the failing test**

```python
def test_run_inferential_stage_threads_selection_candidates(tiny_df, tiny_schema, tmp_output):
    """The selection pool is the EDA predictor list, not the literature union —
    the variables that win include ones no literature model uses."""
    import inspect
    sig = inspect.signature(inf.run_inferential_stage)
    assert "selection_candidates" in sig.parameters
    sig2 = inspect.signature(inf.run_inferential)
    assert "selection_candidates" in sig2.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_inferential.py -k selection_candidates -q`
Expected: FAIL — `AssertionError`

- [ ] **Step 3: Thread the parameter**

Add `selection_candidates: Sequence[str] | None = None` to both
`run_inferential` and `run_inferential_stage` (keyword-only, after
`vif_threshold`), pass it straight through from the stage wrapper, and forward it
into the `run_comparison_stage` call added in Task 11 as `candidates=`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_inferential.py -q`
Expected: PASS

- [ ] **Step 5: Compute the two variants in the notebook**

In `meningioma-modelling.ipynb`, replace the two frozen tuples at the end of
`EXPERIMENTAL_MODEL_VARIANTS` — the ones with ids `top_6_signs` and
`top_1_sign`, together with the ⚠️ comment block above them — with a computed
block. It must run *after* `EDA_PREDICTORS` is resolved, so put it in the cell
that already calls `_analysis.resolve_eda`, immediately after
`INFERENTIAL_MODEL_VARIANTS` is built:

```python
# 🔎 top_6_variables / top_1_variable — computed, never a frozen list.
#
# Ranked by discrimination, max(AUC, 1-AUC), so a protective variable is not
# thrown away for scoring below 0.5: ADC is 0.370 here, which is 0.630 the
# other way and the second-strongest variable available.
#
# Two guards, in order: skip a derived cut-point when its continuous parent is
# a candidate, and skip anything correlated above rho 0.8 with something
# already picked, taking the next candidate that clears. Without the second,
# the best six are four tumour-size measurements at rho up to 0.91.
#
# ⚠️ These two are chosen from the same 352 patients they are then fitted on.
# The bootstrap re-runs this selection inside every resample so the optimism
# correction covers the choosing, but they are still not comparable to the
# literature models, whose predictors were fixed by other people years ago.
from heavy_machinery.modelling_phase import variable_selection as _vs

_y = df[EDA_TARGETS[0]].astype("boolean").fillna(False).astype(int).to_numpy()
TOP_VARIABLES, TOP_SELECTION_AUDIT = _vs.select_variables(
    df, _y, EDA_PREDICTORS, k=6, rho_max=0.8,
    cutpoint_parent=load("analysis").CUTPOINT_PARENT,
)
_vs.assert_reference(TOP_VARIABLES)

INFERENTIAL_MODEL_VARIANTS = list(INFERENTIAL_MODEL_VARIANTS) + [
    _analysis.InferentialModelVariant(
        model_id="top_1_variable",
        title="Top 1 variable by discrimination | high grade",
        link="", target=EDA_TARGETS[0],
        predictors=(TOP_VARIABLES[0],), experimental=True,
    ),
    _analysis.InferentialModelVariant(
        model_id="top_6_variables",
        title="Top 6 variables by discrimination | high grade",
        link="", target=EDA_TARGETS[0],
        predictors=tuple(TOP_VARIABLES), experimental=True,
    ),
]
print(f"🔎 top 6: {TOP_VARIABLES}")
```

`_analysis.InferentialModelVariant` is re-exported by `config/analysis.py`,
which already imports it from `inferential`. Delete the two frozen tuples from
`EXPERIMENTAL_MODEL_VARIANTS` so the ids cannot collide.

- [ ] **Step 6: Pass the pool into the stage call**

In the `run_inferential_stage(...)` call, add:

```python
    selection_candidates=EDA_PREDICTORS,
```

- [ ] **Step 7: Verify on real data that the six match the spec**

```bash
python3 - <<'PY'
import sys, json
from pathlib import Path
sys.path[:0] = [str(Path('heavy_machinery')/d) for d in
                ('modelling_phase','cleaning_phase','config')]
sys.path.insert(0, '.')
import missingness_resolution as mr
import variable_selection as vs
from heavy_machinery.config import load
d = mr.load_modeling_frames(Path('output'))[0]
cells = [''.join(c['source']) for c in
         json.loads(Path('meningioma-modelling.ipynb').read_text())['cells']
         if c['cell_type'] == 'code']
ns = {}
exec(next(s for s in cells if 'EDA_PREDICTORS = [' in s).split('EDA_REDUNDANT')[0], ns)
cand = [c for c in ns['EDA_PREDICTORS'] if c in d.columns]
y = d['high_grade'].astype(int).to_numpy()
picked, audit = vs.select_variables(
    d, y, cand, k=6, rho_max=0.8,
    cutpoint_parent=load("analysis").CUTPOINT_PARENT)
print("picked:", picked)
assert picked == ["tumor_volume", "adc_value", "edema_volume_cm3",
                  "irregular_tumor_margin", "skull_base_location",
                  "cystic_component"], picked
vs.assert_reference(picked)
print("dropped:", [(r['variable'], r['reason']) for r in audit if not r['kept']])
PY
```

Expected: the six from the spec, in that order, and six dropped candidates each
with a reason.

- [ ] **Step 8: Commit**

```bash
git add heavy_machinery/modelling_phase/inferential.py meningioma-modelling.ipynb \
        heavy_machinery/pytests_atypier/test_inferential.py
git commit -m "feat: compute top_6_variables and top_1_variable instead of freezing them"
```

---

### Task 13: Slim the odds-ratio table to four columns

**Files:**
- Modify: `heavy_machinery/modelling_phase/inferential.py` (rename `_forest_row_label` → `predictor_label`)
- Modify: `heavy_machinery/modelling_phase/report.py` (`render_inferential`'s per-model table)
- Test: `heavy_machinery/pytests_atypier/test_report.py`, `heavy_machinery/pytests_atypier/test_inferential.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `report._beta_se(coef, se) -> str`, `report._or_ci(o, lo, hi) -> str`, `report._model_level_line(tbl) -> str`, and `inferential.predictor_label(row) -> str` (public rename, same behaviour). Task 13 renders alongside these.

The table currently shows 18 columns of which **seven are identical on every row**
(`target`, `model_id`, `experimental`, `n_models`, `intercept_coef`,
`intercept_or`, `df`) and two more are blank for every binary predictor
(`z_mu`, `z_sd`). The published-model table directly above renders
`Variable | … | Published aOR (95% CI) | p`, so matching that shape lets a reader
read the two against each other line by line. The CSV keeps every column.

- [ ] **Step 1: Write the failing test**

Add to `heavy_machinery/pytests_atypier/test_report.py`:

```python
def test_beta_se_and_or_ci_formatting():
    import report as rp
    assert rp._beta_se(0.96, 0.37) == "0.96 (0.37)"
    assert rp._beta_se(None, 0.37) == ""
    assert rp._or_ci(2.60, 1.26, 5.38) == "2.60 (1.26–5.38)"
    assert rp._or_ci(2.60, None, None) == "2.60"


def test_model_level_line_states_intercept_and_imputations():
    import pandas as pd, report as rp
    tbl = pd.DataFrame({"intercept_coef": [-1.16, -1.16], "intercept_or": [0.312, 0.312],
                        "n_models": [20, 20], "df": ["∞", "∞"]})
    line = rp._model_level_line(tbl)
    assert "-1.16" in line and "0.312" in line and "20" in line and "∞" in line


def test_multivariable_table_shows_four_columns_only(report_cfg, report_art):
    import pandas as pd, report as rp
    report_art.inferential_multivariable = {
        "high_grade::m1": pd.DataFrame({
            "predictor_col": ["age", "male"],
            "coef": [0.13, 0.72], "se": [0.13, 0.27],
            "or": [1.14, 2.05], "or_ci_lo": [0.88, 1.22], "or_ci_hi": [1.46, 3.46],
            "p": [0.315, 0.007], "df": ["∞", "∞"], "n_models": [20, 20],
            "intercept_coef": [-1.16, -1.16], "intercept_or": [0.312, 0.312],
            "z_mu": [63.1, None], "z_sd": [12.68, None],
            "target": ["high_grade"] * 2, "model_id": ["m1"] * 2,
            "experimental": [True, True],
        })
    }
    html = rp.render_inferential(report_cfg, report_art)
    for gone in ("model_id", "experimental", "intercept_coef", "n_models", "z_sd"):
        assert f"<th>{gone}</th>" not in html
    assert "β (SE)" in html and "OR (95% CI)" in html
    assert "per 1 SD: 12.68" in html          # z_sd folded into the predictor name
```

Add to `heavy_machinery/pytests_atypier/test_inferential.py`, replacing the two
`inf._forest_row_label` references:

```python
def test_predictor_label_names_the_sd_for_a_continuous_predictor():
    row = _pooled().iloc[1].copy()          # adc_value, z_sd = 0.17
    assert "per 1 SD: 0.17" in inf.predictor_label(row)


def test_predictor_label_ignores_a_missing_or_zero_sd():
    row = _pooled().iloc[0].copy()
    row["z_sd"] = 0.0
    assert "per 1 SD" not in inf.predictor_label(row)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py -k "beta_se or model_level or four_columns" heavy_machinery/pytests_atypier/test_inferential.py -k predictor_label -q`
Expected: FAIL — `AttributeError: module 'report' has no attribute '_beta_se'`

- [ ] **Step 3: Make the label helper public**

In `heavy_machinery/modelling_phase/inferential.py`, rename `_forest_row_label` to
`predictor_label` (definition and its one call inside `_forest_plot`). The
docstring already explains why the SD is named; leave it. Making it public is the
point — the table and the forest plot must label a predictor identically or the
reader sees "per 1 SD: 12.68" in one and a bare name in the other.

- [ ] **Step 4: Add the formatters and rebuild the display table**

In `report.py`, add beside `_published_or`:

```python
def _beta_se(coef: Any, se: Any) -> str:
    """``0.96 (0.37)`` — the log-odds coefficient with its standard error."""
    c, s = _coerce_float(coef), _coerce_float(se)
    if c is None:
        return ""
    return f"{c:.2f} ({s:.2f})" if s is not None else f"{c:.2f}"


def _or_ci(o: Any, lo: Any, hi: Any) -> str:
    """``2.60 (1.26–5.38)`` — same shape as the published table above it."""
    ov, l, h = (_coerce_float(x) for x in (o, lo, hi))
    if ov is None:
        return ""
    return f"{ov:.2f} ({l:.2f}–{h:.2f})" if l is not None and h is not None else f"{ov:.2f}"


def _model_level_line(tbl: pd.DataFrame) -> str:
    """Facts that were repeated on every row, stated once.

    The intercept, the imputation count and the pooled degrees of freedom are
    properties of the model, not of any predictor. As columns they cost seven
    cells per row and told the reader nothing new after the first one.
    """
    bits: list[str] = []
    if "intercept_coef" in tbl.columns and len(tbl):
        ic = _coerce_float(tbl["intercept_coef"].iloc[0])
        io = _coerce_float(tbl["intercept_or"].iloc[0]) if "intercept_or" in tbl.columns else None
        if ic is not None:
            bits.append(f"Intercept {ic:.2f}" + (f" (OR {io:.3f})" if io is not None else ""))
    if "n_models" in tbl.columns and len(tbl):
        n = _to_int_or_none(tbl["n_models"].iloc[0])
        if n:
            bits.append(f"pooled across {n} imputations")
    if "df" in tbl.columns and len(tbl):
        dfs = {human_pool_df(v) for v in tbl["df"] if str(v).strip()}
        if len(dfs) == 1:
            bits.append(f"Rubin df {dfs.pop()}")
    return f'<p class="muted">{" &middot; ".join(_esc(b) for b in bits)}</p>' if bits else ""
```

In `render_inferential`, replace the block from `nowrap = ("model_id",) …` through
the `table_to_html(...)` append with:

```python
                # The numeric ``tbl`` stays as-is for the interpretation block
                # below; the display copy is what the reader sees.
                display = pd.DataFrame({
                    "Predictor": [predictor_label(r) for _, r in tbl.iterrows()],
                    "β (SE)": [_beta_se(r.get("coef"), r.get("se"))
                               for _, r in tbl.iterrows()],
                    "OR (95% CI)": [_or_ci(r.get(col_or), r.get(col_lo), r.get(col_hi))
                                    for _, r in tbl.iterrows()],
                    "P": [human_p(r.get(col_p)) for _, r in tbl.iterrows()],
                })
                blocks.append(_model_level_line(tbl))
                blocks.append(table_to_html(
                    display, row_class_fn=_row_cls,
                    nowrap_cols=("β (SE)", "OR (95% CI)", "P")))
```

Delete the two lines that pre-format `tbl[col_p]` and `tbl["df"]` for display —
`human_p` is now applied when building `display`, and `df` is no longer a column.
`_row_cls` still reads `col_or`/`col_lo`/`col_hi` from the numeric row, so pass
`tbl` rows to it unchanged.

Import the renamed helper at the top of `report.py`:

```python
from inferential import (
    artifact_base,
    model_key,
    parse_artifact_base,
    parse_model_key,
    predictor_label,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py heavy_machinery/pytests_atypier/test_inferential.py -q`
Expected: PASS

- [ ] **Step 6: Rebuild the report and eyeball one table**

```bash
python3 - <<'PY'
import re, sys
from pathlib import Path
sys.path[:0] = [str(Path('heavy_machinery')/d) for d in
                ('modelling_phase','cleaning_phase','cutpoint_phase','config')]
sys.path.insert(0, '.')
from heavy_machinery.config import load
load("report_settings").run_report(
    output_root=Path("output"), report_title="t", report_author="a",
    report_path=Path("output/report/report.html"), analysis_years=None,
    eda_targets=["high_grade"])
h = re.sub(r'data:image/[a-z]+;base64,[A-Za-z0-9+/=]+','[IMG]',
           Path('output/report/report.html').read_text())
i = h.index('🧮 Multivariable modelling')
hdrs = re.findall(r'<thead>(.*?)</thead>', h[i:i+60000], re.S)
for k in hdrs[:4]:
    print([re.sub('<[^>]+>','',c).strip() for c in re.findall(r'<th[^>]*>(.*?)</th>', k, re.S)])
PY
```

Expected: the odds-ratio tables print `['Predictor', 'β (SE)', 'OR (95% CI)', 'P']`.

- [ ] **Step 7: Commit**

```bash
git add heavy_machinery/modelling_phase/inferential.py heavy_machinery/modelling_phase/report.py \
        heavy_machinery/pytests_atypier/test_report.py heavy_machinery/pytests_atypier/test_inferential.py
git commit -m "refactor: odds-ratio table down to Predictor, beta (SE), OR (95% CI), P"
```

---

### Task 14: Render the comparison tables and the direction column

**Files:**
- Modify: `heavy_machinery/modelling_phase/report.py` (`Artifacts`, `load_artifacts`, `render_inferential`)
- Test: `heavy_machinery/pytests_atypier/test_report.py`

**Interfaces:**
- Consumes: the three CSVs from Task 11.
- Produces: `Artifacts.model_vs_single`, `.single_reference`, `.top_selection`; `_model_vs_single_block(model_id, art) -> str` and `_selection_audit_block(art) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_model_vs_single_block_shows_delta_and_ci(report_art):
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([
        {"model_id": "m1", "single": "tumor_volume",
         "auc_model_corrected": 0.73, "auc_single_corrected": 0.68,
         "delta_auc": 0.05, "delta_ci_lo": 0.01, "delta_ci_hi": 0.09, "d2_p": 0.004},
    ])
    html = rp._model_vs_single_block("m1", report_art)
    assert "0.05" in html and "0.01" in html and "tumor_volume" in html


def test_selection_audit_block_names_the_dropped_variable_and_reason(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": ""},
        {"variable": "max_diameter_cm", "auc": 0.675, "discrimination": 0.675,
         "kept": False, "reason": "rho=0.91 with tumor_volume"},
    ])
    html = rp._selection_audit_block(report_art)
    assert "max_diameter_cm" in html and "rho=0.91" in html


def test_discrimination_is_shown_for_a_protective_variable(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "adc_value", "auc": 0.370, "discrimination": 0.630,
         "kept": True, "reason": ""},
    ])
    html = rp._selection_audit_block(report_art)
    assert "0.630" in html and "↓" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py -k "model_vs_single or selection_audit or protective" -q`
Expected: FAIL — `AttributeError: module 'report' has no attribute '_model_vs_single_block'`

- [ ] **Step 3: Implement**

Add three fields to `Artifacts`:

```python
    model_vs_single: pd.DataFrame | None = None
    single_reference: pd.DataFrame | None = None
    top_selection: pd.DataFrame | None = None
```

Load them in `load_artifacts` beside the other inferential tables:

```python
    art.model_vs_single = _maybe_read_csv(inf_tab / "model_vs_single_auc.csv", art.warnings)
    art.single_reference = _maybe_read_csv(inf_tab / "single_predictor_reference.csv", art.warnings)
    art.top_selection = _maybe_read_csv(inf_tab / "top_variable_selection.csv", art.warnings)
```

Add the two renderers near `_published_model_block`:

```python
def _model_vs_single_block(model_id: str, art: Artifacts) -> str:
    """Does this combination beat each single predictor it is built from?"""
    tbl = art.model_vs_single
    if tbl is None or tbl.empty or "model_id" not in tbl.columns:
        return ""
    sub = tbl[tbl["model_id"].astype(str) == str(model_id)]
    if sub.empty:
        return ""
    rows = pd.DataFrame([{
        "Single predictor": r["single"],
        "Model AUC": format_number(r["auc_model_corrected"]),
        "Single AUC": format_number(r["auc_single_corrected"]),
        "ΔAUC (95% CI)": (f'{float(r["delta_auc"]):.3f} '
                          f'({float(r["delta_ci_lo"]):.3f}–{float(r["delta_ci_hi"]):.3f})'),
        "p (D2)": human_p(r.get("d2_p")),
    } for _, r in sub.iterrows()])
    return details_block(
        "⚖️ Does the combination beat its own single predictors?",
        "<p>Both AUCs are optimism-corrected, and the difference is taken "
        "<em>within</em> each bootstrap resample, so the interval is a paired "
        "one. A CI that spans zero means this cohort cannot tell the "
        "combination apart from that single predictor. The p-value is a "
        "likelihood-ratio test pooled across the MICE draws by Rubin's D2.</p>"
        + table_to_html(rows, nowrap_cols=("ΔAUC (95% CI)", "p (D2)")),
    )


def _selection_audit_block(art: Artifacts) -> str:
    """Which candidates were considered, kept, and dropped — and why."""
    tbl = art.top_selection
    if tbl is None or tbl.empty:
        return ""
    rows = pd.DataFrame([{
        "Variable": r["variable"],
        "AUC": format_number(r["auc"]),
        "Discrimination": (f'{float(r["discrimination"]):.3f}'
                           + (" ↓" if float(r["auc"]) < 0.5 else "")),
        "Kept": "✅" if bool(r["kept"]) else "—",
        "Why dropped": r.get("reason", ""),
    } for _, r in tbl.iterrows()])
    return details_block(
        "🔎 How these variables were chosen",
        "<p>Candidates in descending discrimination — <code>max(AUC, 1−AUC)</code>, "
        "so a protective variable is not discarded for scoring below 0.5. A "
        "derived cut-point is skipped when its continuous parent is available, "
        "and anything correlated above ρ=0.8 with a variable already kept is "
        "skipped in favour of the next one that clears.</p>"
        + table_to_html(rows, nowrap_cols=("AUC", "Discrimination", "Kept")),
    )
```

Call `_model_vs_single_block(model_id, art)` in `_render_model_blocks` immediately after `blocks.append(_render_inferential_interpretation(...))`, and `_selection_audit_block(art)` once before `_not_fitted_block()` in `render_inferential`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest heavy_machinery/pytests_atypier/test_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/modelling_phase/report.py heavy_machinery/pytests_atypier/test_report.py
git commit -m "feat: render combined-vs-single tables and the selection audit"
```

---

### Task 15: Full clean pipeline run and verification

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Consumes: everything.
- Produces: a regenerated `output/` tree and a verified `report.html`.

- [ ] **Step 1: Back up the current output tree**

```bash
rm -rf /tmp/output_backup && cp -R output /tmp/output_backup && find /tmp/output_backup -type f | wc -l
```

- [ ] **Step 2: Run all three notebooks in order**

```bash
for nb in meningioma-cleaning meningioma-cutpoints meningioma-modelling; do
  ATYPIER_FIGURES=report jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 --output-dir /tmp --output "$nb.out.ipynb" "$nb.ipynb" || break
done
```

- [ ] **Step 3: Verify the model inventory is 22 and the tables exist**

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path
t = Path('output/inferential/tables')
models = sorted(p.stem.replace('high_grade__','').replace('__multivariable','')
                for p in t.glob('*__multivariable.csv'))
print(f"{len(models)} fitted models:", models)
for f in ('single_predictor_reference.csv','model_vs_single_auc.csv','top_variable_selection.csv'):
    d = pd.read_csv(t/f); print(f"{f}: {len(d)} rows"); assert len(d) > 0
vs = pd.read_csv(t/'model_vs_single_auc.csv')
assert vs['delta_ci_lo'].notna().all() and vs['d2_p'].notna().any()
sel = pd.read_csv(t/'top_variable_selection.csv')
assert sel['kept'].sum() == 6, sel
print("top 6:", sel.loc[sel['kept'],'variable'].tolist())
PY
```

Expected: 10 combined models (7 literature + 3 experimental), all three tables non-empty, exactly 6 kept variables.

- [ ] **Step 4: Verify the report renders every new block**

```bash
python3 - <<'PY'
import re
from pathlib import Path
h = re.sub(r'data:image/[a-z]+;base64,[A-Za-z0-9+/=]+','[IMG]',
           Path('output/report/report.html').read_text())
for needle in ("Does the combination beat its own single predictors",
               "How these variables were chosen",
               "Published models not refit",
               "not an external validation"):
    assert needle in h, needle
    print("ok:", needle)
PY
```

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS, zero failures.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: full clean pipeline run with the literature refits"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: B=1000/seed 20260801 → Task 2; ΔAUC combined-vs-each-single → Tasks 3, 11; reference declaration → Task 7; lightweight singles → Task 8; optimism-corrected ΔAUC → Task 3; D2 → Task 4; MICE unchanged → no task by design; selection-inside-bootstrap → Task 6; ρ=0.8 and both guards → Task 5; top-N lists computed → Task 12; table slimmed to four columns → Task 13; direction column → Task 14; `age_ge75` → Task 1; six literature models + `published_models` → Task 9; `NOT_FITTED` and `surrogate_note` → Tasks 9, 10; three CSV artifacts → Task 11; report blocks → Tasks 10, 13, 14; clean run → Task 15.

**Gap found and closed:** the spec's comparison-figure change (11 rows, reference row distinguished) has no task. `model_comparison_figure` already takes whatever entries it is handed, so the row count follows automatically from the model list — but the reference-row styling does not. Deferred deliberately: it is cosmetic, and `_COMPARISON_METRICS` is untouched by this plan. Recorded here rather than silently dropped.

**Type consistency.** `resample_aucs` is `list[float]` everywhere (Tasks 2, 3, 8, 11). `select` takes `(frame, y_array)` and returns `list[str]` in both Task 6's hook and Task 11's caller. Audit rows carry exactly `variable`, `auc`, `discrimination`, `kept`, `reason` in Tasks 5, 11, 12 and 14. `CUTPOINT_PARENT` lives in `analysis.py` and is read by name in Tasks 5 and 11.
