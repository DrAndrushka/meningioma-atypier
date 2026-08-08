# Report Correctness Fixes (Spec §5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the eight correctness problems (spec §5) in `report.html` and `threshold_report.html`, then re-run the pipeline so the already-committed midline fix finally reaches the data and both reports.

**Architecture:** All changes are edits to the existing pipeline — `heavy_machinery/modelling_phase/eda.py` (FDR family), `heavy_machinery/modelling_phase/report.py` + `heavy_machinery/threshold_phase/threshold_report.py` (footnotes, flags), a new shared constant in `heavy_machinery/config/analysis.py` (bootstrap count), and the modelling notebook's predictor lists. The final task re-runs the three notebooks in order and verifies the midline flip end to end.

**Tech Stack:** Python 3 / pandas / matplotlib, R `mice` via subprocess (required for the modelling notebook), pytest suite in `heavy_machinery/pytests_atypier/`.

**Source spec:** `/Users/andriszaguzovs/Downloads/html_report_improvement_spec.md` §5 (plus §5.1's dependency on a pipeline re-run).

## Global Constraints

- Scope is `report.html` + `threshold_report.html` only; improve the existing pipeline, do not rebuild it (spec preamble).
- Run pytest and the notebooks **from the repo root** — imports resolve via `pytest.ini` / `pyrightconfig.json` (bare imports like `import eda` work only from there).
- Baseline suite is fully green (381 tests). Any failure after your change is a regression you introduced.
- Bootstrap resamples harmonised to **1000** everywhere, printed once per report in the Methods block (spec 5.3).
- Cohort event rate always **“29.8% (105/352)”** — 1 decimal place with the fraction; any subset rate must carry its own n (spec 5.4).
- Every SVG rewrites cosmetically on every pipeline run — never use `diff` on SVGs to prove or disprove a change.
- Test import idiom in `heavy_machinery/pytests_atypier/`: bare module imports (`import eda`, `from schema_infer import ColSpec`), `matplotlib.use("Agg")` before pyplot.
- Commit after each task. Do not commit `output/` artifacts unless git already tracks the specific file.

## Key background facts (verified 2026-08-08)

- **Midline (spec 5.1) is already fixed in code but not in data.** Commit `f1e4bab` changed the cleaning-notebook derivation from `side in {'left','right'}` (lateralised — the inverted rule the spec found) to `side == 'midline'`. But `output/datasets/unimputed_df.parquet` still holds the old flag: `midline` True = 317/352. After re-run it must be True = 35/352 (`side` value counts: right 159, left 158, midline 35). Every downstream number the spec quotes (Sens 97.1%/Spec 13.0%) will legitimately flip.
- The EDA sweep call site is `meningioma-modelling.ipynb` (search `screen_associations(`), passing `EDA_PREDICTORS`; the sweep itself is `screen_associations()` at `heavy_machinery/modelling_phase/eda.py:1078`, FDR applied per target at lines 1183–1188 via `benjamini_hochberg()` (line 79).
- Bootstrap counts live at `heavy_machinery/modelling_phase/model_validation.py:199,378` (`n_bootstrap: int = 1000`) and `heavy_machinery/threshold_phase/calibration.py:193,219,318` + `heavy_machinery/threshold_phase/combinations.py:326` (`n_boot: int = 500`).
- The `outcome rate {:.0%}` string is `heavy_machinery/modelling_phase/performance_plots.py:243`.
- Threshold report: "Share of range" table is built at `heavy_machinery/threshold_phase/threshold_report.py:1076` inside `render_usefulness` (line 1029), which also renders the Uncut-vs-Cut calibration comparison (`data.table("calibration")`, line 1030). Evidence vocabulary (`fragile`/`survives`) is defined in `heavy_machinery/threshold_phase/evidence.py:27-80,186-203` and rendered by `render_evidence` (`threshold_report.py:960`).
- DDA continuous stats: `output/dda/tables/dda_continuous.csv` with columns `column,kind,n,...,min,...,max,...`; loaded as `art.dda_continuous` (`report.py:770`), rendered in `render_dda` (`report.py:1545`).
- Report helpers available in `report.py`: `warning_box(msg)` (line 541), `info_box(msg)` (547), `details_block(summary, inner_html)` (630), `table_to_html(df)` (551).
- Threshold-report helpers: `_table(df)` (293), `_lead(text)`, `_answer(text)` (301), `details_block` imported from the modelling report module.

---

### Task 1: FDR family without redundant tests (spec 5.2)

The current Tab16 sweep BH-corrects across 41 predictors that include five continuous variables *and their own dichotomisations*, plus binary recodes of nominal parents. Fix: `screen_associations()` gains an `fdr_family` parameter — only predictors in the family enter the Benjamini–Hochberg correction; the rest keep their raw p, get `p_fdr = NaN`, and are marked `in_fdr_family = False` so the report can shunt them into a collapsed exploratory block.

**Files:**
- Modify: `heavy_machinery/modelling_phase/eda.py:1078-1188` (signature + FDR block)
- Modify: `meningioma-modelling.ipynb` (the cell defining `EDA_PREDICTORS`, and the `screen_associations(` call cell)
- Test: `heavy_machinery/pytests_atypier/test_eda.py`

**Interfaces:**
- Produces: `screen_associations(..., fdr_family: Sequence[str] | None = None)`; output DataFrame gains bool column `in_fdr_family`. `fdr_family=None` keeps today's behaviour (every predictor in the family). `associations.csv` carries the new column; Task 8's re-run materialises it for the report.
- The redundant set (excluded from the family, per spec 5.2 plus the same rule applied to `midline`, which after the 5.1 fix is a pure recode of `side`):
  `tumor_volume_ge15.1`, `adc_value_le0.72`, `max_diameter_cm_ge3.81`, `edema_volume_ge4.76`, `edema_index_ge0.0617`, `male_sex`, `irregular_margin`, `skull_base_location`, `venous_sinus_invasion`, `midline`.
  The parents (`tumor_volume`, `adc_value`, `max_diameter_cm`, `edema_volume_cm3`, `edema_index`, `sex`, `tumor_margin`, `tumor_location`, `sinus_invasion`, `side`) stay in the family.

- [ ] **Step 1: Write the failing test**

Append to `heavy_machinery/pytests_atypier/test_eda.py`:

```python
def _fdr_family_fixture():
    rng = np.random.default_rng(7)
    n = 80
    y = pd.Series(rng.integers(0, 2, n).astype(bool), name="high_grade")
    df = pd.DataFrame({
        "high_grade": y,
        "vol": rng.normal(10, 3, n) + y * 2.0,
        "vol_ge10": pd.array(rng.integers(0, 2, n).astype(bool)),
        "adc": rng.normal(1.0, 0.2, n) - y * 0.1,
    })
    schema = {
        "high_grade": ColSpec(name="high_grade", kind="binary"),
        "vol": ColSpec(name="vol", kind="continuous"),
        "vol_ge10": ColSpec(name="vol_ge10", kind="binary"),
        "adc": ColSpec(name="adc", kind="continuous"),
    }
    return df, schema


def test_fdr_family_limits_correction_to_family(tmp_path):
    df, schema = _fdr_family_fixture()
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        fdr_family=["vol", "adc"],
        output_root=tmp_path,
    )
    fam = out[out["in_fdr_family"]]
    extra = out[~out["in_fdr_family"]]
    assert sorted(fam["predictor"]) == ["adc", "vol"]
    assert list(extra["predictor"]) == ["vol_ge10"]
    # excluded row: raw p kept, no corrected p, never flagged significant
    assert extra["p"].notna().all()
    assert extra["p_fdr"].isna().all()
    assert not extra["fdr_significant"].any()
    # family rows: BH computed over exactly the 2 family tests
    expected = eda.benjamini_hochberg(fam["p"])
    assert np.allclose(fam["p_fdr"].values, expected.values)


def test_fdr_family_none_keeps_everything_in_family(tmp_path):
    df, schema = _fdr_family_fixture()
    out = eda.screen_associations(
        df, schema,
        targets=["high_grade"],
        predictors=["vol", "vol_ge10", "adc"],
        output_root=tmp_path,
    )
    assert out["in_fdr_family"].all()
    assert out["p_fdr"].notna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_eda.py -k fdr_family -v`
Expected: FAIL — `screen_associations() got an unexpected keyword argument 'fdr_family'`.

- [ ] **Step 3: Implement in `eda.py`**

Add the parameter to the signature at line 1078 (after `predictors`):

```python
def screen_associations(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    targets: Sequence[str],
    predictors: Sequence[str] | None = None,
    fdr_family: Sequence[str] | None = None,
    positive_class: dict | None = None,
    fdr_alpha: float = 0.05,
    output_root: Path | str = "output",
) -> pd.DataFrame:
```

Document it in the docstring:

```python
    fdr_family    : predictors whose tests form the multiplicity family.
                    BH correction runs over these only; predictors outside
                    the family keep their raw p, get ``p_fdr = NaN`` and
                    ``in_fdr_family = False`` (exploratory, uncorrected).
                    ``None`` (default) puts every predictor in the family.
```

Replace the FDR block (currently lines 1183–1188) with:

```python
    # FDR per target, restricted to the declared multiplicity family
    family = set(fdr_family) if fdr_family is not None else set(out["predictor"])
    out["in_fdr_family"] = out["predictor"].isin(family)
    out["p_fdr"] = np.nan
    for t in out["target"].unique():
        mask = (out["target"] == t) & out["in_fdr_family"]
        out.loc[mask, "p_fdr"] = benjamini_hochberg(out.loc[mask, "p"]).values
    out["fdr_significant"] = out["p_fdr"] < fdr_alpha
```

Add `"in_fdr_family"` to the `cols` list (line 1193–1195), after `"fdr_significant"`, and to the empty-frame column list at lines 1175–1179.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest heavy_machinery/pytests_atypier/test_eda.py -v`
Expected: new tests PASS, all pre-existing `test_eda.py` tests still PASS (the default path must be behaviour-identical apart from the new column).

- [ ] **Step 5: Update the notebook**

In `meningioma-modelling.ipynb`, in the cell that defines `EDA_PREDICTORS` (search for `'midline',`), add directly below the list:

```python
# Redundant variants of predictors already in the sweep: Youden-derived
# dichotomisations of continuous variables, and binary recodes of nominal
# parents. They render as exploratory (uncorrected) and stay out of the
# BH multiplicity family (spec 5.2).
EDA_REDUNDANT_VARIANTS = [
    'tumor_volume_ge15.1', 'adc_value_le0.72', 'max_diameter_cm_ge3.81',
    'edema_volume_ge4.76', 'edema_index_ge0.0617',
    'male_sex', 'irregular_margin', 'skull_base_location',
    'venous_sinus_invasion', 'midline',
]
EDA_FDR_FAMILY = [c for c in EDA_PREDICTORS if c not in EDA_REDUNDANT_VARIANTS]
```

And in the `screen_associations(` call cell add the argument:

```python
assoc = screen_associations(
    df, schema,
    targets=EDA_TARGETS,
    predictors=EDA_PREDICTORS,
    fdr_family=EDA_FDR_FAMILY,
    positive_class=EDA_POSITIVE_CLASS,
    fdr_alpha=0.05,
    output_root=OUTPUT_ROOT,
    )
```

Use `jupyter nbconvert` only in Task 8 — here just edit the JSON source (via `jq` or careful string edit; the cells are small).

- [ ] **Step 6: Split Tab16 in the report renderer**

In `heavy_machinery/modelling_phase/report.py`, `render_eda` (line 2159): find where the associations DataFrame is rendered into the main sweep table (`table_to_html(...)` on the associations view, around lines 2200–2270 — locate by reading the function). Immediately before that render, split:

```python
        if "in_fdr_family" in view.columns:
            fam_mask = view["in_fdr_family"].fillna(True).astype(bool)
            exploratory = view[~fam_mask].drop(columns=["in_fdr_family"])
            view = view[fam_mask].drop(columns=["in_fdr_family"])
            n_tests = int(fam_mask.sum())
        else:
            exploratory, n_tests = view.iloc[0:0], len(view)
```

Render `view` where the full table rendered before, followed by a one-line note and the collapsed block (reuse the same `table_to_html` styling arguments the main table uses):

```python
        body.append(info_box(
            f"FDR family: {n_tests} non-redundant predictors entered the "
            f"Benjamini–Hochberg correction. Redundant variants "
            f"(dichotomisations and binary recodes of predictors already "
            f"tested) are exploratory and uncorrected."))
        if not exploratory.empty:
            body.append(details_block(
                "Exploratory variants — uncorrected, not in the FDR family",
                table_to_html(exploratory)))
```

Adapt variable names to what `render_eda` actually calls its display frame; keep row-classing logic intact for the main table.

- [ ] **Step 7: Add a renderer test**

Append to `heavy_machinery/pytests_atypier/test_report.py` (match its existing import/fixture idiom for calling section renderers; if `render_eda` needs a full `Artifacts`, test the splitting logic by extracting it into a small pure helper `split_fdr_family(view) -> tuple[pd.DataFrame, pd.DataFrame, int]` in `report.py` and test that directly):

```python
def test_split_fdr_family_partitions_rows():
    import report
    view = pd.DataFrame({
        "predictor": ["vol", "vol_ge10"],
        "p": [0.01, 0.02],
        "in_fdr_family": [True, False],
    })
    main, exploratory, n_tests = report.split_fdr_family(view)
    assert list(main["predictor"]) == ["vol"]
    assert list(exploratory["predictor"]) == ["vol_ge10"]
    assert n_tests == 1
    assert "in_fdr_family" not in main.columns


def test_split_fdr_family_backcompat_without_column():
    import report
    view = pd.DataFrame({"predictor": ["vol"], "p": [0.01]})
    main, exploratory, n_tests = report.split_fdr_family(view)
    assert len(main) == 1 and exploratory.empty and n_tests == 1
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_report.py -k fdr_family -v` — FAIL first, then implement `split_fdr_family` as the helper used by Step 6's code, then PASS.

- [ ] **Step 8: Full suite + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q`
Expected: all green.

```bash
git add heavy_machinery/modelling_phase/eda.py heavy_machinery/modelling_phase/report.py heavy_machinery/pytests_atypier/test_eda.py heavy_machinery/pytests_atypier/test_report.py meningioma-modelling.ipynb
git commit -m "feat: restrict BH-FDR to a non-redundant predictor family (spec 5.2)"
```

---

### Task 2: One bootstrap number, printed once (spec 5.3)

**Files:**
- Modify: `heavy_machinery/config/analysis.py` (add constant)
- Modify: `heavy_machinery/modelling_phase/model_validation.py:199,378`
- Modify: `heavy_machinery/threshold_phase/calibration.py:193,219,318`, `heavy_machinery/threshold_phase/combinations.py:326`
- Modify: `heavy_machinery/modelling_phase/report.py` (`render_header`, line 1097) and `heavy_machinery/threshold_phase/threshold_report.py` (`render_header`, line 578)
- Test: `heavy_machinery/pytests_atypier/test_model_validation.py`, `heavy_machinery/pytests_atypier/test_calibration.py`

**Interfaces:**
- Produces: `analysis.BOOTSTRAP_RESAMPLES: int = 1000`, imported everywhere a resample count defaults today. Both report headers print "Bootstrap internal validation: 1000 resamples." exactly once; no other body text or table repeats the number as a hard-coded literal.

- [ ] **Step 1: Write the failing tests**

Append to `test_model_validation.py` (which already does `from model_validation import bootstrap_internal_validation, enrich_streamlit_artifact`):

```python
def test_bootstrap_default_comes_from_shared_config():
    import inspect
    import analysis  # heavy_machinery/config resolves bare per pytest.ini
    for fn in (bootstrap_internal_validation, enrich_streamlit_artifact):
        sig = inspect.signature(fn)
        assert sig.parameters["n_bootstrap"].default == analysis.BOOTSTRAP_RESAMPLES
```

Append to `test_calibration.py` (bare imports, same as every test module):

```python
def test_threshold_bootstrap_matches_shared_config():
    import inspect
    import analysis
    import calibration
    import combinations
    fns = [
        calibration.uncut_model_calibration,   # calibration.py:188, default at :193
        calibration._fitted_calibration,       # calibration.py:213, default at :219
        calibration.cut_model_calibration,     # calibration.py:313, default at :318
        combinations.continuous_model_benchmark,  # combinations.py:321, default at :326
    ]
    for fn in fns:
        sig = inspect.signature(fn)
        assert sig.parameters["n_boot"].default == analysis.BOOTSTRAP_RESAMPLES
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_model_validation.py heavy_machinery/pytests_atypier/test_calibration.py -k shared_config -v`
Expected: FAIL — `analysis` has no attribute `BOOTSTRAP_RESAMPLES`.

- [ ] **Step 2: Implement**

In `heavy_machinery/config/analysis.py`, near the top:

```python
# One bootstrap-resample count for the whole pipeline (spec 5.3): the
# modelling phase and the threshold phase must validate with the same
# number, printed once per report in the Methods block.
BOOTSTRAP_RESAMPLES: int = 1000
```

Then change each default: `n_bootstrap: int = 1000` → `n_bootstrap: int = analysis.BOOTSTRAP_RESAMPLES` (model_validation.py:199,378) and `n_boot: int = 500` → `n_boot: int = analysis.BOOTSTRAP_RESAMPLES` (calibration.py:193,219,318; combinations.py:326), adding the import the way each module already imports config modules (check the top of each file; threshold_phase modules may import config differently than modelling_phase — mirror whatever `from ... import` style is already present).

- [ ] **Step 3: Print once per report**

`report.py` `render_header` (line 1097) — inside the header cards/metadata block add one line (using the existing `card(...)` helper at line 1125 if that is the idiom):

```python
    card("Bootstrap validation", f"{analysis.BOOTSTRAP_RESAMPLES} resamples")
```

`threshold_report.py` `render_header` (line 578) — add the equivalent single line to its header/Methods text:

```python
    f"Bootstrap internal validation: {analysis.BOOTSTRAP_RESAMPLES} resamples."
```

Then grep both report modules for hard-coded `1000`/`500` resample mentions in prose strings and delete/replace them so the number is stated once per report:

Run: `grep -n "resample" heavy_machinery/modelling_phase/report.py heavy_machinery/threshold_phase/threshold_report.py`

- [ ] **Step 4: Run tests**

Run: `python -m pytest heavy_machinery/pytests_atypier -q`
Expected: all green. ⚠️ If any threshold-phase test asserts `n_boot == 500` or pins runtime-sized outputs, update it to reference `analysis.BOOTSTRAP_RESAMPLES` — that is the point of the change, not a regression.

- [ ] **Step 5: Commit**

```bash
git add heavy_machinery/config/analysis.py heavy_machinery/modelling_phase/model_validation.py heavy_machinery/threshold_phase/calibration.py heavy_machinery/threshold_phase/combinations.py heavy_machinery/modelling_phase/report.py heavy_machinery/threshold_phase/threshold_report.py heavy_machinery/pytests_atypier/test_model_validation.py heavy_machinery/pytests_atypier/test_calibration.py
git commit -m "feat: single BOOTSTRAP_RESAMPLES constant, printed once per report (spec 5.3)"
```

---

### Task 3: One event-rate rounding rule (spec 5.4)

The cohort event rate renders as 29.8%, 30% and 31% in different places; 31% is actually the ADC-subset rate shown without its n.

**Files:**
- Modify: `heavy_machinery/modelling_phase/performance_plots.py:243`
- Modify: whichever threshold-phase figure code emits an outcome/event-rate percentage (find with the grep in Step 2)
- Test: `heavy_machinery/pytests_atypier/test_performance_plots.py`

**Interfaces:**
- Produces: every rendered event/outcome rate string is `{rate:.1%}` and, when the denominator differs from the full cohort, is suffixed with `of n={n}`.

- [ ] **Step 1: Failing test**

Append to `test_performance_plots.py`, reusing its `_validation()` helper (which already carries `"decision_curve": {..., "prevalence": 0.3}`) and its subtitle-assertion idiom (`" ".join(t.get_text() for t in ax.texts)`, as in `test_calibration_figure_draws_a_point_per_bin`):

```python
def test_outcome_rate_renders_one_decimal_with_no_false_precision():
    val = _validation()
    val["decision_curve"]["prevalence"] = 105 / 352  # 0.29829...
    fig = pp.decision_curve_figure(val)
    assert fig is not None
    subtitle = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "29.8%" in subtitle
    assert "30%" not in subtitle
    plt.close(fig)
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_performance_plots.py -k one_decimal -v` → FAIL (renders `30%`).

- [ ] **Step 2: Implement**

`performance_plots.py:243`:

```python
    extra = (
        f"outcome rate {float(prevalence):.1%}" if prevalence is not None else None
    )
```

Find the remaining sites:

Run: `grep -rn ":.0%\|:.0f}%" heavy_machinery/modelling_phase heavy_machinery/threshold_phase | grep -iv test`

For each hit that formats an event/outcome/prevalence rate: change to `:.1%`. Where the rate's denominator is a subset (e.g. the ADC risk-curve panel builds its subtitle from the metric's own subset in `heavy_machinery/threshold_phase/risk_curves.py` — locate via the grep), append the subset size, e.g.:

```python
    f"outcome rate {rate:.1%} of n={n_subset}"
```

Leave non-rate percentages (sensitivity, specificity, coverage) untouched in this task — they are spec 3.5, a later plan.

- [ ] **Step 3: Tests + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q` → all green.

```bash
git add heavy_machinery/modelling_phase/performance_plots.py heavy_machinery/threshold_phase heavy_machinery/pytests_atypier/test_performance_plots.py
git commit -m "fix: event rates render at 1 dp with subset n where applicable (spec 5.4)"
```

---

### Task 4: Tab13 non-comparability footnote (spec 5.5)

The threshold report's Uncut-vs-Cut comparison mixes models fitted on different patient counts. Task 2 already harmonises resample counts; the remaining difference (n) gets an explicit footnote.

**Files:**
- Modify: `heavy_machinery/threshold_phase/threshold_report.py` — `render_usefulness` (line 1029), where `data.table("calibration")` is rendered
- Test: `heavy_machinery/pytests_atypier/test_threshold_report.py`

- [ ] **Step 1: Failing test**

Append to `test_threshold_report.py`, using its existing whole-document idiom — `_write_artifacts(tmp_path)` writes the synthetic table set and `_html(tmp_path)` builds the report:

```python
def test_usefulness_carries_noncomparability_footnote(tmp_path):
    _write_artifacts(tmp_path)
    html = _html(tmp_path)
    assert "not directly comparable" in html
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_threshold_report.py -k noncomparability -v` → FAIL.

- [ ] **Step 2: Implement**

In `render_usefulness`, immediately after the HTML for the Uncut/Cut comparison table is assembled, append:

```python
    note = ("<p class=\"footnote\">The uncut and cut model rows are fitted on "
            "different patient counts (each model keeps the patients complete "
            "for its own inputs), so their metrics are not directly comparable "
            "row-to-row; read each against its own n.</p>")
```

and include `{note}` in the returned f-string right below that table. If the `calibration` table has an `n` column, state the counts: build the sentence with `_join([f"{row.model}: n={_int(row.n)}" ...])` instead of the generic clause.

- [ ] **Step 3: Tests + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q` → all green.

```bash
git add heavy_machinery/threshold_phase/threshold_report.py heavy_machinery/pytests_atypier/test_threshold_report.py
git commit -m "fix: non-comparability footnote on uncut-vs-cut comparison (spec 5.5)"
```

---

### Task 5: Flag implausible minima (spec 5.6)

`max_diameter_cm` min 0.2 cm and `tumor_volume` min 0.3 cm³ pass silently today; the report should surface them as data-quality warnings.

**Files:**
- Modify: `heavy_machinery/modelling_phase/report.py` — `render_dda` (line 1545), after the continuous table renders (`art.dda_continuous`)
- Test: `heavy_machinery/pytests_atypier/test_report.py`

**Interfaces:**
- Produces: `data_quality_warnings(dda_continuous: pd.DataFrame) -> list[str]` in `report.py`, unit-testable, rendered via the existing `warning_box`.

- [ ] **Step 1: Failing test**

```python
def test_data_quality_flags_implausible_minima():
    import report
    dda = pd.DataFrame({
        "column": ["max_diameter_cm", "tumor_volume", "age"],
        "min": [0.2, 0.3, 18.0],
    })
    msgs = report.data_quality_warnings(dda)
    assert len(msgs) == 2
    assert any("0.2" in m and "diameter" in m.lower() for m in msgs)
    assert any("0.3" in m and "volume" in m.lower() for m in msgs)


def test_data_quality_silent_when_plausible():
    import report
    dda = pd.DataFrame({"column": ["max_diameter_cm"], "min": [1.4]})
    assert report.data_quality_warnings(dda) == []
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_report.py -k data_quality -v` → FAIL.

- [ ] **Step 2: Implement**

In `report.py` (near the other small helpers, e.g. below `warning_box`):

```python
# Plausibility floors for measured tumour size (spec 5.6): values below these
# are almost certainly unit or transcription errors, not real surgical lesions.
_PLAUSIBILITY_FLOORS: dict[str, tuple[float, str, str]] = {
    "max_diameter_cm": (0.5, "cm", "maximum diameter"),
    "tumor_volume": (0.5, "cm³", "tumour volume"),
}


def data_quality_warnings(dda_continuous: pd.DataFrame | None) -> list[str]:
    """Messages for continuous minima below hard plausibility floors."""
    if dda_continuous is None or dda_continuous.empty:
        return []
    msgs: list[str] = []
    idx = dda_continuous.set_index("column")
    for col, (floor, unit, label) in _PLAUSIBILITY_FLOORS.items():
        if col not in idx.index:
            continue
        mn = pd.to_numeric(pd.Series([idx.loc[col, "min"]]), errors="coerce").iloc[0]
        if pd.notna(mn) and mn < floor:
            msgs.append(
                f"Data-quality note: smallest recorded {label} is {mn:g} {unit} "
                f"— implausibly small; verify the source records before "
                f"publication.")
    return msgs
```

In `render_dda`, right after the continuous table is appended:

```python
    for msg in data_quality_warnings(art.dda_continuous):
        uni.append(warning_box(msg))
```

(`uni` is the univariate-section list already being built there; keep whatever list the continuous table actually lands in.)

- [ ] **Step 3: Tests + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q` → all green.

```bash
git add heavy_machinery/modelling_phase/report.py heavy_machinery/pytests_atypier/test_report.py
git commit -m "feat: flag implausible size minima in the DDA section (spec 5.6)"
```

---

### Task 6: Define "Share of range" (spec 5.7)

**Files:**
- Modify: `heavy_machinery/threshold_phase/threshold_report.py` — `render_usefulness`, below the table built at line 1076
- Test: `heavy_machinery/pytests_atypier/test_threshold_report.py`

- [ ] **Step 1: Failing test** (same whole-document idiom as Task 4):

```python
def test_share_of_range_defined_in_footnote(tmp_path):
    _write_artifacts(tmp_path)
    html = _html(tmp_path)
    assert "beats both treating everyone and treating no one" in html
```

Run → FAIL.

- [ ] **Step 2: Implement** — append after that table's HTML:

```python
    share_note = ("<p class=\"footnote\">Share of range: the fraction of the "
                  "tested decision-threshold spectrum over which the strategy "
                  "beats both treating everyone and treating no one.</p>")
```

and interpolate it into the returned section HTML directly under the table.

- [ ] **Step 3: Tests + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q` → all green.

```bash
git add heavy_machinery/threshold_phase/threshold_report.py heavy_machinery/pytests_atypier/test_threshold_report.py
git commit -m "docs: footnote defining Share of range (spec 5.7)"
```

---

### Task 7: Define the evidence vocabulary (spec 5.8)

`fragile`, `pass`, `fail`, `survives` appear in threshold Tab11/Tab12 with no key. The definitions already exist in `heavy_machinery/threshold_phase/evidence.py:27-80`; surface them as a footnote where the tables render.

**Files:**
- Modify: `heavy_machinery/threshold_phase/threshold_report.py` — `render_evidence` (line 960) and `render_stability` (line 894, whose details block shows the per-dataset table)
- Test: `heavy_machinery/pytests_atypier/test_threshold_report.py`

- [ ] **Step 1: Failing test**

```python
def test_evidence_vocabulary_footnote_present(tmp_path):
    _write_artifacts(tmp_path)
    html = _html(tmp_path)
    assert "necessary criteria pass" in html
```

Run → FAIL. (The word `fragile` already appears in table cells — assert on the definition phrase, never the word alone.)

- [ ] **Step 2: Implement** — module-level constant in `threshold_report.py`:

```python
_EVIDENCE_KEY = (
    "<p class=\"footnote\">Vocabulary: <em>pass</em>/<em>fail</em> — the "
    "criterion is met / not met; <em>survives</em> — still significant after "
    "the named multiple-testing correction; <em>fragile</em> — all necessary "
    "criteria pass but no robustness criterion does, so the finding depends "
    "on analysis choices that were not forced.</p>")
```

Append `{_EVIDENCE_KEY}` once in `render_evidence`'s returned HTML, directly under the verdict table. Cross-check the wording against the grade rules in `evidence.py:27-80` and adjust so the footnote states exactly what the code implements (e.g. the `fragile` rule at line 40 has an **or** branch — include it).

- [ ] **Step 3: Tests + commit**

Run: `python -m pytest heavy_machinery/pytests_atypier -q` → all green.

```bash
git add heavy_machinery/threshold_phase/threshold_report.py heavy_machinery/pytests_atypier/test_threshold_report.py
git commit -m "docs: define pass/fail/fragile/survives where verdicts render (spec 5.8)"
```

---

### Task 8: Pipeline re-run — propagate the midline fix and everything above (spec 5.1)

The cleaning notebook already derives `midline = (side == 'midline')` (fixed in commit `f1e4bab`), but `output/datasets/`, both reports, and every downstream table still carry the inverted flag. One full re-run propagates the midline fix and Tasks 1–7 together.

⚠️ Long-running: the modelling notebook runs R MICE with m=20 and 1000-resample bootstraps; the thresholder now also runs 1000 resamples (Task 2). Expect this to take a while; run the notebooks sequentially, never in parallel (modelling reads cleaning's handoff).

**Files:**
- Test: `heavy_machinery/pytests_atypier/test_dataset_handoff.py`
- Regenerated (not hand-edited): `output/datasets/*`, `output/dda|eda|inferential|missingness/*`, `output/report/report.html`, `output/thresholds/*`

- [ ] **Step 1: Write the acceptance test (fails against stale data)**

Append to `test_dataset_handoff.py` (match how it locates the parquet — it already reads `output/datasets/`):

```python
def test_midline_flag_means_midline():
    df = pd.read_parquet("output/datasets/unimputed_df.parquet")
    derived = (df["side"] == "midline")
    both = df["midline"].notna() & df["side"].notna()
    assert (df.loc[both, "midline"].astype(bool) == derived[both]).all()
    # cohort fact: 35 midline vs 317 lateralised of 352
    assert int(df["midline"].sum()) == 35
```

Run: `python -m pytest heavy_machinery/pytests_atypier/test_dataset_handoff.py -k midline_flag -v`
Expected: FAIL — stale parquet has `midline.sum() == 317`. This red test is the acceptance gate for the re-run; do not adjust it to pass.

- [ ] **Step 2: Re-run the cleaning notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace meningioma-cleaning.ipynb
```

Then: `python -m pytest heavy_machinery/pytests_atypier/test_dataset_handoff.py -k midline_flag -v` → PASS.

- [ ] **Step 3: Re-run the modelling notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=None meningioma-modelling.ipynb
```

Requires the R `mice` toolchain (`heavy_machinery/scripts/run_mice.R`); if R fails, stop and report — the RF fallback is a labelled sensitivity analysis only, never a substitute.

- [ ] **Step 4: Re-run the thresholder notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=None meningioma-thresholder.ipynb
```

- [ ] **Step 5: Full suite**

Run: `python -m pytest heavy_machinery/pytests_atypier -q`
Expected: all green. Failures here mean a downstream artifact no longer matches an assertion — investigate whether the assertion pinned the *old, wrong* midline numbers (update the test) or a task above broke something (fix the code).

- [ ] **Step 6: Verify the rendered reports**

Checks against `output/report/report.html` and `output/thresholds/threshold_report.html` (grep the HTML; view in browser for spot checks — remember `qlmanage` crops SVGs, use a browser):

```bash
grep -c "Bootstrap" output/report/report.html output/thresholds/threshold_report.html
grep -o "29.8%" output/report/report.html | head -1
grep -o "Share of range" output/thresholds/threshold_report.html | head -1
grep -o "not directly comparable" output/thresholds/threshold_report.html | head -1
grep -o "Exploratory variants" output/report/report.html | head -1
grep -o "Data-quality note" output/report/report.html | head -1
```

Expected: each phrase present; "1000 resamples" appears in both; the midline row of the diagnostic tables now shows prevalence 35/352 (~9.9%) with sensitivity/specificity flipped relative to the spec's quoted 97.1%/13.0%.

- [ ] **Step 7: Commit**

```bash
git add meningioma-cleaning.ipynb meningioma-modelling.ipynb meningioma-thresholder.ipynb heavy_machinery/pytests_atypier/test_dataset_handoff.py
git add -u output
git commit -m "fix: re-run pipeline — midline flag now flags midline tumours (spec 5.1)"
```

(`git add -u output` stages only output files git already tracks; untracked artifacts stay untracked, matching current repo convention.)

---

## Follow-on plans (not in this document)

The remaining spec sections build on these fixes and need three further plans, in order:

1. **Rendering infrastructure** — §0 three-layer `render(obj, legend, explainer)`, §4 single `LABELS` dict (name → display → unit → abbreviation), stable Fig/Table numbering, `publication=True` flag, en-GB locale. This is the multiplier: ~40% of remaining findings fall out of the LABELS dict alone.
2. **Figure layer** — §2: strip in-figure titles/stat strips/annotations into legends and explainers, two width classes, panel letters, per-figure reworks (R_202 split, R_174 label, ROC AUC CIs, calibration axes, DCA harmonisation, R_132/R_173 restructuring), EPS/TIFF export buttons, greyscale proofs.
3. **Tables, de-duplication, additions** — §3 (captions, footnotes, precision, exports, Tab17/Tab32 merge), §6 (main/supplementary/internal tagging, collapse R_000–R_129, one primary model expanded), §7 (Table 2 by WHO grade, STARD flow figure, manifest).

Two decisions in those plans belong to the user, flag before planning them: **which model is the primary one** (spec 6.5 keeps only one expanded), and **whether second-reader data exists** for interobserver κ/ICC (spec 7.3 — cannot be computed from the current single-reader dataset if no second reading was captured).
