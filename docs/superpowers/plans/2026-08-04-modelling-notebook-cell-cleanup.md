# Modelling Notebook Cell Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip logic out of `meningioma-modelling.ipynb` code cells so each cell holds only inputs, function calls, and comments — moving every displaced line into `heavy_machinery/` with its explanatory comments intact, and fixing the `_panel_key` substring bug the move exposes.

**Architecture:** Three module additions absorb what the notebook currently computes inline: `validation.validate_unimputed_handoff` (path building + validate + report), a `verbose` branch inside `dataset_handoff.load_modelling_handoff` (imputation-method reporting), and three new `marker_panel` helpers (`panel_key`, `load_panel_artifacts`, `model_links_from_variants`) wired into `run_marker_panel` so the panel discovers its own artifacts, MICE draws, and paper links. The notebook then edits down to calls. Every number is unchanged; the only intended output change is two model **names**, from the bug fix in Task 4.

**Tech Stack:** Python 3, pandas, pandera, pytest, Jupyter (`jupyter nbconvert --execute`). Modules under `heavy_machinery/{cleaning_phase,modelling_phase,config}/`; tests under `heavy_machinery/pytests_atypier/`.

## Global Constraints

- **Byte-identical output, with exactly one documented exception.** Every file under `output/` must be unchanged after the refactor *except* the two experimental model names, which the Task 4 bug fix corrects. Byte-identical output is the house standard for refactors here (see `README.md` §changelog: "all thirteen output CSVs byte-for-byte unchanged"), so the exception is enumerated rather than waved through. The complete allowed diff is:

  | File | Column | Before | After |
  |---|---|---|---|
  | `output/panel/tables/10_model_vs_single.csv` | `model` | `experimental_1` | `experimental_model_1` |
  | `output/panel/tables/10_model_vs_single.csv` | `model` | `experimental_2` | `experimental_model_2` |
  | `output/panel/tables/13_model_reading_view.csv` | `Model` | `Experimental 1` | `Experimental model 1` |
  | `output/panel/tables/13_model_reading_view.csv` | `Model` | `Experimental 2` | `Experimental model 2` |
  | `output/report/report.html` | — | the same four strings | the same four strings |

  **Any other difference is a regression.** No number, no row order, no figure, no other file may move. Verified by `diff -r` in Task 6.
- **Run everything from the repo root** `/Users/andriszaguzovs/TheLibraryOfCode/meningioma-atypier/`. `pytest.ini` sets `testpaths = heavy_machinery/pytests_atypier` and the `pythonpath` entries that make flat sibling imports (`import marker_panel`) work.
- **Modules use flat sibling imports** (`from inferential import ...`, `import marker_panel as mp`), not `heavy_machinery.`-prefixed ones. The notebook uses the `heavy_machinery.`-prefixed form. Do not mix them.
- **No new dependencies.** Nothing gets added to `requirements.txt`.
- **Comments are moved, never deleted.** Every explanatory comment and docstring removed from a notebook cell must reappear on the module code that took over the job. Commented-out lines in the notebook (e.g. `#df.head(3)`, `#assoc`, the `#'progesterone_pos'` entries inside the predictor lists) are user scratchpad — leave them exactly where they are.
- **What may stay in a notebook code cell:** lines that encode an input the user chooses (column lists, target lists, titles, `NON_IMAGING`), function calls, and comments. Nothing else — no `def`, no `if`/`else`, no `for`, no comprehension, no f-string built for printing.
- **Imports stay in cell 2** in the explicit `from X import y` form (user decision), minus the dead ones. Do not collapse them into a namespace object.

---

## File Structure

| File | Change | Responsibility after the change |
|---|---|---|
| `heavy_machinery/cleaning_phase/validation.py` | Modify — add `validate_unimputed_handoff` | Also owns the one-call path-build + validate + report for the unimputed EDA cohort, mirroring the existing `validate_imputed_frames` |
| `heavy_machinery/cleaning_phase/dataset_handoff.py` | Modify — add `verbose` to `load_modelling_handoff` | Also reports what it loaded and which imputation method the next stage will use |
| `heavy_machinery/modelling_phase/marker_panel.py` | Modify — add `panel_key`, `load_panel_artifacts`, `model_links_from_variants`, `_panel_draws`; extend `run_marker_panel` | Also discovers its own model artifacts, MICE draws and paper links from `output_root` + the variant list, and keys them correctly |
| `heavy_machinery/pytests_atypier/test_validation.py` | **Create** — 3 tests | First test file for `validation.py`; nothing currently covers it |
| `heavy_machinery/pytests_atypier/test_dataset_handoff.py` | Modify — add 3 tests | |
| `heavy_machinery/pytests_atypier/test_marker_panel.py` | Modify — add 8 tests | Includes the regression test for the `_model` substring bug |
| `CHANGES.md`, `README.md` | Modify — record the model rename | Task 6, since the rename is only confirmed once the notebook has run |
| `meningioma-modelling.ipynb` | Modify — cells 2, 4, 5, 22 | Inputs, calls, comments |

---

### Task 1: Capture the baseline

Nothing can be verified without a copy of the current `output/` tree. This must happen before any source file is touched.

**Files:**
- Create: `/tmp/output_baseline/` (outside the repo, so `.gitignore` is irrelevant)

- [ ] **Step 1: Confirm the working tree is clean**

```bash
git -C /Users/andriszaguzovs/TheLibraryOfCode/meningioma-atypier status --porcelain
```

Expected: empty output. If it is not empty, stop and report to the user — a dirty tree means the baseline may not correspond to committed code.

- [ ] **Step 2: Confirm `output/` is populated**

```bash
find output -type f | wc -l
```

Expected: a few hundred files. If it is 0 or very small, stop and tell the user the notebooks need a full run first — there is nothing to diff against.

- [ ] **Step 3: Copy the tree**

```bash
rm -rf /tmp/output_baseline && cp -R output /tmp/output_baseline
```

- [ ] **Step 4: Verify the copy is identical**

```bash
diff -r /tmp/output_baseline output && echo "BASELINE OK"
```

Expected: `BASELINE OK` with no diff lines above it.

- [ ] **Step 5: Record the baseline file count for Task 6**

```bash
find /tmp/output_baseline -type f | wc -l > /tmp/output_baseline_count.txt && cat /tmp/output_baseline_count.txt
```

Expected: prints a number. No commit for this task — nothing in the repo changed.

---

### Task 2: `validate_unimputed_handoff` in validation.py

Absorbs notebook cell 5, which builds a path, validates, and prints — three lines of plumbing the module should own. `validate_imputed_frames` in the same file already prints its own result, so this makes the two halves symmetric.

**Files:**
- Modify: `heavy_machinery/cleaning_phase/validation.py` (append after `load_schema_validation`, around line 54)
- Create: `heavy_machinery/pytests_atypier/test_validation.py`

**Interfaces:**
- Consumes: `load_schema_validation(path)` — existing, same file.
- Produces: `validate_unimputed_handoff(df: pd.DataFrame, output_root: Path | str = "output") -> pa.DataFrameSchema`. Returns the loaded schema so the caller can reuse it for the imputed draws later. Task 5 (notebook) depends on this exact name and return type.

- [ ] **Step 1: Write the failing tests**

Nothing currently covers `validation.py` — create `heavy_machinery/pytests_atypier/test_validation.py` with exactly this content. The flat `import validation` form is what `pytest.ini`'s `pythonpath` supports; do not write `from heavy_machinery.cleaning_phase import validation`.

```python
"""Pandera validation of the cleaning → modelling handoff."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from validation import validate_unimputed_handoff


def _write_schema(output_root) -> pa.DataFrameSchema:
    schema = pa.DataFrameSchema({
        "age": pa.Column("float64", nullable=True),
        "grade": pa.Column("int64", nullable=True),
    }, strict=True)
    path = output_root / "cleaning" / "schema_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema.to_json(path, indent=4)
    return schema


def test_validate_unimputed_handoff_loads_validates_and_returns_the_schema(
    tmp_output, capsys,
):
    """One call replaces the notebook's path-build + validate + print.

    The schema comes back because the modelling notebook validates the MICE
    draws against the same object later on.
    """
    _write_schema(tmp_output)
    df = pd.DataFrame({"age": [40.0, 50.0], "grade": [1, 2]})

    returned = validate_unimputed_handoff(df, tmp_output)

    assert isinstance(returned, pa.DataFrameSchema)
    assert set(returned.columns) == {"age", "grade"}
    assert "✅" in capsys.readouterr().out


def test_validate_unimputed_handoff_rejects_a_frame_that_breaks_the_schema(tmp_output):
    """A stray column is exactly what a strict handoff schema exists to catch."""
    _write_schema(tmp_output)
    df = pd.DataFrame({"age": [40.0], "grade": [1], "surprise": ["x"]})

    with pytest.raises(pa.errors.SchemaErrors):
        validate_unimputed_handoff(df, tmp_output)


def test_validate_unimputed_handoff_raises_when_the_json_is_missing(tmp_output):
    """A missing schema artifact means cleaning never finished. Say so."""
    with pytest.raises(FileNotFoundError):
        validate_unimputed_handoff(pd.DataFrame({"age": [1.0]}), tmp_output)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_validation.py -v
```

Expected: FAIL at collection — `ImportError: cannot import name 'validate_unimputed_handoff' from 'validation'`.

- [ ] **Step 3: Write the implementation**

Insert into `heavy_machinery/cleaning_phase/validation.py`, immediately after `load_schema_validation` and before `validate_imputed_frames`:

```python
def validate_unimputed_handoff(
    df: pd.DataFrame,
    output_root: str | Path = "output",
    ) -> pa.DataFrameSchema:
    """Pandera-validate the unimputed EDA cohort and return the schema.

    The schema is returned rather than discarded because the same object
    validates every MICE draw in the modelling notebook's §04, immediately
    before fitting.
    """
    schema_validation = load_schema_validation(Path(output_root) / "cleaning" / "schema_validation.json")
    schema_validation(df, lazy=True)
    print("✅ Pandera validated unimputed handoff (EDA cohort)")
    return schema_validation
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_validation.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full suite to check nothing regressed**

```bash
python -m pytest -q
```

Expected: all pass. The suite was fully green before this change, so any failure here is caused by this change — do not proceed past a failure.

- [ ] **Step 6: Commit**

```bash
git add heavy_machinery/cleaning_phase/validation.py heavy_machinery/pytests_atypier/test_validation.py
git commit -m "refactor: fold the notebook's unimputed-handoff validation into validation.py"
```

---

### Task 3: Handoff reporting in dataset_handoff.py

Absorbs notebook cell 4's `if IMPUTATION_METHOD == "mice": ... else: ...` block — the only branch left in the notebook.

**Files:**
- Modify: `heavy_machinery/cleaning_phase/dataset_handoff.py:132-141`
- Test: `heavy_machinery/pytests_atypier/test_dataset_handoff.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_modelling_handoff(output_root="output", *, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, ColSpec], str]`. Return type is unchanged; only the keyword and the printing are new.

- [ ] **Step 1: Write the failing tests**

Append to `heavy_machinery/pytests_atypier/test_dataset_handoff.py`. The setup below is copied from the existing `test_load_modelling_handoff` (line 138) and `test_detect_imputation_method_mice` (line 38) — every helper it calls is already imported at the top of that file, so no new imports are needed.

```python
def _stage_simple_handoff(tmp_output, tiny_df, tiny_schema) -> None:
    """A complete simple-imputation handoff on disk, as cleaning §16 leaves it."""
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / SIMPLE_MODELING_DATASET_NAME,
        context=SIMPLE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    export_schema_summary(tiny_schema, tmp_output)


def test_load_modelling_handoff_reports_what_it_loaded(tmp_output, tiny_df, tiny_schema, capsys):
    """The notebook used to print this itself, with an if/else on the method.

    A branch in a notebook cell is a branch nobody tests. It lives here now.
    """
    _stage_simple_handoff(tmp_output, tiny_df, tiny_schema)

    df, schema, method = load_modelling_handoff(tmp_output)
    out = capsys.readouterr().out

    assert method == "simple"
    assert f"{len(df)} rows" in out
    assert f"{len(schema)} schema columns" in out
    assert "one imputed parquet" in out


def test_load_modelling_handoff_reports_the_mice_branch(tmp_output, tiny_df, tiny_schema, capsys):
    """MICE means the inferential stage pools draws. The reader is told which."""
    prepare_datasets_dir(tmp_output)
    stage_unimputed_dataset(tiny_df, tmp_output)
    _save_dataset_parquet(
        tiny_df,
        _datasets_dir(tmp_output) / MICE_MODELING_DATASET_NAME,
        context=MICE_MODELING_DATASET_NAME,
        dtype_reference=tiny_df,
    )
    export_schema_summary(tiny_schema, tmp_output)

    _, _, method = load_modelling_handoff(tmp_output)

    assert method == "mice"
    assert "MICE" in capsys.readouterr().out


def test_load_modelling_handoff_can_be_quiet(tmp_output, tiny_df, tiny_schema, capsys):
    """Callers that are not a notebook should not have to eat the printing."""
    _stage_simple_handoff(tmp_output, tiny_df, tiny_schema)

    load_modelling_handoff(tmp_output, verbose=False)

    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_dataset_handoff.py -k reports_what_it_loaded -v
```

Expected: FAIL — the assertion on `"rows" in out` fails because nothing is printed yet.

- [ ] **Step 3: Write the implementation**

Replace `heavy_machinery/cleaning_phase/dataset_handoff.py:132-141` with:

```python
def load_modelling_handoff(
    output_root: Path | str = "output",
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, ColSpec], str]:
    """Load unimputed cohort, committed schema, and imputation method for modelling.

    ``verbose`` reports the row/column counts and which imputation method the
    inferential stage will therefore use — MICE pools several imputed draws,
    simple reads one parquet. The modelling notebook used to branch on this
    itself; a branch in a notebook cell is a branch nobody tests.
    """
    output_root = Path(output_root)
    imputation_method = detect_imputation_method(output_root)
    df = load_unimputed_dataset(output_root)
    schema = load_schema_from_handoff(output_root)
    validate_schema_against_frame(df, schema)
    if verbose:
        print(f"Loaded unimputed cohort: {len(df)} rows, {len(schema)} schema columns")
        if imputation_method == "mice":
            print("Imputation method: MICE — inferential stage (§04) pools multiple imputed draws.")
        else:
            print("Imputation method: simple — inferential stage (§04) uses one imputed parquet.")
    return df, schema, imputation_method
```

The two `print` strings are copied verbatim from notebook cell 4, including the `§04` references and the em dashes. Do not reword them — Task 6 compares nothing about stdout, but the user reads these lines every run.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_dataset_handoff.py -v
```

Expected: all pass, including the three new ones.

- [ ] **Step 5: Run the full suite**

```bash
python -m pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add heavy_machinery/cleaning_phase/dataset_handoff.py heavy_machinery/pytests_atypier/test_dataset_handoff.py
git commit -m "refactor: move the handoff report out of the modelling notebook"
```

---

### Task 4: Marker panel discovers its own artifacts, draws and links

The largest cleanup. Notebook cell 22 currently defines a function, runs two dict comprehensions, and imports two extra modules — 53 lines that all exist to hand `run_marker_panel` three arguments it can work out itself.

**🔧 This task also fixes the `_panel_key` bug (user-approved).** The notebook's `_panel_key` uses `name.replace("_model", "")`, which strips *every* occurrence of `_model`, not just the trailing one. On the artifact stem `high_grade_experimental_model_1_model` that yields `experimental_1`, and on the variant id `experimental_model_1` it also yields `experimental_1` — the two sides meet only because both are mangled the same way. Any model id containing `_model` anywhere else would silently mismatch instead.

`inferential._artifact_model_id` already does this job correctly: it strips `_model` only as a suffix, then the `{target}_` prefix. Check it against both kinds of input:

| input | `_artifact_model_id(input, "high_grade")` |
|---|---|
| `high_grade_yao_et_al_2022_model` (artifact stem) | `yao_et_al_2022` |
| `yao_et_al_2022` (variant id) | `yao_et_al_2022` |
| `high_grade_experimental_model_1_model` (artifact stem) | `experimental_model_1` |
| `experimental_model_1` (variant id) | `experimental_model_1` |

Both sides still meet, and now they meet on the true id. **The visible consequence:** `marker_panel._model_label` turns the key into the name printed in the report, so the two experimental rows change from "Experimental 1" / "Experimental 2" to "Experimental model 1" / "Experimental model 2". That is the entire allowed output diff for this whole plan — see Global Constraints. Nothing numeric moves: the keys change on both the artifact side and the link side simultaneously, and row order comes from `sorted(glob(...))` over filenames, which the rename does not touch.

Reuse `_artifact_model_id` rather than writing a third copy. Importing an underscore-private across sibling modules is established practice in this codebase — `dataset_handoff.py:14-25` imports six of them from `missingness_resolution`.

**Files:**
- Modify: `heavy_machinery/modelling_phase/marker_panel.py` (add three helpers above `run_marker_panel` at line 949; extend `run_marker_panel`'s signature and body)
- Test: `heavy_machinery/pytests_atypier/test_marker_panel.py`

**Interfaces:**
- Consumes: `model_calculator.load_model_artifact(path) -> dict`; `missingness_resolution.load_imputed_frames(output_root) -> list[pd.DataFrame]`; `inferential.normalize_inferential_variants(variants=..., default_target=...) -> list[InferentialModelVariant]` (accepts raw 5-tuples `(id, title, link, target, [predictors])` as well as `InferentialModelVariant` objects, and exposes `.model_id` / `.link`).
- Produces:
  - `panel_key(name: str, target: str) -> str`
  - `load_panel_artifacts(output_root: Path | str, target: str) -> dict[str, dict]`
  - `model_links_from_variants(variants: Sequence, target: str) -> dict[str, str]`
  - `run_marker_panel(..., variants: Sequence = (), artifacts: dict | None = None, draws: Sequence | None = None, model_links: Mapping | None = None, ...)` — `artifacts=None` and `draws=None` now mean "load them from `output_root`"; `model_links=None` means "derive from `variants`". Task 5 (notebook) depends on the `variants=` keyword.

- [ ] **Step 1: Write the failing tests**

Append to `heavy_machinery/pytests_atypier/test_marker_panel.py`. `count_frame()`, `panel_accuracy_table()`, `TARGET` and the `tmp_output` fixture already exist in that file / `conftest.py`.

```python
# --------------------------------------------------------------------------
# Self-discovery: artifacts, draws and links
# --------------------------------------------------------------------------
def _write_artifact(art_dir, stem: str) -> None:
    """Smallest artifact `load_model_artifact` will accept."""
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / f"{stem}.json").write_text(json.dumps({
        "model_name": stem,
        "target": TARGET,
        "coefficients": {"const": -1.0, "sign_0": 0.5},
        "features": [{"name": "sign_0", "kind": "binary"}],
    }), encoding="utf-8")


def test_panel_key_maps_an_artifact_stem_and_a_variant_id_onto_one_key():
    """An artifact filename and the variant id that produced it must agree."""
    assert mp.panel_key("high_grade_yao_et_al_2022_model", TARGET) == "yao_et_al_2022"
    assert mp.panel_key("yao_et_al_2022", TARGET) == "yao_et_al_2022"
    assert mp.panel_key("high_grade_experimental_model_1_model", TARGET) == "experimental_model_1"
    assert mp.panel_key("experimental_model_1", TARGET) == "experimental_model_1"


def test_panel_key_strips_model_only_as_a_suffix():
    """The regression this replaced: `.replace()` stripped every occurrence.

    `experimental_model_1` used to collapse to `experimental_1`, losing part
    of the real id. It matched anyway only because the artifact stem was
    mangled identically — an id with `_model` anywhere else would not have
    been so lucky.
    """
    assert mp.panel_key("high_grade_model_free_zone_model", TARGET) == "model_free_zone"
    assert mp.panel_key("model_free_zone", TARGET) == "model_free_zone"


def test_load_panel_artifacts_reads_the_model_artifact_directory(tmp_output):
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    artifacts = mp.load_panel_artifacts(tmp_output, TARGET)
    assert set(artifacts) == {"yao_et_al_2022"}
    assert artifacts["yao_et_al_2022"]["coefficients"]["const"] == -1.0


def test_load_panel_artifacts_is_empty_when_nothing_has_been_fitted(tmp_output):
    """A panel run before the inferential stage is not an error."""
    assert mp.load_panel_artifacts(tmp_output, TARGET) == {}


def test_model_links_from_variants_keeps_papers_and_drops_our_own():
    """Experimental variants carry an empty link and must not get a citation."""
    variants = [
        ("yao_et_al_2022", "Yao et al. 2022", "https://example.org/yao",
         TARGET, ["sign_0"]),
        ("experimental_model_1", "model 1", "", TARGET, ["sign_1"]),
    ]
    links = mp.model_links_from_variants(variants, TARGET)
    assert links == {"yao_et_al_2022": "https://example.org/yao"}


def test_run_marker_panel_finds_its_own_artifacts_and_links(tmp_output):
    """Passing `variants` replaces the notebook's two dict comprehensions."""
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
        variants=[("yao_et_al_2022", "Yao et al. 2022",
                   "https://example.org/yao", TARGET, ["sign_0"])],
    )
    models = tables["10_model_vs_single"]
    assert set(models["model"]) == {"yao_et_al_2022"}
    assert models["source_link"].iloc[0] == "https://example.org/yao"


def test_run_marker_panel_still_honours_artifacts_passed_in_explicitly(tmp_output):
    """An explicit empty dict means empty, not 'go and look'."""
    _write_artifact(tmp_output / "inferential" / "model_artifacts",
                    f"{TARGET}_yao_et_al_2022_model")
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40, artifacts={},
    )
    assert tables["10_model_vs_single"].empty


def test_run_marker_panel_survives_a_cohort_with_no_mice_draws(tmp_output):
    """Simple imputation leaves no MICE directory. That is not a crash."""
    tables = mp.run_marker_panel(
        count_frame(), target=TARGET, accuracy_table=panel_accuracy_table(),
        output_root=tmp_output, n_boot=40,
    )
    assert "11_imputation_stability" in tables
```

Add `import json` to the top of `test_marker_panel.py` if it is not already there.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -k "panel_key or load_panel_artifacts or model_links_from_variants or finds_its_own or explicitly or no_mice_draws" -v
```

Expected: FAIL — `AttributeError: module 'marker_panel' has no attribute 'panel_key'` for most, and a `TypeError` about an unexpected keyword `variants` for the run tests.

- [ ] **Step 3: Add the three helpers**

Insert into `heavy_machinery/modelling_phase/marker_panel.py`, immediately above `def run_marker_panel(` (currently line 949, after `_write_table`):

```python
def panel_key(name: str, target: str) -> str:
    """Artifact filenames and variant ids, reduced to the same key.

    ``high_grade_experimental_model_1_model.json`` becomes
    ``experimental_model_1``, so the variant id ``experimental_model_1`` has
    to lose the same affixes or the two never meet. The key is also the model
    name :func:`_model_label` prints in the report.

    Delegates to :func:`inferential._artifact_model_id` rather than doing its
    own string surgery. The notebook's version used ``str.replace``, which
    strips ``_model`` wherever it appears rather than only as a suffix; the
    two sides agreed only because both were mangled the same way, and an id
    containing ``_model`` in the middle would have silently mismatched.
    """
    from inferential import _artifact_model_id

    return _artifact_model_id(name, target)


def load_panel_artifacts(output_root: Path | str, target: str) -> dict[str, dict]:
    """Every fitted model artifact under ``output/inferential/model_artifacts/``.

    Empty when the inferential stage has not run — a panel without models
    still answers aim 1, so this is a missing section, not an error.
    """
    from model_calculator import load_model_artifact

    art_dir = Path(output_root) / "inferential" / "model_artifacts"
    if not art_dir.exists():
        return {}
    return {
        panel_key(path.stem, target): load_model_artifact(path)
        for path in sorted(art_dir.glob("*_model.json"))
    }


def model_links_from_variants(variants: Sequence, target: str) -> dict[str, str]:
    """The paper each published predictor set came from, keyed like the artifacts.

    So the model comparison table links out to what it is being compared
    against. Our own experimental variants carry an empty link and get no
    citation — inventing one is worse than leaving the cell blank.
    """
    from inferential import normalize_inferential_variants

    return {
        panel_key(var.model_id, target): var.link
        for var in normalize_inferential_variants(variants=list(variants),
                                                  default_target=target)
        if var.link
    }


def _panel_draws(output_root: Path | str) -> list[pd.DataFrame]:
    """The MICE draws, or none when the cohort was filled by simple imputation.

    The draws are a stability check, not an input to any published number, so
    a cohort without them loses one table rather than the whole panel.
    """
    from missingness_resolution import load_imputed_frames

    manifest = Path(output_root) / "missingness" / "mice" / "manifest.json"
    if not manifest.exists():
        return []
    return load_imputed_frames(output_root)
```

The imports are function-local on purpose: `marker_panel` is imported by `report.py`, and pulling `model_calculator` and `missingness_resolution` in at module scope would widen an already load-bearing import graph.

- [ ] **Step 4: Verify the MICE manifest path is right**

The `_panel_draws` guard must point at the directory `load_imputed_frames` actually reads.

```bash
grep -n "_mice_dataset_dir" -A 6 heavy_machinery/cleaning_phase/missingness_resolution.py | head -20
```

Correct the path inside `_panel_draws` to match what `_mice_dataset_dir` returns. Then confirm against the live tree:

```bash
ls output/missingness/mice/manifest.json
```

Expected: the file exists (the README lists `missingness/mice/manifest.json` as a key artifact). If the real location differs, use the real one.

- [ ] **Step 5: Wire the helpers into `run_marker_panel`**

Change the signature (currently `marker_panel.py:949-962`) — three defaults move from "nothing" to "work it out":

```python
def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    variants: Sequence = (),
    artifacts: dict[str, dict] | None = None,
    draws: Sequence[pd.DataFrame] | None = None,
    model_links: Mapping[str, str] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    draw_n_boot: int = DEFAULT_DRAW_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> dict[str, pd.DataFrame]:
```

Add this paragraph to the end of the existing docstring, before the closing `"""`:

```
    ``artifacts``, ``draws`` and ``model_links`` left at ``None`` are found
    rather than passed: the fitted models under ``output_root``, the MICE
    draws beside them, and the paper links carried by ``variants``. A caller
    that passes an empty dict or list means empty, and gets empty.
```

Then insert at the top of the body, immediately after `root = Path(output_root) / "panel"`:

```python
    if artifacts is None:
        artifacts = load_panel_artifacts(output_root, target)
    if draws is None:
        draws = _panel_draws(output_root)
    if model_links is None:
        model_links = model_links_from_variants(variants, target)
```

Then delete the now-redundant `or {}` / `list(draws)` fallbacks further down the body — change `artifacts or {}` to `artifacts` in the `model_vs_single(...)` call, and leave `list(draws)` in the `imputation_stability(...)` call as it is (it still needs to be a list).

- [ ] **Step 6: Run the marker panel tests**

```bash
python -m pytest heavy_machinery/pytests_atypier/test_marker_panel.py -v
```

Expected: all pass, new and pre-existing. The pre-existing `test_run_marker_panel_writes_every_table_and_figure` passes no `draws` — it now hits `_panel_draws`, which must return `[]` for a `tmp_output` with no MICE manifest. If that test fails, the Step 4 path is wrong.

- [ ] **Step 7: Run the full suite**

```bash
python -m pytest -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add heavy_machinery/modelling_phase/marker_panel.py heavy_machinery/pytests_atypier/test_marker_panel.py
git commit -m "fix: derive marker panel model keys by suffix, not by substring

The keys are built by stripping the target prefix and the _model suffix from
an artifact filename and from the variant id, so the two meet. str.replace
stripped _model wherever it appeared, so experimental_model_1 collapsed to
experimental_1 — matching only because the artifact stem was mangled the same
way. Reuses inferential._artifact_model_id, which strips the suffix only.

The two experimental models are now named Experimental model 1 and 2 in the
comparison table. No number changes."
```

- [ ] **Step 9: Confirm the rename is the only behavioural change**

```bash
python -m pytest heavy_machinery/pytests_atypier/ -q
```

Expected: all pass. Then report to the user which model keys changed and confirm no test asserting a numeric value needed editing. If you had to change a number in any existing test to make it pass, stop — that means the fix moved a statistic, which it must not.

---

### Task 5: Clean the notebook cells

Every module change is now in place. This task edits `meningioma-modelling.ipynb` only.

**Files:**
- Modify: `meningioma-modelling.ipynb` cells 2, 4, 5, 22

**Interfaces:**
- Consumes: `validation.validate_unimputed_handoff` (Task 2), `load_modelling_handoff` verbose printing (Task 3), `marker_panel.run_marker_panel(variants=...)` (Task 4).
- Produces: nothing downstream depends on this task except Task 6's verification run.

Use the `NotebookEdit` tool for each cell, not `Edit` — the notebook is JSON and hand-editing it corrupts the outputs.

- [ ] **Step 1: Confirm the cell indices before editing**

```bash
python3 -c "
import json
nb=json.load(open('meningioma-modelling.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        print(i, repr(''.join(c['source']).splitlines()[0] if c['source'] else ''))
"
```

Expected: index 2 starts `import pandas as pd`, index 4 starts `df, schema, IMPUTATION_METHOD =`, index 5 starts `schema_validation =`, index 22 starts `from heavy_machinery.cleaning_phase.missingness_resolution import load_imputed_frames`. If the indices differ, use the ones you see — do not edit by index blindly.

- [ ] **Step 2: Replace cell 2**

Three dead imports go (`pandera.pandas as pa`, `format_table_for_display`, `preview_multivariable_cases` — none is referenced anywhere in the notebook), `load_schema_validation` becomes `validate_unimputed_handoff`, and `run_marker_panel` moves up from cell 22.

```python
import pandas as pd
pd.set_option("display.max_columns", None)

from pathlib import Path

from IPython.display import display

from heavy_machinery.config import load
from heavy_machinery.cleaning_phase.dataset_handoff import load_modelling_handoff
from heavy_machinery.cleaning_phase.missingness_resolution import load_modeling_frames
from heavy_machinery.cleaning_phase.validation import validate_unimputed_handoff, validate_imputed_frames
from heavy_machinery.modelling_phase.eda import screen_associations
from heavy_machinery.modelling_phase.diagnostic_accuracy import screen_diagnostic_accuracy
from heavy_machinery.modelling_phase.inferential import run_inferential_stage
from heavy_machinery.modelling_phase.marker_panel import run_marker_panel

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # do not wipe — reads cleaning outputs

#🟧🟧🟧 None = all years; e.g. [2025] for one cohort year

ANALYSIS_YEARS: list[int] | None = None
```

- [ ] **Step 3: Replace cell 4**

```python
df, schema, IMPUTATION_METHOD = load_modelling_handoff(OUTPUT_ROOT)
#df.head(3)
```

- [ ] **Step 4: Replace cell 5**

```python
schema_validation = validate_unimputed_handoff(df, OUTPUT_ROOT)
```

- [ ] **Step 5: Replace cell 22**

53 lines become 15. `NON_IMAGING` stays with its comment because it is a real input. `PANEL_TARGET` is inlined — it was only ever `EDA_TARGETS[0]`. `INFERENTIAL_MODEL_VARIANTS` (from cell 13) replaces the two raw lists: it is the same set of models with the same ids and links, already filtered to the ones that could actually be fitted, so the links it carries are exactly the ones with artifacts to attach to.

```python
# Not read off a scan. The accuracy table carries these too, and a section
# about MRI markers should not silently include them.
NON_IMAGING = {
    "sex_male", "hist_necrosis", "progesterone_pos",
}

panel_tables = run_marker_panel(
    df,
    target=EDA_TARGETS[0],
    accuracy_table=diag_acc,
    output_root=OUTPUT_ROOT,
    exclude=NON_IMAGING,
    variants=INFERENTIAL_MODEL_VARIANTS,
)

display(panel_tables["02_marker_panel_reading_view"])
display(panel_tables["09_selection_correction"])
display(panel_tables["13_model_reading_view"])
```

- [ ] **Step 6: Check no logic survived in any code cell**

```bash
python3 -c "
import json, re
nb = json.load(open('meningioma-modelling.ipynb'))
bad = re.compile(r'^\s*(def |class |if |elif |else:|for |while |try:|except )')
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    for n, line in enumerate(''.join(c['source']).splitlines(), 1):
        if bad.match(line):
            print(f'cell {i} line {n}: {line}')
print('SCAN DONE')
"
```

Expected: `SCAN DONE` with no lines above it.

- [ ] **Step 7: Check the commented-out lines all survived**

```bash
python3 -c "
import json
nb = json.load(open('meningioma-modelling.ipynb'))
code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
for wanted in ['#df.head(3)', '#assoc', '#full_inferential_table',
               \"#load('analysis').print_copy_pasteable_columns(df)\".replace(chr(39), chr(34)),
               '#🟧🟧🟧 Full table', \"#'progesterone_pos': True\"]:
    print(('OK  ' if wanted in code else 'MISSING '), wanted)
"
```

Expected: every line prints `OK`. A `MISSING` means a user comment was lost — restore it before continuing.

- [ ] **Step 8: Commit**

```bash
git add meningioma-modelling.ipynb
git commit -m "refactor: modelling notebook cells hold only inputs, calls and comments"
```

---

### Task 6: Verify the output changed in exactly one way

The refactor is worthless if a single number moved. This runs the notebook end to end, compares against the Task 1 baseline, and confirms the only difference is the four model-name strings from the Task 4 bug fix.

**Files:**
- No source changes. Produces a verification result only.

- [ ] **Step 1: Run the notebook end to end**

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 meningioma-modelling.ipynb
```

Expected: completes without error. If a cell raises, read the traceback and fix the cause in the module (not by putting logic back into the notebook) before continuing. Note this rewrites the notebook's stored outputs, which is expected and belongs in the commit at Step 7.

- [ ] **Step 2: List which files differ at all**

```bash
diff -rq /tmp/output_baseline output
```

Expected: **at most** these three lines, in any order —

```
Files /tmp/output_baseline/panel/tables/10_model_vs_single.csv and output/panel/tables/10_model_vs_single.csv differ
Files /tmp/output_baseline/panel/tables/13_model_reading_view.csv and output/panel/tables/13_model_reading_view.csv differ
Files /tmp/output_baseline/report/report.html and output/report/report.html differ
```

Any fourth file — any `.svg`, anything under `eda/`, `inferential/`, `dda/`, `datasets/` — is a regression. Stop and debug it. Do not proceed and do not describe the run as verified.

- [ ] **Step 3: Confirm the two CSVs differ only in the model-name column**

```bash
diff /tmp/output_baseline/panel/tables/10_model_vs_single.csv output/panel/tables/10_model_vs_single.csv
diff /tmp/output_baseline/panel/tables/13_model_reading_view.csv output/panel/tables/13_model_reading_view.csv
```

Expected: paired `<` / `>` lines where the **only** textual change on each line is `experimental_1` → `experimental_model_1`, `experimental_2` → `experimental_model_2` (in `10_`) or `Experimental 1` → `Experimental model 1`, `Experimental 2` → `Experimental model 2` (in `13_`). Every number on those lines must be character-for-character the same, and the row count and row order must be unchanged.

Prove that mechanically rather than by eye:

```bash
diff <(sed 's/experimental_model_/experimental_/g' output/panel/tables/10_model_vs_single.csv) /tmp/output_baseline/panel/tables/10_model_vs_single.csv && echo "10_ IS NAME-ONLY"
diff <(sed 's/Experimental model /Experimental /g' output/panel/tables/13_model_reading_view.csv) /tmp/output_baseline/panel/tables/13_model_reading_view.csv && echo "13_ IS NAME-ONLY"
```

Expected: both `IS NAME-ONLY` lines print with no diff above them. If either prints a diff, a number moved — that is a regression, not a rename.

- [ ] **Step 4: Confirm report.html differs only by the same strings**

```bash
diff <(sed -e 's/experimental_model_/experimental_/g' -e 's/Experimental model /Experimental /g' output/report/report.html) /tmp/output_baseline/report/report.html && echo "REPORT IS NAME-ONLY"
```

Expected: `REPORT IS NAME-ONLY`. If a diff appears, check first whether `report.html` embeds a build date — a date-only difference is benign:

```bash
diff <(sed -E -e 's/experimental_model_/experimental_/g' -e 's/Experimental model /Experimental /g' -e 's/[0-9]{4}-[0-9]{2}-[0-9]{2}//g' output/report/report.html) <(sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2}//g' /tmp/output_baseline/report/report.html)
```

Expected: empty. Anything else is a regression.

- [ ] **Step 5: Confirm the file count matches**

```bash
find output -type f | wc -l && cat /tmp/output_baseline_count.txt
```

Expected: the two numbers are equal.

- [ ] **Step 6: Run the full test suite one final time**

```bash
python -m pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit the re-executed notebook**

```bash
git add meningioma-modelling.ipynb
git commit -m "chore: re-execute the modelling notebook after the cell cleanup"
```

- [ ] **Step 8: Record the rename in CHANGES.md and README.md**

The model rename is visible to anyone reading `report.html`, so it belongs in the changelog rather than only in a commit message. Follow the existing row format in `CHANGES.md` — read the top few entries first and match them.

Content to record: the marker panel's model keys are now derived with `inferential._artifact_model_id` instead of a `str.replace` that stripped `_model` wherever it appeared, so the two experimental models are named "Experimental model 1" and "Experimental model 2" in the comparison table rather than "Experimental 1" and "Experimental 2". No number changed; the notebook cells that built those keys are gone.

Also check whether `README.md:439` (the marker panel section) names either model, and update it if so:

```bash
grep -n "Experimental 1\|Experimental 2\|experimental_1\|experimental_2" README.md CHANGES.md
```

- [ ] **Step 9: Commit the docs**

```bash
git add CHANGES.md README.md
git commit -m "docs: record the marker panel model-key fix"
```

- [ ] **Step 10: Report the verification result to the user**

State plainly, with actual numbers rather than summary adjectives: how many files `diff -rq` reported as differing, whether all three `IS NAME-ONLY` proofs printed, whether the file counts matched, and how many tests passed. If anything differed beyond the four approved strings, say exactly what differed — do not describe the refactor as complete.

---

## Notes for the implementer

- **`config/analysis.py` is untouched.** Notebook cell 13 (`_analysis = load("analysis")` plus three `resolve_*` calls) is already nothing but function calls and one input dict, so it is already compliant. Folding the three calls into one would be churn, not cleanup.
- **Cells 8, 10, 12 are the point of the notebook.** `EDA_TARGETS`, `EDA_PREDICTORS`, `LITERATURE_MODEL_VARIANTS`, `EXPERIMENTAL_MODEL_VARIANTS` are exactly the "lines specifically encoding something" the rule protects. They stay in full, comments and commented-out entries included. Do not move them into a config module.
- **Cells 16, 19, 20, 25, 26 are already calls and inputs.** No change.
- **If a cell will not run after an edit,** the fix belongs in the module. Putting a helper back into the notebook to unblock yourself defeats the task — report the blocker instead.
