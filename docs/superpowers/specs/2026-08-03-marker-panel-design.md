# Marker panel — design

Date: 2026-08-03
Branch: `esnr-threshold-report`
Status: approved, not yet implemented

## What this is for

The study has two stated aims that `report.html` does not currently answer in
one place:

1. **Biomarker analysis** — identify the MRI markers most specific for
   high-grade meningioma.
2. **Feature set** — establish whether a *combination* of several radiological
   features predicts high-grade meningioma better than each feature alone.

Both are answerable from artifacts the pipeline already produces, but the
answers are scattered: per-marker specificity sits inside the EDA section's
diagnostic-accuracy table, and the combination question is answered only in a
different deliverable (`output/thresholds/threshold_report.html`).

This design adds one section to `output/report/report.html` that answers both,
standalone, positioned after multivariable modelling and before the appendix.

### Why it belongs in `report.html` specifically

The working loop is: run cleaning and modelling → run the thresholds notebook →
paste the resulting cut-points into `DERIVATIONS` in the cleaning notebook →
re-run cleaning and modelling. On that second pass the cohort carries the
threshold-derived binary flags as columns, so `report.html` is the first
artifact in which the cohort-specific cut-points and the qualitative MRI signs
coexist. It is also the only report that gets read.

The section must therefore not require the threshold report to be open.

## Decisions

| Question | Decision |
|---|---|
| Placement | New section between `render_inferential` and `render_appendix` |
| Where computation lives | New module `modelling_phase/marker_panel.py`; `report.py` stays a pure renderer |
| Estimators | Reuse `threshold_phase/combinations.py` unchanged, via an adapter |
| Marker list | Read from `output/eda/tables/diagnostic_accuracy.csv`, minus a `NON_IMAGING` exclude list |
| Aim 1 ranking | Positive likelihood ratio, shown with case yield |
| Aim 2 headline | Count score ("how many signs are present") |
| Aim 2 statistic | Multivariable model AUC vs best single marker AUC, both optimism-corrected |
| Aim 2 detail | AND/OR rule menu, in a collapsible dropdown |
| Dataset | Observed data primary; 20 MICE draws as a stability check |
| Denominator | One shared set of 301 patients for the head-to-head |

## Architecture

### New module: `heavy_machinery/modelling_phase/marker_panel.py`

Computes and saves. Renders nothing. This preserves the contract in
`report.py`'s docstring — "Reads CSV/SVG under `output/` (no refitting)".

Called from a new notebook cell §04.5, between multivariable modelling (§04)
and build-report (§05).

### Reuse, not reimplementation

`combinations.py` touches only four members of a `CutPoint`: `.col`, `.label`,
`.short_label` and `.flag(df)`. A duck-typed adapter therefore makes every
threshold-phase estimator work on plain binary columns with **no change to
`combinations.py`**:

```python
class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a CutPoint so combinations.py accepts it."""
    col: str
    label: str

    @property
    def short_label(self) -> str:
        return self.label

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].astype("boolean")
```

This buys `shared_cohort`, `flag_frame`, `combine_flags`, `single_rule_table`,
`pair_rule_table`, `count_score_table`, `count_threshold_table` and
`bootstrap_best_rule` — one set of estimators serving two reports, which is the
only way the two reports cannot contradict each other.

`marker_panel.py` adds exactly one estimator of its own: the positive
likelihood ratio and its confidence interval, which
`diagnostic_accuracy.binary_diagnostic_metrics` does not currently compute.

### Marker selection

Read the `binary` and `derived_binary` rows for `target == "high_grade"` from
`output/eda/tables/diagnostic_accuracy.csv`. Drop:

- the outcome itself (`high_grade`),
- any column named in `NON_IMAGING`.

`NON_IMAGING` lives in the notebook cell, not in the module, so it is visible
and editable without touching `.py` files. It exists because
`diagnostic_accuracy.csv` also carries non-imaging predictors — `sex_male` is
`derived_binary` and would otherwise walk into a section about MRI markers.
Histology predictors are pruned upstream in the notebook.

Reading the list rather than hard-coding it means whatever is activated or
dropped in `DERIVATIONS` flows through automatically.

### Data flow

```
notebook §04.5
  └── marker_panel.run_marker_panel(df_unimputed, draws, output_root, ...)
        ├── output/panel/tables/*.csv
        └── output/panel/figures/*.svg
                    │
notebook §05  build_report
  └── report.load_artifacts  → reads output/panel/
  └── report.render_marker_panel → section_block(...)
  └── report.build_report → inserted before render_appendix
```

## Outputs

### `output/panel/tables/`

| File | Contents |
|---|---|
| `01_marker_panel.csv` | Per marker: `n_used`, `present_n`, TP/FP/FN/TN, sensitivity + CI, specificity + CI, PPV, NPV, AUC, LR+ + CI, `chance_overlap` flag |
| `02_marker_panel_reading_view.csv` | The same, formatted for the report table |
| `03_shared_cohort.csv` | The shared set: n, events, the markers it required, and how many patients each marker cost |
| `04_single_rules.csv` | Each marker alone, scored on the shared set |
| `05_rule_menu.csv` | Singles + AND/OR pairs, scored on the shared set |
| `06_rule_reading_view.csv` | Rules ranked by Youden J, clinician-facing columns |
| `07_count_score.csv` | Observed high-grade rate at each count of signs present, Wilson CIs |
| `08_count_thresholds.csv` | The count used as a test: "≥1 sign", "≥2 signs", … |
| `09_selection_correction.csv` | Apparent J, optimism, corrected J — for the best single and the best combination |
| `10_model_vs_single.csv` | Per model: artifact corrected AUC (n = 352, imputed), coefficients re-scored on the shared set (n = 301, observed, apparent), and the best single marker's corrected AUC on the shared set |
| `11_imputation_stability.csv` | Across the 20 MICE draws: how often the same marker ranks first, how often the same rule wins, how often the combination still beats the best single |

### `output/panel/figures/`

| File | Contents |
|---|---|
| `lr_forest.svg` | LR+ per marker with confidence intervals, log x-axis, one row per marker, reference line at 1 |
| `count_score.svg` | Observed risk against number of signs present, Wilson CIs |
| `rule_space.svg` | Every rule in sensitivity–specificity space, singles marked apart (reuses `combinations.combination_figure`) |

## What the section says

**Title:** `🎯 Which MRI markers, and do they combine?`

### Aim 1 — which sign argues hardest for high grade

One lead sentence naming the strongest marker with its LR+ and how many
high-grade tumours it flags, then one table sorted by LR+:

| Marker | Present in | Catches | Sens (CI) | Spec (CI) | LR+ (CI) |
|---|---|---|---|---|---|

**Positive likelihood ratio (LR+)** = sensitivity / (1 − specificity): how many
times more likely the sign is to appear in a high-grade tumour than a benign
one. Confidence interval computed on the log scale (Katz).

Two guards, both required:

- The **Catches** column (true positives out of all high-grade tumours in the
  denominator) is what stops the specificity trap. `brain_invasion` has the
  highest specificity in the cohort (0.996) purely because it is almost never
  seen — it flags 5 of 105 high-grade tumours. Ranking on specificity alone
  would put a near-useless sign first.
- Any marker whose LR+ interval includes 1 is printed as "not distinguishable
  from chance" instead of being given a rank.

Then `lr_forest.svg`.

### Aim 2 — does a combination beat any single sign

Three layers, in decreasing order of prominence.

**Headline — count score.** One sentence and `count_score.svg`: observed
high-grade risk against how many of the N signs are present, with Wilson
confidence intervals. This is the literal answer to the study aim and involves
no selection of a winning rule, so it needs no optimism correction.

**Statistic — model vs best single.** A short table of each multivariable
model's AUC against the best single marker's AUC, with three labelled columns
per model: its own optimism-corrected AUC from its artifact (n = 352, imputed),
its coefficients re-scored on the shared set (n = 301, observed, apparent), and
the best single marker's optimism-corrected AUC on that same shared set. The
re-scored column is the like-for-like comparison; the artifact column is
carried alongside so the two are never confused. One sentence states that the
re-scored figure is not itself optimism-corrected, and in which direction that
biases it.

**Detail — the rule menu, in a dropdown.** All singles and AND/OR pairs ranked
by Youden J, with one sentence stating how much the winner's advantage shrank
once selection was accounted for.

### Two correctness constraints

**1. Selection optimism must be corrected on both sides.** With ~15 markers
there are ~105 pairs × 2 logics ≈ 210 candidate rules. The best of 210 will
beat the best single marker on the data that chose it, even when no rule is
genuinely better. `CHANGES.md` records this exact error already occurring once
in this project: a gain reported as +0.008 was +0.050 once both sides were
corrected the same way. `bootstrap_best_rule` is the correction, and it must be
applied to the best *single* as well as the best *combination* — "best of 15"
is also a choice made on these patients.

**2. The model comparison re-scores, it does not refit.** The multivariable
models are fitted on the 20 MICE draws and pooled by Rubin's rules (n = 352,
`n_models = 20`), so their saved AUCs are not on the same patients as the
markers. `marker_panel.py` loads the saved coefficients from
`output/inferential/model_artifacts/*_model.json` and applies them to the
observed values of the shared set, producing an AUC on the same patients the
markers are scored on. No model is refitted; only the patient set changes.

The re-scored AUC is apparent — correcting it would require re-running the
bootstrap on the shared set, which is refitting and therefore out of scope. The
coefficients were fitted on data that includes these patients, so the re-scored
figure is optimistic; the artifact's own corrected AUC bounds how much
(0.016–0.042 on the current models). Both columns are printed and labelled, and
the section states this rather than implying the two are interchangeable.

## Dataset and denominator

**Primary: observed data.** Sensitivity, specificity and LR+ describe how a
sign performs when a radiologist looks at the scan. Imputing the marker reports
the accuracy of a finding nobody observed, which changes what the number means
for a rule intended to be applied at the scanner.

**Stability check: the 20 MICE draws** at
`output/missingness/mice/imputed_0NN.parquet`, reported as reproduction rates
in `11_imputation_stability.csv`. Rubin's rules can average an estimate but
cannot average a *choice* — a different rule can win in each draw — so the
honest imputed output is "the same rule won in X% of draws", not a pooled
winner. This is the same pattern the threshold phase already uses in
`18_imputation_stability.csv`.

**Denominator: one shared set for the head-to-head.** On the current cohort
that is 301 of 352 patients. The loss is not the qualitative signs, which are
nearly complete (348–352 of 352); it is ADC (309), tumour volume (329) and
edema volume (333). A Youden J compared across two different patient sets is
not a comparison, so every rule in `04`–`06` and `09`–`10` is scored on the
same patients. The aim-1 marker table in `01` keeps each marker's own `n_used`
and prints it, because that table ranks markers rather than comparing rules.

`03_shared_cohort.csv` records which markers cost how many patients, so the
denominator is auditable rather than asserted.

## Error handling

Follows the conventions already in `report.py`:

- Missing or empty panel tables → `warning_box(...)`, section still renders.
- Fewer than two usable markers after exclusion → `info_box(...)` explaining
  that a combination question needs at least two markers; no crash.
- Missing model artifacts → the model-vs-single table is omitted and the count
  score and rule menu still render.
- A marker with zero positives or zero negatives in the shared set → excluded
  from the rule search with a stated reason in `03_shared_cohort.csv`, rather
  than producing an undefined LR+.

## Testing

New file `heavy_machinery/pytests_atypier/test_marker_panel.py`:

- LR+ and its confidence interval against a hand-computed 2×2.
- A marker with specificity 1.0 produces a stated result rather than a division
  by zero.
- `BinaryMarker` produces flags identical to an equivalent `CutPoint`, so the
  reuse claim is verified rather than assumed.
- The `NON_IMAGING` exclude list actually excludes, and the outcome column is
  never treated as a marker.
- The shared set contains exactly the patients with every marker observed.
- `bootstrap_best_rule` is applied to the single side as well as the
  combination side — a regression test for the `CHANGES.md` bug.
- Determinism: two runs with the same seed produce identical tables.

Additions to `test_report.py`:

- The section appears in the built HTML, positioned after multivariable
  modelling and before the appendix.
- With no panel tables on disk the section degrades to a warning box and the
  report still builds.

## Out of scope

- Changing the EDA section's existing diagnostic-accuracy table.
- Any change to `threshold_phase/combinations.py`.
- Refitting any multivariable model.
- Triples and larger AND/OR rules. `pair_rule_table` supports them, but an AND
  of three signs on ~90 events lands in single figures, and a sensitivity
  computed from eight patients is not a number worth reporting.
