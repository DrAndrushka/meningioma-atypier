# Literature model refit — design

Source: `literature_model_refit_spec.md` (Downloads, 2026-08-16).
Status: approved 2026-08-16. Supersedes nothing; extends the multivariable phase.

## What this is for

Seven previously published multivariable models for high-grade meningioma are
refit in our cohort (n=352; 247 grade 1, 105 grade 2–3), each measured against
the single predictors it is built from. The research question is Zhang 2020's
and Peng 2021's: **does combining imaging features beat the best single one, and
by how much?** Those two papers are the only published meningioma-grading work
reporting combined-versus-each-single AUC, with gaps of ΔAUC 0.05 and ΔC 0.095
respectively. Our answer has to be on the same footing to be comparable.

These are **refits, not external validations**. Three of the seven substitute
`irregular_tumor_margin` for the original papers' tumour–brain interface, a
related but non-identical construct. The design carries that caveat in config so
it reaches the report without depending on anyone remembering it.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Bootstrap resamples | **B = 1000** | Deviates from the source spec's 2000. Keeps the existing constant, so no already-published number moves for that reason alone, and halves the runtime. Footnote the deviation. |
| Bootstrap seed | **20260801**, master seed | The spec's value. Replaces the modelling phase's current per-resample `RandomState(i)` scheme. |
| Scope of the seed change | Modelling phase **only** | The cut-point phase already uses `SEED = 20260801` (`wobble.py`), and with B staying at 1000 its resample count is unchanged too. The report's Methods card claim of one shared count across the pipeline therefore stays true with no edit. |
| ΔAUC comparison | Combined vs **each of its own singles** | Reproduces what Zhang and Peng actually published. A single shared denominator would not. |
| Reference model | **`top_1_variable`** — the computed best single predictor | Currently `tumor_volume` (0.679). No separate `model_0`: the reference *is* the top-1 model. See "Data-driven selection". |
| `top_6` / `top_1` selection | **Computed each run**, not a hard-coded list | Was a frozen list ranked by LR+. Now ranked by AUC, which is the metric the deliverable compares on. |
| Selection guard 1 | Drop a derived cut-point when its **continuous parent** is a candidate | The parent carries more information and does not stack cut-point optimism on model optimism (source spec footnote 4). |
| Selection guard 2 | Drop a candidate **collinear (ρ>0.8)** with one already picked, and take the next one that is not | Without it the top 6 by AUC is four tumour-size variables at ρ up to 0.91. |
| Selection audit | Every drop **recorded and footnoted** | A selection nobody can see is a selection nobody can check. |
| Single-predictor models | **Lightweight** — AUC only | 38 models (11 combined + 27 singles; see "Model inventory" below) otherwise means 38 report folds instead of 11. |
| ΔAUC scale | **Optimism-corrected**, differenced within resamples | Combined models have more parameters, so apparent ΔAUC overstates the very effect being claimed. |
| Nested test | **D2 pooling** (Li, Meng, Raghunathan & Rubin 1991) | A valid pooled LR test across 20 MICE draws otherwise needs Meng–Rubin D3. D2 is standard, citable, and ~20 lines. Label the column D2. |
| MICE | **unchanged** | Already m=20, maxit=20, seed=42 — matches the spec exactly. |
| Selection inside the bootstrap | **In scope**, for `top_6_variables` | Each resample re-ranks, re-guards and re-picks its own six before refitting. Otherwise the only data-selected model in the figure is the only one whose optimism is under-corrected. |
| Collinearity threshold | **ρ = 0.8** | Conventional. Nothing in this cohort sits near the boundary, so 0.7 or 0.9 would pick the same six — but the number is published, so it is stated. |
| AUC direction in tables | **Show discrimination beside raw AUC**, arrow on protective variables | A raw AUC of 0.370 for `adc_value` reads as "useless" when it means 0.630 the other way. Two of the six picked variables are protective. |

### Deviations from the source spec

1. **B = 1000, not 2000.** Stated above. Everything else in the spec's pipeline
   requirements block is met as written.
2. **`radeesri_lekhavat_2023` renamed `radeesri_2023`** to match the spec's IDs.
   Its predictors are unchanged and already correct.

## Model inventory — 11 combined models, 27 singles, 47 comparison rows

*(Updated 2026-08-17, final whole-branch review, Item 5: this section
originally said "22 fits" and listed 12 singles from before `top_1_variable`
and `top_6_variables` were wired through `run_comparison_stage`/
`write_streamlit_artifacts` as real fitted variants. The counts below are
read off a clean three-notebook run, not estimated.)*

**7 literature models.** `radeesri_2023`, `spille_2020`, `zhang_2020`,
`funari_2023`, `kawahara_2012`, `lin_2014`, `peng_2021`.

**2 experimental models.** `experimental_model_1`, `experimental_model_2` —
hand-picked predictor sets, unchanged by this design.

**2 data-selected models.** `top_1_variable` and `top_6_variables` — both ARE
separate fits (each gets its own report fold, forest plot, VIF table, and
calculator artifact), contrary to this section's earlier claim that
`top_1_variable` was folded into the reference rather than fitted. The
reference/ΔAUC-denominator role and the fitted model are the same object
(`tumor_volume` today), but that object is still one of the 11 rows here, not
a label pinned onto another model's row.

11 total: 7 + 2 + 2.

**27 single-predictor models — the union of every predictor across ALL 11
combined models above, not just the 7 literature ones.** The extra 15 beyond
the literature union come from `experimental_model_1`/`experimental_model_2`
(hand-picked, e.g. `dural_tail`, `mass_effect`, `dwi_hyperintensity`) and from
`top_6_variables`' own data-driven picks that no literature model uses (e.g.
`adc_value`, `cystic_component`, `skull_base_location` — the reason
`run_comparison_stage`'s candidate pool is explicitly documented as not the
literature union):

`adc_value`, `adc_value_le0.72`, `age`, `age_ge75`, `calcification`,
`capsular_enhancement`, `cortical_destruction`, `cystic_component`,
`dural_tail`, `dwi_hyperintensity`, `edema_volume_cm3`,
`edema_volume_ge4.76`, `hemorrhage`, `heterogeneous_enhancement`,
`hyperostosis`, `irregular_tumor_margin`, `male`, `mass_effect`,
`necrosis_or_hemorrhage`, `perifocal_edema`, `skull_base_location`,
`t1_hypointensity`, `t2_hyperintensity`, `transfalcine_extension`,
`transsinus_extension`, `tumor_volume`, `tumor_volume_ge15.1`.

**47 combined-vs-single comparison rows** — one per (combined model, single
predictor it is built from) pair: `experimental_model_1` 9, `experimental_model_2`
10, `zhang_2020` 4, `lin_2014` 4, `funari_2023` 3, `peng_2021` 3, `radeesri_2023`
3, `top_6_variables` 6, `kawahara_2012` 2, `spille_2020` 2, `top_1_variable` 1.

**Selection audit table — 12 full-cohort rows, 26 total.** `top_variable_selection
.csv`'s full-cohort walk (`vs.select_variables` over `EDA_PREDICTORS`, k=6)
stops as soon as six are kept, producing 12 rows — 6 kept, 6 dropped for a
cut-point/collinearity reason. Since Task-16-review Finding 3, every OTHER
candidate that won at least one of `top_6_variables`' bootstrap resamples
also gets a row (blank auc/discrimination/kept/reason — the full-cohort walk
never reached it), appended after the audited 12 and sorted by resample count.
21 candidates win at least one resample in production (7 already among the 12,
14 not), for 26 rows total today; the exact extra-row count is a property of
the data and the bootstrap seed, not a fixed constant.

`recurrent_meningioma`, the old `top_1_sign`, drops out of the model list
entirely: it was top by LR+ but ranks 13th by AUC, and appears in no literature
model. It remains in the marker panel, where LR+ is the right metric — though
it is the single highest resample-selection-count variable (326 of 1000) that
never enters the full-cohort audit's own 12 rows, exactly the instability
Finding 3 above makes visible.

## Architecture

### Data layer — one derivation

`age_ge75` joins the cleaning notebook's `DERIVATIONS`, marked model-only the
same way `necrosis_or_hemorrhage` is:

```python
_derivations.Apply(
    name="age_ge75", source="age",
    fn=lambda s: s.astype("Float64") >= 75,
    kind="binary", hide_parent=False, eda_in_derived=None,
    rule="age >= 75",
    reason="Lin BJ et al., J Neurosurg 2014;121(5):1201-1208 — 2 of 12 score points.",
)
```

`eda_in_derived=None` keeps it out of the EDA screen, the FDR families and the
marker panel. It exists to fit one published model.

**Cell counts are already checked**: 46 grade-1 and 23 grade-2/3 are ≥ 75, both
above the source spec's n<20 floor. It is fitted dichotomised as published; the
continuous-age fallback in the source spec is not used. A test pins this so the
fallback question is re-raised if the cohort changes.

### Model layer

`LITERATURE_MODEL_VARIANTS` gains 6 entries in the existing 5-tuple form.
`config/published_models.py` gains the 6 papers in its existing shape, plus two
new optional fields:

- `surrogate_note` — set on `kawahara_2012`, `lin_2014` and `peng_2021`, stating
  the interface substitution and that the refit is not an external validation.
  Rendered above the model's table.
- `NOT_FITTED` — a sibling dict recording the five models deliberately excluded
  (Azeemuddin 2018, Yao 2022, Amano 2022, Duarte Gomes / Quintas-Neves 2026,
  Hale 2018) and the reason for each. These reasons currently exist only in a
  commit message; config is where someone about to re-add one would look.

Two papers have deliberate gaps, left empty rather than filled:

- **Kawahara 2012** transcribed in full from the publisher PDF (obtained
  2026-08-17); the cells previously left empty are now filled. Multivariable
  Table 3: unclear tumour-brain interface aOR 42.0 (4.5-390) p=0.001;
  heterogeneous enhancement aOR 8.3 (1.7-40.4) p=0.009. n=65, 39 benign and 26
  high-grade, WHO 2000. Published equation z = -1.979 + 3.738*TBI +
  2.112*heterogeneity, giving 98% probability of high grade when both factors are
  present and 12.1% when neither. The sign of the second term is ambiguous in the
  extracted text and was resolved by reconstructing all four of the paper's own
  probabilities exactly.
- **Zhang 2020** reports β coefficients only, no ORs. Its terms carry β with the
  scale named.

**Kawahara needs a stronger surrogate note than the other two, and the PDF lets
it quote numbers.** The paper scored tumoral margin and tumour-brain interface as
*separate* factors and published both univariable effects: unclear interface
OR 71.8 (8.4-612) and irregular margin OR 10.3 (3.2-33). The
multivariable model kept the interface and dropped the margin. Our refit
substitutes `irregular_tumor_margin` for the interface, so it substitutes the
variable that was roughly sevenfold weaker on the authors' own data and which
they specifically discarded. The note quotes both numbers rather than asking a
reader to take the caveat on trust. Kawahara also assessed **negative**
capsular enhancement (OR 19.2, 5.4-69) and did not retain it, worth stating
because `lin_2014` does use `capsular_enhancement` — coded PRESENT, the
opposite direction, so the two papers do not agree with each other here.

*(Corrected 2026-08-17, final whole-branch review, Blocker 2: the word
"negative" — the published OR is for the ABSENCE of capsular enhancement —
was missing here and in `published_models.py`'s `kawahara_2012` entry, which
reversed the clinical claim. The "each p<0.001" attached to the two
univariable ORs above could not be confirmed in any reachable source and has
been dropped rather than left unsourced.)*

### Comparison layer

Pairing is sound because every model is fitted on the same patients: bootstrap
validation runs on `imputed_frames[0]`, which has no missing values, so the
complete-case frame is all 352 rows for all 11 combined models and 27 singles. Differences are therefore
genuinely paired, not two independent estimates.

`bootstrap_internal_validation` gains an option to return its **per-resample AUC
vector** alongside the aggregates it already returns. For a (combined, single)
pair, ΔAUC is differenced within each resample and the CI is the percentiles of
that difference distribution.

A new module, `heavy_machinery/modelling_phase/model_comparison.py`, owns:

- fitting and validating the singles (27 today — see "Model inventory" above)
- the paired ΔAUC with CI
- D2 pooling for the nested LR test
- the reference declaration check

It reuses `build_complete_case_frame` and `bootstrap_internal_validation` rather
than reimplementing either.

### Selection inside the bootstrap

`top_6_variables` is the only model whose predictors are chosen from the same
cohort it is fitted on, so it is the only one where the bootstrap must also
re-run the *choosing*. `bootstrap_internal_validation` gains an optional
`select` callable: when supplied, each resample re-ranks all candidates by AUC on
that resample, re-applies both guards, picks its own six, fits them, and scores
against the original cohort. The resulting optimism covers selection as well as
coefficients.

The six chosen on the full cohort remain the reported model; the selector only
changes what the correction measures. A count of how often each variable is
re-selected across the 1000 resamples goes into `top_variable_selection.csv` —
a variable picked in 400 of 1000 resamples is a different claim from one picked
in 990, and that distinction is otherwise invisible.

`experimental_model_1` is deliberately excluded from this: its optimism comes
from cut-points found in the cut-point phase, which the resample would have to
re-derive. Out of scope, caveat text unchanged.

### Data-driven selection — `top_1_variable` and `top_6_variables`

Both were frozen lists ranked by positive likelihood ratio. They become computed,
ranked by **univariate AUC** — the metric the whole deliverable compares on, and
the one that makes `top_1_variable` the reference by construction rather than by
a second, separate choice.

The selection walks candidates in descending AUC and applies two guards:

1. **Parent/child.** Skip a derived cut-point when its continuous parent is also
   a candidate. Drops `max_diameter_cm_ge3.81`, `tumor_volume_ge15.1`,
   `edema_volume_ge4.76`, `adc_value_le0.72`, `edema_index_ge0.0617`.
2. **Collinearity.** Skip a candidate whose |Spearman ρ| exceeds 0.8 against any
   already-picked variable, and continue to the next one that clears it.

Without guard 2 the top six by AUC are four tumour-size variables at ρ up to
0.91 — the same double-counting the notebook already warns about for the count
score. With both guards the six are distinct constructs:

| # | Variable | AUC | Construct |
|---|---|---|---|
| 1 | `tumor_volume` | 0.679 | size |
| 2 | `adc_value` | 0.370 ↓ | diffusion |
| 3 | `edema_volume_cm3` | 0.628 | edema |
| 4 | `irregular_tumor_margin` | 0.623 | margin |
| 5 | `skull_base_location` | 0.386 ↓ | location |
| 6 | `cystic_component` | 0.592 | internal structure |

`top_1_variable` is the first row, `tumor_volume`, which is also the ΔAUC
reference.

**Every drop is recorded**, with the variable, its AUC, and the reason — either
the parent it dichotomises or the ρ and the variable it clashed with. Written to
`top_variable_selection.csv` and rendered as a footnote under the model in the
report, so a reader can see that `max_diameter_cm` was second on AUC and was
dropped for ρ=0.91 against `tumor_volume`, not overlooked.

The reference is the maximum of ~30 univariate AUCs and is therefore biased
upward by selection. That makes every ΔAUC **conservative** — the gap is
understated, not inflated. Kept and footnoted rather than corrected, matching
what Zhang and Peng did.

Config still pins the expected winner and the run **raises** if a different
variable takes first place, so a 0.004 shift cannot silently move the denominator
under the manuscript without anyone noticing.

## Outputs

### `output/inferential/tables/`

- `single_predictor_reference.csv` — one row per single predictor: name, n,
  events, apparent AUC, optimism-corrected AUC.
- `model_vs_single_auc.csv` — one row per (model × one of its own predictors):
  both corrected AUCs, ΔAUC, CI bounds, D2 p-value.
- `top_variable_selection.csv` — the selection audit: every candidate considered
  in AUC order, its raw AUC, its discrimination (raw flipped when below 0.5),
  whether it was kept or dropped, the reason for each drop, and how many of the
  1000 resamples re-selected it. Source of the footnote under `top_6_variables`.

### `report.html`

Inside each literature model's existing fold, below its odds-ratio table: the
combined-vs-single comparison, and the `surrogate_note` where one is set. No new
top-level section.

The existing `high_grade__model_comparison.png` grows to **11 rows** — 7
literature, 2 experimental, 2 data-selected (`top_1_variable`, `top_6_variables`;
see "Model inventory" above — `top_1_variable` is a fitted row like any other,
not a label promoted onto another model's row) — keeping its three panels of
apparent versus optimism-corrected AUC, Brier and calibration slope. The
reference row (`top_1_variable`) is visually distinguished. One figure, not
two: 11 rows fits without crowding.

## Testing

- Paired differencing uses identical resample indices for both models of a pair.
- The reference assertion raises when a better-AUC variable is introduced.
- D2 pooling reproduces a worked example from the source paper.
- `age_ge75` cell counts stay above the n<20 floor.
- Kawahara's empty result cells render as empty, not as `nan`.
- Every literature model's predictor list matches `published_models.py`.

## Cost

Modelling phase goes from ~13 s to roughly **90 seconds** (pre-implementation
estimate; the model count it was based on, 22, is superseded — see "Model
inventory" above, now 11 combined + 27 singles). Cut-point phase is unchanged
in cost; only its seed usage is confirmed. Measured on the final
whole-branch-review clean run (2026-08-17): `meningioma-modelling.ipynb`
end-to-end (EDA, all 38 model fits/bootstraps, marker panel, report build)
took ~116 s.

## Out of scope

- No dichotomised `tumor_volume` variant. Source spec footnote 4: importing a
  foreign cohort's cut-point adds a second layer of optimism.
- No refit of the five excluded models.
- No re-selection inside the bootstrap for `experimental_model_1`. Three of its
  nine predictors are cohort-derived cut-points, so it carries the same kind of
  optimism one layer down — but fixing that means folding the threshold search
  into the resample, which spans the cut-point phase. Separate work; the model's
  caveat text stays.
- No new top-level report section.
