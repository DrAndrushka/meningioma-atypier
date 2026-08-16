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
| Single-predictor models | **Lightweight** — AUC only | 22 models otherwise means 22 report folds and 22 comparison rows. |
| ΔAUC scale | **Optimism-corrected**, differenced within resamples | Combined models have more parameters, so apparent ΔAUC overstates the very effect being claimed. |
| Nested test | **D2 pooling** (Li, Meng, Raghunathan & Rubin 1991) | A valid pooled LR test across 20 MICE draws otherwise needs Meng–Rubin D3. D2 is standard, citable, and ~20 lines. Label the column D2. |
| MICE | **unchanged** | Already m=20, maxit=20, seed=42 — matches the spec exactly. |

### Deviations from the source spec

1. **B = 1000, not 2000.** Stated above. Everything else in the spec's pipeline
   requirements block is met as written.
2. **`radeesri_lekhavat_2023` renamed `radeesri_2023`** to match the spec's IDs.
   Its predictors are unchanged and already correct.

## Model inventory — 22 fits

**7 literature models.** `radeesri_2023` (exists), `spille_2020`, `zhang_2020`,
`funari_2023`, `kawahara_2012`, `lin_2014`, `peng_2021`.

**3 experimental models.** `experimental_model_1` and `experimental_model_2` are
unchanged. `top_6_signs` becomes **computed** rather than a frozen list — see
"Data-driven selection" — and is renamed `top_6_variables`, because under AUC
ranking four of its six are continuous measurements rather than binary signs.

**`top_1_variable` is not a separate fit.** It is the label for whichever single
predictor tops the AUC ranking, which is currently `tumor_volume` — already one
of the 12 singles below. It is therefore that model, promoted to a row in the
comparison figure and used as the ΔAUC denominator, not an extra fit. This is
the change that makes the reference and the top-1 model the same object rather
than two choices that could disagree.

**12 single-predictor models** — the union of every predictor appearing in any
literature model:

| Predictor | Feeds |
|---|---|
| `necrosis_or_hemorrhage` | Radeesri |
| `hyperostosis` | Radeesri |
| `perifocal_edema` | Radeesri, Zhang, Funari |
| `edema_volume_cm3` | Spille |
| `heterogeneous_enhancement` | Spille, Kawahara, Lin |
| `calcification` | Zhang |
| `irregular_tumor_margin` | Zhang, Funari, Kawahara, Lin, Peng |
| `skull_base_location` | Zhang, Peng |
| `tumor_volume` | Funari — and currently `top_1_variable`, the reference |
| `cortical_destruction` | Peng |
| `capsular_enhancement` | Lin |
| `age_ge75` | Lin |

All 12 are new fits — none is currently modelled alone.

**New or changed: 19 of the 22** — 12 singles, 6 literature models, and
`top_6_variables` refitted on a different predictor set. Only `radeesri_2023`,
`experimental_model_1` and `experimental_model_2` are untouched.

`recurrent_meningioma`, the old `top_1_sign`, drops out of the model list
entirely: it was top by LR+ but ranks 13th by AUC, and appears in no literature
model. It remains in the marker panel, where LR+ is the right metric.

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

- **Kawahara 2012** reports two independent predictors but no retrievable
  ORs/CIs. Result cells stay empty until the publisher PDF is obtained.
- **Zhang 2020** reports β coefficients only, no ORs. Its terms carry β with the
  scale named.

### Comparison layer

Pairing is sound because every model is fitted on the same patients: bootstrap
validation runs on `imputed_frames[0]`, which has no missing values, so the
complete-case frame is all 352 rows for all 22 models. Differences are therefore
genuinely paired, not two independent estimates.

`bootstrap_internal_validation` gains an option to return its **per-resample AUC
vector** alongside the aggregates it already returns. For a (combined, single)
pair, ΔAUC is differenced within each resample and the CI is the percentiles of
that difference distribution.

A new module, `heavy_machinery/modelling_phase/model_comparison.py`, owns:

- fitting and validating the 12 singles
- the paired ΔAUC with CI
- D2 pooling for the nested LR test
- the reference declaration check

It reuses `build_complete_case_frame` and `bootstrap_internal_validation` rather
than reimplementing either.

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
  in AUC order, its AUC, whether it was kept or dropped, and the reason for each
  drop. Source of the footnote under `top_6_variables`.

### `report.html`

Inside each literature model's existing fold, below its odds-ratio table: the
combined-vs-single comparison, and the `surrogate_note` where one is set. No new
top-level section.

The existing `high_grade__model_comparison.png` grows to **11 rows** — 7
literature, 3 experimental, 1 reference (`top_1_variable`, which is the
`tumor_volume` single promoted to its own row) — keeping its three panels of apparent
versus optimism-corrected AUC, Brier and calibration slope. The reference row is
visually distinguished. One figure, not two: 11 rows fits without crowding.

## Testing

- Paired differencing uses identical resample indices for both models of a pair.
- The reference assertion raises when a better-AUC variable is introduced.
- D2 pooling reproduces a worked example from the source paper.
- `age_ge75` cell counts stay above the n<20 floor.
- Kawahara's empty result cells render as empty, not as `nan`.
- Every literature model's predictor list matches `published_models.py`.

## Cost

Modelling phase goes from ~13 s to roughly **1 minute** (22 models × 1000
resamples, 4 workers). Cut-point phase is unchanged in cost; only its seed usage
is confirmed. A full clean pipeline run stays under two minutes.

## Out of scope

- No dichotomised `tumor_volume` variant. Source spec footnote 4: importing a
  foreign cohort's cut-point adds a second layer of optimism.
- No refit of the five excluded models.
- **Open — not yet decided.** Re-running the selection inside each bootstrap
  resample for `top_6_variables` / `top_1_variable`. Because selection is now
  code rather than a frozen list, each resample *could* re-rank, re-guard and
  re-pick its own six before refitting. Without it the bootstrap corrects the
  coefficients but gives the selection a free pass, so those two models'
  corrected AUCs stay optimistic in a way the literature models' are not — and
  they sit in the same comparison figure. Recommended in scope; costs roughly
  3–5× their validation time (28 AUCs and one correlation matrix per resample,
  cheap at n=352). **Decide before the implementation plan is written.**
- No new top-level report section.
