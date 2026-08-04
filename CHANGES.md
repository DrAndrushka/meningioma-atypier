# Threshold report — what changed and which numbers moved

Branch `esnr-threshold-report`, run of 2026-08-03. Seed `20260801` and the
bootstrap budgets (2000 for cut-points, 500 for risk curves, m = 20 MICE draws)
are unchanged, so every difference below is a fix, not a re-roll.

The report is regenerated from the notebook; nothing in
`output/thresholds/threshold_report.html` was hand-edited.

---

## 2026-08-04 — §04.5 marker panel: same numbers, 14 seconds instead of 102 minutes

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
| Shared-set correction (2 × 500 resamples) | 9.0 min | 1.3 s |
| §04.5 end to end | 102 min | **13.9 s** |

---

## Numbers that moved

| What | Was | Now | Why |
|---|---|---|---|
| Best single criterion, J after correction | *(blank cell)* | **0.232** (full data) / **0.212** (shared denominator) | The cell was empty. Computed with the same selection-optimism bootstrap used for the combination (P1.1). |
| Best single criterion, identity | Tumor volume ≥ 15.1 | **Edema volume ≥ 4.76** on the shared denominator | On one patient set the ranking of the singles changes (P1.2). |
| Gain of the best combination | +0.008 | **+0.050** (full data) / **+0.069** (shared denominator) | The old gain scored a *corrected* combination against an *uncorrected* single. Both sides are now corrected the same way (P1.1). |
| Selection optimism, best single | *(not computed)* | **0.042** full data, **0.046** shared | New. Near-identical to the combination's 0.042–0.044, which is why the old comparison was so misleading. |
| Winner stability, best single | *(not computed)* | **40.8%** full data, **32.0%** shared | New. The single is no more stable than the combination. |
| Zero-inflation share, edema volume | "a quarter of this cohort" | **36.6%** of measured patients (122 / 333) | The old figure was prose from an earlier run. Edema index is 35.0% (113 / 323) (P1.3). |
| Edema volume non-linearity, non-zero values only | *(not computed)* | **p = 0.180** (whole cohort p = 0.007) | New. The curvature does not survive removing the zeros (P1.3). |
| Edema index non-linearity, non-zero values only | *(not computed)* | **p = 0.675** (whole cohort p = 0.047) | New, same finding, stronger (P1.3). |
| Threshold verdict | "4 of 4 — yes" | **1 strong, 2 moderate, 1 fragile** | The binary p < 0.05 gate is replaced by a five-criterion graded hierarchy (P0.2). |
| Non-linearity p, adjusted | *(not computed)* | Holm **0.029 / 0.042 / 0.029 / 0.047** — all four survive. Bonferroni **0.037 / 0.084 / 0.029 / 0.189** — drops tumour volume and edema index | New. Multiplicity is now stated, not declined (P1.4). |
| Calibration slope, uncut four-measurement model | *(not computed)* | **1.00 apparent → 0.911 bootstrap-corrected** | New (P1.5). |
| Calibration slope, multivariable models | not in this report | **0.773 – 0.916 corrected**, worst for the 10-predictor model | Read from the modelling phase's own artifacts (P1.5). |
| Calibration **intercept**, multivariable models | apparent only (0.00) | **−0.005 to +0.000 corrected** | The modelling phase now bootstrap-corrects the intercept as well as the slope, so the threshold report's cell is filled instead of blank. |
| Brier score, uncut model | *(not computed)* | **0.189 apparent → 0.195 corrected** | New (P1.5). |
| Net benefit | *(not computed)* | Uncut model best over **58.9%** of the 5–60% threshold range; best single cut-point best over **0%** | New (P1.5). |
| Accrual window | not stated | **2018–2026**, 9 calendar years | Derived in `cohort_summary()` from `entry_year` (P3). |
| "flags 34 of the 34+62 high-grade tumours" | printed the arithmetic | **"flags 34 of the 96 … wrongly flagging 25 of the 213 benign ones"** | Arithmetic bug. The benign denominator is that metric's own complete cases (FP + TN = 213), not the cohort's 247 (P2). |
| Single cut-point performance | "AUC ≈ 0.64" | **"Balanced accuracy 0.63*"** with a footnote | (sens+spec)/2 is not an AUC and sat in a column next to real AUCs. The value also moved 0.64 → 0.63 because it now derives from the shared-denominator best single (P1.2, P2). |
| Edema index `spec_ge_90` | "≥3.89, J −0.03" printed as a rule | **"no cut-point attains ≥ 90% specificity with above-chance sensitivity"** | A worse-than-chance rule is a finding, not a rule (P2). |
| Edema volume / index `sens_ge_90` | blank row | **"not attainable — no cut-point on this cohort reaches ≥ 90% sensitivity …"** | A blank cell reads as a missing value (P2). |
| Risk-curve bootstrap count | "several hundred resamples" | **500**, with the cut-point budget (**2000**) named separately | Templated from the manifest. They really are two budgets (P2). |

### What did **not** move

Cut-points, risk curves, crossings, MICE stability rates, the count score
(11% → 20% → 32% → 41% → 66%) and the uncut model's AUC (0.69) are all
unchanged. No estimator was touched; the fixes are to how results were
compared, graded and described.

---

## Conclusions that changed

**1. "All four measurements have a threshold" is not what the evidence
supports.** Under the pre-specified hierarchy: edema index *strong*, tumour
volume and edema volume *moderate* (both fail the log-scale test), ADC
*fragile*.

**2. ADC's "threshold" restates the 50%-risk crossing.** Its knee is at 0.662
and the bootstrap interval of its 50%-risk crossing is 0.598–0.716 (crossing at
0.663). The knee sits inside it, so it carries no information the crossing does
not — and it reproduces in only 60% of MICE draws.

**3. The edema thresholds are largely presence versus absence.** Risk is 19%
(13–27) with no edema and 36% (30–43) with any — a 1.9-fold difference before a
cut-point is drawn. Refitted on non-zero values only, the curvature is gone
(p = 0.18 and p = 0.68).

**4. "Combining bought nothing" survives, but on different grounds.** The gain
is no longer negligible (+0.069 on one denominator, above the 0.05 bar). What
sinks it is stability: the winning combination holds in 39.6% of resampled
cohorts. The report now says that instead.

**5. No single cut-point is ever the best strategy on the decision curve.** The
uncut four-measurement model leads over 58.9% of the plausible threshold range;
the best single cut-point leads over none of it, though it does beat treat-all
and treat-none between 20% and 41%.

---

## Where I disagree with the brief

**The evidence grades did not come out as predicted, and I did not tune them to.**
The brief expected edema volume *strong*, ADC *moderate*, tumour volume
*fragile*, edema index *weak*. The five specified criteria produce edema index
*strong*, tumour and edema volume *moderate*, ADC *fragile*. Two specific
divergences are worth a decision before submission:

- **Edema index grades *strong* on five binary criteria while being the weakest
  measurement in the study**: AUC 0.58, non-linearity p = 0.047 (Holm 0.047,
  right on the line), knee interval spanning 3.6× end to end, and its curvature
  vanishing entirely once the zeros are removed. The hierarchy cannot see
  precision, only pass/fail. Knee-interval width, AUC and the non-zero refit p
  are therefore reported *next to* the grade and explicitly do not score it —
  promoting them now would be the after-the-fact rule-making the hierarchy
  exists to prevent. **If precision should gate the verdict, specify it as a
  sixth criterion before the next run, not after this one.**

- **Tumour volume grades *moderate*, not *fragile*, despite log-scale p = 0.97.**
  Under the specified split, scale robustness is one of two robustness checks;
  failing one of two is *moderate* by construction. Grading it *fragile* would
  require making scale robustness necessary rather than robustness — a
  defensible choice, but a different hierarchy from the one specified.

**Two of the four "rendering bugs" are not in the source.** The HTML held
`counted in patients, not in axis units` and `86/127/15/99` correctly; those
were line-wraps in whatever rendered the page. The token-splitting is real
though, so numeric cells are now `white-space: nowrap` and cannot break
mid-number. The other two (`34+62`, `4 of 4 … has`) were genuine and are fixed.

**The corrected single J is 0.232, not the 0.272 the brief inferred.** The
inference assumed the +0.008 delta was corrected-vs-corrected; it was
corrected-vs-uncorrected. That also reconciles it with section 4's per-metric
0.23 for tumour volume, which corrects for choosing a *cut-point* rather than
for choosing *which metric to report*.

**~~The multivariable models have no bootstrap-corrected calibration
intercept.~~ Fixed.** `model_validation.bootstrap_internal_validation` now
accumulates intercept optimism alongside slope optimism in the same loop, and
exports `intercept_corrected` plus a "Calibration intercept" metrics row. Both
notebooks were re-run.

The result is worth stating plainly because it is anticlimactic: **every
corrected intercept lands between −0.005 and +0.000.** Calibration-in-the-large
is anchored to the sample's own event rate, so it barely moves under
resampling. That is a real finding, not an empty column — these models are not
systematically over- or under-predicting risk, and the entire calibration
shortfall is in the slope. The report says so rather than presenting a column of
zeros.

Verified that nothing else moved: every model's AUC, Brier, calibration slope
and shrunken coefficients are byte-identical to the previous artifacts. MICE was
not re-run (the modelling notebook reads the cached draws), so the threshold
phase's stability numbers are unaffected.

---

## New artifacts

Tables `27`–`44` and figures `33`, `42`, `45` under `output/thresholds/`.
File numbers follow write order within the notebook, so the new tables added to
earlier sections (34–39) sort after the older ones from later sections. The
numbers are filename prefixes, not an outline.

| File | Contents |
|---|---|
| `27_evidence_criteria.csv` | The five pre-specified criteria and their rules |
| `28_threshold_evidence.csv` | Per metric: five criteria, grade, limiting criterion, non-scoring context |
| `29_threshold_evidence_reading_view.csv` | The graded verdict in reading order |
| `30`–`32_shared_combination_*.csv` | The whole §5 head-to-head on 304 patients |
| `33_shared_combination_space.svg` | Same, as a figure |
| `34_zero_inflation.csv` | Exact zeros per metric, risk on each side |
| `35_presence_rules.csv` | "Is there any at all?" scored as a yes/no test |
| `36_risk_curves_nonzero_only.csv` | Risk curves refitted on non-zero values |
| `37_zero_inflation_comparison.csv` | The three fits side by side |
| `38`–`39_nonlinearity_multiplicity*.csv` | Raw, Holm- and Bonferroni-adjusted p |
| `40`–`41_calibration*.csv` | Slope, intercept, Brier, calibration bins |
| `42_calibration.svg` | Calibration plots |
| `43`–`44_net_benefit*.csv` | Decision curve and per-strategy summary |
| `45_net_benefit.svg` | The decision curve |

## New modules and tests

`heavy_machinery/threshold_phase/evidence.py` (evidence hierarchy, Holm,
Bonferroni) and `calibration.py` (calibration, net benefit). Test count went
from 419 to 523, all passing.
