"""Published multivariable models, transcribed from their source papers.

Reference data only — nothing here is fitted, resampled, or derived from this
cohort. ``report.render_inferential`` prints it directly above the model this
pipeline fitted, so a reader can put the published odds ratio and ours side by
side and see where the two cohorts disagree.

Keyed by the ``model_id`` used in ``LITERATURE_MODEL_VARIANTS`` (modelling
notebook §02). Every number must be quoted from the paper as printed; leave a
field empty rather than filling it from a related figure, and leave
``performance`` empty when the paper reports no AUC or c-statistic for the
multivariable model. ``column`` is the column in *this* cohort that stands in
for the paper's variable — the honest place to record a mapping that is not
always one-to-one.
"""
from __future__ import annotations

# ``reference_number`` is the paper's position in the *manuscript's* reference
# list, which is what AJNR figures and tables cite — the journal numbers
# references in order of first appearance and prints the number as a superscript
# rather than an author-year. It lives here because nothing the pipeline
# computes can know it, and it must be re-checked whenever the reference list is
# renumbered. Verified against the manuscript on 2026-08-25.
PUBLISHED_MODELS: dict[str, dict] = {
    "radeesri_2023": {
        "reference_number": 8,
        "citation": (
            "Radeesri K, Lekhavat V. The Role of Pre-Operative MRI for Prediction "
            "of High-Grade Intracranial Meningioma: A Retrospective Study. "
            "Asian Pacific J Cancer Prev 2023;24(3):819–825."
        ),
        "cohort": (
            "210 of 327 consecutive intracranial meningiomas had a pre-operative "
            "MRI available; 50 of the 210 were high-grade. The paper does not state "
            "how many of those entered the multivariable model."
        ),
        "outcome": "High-grade meningioma (WHO grade 2–3, 2021 CNS classification).",
        # Empty on purpose: the paper reports no AUC, c-statistic, sensitivity or
        # specificity for the multivariable model, so there is nothing to compare
        # our validated AUC against.
        "performance": "",
        "terms": [
            {
                "variable": "Necrosis or hemorrhage",
                "meaning": (
                    "A non-enhancing patch inside the tumour (dead tissue), or "
                    "blooming on SWI/GRE showing old bleeding. Scored as one finding."
                ),
                "or": 2.94, "ci_lo": 1.15, "ci_hi": 7.48, "p": 0.024,
                "column": "necrosis_or_hemorrhage",
            },
            {
                "variable": "Hyperostosis",
                "meaning": (
                    "Thickened skull bone next to the tumour. The only protective "
                    "term in the model — it argued for low grade, not high."
                ),
                "or": 0.31, "ci_lo": 0.12, "ci_hi": 0.79, "p": 0.014,
                "column": "hyperostosis",
            },
            {
                "variable": "Brain edema",
                "meaning": (
                    "Peritumoral swelling: bright signal on T2 or FLAIR in the brain "
                    "immediately around the tumour."
                ),
                "or": 2.33, "ci_lo": 1.13, "ci_hi": 4.81, "p": 0.022,
                "column": "perifocal_edema",
            },
        ],
    },
    # ------------------------------------------------------------------
    # The six entries below are transcribed from the numbers Andris supplied
    # in literature_model_refit_spec.md (Downloads, 2026-08-16), which is the
    # origin of record for every figure here. Every OR/CI/p/beta/cohort figure
    # is quoted from that spec as printed; nothing here is rounded, derived, or
    # backfilled from a related figure. Where the spec itself has no number for
    # a field (a score weight rather than an OR, or a bound the paper never
    # published), the field is left empty rather than estimated.
    "spille_2020": {
        "reference_number": 7,
        "citation": (
            "Spille DC, Adeli A, Sporns PB, et al. Predicting the risk of "
            "postoperative recurrence and high-grade histology in patients with "
            "intracranial meningiomas using routine preoperative MRI. "
            "Neurosurg Rev 2021;44(2):1109–1117."
        ),
        "cohort": (
            "565 intracranial meningiomas; 516 grade 1, 49 (9%) "
            "high-grade. Single-centre, Münster, Germany. WHO 2016."
        ),
        "outcome": "High-grade meningioma (WHO 2016 grade 2–3).",
        # Empty on purpose: the spec states explicitly that no combined-model
        # AUC is reported — only the edema-volume single-predictor ROC AUC of
        # 0.645 (sens 0.70, spec 0.60), which is not the multivariable model's
        # own performance and so does not belong in this field.
        "performance": "",
        "terms": [
            {
                "variable": "Edema volume",
                "meaning": (
                    "Peritumoral edema volume in cm³, by the spheroid formula "
                    "V = 4/3·π·r₁·r₂·r₃ minus tumour volume. Multivariable "
                    "(adjusted) result quoted here; the paper's univariable OR "
                    "was also 1.00, p=.003."
                ),
                "or": 1.00, "ci_lo": 1.00, "ci_hi": 1.01, "p": 0.037,
                "column": "edema_volume_cm3",
            },
            {
                "variable": "Heterogeneous contrast enhancement",
                "meaning": (
                    "Heterogeneous versus homogeneous appearance on "
                    "gadolinium-enhanced T1. Multivariable (adjusted) result "
                    "quoted here; the paper's univariable OR was 3.10 "
                    "(1.67–5.78), p<.001."
                ),
                "or": 2.51, "ci_lo": 1.20, "ci_hi": 5.25, "p": 0.014,
                "column": "heterogeneous_enhancement",
            },
        ],
    },
    "zhang_2020": {
        "reference_number": 9,
        "citation": (
            "Zhang S, Chiang GC, Knapp JM, et al. Grading meningiomas utilizing "
            "multiparametric MRI with inclusion of susceptibility weighted "
            "imaging and quantitative susceptibility mapping. "
            "J Neuroradiol 2020;47(4):272–277."
        ),
        "cohort": (
            "129 meningiomas; 92 grade 1 (71.3%), 37 (28.7%) high-grade. "
            "Single-centre, Weill Cornell, USA. Morphological arm only — the "
            "SWI/QSM/ADC quantitative model is not refit here."
        ),
        "outcome": "High-grade meningioma (WHO grade 2–3).",
        # This is the anchor for the combined-vs-single research question this
        # refit exists to answer: the paper's own combined-model AUC beats its
        # best single predictor (peritumoral edema alone, AUC 0.66, 0.55–0.76)
        # by ΔAUC 0.05.
        "performance": (
            "Combined morphological model AUC 0.71 (95% CI 0.61–0.81). "
            "Best single predictor alone (peritumoral edema) AUC 0.66 "
            "(0.55–0.76) — combined beats best single by ΔAUC 0.05, the "
            "benchmark this refit's comparison is measured against. No "
            "internal-validation resampling reported."
        ),
        # beta is the log-odds coefficient from their multivariable logistic
        # model of the morphological arm — no SE is published, so no CI can be
        # derived from it, and it is not exponentiated into an OR here.
        "terms": [
            {
                "variable": "Calcification",
                "meaning": "Visible tumour calcification, present/absent.",
                # Reports β only, no ORs/CIs — do not exponentiate to invent one.
                "beta": 0.874, "or": "", "ci_lo": "", "ci_hi": "", "p": 0.110,
                "column": "calcification",
            },
            {
                "variable": "Peritumoral edema",
                "meaning": "Peritumoral edema, present/absent.",
                "beta": 0.554, "or": "", "ci_lo": "", "ci_hi": "", "p": 0.042,
                "column": "perifocal_edema",
            },
            {
                "variable": "Tumor border",
                "meaning": (
                    "Ill-defined versus well-defined tumour margin — their "
                    "'tumor border', mapped to this cohort's irregular tumour "
                    "margin flag."
                ),
                "beta": 0.862, "or": "", "ci_lo": "", "ci_hi": "", "p": 0.024,
                "column": "irregular_tumor_margin",
            },
            {
                "variable": "Tumor location (non-skull base = risk direction)",
                "meaning": (
                    "Convexity versus other location. Substituted here with "
                    "skull base versus non-skull base — directionally "
                    "equivalent; non-skull-base is the high-grade risk "
                    "direction in both this paper and Peng 2021, so their "
                    "positive β (+0.545) is a non-skull-base effect. This "
                    "cohort's skull_base_location is coded the opposite way "
                    "(True = at skull base) — see peng_2021's note on the "
                    "same column for the reciprocal-OR treatment; no "
                    "reciprocal is computed here since this paper reports β, "
                    "not an OR, and the sign flip is stated instead."
                ),
                "beta": 0.545, "or": "", "ci_lo": "", "ci_hi": "", "p": 0.039,
                "column": "skull_base_location",
            },
        ],
    },
    "funari_2023": {
        "reference_number": 10,
        "citation": (
            "Funari A, De la Garza Ramos R, Cezayirli P, et al. Imaging score "
            "for differentiation of meningioma grade. "
            "Neuroradiology 2023;65(3):453–462."
        ),
        "cohort": (
            "90 meningiomas; 45 grade 1, 45 grade 2. Single-centre, "
            "Montefiore/Einstein, Bronx, USA. Mean age 59.9, 70% female."
        ),
        "outcome": "WHO grade 2 meningioma (grade 3 not included in their cohort).",
        "performance": (
            "Point score ≥3 of 5: recall 0.80, precision 0.73, F1 0.77, "
            "accuracy 0.76, AUC 0.82. Calibration by score: 0 pts → 0% grade 2; "
            "1 → 25.0%; 2 → 38.5%; 3 → 65.4%; ≥4 → 83.3%. No internal-"
            "validation resampling; no per-single-predictor AUCs reported."
        ),
        "terms": [
            {
                "variable": "Tumor volume ≥36.0 cc",
                "meaning": (
                    "Volumetric tumour measurement, dichotomised at 36 cc in "
                    "the paper's score (worth 2 of their 5 points). Refit here "
                    "with tumor_volume left continuous — their cut-point is not "
                    "imported, so no odds ratio from their score applies to "
                    "this parameterisation."
                ),
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "tumor_volume",
            },
            {
                "variable": "Irregular tumor borders",
                "meaning": "Present/absent; worth 2 of the paper's 5 score points.",
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "irregular_tumor_margin",
            },
            {
                "variable": "Peritumoral edema",
                "meaning": "Present/absent; worth 1 of the paper's 5 score points.",
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "perifocal_edema",
            },
        ],
    },
    "kawahara_2012": {
        "reference_number": 11,
        "cross_reference": (
            "Capsular enhancement points the OPPOSITE way in Lin et al. 2014, "
            "also refit here. These authors scored its ABSENCE as the high-grade "
            "finding (negative capsular enhancement, univariable OR 19.2, "
            "5.4-69) and did not retain it in their multivariable model; Lin "
            "scores its PRESENCE at 3 of 12 points toward high grade. Both are "
            "transcribed correctly — the two papers genuinely disagree on the "
            "direction of this sign. Our own fitted estimate for the same "
            "column appears in Lin's odds-ratio table below."
        ),
        "citation": (
            "Kawahara Y, Nakada M, Hayashi Y, et al. Prediction of high-grade "
            "meningioma by preoperative MRI assessment. "
            "J Neurooncol 2012;108(1):147–152."
        ),
        "cohort": "65 meningiomas; 39 benign, 26 high-grade. Single-centre, Kanazawa University, Japan. WHO 2000.",
        "outcome": "High-grade meningioma.",
        # Transcribed from the publisher PDF (obtained 2026-08-17). Odds ratios
        # below are Table 3; the logistic equation and its four calibration
        # probabilities are Table 4. Equation: z = -1.979 + 3.738*TBI +
        # 2.112*heterogeneity. The sign of the second term reads ambiguously
        # in the extracted text; it is PLUS, confirmed two ways —
        # reconstructing all four of these probabilities exactly, and
        # exp(3.738)=42.0, exp(2.112)=8.3 matching the Table 3 odds ratios.
        # Interface has the larger aOR (42.0 vs 8.3), so it must give the
        # larger of the two single-factor probabilities — 85.3%, not 53.3%.
        "performance": (
            "Published logistic equation z = -1.979 + 3.738*TBI + "
            "2.112*heterogeneity. Probability of high grade: 12.1% with "
            "neither factor present, 85.3% with unclear interface alone, "
            "53.3% with heterogeneous enhancement alone, 98.0% with both."
        ),
        "terms": [
            {
                "variable": "Unclear tumor-brain interface",
                "meaning": (
                    "Absence of a clear demarcation or CSF cleft between "
                    "tumour and adjacent brain. Substituted here with "
                    "irregular tumour margin — a related but non-identical "
                    "construct; see the surrogate note below."
                ),
                "or": 42.0, "ci_lo": 4.5, "ci_hi": 390, "p": 0.001,
                "column": "irregular_tumor_margin",
            },
            {
                "variable": "Heterogeneous enhancement",
                "meaning": (
                    "Heterogeneous versus homogeneous appearance on "
                    "post-contrast T1."
                ),
                "or": 8.3, "ci_lo": 1.7, "ci_hi": 40.4, "p": 0.009,
                "column": "heterogeneous_enhancement",
            },
        ],
        "surrogate_note": (
            "Refit with irregular_tumor_margin standing in for the paper's "
            "tumour-brain interface. These authors scored the two as SEPARATE "
            "factors and published both: unclear interface OR 71.8 (8.4-612) "
            "and irregular tumoral margin OR 10.3 (3.2-33) on univariable "
            "analysis. Their multivariable model kept the interface "
            "and dropped the margin. Our substitute is therefore the weaker of "
            "the two on their own data, by roughly sevenfold, and the one they "
            "specifically discarded. Expect this refit to underperform the "
            "published model for that reason alone. They also assessed negative "
            "capsular enhancement (OR 19.2, 5.4-69) and did not retain it. This "
            "is a refit, not an external validation."
        ),
    },
    "lin_2014": {
        "reference_number": 12,
        "cross_reference": (
            "Capsular enhancement points the OPPOSITE way in Kawahara et al. "
            "2012, also refit here. Lin scores its PRESENCE at 3 of 12 points "
            "toward high grade; Kawahara scored its ABSENCE as the high-grade "
            "finding (negative capsular enhancement, univariable OR 19.2, "
            "5.4-69) and dropped it from their multivariable model. Both are "
            "transcribed correctly — the two papers genuinely disagree on the "
            "direction of this sign. See this model's own odds-ratio table "
            "below for what this cohort makes of it."
        ),
        "citation": (
            "Lin BJ, Chou KN, Kao HW, et al. Correlation between magnetic "
            "resonance imaging grading and pathological grading in meningioma. "
            "J Neurosurg 2014;121(5):1201–1208."
        ),
        "cohort": "120 meningiomas; 90 grade 1, 30 grade 2/3. Single-centre, Taiwan.",
        "outcome": "High-grade meningioma (WHO grade 2–3).",
        # Empty on purpose: this is a summed weighted score (12 points total),
        # not a logistic regression — the paper reports no AUC or c-statistic
        # for it, and no per-term OR/CI/p exists to transcribe.
        "performance": "",
        "terms": [
            {
                "variable": "Age ≥75 years",
                "meaning": "Patient age dichotomised at 75. Worth 2 of the scale's 12 points.",
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "age_ge75",
            },
            {
                "variable": "Indistinct tumor-brain interface",
                "meaning": (
                    "Loss of clear tumour-parenchyma demarcation — the "
                    "scale's heaviest-weighted term, 5 of its 12 points. "
                    "Substituted here with irregular tumour margin; see the "
                    "surrogate note below."
                ),
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "irregular_tumor_margin",
            },
            {
                "variable": "Capsular enhancement",
                "meaning": "Enhancing rim/layer around the tumour. Worth 3 of the scale's 12 points.",
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "capsular_enhancement",
            },
            {
                "variable": "Heterogeneous tumor enhancement",
                "meaning": "Heterogeneous versus homogeneous post-contrast enhancement. Worth 2 of the scale's 12 points.",
                "or": "", "ci_lo": "", "ci_hi": "", "p": "",
                "column": "heterogeneous_enhancement",
            },
        ],
        "surrogate_note": (
            "Refit with irregular_tumor_margin standing in for the paper's "
            "indistinct tumour-brain interface — the heaviest-weighted term in "
            "their 12-point scale (5 points, more than any other factor). Our "
            "dataset scores irregular tumour border/contour, not CSF-cleft "
            "clarity or tumour-parenchyma demarcation specifically; these are "
            "related but non-identical constructs, and irregular margin is "
            "present in 157/352 (45%) of this cohort. See kawahara_2012's note "
            "for this substitution's published effect sizes on another cohort. "
            "This is a refit, not an external validation."
        ),
    },
    "peng_2021": {
        "reference_number": 13,
        "citation": (
            "Peng S, Cheng Z, Guo Z. Diagnostic nomogram model for predicting "
            "preoperative pathological grade of meningioma. "
            "Transl Cancer Res 2021;10(9):4057–4064."
        ),
        "cohort": (
            "215 meningiomas; 168 (78%) grade 1, 47 (22%) high-grade. "
            "Single-centre, Shanghai Ninth People's Hospital, China. WHO 2016."
        ),
        "outcome": "High-grade meningioma (WHO 2016 grade 2–3).",
        "performance": (
            "Combined nomogram C-index 0.874 (95% CI 0.818–0.929); 0.868 on "
            "1000-resample bootstrap. Hosmer-Lemeshow p=.378 (good "
            "calibration). Decision-curve analysis: highest net benefit at a "
            "29% threshold probability. Best single predictor alone (tumour-"
            "brain interface) C 0.779 — combined beats best single by ΔC 0.095, "
            "the second benchmark this refit's comparison is measured against."
        ),
        "terms": [
            {
                "variable": "Tumour-brain interface",
                "meaning": (
                    "Clear = clear peritumoural rim, regular shape without "
                    "obvious lobules, distinct CSF cleft at the margin; "
                    "anything else scored unclear. Unclear in 48/215 (22%) of "
                    "their cohort, 65% of those high-grade. Substituted here "
                    "with irregular tumour margin; see the surrogate note "
                    "below. Single-predictor AUC 0.779, sens 66%, spec 90%."
                ),
                "or": "", "ci_lo": "", "ci_hi": "", "p": "<0.001",
                "column": "irregular_tumor_margin",
            },
            {
                "variable": "Bone invasion",
                "meaning": (
                    "Bone destruction on non-contrast CT, present "
                    "preoperatively in 70% of their high-grade cases. "
                    "Single-predictor AUC 0.729."
                ),
                "or": "", "ci_lo": "", "ci_hi": "", "p": "<0.001",
                "column": "cortical_destruction",
            },
            {
                "variable": "Tumour location (non-skull base = risk direction)",
                "meaning": (
                    "Skull base versus non-skull base; non-skull base is the "
                    "risk-increasing direction in their coding. This cohort's "
                    "skull_base_location is coded the opposite way (True = at "
                    "skull base). As published for non-skull base: OR 3.042 "
                    "(1.223-7.567). Re-expressed for this cohort's coding, "
                    "where True means at the skull base, that is its "
                    "reciprocal: OR 0.33 (0.13-0.82). Our fitted estimate "
                    "should be compared against 0.33, not 3.042 — this is "
                    "algebra on the published estimate (1/OR, 1/ci_hi, "
                    "1/ci_lo), not a new number; the or/ci_lo/ci_hi fields "
                    "below remain the published 3.042 (1.223-7.567) as "
                    "printed. Single-predictor AUC 0.583."
                ),
                "or": 3.042, "ci_lo": 1.223, "ci_hi": 7.567, "p": 0.017,
                "column": "skull_base_location",
            },
        ],
        "surrogate_note": (
            "Refit with irregular_tumor_margin standing in for the paper's "
            "tumour-brain interface — their strongest single predictor, "
            "AUC 0.779, unclear in only 22% of their cohort versus 157/352 "
            "(45%) irregular margin in this one. Our variable fires on a "
            "broader set of tumours than theirs. See kawahara_2012's note for "
            "this substitution's published effect sizes on another cohort. "
            "This is a refit, not an external validation."
        ),
    },
}




def published_model(model_id: str) -> dict | None:
    """The published model for ``model_id``, or None when none is recorded."""
    return PUBLISHED_MODELS.get(str(model_id or ""))
