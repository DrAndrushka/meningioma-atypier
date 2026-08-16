"""S1, S2, the number formatting they share, and the HTML proof sheet."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heavy_machinery.config import load as _load_config  # noqa: F401  (sys.path)

import ajnr_format as fm
import manuscript_tables as mt
import report_html as rh


# --- formatting ------------------------------------------------------------
def test_p_values_drop_the_leading_zero_and_extremes_become_bounds():
    assert fm.fmt_p(0.042) == ".04"
    assert fm.fmt_p(0.0065) == ".006"
    assert fm.fmt_p(0.0000001) == "<.001"
    assert fm.fmt_p(0.999) == ">.99"


def test_estimates_intervals_and_units_keep_their_own_precision():
    """Only P values drop the leading zero; applying that rule to an odds ratio
    is the slip. 15.1 must not become 15.100, and 0.0617 must keep its digits.
    """
    assert fm.fmt_est(0.72) == "0.72"
    assert fm.fmt_est_ci(0.68, 0.61, 0.74) == "0.68 (0.61–0.74)"
    assert fm.fmt_ci(0.61, 0.74) == "0.61–0.74"
    assert "-" not in fm.fmt_ci(0.61, 0.74)
    assert fm.fmt_value(15.1, 1) == "15.1"
    assert fm.fmt_value(0.0617, 4) == "0.0617"
    assert fm.fmt_pct(1.41) == "141%"


def test_a_missing_value_is_an_em_dash_not_a_nan():
    for value in (None, np.nan, float("inf"), "abc"):
        assert fm.fmt_p(value) == fm.BLANK
        assert fm.fmt_est(value) == fm.BLANK
    assert (fm.yes_no(True), fm.yes_no(False), fm.yes_no(None)) == (
        "Yes", "No", fm.BLANK)


# --- the inputs ------------------------------------------------------------
def _inputs(**over):
    base = {
        "eligible": pd.DataFrame([{"col": "adc_value", "stratum": "all"}]),
        "agreement": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                    "cutoff_min": 0.72, "cutoff_max": 0.79,
                                    "spread_vs_iqr": 0.41}]),
        "wobble": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                 "cutpoint": 0.72, "ci_lo": 0.69, "ci_hi": 0.85,
                                 "stability_ratio": 0.93}]),
        "imputation": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                     "diverges": False, "draw_min": 0.72,
                                     "draw_max": 0.73}]),
        "dichotomy": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                    "cutoff": 0.72, "n": 309,
                                    "auc_continuous": 0.627,
                                    "auc_continuous_lo": 0.555,
                                    "auc_continuous_hi": 0.694,
                                    "auc_binary": 0.618, "auc_binary_lo": 0.564,
                                    "auc_binary_hi": 0.670,
                                    "information_retained": 0.93,
                                    "auc_loss_p": 0.72}]),
    }
    base.update(over)
    return base


def _s1(**over):
    kw = _inputs(**over)
    return mt.supplemental_s1(kw["eligible"], agreement=kw["agreement"],
                              wobble=kw["wobble"], imputation=kw["imputation"])


def _s2(**over):
    kw = _inputs(**over)
    return mt.supplemental_s2(kw["eligible"], dichotomy=kw["dichotomy"])


# --- S1 --------------------------------------------------------------------
def test_s1_locates_the_cutpoint_when_all_three_checks_hold():
    """A ratio alone says nothing about where the cut-point actually sits."""
    row = _s1().loc["ADC (mean)"]
    assert "0.69–0.85" in row["Bootstrap 95% interval"]
    assert "0.93× IQR" in row["Bootstrap 95% interval"]
    assert row["Published cut-point"] == "≤ 0.72 ×10⁻³ mm²/s"
    assert row["Cut-point locatable"] == "Yes"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        # A wide bootstrap interval alone is enough to lose it ...
        ({"wobble": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                   "cutpoint": 0.72, "ci_lo": 0.4, "ci_hi": 1.2,
                                   "stability_ratio": 1.4}])}, "No"),
        # ... and so are scattered selection rules ...
        ({"agreement": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                      "cutoff_min": 0.5, "cutoff_max": 1.1,
                                      "spread_vs_iqr": 0.8}])}, "No"),
        # ... and divergence across imputations.
        ({"imputation": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                       "diverges": True, "draw_min": 0.9,
                                       "draw_max": 1.1}])}, "No"),
        # A check that could not be run leaves no verdict at all.
        ({"imputation": pd.DataFrame()}, fm.BLANK),
    ],
)
def test_s1_needs_every_stability_check_to_hold(override, expected):
    assert _s1(**override).loc["ADC (mean)", "Cut-point locatable"] == expected


# --- S2 --------------------------------------------------------------------
def test_s2_reports_both_aucs_and_accepts_a_small_insignificant_loss():
    row = _s2().loc["ADC (mean)"]
    assert row["AUC as a number"] == "0.63 (0.56–0.69)"
    assert row["AUC as yes/no"] == "0.62 (0.56–0.67)"
    assert row["Acceptable loss"] == "Yes"


def test_s2_rejects_a_large_loss_and_a_significant_one_alike():
    """The two conditions catch different failures, so both are required."""
    dich = _inputs()["dichotomy"].assign(information_retained=0.70,
                                         auc_loss_p=0.40)
    assert _s2(dichotomy=dich).loc["ADC (mean)", "Acceptable loss"] == "No"

    dich = _inputs()["dichotomy"].assign(information_retained=0.95,
                                         auc_loss_p=0.01)
    assert _s2(dichotomy=dich).loc["ADC (mean)", "Acceptable loss"] == "No"

    # Retention above 100% is shown as it is, not clipped.
    dich = _inputs()["dichotomy"].assign(information_retained=1.41)
    assert _s2(dichotomy=dich).loc["ADC (mean)",
                                   "Discrimination retained"] == "141%"


# --- the footnotes ---------------------------------------------------------
def test_each_footnote_states_the_rule_and_thresholds_behind_its_verdict():
    """One definition, used by the code and printed in the note."""
    assert "Cut-point locatable: Yes when" in mt.s1_footnote()
    assert "Acceptable loss: Yes when" in mt.s2_footnote()
    assert f"{mt.SPREAD_LIMIT:.2f} × IQR" in mt.s1_footnote()
    assert f"{mt.BOOTSTRAP_LIMIT:.2f} × IQR" in mt.s1_footnote()
    assert f"{mt.RETAINED_FLOOR:.0%}" in mt.s2_footnote()
    assert "above 100%" in mt.s2_footnote()
    assert mt.s1_footnote().startswith("Note:—")
    assert mt.s2_footnote().startswith("Note:—")


# --- the HTML --------------------------------------------------------------
def test_the_page_is_self_contained_and_reproducible():
    """No fetches: it has to survive being emailed."""
    page = rh.build([rh.table_section("T", _s1(), note="n", index_header="M")],
                    title="Demo", generated_at="fixed")
    for token in ("<script src=", 'href="http', '<link rel="stylesheet"'):
        assert token not in page
    assert "<style>" in page

    page = rh.build([rh.table_section(mt.S1_TITLE, _s1(), note=mt.s1_footnote(),
                                      index_header="Measurement")],
                    title="Demo", generated_at="fixed")
    assert "Table S1." in page
    assert "Cut-point locatable: Yes when" in page

    # Pinning the timestamp makes the page diffable.
    assert rh.build([], title="T", generated_at="fixed") == \
        rh.build([], title="T", generated_at="fixed")


def test_rendered_cells_are_graded_and_escaped():
    page = rh.render_table(_s1(), index_header="Measurement")
    assert 'class="yes"' in page

    frame = pd.DataFrame([["<script>x</script>"]], index=["r"], columns=["c"])
    assert "<script>" not in rh.render_table(frame)


def test_sections_embed_their_figure_or_degrade_to_a_placeholder(tmp_path):
    page = rh.placeholder_section("Table 1", columns=["a", "b"], reason="why")
    assert "Not yet built" in page and "<li>a</li>" in page and "why" in page

    assert "Not yet built" in rh.figure_section("Fig", tmp_path / "absent.png")

    image = tmp_path / "f.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    page = rh.figure_section("Fig", image)
    assert "data:image/png;base64," in page
    assert str(image) not in page


# --- against the real cohort ----------------------------------------------
def test_the_real_s1_and_s2_cover_every_eligible_measurement(real_cohort):
    import bend_location as bl
    import criteria as cr
    import dichotomy as di
    import eligibility as el
    import imputation as imp
    import nonlinearity as nl
    import separation as sep
    import wobble as wb

    frozen = {"adc_value": 0.72, "max_diameter_cm": 3.81, "tumor_volume": 15.1,
              "edema_volume_cm3": 4.76, "edema_index": 0.0617}
    fits = nl.fit_all(real_cohort)
    separation = sep.separation_table(real_cohort)
    bend = bl.bend_table(real_cohort, fits=fits)
    eligible = el.eligible(el.carry_forward(separation, bend))
    wobble, _ = wb.wobble_table(real_cohort, eligible, n_boot=60, frozen=frozen)

    s1 = mt.supplemental_s1(
        eligible,
        agreement=cr.agreement(cr.criteria_table(real_cohort, eligible),
                               real_cohort),
        wobble=wobble,
        imputation=imp.per_draw_cutpoints(imp.load_draws("output"), eligible,
                                          frozen=frozen))
    s2 = mt.supplemental_s2(
        eligible, dichotomy=di.dichotomy_table(real_cohort, eligible, frozen))

    assert len(s1) == len(s2) == 5
    assert not (s1 == "nan").to_numpy().any()
    assert not (s2 == "nan").to_numpy().any()
    assert set(s1["Cut-point locatable"]) <= {"Yes", "No"}
    assert set(s2["Acceptable loss"]) <= {"Yes", "No"}


# --- the dashboard cards ---------------------------------------------------
def _verdict_inputs(**over):
    kw = _inputs(**over)
    base = {
        "nonlinearity": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                       "bent_clinical": True, "lr_p": 0.009,
                                       "lr_p_log": 0.02, "scales_agree": True}]),
        "segmented": pd.DataFrame([{"col": "adc_value", "stratum": "all",
                                    "breakpoint": 0.77, "ci_lo": 0.71,
                                    "ci_hi": 0.89, "davies_p": 0.019,
                                    "delta_aic": -4.96, "n": 309,
                                    "breakpoint_supported": True}]),
        "presence": pd.DataFrame(),
    }
    base.update({k: v for k, v in over.items() if k in
                 ("nonlinearity", "segmented", "presence")})
    kw["dichotomy"] = kw["dichotomy"].assign(
        or_per_sd=0.62, or_per_sd_lo=0.47, or_per_sd_hi=0.83, or_per_sd_p=0.0014,
        or_binary=4.12, or_binary_lo=2.28, or_binary_hi=7.45, auc_loss=0.009)
    return kw["eligible"], kw["dichotomy"], base


def _verdicts(**over):
    eligible, dichotomy, base = _verdict_inputs(**over)
    return mt.threshold_verdicts(eligible, dichotomy=dichotomy,
                                 wobble=_inputs(**over)["wobble"], **base)


def test_a_card_that_meets_everything_says_so_and_sorts_first():
    entry = _verdicts()[0]
    assert entry["supported"]
    assert entry["criteria_line"] == f"All {mt.CRITERIA_TOTAL} criteria met"
    assert entry["reason"] == ""          # nothing to explain away

    entries = _verdicts() + [dict(entry, supported=False, measurement="Zzz")]
    ordered = sorted(entries, key=lambda e: (not e["supported"],
                                             e["measurement"]))
    assert ordered[0]["supported"] and not ordered[-1]["supported"]


def test_a_failing_card_counts_and_names_what_it_missed():
    """The most fundamental failure comes first: with no bend at all, the later
    tests have nothing to be about."""
    nonlin = pd.DataFrame([{"col": "adc_value", "stratum": "all",
                            "bent_clinical": False, "lr_p": 0.50,
                            "lr_p_log": 0.77, "scales_agree": True}])
    seg = pd.DataFrame([{"col": "adc_value", "stratum": "all",
                         "breakpoint": 6.25, "ci_lo": 2.07, "ci_hi": 6.40,
                         "davies_p": 0.30, "delta_aic": 1.2, "n": 352,
                         "breakpoint_supported": False}])
    entry = _verdicts(nonlinearity=nonlin, segmented=seg)[0]
    assert entry["criteria_line"] == f"1 of {mt.CRITERIA_TOTAL} criteria met"
    assert entry["failures"] == ["no bend", "break no better than chance",
                                 "break too small to matter"]
    assert entry["reason"].startswith("Not met: no bend;")

    nonlin = pd.DataFrame([{"col": "adc_value", "stratum": "all",
                            "bent_clinical": False, "lr_p": 0.50,
                            "lr_p_log": 0.77, "scales_agree": False}])
    assert _verdicts(nonlinearity=nonlin)[0]["failures"][0] == "no bend"


def test_the_works_line_names_the_better_form():
    # The DeLong comparison is not significant here, so neither form wins.
    assert _verdicts()[0]["works"] == (
        "Works both ways, with no measurable difference between the two.")

    _, dich, base = _verdict_inputs()
    entry = mt.threshold_verdicts(
        _verdict_inputs()[0],
        dichotomy=dich.assign(auc_loss=0.042, auc_loss_p=0.010), **base)[0]
    assert entry["works"] == "Works both ways, better as a number."

    entry = mt.threshold_verdicts(
        _verdict_inputs()[0],
        dichotomy=dich.assign(auc_loss=-0.040, auc_loss_p=0.004), **base)[0]
    assert entry["works"] == "Works both ways, better as a yes/no."

    # A missing column must not raise; it means the question was not asked.
    entry = mt.threshold_verdicts(_verdict_inputs()[0],
                                  dichotomy=dich.drop(columns=["auc_loss"]),
                                  **base)[0]
    assert "no measurable difference" in entry["works"]

    # Edema index: presence predicts, amount does not.
    sign_not_dose = dich.assign(
        or_per_sd=1.09, or_per_sd_lo=0.87, or_per_sd_hi=1.38, or_per_sd_p=0.45,
        or_binary=2.65, or_binary_lo=1.56, or_binary_hi=4.50)
    presence = pd.DataFrame([{"col": "adc_value", "stratum": "all",
                              "presence_matters": True, "amount_matters": False,
                              "presence_or": 2.35, "presence_lo": 1.36,
                              "presence_hi": 4.04, "amount_or": 0.85,
                              "amount_lo": 0.63, "amount_hi": 1.14}])
    eligible, _, presence_base = _verdict_inputs(presence=presence)
    entry = mt.threshold_verdicts(eligible, dichotomy=sign_not_dose,
                                  **presence_base)[0]
    assert entry["works"] == ("Works as a yes/no only — having any predicts, "
                              "how much does not.")

    # A null measurement without a zero pile gets the generic line.
    null = dich.assign(
        or_per_sd=1.02, or_per_sd_lo=0.80, or_per_sd_hi=1.30, or_per_sd_p=0.87,
        or_binary=1.10, or_binary_lo=0.70, or_binary_hi=1.72)
    entry = mt.threshold_verdicts(_verdict_inputs()[0], dichotomy=null,
                                  **base)[0]
    assert entry["works"] == "Predicts neither as a number nor as a yes/no."


def test_a_card_renders_its_thumbnail_inline_when_it_has_one():
    entry = dict(_verdicts()[0], thumbnail="ZmFrZQ==")
    card = rh._card(entry)
    assert 'class="spark"' in card and "data:image/png;base64,ZmFrZQ==" in card
    assert 'class="spark"' not in rh._card(_verdicts()[0])
