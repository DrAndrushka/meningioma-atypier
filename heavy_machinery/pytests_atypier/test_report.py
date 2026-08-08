"""Tests for report.py — HTML sections, glossaries, inferential layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import report as rp
from report import (
    Artifacts,
    MissingThresholds,
    ReportConfig,
    build_report,
    classify_missing,
    classify_or_direction,
    classify_significance,
    details_block,
    human_p,
    human_pool_df,
    info_box,
    load_artifacts,
    main,
    render_appendix,
    render_cleaning,
    render_dda,
    render_eda,
    render_header,
    render_inferential,
    render_marker_panel,
    render_missingness,
    render_schema,
    svg_grid,
    table_to_html,
    warning_box,
    write_html,
)


@pytest.fixture
def report_cfg(tmp_output):
    return ReportConfig(output_root=tmp_output, title="Test", targets=("event",))


@pytest.fixture
def report_art(tmp_output):
    (tmp_output / "cleaning").mkdir(parents=True)
    pd.DataFrame([{"step": "final", "n_rows": 4}]).to_csv(
        tmp_output / "cleaning" / "cleaning_summary.csv", index=False,
    )
    (tmp_output / "dda" / "tables").mkdir(parents=True)
    pd.DataFrame([{"n_rows": 4, "n_cols": 3}]).to_csv(
        tmp_output / "dda" / "tables" / "dda_overall.csv", index=False,
    )
    return Artifacts(output_root=tmp_output, cleaning_summary=pd.read_csv(
        tmp_output / "cleaning" / "cleaning_summary.csv",
    ))


def test_embed_svg_src(tmp_path):
    p = tmp_path / "x.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    assert rp._embed_svg_src(p).startswith("data:image/svg+xml;base64,")


def test_figure_img_html(tmp_path):
    p = tmp_path / "x.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    html = rp._figure_img_html(p)
    assert "<img" in html


def test_esc():
    assert rp._esc("<b>") == "&lt;b&gt;"
    assert rp._esc(None) == ""


def test_human_pool_df():
    assert human_pool_df(12.0) == "12"
    assert human_pool_df(float("inf")) == "∞"


def test_human_p():
    assert human_p(0.0001) == "<0.001"
    assert human_p("<0.001") == "<0.001"


def test_coerce_p():
    assert rp._coerce_p("<0.001") is not None
    assert rp._coerce_p(None) is None


def test_coerce_float():
    assert rp._coerce_float("1.5") == 1.5
    assert rp._coerce_float("bad") is None


def test_classify_significance():
    assert classify_significance(0.01, 0.02) == "sig-fdr"


def test_classify_or_direction():
    assert classify_or_direction(2.0, 1.2, 3.0) == "or-risk"


def test_classify_missing():
    assert classify_missing(50.0) == "missing-severe"


def test_warning_box():
    assert "warning-box" in warning_box("oops")


def test_info_box():
    assert "info-box" in info_box("note")


def test_table_to_html():
    df = pd.DataFrame({"a": [1], "b": [2]})
    html = table_to_html(df)
    assert "<table" in html


def test_svg_grid(tmp_path):
    p = tmp_path / "x.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    assert "figure-grid" in svg_grid([p])


def test_details_block():
    assert "<details" in details_block("sum", "<p>x</p>")


def test_maybe_read_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n")
    warns = []
    df = rp._maybe_read_csv(p, warns)
    assert df is not None and len(df) == 1


def test_load_schema_any(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text(json.dumps({"age": {"kind": "continuous"}}))
    warns = []
    df = rp._load_schema_any(p, warns)
    assert df is not None and "column" in df.columns


def test_load_artifacts(report_cfg, tmp_output):
    art = load_artifacts(report_cfg)
    assert art.output_root == tmp_output


def test_format_authors():
    assert rp._format_authors("") == ""
    assert rp._format_authors("Jane Doe") == "Jane Doe"
    assert rp._format_authors("Jane Doe, John Smith") == "Jane Doe, John Smith"
    assert rp._format_authors("A, B, C") == "A, B, C"
    assert rp._format_authors("A, B, and C") == "A, B, C"
    six = "Arturs Balodis, Sigita Zālīte, Roberts Tumeļkāns, Valērija Aksjonova, Elizabete Stankeviča, Andris Zaguzovs"
    assert rp._format_authors(six).endswith(", Andris Zaguzovs")


def test_render_header(report_cfg, report_art):
    html = render_header(report_cfg, report_art)
    assert "Test" in html
    report_cfg.author = "Alice, Bob, Carol"
    html = render_header(report_cfg, report_art)
    assert 'class="report-authors"' in html
    assert "Alice, Bob, Carol" in html
    assert "Author" not in html or 'class="label">Author<' not in html


def test_render_cleaning(report_cfg, report_art):
    assert "<section" in render_cleaning(report_cfg, report_art)


def test_cohort_flow_table_uses_log_counts_and_shows_zero_duplicates():
    art = Artifacts(
        output_root=Path("."),
        cleaning_summary=pd.DataFrame([
            {"step": "raw_data", "detail": "rows", "n_rows": 10, "n_columns": 2, "criterion": ""},
            {"step": "duplicate_audit",
             "detail": "no duplicates found",
             "n_rows": 10, "n_columns": 2, "criterion": ""},
            {"step": "drop_rows", "detail": "grade exists", "n_rows": 9,
             "n_columns": 2, "criterion": "WHO grade recorded"},
            {"step": "final", "detail": "done", "n_rows": 9, "n_columns": 2, "criterion": ""},
        ]),
        cleaning_log=pd.DataFrame([
            {"step": "drop_rows", "reason": "grade exists", "criterion": "WHO grade recorded",
             "n_before": 10.0, "n_dropped": 1.0, "n_remaining": 9.0},
        ]),
    )
    html = rp._cohort_flow_table(art)
    # Duplicate audit leads the table even with nothing found.
    assert "Duplicate ID audit" in html
    assert "no duplicates found" in html
    # n_before/n_dropped come from the log, as ints not floats.
    assert "<td>10</td>" in html and "<td>1</td>" in html
    assert "10.0" not in html
    assert "WHO grade recorded" in html
    assert "Analysed cohort" in html


def test_cohort_flow_table_falls_back_to_summary_without_log():
    art = Artifacts(
        output_root=Path("."),
        cleaning_summary=pd.DataFrame([
            {"step": "drop_rows", "detail": "grade exists", "n_rows": 9,
             "n_columns": 2, "criterion": "WHO grade recorded"},
        ]),
    )
    html = rp._cohort_flow_table(art)
    assert "grade exists" in html and "WHO grade recorded" in html


def test_derived_tables_split_added_from_recoded():
    log = pd.DataFrame([
        {"derivation": "high_grade", "source": "who_grade", "kind": "binary",
         "rule": "who_grade in {2, 3}", "rows_nonmissing": 9, "rows_missing": 0,
         "schema_action": "added ColSpec (binary) for high_grade", "reason": "WHO 2021."},
        {"derivation": "edema_volume_cm3", "source": "perifocal_edema", "kind": "continuous",
         "rule": "set to 0 where edema absent", "rows_nonmissing": 8, "rows_missing": 1,
         "schema_action": "updated ColSpec (continuous) for edema_volume_cm3",
         "reason": "Structural zero."},
        {"derivation": "skipped_one", "source": "nope", "kind": "binary", "rule": "",
         "rows_nonmissing": "", "rows_missing": "",
         "schema_action": "skipped (inactive)", "reason": ""},
    ])
    html = rp._derived_tables(log)
    assert "Derived variables" in html and "Recoded variables" in html
    assert "who_grade in {2, 3}" in html
    assert "set to 0 where edema absent" in html
    # Skipped derivations belong to the raw log only, not these tables.
    assert "skipped_one" not in html


def test_cleaning_provenance_reports_each_shape():
    art = Artifacts(
        output_root=Path("."),
        cleaning_summary=pd.DataFrame([
            {"step": "raw_data", "detail": "rows", "n_rows": 10, "n_columns": 5, "criterion": ""},
            {"step": "apply_schema", "detail": "coerced dtypes", "n_rows": 10,
             "n_columns": 4, "criterion": ""},
            {"step": "final", "detail": "done", "n_rows": 9, "n_columns": 6, "criterion": ""},
        ]),
    )
    html = rp._cleaning_provenance(art)
    assert "10 rows × 5 columns" in html
    assert "10 rows × 4 columns" in html
    assert "9 rows × 6 columns" in html
    assert "coerced dtypes" in html


def test_render_cleaning_coercion_audit(report_cfg, report_art):
    report_art.cleaning_summary = pd.DataFrame([{
        "step": "final", "detail": "done", "n_rows": 10, "n_columns": 5, "criterion": "",
    }])
    report_art.schema_coercion = pd.DataFrame([
        {
            "column": "tumor_volume",
            "kind": "continuous",
            "value_before": "NAV SECTRA - NOSŪTĪTS",
            "value_after": "(missing)",
            "n": 21,
        },
        {
            "column": "tumor_volume",
            "kind": "continuous",
            "value_before": "01",
            "value_after": "1",
            "n": 3,
        },
        {
            "column": "tumor_volume",
            "kind": "continuous",
            "value_before": "1.10",
            "value_after": "1.1",
            "n": 2,
        },
        {
            "column": "pid",
            "kind": "id",
            "value_before": "1.0",
            "value_after": "1.0",
            "n": 1,
        },
        {
            "column": "pid",
            "kind": "id",
            "value_before": "2.0",
            "value_after": "2.0",
            "n": 1,
        },
        {
            "column": "pid",
            "kind": "id",
            "value_before": "NAV",
            "value_after": "(missing)",
            "n": 1,
        },
    ])
    html = render_cleaning(report_cfg, report_art)
    # continuous missing + leading-zero fold + trailing-zero fold + id missing + id fold
    assert "Coerced value audit (5)" in html
    assert "NAV SECTRA - NOSŪTĪTS" in html
    assert "(missing)" in html
    assert "(various)" in html
    assert "leading-zero integer" in html
    assert "trailing-zero decimal" in html
    assert "<td>01</td>" not in html
    assert "<td>1.10</td>" not in html


def test_render_schema(report_cfg, report_art):
    report_art.schema_summary = pd.DataFrame([{"column": "age", "kind": "continuous"}])
    assert "<section" in render_schema(report_cfg, report_art)


def test_render_dda(report_cfg, report_art):
    report_art.dda_overall = pd.DataFrame([{"n_rows": 4}])
    html = render_dda(report_cfg, report_art)
    assert "<section" in html
    assert "1️⃣ DDA - univariate" in html
    assert "2️⃣ DDA - bivariate" in html
    assert "3️⃣ DDA - trivariate" in html
    # Univariate appears before bivariate before trivariate in the HTML
    assert html.index("1️⃣ DDA - univariate") < html.index("2️⃣ DDA - bivariate")
    assert html.index("2️⃣ DDA - bivariate") < html.index("3️⃣ DDA - trivariate")


def test_group_dda_bivariate_figures(tmp_path):
    a = tmp_path / "age__by__sex.svg"
    b = tmp_path / "age__by__adc_value.svg"
    c = tmp_path / "sex__by__adc_value.svg"
    for p in (a, b, c):
        p.write_text("<svg></svg>", encoding="utf-8")
    groups = rp._group_dda_bivariate_figures([a, b, c])
    assert list(groups) == ["age", "sex"]
    assert len(groups["age"]) == 2
    assert len(groups["sex"]) == 1


def test_group_dda_trivariate_figures(tmp_path):
    a = tmp_path / "vol__vs__diam__by__high_grade.svg"
    b = tmp_path / "vol__vs__diam__by__sex.svg"
    c = tmp_path / "age__vs__adc__by__sex.svg"
    for p in (a, b, c):
        p.write_text("<svg></svg>", encoding="utf-8")
    groups = rp._group_dda_trivariate_figures([a, b, c])
    assert list(groups) == ["age__vs__adc", "vol__vs__diam"]
    assert len(groups["vol__vs__diam"]) == 2


def test_render_dda_bivariate_key_dropdowns(report_cfg, report_art, tmp_path):
    a = tmp_path / "age__by__sex.svg"
    b = tmp_path / "sex__by__adc_value.svg"
    for p in (a, b):
        p.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    report_art.dda_bivariate_figures = [a, b]
    html = render_dda(report_cfg, report_art)
    assert "🔑 Age (1)" in html
    assert "🔑 Sex (1)" in html


def test_render_dda_trivariate_key_dropdowns(report_cfg, report_art, tmp_path):
    a = tmp_path / "tumor_volume__vs__max_diameter_cm__by__high_grade.svg"
    a.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    report_art.dda_trivariate_figures = [a]
    html = render_dda(report_cfg, report_art)
    assert "3️⃣ DDA - trivariate (1)" in html
    assert "Tumor volume (cm³) vs Max diameter (cm) (1)" in html


def test_dda_continuous_for_report_rounds_to_two_decimals():
    df = pd.DataFrame([{
        "column": "age",
        "kind": "continuous",
        "n": 365,
        "n_unique": 61,
        "missing_pct": 0.0,
        "mean": 62.9972602739726,
        "std": 12.764454988545294,
        "mode": 1.0,
    }])
    out = rp._dda_continuous_for_report(df)
    assert out.loc[0, "mean"] == 63
    assert out.loc[0, "std"] == 12.76
    assert out.loc[0, "n"] == 365


def test_render_dda_continuous_table_in_report(report_cfg, report_art):
    report_art.dda_continuous = pd.DataFrame([{
        "column": "age",
        "kind": "continuous",
        "n": 365,
        "n_unique": 61,
        "missing_pct": 0.0,
        "mean": 62.9972602739726,
        "median": 64.0,
    }])
    html = render_dda(report_cfg, report_art)
    assert "Continuous / count variables" in html
    assert "62.9972602739726" not in html
    assert ">63<" in html or ">63.0<" in html


def test_dda_glossary():
    assert "missing_pct" in rp._dda_glossary()


def test_eda_glossary():
    html = rp._eda_glossary()
    assert "mann_whitney_u" in html
    assert "rank_biserial_r" in html
    assert "Sensitivity" in html


def test_inferential_glossary():
    html = rp._inferential_glossary()
    assert "epv" in html
    assert "Rubin pooling" in html
    assert "vif" in html


def test_missingness_glossary():
    html = rp._missingness_glossary()
    assert "mice" in html
    assert "method by variable type" in html
    assert "missingness_resolution.py" in html


def test_render_missingness(report_cfg, report_art):
    assert "<section" in render_missingness(report_cfg, report_art)


def test_render_eda(report_cfg, report_art):
    html = render_eda(report_cfg, report_art)
    assert "<section" in html
    assert "What do these metrics mean?" in html


def test_render_diagnostic_accuracy(report_cfg, report_art):
    report_art.associations = pd.DataFrame({
        "target": ["event"], "predictor": ["age"], "kind": ["continuous"],
        "test": ["spearman"], "effect_label": ["spearman_rho"], "effect": [0.2],
        "p": [0.04], "p_fdr": [0.08], "n_used": [4],
    })
    report_art.diagnostic_accuracy = pd.DataFrame({
        "target": ["event", "event"],
        "predictor": ["perifocal_edema", "age"],
        "n_used": [4, 4],
        "TP": [2, np.nan], "FP": [0, np.nan], "FN": [1, np.nan], "TN": [1, np.nan],
        "sensitivity": [0.667, np.nan],
        "sensitivity_lo": [0.30, np.nan],
        "sensitivity_hi": [0.90, np.nan],
        "specificity": [1.0, np.nan],
        "specificity_lo": [0.40, np.nan],
        "specificity_hi": [1.0, np.nan],
        "PPV": [1.0, np.nan],
        "PPV_lo": [0.50, np.nan],
        "PPV_hi": [1.0, np.nan],
        "NPV": [0.5, np.nan],
        "NPV_lo": [0.10, np.nan],
        "NPV_hi": [0.90, np.nan],
        "accuracy": [0.75, np.nan],
        "accuracy_lo": [0.40, np.nan],
        "accuracy_hi": [0.95, np.nan],
        "AUC": [0.833, np.nan],
        "p": [0.01, np.nan],
        "p_fdr": [0.02, np.nan],
        "note": ["", "Skipped: requires predefined cutoff"],
    })
    html = render_eda(report_cfg, report_art)
    assert "Like in that research: univariate diagnostic accuracy" in html
    assert "Peritumoral edema" in html
    assert "66.7% [30.0" in html
    assert "sig-fdr" in html
    assert "Sensitivity (95% CI)" in html


def test_render_inferential(report_cfg, report_art):
    report_art.inferential_multivariable = {
        "event": pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
    }
    report_art.inferential_cases = pd.DataFrame([{
        "target": "event",
        "model_id": "",
        "model_title": "",
        "n_complete_cases": 40,
        "n_outcome_events": 12,
        "n_design_columns": 3,
        "epv": 4.0,
    }])
    html = render_inferential(report_cfg, report_art)
    assert "<section" in html
    assert 'class="epv-card"' in html
    assert "Underpowered" in html
    assert "What do these metrics mean?" in html


def test_render_inferential_multiple_variants(report_cfg, report_art):
    report_art.inferential_multivariable = {
        "high_grade::atypier_primary": pd.DataFrame({
            "predictor_col": ["sex"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
        "high_grade::bondo_et_al": pd.DataFrame({
            "predictor_col": ["tumor_margin"], "or": [0.5], "or_ci_lo": [0.2],
            "or_ci_hi": [0.9], "p": [0.04],
        }),
    }
    report_art.inferential_vif = {
        "high_grade::atypier_primary": pd.DataFrame({"predictor": ["sex"], "vif": [1.2]}),
        "high_grade::bondo_et_al": pd.DataFrame({"predictor": ["tumor_margin"], "vif": [1.1]}),
    }
    report_art.inferential_model_titles = {
        "high_grade::atypier_primary": "meningioma_atypier primary model",
        "high_grade::bondo_et_al": "Bondo et al.",
    }
    report_art.inferential_model_links = {
        "high_grade::bondo_et_al": "https://example.com/bondo",
    }
    report_art.inferential_cases = pd.DataFrame([
        {
            "target": "high_grade", "model_id": "atypier_primary",
            "model_title": "meningioma_atypier primary model",
            "n_complete_cases": 40, "n_outcome_events": 12,
            "n_design_columns": 3, "epv": 4.0,
        },
        {
            "target": "high_grade", "model_id": "bondo_et_al",
            "model_title": "Bondo et al.",
            "n_complete_cases": 38, "n_outcome_events": 12,
            "n_design_columns": 2, "epv": 6.0,
        },
    ])
    report_cfg.targets = ("high_grade",)
    html = render_inferential(report_cfg, report_art)
    assert "meningioma_atypier primary model" in html
    assert "Bondo et al." in html
    assert 'href="https://example.com/bondo"' in html
    assert "Interpretation" in html
    assert "What do these metrics mean?" in html


def test_render_inferential_experimental_last(report_cfg, report_art):
    report_art.inferential_multivariable = {
        "high_grade::experimental": pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
        "high_grade::yao_et_al_2022": pd.DataFrame({
            "predictor_col": ["sex"], "or": [1.5], "or_ci_lo": [1.0],
            "or_ci_hi": [2.0], "p": [0.04],
        }),
    }
    report_art.inferential_model_titles = {
        "high_grade::experimental": "meningioma_atypier experimental",
        "high_grade::yao_et_al_2022": "Yao et al. 2022",
    }
    report_art.inferential_cases = pd.DataFrame([
        {
            "target": "high_grade", "model_id": "experimental",
            "model_title": "meningioma_atypier experimental",
            "experimental": True,
            "n_complete_cases": 40, "n_outcome_events": 12,
            "n_design_columns": 3, "epv": 4.0,
        },
        {
            "target": "high_grade", "model_id": "yao_et_al_2022",
            "model_title": "Yao et al. 2022",
            "experimental": False,
            "n_complete_cases": 38, "n_outcome_events": 12,
            "n_design_columns": 2, "epv": 6.0,
        },
    ])
    report_art.inferential_model_experimental = {
        "high_grade::experimental": True,
        "high_grade::yao_et_al_2022": False,
    }
    report_cfg.targets = ("high_grade",)
    html = render_inferential(report_cfg, report_art)
    assert html.index("Yao et al. 2022") < html.index("meningioma_atypier experimental")
    assert "Literature-based models" in html
    assert "Experimental model" in html
    assert html.index("Literature-based models") < html.index("Experimental model")


def test_render_inferential_legacy_id_fallback_without_experimental_column(report_cfg, report_art):
    report_art.inferential_multivariable = {
        "high_grade::experimental_model_1": pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
        "high_grade::yao_et_al_2022": pd.DataFrame({
            "predictor_col": ["sex"], "or": [1.5], "or_ci_lo": [1.0],
            "or_ci_hi": [2.0], "p": [0.04],
        }),
    }
    report_art.inferential_model_titles = {
        "high_grade::experimental_model_1": "Custom model",
        "high_grade::yao_et_al_2022": "Yao et al. 2022",
    }
    report_art.inferential_cases = pd.DataFrame([
        {
            "target": "high_grade", "model_id": "experimental_model_1",
            "model_title": "Custom model",
            "n_complete_cases": 40, "n_outcome_events": 12,
            "n_design_columns": 3, "epv": 4.0,
        },
        {
            "target": "high_grade", "model_id": "yao_et_al_2022",
            "model_title": "Yao et al. 2022",
            "n_complete_cases": 38, "n_outcome_events": 12,
            "n_design_columns": 2, "epv": 6.0,
        },
    ])
    report_cfg.targets = ("high_grade",)
    html = render_inferential(report_cfg, report_art)
    assert html.index("Yao et al. 2022") < html.index("Custom model")
    assert "Experimental model" in html


def test_to_int_or_none():
    assert rp._to_int_or_none(4.0) == 4
    assert rp._to_int_or_none("x") is None


def test_first_present():
    df = pd.DataFrame({"b": [1]})
    assert rp._first_present(df, ["a", "b"]) == "b"


def test_render_inferential_interpretation():
    tbl = pd.DataFrame({"predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
                        "or_ci_hi": [3.0], "p": [0.01]})
    html = rp._render_inferential_interpretation(
        "event", tbl, "predictor_col", "or", "or_ci_lo", "or_ci_hi", "p",
    )
    assert "Interpretation" in html


def test_eda_direction_phrase():
    r = pd.Series({
        "predictor": "age", "test": "spearman", "effect": 0.3,
        "effect_label": "spearman_rho",
    })
    assert "age" in rp._eda_direction_phrase(r, "event")
    assert "higher rate" in rp._eda_direction_phrase(r, "event")

    r_phi = pd.Series({
        "predictor": "progesterone_pos", "test": "fisher_exact", "effect": -0.13,
        "effect_label": "phi",
    })
    assert "lower rate" in rp._eda_direction_phrase(r_phi, "high_grade")


def test_render_eda_interpretation(report_cfg):
    sub = pd.DataFrame({
        "predictor": ["age"], "test": ["spearman"], "effect": [0.2],
        "effect_label": ["spearman_rho"], "p": [0.04], "p_fdr": [0.08],
    })
    html = rp._render_eda_interpretation("event", sub, report_cfg)
    assert isinstance(html, str)


def test_render_appendix(report_cfg, report_art):
    html = render_appendix(report_cfg, report_art)
    assert isinstance(html, str)
    assert "Environment &amp; package versions" in html or "Environment & package versions" in html
    assert "<details" in html
    assert "pandas" in html
    assert "Python" in html


def test_render_environment_appendix():
    art = rp.Artifacts(output_root=Path("output"))
    art.mice_manifest = {
        "r_version": "R version 4.6.1 (2026-06-24)",
        "mice_version": "3.19.0",
        "jsonlite_version": "2.0.0",
    }
    html = rp._render_environment_appendix(art)
    assert "Computer / runtime" in html
    assert "Package versions" in html
    assert "pandas" in html
    assert "numpy" in html
    # Python vs R modules are explicitly separated, with R versions present.
    assert "Python modules" in html
    assert "R modules" in html
    assert "mice" in html
    assert "jsonlite" in html
    assert "3.19.0" in html


def test_system_specs_processor_and_graphics(monkeypatch):
    monkeypatch.setattr(
        rp, "_processor_description",
        lambda: "Intel(R) Core(TM) i7-12700K — 12 cores, 20 threads",
    )
    monkeypatch.setattr(
        rp, "_graphics_descriptions",
        lambda: ["NVIDIA GeForce RTX 3080, 10 GB VRAM, driver 31.0.15.4614"],
    )
    monkeypatch.setattr(rp, "_total_memory_gb", lambda: 32.0)
    rows = {r["item"]: r["value"] for r in rp._system_specs_rows()}
    assert "12 cores" in rows["Processor"]
    assert "NVIDIA GeForce RTX 3080" in rows["Graphics"]
    assert rows["RAM"] == "32.0 GB"


def test_wrap_html():
    assert "<html" in rp._wrap_html("T", "<body/>")


def test_build_report(report_cfg):
    html = build_report(report_cfg)
    assert "<html" in html


def test_write_html(tmp_path):
    out = write_html("<html><body>x</body></html>", tmp_path / "r.html")
    assert out.exists()


def test_parse_args(tmp_path):
    args = rp._parse_args(["--output-root", str(tmp_path)])
    assert args.output_root == tmp_path


def test_main(tmp_output):
    code = main(["--output-root", str(tmp_output), "--out", str(tmp_output / "r.html")])
    assert code == 0
    assert (tmp_output / "r.html").exists()


# --------------------------------------------------------------------------
# Marker panel section
# --------------------------------------------------------------------------
@pytest.fixture
def panel_output(tmp_output):
    tables = tmp_output / "panel" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([
        {"marker": "cortical_destruction", "label": "Cortical destruction",
         "n_used": 300, "present_n": 50, "catches": 27, "n_high_grade": 105,
         "lr_pos": 2.76, "lr_pos_lo": 1.66, "lr_pos_hi": 4.59,
         "chance_overlap": False, "continuity_corrected": False},
    ]).to_csv(tables / "01_marker_panel.csv", index=False)
    pd.DataFrame([
        {"Marker": "Cortical destruction", "Present in": "50/300",
         "Catches": "27 of 105", "Sens (95% CI)": "26% (19–35)",
         "Spec (95% CI)": "91% (86–94)", "LR+ (95% CI)": "2.8 (1.7–4.6)"},
    ]).to_csv(tables / "02_marker_panel_reading_view.csv", index=False)
    pd.DataFrame([
        {"item": "Patients in the shared set", "value": 301, "note": "every marker observed"},
    ]).to_csv(tables / "03_shared_cohort.csv", index=False)
    pd.DataFrame([
        {"n_criteria_met": 0, "n": 100, "n_high_grade": 11, "risk": 0.11,
         "risk_lo": 0.06, "risk_hi": 0.19},
        {"n_criteria_met": 1, "n": 90, "n_high_grade": 30, "risk": 0.33,
         "risk_lo": 0.24, "risk_hi": 0.43},
    ]).to_csv(tables / "07_count_score.csv", index=False)
    pd.DataFrame([
        {"side": "best single", "best_rule": "Cortical destruction",
         "J_apparent": 0.300, "optimism": 0.056, "J_corrected": 0.244,
         "winner_stability": 0.41, "n_bootstrap": 500,
         "gain_apparent": 0.120, "gain_corrected": 0.133,
         "correction_effect": "widens"},
        {"side": "best combination", "best_rule": "Cortical destruction OR Edema",
         "J_apparent": 0.420, "optimism": 0.043, "J_corrected": 0.377,
         "winner_stability": 0.33, "n_bootstrap": 500,
         "gain_apparent": 0.120, "gain_corrected": 0.133,
         "correction_effect": "widens"},
    ]).to_csv(tables / "09_selection_correction.csv", index=False)
    pd.DataFrame([
        {"Rule": "Cortical destruction OR Edema", "Type": "or", "n": 301,
         "Sens (95% CI)": "58% (48–67)", "Spec (95% CI)": "65% (58–71)",
         "PPV (95% CI)": "44% (36–53)", "NPV (95% CI)": "76% (69–82)",
         "TP/FP/FN/TN": "55/70/40/136", "OR (95% CI)": "2.7 (1.6–4.4)", "J": 0.23},
    ]).to_csv(tables / "06_rule_reading_view.csv", index=False)
    pd.DataFrame([
        {"model": "amano_et_al_2021", "n_scored": 301, "n_complete_own": 324,
         "denominator": "the patients every model could score",
         "auc_shared_apparent": 0.74, "auc_artifact_corrected": 0.733,
         "auc_artifact_apparent": 0.749, "best_single_rule": "Cortical destruction",
         "n_best_single": 344, "best_single_auc_corrected": 0.622,
         "best_single_J_corrected": 0.244, "note": ""},
    ]).to_csv(tables / "10_model_vs_single.csv", index=False)
    pd.DataFrame([
        {"item": "Draws", "value": 20, "note": "20 scorable"},
        {"item": "Winning rule reproduced", "value": 0.4,
         "note": "most often: Cortical destruction OR Edema"},
    ]).to_csv(tables / "11_imputation_stability.csv", index=False)
    pd.DataFrame([
        {"k_markers": 2, "min_n": 10, "n_bins_usable": 2, "direction": "rises",
         "low_count": 0, "low_n": 100, "low_risk": 0.11,
         "high_count": 1, "high_n": 90, "high_risk": 0.33, "note": ""},
    ]).to_csv(tables / "12_count_headline.csv", index=False)
    pd.DataFrame([
        {"Model": "Amano et al 2021", "Patients scored": "301",
         "Model AUC here (apparent)": "0.740",
         "Model AUC, own patients (corrected)": "0.733",
         "Model AUC, own patients (apparent)": "0.749",
         "Best single sign": "Cortical destruction",
         "Best single AUC (corrected)": "0.622",
         "Best single Youden J (corrected)": "0.244", "Note": ""},
    ]).to_csv(tables / "13_model_reading_view.csv", index=False)
    pd.DataFrame([
        {"What was checked": "Draws", "Result": "20", "Detail": "20 scorable"},
        {"What was checked": "Winning rule reproduced", "Result": "40%",
         "Detail": "most often: Cortical destruction OR Edema"},
    ]).to_csv(tables / "14_stability_reading_view.csv", index=False)

    figures = tmp_output / "panel" / "figures"
    figures.mkdir(parents=True)
    for name in ("lr_forest.svg", "count_score.svg", "rule_space.svg"):
        (figures / name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
            encoding="utf-8",
        )
    return tmp_output


def test_load_artifacts_finds_the_panel_tables(panel_output):
    art = load_artifacts(ReportConfig(output_root=panel_output, title="T"))
    assert art.panel_marker_reading_view is not None
    assert art.panel_selection_correction is not None
    assert len(art.panel_figures) == 3


def test_marker_panel_section_answers_both_aims(panel_output):
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "Cortical destruction" in html
    assert "2.8 (1.7–4.6)" in html
    assert "301" in html


def test_marker_panel_section_quotes_the_corrected_gain_not_the_apparent_one(
    panel_output,
):
    """+0.133 is the corrected gain; +0.120 is the uncorrected one, and 0.42 is
    the combination's apparent Youden J.

    Quoting an apparent number where the corrected one belongs is the
    CHANGES.md mistake in prose form, so the headline gain must be the
    corrected one *and* the apparent figures must not be able to stand in for
    it. Asserting only that the corrected value appears cannot catch that.
    """
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "<strong>+0.133</strong>" in html
    assert "<strong>+0.120</strong>" not in html
    assert "0.42" not in html


def test_the_correction_sentence_reports_which_way_correction_moved_the_gap(
    panel_output,
):
    """On the real cohort the corrected gap is the *larger* one.

    The best-of-16-singles side carries more selection optimism than the
    best-of-many-combinations side, so prose asserting "the uncorrected gap is
    larger" is false there. The direction is a column, and the sentence follows
    it.
    """
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "widens" in html
    assert "+0.120" in html          # the uncorrected gap, named as such
    assert "0.056" in html and "0.043" in html   # what each side cost to choose


def test_the_correction_sentence_flips_when_correction_narrows_the_gap(
    panel_output,
):
    """The same page, opposite data: the wording must follow the table."""
    tables = panel_output / "panel" / "tables"
    corr = pd.read_csv(tables / "09_selection_correction.csv")
    corr["gain_apparent"] = 0.200
    corr["correction_effect"] = "narrows"
    corr.to_csv(tables / "09_selection_correction.csv", index=False)

    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "narrows" in html
    assert "widens" not in html


def test_the_headline_sentence_follows_the_measured_direction(panel_output):
    """The aim-2 lead said "Risk rises" whatever the table held.

    On the real cohort that sentence read "Risk rises from 0% with 3 of the
    signs present to 0% with 15" — the two thinnest bins, and not a rise. The
    direction now comes from ``12_count_headline.csv``.
    """
    tables = panel_output / "panel" / "tables"
    pd.DataFrame([
        {"k_markers": 2, "min_n": 10, "n_bins_usable": 2, "direction": "falls",
         "low_count": 0, "low_n": 100, "low_risk": 0.33,
         "high_count": 1, "high_n": 90, "high_risk": 0.11, "note": ""},
    ]).to_csv(tables / "12_count_headline.csv", index=False)

    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "Risk falls from 33%" in html
    assert "Risk rises" not in html


def test_the_headline_sentence_quotes_the_denominators_behind_it(panel_output):
    """A bin holding one patient is what made the old sentence wrong; showing
    each endpoint's patient count makes a thin endpoint visible on the page."""
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "Risk rises from 11% among the 100 patients" in html
    assert "33% among the 90 with 1" in html


def test_the_panel_tables_never_show_raw_machine_column_names(panel_output):
    """Every other table in the section goes through a reading view; these two
    used to be dumped straight from the CSV."""
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    for machine_name in ("auc_shared_apparent", "auc_artifact_corrected",
                         "best_single_J_corrected", "n_bootstrap", "n_scored"):
        assert machine_name not in html


def test_the_model_prose_names_which_column_compares_with_which(panel_output):
    """A Youden J of 0.24 beside an AUC of 0.74 reads as three times worse.

    They are different scales, so the page has to say which column is the
    like-for-like one — the corrected single-marker AUC, not the J.
    """
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "Best single AUC (corrected)" in html
    assert "0.622" in html
    assert "(J + 1) / 2" in html


def test_the_model_prose_names_the_one_patient_set_the_models_share(panel_output):
    """Seven models with seven different denominators in one column invites the
    reader to subtract them. They are restricted to one set, and it is named —
    together with the wider set the single-sign columns are scored on."""
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "one shared set of 301 patients" in html
    assert "the patients every model could score" in html
    assert "the 344 patients with every marker observed" in html


def test_the_panel_warning_shows_no_escaped_markup(tmp_output):
    cfg = ReportConfig(output_root=tmp_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "&lt;code&gt;" not in html
    assert "output/panel/" in html


def test_marker_panel_degrades_to_a_warning_when_nothing_was_computed(tmp_output):
    cfg = ReportConfig(output_root=tmp_output, title="T")
    html = rp.render_marker_panel(cfg, load_artifacts(cfg))
    assert "warning" in html.lower()


def test_the_panel_section_sits_between_modelling_and_the_appendix(panel_output):
    cfg = ReportConfig(output_root=panel_output, title="T")
    html = build_report(cfg)
    assert html.index("Multivariable modelling") < html.index("Which MRI markers")
    assert html.index("Which MRI markers") < html.index("📎 Appendix")


# --------------------------------------------------------------------------
# The model table's Note cell carries the paper link
# --------------------------------------------------------------------------
def test_model_note_url_becomes_an_anchor():
    view = pd.DataFrame([{"Model": "Yao et al 2022",
                          "Note": "https://pubmed.ncbi.nlm.nih.gov/30317276/"}])
    out = rp._linked_model_notes(view)
    cell = out.iloc[0]["Note"]
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/30317276/"' in cell
    assert 'rel="noopener noreferrer"' in cell
    assert "Paper" in cell


def test_model_note_keeps_surrounding_text():
    view = pd.DataFrame([{"Model": "m", "Note": "not scorable here https://x.org/a"}])
    cell = rp._linked_model_notes(view).iloc[0]["Note"]
    assert "not scorable here" in cell
    assert 'href="https://x.org/a"' in cell


def test_model_note_without_a_url_is_escaped_not_linked():
    view = pd.DataFrame([{"Model": "m", "Note": "one outcome class only"}])
    cell = rp._linked_model_notes(view).iloc[0]["Note"]
    assert cell == "one outcome class only"
    assert "<a " not in cell


def test_model_note_escapes_markup_around_the_url():
    """The column is emitted as safe HTML, so everything else must be escaped."""
    view = pd.DataFrame([{"Model": "m",
                          "Note": "<script>alert(1)</script> https://x.org/a"}])
    cell = rp._linked_model_notes(view).iloc[0]["Note"]
    assert "<script>" not in cell
    assert "&lt;script&gt;" in cell


def test_model_note_survives_a_view_without_the_column():
    view = pd.DataFrame([{"Model": "m"}])
    assert list(rp._linked_model_notes(view).columns) == ["Model"]
