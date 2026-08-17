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
    render_missingness,
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


def test_png_embedding_helpers(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert rp._embed_png_src(p).startswith("data:image/png;base64,")
    assert "<img" in rp._figure_img_html(p)
    assert "figure-grid" in svg_grid([p])


def test_report_css_caps_embedded_images_to_the_page():
    """High-dpi PNG files stay full-size on disk; HTML must not display them native."""
    assert "img { max-width: 100%; height: auto; }" in rp._CSS
    assert "max-width: 36rem" in rp._CSS
    assert "max-width: none" not in rp._CSS


def test_scalar_helpers():
    assert rp._esc("<b>") == "&lt;b&gt;"
    assert rp._esc(None) == ""
    assert human_pool_df(12.0) == "12"
    assert human_pool_df(float("inf")) == "∞"
    assert human_p(0.0001) == "<0.001"
    assert human_p("<0.001") == "<0.001"
    assert rp._coerce_p("<0.001") is not None
    assert rp._coerce_p(None) is None
    assert rp._coerce_float("1.5") == 1.5
    assert rp._coerce_float("bad") is None
    assert rp._to_int_or_none(4.0) == 4
    assert rp._to_int_or_none("x") is None
    assert rp._first_present(pd.DataFrame({"b": [1]}), ["a", "b"]) == "b"


def test_classifiers():
    assert classify_significance(0.01, 0.02) == "sig-fdr"
    assert classify_or_direction(2.0, 1.2, 3.0) == "or-risk"
    assert classify_missing(50.0) == "missing-severe"


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
    # predictor_label folds z_sd into the name; SD >= 10 rounds to whole units
    # (existing, shared behaviour with the forest plot — see inferential.py).
    assert "per 1 SD: 13" in html


def test_html_building_blocks():
    assert "warning-box" in warning_box("oops")
    assert "info-box" in info_box("note")
    assert "<table" in table_to_html(pd.DataFrame({"a": [1], "b": [2]}))
    assert "<details" in details_block("sum", "<p>x</p>")
    assert "<html" in rp._wrap_html("T", "<body/>")


def test_file_loaders(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("a\n1\n")
    warns = []
    df = rp._maybe_read_csv(csv, warns)
    assert df is not None and len(df) == 1

    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"age": {"kind": "continuous"}}))
    df = rp._load_schema_any(schema, warns)
    assert df is not None and "column" in df.columns


def test_load_artifacts_and_build_report(report_cfg, tmp_output):
    assert load_artifacts(report_cfg).output_root == tmp_output
    assert "<html" in build_report(report_cfg)


def test_cli_entry_points(tmp_path, tmp_output):
    assert write_html("<html><body>x</body></html>", tmp_path / "r.html").exists()
    assert rp._parse_args(["--output-root", str(tmp_path)]).output_root == tmp_path
    code = main(["--output-root", str(tmp_output), "--out", str(tmp_output / "r.html")])
    assert code == 0
    assert (tmp_output / "r.html").exists()


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


def test_plain_sections_render(report_cfg, report_art):
    assert "<section" in render_cleaning(report_cfg, report_art)
    assert "<section" in render_missingness(report_cfg, report_art)
    html = render_eda(report_cfg, report_art)
    assert "<section" in html
    assert "💡 Interpretation" not in html
    assert "Full Sweep" not in html
    assert "Like in that research" not in html
    assert "Exploratory variants" not in html


def test_cohort_flow_table_uses_log_counts_and_falls_back_without_one():
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

    # Without a log, the same facts come off the summary.
    bare = Artifacts(
        output_root=Path("."),
        cleaning_summary=pd.DataFrame([
            {"step": "drop_rows", "detail": "grade exists", "n_rows": 9,
             "n_columns": 2, "criterion": "WHO grade recorded"},
        ]),
    )
    html = rp._cohort_flow_table(bare)
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


def test_render_dda_orders_the_blocks_and_splits_native_from_derived(report_cfg, report_art):
    report_art.dda_overall = pd.DataFrame([{"n_rows": 4}])
    report_art.dda_derived_columns = frozenset({"high_grade", "male_sex"})
    report_art.dda_binary = pd.DataFrame([
        {"column": "dural_tail", "kind": "binary", "n": 10},
        {"column": "high_grade", "kind": "binary", "n": 10},
        {"column": "male_sex", "kind": "binary", "n": 10},
    ])
    report_art.dda_continuous = pd.DataFrame([
        {"column": "age", "kind": "continuous", "n": 10, "min": 20.0},
    ])
    html = render_dda(report_cfg, report_art)
    assert "<section" in html
    assert "1️⃣ DDA - univariate" in html
    assert "2️⃣ DDA - bivariate" in html
    assert "3️⃣ DDA - trivariate" in html
    # Univariate appears before bivariate before trivariate in the HTML
    assert html.index("1️⃣ DDA - univariate") < html.index("2️⃣ DDA - bivariate")
    assert html.index("2️⃣ DDA - bivariate") < html.index("3️⃣ DDA - trivariate")
    assert "✅ Binary variables (3)" in html
    assert "🌱 Native (1)" in html
    assert "🧩 Derived (2)" in html
    assert "📏 Continuous / count variables (1)" in html
    # Continuous has no flagged derived cols → plain table, no Native/Derived split
    cont_start = html.index("📏 Continuous / count variables (1)")
    cont_chunk = html[cont_start:cont_start + 800]
    assert "🌱 Native" not in cont_chunk


def test_render_dda_hides_hidden_parent_columns(report_cfg, report_art):
    report_art.hidden_parent_columns = frozenset({"sex"})
    report_art.dda_categorical = pd.DataFrame([
        {"column": "sex", "kind": "nominal", "n": 10},
        {"column": "side", "kind": "nominal", "n": 10},
    ])
    html = render_dda(report_cfg, report_art)
    assert "🏷️ Categorical / ordinal variables (1)" in html
    assert ">side<" in html or "side" in html
    assert ">Nominal<" in html
    # the hidden parent should not appear as a table cell value for column
    assert ">sex<" not in html


def test_group_dda_figures_by_key(tmp_path):
    bi = [tmp_path / "age__by__sex.png",
          tmp_path / "age__by__adc_value.png",
          tmp_path / "sex__by__adc_value.png"]
    tri = [tmp_path / "vol__vs__diam__by__high_grade.png",
           tmp_path / "vol__vs__diam__by__sex.png",
           tmp_path / "age__vs__adc__by__sex.png"]
    for p in (*bi, *tri):
        p.write_text("<svg></svg>", encoding="utf-8")

    groups = rp._group_dda_bivariate_figures(bi)
    assert list(groups) == ["age", "sex"]
    assert len(groups["age"]) == 2
    assert len(groups["sex"]) == 1

    groups = rp._group_dda_trivariate_figures(tri)
    assert list(groups) == ["age__vs__adc", "vol__vs__diam"]
    assert len(groups["vol__vs__diam"]) == 2


def test_render_dda_key_dropdowns(report_cfg, report_art, tmp_path):
    a = tmp_path / "age__by__sex.png"
    b = tmp_path / "sex__by__adc_value.png"
    c = tmp_path / "tumor_volume__vs__max_diameter_cm__by__high_grade.png"
    for p in (a, b, c):
        p.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    report_art.dda_bivariate_figures = [a, b]
    report_art.dda_trivariate_figures = [c]
    html = render_dda(report_cfg, report_art)
    assert "🔑 Age (1)" in html
    assert "🔑 Sex (1)" in html
    assert "3️⃣ DDA - trivariate (1)" in html
    assert "Tumor volume (cm³) vs Max diameter (cm) (1)" in html


def test_dda_continuous_rounds_to_two_decimals(report_cfg, report_art):
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


def test_eda_paper_display_table_blanks_nan_and_keeps_nn_pct():
    rows = pd.DataFrame([{
        "row_role": "variable",
        "predictor": "dural_tail",
        "level": np.nan,
        "grade1": "10/50 (20.0%)",
        "grade23": np.nan,
        "effect": np.nan,
        "auc": "nan",
        "p_fdr": np.nan,
    }])
    out = rp._eda_paper_display_table(
        rows, table_kind="binary", target="high_grade",
    )
    assert list(out.columns) == [
        "Variable", "WHO Grade 1 n/N (%)", "WHO Grade 2–3 n/N (%)",
        "OR (95% CI)", "FDR-p",
    ]
    row = out.iloc[0]
    assert row["WHO Grade 1 n/N (%)"] == "10/50 (20.0%)"
    assert row["WHO Grade 2–3 n/N (%)"] == ""
    assert row["OR (95% CI)"] == ""
    assert row["FDR-p"] == ""
    assert "AUC" not in out.columns
    assert "nan" not in " ".join(map(str, row.values)).lower()

    levels = pd.DataFrame([{
        "row_role": "level",
        "predictor": "side",
        "level": "left",
        "grade1": "10/50 (20.0%)",
        "grade23": "20/50 (40.0%)",
        "effect": "2.00 (1.10–3.50)",
        "auc": "",
        "p_fdr": "",
        "p_level": "0.020",
    }])
    out = rp._eda_paper_display_table(
        levels, table_kind="nominal", target="high_grade",
    )
    assert list(out.columns) == [
        "Variable", "WHO Grade 1 n/N (%)", "WHO Grade 2–3 n/N (%)",
        "OR (95% CI)", "FDR-p",
    ]


def test_render_eda_paper_tables_native_derived(report_cfg, report_art):
    report_art.associations = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["dural_tail", "male_sex"],
        "kind": ["binary", "binary"],
        "test": ["chi2", "chi2"],
        "effect_label": ["phi", "phi"],
        "effect": [0.2, 0.1],
        "p": [0.01, 0.04],
        "p_fdr": [0.02, 0.08],
        "n_used": [100, 100],
        "in_fdr_family": [True, True],
    })
    report_art.eda_derived_columns = frozenset({"male_sex"})
    report_art.eda_paper_tables = pd.DataFrame([
        {
            "target": "high_grade", "table_kind": "binary", "predictor": "dural_tail",
            "row_role": "variable", "level": "",
            "grade1": "10/50 (20.0%)", "grade23": "20/50 (40.0%)",
            "effect": "2.00 (1.10–3.50)", "auc": "", "p_fdr": "0.020",
            "p_level": "", "sort_p": 0.02,
        },
        {
            "target": "high_grade", "table_kind": "binary", "predictor": "male_sex",
            "row_role": "variable", "level": "",
            "grade1": "15/50 (30.0%)", "grade23": "25/50 (50.0%)",
            "effect": "1.80 (1.00–3.20)", "auc": "", "p_fdr": "0.080",
            "p_level": "", "sort_p": 0.08,
        },
        {
            "target": "high_grade", "table_kind": "continuous", "predictor": "age",
            "row_role": "variable", "level": "",
            "grade1": "60.00 [50.00–70.00]", "grade23": "65.00 [55.00–75.00]",
            "effect": "1.20 (1.00–1.45)", "auc": "0.60 (0.50–0.70)",
            "p_fdr": "0.100", "p_level": "", "sort_p": 0.1,
        },
        {
            "target": "high_grade", "table_kind": "ordinal", "predictor": "age_bins",
            "row_role": "variable", "level": "",
            "grade1": "", "grade23": "",
            "effect": "", "auc": "", "p_fdr": "0.200",
            "p_level": "", "sort_p": 0.2,
        },
    ])
    html = render_eda(report_cfg, report_art)
    assert "🌱 Native" in html
    assert "🧩 Derived" in html
    assert "WHO Grade 1 n/N (%)" in html
    assert "OR per SD (95% CI)" in html
    assert "Dichotomous" in html
    assert "Interval/Ratio" in html
    assert "Ordinal" in html
    assert "eda-kind-divider" in html
    assert "eda-col-header" in html
    assert "eda-paper-stack" in html
    # The AJNR footnote: one Note:— paragraph, then the abbreviation list.
    assert "Note:&mdash;" in html
    assert "Benjamini&ndash;Hochberg false discovery rate procedure" in html
    assert "Native and derived variables form separate families" in html
    assert "not portable" in html
    assert "AUC indicates area under the receiver operating characteristic" in html
    assert "Full Sweep" not in html
    assert "Like in that research" not in html
    # One fold per origin, each holding that origin's table AND its forest.
    # The wrapper fold and the separate forest folds are gone: reading a table
    # and its plot should not mean opening two places, and a native q must not
    # sit next to a derived one.
    assert "📊 Paper-style table" not in html
    assert "🌲 Native forest" not in html and "🌲 Derived forest" not in html
    assert html.index("🌱 Native") < html.index("🧩 Derived")

    native_fold = html[html.index("🌱 Native"):html.index("🧩 Derived")]
    assert "<img" in native_fold
    assert "Native unadjusted odds ratios" in native_fold
    assert "Derived unadjusted odds ratios" not in native_fold

    derived_fold = html[html.index("🧩 Derived"):]
    assert "Derived unadjusted odds ratios" in derived_fold

    # Native / Derived stay separate; each origin is exactly one <table>.
    paper = rp._render_eda_native_derived_block(
        report_art.eda_paper_tables,
        target="high_grade",
        derived_cols=frozenset({"male_sex"}),
        n_fdr_family=2,
    )
    assert "<h4>" not in paper  # datatypes are divider rows, not separate headings
    # The block no longer folds itself — render_eda pairs each origin with its
    # own forest and folds the two together, so this returns bare table html.
    assert "<details" not in paper

    def _one(only):
        return rp._render_eda_native_derived_block(
            report_art.eda_paper_tables,
            target="high_grade",
            derived_cols=frozenset({"male_sex"}),
            n_fdr_family=2,
            only=only,
        )

    details = [_one("native"), _one("derived")]
    assert details[0].count('<table class="report">') == 1
    assert details[1].count('<table class="report">') == 1
    # Each origin renders only its own rows.
    assert "Dural tail" in details[0] and "Dural tail" not in details[1]
    assert "Male sex" in details[1] and "Male sex" not in details[0]
    # Native stacks multiple datatypes inside that single table
    assert details[0].count("eda-kind-divider") >= 2
    assert "Interval/Ratio" in details[0]
    assert "Dichotomous" in details[0]
    assert "Ordinal" in details[0]
    # Derived is its own one-table stack (binary only in this fixture)
    assert details[1].count("eda-kind-divider") == 1
    assert "Dichotomous" in details[1]


def test_render_eda_omits_excluded_columns(report_cfg, report_art):
    report_art.associations = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["dural_tail", "hidden_pred"],
        "kind": ["binary", "binary"],
        "test": ["chi2", "chi2"],
        "effect_label": ["phi", "phi"],
        "effect": [0.2, 0.9],
        "p": [0.01, 0.001],
        "p_fdr": [0.02, 0.002],
        "n_used": [100, 100],
        "in_fdr_family": [True, True],
    })
    report_art.eda_excluded_columns = frozenset({"hidden_pred"})
    report_art.eda_paper_tables = pd.DataFrame([
        {
            "target": "high_grade", "table_kind": "binary", "predictor": "dural_tail",
            "row_role": "variable", "level": "",
            "grade1": "10/50 (20.0%)", "grade23": "20/50 (40.0%)",
            "effect": "2.00 (1.10–3.50)", "auc": "", "p_fdr": "0.020",
            "p_level": "", "sort_p": 0.02,
        },
        {
            "target": "high_grade", "table_kind": "binary", "predictor": "hidden_pred",
            "row_role": "variable", "level": "",
            "grade1": "1/50 (2.0%)", "grade23": "40/50 (80.0%)",
            "effect": "9.00 (2.00–20.0)", "auc": "", "p_fdr": "0.002",
            "p_level": "", "sort_p": 0.002,
        },
    ])
    html = render_eda(report_cfg, report_art)
    assert "dural_tail" in html
    assert "hidden_pred" not in html
    assert "Hidden pred" not in html


def test_render_inferential(report_cfg, report_art):
    tbl = pd.DataFrame({
        "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
        "or_ci_hi": [3.0], "p": [0.01],
    })
    report_art.inferential_multivariable = {"event": tbl}
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

    assert "Interpretation" in rp._render_inferential_interpretation(
        "event", tbl, "predictor_col", "or", "or_ci_lo", "or_ci_hi", "p",
    )


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


@pytest.mark.parametrize(
    ("model_id", "title", "flagged"),
    [
        # Flagged by the explicit `experimental` column ...
        ("experimental", "meningioma_atypier experimental", True),
        # ... and by the legacy id prefix when that column is absent.
        ("experimental_model_1", "Custom model", False),
    ],
)
def test_render_inferential_puts_the_experimental_model_last(
    report_cfg, report_art, model_id, title, flagged,
):
    key = f"high_grade::{model_id}"
    report_art.inferential_multivariable = {
        key: pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
        "high_grade::yao_et_al_2022": pd.DataFrame({
            "predictor_col": ["sex"], "or": [1.5], "or_ci_lo": [1.0],
            "or_ci_hi": [2.0], "p": [0.04],
        }),
    }
    report_art.inferential_model_titles = {
        key: title,
        "high_grade::yao_et_al_2022": "Yao et al. 2022",
    }
    case = {
        "target": "high_grade", "model_id": model_id, "model_title": title,
        "n_complete_cases": 40, "n_outcome_events": 12,
        "n_design_columns": 3, "epv": 4.0,
    }
    other = {
        "target": "high_grade", "model_id": "yao_et_al_2022",
        "model_title": "Yao et al. 2022",
        "n_complete_cases": 38, "n_outcome_events": 12,
        "n_design_columns": 2, "epv": 6.0,
    }
    if flagged:
        case["experimental"] = True
        other["experimental"] = False
        report_art.inferential_model_experimental = {
            key: True, "high_grade::yao_et_al_2022": False,
        }
    report_art.inferential_cases = pd.DataFrame([case, other])
    report_cfg.targets = ("high_grade",)
    html = render_inferential(report_cfg, report_art)
    assert html.index("Yao et al. 2022") < html.index(title)
    assert "Literature-based models" in html
    assert "Experimental model" in html
    assert html.index("Literature-based models") < html.index("Experimental model")


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


def _model_vs_single_row(**overrides):
    row = {
        "model_id": "m1", "single": "tumor_volume",
        "auc_model_corrected": 0.697, "auc_single_corrected": 0.679,
        "delta_auc_corrected": 0.018, "delta_auc_apparent": 0.026,
        "delta_ci_lo": -0.006, "delta_ci_hi": 0.057, "d2_p": 0.014873,
        "n_resamples": 1000,
    }
    row.update(overrides)
    return row


def test_model_vs_single_block_returns_empty_without_a_table(report_art):
    import report as rp
    assert rp._model_vs_single_block("m1", report_art) == ""


def test_model_vs_single_block_returns_empty_for_a_different_model(report_art):
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row()])
    assert rp._model_vs_single_block("some_other_model", report_art) == ""


def test_model_vs_single_block_shows_both_deltas_and_the_single_predictor(report_art):
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row()])
    html = rp._model_vs_single_block("m1", report_art)
    assert "tumor_volume" in html
    # Corrected point estimate and apparent point estimate are different
    # numbers and must both appear (0.018 corrected vs 0.026 apparent).
    assert "0.018" in html
    assert "0.026" in html
    assert "-0.006" in html and "0.057" in html


def test_model_vs_single_block_puts_the_ci_next_to_apparent_not_corrected(report_art):
    """Requirement A: the CI is a patient-resampling interval centred on the
    APPARENT delta, not the optimism-corrected one. A reader who pairs
    delta_auc_corrected with delta_ci_lo/hi would be silently wrong, so the
    corrected point estimate and the apparent-delta-with-CI must live in
    separately headed columns, and the header naming the CI's column must
    say "apparent", never bare "ΔAUC (95% CI)" that could be misread as
    belonging to the corrected number."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row()])
    html = rp._model_vs_single_block("m1", report_art)
    assert "ΔAUC corrected" in html
    assert "ΔAUC apparent" in html
    # The table's own "ΔAUC apparent" column header (its last occurrence --
    # the prose may mention "apparent" earlier) must sit immediately before
    # the CI value in the data row, not under a bare "corrected" header.
    apparent_idx = html.rindex("ΔAUC apparent")
    ci_idx = html.index("-0.006")
    assert apparent_idx < ci_idx
    assert ci_idx - apparent_idx < 400


def test_model_vs_single_block_paragraph_explains_apparent_vs_corrected(report_art):
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row()])
    html = rp._model_vs_single_block("m1", report_art)
    lowered = html.lower()
    assert "apparent" in lowered and "corrected" in lowered
    assert "overfitting" in lowered or "optimism" in lowered


def test_model_vs_single_block_paragraph_explains_the_p_value_is_a_different_uncorrected_question(report_art):
    """Requirement B: d2_p is a full-cohort, no-optimism-correction
    likelihood-ratio test, and it can be significant while the (corrected)
    ΔAUC's apparent-delta CI spans zero -- as it genuinely does for
    funari_2023/tumor_volume and spille_2020/edema_volume_cm3. The paragraph
    must say the two answer different questions and that the p-value carries
    no optimism correction, so a reader does not conclude one of them is
    wrong."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([
        _model_vs_single_row(model_id="funari_2023", single="tumor_volume",
                              delta_auc_corrected=0.018, delta_auc_apparent=0.026,
                              delta_ci_lo=-0.006, delta_ci_hi=0.057, d2_p=0.014873),
    ])
    html = rp._model_vs_single_block("funari_2023", report_art)
    lowered = html.lower()
    assert "likelihood" in lowered
    assert "no optimism correction" in lowered or "not optimism-corrected" in lowered \
        or "no correction for optimism" in lowered
    assert "different question" in lowered or "different questions" in lowered


def test_model_vs_single_block_states_corrected_delta_has_no_interval(report_art):
    """A related gap the reviewer flagged: after being told not to use the
    apparent CI or the p-value for the corrected delta's uncertainty, a
    reader is left wondering what to use instead. The answer is nothing --
    it is a point estimate -- and the block must say so."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row()])
    html = rp._model_vs_single_block("m1", report_art)
    lowered = html.lower()
    assert "point estimate" in lowered
    assert "no confidence interval" in lowered or "no interval" in lowered


def test_model_vs_single_block_resample_count_is_read_from_the_data(report_art):
    """The "resampled ... times" phrase must track model_vs_single_auc.csv's
    own n_resamples column, not a literal 1000 -- a re-run with a different
    bootstrap count must not leave stale prose behind."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row(n_resamples=250)])
    html = rp._model_vs_single_block("m1", report_art)
    assert "250" in html
    assert "1000" not in html


def test_model_vs_single_block_single_auc_and_delta_use_three_decimals(report_art):
    """format_number trims trailing zeros, which makes ΔAUC corrected and
    ΔAUC apparent look like different precisions side by side, and turns an
    exact 0.000 into a bare '0' that reads as missing data."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row(
        model_id="top_1_variable", single="tumor_volume",
        auc_single_corrected=0.630, delta_auc_corrected=0.0,
        delta_auc_apparent=0.0, delta_ci_lo=0.0, delta_ci_hi=0.0,
    )])
    html = rp._model_vs_single_block("top_1_variable", report_art)
    assert "0.630" in html   # not the ragged "0.63"
    assert "0.000" in html   # not the bare "0"
    assert ">0<" not in html


def test_model_vs_single_block_multiple_singles_all_render(report_art):
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([
        _model_vs_single_row(single="tumor_volume"),
        _model_vs_single_row(single="irregular_tumor_margin", delta_auc_corrected=0.072,
                              delta_auc_apparent=0.082, delta_ci_lo=0.038, delta_ci_hi=0.128,
                              d2_p=0.000034),
    ])
    html = rp._model_vs_single_block("m1", report_art)
    assert "tumor_volume" in html and "irregular_tumor_margin" in html
    assert "beat its own single" in html.lower() or "beat its own single" in html


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


def test_selection_audit_block_renders_a_resample_only_row_with_blank_kept_cell(report_art):
    """Finding 3, final whole-branch review: a candidate the full-cohort
    audit walk never reached has NaN in every one of auc/discrimination/kept/
    reason, not just the ones already known to render blank. ``bool(NaN)`` is
    truthy, so a naive ``"✅" if bool(kept) else "—"`` would mislabel a
    never-evaluated candidate "kept" -- the Kept cell for such a row must be
    blank, not "✅" and not the literal string "nan"."""
    import re

    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": "", "resample_selection_count": 900},
        {"variable": "hyperostosis", "auc": float("nan"),
         "discrimination": float("nan"), "kept": float("nan"),
         "reason": float("nan"), "resample_selection_count": 316},
    ])
    html = rp._selection_audit_block(report_art)
    assert "hyperostosis" in html and "316" in html
    row_html = re.search(
        r"<tr[^>]*>(?:(?!</tr>).)*hyperostosis(?:(?!</tr>).)*</tr>", html, re.S)
    assert row_html is not None
    assert "✅" not in row_html.group(0)
    assert "nan" not in row_html.group(0).lower()


def test_discrimination_is_shown_for_a_protective_variable(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "adc_value", "auc": 0.370, "discrimination": 0.630,
         "kept": True, "reason": ""},
    ])
    html = rp._selection_audit_block(report_art)
    assert "0.630" in html and "↓" in html


def test_selection_audit_block_returns_empty_without_a_table(report_art):
    import report as rp
    assert rp._selection_audit_block(report_art) == ""


def test_selection_audit_block_shows_resample_selection_counts(report_art):
    """The counts are the evidence for how stable the chosen six are, so they
    must be rendered, not just loaded."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.678504, "discrimination": 0.678504,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
        {"variable": "max_diameter_cm", "auc": 0.674745, "discrimination": 0.674745,
         "kept": False, "reason": "rho=0.91 with tumor_volume",
         "resample_selection_count": 382},
        {"variable": "cystic_component", "auc": 0.592481, "discrimination": 0.592481,
         "kept": True, "reason": float("nan"), "resample_selection_count": 342},
    ])
    html = rp._selection_audit_block(report_art)
    assert "615" in html and "382" in html and "342" in html


def test_selection_audit_block_explains_what_a_low_resample_count_means(report_art):
    import pandas as pd, report as rp
    n_resamples = 1000
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.678504, "discrimination": 0.678504,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615,
         "resample_selection_total": n_resamples},
        {"variable": "max_diameter_cm", "auc": 0.674745, "discrimination": 0.674745,
         "kept": False, "reason": "rho=0.91 with tumor_volume",
         "resample_selection_count": 382, "resample_selection_total": n_resamples},
    ])
    html = rp._selection_audit_block(report_art)
    lowered = html.lower()
    assert "resample" in lowered
    # The resample total must come from top_variable_selection.csv's OWN
    # resample_selection_total column (Item 4, final whole-branch review) --
    # never a literal "1000" and never model_vs_single_auc.csv's n_resamples
    # (see the dedicated "not borrowed" test below) -- assert against the
    # value actually in the fixture's data, and prove it moves with the data
    # in a second call.
    assert str(n_resamples) in html
    assert "stable" in lowered or "unstable" in lowered

    other = report_art.top_selection.copy()
    other["resample_selection_total"] = 250
    report_art.top_selection = other
    html2 = rp._selection_audit_block(report_art)
    assert "250" in html2
    assert str(n_resamples) not in html2


def test_selection_audit_block_denominator_is_not_borrowed_from_model_vs_single(
    report_art,
):
    """Item 4, final whole-branch review: model_vs_single_auc.csv's
    n_resamples counts PATIENT resamples that kept both outcome classes;
    top_variable_selection.csv's own resample_selection_total counts
    SELECTION resamples from a different loop with different drop rules.
    Both are 1000 in production today, which is exactly why a fixture where
    they DIFFER is the only way to catch one silently standing in for the
    other."""
    import pandas as pd, report as rp
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row(n_resamples=1000)])
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615,
         "resample_selection_total": 777},
    ])
    html = rp._selection_audit_block(report_art)
    assert "Selected in resamples (of 777)" in html
    assert "Selected in resamples (of 1000)" not in html


def test_selection_audit_block_derives_the_collinearity_sentence_from_the_table(report_art):
    """Important finding fix: the "picked almost as often as tumor_volume"
    claim must be computed from resample_selection_count, not typed. Use
    counts that differ from production (382 vs 615) to prove the sentence
    isn't hardcoded, and check the actual counts appear rather than a
    characterisation like "almost as often" or "coin flip"."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 900},
        {"variable": "max_diameter_cm", "auc": 0.675, "discrimination": 0.675,
         "kept": False, "reason": "rho=0.42 with tumor_volume",
         "resample_selection_count": 111},
    ])
    html = rp._selection_audit_block(report_art)
    assert "still won 111 resamples" in html
    assert "900" in html
    assert "0.42" in html
    # The old hardcoded pair (382, 615) must not leak in from a stale string.
    assert "382" not in html
    assert "615" not in html


def test_selection_audit_block_picks_the_highest_count_dropped_collinear_row(report_art):
    """When more than one variable was dropped for collinearity, the
    sentence must name the one with the highest resample_selection_count,
    not just the first row in the table."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 900},
        {"variable": "weak_collinear", "auc": 0.60, "discrimination": 0.60,
         "kept": False, "reason": "rho=0.85 with tumor_volume",
         "resample_selection_count": 5},
        {"variable": "max_diameter_cm", "auc": 0.675, "discrimination": 0.675,
         "kept": False, "reason": "rho=0.91 with tumor_volume",
         "resample_selection_count": 382},
    ])
    html = rp._selection_audit_block(report_art)
    # The derived sentence names the higher-count row (max_diameter_cm, 382),
    # not the weaker one (weak_collinear, 5) -- even though both rows are
    # still listed in the table itself.
    assert "still won 382" in html
    assert "still won 5" not in html
    # weak_collinear appears exactly once: its own table row, not a second
    # time inside a derived sentence about it.
    assert html.count("weak_collinear") == 1


def test_selection_audit_block_omits_the_collinearity_sentence_when_none_dropped_for_it(report_art):
    """No candidate was dropped for correlation with a kept variable (e.g.
    only cut-point rows were dropped) -- the block must not invent a claim
    about a pair that doesn't exist."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
        {"variable": "tumor_volume_ge15.1", "auc": 0.639, "discrimination": 0.639,
         "kept": False, "reason": "cut-point of tumor_volume",
         "resample_selection_count": 0},
    ])
    html = rp._selection_audit_block(report_art)
    assert "still won" not in html
    assert "0.91" not in html
    assert "dropped by the full-cohort selection only for being too correlated" not in html


def test_selection_audit_block_explains_cutpoint_zero_is_structural(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
        {"variable": "tumor_volume_ge15.1", "auc": 0.639, "discrimination": 0.639,
         "kept": False, "reason": "cut-point of tumor_volume",
         "resample_selection_count": 0},
    ])
    html = rp._selection_audit_block(report_art)
    lowered = html.lower()
    assert "structural" in lowered
    assert "deterministic" in lowered or "guard" in lowered


def test_selection_audit_block_header_shows_the_resample_denominator_when_known(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615,
         "resample_selection_total": 1000},
    ])
    html = rp._selection_audit_block(report_art)
    assert "Selected in resamples (of 1000)" in html


def test_selection_audit_block_header_omits_denominator_when_unknown(report_art):
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
    ])
    html = rp._selection_audit_block(report_art)
    assert "Selected in resamples" in html
    assert "Selected in resamples (of" not in html


def test_selection_audit_block_never_renders_nan_or_none_literals(report_art):
    """Blank/NaN reasons (kept variables have no drop reason) and a missing
    resample count must never render as the literal text 'nan' or 'None'."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
        {"variable": "cystic_component", "auc": 0.592, "discrimination": 0.592,
         "kept": True, "reason": None},
    ])
    html = rp._selection_audit_block(report_art)
    assert ">nan<" not in html
    assert ">None<" not in html


def test_selection_audit_block_missing_resample_count_column_does_not_crash(report_art):
    """Older / hand-built tables (like the brief's own sample rows) may not
    carry resample_selection_count at all -- the block must degrade
    gracefully rather than raising KeyError."""
    import pandas as pd, report as rp
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": ""},
    ])
    html = rp._selection_audit_block(report_art)
    assert "tumor_volume" in html
    assert ">nan<" not in html and ">None<" not in html


def test_render_inferential_wires_the_combined_vs_single_and_selection_blocks(report_cfg, report_art):
    """Integration check: both new blocks must actually be reachable from
    render_inferential, not just callable in isolation."""
    import pandas as pd, report as rp
    report_art.inferential_multivariable = {
        "high_grade::m1": pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
    }
    report_art.model_vs_single = pd.DataFrame([_model_vs_single_row(model_id="m1")])
    report_art.top_selection = pd.DataFrame([
        {"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
         "kept": True, "reason": float("nan"), "resample_selection_count": 615},
    ])
    report_cfg.targets = ("high_grade",)
    html = render_inferential(report_cfg, report_art)
    assert "beat its own single" in html
    assert "How these variables were chosen" in html
    assert ">nan<" not in html and ">None<" not in html


def test_load_artifacts_reads_the_combined_vs_single_tables(report_cfg, tmp_output):
    import pandas as pd
    inf_tab = tmp_output / "inferential" / "tables"
    inf_tab.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_model_vs_single_row()]).to_csv(
        inf_tab / "model_vs_single_auc.csv", index=False)
    pd.DataFrame([{"predictor": "tumor_volume", "n": 352, "events": 105,
                    "auc_apparent": 0.68, "auc_corrected": 0.679}]).to_csv(
        inf_tab / "single_predictor_reference.csv", index=False)
    pd.DataFrame([{"variable": "tumor_volume", "auc": 0.679, "discrimination": 0.679,
                    "kept": True, "reason": "", "resample_selection_count": 615}]).to_csv(
        inf_tab / "top_variable_selection.csv", index=False)
    art = load_artifacts(report_cfg)
    assert art.model_vs_single is not None
    assert "tumor_volume" in art.model_vs_single["single"].values
    assert art.single_reference is not None
    assert "tumor_volume" in art.single_reference["predictor"].values
    assert art.top_selection is not None
    assert "tumor_volume" in art.top_selection["variable"].values


def test_render_appendix_lists_the_environment(report_cfg, report_art):
    html = render_appendix(report_cfg, report_art)
    assert isinstance(html, str)
    assert "Environment &amp; package versions" in html or "Environment & package versions" in html
    assert "<details" in html
    assert "pandas" in html
    assert "Python" in html

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
        {"k_markers": 2, "min_n": 10, "n_bins_usable": 2, "direction": "rises",
         "low_count": 0, "low_n": 100, "low_risk": 0.11,
         "high_count": 1, "high_n": 90, "high_risk": 0.33, "note": ""},
    ]).to_csv(tables / "12_count_headline.csv", index=False)

    # The panel now renders inside its EDA target, so the target has to exist.
    eda_tab = tmp_output / "eda" / "tables"
    eda_tab.mkdir(parents=True)
    pd.DataFrame([
        {"target": "high_grade", "predictor": "cortical_destruction",
         "kind": "binary", "test": "chi2", "effect_label": "OR", "effect": 3.42,
         "p": 0.001, "p_fdr": 0.004, "n_used": 351, "auc_univariate": 0.58},
    ]).to_csv(eda_tab / "associations.csv", index=False)

    figures = tmp_output / "panel" / "figures"
    figures.mkdir(parents=True)
    for name in ("lr_forest.png", "count_score.png"):
        (figures / name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
            encoding="utf-8",
        )
    return tmp_output


def test_the_binary_block_answers_aim_one(panel_output):
    """Every other table in the section goes through a reading view; the panel
    tables used to be dumped straight from the CSV."""
    cfg = ReportConfig(output_root=panel_output, title="T")
    art = load_artifacts(cfg)
    assert art.panel_marker_reading_view is not None
    assert len(art.panel_figures) == 2

    html = rp._binary_marker_block(art)
    assert "⋈ Binary" in html
    assert "Cortical destruction" in html
    assert "2.8 (1.7–4.6)" in html
    for machine_name in ("auc_shared_apparent", "auc_artifact_corrected",
                         "best_single_J_corrected", "n_bootstrap", "n_scored"):
        assert machine_name not in html


def test_the_binary_block_is_absent_when_no_panel_was_computed(tmp_output):
    """No dropdown at all — a reader of the EDA target should not meet a
    warning about artifacts they never asked for."""
    cfg = ReportConfig(output_root=tmp_output, title="T")
    assert rp._binary_marker_block(load_artifacts(cfg)) == ""


def test_the_binary_block_sits_inside_the_eda_target(panel_output):
    """The panel is a screening result, so it hangs under the target it was
    screened against — not as a section of its own."""
    cfg = ReportConfig(output_root=panel_output, title="T",
                       targets=("high_grade",))
    html = build_report(cfg)
    assert "Which MRI markers" not in html
    eda = html.index("Exploratory association screening")
    inferential = html.index("Multivariable modelling")
    assert eda < html.index("⋈ Binary") < inferential
    assert html.index("🎯 Target: <code>high_grade</code>") < html.index("⋈ Binary")


def test_data_quality_warnings():
    import report

    # Implausible minima are flagged, one message per column.
    msgs = report.data_quality_warnings(pd.DataFrame({
        "column": ["max_diameter_cm", "tumor_volume", "age"],
        "min": [0.2, 0.3, 18.0],
    }))
    assert len(msgs) == 2
    assert any("0.2" in m and "diameter" in m.lower() for m in msgs)
    assert any("0.3" in m and "volume" in m.lower() for m in msgs)

    # Plausible minima stay silent.
    assert report.data_quality_warnings(
        pd.DataFrame({"column": ["max_diameter_cm"], "min": [1.4]}),
    ) == []

    # A missing "min" column returns [] without raising.
    assert report.data_quality_warnings(
        pd.DataFrame({"column": ["max_diameter_cm", "tumor_volume"]}),
    ) == []

    # Duplicate rows for one column collapse to a single warning at the minimum.
    msgs = report.data_quality_warnings(pd.DataFrame({
        "column": ["max_diameter_cm", "max_diameter_cm", "age"],
        "min": [0.2, 0.6, 25.0],
    }))
    assert len(msgs) == 1
    assert "0.2" in msgs[0] and "diameter" in msgs[0].lower()


def _paper_pair():
    return pd.DataFrame({
        "target": ["y", "y"], "table_kind": ["binary", "binary"],
        "predictor": ["adc_value", "adc_value_le0.72"],
        "row_role": ["variable", "variable"], "level": ["", ""],
        "grade1": ["", ""], "grade23": ["", ""],
        "effect": ["1.00 (0.50–2.00)", "1.00 (0.50–2.00)"], "auc": ["", ""],
        "p_fdr": ["0.01", "0.02"], "p_level": ["", ""], "sort_p": [0.01, 0.02],
    })


def test_the_native_family_rejects_restatements_and_hidden_parents():
    """A restatement is checked against the derivation log, not the list that
    decided the split.

    ``adc_value_le0.72`` is ``adc_value`` with a line drawn through it, so
    correcting it natively tests the same information twice and moves every
    native q. This is how multiple_meningiomas ended up with no q at all. A
    hidden parent is the mirror case: it was replaced by a flag, so showing
    both puts one fact in twice.
    """
    with pytest.raises(ValueError, match="alongside the column they restate"):
        rp._render_eda_native_derived_block(
            _paper_pair(), target="y", derived_cols=frozenset(),
            n_fdr_family=2,
            derived_sources={"adc_value_le0.72": ["adc_value"]},
        )

    with pytest.raises(ValueError, match="Hidden parent"):
        rp._render_eda_native_derived_block(
            _paper_pair(), target="y",
            derived_cols=frozenset({"adc_value_le0.72"}),
            n_fdr_family=1, hidden_parents=frozenset({"adc_value"}),
        )


def test_the_footnote_names_what_was_dropped_and_what_replaced_it():
    """A reader who knows sex was recorded must be told it is here as "Male"."""
    html = rp._render_eda_native_derived_block(
        _paper_pair(), target="y",
        derived_cols=frozenset({"adc_value_le0.72"}),
        n_fdr_family=1, n_derived_family=1,
        hidden_parents=frozenset({"sex"}),
        hidden_replacements={"sex": ["male"]},
        derived_sources={"adc_value_le0.72": ["adc_value"]},
    )
    assert "Replaced by derived flags" in html
    assert "Male" in html


def test_model_overview_block_carries_both_comparators(report_cfg, report_art):
    """The overview's whole point is that the two Δ columns answer different
    questions and often disagree — a model can beat its own best ingredient and
    still lose to the shared reference. Both must render, side by side."""
    import pandas as pd, report as rp
    report_art.model_overview = pd.DataFrame([{
        "model_id": "zhang_2020", "n_predictors": 4,
        "auc_apparent": 0.703, "auc_corrected": 0.688,
        "best_own_single": "irregular_tumor_margin", "best_own_auc_corrected": 0.625,
        "delta_own_corrected": 0.063, "delta_own_apparent": 0.080,
        "delta_own_ci_lo": 0.031, "delta_own_ci_hi": 0.132,
        "reference": "tumor_volume", "reference_auc_corrected": 0.679,
        "delta_ref_corrected": 0.009, "delta_ref_apparent": 0.024,
        "delta_ref_ci_lo": -0.045, "delta_ref_ci_hi": 0.094,
    }])
    html = rp._model_overview_block(report_art)
    assert "+0.063 (+0.031 to +0.132)" in html      # beats its own ingredient
    assert "+0.009 (-0.045 to +0.094)" in html      # but not tumour volume
    assert "Tumor volume" in html                    # reference named in a header


def test_model_overview_block_leaves_a_one_predictor_model_empty(report_art):
    """A one-predictor model has no combination to test. Its Δ cells must be
    blank rather than zero — zero would claim the comparison was made and came
    out even."""
    import pandas as pd, report as rp
    report_art.model_overview = pd.DataFrame([{
        "model_id": "top_1_variable", "n_predictors": 1,
        "auc_apparent": 0.679, "auc_corrected": 0.660,
        "best_own_single": None, "best_own_auc_corrected": None,
        "delta_own_corrected": None, "delta_own_apparent": None,
        "delta_own_ci_lo": None, "delta_own_ci_hi": None,
        "reference": None, "reference_auc_corrected": None,
        "delta_ref_corrected": None, "delta_ref_apparent": None,
        "delta_ref_ci_lo": None, "delta_ref_ci_hi": None,
    }])
    html = rp._model_overview_block(report_art)
    assert "0.660" in html
    assert ">nan<" not in html and ">None<" not in html
    assert "+0.000" not in html


def test_delta_cell_formats_point_estimate_then_interval():
    import report as rp
    assert rp._delta_cell(0.063, 0.031, 0.132) == "+0.063 (+0.031 to +0.132)"
    assert rp._delta_cell(-0.051, -0.105, 0.036) == "-0.051 (-0.105 to +0.036)"
    assert rp._delta_cell(0.063, None, None) == "+0.063"
    assert rp._delta_cell(None, 0.0, 1.0) == ""


def test_comparison_block_explains_a_calibration_slope_above_one(tmp_path, report_art):
    """Two of the eleven models come out marginally above 1.0 on the corrected
    calibration slope. Read naively that says "under-confident"; it actually
    means a two-predictor model on 352 patients had nothing to overfit, so the
    correction landed on 1.0 plus noise. The block must say so, or the number
    reads as a finding."""
    import report as rp
    fig = tmp_path / "high_grade__model_comparison.png"
    fig.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    report_art.inferential_figures = [fig]
    html = rp._render_model_comparison("high_grade", report_art)
    assert "no optimism to remove" in html
    assert "not that the model is under-confident" in html
    # And it must not claim the apparent slope was measured — it is asserted.
    assert "exactly 1.0 for every model by construction" in html
