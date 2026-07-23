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
    EffectThresholds,
    MissingThresholds,
    ReportConfig,
    build_report,
    classify_missing,
    classify_or_direction,
    classify_significance,
    details_block,
    effect_badge,
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


def test_strength_tier():
    assert rp._strength_tier(0.5, "corr", EffectThresholds()) == "strong"


def test_effect_badge():
    html = effect_badge(0.5, kind="corr")
    assert "badge" in html


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


def test_render_schema(report_cfg, report_art):
    report_art.schema_summary = pd.DataFrame([{"column": "age", "kind": "continuous"}])
    assert "<section" in render_schema(report_cfg, report_art)


def test_render_dda(report_cfg, report_art):
    report_art.dda_overall = pd.DataFrame([{"n_rows": 4}])
    html = render_dda(report_cfg, report_art)
    assert "<section" in html
    assert "1️⃣ DDA - univariate" in html
    assert "2️⃣ DDA - bivariate" in html
    # Univariate appears before bivariate in the HTML
    assert html.index("1️⃣ DDA - univariate") < html.index("2️⃣ DDA - bivariate")


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


def test_render_dda_bivariate_key_dropdowns(report_cfg, report_art, tmp_path):
    a = tmp_path / "age__by__sex.svg"
    b = tmp_path / "sex__by__adc_value.svg"
    for p in (a, b):
        p.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    report_art.dda_bivariate_figures = [a, b]
    html = render_dda(report_cfg, report_art)
    assert "🔑 Age (1)" in html
    assert "🔑 Sex (1)" in html


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
    assert "Peritumoral Edema" in html
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
    r = pd.Series({"predictor": "age", "test": "spearman", "effect": 0.3})
    assert "age" in rp._eda_direction_phrase(r, "event")


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
