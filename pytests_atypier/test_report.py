"""Tests for report.py — one test per function."""

from __future__ import annotations

import json
from pathlib import Path

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
    render_final_conclusion,
    render_focus_predictor,
    render_header,
    render_inferential,
    render_missingness,
    render_schema,
    render_stats_decoder,
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


def test_focus_eda_figure(tmp_path):
    p = tmp_path / "event__age.svg"
    p.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    html = rp._focus_eda_figure(p)
    assert "img" in html or html == ""


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


def test_render_header(report_cfg, report_art):
    html = render_header(report_cfg, report_art)
    assert "Test" in html


def test_render_cleaning(report_cfg, report_art):
    assert "<section" in render_cleaning(report_cfg, report_art)


def test_render_schema(report_cfg, report_art):
    report_art.schema_summary = pd.DataFrame([{"column": "age", "kind": "continuous"}])
    assert "<section" in render_schema(report_cfg, report_art)


def test_render_dda(report_cfg, report_art):
    report_art.dda_overall = pd.DataFrame([{"n_rows": 4}])
    assert "<section" in render_dda(report_cfg, report_art)


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


def test_render_missingness(report_cfg, report_art):
    assert "<section" in render_missingness(report_cfg, report_art)


def test_render_eda(report_cfg, report_art):
    assert "<section" in render_eda(report_cfg, report_art)


def test_render_inferential(report_cfg, report_art):
    report_art.inferential_multivariable = {
        "event": pd.DataFrame({
            "predictor_col": ["age"], "or": [2.0], "or_ci_lo": [1.2],
            "or_ci_hi": [3.0], "p": [0.01],
        }),
    }
    report_art.inferential_cases = pd.DataFrame([{
        "target": "event",
        "n_complete_cases": 40,
        "n_outcome_events": 12,
        "n_design_columns": 3,
        "epv": 4.0,
    }])
    html = render_inferential(report_cfg, report_art)
    assert "<section" in html
    assert "EPV = 4" in html


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


def test_render_stats_decoder():
    assert "Spearman" in render_stats_decoder() or "spearman" in render_stats_decoder().lower()


def test_dda_row_for_column(report_art):
    report_art.dda_continuous = pd.DataFrame([{"column": "age", "mean": 55.0}])
    row, kind = rp._dda_row_for_column(report_art, "age")
    assert row is not None


def test_figures_for_column(tmp_path):
    p = tmp_path / "age__hist.svg"
    p.write_text("x")
    assert rp._figures_for_column([p], "age") == [p]


def test_inferential_matches():
    assert rp._inferential_matches("sex_F", "sex")


def test_onehot_modeled_level():
    assert rp._onehot_modeled_level("sex_F", "sex") == "F"


def test_invert_or_ci():
    o, lo, hi = rp._invert_or_ci(2.0, 1.5, 3.0)
    assert lo < o < hi or o > 0


def test_or_ci_phrase():
    assert "2.00" in rp._or_ci_phrase(2.0, 1.5, 3.0)


def test_infer_focus_reference():
    dda_row = pd.Series({"first_mode": "M", "second_mode": "F"})
    ref = rp._infer_focus_reference("sex", "F", None, dda_row)
    assert ref == "M"


def test_render_focus_dda_routes():
    row = pd.Series({"column": "age", "kind": "continuous", "mean": 55.0})
    html = rp._render_focus_dda_routes(row, "age")
    assert isinstance(html, str)


def test_render_focus_route_or_card():
    html = rp._render_focus_route_or_card(
        "event", "sex", "F", "M", 2.0, 1.5, 3.0, 0.04,
    )
    assert "focus-route" in html


def test_focus_stat_cards():
    row = pd.Series({"mean": 55.0, "median": 54.0})
    assert "stat-card" in rp._focus_stat_cards(row, kind="continuous")


def test_render_focus_predictor(report_cfg, report_art):
    report_cfg.focus_predictor = "grade"
    report_art.schema_summary = pd.DataFrame([{
        "column": "grade",
        "kind": "ordinal",
        "keep": True,
        "levels": [1, 2, 3],
        "note": "",
    }])
    report_art.dda_continuous = pd.DataFrame([{"column": "age", "mean": 55.0}])
    html = render_focus_predictor(report_cfg, report_art)
    assert isinstance(html, str)
    assert "1 &lt; 2 &lt; 3" in html


def test_focus_schema_display_row():
    ordinal = rp._focus_schema_display_row(pd.Series({
        "kind": "ordinal", "keep": True, "levels": ["low", "mid", "high"],
    }))
    assert ordinal.loc[0, "levels"] == "low < mid < high"
    nominal = rp._focus_schema_display_row(pd.Series({
        "kind": "nominal", "keep": True, "levels": ["primary", "recurrent"],
    }))
    assert nominal.loc[0, "levels"] == "primary, recurrent"


def test_render_final_conclusion(report_cfg, report_art):
    assert isinstance(render_final_conclusion(report_cfg, report_art), str)


def test_render_appendix(report_cfg, report_art):
    html = render_appendix(report_cfg, report_art)
    assert isinstance(html, str)
    assert "Environment &amp; package versions" in html or "Environment & package versions" in html
    assert "<details" in html
    assert "pandas" in html
    assert "Python" in html


def test_render_environment_appendix():
    html = rp._render_environment_appendix()
    assert "Computer / runtime" in html
    assert "Package versions" in html
    assert "pandas" in html
    assert "numpy" in html


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
