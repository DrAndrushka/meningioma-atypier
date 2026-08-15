"""Tests for paper-style EDA table builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from schema_infer import ColSpec

from eda_paper_tables import (
    build_eda_paper_tables,
    parse_or_ci,
    univariate_or_forest_data,
)


def test_build_eda_paper_tables_binary_and_continuous(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "high_grade": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "sign_a": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "age": rng.normal(60, 12, n),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "sign_a": ColSpec("sign_a", "binary"),
        "age": ColSpec("age", "continuous"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["sign_a", "age"],
        "kind": ["binary", "continuous"],
        "test": ["chi2", "mann_whitney_u"],
        "p": [0.01, 0.02],
        "p_fdr": [0.02, 0.03],
        "in_fdr_family": [True, True],
        "positive_class": [True, True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    assert (tmp_path / "eda" / "tables" / "eda_paper_tables.csv").exists()
    assert set(out["table_kind"]) == {"binary", "continuous"}
    binary = out[out["predictor"] == "sign_a"].iloc[0]
    assert "/" in binary["grade1"] and "%" in binary["grade1"]
    assert binary["auc"] == ""
    cont = out[out["predictor"] == "age"].iloc[0]
    assert "[" in cont["grade1"]
    assert "OR" not in cont["effect"] or "(" in cont["effect"]


def test_build_eda_paper_tables_categorical_reference_coding(tmp_path):
    df = pd.DataFrame({
        "high_grade": [False, False, False, True, True, True] * 10,
        "side": ["left", "right", "midline", "left", "right", "midline"] * 10,
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "side": ColSpec("side", "nominal"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade"],
        "predictor": ["side"],
        "kind": ["nominal"],
        "test": ["chi2"],
        "p": [0.04],
        "p_fdr": [0.05],
        "in_fdr_family": [True],
        "positive_class": [True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    cat = out[out["predictor"] == "side"]
    assert set(cat["table_kind"]) == {"nominal"}
    assert (cat["row_role"] == "variable").sum() == 1
    assert (cat["row_role"] == "reference").sum() == 1
    assert (cat["row_role"] == "level").sum() >= 1
    # parent first
    assert cat.iloc[0]["row_role"] == "variable"
    # n/N (%) on every level, same format as dichotomous
    for _, row in cat[cat["row_role"].isin(["reference", "level"])].iterrows():
        assert "/" in row["grade1"] and "%" in row["grade1"]
        assert "/" in row["grade23"] and "%" in row["grade23"]


def test_build_eda_paper_tables_uses_declared_positive_class(tmp_path):
    df = pd.DataFrame({
        "high_grade": [False, False, True, True, False, True] * 10,
        "side": ["left", "left", "left", "right", "right", "midline"] * 10,
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "side": ColSpec("side", "nominal", positive_class="right"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade"],
        "predictor": ["side"],
        "kind": ["nominal"],
        "test": ["chi2"],
        "p": [0.04],
        "p_fdr": [0.05],
        "in_fdr_family": [True],
        "positive_class": [True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    cat = out[out["predictor"] == "side"]
    ref = cat[cat["row_role"] == "reference"]
    assert len(ref) == 1
    assert ref.iloc[0]["level"] != "right"
    assert "right" in set(cat.loc[cat["row_role"] == "level", "level"])
    assert ref.iloc[0]["effect"] == "— (ref)"


def test_binary_paper_table_inverts_when_false_is_positive(tmp_path):
    df = pd.DataFrame({
        "high_grade": [False] * 20 + [True] * 20,
        "sign": [False] * 15 + [True] * 5 + [False] * 5 + [True] * 15,
    })
    assoc = pd.DataFrame({
        "target": ["high_grade"],
        "predictor": ["sign"],
        "kind": ["binary"],
        "test": ["chi2"],
        "p": [0.01],
        "p_fdr": [0.02],
        "in_fdr_family": [True],
        "positive_class": [True],
    })
    default = build_eda_paper_tables(
        df,
        {"high_grade": ColSpec("high_grade", "binary"),
         "sign": ColSpec("sign", "binary")},
        assoc, output_root=tmp_path / "default",
    )
    inverted = build_eda_paper_tables(
        df,
        {"high_grade": ColSpec("high_grade", "binary"),
         "sign": ColSpec("sign", "binary", positive_class=False)},
        assoc, output_root=tmp_path / "inverted",
    )
    def_or = parse_or_ci(default.iloc[0]["effect"])
    inv_or = parse_or_ci(inverted.iloc[0]["effect"])
    assert def_or is not None and inv_or is not None
    assert def_or[0] > 1
    assert inv_or[0] < 1


def test_build_eda_paper_tables_skips_excluded(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "high_grade": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "keep_me": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "hide_me": pd.array(rng.integers(0, 2, n), dtype="boolean"),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "keep_me": ColSpec("keep_me", "binary"),
        "hide_me": ColSpec("hide_me", "binary"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["keep_me", "hide_me"],
        "kind": ["binary", "binary"],
        "test": ["chi2", "chi2"],
        "p": [0.01, 0.02],
        "p_fdr": [0.02, 0.03],
        "in_fdr_family": [True, True],
        "positive_class": [True, True],
    })
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["hide_me"]}).to_csv(
        cleaning / "eda_excluded_columns.csv", index=False,
    )
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    assert set(out["predictor"]) == {"keep_me"}


def test_build_eda_paper_tables_keeps_exploratory_derived(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame({
        "high_grade": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "dural_tail": pd.array(rng.integers(0, 2, n), dtype="boolean"),
        "venous_sinus_invasion": pd.array(rng.integers(0, 2, n), dtype="boolean"),
    })
    schema = {
        "high_grade": ColSpec("high_grade", "binary"),
        "dural_tail": ColSpec("dural_tail", "binary"),
        "venous_sinus_invasion": ColSpec("venous_sinus_invasion", "binary"),
    }
    assoc = pd.DataFrame({
        "target": ["high_grade", "high_grade"],
        "predictor": ["dural_tail", "venous_sinus_invasion"],
        "kind": ["binary", "binary"],
        "test": ["chi2", "chi2"],
        "p": [0.01, 0.04],
        "p_fdr": [0.02, np.nan],
        "in_fdr_family": [True, False],
        "positive_class": [True, True],
    })
    cleaning = tmp_path / "cleaning"
    cleaning.mkdir(parents=True)
    pd.DataFrame({"column": ["venous_sinus_invasion"]}).to_csv(
        cleaning / "eda_derived_columns.csv", index=False,
    )
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    assert set(out["predictor"]) == {"dural_tail", "venous_sinus_invasion"}
    derived = out[out["predictor"] == "venous_sinus_invasion"].iloc[0]
    assert derived["p_fdr"] != ""
    native = out[out["predictor"] == "dural_tail"].iloc[0]
    assert native["p_fdr"] != ""


def test_parse_or_ci_accepts_en_dash_and_hyphen():
    assert parse_or_ci("2.00 (1.10–3.50)") == (2.0, 1.10, 3.50)
    assert parse_or_ci("1.20 (1.00-1.45)") == (1.20, 1.00, 1.45)
    assert parse_or_ci("") is None
    assert parse_or_ci("ref") is None


def test_univariate_or_forest_data_skips_reference_and_excluded():
    paper = pd.DataFrame([
        {
            "target": "high_grade", "table_kind": "binary",
            "predictor": "dural_tail", "row_role": "variable", "level": "",
            "effect": "2.00 (1.10–3.50)",
        },
        {
            "target": "high_grade", "table_kind": "nominal",
            "predictor": "side", "row_role": "reference", "level": "right",
            "effect": "",
        },
        {
            "target": "high_grade", "table_kind": "nominal",
            "predictor": "side", "row_role": "level", "level": "left",
            "effect": "1.50 (1.10–2.00)",
        },
        {
            "target": "high_grade", "table_kind": "continuous",
            "predictor": "age", "row_role": "variable", "level": "",
            "effect": "1.20 (1.00–1.45)",
        },
        {
            "target": "high_grade", "table_kind": "binary",
            "predictor": "hidden_pred", "row_role": "variable", "level": "",
            "effect": "9.00 (2.00–20.0)",
        },
        {
            "target": "other", "table_kind": "binary",
            "predictor": "dural_tail", "row_role": "variable", "level": "",
            "effect": "3.00 (1.00–9.00)",
        },
    ])
    out = univariate_or_forest_data(
        paper, target="high_grade", excluded={"hidden_pred"},
    )
    assert set(out["predictor"]) == {"dural_tail", "side", "age"}
    age = out[out["predictor"] == "age"].iloc[0]
    assert age["label"] == "Age (per 1 SD)"
    side = out[out["predictor"] == "side"].iloc[0]
    assert "vs" in side["label"]
    assert (out["or"] == 3.0).sum() == 0
    only_age = univariate_or_forest_data(paper, include={"age"})
    assert set(only_age["predictor"]) == {"age"}


# ---------------------------------------------------------------------------
# Forest significance: the grey/black mark must agree with the drawn interval
# ---------------------------------------------------------------------------

def _forest_paper() -> pd.DataFrame:
    """One row of each kind, with an FDR-p that disagrees with its own CI."""
    return pd.DataFrame([
        # q significant AND CI excludes 1 -> black
        {"target": "high_grade", "table_kind": "binary", "predictor": "dural_tail",
         "row_role": "variable", "level": "", "effect": "2.00 (1.10–3.50)",
         "p_fdr": "<0.001", "p_level": "", "sort_p": 0.0005},
        # q significant BUT CI crosses 1 -> grey (the bug)
        {"target": "high_grade", "table_kind": "continuous", "predictor": "edema_index",
         "row_role": "variable", "level": "", "effect": "0.99 (0.78–1.26)",
         "p_fdr": "0.027", "p_level": "0.942", "sort_p": 0.027},
        # q not significant, CI excludes 1 -> grey
        {"target": "high_grade", "table_kind": "binary", "predictor": "hemorrhage",
         "row_role": "variable", "level": "", "effect": "2.23 (1.04–4.75)",
         "p_fdr": "0.053", "p_level": "", "sort_p": 0.053},
        # level row: inherits the omnibus q, own contrast crosses 1 -> grey
        {"target": "high_grade", "table_kind": "nominal", "predictor": "side",
         "row_role": "reference", "level": "right", "effect": "",
         "p_fdr": "", "p_level": "", "sort_p": 0.007},
        {"target": "high_grade", "table_kind": "nominal", "predictor": "side",
         "row_role": "level", "level": "left", "effect": "0.67 (0.42–1.08)",
         "p_fdr": "", "p_level": "0.101", "sort_p": 0.007},
    ])


def test_forest_data_carries_numeric_fdr_p():
    """Every plotted row knows its FDR-p, including ``<0.001`` and level rows."""
    out = univariate_or_forest_data(_forest_paper(), target="high_grade")
    q = out.set_index("predictor")["p_fdr"]
    assert q["dural_tail"] == 0.0005
    assert q["edema_index"] == 0.027
    assert q["hemorrhage"] == 0.053
    assert q["side"] == 0.007  # level row inherits the variable's q


def test_forest_greys_rows_that_are_not_fdr_significant():
    """Grey = not FDR-significant, or an interval that touches the null."""
    import matplotlib.pyplot as plt
    import plot_style as ps

    from eda_paper_tables import draw_univariate_or_forest

    plot_df = univariate_or_forest_data(_forest_paper(), target="high_grade")
    fig, ax = draw_univariate_or_forest(plot_df, fdr_alpha=0.05)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Marker colour per row, matched back to the row label by y position.
    colours = {}
    for line in ax.lines:
        if line.get_marker() != "s":
            continue
        y = int(round(line.get_ydata()[0]))
        colours[labels[len(labels) - 1 - y]] = line.get_color()
    black = [lab for lab, c in colours.items() if c == ps.OKABE["black"]]
    assert any("Dural tail" in lab for lab in black)
    assert all("Edema index" not in lab for lab in black)   # q<0.05 but CI crosses 1
    assert all("Hemorrhage" not in lab for lab in black)    # CI excludes 1 but q>=0.05
    assert all("Side" not in lab for lab in black)          # level contrast crosses 1
    plt.close(fig)


def test_forest_never_draws_a_black_row_across_the_null():
    """The invariant: a full-ink row can never straddle OR = 1."""
    import matplotlib.pyplot as plt
    import plot_style as ps

    from eda_paper_tables import draw_univariate_or_forest

    plot_df = univariate_or_forest_data(_forest_paper(), target="high_grade")
    fig, ax = draw_univariate_or_forest(plot_df, fdr_alpha=0.05)
    bars = 0
    for line in ax.lines:
        ys = line.get_ydata()
        # Interval bars are horizontal (both ends on the same row); the dashed
        # reference line at OR = 1 is vertical.
        if line.get_marker() == "s" or len(ys) != 2 or ys[0] != ys[1]:
            continue
        bars += 1
        lo, hi = sorted(line.get_xdata())
        if line.get_color() == ps.OKABE["black"]:
            assert not (lo <= 1.0 <= hi)
    assert bars == len(plot_df)
    plt.close(fig)


def test_continuous_rows_keep_the_p_of_the_model_they_report(tmp_path):
    """The per-SD OR must be shipped with the p from that same logit fit."""
    rng = np.random.default_rng(3)
    n = 200
    age = rng.normal(60, 12, n)
    y = rng.random(n) < 1 / (1 + np.exp(-(age - 60) / 8))
    df = pd.DataFrame({"high_grade": pd.array(y, dtype="boolean"), "age": age})
    schema = {"high_grade": ColSpec("high_grade", "binary"),
              "age": ColSpec("age", "continuous")}
    assoc = pd.DataFrame({
        "target": ["high_grade"], "predictor": ["age"], "kind": ["continuous"],
        "test": ["mann_whitney_u"], "p": [0.5], "p_fdr": [0.5],
        "in_fdr_family": [True], "positive_class": [True],
    })
    out = build_eda_paper_tables(df, schema, assoc, output_root=tmp_path)
    row = out[out["predictor"] == "age"].iloc[0]
    assert row["p_level"] not in ("", None)
    assert float(str(row["p_level"]).lstrip("<")) < 0.05  # the fit is strong


def test_the_forest_refuses_to_plot_a_number_the_table_does_not_show():
    """A reader trusts the plot over the table, so they must not disagree."""
    paper = pd.DataFrame({
        "target": ["y"], "table_kind": ["binary"], "predictor": ["male"],
        "row_role": ["variable"], "level": [""],
        "effect": ["9.99 (1.00–2.00)"],          # estimate does not match its CI text
        "p_fdr": ["0.01"], "sort_p": [0.01],
    })
    # parse_or_ci reads 9.99/1.00/2.00 back out, so the round-trip matches and
    # this row is fine; corrupt the rendering instead.
    paper.loc[0, "effect"] = "9.99 (1.00–2.00) "  # trailing space is stripped
    assert not univariate_or_forest_data(paper, target="y").empty

    paper.loc[0, "effect"] = "1.5 (1.00–2.00)"    # 1.5 renders as "1.50"
    with pytest.raises(ValueError, match="while the table prints"):
        univariate_or_forest_data(paper, target="y")
