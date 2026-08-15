"""Paper-style univariate EDA tables (native / derived × datatype).

Builds ``output/eda/tables/eda_paper_tables.csv`` from the analysis frame and
the associations screen — used by the HTML report and the manuscript forest.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cleaning import format_table_for_csv as _format_table_for_csv
from delong import auc_ci_auto
from diagnostic_accuracy import (
    _encode_feature_present,
    _odds_ratio_ci,
)
from eda import (
    _encode_binary_target,
    benjamini_hochberg,
    compute_univariate_auc,
)
from inferential import _fit_logit_robust
from scales import is_log_scaled, standardise
from schema_infer import ColSpec, match_level

_PAPER_COLUMNS = [
    "target",
    "table_kind",  # nominal | ordinal | continuous | binary
    "predictor",
    "row_role",  # variable | level | reference
    "level",
    "grade1",
    "grade23",
    "effect",  # OR (95% CI) or OR per SD (95% CI)
    "auc",
    "p_fdr",
    "p_level",
    "q_level",  # BH across level rows only — their own family, native/derived
    "sort_p",
]

# Carried from _categorical_rows to _apply_level_fdr, then dropped before the
# CSV is written. BH needs the unrounded Wald p; ``p_level`` has already been
# squashed to "<0.001", which would tie three level rows at 0.0005.
_LEVEL_P_RAW = "p_level_raw"


def _fmt_p(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        s = str(v)
        return s
    if not np.isfinite(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def _fmt_or_ci(or_: float, lo: float, hi: float) -> str:
    if not np.isfinite(or_) or not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"{or_:.2f} ({lo:.2f}–{hi:.2f})"


def _fmt_nn_pct(k: int, n: int) -> str:
    if n <= 0:
        return ""
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def _fmt_median_iqr(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return ""
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]"


def _fmt_auc_ci(auc: float, lo: float, hi: float) -> str:
    if not np.isfinite(auc):
        return ""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{auc:.2f}"
    return f"{auc:.2f} ({lo:.2f}–{hi:.2f})"


def _logit_or_ci(y: pd.Series, x: pd.Series) -> tuple[float, float, float, float]:
    """Univariate logit OR + 95% CI + Wald p for a numeric predictor."""
    pair = pd.concat([
        pd.to_numeric(y, errors="coerce").rename("y"),
        pd.to_numeric(x, errors="coerce").rename("x"),
    ], axis=1).dropna()
    if len(pair) < 5 or pair["y"].nunique() < 2 or pair["x"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    yv = pair["y"].astype(float)
    xv = pair[["x"]].astype(float)
    Xc = sm.add_constant(xv, has_constant="add")
    model = _fit_logit_robust(yv, Xc)
    if model is None or "x" not in getattr(model, "params", pd.Series(dtype=float)).index:
        return np.nan, np.nan, np.nan, np.nan
    beta = float(model.params["x"])
    se = float(model.bse["x"]) if "x" in model.bse.index else np.nan
    p = float(model.pvalues["x"]) if "x" in model.pvalues.index else np.nan
    if not np.isfinite(se) or se <= 0:
        return float(np.exp(beta)), np.nan, np.nan, p
    return (
        float(np.exp(beta)),
        float(np.exp(beta - 1.96 * se)),
        float(np.exp(beta + 1.96 * se)),
        p,
    )


def _binary_rows(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    *,
    positive_class: Any,
    p_fdr: Any,
    feature_positive: Any = None,
) -> list[dict[str, Any]]:
    y_enc, _ = _encode_binary_target(df[target], positive_class)
    x_enc = _encode_feature_present(df[predictor])
    pair = pd.concat([y_enc.rename("y"), x_enc.rename("x")], axis=1).dropna()
    if pair.empty:
        return []
    y = pair["y"].astype(int)
    x = pair["x"].astype(int)
    # x=1 is "present". Default positive class is True. Invert when the
    # declared index class is absent, so n(%) and OR are for that class.
    if match_level([False, True], feature_positive) is False:
        x = (1 - x).astype(int)
    # y=1 → WHO 2–3 (positive); y=0 → WHO 1
    g1 = y == 0
    g23 = y == 1
    k1, n1 = int(((x == 1) & g1).sum()), int(g1.sum())
    k23, n23 = int(((x == 1) & g23).sum()), int(g23.sum())
    tp = int(((x == 1) & (y == 1)).sum())
    fp = int(((x == 1) & (y == 0)).sum())
    fn = int(((x == 0) & (y == 1)).sum())
    tn = int(((x == 0) & (y == 0)).sum())
    or_, lo, hi = _odds_ratio_ci(tp, fp, fn, tn)
    return [{
        "target": target,
        "table_kind": "binary",
        "predictor": predictor,
        "row_role": "variable",
        "level": "",
        "grade1": _fmt_nn_pct(k1, n1),
        "grade23": _fmt_nn_pct(k23, n23),
        "effect": _fmt_or_ci(or_, lo, hi),
        "auc": "",  # binary AUC column left empty by design
        "p_fdr": _fmt_p(p_fdr),
        "p_level": "",
        "sort_p": _coerce_sort_p(p_fdr),
    }]


def _coerce_sort_p(p_fdr: Any) -> float:
    if p_fdr is None or (isinstance(p_fdr, float) and not np.isfinite(p_fdr)):
        return 1.0
    s = str(p_fdr)
    if s.startswith("<"):
        return 0.0005
    try:
        return float(p_fdr)
    except (TypeError, ValueError):
        return 1.0


def _continuous_rows(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    *,
    positive_class: Any,
    p_fdr: Any,
    kind: str,
) -> list[dict[str, Any]]:
    y_enc, _ = _encode_binary_target(df[target], positive_class)
    x = pd.to_numeric(df[predictor], errors="coerce")
    pair = pd.concat([y_enc.rename("y"), x.rename("x")], axis=1).dropna()
    if pair.empty:
        return []
    g1 = pair.loc[pair["y"] == 0, "x"]
    g23 = pair.loc[pair["y"] == 1, "x"]
    # Right-skewed measurements are standardised on the log scale — declared in
    # ``scales``, not here, so this table cannot disagree with the threshold and
    # cut-point phases about what one SD means. Medians and IQRs above stay in
    # clinical units; only the odds ratio is affected.
    z = pd.Series(
        standardise(pair["x"].to_numpy(dtype=float), is_log_scaled(predictor)),
        index=pair.index,
    )
    # ``p_model`` belongs to the per-SD OR printed on this row. The screen's
    # ``p_fdr`` comes from a rank test, which can disagree with a logit slope
    # on a skewed predictor, so both are kept rather than one standing in for
    # the other.
    or_, lo, hi, p_model = _logit_or_ci(pair["y"], z)
    # DeLong — the same estimator, and so the same interval, the cut-point
    # phase publishes for these measurements; declared once in ``delong``. It
    # was a 400-draw percentile bootstrap here, whose lower bound moved in the
    # second decimal with the random seed, so one AUC reached print with two
    # different intervals one report apart.
    auc, auc_lo, auc_hi = auc_ci_auto(
        pair["y"].to_numpy(), pair["x"].to_numpy(),
    )
    # Prefer associations AUC if the interval could not be computed
    if not np.isfinite(auc):
        auc = compute_univariate_auc(
            df, target, predictor, kind,
            positive_class=positive_class, target_kind="binary",
        )
        auc_lo = auc_hi = np.nan
    return [{
        "target": target,
        "table_kind": "continuous",
        "predictor": predictor,
        "row_role": "variable",
        "level": "",
        "grade1": _fmt_median_iqr(g1),
        "grade23": _fmt_median_iqr(g23),
        "effect": _fmt_or_ci(or_, lo, hi),
        "auc": _fmt_auc_ci(auc, auc_lo, auc_hi),
        "p_fdr": _fmt_p(p_fdr),
        "p_level": _fmt_p(p_model),
        "sort_p": _coerce_sort_p(p_fdr),
    }]


def _categorical_rows(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    *,
    positive_class: Any,
    p_fdr: Any,
    ordered_levels: list | None,
    table_kind: str,
    feature_positive: Any = None,
) -> list[dict[str, Any]]:
    y_enc, _ = _encode_binary_target(df[target], positive_class)
    x = df[predictor]
    pair = pd.concat([y_enc.rename("y"), x.rename("x")], axis=1).dropna()
    if pair.empty:
        return []
    if ordered_levels:
        levels = [lv for lv in ordered_levels if (pair["x"] == lv).any()]
        extras = [lv for lv in pair["x"].astype(str).unique() if lv not in levels]
        levels = levels + extras
    else:
        # Fallback reference = most frequent level
        levels = list(pair["x"].astype(str).value_counts().index)
    if len(levels) < 2:
        return []
    pos = match_level(levels, feature_positive)
    if pos is not None:
        rest = [lv for lv in levels if lv != pos and str(lv) != str(pos)]
        ref = rest[0] if rest else pos
    else:
        ref = levels[0]
    n1 = int((pair["y"] == 0).sum())
    n23 = int((pair["y"] == 1).sum())
    p_num = _coerce_sort_p(p_fdr)
    kind = "ordinal" if table_kind == "ordinal" else "nominal"
    rows: list[dict[str, Any]] = [{
        "target": target,
        "table_kind": kind,
        "predictor": predictor,
        "row_role": "variable",
        "level": "",
        "grade1": "",
        "grade23": "",
        "effect": "",
        "auc": "",
        "p_fdr": _fmt_p(p_fdr),
        "p_level": "",
        "sort_p": p_num,
    }]
    for lv in levels:
        mask = pair["x"].astype(str) == str(lv)
        k1 = int(((pair["y"] == 0) & mask).sum())
        k23 = int(((pair["y"] == 1) & mask).sum())
        is_ref = str(lv) == str(ref)
        if is_ref:
            effect, p_level, p_lv = "— (ref)", "", np.nan
            role = "reference"
        else:
            # 2×2: level vs rest-of-reference collapsed as level vs ref only
            in_lv = mask
            in_ref = pair["x"].astype(str) == str(ref)
            sub = pair.loc[in_lv | in_ref]
            x_bin = (sub["x"].astype(str) == str(lv)).astype(float)
            or_, lo, hi, p_lv = _logit_or_ci(sub["y"], x_bin)
            effect = _fmt_or_ci(or_, lo, hi)
            p_level = _fmt_p(p_lv)
            role = "level"
        rows.append({
            "target": target,
            "table_kind": kind,
            "predictor": predictor,
            "row_role": role,
            "level": str(lv),
            "grade1": _fmt_nn_pct(k1, n1),
            "grade23": _fmt_nn_pct(k23, n23),
            "effect": effect,
            "auc": "",
            "p_fdr": "",
            "p_level": p_level,
            _LEVEL_P_RAW: float(p_lv) if role == "level" else np.nan,
            "sort_p": p_num,
        })
    return rows


def _apply_level_fdr(
    out: pd.DataFrame,
    *,
    derived_cols: frozenset[str] | set[str],
    native_preds: frozenset[str] | set[str],
) -> pd.DataFrame:
    """Benjamini–Hochberg across level rows, native and derived kept apart.

    A level row's p is the Wald p of its own level-vs-reference logit, so those
    p's are as multiple as the variable rows above them and must not be printed
    raw in a column headed FDR-p. The split mirrors
    ``apply_native_and_derived_fdr``: derived flags never join the native
    family, and a predictor in neither family gets no q at all — its level rows
    stay blank, exactly as its variable row does.
    """
    out = out.copy()
    out["q_level"] = ""
    if out.empty or _LEVEL_P_RAW not in out.columns:
        return out
    derived = {str(c) for c in derived_cols}
    native = {str(p) for p in native_preds} - derived
    preds = out["predictor"].astype(str)
    is_level = out["row_role"].astype(str) == "level"
    p_raw = pd.to_numeric(out[_LEVEL_P_RAW], errors="coerce")
    for target in out["target"].astype(str).unique():
        in_target = out["target"].astype(str) == target
        for family in (native, derived):
            mask = in_target & is_level & preds.isin(family) & p_raw.notna()
            if not mask.any():
                continue
            q = benjamini_hochberg(p_raw[mask])
            out.loc[mask, "q_level"] = [_fmt_p(v) for v in q]
    return out


def build_eda_paper_tables(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    associations: pd.DataFrame,
    *,
    output_root: Path | str | None = None,
    excluded_predictors: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Build long paper-table rows for every associations predictor."""
    if associations is None or associations.empty:
        out = pd.DataFrame(columns=_PAPER_COLUMNS)
        if output_root is not None:
            tabs = Path(output_root) / "eda" / "tables"
            tabs.mkdir(parents=True, exist_ok=True)
            out.to_csv(tabs / "eda_paper_tables.csv", index=False)
        return out

    excluded = set(excluded_predictors or ())
    if output_root is not None:
        for fname in ("eda_excluded_columns.csv", "hidden_parent_columns.csv"):
            excl_path = Path(output_root) / "cleaning" / fname
            if not excl_path.exists():
                continue
            try:
                excl_tbl = pd.read_csv(excl_path)
                if "column" in excl_tbl.columns:
                    excluded |= {str(c) for c in excl_tbl["column"].dropna().tolist()}
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    # FDR family is a multiplicity correction, not an inclusion filter.
    # ``eda_in_derived=True`` columns stay in the Derived block; they get a
    # separate BH so FDR-p is shown without counting toward the native family.
    view = associations.copy()
    if "test" in view.columns:
        view = view[view["test"].astype(str) != "skip"]
    if excluded:
        view = view[~view["predictor"].astype(str).isin(excluded)]

    derived_cols: frozenset[str] | set[str] = frozenset()
    native_preds: set[str] = set()
    if not view.empty:
        from eda import apply_native_and_derived_fdr, _load_eda_derived_columns

        derived_cols = (
            _load_eda_derived_columns(output_root) if output_root is not None
            else frozenset()
        )
        if "in_fdr_family" in view.columns:
            fam = [
                str(p) for p in view.loc[
                    view["in_fdr_family"].fillna(True).astype(bool), "predictor"
                ].unique()
            ]
        else:
            fam = None
        view = apply_native_and_derived_fdr(
            view, derived_cols=derived_cols, fdr_family=fam,
        )
        # Read the native family back off the call that defined it, so the
        # level correction cannot drift from the variable correction.
        native_preds = {
            str(p) for p in view.loc[
                view["in_fdr_family"].astype(bool), "predictor"
            ].unique()
        }

    for _, assoc in view.iterrows():
        target = str(assoc["target"])
        pred = str(assoc["predictor"])
        if target not in df.columns or pred not in df.columns:
            continue
        kind = str(assoc.get("kind") or getattr(schema.get(pred), "kind", ""))
        pos = assoc.get("positive_class", None)
        p_fdr = assoc.get("p_fdr", np.nan)
        if kind == "binary":
            spec = schema.get(pred)
            rows.extend(_binary_rows(
                df, target, pred, positive_class=pos, p_fdr=p_fdr,
                feature_positive=getattr(spec, "positive_class", None),
            ))
        elif kind in ("continuous", "count"):
            rows.extend(_continuous_rows(
                df, target, pred, positive_class=pos, p_fdr=p_fdr, kind=kind,
            ))
        elif kind in ("nominal", "ordinal"):
            spec = schema.get(pred)
            levels = spec.ordered_levels if spec is not None else None
            rows.extend(_categorical_rows(
                df, target, pred, positive_class=pos, p_fdr=p_fdr,
                ordered_levels=levels, table_kind=kind,
                feature_positive=getattr(spec, "positive_class", None),
            ))

    cols = _PAPER_COLUMNS + [_LEVEL_P_RAW]
    out = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    # Level rows carry their own multiplicity, so they get their own BH family.
    out = _apply_level_fdr(
        out, derived_cols=derived_cols, native_preds=native_preds,
    ).drop(columns=[_LEVEL_P_RAW])
    if not out.empty:
        role_ord = {"variable": 0, "reference": 1, "level": 2}
        kind_ord = {"nominal": 0, "ordinal": 1, "continuous": 2, "binary": 3}
        out["_role_ord"] = out["row_role"].map(role_ord).fillna(9)
        out["_kind_ord"] = out["table_kind"].map(kind_ord).fillna(9)
        out = out.sort_values(
            ["target", "_kind_ord", "sort_p", "predictor", "_role_ord", "level"],
            ascending=[True, True, True, True, True, True],
        ).drop(columns=["_role_ord", "_kind_ord"]).reset_index(drop=True)

    if output_root is not None:
        tabs = Path(output_root) / "eda" / "tables"
        tabs.mkdir(parents=True, exist_ok=True)
        _format_table_for_csv(out).to_csv(tabs / "eda_paper_tables.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Unadjusted OR forest (same numbers as the paper table)
# ---------------------------------------------------------------------------

_OR_CI_PAT = re.compile(
    r"(?P<est>\d+\.?\d*)\s*\((?P<lo>\d+\.?\d*)\s*[–-]\s*(?P<hi>\d+\.?\d*)\)"
)


def parse_or_ci(text: Any) -> tuple[float, float, float] | None:
    """Parse ``1.20 (1.00–1.45)`` from a paper-table effect cell."""
    m = _OR_CI_PAT.search(str(text))
    if m is None:
        return None
    return tuple(map(float, m.group("est", "lo", "hi")))


def _forest_row_label(row: pd.Series, refs: pd.Series) -> str:
    from plot_style import prettify_label, prettify_level

    name = prettify_label(row["predictor"])
    kind = str(row["table_kind"])
    if kind == "continuous":
        return f"{name} (per 1 SD)"
    if kind in ("nominal", "ordinal"):
        ref = refs.get(row["predictor"], "")
        lvl = prettify_level(row["level"], row["predictor"])
        ref_lab = prettify_level(ref, row["predictor"]) if str(ref) else ""
        if ref_lab:
            return f"{name}: {lvl} vs {ref_lab}"
        return f"{name}: {lvl}"
    return name


def univariate_or_forest_data(
    paper: pd.DataFrame,
    *,
    target: str | None = None,
    excluded: frozenset[str] | set[str] | None = None,
    include: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Rows with ``label`` / ``or`` / ``lo`` / ``hi`` / ``p_fdr`` for the forest.

    ``p_fdr`` is numeric (``NaN`` when unknown) and is the predictor's
    FDR-adjusted p, carried onto its level rows as well so every plotted row
    can be marked significant or not.
    """
    empty = pd.DataFrame(
        columns=["table_kind", "predictor", "level", "label", "or", "lo", "hi", "p_fdr"],
    )
    if paper is None or paper.empty:
        return empty

    sub = paper.copy()
    if target is not None and "target" in sub.columns:
        sub = sub[sub["target"].astype(str) == str(target)]
    if excluded:
        sub = sub[~sub["predictor"].astype(str).isin(excluded)]
    if include is not None:
        sub = sub[sub["predictor"].astype(str).isin(include)]
    if sub.empty:
        return empty

    refs = (
        sub.loc[sub["row_role"].astype(str).eq("reference"), ["predictor", "level"]]
        .drop_duplicates("predictor")
        .set_index("predictor")["level"]
        if "row_role" in sub.columns
        else pd.Series(dtype=object)
    )

    # ``sort_p`` is the predictor's numeric FDR-p repeated on its level rows;
    # ``p_fdr`` is the formatted one and is blank on level rows.
    q_by_pred = _forest_q_by_predictor(sub)

    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        parsed = parse_or_ci(row.get("effect"))
        if parsed is None:
            continue
        est, lo, hi = parsed
        # The forest and the table must show the same number. They are built
        # from one source, so this can only fail if the parse drifts from the
        # formatter — but a plot that disagrees with the table beside it is the
        # kind of thing a reader trusts the plot on, so it is checked rather
        # than assumed.
        rendered = _fmt_or_ci(est, lo, hi)
        if rendered != str(row.get("effect")).strip():
            raise ValueError(
                f"Forest row for {row['predictor']!r} would plot "
                f"{rendered!r} while the table prints "
                f"{str(row.get('effect')).strip()!r}."
            )
        rows.append({
            "table_kind": row["table_kind"],
            "predictor": row["predictor"],
            "level": row["level"],
            "label": _forest_row_label(row, refs),
            "or": est,
            "lo": lo,
            "hi": hi,
            "p_fdr": q_by_pred.get(str(row["predictor"]), np.nan),
        })
    return pd.DataFrame(rows)


def _forest_q_by_predictor(sub: pd.DataFrame) -> dict[str, float]:
    """Numeric FDR-p per predictor, read from ``sort_p`` or ``p_fdr``."""
    out: dict[str, float] = {}
    for pred, grp in sub.groupby(sub["predictor"].astype(str)):
        vals = []
        for col in ("sort_p", "p_fdr"):
            if col not in grp.columns:
                continue
            for raw in grp[col]:
                if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
                    continue
                if str(raw).strip() == "":
                    continue
                vals.append(_coerce_sort_p(raw))
            if vals:
                break
        out[pred] = min(vals) if vals else np.nan
    return out


def draw_univariate_or_forest(plot_df: pd.DataFrame, *, fdr_alpha: float = 0.05):
    """Forest of unadjusted ORs. Caller owns saving / closing the figure.

    A row is drawn in full ink only when its FDR-p clears ``fdr_alpha`` **and**
    its interval excludes 1; everything else is grey. Both conditions are
    needed because the FDR-p comes from the screening test (a rank test for
    continuous predictors) while the interval comes from a logistic fit, and on
    a skewed predictor the two can disagree.
    """
    from plot_style import FIG_WIDTH_DOUBLE, apply_plot_style, forest_lr

    apply_plot_style()
    if "p_fdr" in plot_df.columns:
        q = pd.to_numeric(plot_df["p_fdr"], errors="coerce").to_numpy(dtype=float)
        ns = ~(q < fdr_alpha)  # NaN -> not significant
    else:
        ns = None
    fig, ax = forest_lr(
        plot_df["label"].tolist(),
        plot_df["or"].to_numpy(dtype=float),
        plot_df["lo"].to_numpy(dtype=float),
        plot_df["hi"].to_numpy(dtype=float),
        ref=1.0,
        xlabel="Unadjusted odds ratio (95% CI)",
        value_header="OR (95% CI)",
        width=FIG_WIDTH_DOUBLE,
        ns=ns,
    )
    return fig, ax
