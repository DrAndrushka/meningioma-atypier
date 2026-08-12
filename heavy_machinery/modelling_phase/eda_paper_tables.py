"""Paper-style univariate EDA tables (native / derived × datatype).

Builds ``output/eda/tables/eda_paper_tables.csv`` from the analysis frame and
the associations screen — used by the HTML report only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cleaning import format_table_for_csv as _format_table_for_csv
from diagnostic_accuracy import (
    _encode_feature_present,
    _odds_ratio_ci,
)
from eda import _encode_binary_target, compute_univariate_auc
from inferential import _fit_logit_robust
from schema_infer import ColSpec

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
    "sort_p",
]


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


def _fmt_n_pct(k: int, n: int) -> str:
    if n <= 0:
        return ""
    return f"{k} ({100.0 * k / n:.1f}%)"


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


def _bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 42,
) -> tuple[float, float, float]:
    """ROC AUC point estimate + percentile bootstrap 95% CI."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if len(y) < 5 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        return np.nan, np.nan, np.nan
    if auc < 0.5:
        auc = 1.0 - auc
        scores = -scores
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, ss = y[idx], scores[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            a = float(roc_auc_score(yy, ss))
        except ValueError:
            continue
        boots.append(a if a >= 0.5 else 1.0 - a)
    if len(boots) < 20:
        return auc, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


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
) -> list[dict[str, Any]]:
    y_enc, _ = _encode_binary_target(df[target], positive_class)
    x_enc = _encode_feature_present(df[predictor])
    pair = pd.concat([y_enc.rename("y"), x_enc.rename("x")], axis=1).dropna()
    if pair.empty:
        return []
    y = pair["y"].astype(int)
    x = pair["x"].astype(int)
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
    sd = float(pair["x"].std(ddof=0))
    z = (pair["x"] - pair["x"].mean()) / sd if sd > 0 else pair["x"] * 0.0
    or_, lo, hi, _p = _logit_or_ci(pair["y"], z)
    auc, auc_lo, auc_hi = _bootstrap_auc_ci(
        pair["y"].to_numpy(), pair["x"].to_numpy(),
    )
    # Prefer associations AUC if bootstrap failed
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
        "p_level": "",
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
        # Reference = most frequent level
        levels = list(pair["x"].astype(str).value_counts().index)
    if len(levels) < 2:
        return []
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
            effect, p_level = "— (ref)", ""
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
            "grade1": _fmt_n_pct(k1, n1),
            "grade23": _fmt_n_pct(k23, n23),
            "effect": effect,
            "auc": "",
            "p_fdr": "",
            "p_level": p_level,
            "sort_p": p_num,
        })
    return rows


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
    # Prefer FDR-family main predictors for the paper tables
    view = associations.copy()
    if "in_fdr_family" in view.columns:
        view = view[view["in_fdr_family"].fillna(True).astype(bool)]
    if "test" in view.columns:
        view = view[view["test"].astype(str) != "skip"]
    if excluded:
        view = view[~view["predictor"].astype(str).isin(excluded)]

    for _, assoc in view.iterrows():
        target = str(assoc["target"])
        pred = str(assoc["predictor"])
        if target not in df.columns or pred not in df.columns:
            continue
        kind = str(assoc.get("kind") or getattr(schema.get(pred), "kind", ""))
        pos = assoc.get("positive_class", None)
        p_fdr = assoc.get("p_fdr", np.nan)
        if kind == "binary":
            rows.extend(_binary_rows(
                df, target, pred, positive_class=pos, p_fdr=p_fdr,
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
            ))

    out = pd.DataFrame(rows, columns=_PAPER_COLUMNS) if rows else pd.DataFrame(
        columns=_PAPER_COLUMNS,
    )
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
