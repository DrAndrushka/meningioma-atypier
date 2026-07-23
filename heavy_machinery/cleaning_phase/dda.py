"""Descriptive data analysis (DDA) on every column before inferential testing.

Summary stats plus histogram/box/bar plots where appropriate. No p-values.
Optional bivariate seaborn plots via ``run_dda_bivariate``
(``{x: [partners]}`` — categorical and continuous partners).
Artifacts → ``output/dda/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, trim_mean

from schema_infer import ColSpec
from heavy_machinery.modelling_phase.plot_style import PALETTE, apply_plot_style, prettify_label

apply_plot_style()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "dda" / "figures"
    tabs = root / "dda" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


# CSV tables are saved raw (NaN preserved). Use cleaning.format_table_for_display in the notebook.

# ---------------------------------------------------------------------------
# Per-column stats
# ---------------------------------------------------------------------------

def _stats_continuous(s: pd.Series) -> dict:
    nn = s.dropna()
    n = len(nn)
    if n == 0:
        return {"n": 0, "n_unique": 0, "missing_pct": 100.0,
                "min": np.nan, "p_5th": np.nan, "median": np.nan,
                "mean": np.nan, "trimmed_mean": np.nan, "p_95th": np.nan,
                "max": np.nan, "mode": np.nan, "std": np.nan, "cv": np.nan,
                "iqr": np.nan, "skewness": np.nan, "kurtosis": np.nan}

    q1, q3 = float(nn.quantile(0.25)), float(nn.quantile(0.75))
    mean = float(nn.mean())
    std = float(nn.std(ddof=1)) if n > 1 else np.nan
    cv = float(std / mean) if (std == std and abs(mean) > 1e-12) else np.nan
    mode_vals = nn.mode()
    return {
        "n": int(n),
        "n_unique": int(nn.nunique()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "min": float(nn.min()),
        "p_5th": float(nn.quantile(0.05)),
        "median": float(nn.median()),
        "mean": mean,
        "trimmed_mean": float(trim_mean(nn, 0.1)) if n >= 5 else np.nan,
        "p_95th": float(nn.quantile(0.95)),
        "max": float(nn.max()),
        "mode": float(mode_vals.iloc[0]) if len(mode_vals) == 1 else np.nan,
        "std": std,
        "cv": cv,
        "iqr": q3 - q1,
        "skewness": float(skew(nn)) if n > 2 else np.nan,
        "kurtosis": float(kurtosis(nn, fisher=True)) if n > 3 else np.nan,
    }


def _stats_categorical(s: pd.Series, ordered: bool) -> dict:
    """Shared stats for ordinal, nominal, and binary columns.

    median_category is the middle level when ``ordered=True``; empty for nominal.
    second_mode / second_mode_pct are filled only when there are 3+ distinct values.
    """
    nn = s.dropna()
    n = int(nn.size)
    base = {
        "n": n,
        "n_unique": int(nn.nunique()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "ordered": ordered,
        "first_mode": np.nan, "first_mode_pct": np.nan,
        "second_mode": np.nan, "second_mode_pct": np.nan,
        "rarest": np.nan, "rarest_pct": np.nan, "max_class_imbalance": np.nan,
        "median_category": np.nan,
        "balance": np.nan, "entropy_bin": np.nan,
    }
    if n == 0:
        return base

    vc = nn.value_counts()  # sorted descending
    base["first_mode"] = vc.index[0]
    base["first_mode_pct"] = round(float(vc.iloc[0] / n * 100), 2)
    if len(vc) > 2:
        base["second_mode"] = vc.index[1]
        base["second_mode_pct"] = round(float(vc.iloc[1] / n * 100), 2)
    else:
        base["second_mode"] = np.nan
        base["second_mode_pct"] = np.nan
    base["rarest"] = vc.index[-1]
    rarest_count = int(vc.iloc[-1])
    base["rarest_pct"] = round(float(rarest_count / n * 100), 2)
    base["max_class_imbalance"] = (round(float(vc.iloc[0] / rarest_count), 2)
                                   if rarest_count > 0 else np.nan)

    # Shannon entropy in bits + normalized balance
    p = vc.values / n
    p = p[p > 0]
    H_bits = float(-(p * np.log2(p)).sum())
    base["entropy_bin"] = round(H_bits, 4)
    k = int((vc > 0).sum())
    base["balance"] = round(H_bits / np.log2(k), 4) if k > 1 else np.nan

    # Median category — only for ordinal
    if ordered:
        if isinstance(s.dtype, pd.CategoricalDtype) and s.dtype.ordered:
            cats = list(s.cat.categories)
            codes = pd.Series(pd.Categorical(nn, categories=cats, ordered=True).codes)
            med_code = codes.median()
            if pd.notna(med_code):
                idx = int(np.floor(med_code))
                idx = max(0, min(idx, len(cats) - 1))
                base["median_category"] = cats[idx]
        else:
            try:
                base["median_category"] = float(nn.median())
            except Exception:
                pass
    return base


def _stats_binary(s: pd.Series) -> dict:
    """Stats for binary columns (mode/rarest; no second_mode or median_category)."""
    nn = s.dropna()
    n = int(nn.size)
    base = {
        "n": n,
        "n_unique": int(nn.nunique()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "ordered": False,
        "mode": np.nan, "mode_pct": np.nan,
        "rarest": np.nan, "rarest_pct": np.nan,
        "max_class_imbalance": np.nan,
        "balance": np.nan, "entropy_bin": np.nan,
    }
    if n == 0:
        return base

    vc = nn.value_counts()
    mode_count = int(vc.iloc[0])
    rarest_count = int(vc.iloc[-1])
    base["mode"] = vc.index[0]
    base["mode_pct"] = round(float(mode_count / n * 100), 2)
    base["rarest"] = vc.index[-1]
    base["rarest_pct"] = round(float(rarest_count / n * 100), 2)
    base["max_class_imbalance"] = (
        round(float(mode_count / rarest_count), 2) if rarest_count > 0 else np.nan
    )

    p = vc.values / n
    p = p[p > 0]
    H_bits = float(-(p * np.log2(p)).sum())
    base["entropy_bin"] = round(H_bits, 4)
    k = int((vc > 0).sum())
    base["balance"] = round(H_bits / np.log2(k), 4) if k > 1 else np.nan
    return base


def _stats_datetime(s: pd.Series) -> dict:
    nn = pd.to_datetime(s, errors="coerce").dropna()
    return {
        "n": int(nn.size),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "min": nn.min() if len(nn) else pd.NaT,
        "max": nn.max() if len(nn) else pd.NaT,
        "span_days": (nn.max() - nn.min()).days if len(nn) else np.nan,
    }


def _stats_id(s: pd.Series) -> dict:
    return {
        "n": int(s.notna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 2),
        "n_unique": int(s.nunique(dropna=True)),
    }


# ---------------------------------------------------------------------------
# Per-column plots
# ---------------------------------------------------------------------------

def _plot_continuous(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    paths = []
    nn = s.dropna()
    if nn.empty:
        return paths

    label = prettify_label(name)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(nn, kde=True, ax=ax, color=PALETTE["primary"])
    ax.set_title(f"{label} — distribution")
    ax.set_xlabel(label); ax.set_ylabel("Count")
    p = out_dir / f"{name}__hist.svg"
    _save_fig(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.boxplot(x=nn, ax=ax, color=PALETTE["primary"])
    ax.set_title(f"{label} — box plot")
    ax.set_xlabel(label)
    p = out_dir / f"{name}__box.svg"
    _save_fig(fig, p); paths.append(p)
    return paths


def _ordinal_bar_order(s: pd.Series, ordered_levels: list | None) -> list:
    obs = set(s.dropna())
    if not obs:
        return []
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in obs]
    if ordered_levels:
        head = [lv for lv in ordered_levels if lv in obs]
        tail = sorted((x for x in obs if x not in ordered_levels), key=str)
        return head + tail
    return sorted(obs, key=str)


def _plot_ordinal(
    s: pd.Series, name: str, out_dir: Path, *, ordered_levels: list | None = None,
) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    order = _ordinal_bar_order(nn, ordered_levels)
    label = prettify_label(name)
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(order) + 2), 4.2))
    sns.countplot(x=nn.astype(str), order=[str(o) for o in order], ax=ax,
                  color=PALETTE["primary"])
    ax.set_title(f"{label} — distribution")
    ax.set_xlabel(label); ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fontsize=9, padding=2)
    long_labels = any(len(str(o)) > 6 for o in order)
    plt.setp(ax.get_xticklabels(), rotation=35 if long_labels else 0,
             ha="right" if long_labels else "center")
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_nominal(s: pd.Series, name: str, out_dir: Path, top_n: int = 15) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    vc = nn.value_counts()
    if len(vc) > top_n:
        top = vc.head(top_n)
        top["(other)"] = vc.iloc[top_n:].sum()
        vc = top
    label = prettify_label(name)
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.45 * len(vc) + 0.8)))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color=PALETTE["primary"])
    ax.set_title(f"{label} — counts"); ax.set_xlabel("Count"); ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fontsize=9, padding=3)
    ax.margins(x=0.12)  # headroom so value labels don't clip the right edge
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_binary(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    nn = s.dropna()
    if nn.empty:
        return []
    counts = nn.value_counts().reindex([True, False]).fillna(0).astype(int)
    label = prettify_label(name)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bar_df = pd.DataFrame({"value": ["True", "False"], "count": counts.values})
    sns.barplot(data=bar_df, x="value", y="count", hue="value",
                palette=[PALETTE["good"], PALETTE["bad"]], legend=False, ax=ax)
    ax.set_title(f"{label} — present vs absent")
    ax.set_xlabel(""); ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fontsize=10, padding=3)
    ax.margins(y=0.12)
    p = out_dir / f"{name}__bar.svg"
    _save_fig(fig, p)
    return [p]


def _plot_datetime(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    nn = pd.to_datetime(s, errors="coerce").dropna()
    if nn.empty:
        return []
    monthly = nn.dt.to_period("M").value_counts().sort_index()
    label = prettify_label(name)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(monthly) + 2), 3.8))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o",
            color=PALETTE["primary"])
    ax.set_title(f"{label} — records over time")
    ax.set_xlabel("Month"); ax.set_ylabel("Count")
    # Thin x ticks so dense monthly axes don't overlap.
    step = max(1, len(monthly) // 18)
    ticks = list(range(0, len(monthly), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(monthly.index[i]) for i in ticks])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = out_dir / f"{name}__timeline.svg"
    _save_fig(fig, p)
    return [p]


# ---------------------------------------------------------------------------
# Bivariate distributions (x × partner via seaborn)
# ---------------------------------------------------------------------------

def _is_continuous_like(s: pd.Series, *, max_cat_levels: int = 12) -> bool:
    """Numeric with many distinct values → continuous; few levels stay categorical."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    return int(s.nunique(dropna=True)) > max_cat_levels


def _ordered_levels(s: pd.Series) -> list:
    levels = list(s.dropna().unique())
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if c in set(levels)]
    return sorted(levels, key=str)


def _plot_continuous_density_by_categorical(
    plot_df: pd.DataFrame,
    cont_col: str,
    cat_col: str,
    out_dir: Path,
    *,
    file_stem: str,
    max_facet_levels: int = 4,
) -> Path:
    """Seaborn KDE of a continuous var by a categorical — facet when few levels.

    Matches the aesthetics density style (density curves, facet titles with n=).
    Many levels → single-axes overlaid KDEs instead of a wide facet strip.
    """
    cont_label = prettify_label(cont_col)
    cat_label = prettify_label(cat_col)
    order = _ordered_levels(plot_df[cat_col])
    n_facet = plot_df.groupby(cat_col, observed=True).size()
    path = out_dir / f"{file_stem}.svg"

    if len(order) <= max_facet_levels:
        g = sns.FacetGrid(
            plot_df, col=cat_col, col_order=order,
            sharex=True, sharey=True, height=4.0, aspect=1.15,
        )
        g.map_dataframe(
            sns.kdeplot, x=cont_col, fill=False, linewidth=1.8,
            color=PALETTE["primary"],
        )
        for ax, level in zip(g.axes.flat, order):
            n = int(n_facet.get(level, 0))
            ax.set_title(f"{level} (n={n})")
            ax.set_ylabel("Density")
        g.set_axis_labels(cont_label, "Density")
        g.fig.suptitle(
            f"{cont_label} density by {cat_label}", y=1.03, fontsize=13,
        )
        g.fig.tight_layout()
        g.fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(g.fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.kdeplot(
            data=plot_df, x=cont_col, hue=cat_col, hue_order=order,
            fill=False, linewidth=1.6, palette="Set2", ax=ax,
        )
        ax.set(
            xlabel=cont_label, ylabel="Density",
            title=f"{cont_label} density by {cat_label}",
        )
        _save_fig(fig, path)
    return path


def _plot_bivariate(
    df: pd.DataFrame, x_col: str, by_col: str, out_dir: Path,
    *,
    max_marker_levels: int = 12,
) -> Path | None:
    """One seaborn bivariate figure for ``(x_col, by_col)`` → SVG path.

    Plot choice:
    - continuous × continuous → scatter + OLS trend
    - continuous ↔ categorical → faceted / overlaid KDE density (not boxplots)
    - categorical × categorical → grouped counts

    Categorical partners with ``<2`` or ``>max_marker_levels`` levels are skipped.
    Continuous partners (e.g. ``adc_value``) are always allowed.
    """
    plot_df = df[[x_col, by_col]].dropna()
    if plot_df.empty:
        return None

    x_cont = _is_continuous_like(plot_df[x_col], max_cat_levels=max_marker_levels)
    by_cont = _is_continuous_like(plot_df[by_col], max_cat_levels=max_marker_levels)

    if not by_cont:
        n_levels = int(plot_df[by_col].nunique(dropna=True))
        if n_levels < 2 or n_levels > max_marker_levels:
            return None
    if not x_cont and x_col in plot_df:
        n_x = int(plot_df[x_col].nunique(dropna=True))
        if n_x < 1:
            return None

    file_stem = f"{x_col}__by__{by_col}"

    # Continuous × continuous — scatter + trend
    if x_cont and by_cont:
        x_label = prettify_label(x_col)
        by_label = prettify_label(by_col)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(
            data=plot_df, x=x_col, y=by_col, ax=ax,
            alpha=0.55, color=PALETTE["primary"], edgecolor="none",
        )
        sns.regplot(
            data=plot_df, x=x_col, y=by_col, ax=ax,
            scatter=False, color=PALETTE["accent"],
        )
        ax.set(
            xlabel=x_label, ylabel=by_label,
            title=f"{x_label} vs {by_label}",
        )
        path = out_dir / f"{file_stem}.svg"
        _save_fig(fig, path)
        return path

    # Continuous ↔ categorical — density (aesthetics-style)
    if x_cont and not by_cont:
        return _plot_continuous_density_by_categorical(
            plot_df, cont_col=x_col, cat_col=by_col, out_dir=out_dir,
            file_stem=file_stem,
        )
    if (not x_cont) and by_cont:
        return _plot_continuous_density_by_categorical(
            plot_df, cont_col=by_col, cat_col=x_col, out_dir=out_dir,
            file_stem=file_stem,
        )

    # Categorical × categorical — counts
    x_label = prettify_label(x_col)
    by_label = prettify_label(by_col)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_order = _ordered_levels(plot_df[x_col])
    hue_order = _ordered_levels(plot_df[by_col])
    sns.countplot(
        data=plot_df, x=x_col, hue=by_col,
        order=x_order, hue_order=hue_order,
        palette="Set2", ax=ax,
    )
    ax.set(
        xlabel=x_label, ylabel="Count",
        title=f"{x_label} by {by_label}",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    path = out_dir / f"{file_stem}.svg"
    _save_fig(fig, path)
    return path


_MARKER_KINDS = frozenset({
    "binary", "nominal", "ordinal", "continuous", "count",
})


def build_dda_bivariate_specs(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    x_cols: list[str],
    *,
    max_marker_levels: int = 12,
    marker_kinds: frozenset[str] | set[str] = _MARKER_KINDS,
) -> dict[str, list[str]]:
    """Optional helper: build ``{x: [partners]}`` from schema kinds.

    Prefer an explicit notebook dict when you want full control. Continuous /
    count partners (e.g. ``adc_value``) are included when present in
    ``marker_kinds``.
    """
    partners: list[str] = []
    for col, spec in schema.items():
        if col not in df.columns or not spec.keep or spec.kind == "skip":
            continue
        if spec.kind not in marker_kinds:
            continue
        if spec.kind in ("binary", "nominal", "ordinal"):
            nuniq = int(df[col].nunique(dropna=True))
            if nuniq < 2 or nuniq > max_marker_levels:
                continue
        partners.append(col)

    return {
        x: [m for m in partners if m != x]
        for x in x_cols
        if x in df.columns
    }


def run_dda_bivariate(
    df: pd.DataFrame,
    bivariate_specs: dict[str, list[str]],
    *,
    output_root: Path | str = "output",
    max_marker_levels: int = 12,
) -> list[Path]:
    """Plot each ``{x_col: [partner, ...]}`` pair with seaborn.

    Example::

        run_dda_bivariate(
            df,
            {"age": ["sex", "who_grade", "adc_value"], "sex": ["adc_value"]},
            output_root=OUTPUT_ROOT,
        )

    Writes SVGs under ``output/dda/figures_bivariate/`` (clears prior SVGs there).
    High-cardinality *categorical* partners are skipped; continuous partners are kept.
    """
    output_root = Path(output_root)
    figs_dir = output_root / "dda" / "figures_bivariate"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for old in figs_dir.glob("*.svg"):
        old.unlink()

    paths: list[Path] = []
    for x_col, by_cols in bivariate_specs.items():
        if x_col not in df.columns:
            continue
        for by_col in by_cols:
            if by_col not in df.columns or by_col == x_col:
                continue
            path = _plot_bivariate(
                df, x_col, by_col, figs_dir,
                max_marker_levels=max_marker_levels,
            )
            if path is not None:
                paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dda(
    df: pd.DataFrame,
    schema: dict[str, ColSpec],
    *,
    output_root: Path | str = "output",
) -> dict[str, pd.DataFrame]:
    """
    Run DDA on every kept column in the schema.

    Returns a dict of overview tables: {"continuous": ..., "categorical": ...,
    "binary": ..., "datetime": ..., "id": ...}, also saved as CSV.
    All figures saved as SVG in output/dda/figures/.
    """
    output_root = Path(output_root)
    figs_dir, tabs_dir = _ensure_dirs(output_root)

    rows_cont, rows_cat, rows_bin, rows_dt, rows_id = [], [], [], [], []

    for col, spec in schema.items():
        if col not in df.columns or not spec.keep or spec.kind == "skip":
            continue

        s = df[col]
        if spec.kind in ("continuous", "count"):
            row = {"column": col, "kind": spec.kind, **_stats_continuous(s)}
            rows_cont.append(row)
            _plot_continuous(s, col, figs_dir)

        elif spec.kind in ("ordinal", "nominal"):
            row = {"column": col, "kind": spec.kind,
                   **_stats_categorical(s, ordered=(spec.kind == "ordinal"))}
            rows_cat.append(row)
            if spec.kind == "ordinal":
                _plot_ordinal(s, col, figs_dir, ordered_levels=spec.ordered_levels)
            else:
                _plot_nominal(s, col, figs_dir)

        elif spec.kind == "binary":
            row = {"column": col, "kind": "binary", **_stats_binary(s)}
            rows_bin.append(row)
            _plot_binary(s, col, figs_dir)

        elif spec.kind == "datetime":
            row = {"column": col, "kind": "datetime", **_stats_datetime(s)}
            rows_dt.append(row)
            _plot_datetime(s, col, figs_dir)

        elif spec.kind in ("id", "text"):
            row = {"column": col, "kind": spec.kind, **_stats_id(s)}
            rows_id.append(row)

    # Canonical column order per table (matches the docstring at top of file).
    CONT_ORDER = ["column", "kind", "n", "n_unique", "missing_pct",
                  "min", "p_5th", "median", "mean", "trimmed_mean",
                  "p_95th", "max", "mode", "std", "cv", "iqr",
                  "skewness", "kurtosis"]
    CAT_ORDER = ["column", "kind", "ordered", "n", "n_unique", "missing_pct",
                 "first_mode", "first_mode_pct",
                 "second_mode", "second_mode_pct",
                 "rarest", "rarest_pct", "max_class_imbalance",
                 "median_category", "balance", "entropy_bin"]
    BIN_ORDER = ["column", "kind", "ordered", "n", "n_unique", "missing_pct",
                 "mode", "mode_pct", "rarest", "rarest_pct",
                 "max_class_imbalance", "balance", "entropy_bin"]

    def _reorder(df_tbl, order):
        if df_tbl.empty:
            return df_tbl
        cols = [c for c in order if c in df_tbl.columns]
        extras = [c for c in df_tbl.columns if c not in cols]
        return df_tbl[cols + extras]

    tables = {
        "continuous": _reorder(pd.DataFrame(rows_cont), CONT_ORDER),
        "categorical": _reorder(pd.DataFrame(rows_cat), CAT_ORDER),
        "binary": _reorder(pd.DataFrame(rows_bin), BIN_ORDER),
        "datetime": pd.DataFrame(rows_dt),
        "id_text": pd.DataFrame(rows_id),
    }
    for name, tbl in tables.items():
        if not tbl.empty:
            tbl.to_csv(tabs_dir / f"dda_{name}.csv", index=False)

    # Overall dataset overview
    analysed_tables = (
        tables["continuous"],
        tables["categorical"],
        tables["binary"],
        tables["datetime"],
        tables["id_text"],
    )
    n_cols_analysed = sum(len(tbl) for tbl in analysed_tables)
    overall = pd.DataFrame([{
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_cols_analysed": n_cols_analysed,
        "missing_cells_pct": round(df.isna().mean().mean() * 100, 2),
    }])
    overall.to_csv(tabs_dir / "dda_overall.csv", index=False)
    tables["overall"] = overall
    return tables
