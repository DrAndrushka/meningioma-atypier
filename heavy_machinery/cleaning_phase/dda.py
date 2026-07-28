"""Descriptive data analysis (DDA) on every column before inferential testing.

Summary stats plus histogram/box/bar plots where appropriate. No p-values.
Optional bivariate / trivariate figures. Style, palette, ``n=`` badges, and SVG
export all go through ``plot_style`` (one pipeline for the whole project).
Artifacts → ``output/dda/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, trim_mean
from statsmodels.nonparametric.smoothers_lowess import lowess

from schema_infer import ColSpec
from heavy_machinery.modelling_phase.plot_style import (
    PALETTE,
    annotate_total_n,
    apply_plot_style,
    categorical_palette,
    maybe_science_style,
    prettify_label,
    save_figure,
)

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
    n = int(len(nn))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(nn, kde=True, ax=ax, color=PALETTE["primary"])
    ax.set_title(f"{label} — distribution")
    ax.set_xlabel(label); ax.set_ylabel("Count")
    annotate_total_n(ax, n)
    p = out_dir / f"{name}__hist.svg"
    save_figure(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    sns.boxplot(x=nn, ax=ax, color=PALETTE["primary"])
    ax.set_title(f"{label} — box plot")
    ax.set_xlabel(label)
    annotate_total_n(ax, n)
    p = out_dir / f"{name}__box.svg"
    save_figure(fig, p); paths.append(p)
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
    annotate_total_n(ax, int(len(nn)))
    long_labels = any(len(str(o)) > 6 for o in order)
    plt.setp(ax.get_xticklabels(), rotation=35 if long_labels else 0,
             ha="right" if long_labels else "center")
    p = out_dir / f"{name}__bar.svg"
    save_figure(fig, p)
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
    annotate_total_n(ax, int(len(nn)))
    ax.margins(x=0.12)  # headroom so value labels don't clip the right edge
    p = out_dir / f"{name}__bar.svg"
    save_figure(fig, p)
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
    annotate_total_n(ax, int(len(nn)))
    ax.margins(y=0.12)
    p = out_dir / f"{name}__bar.svg"
    save_figure(fig, p)
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
    annotate_total_n(ax, int(len(nn)))
    # Thin x ticks so dense monthly axes don't overlap.
    step = max(1, len(monthly) // 18)
    ticks = list(range(0, len(monthly), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(monthly.index[i]) for i in ticks])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = out_dir / f"{name}__timeline.svg"
    save_figure(fig, p)
    return [p]


# ---------------------------------------------------------------------------
# Bivariate distributions (x × partner via seaborn)
# ---------------------------------------------------------------------------

def _display_level(level, *, by_col: str = "") -> str:
    """Human-readable group level; ``high_grade`` bool → Low/High grade."""
    if by_col == "high_grade" and level in (True, False, 1, 0):
        return "High grade" if bool(level) else "Low grade"
    if isinstance(level, (bool, np.bool_)):
        return "Yes" if bool(level) else "No"
    return str(level)

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


def _level_counts(s: pd.Series, order: list) -> dict:
    vc = s.value_counts(dropna=True)
    return {lv: int(vc.get(lv, 0)) for lv in order}


def _plot_continuous_density_by_categorical(
    plot_df: pd.DataFrame,
    cont_col: str,
    cat_col: str,
    out_dir: Path,
    *,
    file_stem: str,
) -> Path:
    """Overlapping hist + KDE of a continuous var, colored by categorical.

    Layered semi-transparent histograms with KDE curves (aesthetics-style).
    Legend labels include ``n=``. Support clipped to observed ``[min, max]``.
    """
    cont_label = prettify_label(cont_col)
    cat_label = prettify_label(cat_col)
    order = _ordered_levels(plot_df[cat_col])
    path = out_dir / f"{file_stem}.svg"

    cont = plot_df[cont_col].astype(float)
    x_lo = float(cont.min())
    x_hi = float(cont.max())
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo, x_hi = (x_lo, x_lo + 1.0) if np.isfinite(x_lo) else (0.0, 1.0)
    clip = (x_lo, x_hi)

    palette = categorical_palette(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for level, color in zip(order, palette):
        sub = plot_df.loc[plot_df[cat_col] == level, cont_col].astype(float).dropna()
        n = int(sub.size)
        if n < 1:
            continue
        use_kde = n >= 2 and float(sub.std(ddof=0)) > 0.0
        sns.histplot(
            sub,
            bins=18,
            kde=use_kde,
            stat="count",
            alpha=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            binrange=clip,
            kde_kws={"clip": clip, "cut": 0} if use_kde else None,
            label=f"{level} (n={n})",
            ax=ax,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set(
        xlabel=cont_label, ylabel="Count",
        title=f"{cont_label} distribution by {cat_label}",
    )
    ax.legend(title=cat_label, frameon=True)
    annotate_total_n(ax, len(plot_df))
    save_figure(fig, path)
    return path


def _plot_bivariate(
    df: pd.DataFrame, x_col: str, by_col: str, out_dir: Path,
    *,
    max_marker_levels: int = 12,
) -> Path | None:
    """One seaborn bivariate figure for ``(x_col, by_col)`` → SVG path.

    Plot choice:
    - continuous × continuous → scatter + OLS trend
    - continuous ↔ categorical → overlapping hist + KDE (legend with n=)
    - categorical × categorical → grouped counts

    All figures show ``n=`` (total and/or per level). Uses a distinct shared palette.
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
    n_total = int(len(plot_df))

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
        annotate_total_n(ax, n_total)
        path = out_dir / f"{file_stem}.svg"
        save_figure(fig, path)
        return path

    # Continuous ↔ categorical — hist + KDE (aesthetics-style)
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
    hue_n = _level_counts(plot_df[by_col], hue_order)
    palette = categorical_palette(len(hue_order))
    sns.countplot(
        data=plot_df, x=x_col, hue=by_col,
        order=x_order, hue_order=hue_order,
        palette=palette, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)
    ax.set(
        xlabel=x_label, ylabel="Count",
        title=f"{x_label} by {by_label}",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        labeled = []
        for lab in labels:
            n_lv = next(
                (hue_n[lv] for lv in hue_order if str(lv) == str(lab)),
                int(hue_n.get(lab, 0)) if lab in hue_n else 0,
            )
            labeled.append(f"{lab} (n={n_lv})")
        ax.legend(handles, labeled, title=by_label, frameon=True)
    annotate_total_n(ax, n_total)
    ax.margins(y=0.14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    path = out_dir / f"{file_stem}.svg"
    save_figure(fig, path)
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


def _plot_trivariate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    by_col: str,
    out_dir: Path,
    *,
    max_marker_levels: int = 12,
    lowess_frac: float = 0.4,
    science_style: str | list[str] | tuple[str, ...] | None = None,
) -> Path | None:
    """One SciencePlots trivariate figure for ``(x, y)`` compared across ``by``.

    Plot choice (``by`` always categorical, ordered levels respected):
    - continuous × continuous → scatter + OLS (dashed) + LOESS (solid);
      legend entries match ``{level}`` / ``{level} straight-line fit`` /
      ``{level} smooth trend``
    - continuous × categorical → dodged box + strip, hue = ``by``
    - categorical × categorical → count bars faceted by ``by``

    ``science_style`` selects SciencePlots sheets (default ``science`` + ``nature`` +
    ``no-latex``; try ``ieee``). Corner badge shows total n on single-axis plots.
    High-cardinality categoricals are skipped.
    """
    plot_df = df[[x_col, y_col, by_col]].dropna()
    if plot_df.empty:
        return None

    by_n = int(plot_df[by_col].nunique(dropna=True))
    if by_n < 2 or by_n > max_marker_levels:
        return None
    if _is_continuous_like(plot_df[by_col], max_cat_levels=max_marker_levels):
        return None

    x_cont = _is_continuous_like(plot_df[x_col], max_cat_levels=max_marker_levels)
    y_cont = _is_continuous_like(plot_df[y_col], max_cat_levels=max_marker_levels)
    for col, is_cont in ((x_col, x_cont), (y_col, y_cont)):
        if is_cont:
            continue
        n_lv = int(plot_df[col].nunique(dropna=True))
        if n_lv < 1 or n_lv > max_marker_levels:
            return None

    file_stem = f"{x_col}__vs__{y_col}__by__{by_col}"
    path = out_dir / f"{file_stem}.svg"
    x_label, y_label, by_label = (
        prettify_label(x_col), prettify_label(y_col), prettify_label(by_col),
    )
    by_order = _ordered_levels(plot_df[by_col])
    by_counts = _level_counts(plot_df[by_col], by_order)
    palette = categorical_palette(len(by_order))
    n_total = int(len(plot_df))
    level_labs = [_display_level(lv, by_col=by_col) for lv in by_order]

    with maybe_science_style(science_style):
        if x_cont and y_cont:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            y_floor = float(plot_df[y_col].min()) >= 0
            for level, color, lab in zip(by_order, palette, level_labs):
                sub = plot_df.loc[plot_df[by_col] == level]
                ax.scatter(
                    sub[x_col], sub[y_col],
                    s=28, alpha=0.7, color=color, edgecolors="white",
                    linewidths=0.4, label=lab, zorder=3,
                )
            for level, color, lab in zip(by_order, palette, level_labs):
                sub = plot_df.loc[plot_df[by_col] == level]
                if len(sub) < 2:
                    continue
                x = sub[x_col].to_numpy(dtype=float)
                y = sub[y_col].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(float(x.min()), float(x.max()), 60)
                y_line = slope * x_line + intercept
                if y_floor:
                    y_line = np.clip(y_line, 0, None)
                ax.plot(
                    x_line, y_line, color=color, linestyle="--", linewidth=1.6,
                    label=f"{lab} straight-line fit", zorder=2,
                )
                smo = lowess(y, x, frac=lowess_frac, return_sorted=True)
                y_s = smo[:, 1]
                if y_floor:
                    y_s = np.clip(y_s, 0, None)
                ax.plot(
                    smo[:, 0], y_s, color=color, linewidth=2.4,
                    solid_capstyle="round", label=f"{lab} smooth trend", zorder=2,
                )
            ax.set(
                xlabel=x_label, ylabel=y_label,
                title=f"{x_label} vs {y_label} by {by_label}",
            )
            ax.minorticks_on()
            ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=0.95)
            annotate_total_n(ax, n_total)
            save_figure(fig, path)
            return path

        if x_cont ^ y_cont:
            cont_col, cat_col = (x_col, y_col) if x_cont else (y_col, x_col)
            cont_label, cat_label = (
                (x_label, y_label) if x_cont else (y_label, x_label)
            )
            cat_order = _ordered_levels(plot_df[cat_col])
            n_cat, n_hue = len(cat_order), len(by_order)
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            box_w = 0.55 / max(n_hue, 1)
            gap = 0.04
            centers = np.arange(n_cat, dtype=float)
            for i, (level, color, lab) in enumerate(zip(by_order, palette, level_labs)):
                offset = (i - (n_hue - 1) / 2) * (box_w + gap)
                positions = centers + offset
                data = [
                    plot_df.loc[
                        (plot_df[cat_col] == cat_lv) & (plot_df[by_col] == level),
                        cont_col,
                    ].astype(float).to_numpy()
                    for cat_lv in cat_order
                ]
                ax.boxplot(
                    data, positions=positions, widths=box_w * 0.85,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "#222222", "linewidth": 1.2},
                    whiskerprops={"color": "#555555", "linewidth": 0.8},
                    capprops={"color": "#555555", "linewidth": 0.8},
                    boxprops={
                        "facecolor": color, "alpha": 0.45,
                        "edgecolor": "#333333", "linewidth": 0.8,
                    },
                )
                # strip
                rng = np.random.default_rng(abs(hash((x_col, y_col, by_col, str(level)))) % (2**32))
                for pos, vals in zip(positions, data):
                    if vals.size == 0:
                        continue
                    jitter = rng.uniform(-box_w * 0.25, box_w * 0.25, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, pos) + jitter, vals,
                        s=10, alpha=0.55, color=color, edgecolors="none", zorder=3,
                    )
                # proxy for legend
                ax.plot([], [], color=color, linewidth=6, alpha=0.55, label=f"{lab} (n={by_counts[level]})")
            ax.set_xticks(centers)
            ax.set_xticklabels([str(c) for c in cat_order], rotation=30, ha="right")
            ax.set(
                xlabel=cat_label, ylabel=cont_label,
                title=f"{cont_label} by {cat_label} × {by_label}",
            )
            ax.minorticks_on()
            ax.legend(loc="best", frameon=True, framealpha=0.95, title=by_label)
            annotate_total_n(ax, n_total)
            save_figure(fig, path)
            return path

        # categorical × categorical × by
        x_order = _ordered_levels(plot_df[x_col])
        y_order = _ordered_levels(plot_df[y_col])
        n_panels = len(by_order)
        y_palette = categorical_palette(len(y_order))
        fig, axes = plt.subplots(
            1, n_panels, figsize=(max(3.6 * n_panels, 7.2), 4.5), squeeze=False,
        )
        x_pos = np.arange(len(x_order), dtype=float)
        bar_w = 0.8 / max(len(y_order), 1)
        for ax, level, lab in zip(axes[0], by_order, level_labs):
            sub = plot_df.loc[plot_df[by_col] == level]
            for j, (y_lv, color) in enumerate(zip(y_order, y_palette)):
                counts = [
                    int(((sub[x_col] == x_lv) & (sub[y_col] == y_lv)).sum())
                    for x_lv in x_order
                ]
                offset = (j - (len(y_order) - 1) / 2) * bar_w
                ax.bar(
                    x_pos + offset, counts, width=bar_w * 0.9,
                    color=color, edgecolor="white", linewidth=0.4,
                    label=str(y_lv),
                )
            ax.set_title(f"{lab} (n={by_counts[level]})")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Count")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(x) for x in x_order], rotation=30, ha="right")
            ax.minorticks_on()
            ax.legend(title=y_label, frameon=True, fontsize=8, framealpha=0.95)
        fig.suptitle(f"{x_label} × {y_label} by {by_label}  (n={n_total})", y=1.02)
        save_figure(fig, path)
        return path


def run_dda_trivariate(
    df: pd.DataFrame,
    trivariate_specs: dict[tuple[str, str], list[str]],
    *,
    output_root: Path | str = "output",
    max_marker_levels: int = 12,
    lowess_frac: float = 0.4,
    science_style: str | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Plot each ``{(x, y): [group, …]}`` triple with SciencePlots + matplotlib.

    Example::

        run_dda_trivariate(
            df,
            {("max_diameter_cm", "tumor_volume"): ["high_grade", "sex"]},
            output_root=OUTPUT_ROOT,
            science_style=["science", "nature", "no-latex"],  # or "ieee"
        )

    ``science_style`` — SciencePlots sheet name or list (default
    ``["science", "nature", "no-latex"]``). Try ``ieee`` / ``bright``+``grid``.

    Writes SVGs under ``output/dda/figures_trivariate/`` (clears prior SVGs there).

    Type matrix (``by`` = ordered/unordered categorical):
    - cont × cont → scatter + OLS (``straight-line fit``) + LOESS (``smooth trend``)
    - cont × cat  → dodged box + strip
    - cat × cat   → count bars faceted by by
    """
    output_root = Path(output_root)
    figs_dir = output_root / "dda" / "figures_trivariate"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for old in figs_dir.glob("*.svg"):
        old.unlink()

    paths: list[Path] = []
    for pair, by_cols in trivariate_specs.items():
        if not (isinstance(pair, tuple) and len(pair) == 2):
            continue
        x_col, y_col = pair
        if x_col not in df.columns or y_col not in df.columns:
            continue
        for by_col in by_cols:
            if by_col not in df.columns or by_col in (x_col, y_col):
                continue
            path = _plot_trivariate(
                df, x_col, y_col, by_col, figs_dir,
                max_marker_levels=max_marker_levels,
                lowess_frac=lowess_frac,
                science_style=science_style,
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
