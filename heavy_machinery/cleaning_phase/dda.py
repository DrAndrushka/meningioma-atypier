"""Descriptive data analysis (DDA) on every column before inferential testing.

Figures are strictly *descriptive*: distributions, proportions with their
denominators and Wilson confidence intervals, and non-parametric trends. No
p-values and no fitted models live here — inference belongs to the EDA stage,
so a DDA figure can never be read as a test result.

Optional bivariate / trivariate figures. Style, palette, geometry, and SVG
export all go through ``plot_style`` (one pipeline for the whole project).
Artifacts → ``output/dda/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, trim_mean

from schema_infer import ColSpec, pin_positive_last
from heavy_machinery.modelling_phase.plot_style import (
    FIG_WIDTH_DOUBLE,
    FIG_WIDTH_MEDIUM,
    PALETTE,
    apply_plot_style,
    boxplot_orientation,
    categorical_palette,
    describe_continuous,
    deterministic_rng,
    figure_size,
    histogram_bin_edges,
    kde_curve,
    level_tick_labels,
    lowess_curve,
    maybe_science_style,
    n_subtitle,
    place_legend,
    prettify_label,
    prettify_level,
    proportion_bars,
    raincloud,
    save_figure,
    set_titles,
    width_for_levels,
)

apply_plot_style()

_BOX_HORIZONTAL = boxplot_orientation(False)

# Nominal columns keep this many levels before the tail is pooled into "(other)".
DEFAULT_TOP_N_LEVELS = 15

# Above this many levels (or with long labels) bars are drawn horizontally so
# category names stay readable instead of being rotated to 45°.
_HORIZONTAL_BAR_LEVELS = 6
_LONG_LABEL_CHARS = 12


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    figs = root / "dda" / "figures"
    tabs = root / "dda" / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


def _load_hidden_parent_columns(output_root: Path | str) -> frozenset[str]:
    """Columns with ``hide_parent=True`` (written by apply_derivations)."""
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    if not path.exists():
        return frozenset()
    try:
        tbl = pd.read_csv(path)
    except Exception:
        return frozenset()
    if "column" not in tbl.columns:
        return frozenset()
    return frozenset(str(c) for c in tbl["column"].dropna().tolist())


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

def _numeric_values(s: pd.Series) -> np.ndarray:
    """Finite float values of a series (NaN / inf dropped)."""
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


# A single value holding this much of the mass makes the distribution part
# discrete (structural zeros, floor effects). A Gaussian KDE smooths that spike
# away and understates it badly, so the histogram is left to speak alone.
_POINT_MASS_SHARE = 0.15


def _has_point_mass(values: np.ndarray, share: float = _POINT_MASS_SHARE) -> bool:
    """True when one repeated value dominates — KDE would misrepresent it."""
    if values.size == 0:
        return False
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max()) / float(values.size) > share


def _plot_continuous(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    """One figure per continuous column: marginal box + strip over a histogram.

    Shape, spread, and every raw observation in a single panel-aligned figure —
    the previous split into a separate histogram and box plot forced the reader
    to align two x-axes by eye.

    Counts (not densities) are on the y-axis, and the KDE is rescaled into count
    units and clipped to the observed range, so the curve never claims support
    where no patient exists.
    """
    vals = _numeric_values(s)
    if vals.size == 0:
        return []

    label = prettify_label(name)
    n = int(vals.size)
    lo, hi = float(vals.min()), float(vals.max())
    degenerate = not (hi > lo)

    width = FIG_WIDTH_MEDIUM
    fig, (ax_top, ax_hist) = plt.subplots(
        2, 1, sharex=True,
        figsize=figure_size(width, height=width * 0.78),
        gridspec_kw={"height_ratios": [1, 3.6], "hspace": 0.06},
    )

    counts, edges, _ = ax_hist.hist(
        vals, bins=histogram_bin_edges(vals),
        color=PALETTE["primary"], edgecolor="white", linewidth=0.6, zorder=2,
    )
    smoothable = not degenerate and not _has_point_mass(vals)
    curve = kde_curve(vals, clip=(lo, hi)) if smoothable else None
    if curve is not None:
        xs, dens = curve
        bin_width = float(edges[1] - edges[0])
        ax_hist.plot(
            xs, dens * n * bin_width, color=PALETTE["accent"],
            linewidth=1.4, zorder=3, label="Kernel density",
        )
        place_legend(ax_hist, loc="upper right")
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Count")
    ax_hist.set_ylim(0, max(float(np.max(counts)) * 1.12, 1.0))

    # Marginal summary: box (median / IQR / whiskers) above the raw observations.
    ax_top.boxplot(
        [vals], positions=[0.42], widths=0.42, showfliers=False,
        patch_artist=True,
        medianprops={"color": "#1a1a1a", "linewidth": 1.3},
        whiskerprops={"color": "#4d4d4d", "linewidth": 0.9},
        capprops={"color": "#4d4d4d", "linewidth": 0.9},
        boxprops={
            "facecolor": "white", "edgecolor": PALETTE["primary"],
            "linewidth": 1.0,
        },
        **_BOX_HORIZONTAL,
    )
    rng = deterministic_rng("dda", "continuous", name)
    shown = vals if vals.size <= 500 else rng.choice(vals, size=500, replace=False)
    ax_top.scatter(
        shown, -0.42 + rng.uniform(-0.16, 0.16, size=shown.size),
        s=5, color=PALETTE["primary"], alpha=0.4, edgecolors="none",
    )
    ax_top.set_ylim(-0.95, 0.95)
    ax_top.set_yticks([])
    for side in ("left", "right", "top"):
        ax_top.spines[side].set_visible(False)
    ax_top.tick_params(axis="both", which="both", length=0)

    set_titles(ax_top, label, n_subtitle(n, extra=describe_continuous(vals)))
    fig.align_ylabels([ax_top, ax_hist])
    p = save_figure(fig, out_dir / f"{name}__distribution", tight_layout=False)
    return [p]


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


def _plot_category_proportions(
    s: pd.Series,
    name: str,
    out_dir: Path,
    *,
    order: list,
    note: str | None = None,
) -> list[Path]:
    """Percentage of the cohort in each level, with Wilson 95% CIs and ``k/n``.

    One figure shape for ordinal, nominal, and binary columns. Bars carry their
    denominator and interval so a level with 3 of 6 observations cannot be
    mistaken for one with 150 of 300 — the previous ``stat="density"`` bars
    showed neither, and mislabelled a proportion as a density.
    """
    nn = s.dropna()
    if nn.empty or not order:
        return []

    total = int(nn.size)
    counts = np.array([int((nn == lv).sum()) for lv in order], dtype=float)
    # A pooled "(other)" bucket is prepared by the caller as a plain count.
    labels = level_tick_labels(order, column=name)
    label = prettify_label(name)

    # Labels are already soft-wrapped, so only genuinely long names force the
    # horizontal layout; rotated 45° tick labels are avoided entirely.
    longest_line = max(
        (len(line) for lab in labels for line in lab.split("\n")), default=0,
    )
    horizontal = (
        len(order) > _HORIZONTAL_BAR_LEVELS
        or (len(order) > 3 and longest_line > _LONG_LABEL_CHARS)
    )
    pct_axis = "Percentage of observations (%)"

    if horizontal:
        height = 0.42 * len(order) + 1.5
        fig, ax = plt.subplots(
            figsize=figure_size(FIG_WIDTH_MEDIUM, height=height),
        )
        pct = proportion_bars(
            ax, counts, np.full(counts.shape, total),
            orient="h", color=PALETTE["primary"], width=0.68,
        )
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(pct_axis)
        ax.set_xlim(0, min(100.0, max(float(np.nanmax(pct)) + 24.0, 15.0)))
        ax.set_ylabel("")
    else:
        fig, ax = plt.subplots(
            figsize=figure_size(width_for_levels(len(order))),
        )
        pct = proportion_bars(
            ax, counts, np.full(counts.shape, total),
            orient="v", color=PALETTE["primary"], width=0.62,
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.65, len(order) - 0.35)
        ax.set_ylabel(pct_axis)
        ax.set_ylim(0, min(100.0, max(float(np.nanmax(pct)) * 1.35, 12.0)))
        ax.set_xlabel(label if len(order) > 2 else "")

    set_titles(ax, label, n_subtitle(total, extra=note))
    p = save_figure(fig, out_dir / f"{name}__bar")
    return [p]


def _plot_ordinal(
    s: pd.Series, name: str, out_dir: Path, *, ordered_levels: list | None = None,
) -> list[Path]:
    """Ordinal levels stay in schema order — never re-sorted by frequency."""
    nn = s.dropna()
    if nn.empty:
        return []
    return _plot_category_proportions(
        s, name, out_dir, order=_ordinal_bar_order(nn, ordered_levels),
    )


def _plot_nominal(
    s: pd.Series, name: str, out_dir: Path, top_n: int = DEFAULT_TOP_N_LEVELS,
    *,
    positive_class: Any = None,
) -> list[Path]:
    """Nominal levels sorted by frequency; the tail is pooled and disclosed."""
    nn = s.dropna()
    if nn.empty:
        return []
    vc = nn.value_counts()
    note = None
    if len(vc) > top_n:
        pooled = int(vc.iloc[top_n:].sum())
        n_pooled = int(len(vc) - top_n)
        note = f"{n_pooled} rarest levels pooled ({pooled} observations)"
        order = list(vc.head(top_n).index)
    else:
        order = list(vc.index)
    order = pin_positive_last(order, positive_class)
    return _plot_category_proportions(s, name, out_dir, order=order, note=note)


def _plot_binary(
    s: pd.Series, name: str, out_dir: Path, *, positive_class: Any = None,
) -> list[Path]:
    """Absent before present, one neutral colour — no good/bad encoding.

    Colouring a radiological sign green/red asserts a prognostic direction the
    descriptive stage has not established. A declared ``positive_class`` is
    drawn last so the bar order matches the table contrast (baseline → index).
    """
    nn = s.dropna()
    if nn.empty:
        return []
    observed = set(nn.unique())
    order = [lv for lv in (False, True) if lv in observed]
    if not order:
        order = sorted(observed, key=str)
    order = pin_positive_last(order, positive_class)
    return _plot_category_proportions(s, name, out_dir, order=order)


def _plot_datetime(s: pd.Series, name: str, out_dir: Path) -> list[Path]:
    """Monthly record counts on a true calendar axis.

    Months with no records are drawn as gaps rather than dropped: collapsing
    empty periods compresses the time axis and invents accrual that never
    happened.
    """
    nn = pd.to_datetime(s, errors="coerce").dropna()
    if nn.empty:
        return []
    monthly = nn.dt.to_period("M").value_counts().sort_index()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range, fill_value=0)
    x = monthly.index.to_timestamp()

    label = prettify_label(name)
    width = float(np.clip(0.16 * len(monthly) + 2.4, FIG_WIDTH_MEDIUM, FIG_WIDTH_DOUBLE))
    fig, ax = plt.subplots(figsize=figure_size(width, aspect=0.42))
    ax.bar(
        x, monthly.to_numpy(dtype=float),
        width=pd.Timedelta(days=24), color=PALETTE["primary"],
        edgecolor="white", linewidth=0.4, align="center",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Records")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylim(0, max(float(monthly.max()) * 1.15, 1.0))
    span = f"{full_range[0]} to {full_range[-1]}"
    set_titles(ax, f"{label} — records over time", n_subtitle(int(nn.size), extra=span))
    p = save_figure(fig, out_dir / f"{name}__timeline")
    return [p]


# ---------------------------------------------------------------------------
# Bivariate distributions (x × partner)
# ---------------------------------------------------------------------------

def _display_level(level, *, by_col: str = "") -> str:
    """Human-readable group level (delegates to the shared label map)."""
    return prettify_level(level, by_col)


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
    """Distribution of a continuous variable across categorical levels.

    Drawn as a raincloud (half-violin + box + raw points) instead of layered
    translucent histograms. Overlaid histograms are unreadable with more than
    two groups — the blended overlap reads as a third category, and taller bars
    hide shorter ones outright. Each group here occupies its own slot, so
    density, median/IQR, and every observation are all visible and no group can
    conceal another.

    Densities are clipped to each group's observed range, so the curve cannot
    imply values no patient had.
    """
    cont_label = prettify_label(cont_col)
    cat_label = prettify_label(cat_col)
    order = _ordered_levels(plot_df[cat_col])
    path = out_dir / file_stem

    groups = [
        plot_df.loc[plot_df[cat_col] == lv, cont_col].astype(float).dropna().to_numpy()
        for lv in order
    ]
    counts = [int(g.size) for g in groups]
    palette = categorical_palette(len(order))

    fig, ax = plt.subplots(
        figsize=figure_size(width_for_levels(len(order), per_level=0.85, base=1.6)),
    )
    raincloud(
        ax, groups, colors=palette,
        rng=deterministic_rng("dda", "bivariate", cont_col, cat_col),
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(level_tick_labels(order, counts, column=cat_col))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xlabel(cat_label)
    ax.set_ylabel(cont_label)
    set_titles(
        ax, f"{cont_label} by {cat_label}",
        n_subtitle(int(len(plot_df))),
    )
    return save_figure(fig, path)


def _plot_categorical_by_categorical(
    plot_df: pd.DataFrame,
    x_col: str,
    by_col: str,
    out_dir: Path,
    *,
    file_stem: str,
) -> Path:
    """Composition of ``by_col`` *within* each level of ``x_col``.

    Conditional proportions with Wilson CIs and explicit denominators. The
    previous ``stat="density"`` dodge normalised over the whole table, so bar
    heights answered no well-posed question; a within-``x`` percentage does,
    and the CI keeps a 2-patient cell from looking like a 200-patient one.
    """
    x_label = prettify_label(x_col)
    by_label = prettify_label(by_col)
    x_order = _ordered_levels(plot_df[x_col])
    hue_order = _ordered_levels(plot_df[by_col])
    palette = categorical_palette(len(hue_order))

    totals = np.array([int((plot_df[x_col] == lv).sum()) for lv in x_order], dtype=float)
    centres = np.arange(len(x_order), dtype=float)
    slot = 0.78
    bar_w = slot / max(len(hue_order), 1)

    plot_width = width_for_levels(
        len(x_order) * max(len(hue_order), 1), per_level=0.42, base=2.0,
    )
    fig, ax = plt.subplots(
        figsize=figure_size(plot_width + 1.4, height=plot_width * 0.62),
    )
    top = 0.0
    for j, (hue_lv, color) in enumerate(zip(hue_order, palette)):
        counts = np.array(
            [
                int(((plot_df[x_col] == x_lv) & (plot_df[by_col] == hue_lv)).sum())
                for x_lv in x_order
            ],
            dtype=float,
        )
        offset = (j - (len(hue_order) - 1) / 2.0) * bar_w
        pct = proportion_bars(
            ax, counts, totals, positions=centres + offset,
            width=bar_w * 0.9, color=color,
            label=prettify_level(hue_lv, by_col),
            annotate=len(x_order) * len(hue_order) <= 8,
        )
        top = max(top, float(np.nanmax(pct)) if np.isfinite(pct).any() else 0.0)

    ax.set_xticks(centres)
    ax.set_xticklabels(level_tick_labels(x_order, totals.astype(int), column=x_col))
    ax.set_xlim(-0.6, len(x_order) - 0.4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"% within {x_label.lower()}")
    ax.set_ylim(0, min(105.0, max(top * 1.25, 15.0)))
    place_legend(ax, title=by_label, outside=True)
    set_titles(
        ax, f"{by_label} within {x_label}",
        n_subtitle(int(len(plot_df)), extra="whiskers are 95% Wilson CIs"),
    )
    path = out_dir / file_stem
    return save_figure(fig, path)


def _plot_bivariate(
    df: pd.DataFrame, x_col: str, by_col: str, out_dir: Path,
    *,
    max_marker_levels: int = 12,
) -> Path | None:
    """One bivariate figure for ``(x_col, by_col)`` → SVG path.

    Plot choice:
    - continuous × continuous → scatter + LOESS trend
    - continuous ↔ categorical → raincloud per level
    - categorical × categorical → within-``x`` percentages with Wilson CIs

    All figures report ``n`` (total and per level). Descriptive only: no
    p-values, and no straight-line fit that would imply an untested linear
    model.
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

    # Continuous × continuous — scatter + non-parametric trend
    if x_cont and by_cont:
        x_label = prettify_label(x_col)
        by_label = prettify_label(by_col)
        fig, ax = plt.subplots(figsize=figure_size(FIG_WIDTH_MEDIUM))
        ax.scatter(
            plot_df[x_col].astype(float), plot_df[by_col].astype(float),
            s=12, alpha=0.45, color=PALETTE["primary"], edgecolors="none", zorder=2,
        )
        curve = lowess_curve(
            plot_df[x_col].astype(float).to_numpy(),
            plot_df[by_col].astype(float).to_numpy(),
        )
        if curve is not None:
            ax.plot(
                curve[0], curve[1], color=PALETTE["accent"], linewidth=1.8,
                solid_capstyle="round", zorder=3, label="LOESS trend",
            )
            place_legend(ax, loc="best")
        ax.set_xlabel(x_label)
        ax.set_ylabel(by_label)
        set_titles(ax, f"{x_label} vs {by_label}", n_subtitle(n_total))
        path = out_dir / file_stem
        return save_figure(fig, path)

    # Continuous ↔ categorical — raincloud per level
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

    # Categorical × categorical — within-x percentages with Wilson CIs
    return _plot_categorical_by_categorical(
        plot_df, x_col, by_col, out_dir, file_stem=file_stem,
    )



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


def _filter_hidden_from_bivariate(
    bivariate_specs: dict[str, list[str]],
    hidden: frozenset[str],
) -> dict[str, list[str]]:
    if not hidden:
        return bivariate_specs
    out: dict[str, list[str]] = {}
    for x_col, partners in bivariate_specs.items():
        if x_col in hidden:
            continue
        kept = [p for p in partners if p not in hidden]
        if kept:
            out[x_col] = kept
    return out


def _filter_hidden_from_trivariate(
    trivariate_specs: dict[tuple[str, str], list[str]],
    hidden: frozenset[str],
) -> dict[tuple[str, str], list[str]]:
    if not hidden:
        return trivariate_specs
    out: dict[tuple[str, str], list[str]] = {}
    for pair, by_cols in trivariate_specs.items():
        if not (isinstance(pair, tuple) and len(pair) == 2):
            continue
        x_col, y_col = pair
        if x_col in hidden or y_col in hidden:
            continue
        kept = [c for c in by_cols if c not in hidden]
        if kept:
            out[pair] = kept
    return out


def run_dda_bivariate(
    df: pd.DataFrame,
    bivariate_specs: dict[str, list[str]],
    *,
    output_root: Path | str = "output",
    max_marker_levels: int = 12,
) -> list[Path]:
    """Plot each ``{x_col: [partner, ...]}`` pair.

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
    for pat in ("*.png", "*.tif", "*.eps", "*.svg"):
        for old in figs_dir.glob(pat):
            old.unlink()

    bivariate_specs = _filter_hidden_from_bivariate(
        bivariate_specs, _load_hidden_parent_columns(output_root),
    )

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
    - continuous × continuous → scatter + one LOESS trend per group
    - continuous × categorical → dodged box with its raw points beside it
    - categorical × categorical → % within-x bars with Wilson CIs, faceted by
      ``by`` on a shared 0–100 axis

    Only one trend line per group is drawn. The earlier version overlaid a
    straight-line fit *and* a LOESS curve per group, which tripled the legend
    and invited the reader to pick whichever fit looked stronger.

    ``science_style`` selects SciencePlots sheets (default ``science`` +
    ``nature`` + ``no-latex``; try ``ieee``). High-cardinality categoricals are
    skipped. Descriptive only — no p-values.
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
    path = out_dir / file_stem
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
            fig, ax = plt.subplots(figsize=figure_size(FIG_WIDTH_MEDIUM))
            for level, color, lab in zip(by_order, palette, level_labs):
                sub = plot_df.loc[plot_df[by_col] == level]
                ax.scatter(
                    sub[x_col].astype(float), sub[y_col].astype(float),
                    s=16, alpha=0.5, color=color, edgecolors="none",
                    label=f"{lab} (n={by_counts[level]})", zorder=2,
                )
            for level, color in zip(by_order, palette):
                sub = plot_df.loc[plot_df[by_col] == level]
                curve = lowess_curve(
                    sub[x_col].astype(float).to_numpy(),
                    sub[y_col].astype(float).to_numpy(),
                    frac=lowess_frac,
                )
                if curve is None:
                    continue
                ax.plot(
                    curve[0], curve[1], color=color, linewidth=2.0,
                    solid_capstyle="round", zorder=3,
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.minorticks_on()
            place_legend(ax, title=by_label, loc="best")
            set_titles(
                ax, f"{x_label} vs {y_label} by {by_label}",
                n_subtitle(n_total, extra="lines are LOESS trends"),
            )
            return save_figure(fig, path)

        if x_cont ^ y_cont:
            cont_col, cat_col = (x_col, y_col) if x_cont else (y_col, x_col)
            cont_label, cat_label = (
                (x_label, y_label) if x_cont else (y_label, x_label)
            )
            cat_order = _ordered_levels(plot_df[cat_col])
            n_cat, n_hue = len(cat_order), len(by_order)
            fig, ax = plt.subplots(
                figsize=figure_size(
                    width_for_levels(n_cat * n_hue, per_level=0.42, base=2.0),
                ),
            )
            slot = 0.8
            group_w = slot / max(n_hue, 1)
            box_w = group_w * 0.42
            centers = np.arange(n_cat, dtype=float)
            rng = deterministic_rng("dda", "trivariate", x_col, y_col, by_col)
            for i, (level, color, lab) in enumerate(zip(by_order, palette, level_labs)):
                offset = (i - (n_hue - 1) / 2.0) * group_w
                positions = centers + offset
                data = [
                    plot_df.loc[
                        (plot_df[cat_col] == cat_lv) & (plot_df[by_col] == level),
                        cont_col,
                    ].astype(float).to_numpy()
                    for cat_lv in cat_order
                ]
                # Raw points sit beside their box, never on top of it: overlaid
                # strips hide the median line and redraw whiskered outliers twice.
                for pos, vals in zip(positions, data):
                    if vals.size == 0:
                        continue
                    jitter = rng.uniform(-box_w * 0.34, box_w * 0.34, size=vals.size)
                    ax.scatter(
                        np.full(vals.size, pos - box_w * 0.72) + jitter, vals,
                        s=7, alpha=0.45, color=color, edgecolors="none", zorder=2,
                    )
                ax.boxplot(
                    data, positions=positions + box_w * 0.28, widths=box_w,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "#1a1a1a", "linewidth": 1.2},
                    whiskerprops={"color": "#4d4d4d", "linewidth": 0.8},
                    capprops={"color": "#4d4d4d", "linewidth": 0.8},
                    boxprops={
                        "facecolor": "white", "edgecolor": color, "linewidth": 1.0,
                    },
                    zorder=3,
                )
                ax.plot(
                    [], [], color=color, linewidth=5, alpha=0.6,
                    label=f"{lab} (n={by_counts[level]})",
                )
            ax.set_xticks(centers)
            ax.set_xticklabels(level_tick_labels(cat_order, column=cat_col))
            ax.set_xlim(-0.6, n_cat - 0.4)
            ax.set_xlabel(cat_label)
            ax.set_ylabel(cont_label)
            ax.minorticks_on()
            place_legend(ax, title=by_label, loc="best")
            set_titles(
                ax, f"{cont_label} by {cat_label} × {by_label}",
                n_subtitle(n_total),
            )
            return save_figure(fig, path)

        # categorical × categorical × by — % within each x level, shared 0–100 y
        x_order = _ordered_levels(plot_df[x_col])
        y_order = _ordered_levels(plot_df[y_col])
        n_panels = len(by_order)
        y_palette = categorical_palette(len(y_order))
        panel_w = width_for_levels(len(x_order) * len(y_order), per_level=0.34, base=1.6,
                                   minimum=2.6, maximum=4.4)
        fig, axes = plt.subplots(
            1, n_panels, sharey=True, squeeze=False,
            figsize=figure_size(
                panel_w * n_panels + 2.2, height=panel_w * 1.15,
            ),
        )
        x_pos = np.arange(len(x_order), dtype=float)
        slot = 0.78
        bar_w = slot / max(len(y_order), 1)
        pct_ylabel = f"% within {x_label.lower()}"
        for panel_idx, (ax, level, lab) in enumerate(zip(axes[0], by_order, level_labs)):
            sub = plot_df.loc[plot_df[by_col] == level]
            totals = np.array(
                [int((sub[x_col] == x_lv).sum()) for x_lv in x_order], dtype=float,
            )
            for j, (y_lv, color) in enumerate(zip(y_order, y_palette)):
                counts = np.array(
                    [
                        int(((sub[x_col] == x_lv) & (sub[y_col] == y_lv)).sum())
                        for x_lv in x_order
                    ],
                    dtype=float,
                )
                offset = (j - (len(y_order) - 1) / 2.0) * bar_w
                proportion_bars(
                    ax, counts, totals, positions=x_pos + offset,
                    width=bar_w * 0.9, color=color,
                    label=(
                        prettify_level(y_lv, y_col)
                        if panel_idx == n_panels - 1 else None
                    ),
                    annotate=False,
                )
            ax.set_title(f"{lab} (n={by_counts[level]})")
            ax.set_xlabel(x_label)
            ax.set_ylim(0, 100)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                level_tick_labels(x_order, totals.astype(int), column=x_col),
            )
            ax.set_xlim(-0.6, len(x_order) - 0.4)
            if panel_idx == 0:
                ax.set_ylabel(pct_ylabel)
            if panel_idx == n_panels - 1:
                place_legend(ax, title=y_label, outside=True)
        fig.suptitle(
            f"{x_label} × {y_label} by {by_label}  (n = {n_total}; "
            "bars are % within each column, whiskers are 95% Wilson CIs)",
            y=1.02,
        )
        return save_figure(fig, path)


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
    - cat × cat   → % within-x bars faceted by by (shared 0–100 y)
    """
    output_root = Path(output_root)
    figs_dir = output_root / "dda" / "figures_trivariate"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("*.png", "*.tif", "*.eps", "*.svg"):
        for old in figs_dir.glob(pat):
            old.unlink()

    trivariate_specs = _filter_hidden_from_trivariate(
        trivariate_specs, _load_hidden_parent_columns(output_root),
    )

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
    hidden = _load_hidden_parent_columns(output_root)

    rows_cont, rows_cat, rows_bin, rows_dt, rows_id = [], [], [], [], []

    for col, spec in schema.items():
        if col not in df.columns or not spec.keep or spec.kind == "skip":
            continue
        if col in hidden:
            continue

        s = df[col]
        if spec.kind in ("continuous", "count"):
            row = {"column": col, "kind": spec.kind, **_stats_continuous(s)}
            rows_cont.append(row)
            _plot_continuous(s, col, figs_dir)

        elif spec.kind in ("ordinal", "nominal"):
            row = {"column": col, "kind": spec.kind,
                   "positive_class": spec.positive_class,
                   **_stats_categorical(s, ordered=(spec.kind == "ordinal"))}
            rows_cat.append(row)
            if spec.kind == "ordinal":
                _plot_ordinal(s, col, figs_dir, ordered_levels=spec.ordered_levels)
            else:
                _plot_nominal(s, col, figs_dir, positive_class=spec.positive_class)

        elif spec.kind == "binary":
            row = {"column": col, "kind": "binary",
                   "positive_class": spec.positive_class,
                   **_stats_binary(s)}
            rows_bin.append(row)
            _plot_binary(s, col, figs_dir, positive_class=spec.positive_class)

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
    CAT_ORDER = ["column", "kind", "ordered", "positive_class", "n", "n_unique", "missing_pct",
                 "first_mode", "first_mode_pct",
                 "second_mode", "second_mode_pct",
                 "rarest", "rarest_pct", "max_class_imbalance",
                 "median_category", "balance", "entropy_bin"]
    BIN_ORDER = ["column", "kind", "ordered", "positive_class", "n", "n_unique", "missing_pct",
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
