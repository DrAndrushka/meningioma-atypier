"""Assemble ``output/report/report.html`` from existing pipeline artifacts.

Reads CSV/SVG under ``output/`` (no refitting). Collapsible major sections;
multivariable uses nested literature / experimental dropdowns per target.
CLI: ``python report.py --output-root output``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import functools
import html as _html
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from inferential import (
    artifact_base,
    model_key,
    parse_artifact_base,
    parse_model_key,
)

from cleaning import (
    collapse_coercion_audit_rows,
    format_number,
    format_table_for_display,
)
from plot_style import prettify_caption, prettify_label
from scales import LOG1P_COLUMNS, scale_footnote

# The multivariable section lists no predictors of its own — the model decides
# which survive VIF pruning — so its note names the whole declared set rather
# than the columns that happen to appear in one variant.
_MULTIVARIABLE_SCALE_NOTE = scale_footnote(sorted(LOG1P_COLUMNS))

from heavy_machinery.config import load

analysis = load("analysis")
published_models = load("published_models")


# ---------------------------------------------------------------------------
# Thresholds & constants (Cohen-style defaults, configurable via dataclass)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectThresholds:
    """Tier cutoffs used to translate raw effect magnitudes into badges.

    ``corr_*`` apply to correlation-like effects (Spearman rho, rank-biserial
    r, Cramer's V). ``or_*`` apply to odds ratios via ``abs(log(OR))`` so the
    same scale is used for protective (OR<1) and risk (OR>1) directions.
    """
    corr_weak: float = 0.10
    corr_moderate: float = 0.30
    corr_strong: float = 0.50
    # OR thresholds expressed on the log scale; the chosen anchors map to
    # OR ~ 1.11 / 1.35 / 1.65 (and their reciprocals 0.90 / 0.74 / 0.61).
    or_weak: float = math.log(1.11)
    or_moderate: float = math.log(1.35)
    or_strong: float = math.log(1.65)


@dataclass(frozen=True)
class MissingThresholds:
    low: float = 5.0
    moderate: float = 20.0
    high: float = 40.0


@dataclass
class ReportConfig:
    output_root: Path
    title: str = "Research Data Analysis Report"
    author: str = ""
    targets: Sequence[str] = field(default_factory=tuple)
    schema_path: Path | None = None
    fdr_alpha: float = 0.05
    nominal_alpha: float = 0.05
    effect: EffectThresholds = field(default_factory=EffectThresholds)
    missing: MissingThresholds = field(default_factory=MissingThresholds)


# ---------------------------------------------------------------------------
# Inline CSS (medical-academic, emoji-friendly, color-coded rows + badges)
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --fg: #1f2937;
    --muted: #6b7280;
    --bg: #ffffff;
    --card: #f9fafb;
    --border: #e5e7eb;
    --accent: #3b7ddd;
    --green: #16a34a;
    --green-bg: #dcfce7;
    --yellow: #ca8a04;
    --yellow-bg: #fef9c3;
    --orange: #ea580c;
    --orange-bg: #ffedd5;
    --red: #dc2626;
    --red-bg: #fee2e2;
    --blue: #2563eb;
    --blue-bg: #dbeafe;
    --grey-bg: #f3f4f6;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--fg); background: var(--bg);
    line-height: 1.55; font-size: 15px;
}
.container { max-width: 1180px; margin: 0 auto; padding: 32px 28px 80px; }

h1 { font-size: 30px; margin: 0 0 4px; }
.report-title-block { text-align: center; margin: 0 0 16px; }
.report-title-block h1 { margin-bottom: 10px; }
.report-authors {
    font-size: 16px; color: #374151; margin: 0;
    font-weight: 500; line-height: 1.5;
}
h2 { font-size: 22px; margin: 38px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid var(--border); }
h3 { font-size: 17px; margin: 22px 0 8px; color: #111827; }
h4 { font-size: 15px; margin: 14px 0 6px; color: var(--muted); font-weight: 600; }
p  { margin: 8px 0 12px; }
small, .muted { color: var(--muted); }
code { background: var(--grey-bg); padding: 1px 5px; border-radius: 4px;
       font-size: 13px; }

.report-section { margin-bottom: 28px; }

/* Header dashboard cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr));
         gap: 12px; margin: 14px 0 4px; }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 14px; }
.card .label { font-size: 12px; color: var(--muted); text-transform: uppercase;
               letter-spacing: 0.04em; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 3px; }

/* Tables */
.table-wrap { width: 100%; overflow-x: auto; margin: 8px 0 14px; }
table.report { border-collapse: collapse; width: 100%; font-size: 13.5px;
               margin: 0; }
table.report th, table.report td { padding: 7px 10px; text-align: left;
                                   border-bottom: 1px solid var(--border);
                                   vertical-align: top; }
table.report thead th { background: var(--grey-bg); position: sticky; top: 0;
                        font-weight: 600; }
table.report tbody tr:hover { background: #fafafa; }
table.report th.nowrap, table.report td.nowrap {
    white-space: nowrap;
}
table.report tr.eda-kind-divider th {
    background: var(--grey-bg);
    color: var(--fg);
    font-weight: 700;
    text-align: left;
    border-top: 2px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding-top: 10px;
    padding-bottom: 8px;
}
table.report tr.eda-kind-divider:first-child th {
    border-top: none;
}
table.report tr.eda-kind-divider:hover,
table.report tr.eda-col-header:hover { background: transparent; }
table.report tr.eda-col-header th {
    background: #f9fafb;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
}
/* Keep EDA paper stack as one continuous table (no card gaps between kinds). */
.eda-paper-stack.table-wrap { margin: 8px 0 10px; }
table.report.diagnostic-accuracy th:not(:first-child),
table.report.diagnostic-accuracy td:not(:first-child) {
    text-align: right;
    white-space: nowrap;
}
table.report.diagnostic-accuracy td:first-child { font-weight: 500; }

/* Row color coding */
tr.sig-fdr      { background: var(--green-bg) !important; }
tr.sig-nominal  { background: var(--yellow-bg) !important; }
tr.sig-none     { background: transparent; }
tr.or-risk      { background: #fde8e8 !important; }
tr.or-protective{ background: #dceaff !important; }
tr.or-neutral   { background: transparent; }
tr.missing-low      { background: var(--green-bg) !important; }
tr.missing-medium   { background: var(--yellow-bg) !important; }
tr.missing-high     { background: var(--orange-bg) !important; }
tr.missing-severe   { background: var(--red-bg) !important; }
tr.schema-skip td   { color: var(--muted); opacity: 0.6; font-style: italic; }

/* Badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 12px; font-weight: 600; }
.badge.effect-strong   { background: var(--green-bg);  color: var(--green); }
.badge.effect-moderate { background: var(--yellow-bg); color: var(--yellow); }
.badge.effect-weak     { background: var(--red-bg);    color: var(--red); }
.badge.effect-none     { background: var(--grey-bg);   color: var(--muted); }
.badge.kind            { background: var(--grey-bg);   color: var(--fg); }
.badge.target          { background: var(--blue-bg);   color: var(--blue); }

/* EPV stability gauge */
.epv-card { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
            border: 1px solid var(--border); border-radius: 10px;
            padding: 12px 16px; margin: 10px 0 18px; background: var(--card); }
.epv-card .epv-value { font-size: 26px; font-weight: 700; line-height: 1; }
.epv-card .epv-key   { font-size: 11px; color: var(--muted); text-transform: uppercase;
                       letter-spacing: 0.05em; margin-top: 2px; }
.epv-gauge { position: relative; flex: 1 1 240px; min-width: 200px; height: 14px;
             border-radius: 999px; background: var(--grey-bg); }
.epv-gauge .epv-fill { position: absolute; left: 0; top: 0; bottom: 0;
                       border-radius: 999px; }
.epv-gauge .epv-thr  { position: absolute; top: -6px; bottom: -6px; width: 2px;
                       background: var(--fg); }
.epv-gauge .epv-thr-label { position: absolute; top: 17px; transform: translateX(-50%);
                       font-size: 10.5px; color: var(--muted); white-space: nowrap; }
.epv-pill { display: inline-block; padding: 3px 11px; border-radius: 999px;
            font-size: 12px; font-weight: 600; white-space: nowrap; }
.epv-pill.stable     { background: var(--green-bg);  color: var(--green); }
.epv-pill.borderline { background: var(--yellow-bg); color: var(--yellow); }
.epv-pill.unstable   { background: var(--red-bg);    color: var(--red); }
.epv-detail { width: 100%; font-size: 12.5px; color: var(--muted); margin-top: 2px; }

/* Warning / info boxes */
.warning-box, .info-box {
    border-left: 4px solid var(--yellow); background: var(--yellow-bg);
    padding: 10px 14px; border-radius: 6px; margin: 12px 0;
    font-size: 14px;
}
.warning-box.severe { border-left-color: var(--red); background: var(--red-bg); }
.info-box { border-left-color: var(--accent); background: #eff6ff; }

/* Lead sentence introducing a subsection */
.lead { font-size: 15px; font-weight: 500; margin: 10px 0; }

/* Small print under a table or card — caveats, not the main point */
.footnote { font-size: 12px; color: var(--muted); margin: 4px 0 10px; }

/* Figure grid — display size only. Submission TIF/PNG files are untouched. */
img { max-width: 100%; height: auto; }
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
               gap: 14px; margin: 12px 0 18px; }
.figure-card { border: 1px solid var(--border); border-radius: 8px;
               padding: 8px; background: #fff; }
.figure-card img, .figure-card object {
    width: 100%; max-width: 36rem; height: auto; display: block; margin: 0 auto;
}
.figure-card .caption { font-size: 12px; color: var(--muted);
                        margin-top: 4px; text-align: center;
                        word-break: break-word; }
.figure-card.eda-heatmap-overview {
    max-width: 100%;
    margin: 8px 0 4px;
}
.figure-card.eda-heatmap-overview img {
    width: 100%;
    max-width: 100%;
    min-width: 0;
}

/* Collapsible details */
details.collapsible { margin: 8px 0 14px; }
details.collapsible > summary {
    cursor: pointer; font-weight: 600; padding: 6px 0;
    color: var(--accent);
}

/* Major report sections (top-level dropdowns) */
details.section-collapsible {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: #fff;
    overflow: hidden;
}
details.section-collapsible > summary {
    list-style: none;
    cursor: pointer;
    font-size: 24px;
    font-weight: 700;
    line-height: 1.25;
    padding: 20px 24px;
    color: var(--fg);
    background: var(--card);
    border-bottom: 2px solid transparent;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
details.section-collapsible > summary::-webkit-details-marker { display: none; }
details.section-collapsible > summary::marker { content: ""; }
details.section-collapsible > summary::after {
    content: "▸";
    flex-shrink: 0;
    font-size: 22px;
    color: var(--muted);
    transition: transform 0.15s ease;
}
details.section-collapsible[open] > summary {
    border-bottom-color: var(--border);
}
details.section-collapsible[open] > summary::after {
    transform: rotate(90deg);
}
details.section-collapsible > .section-body {
    padding: 4px 24px 28px;
}
/* Nested model-group dropdowns inside major sections */
details.section-collapsible .section-body > details.collapsible {
    margin: 12px 0 16px;
    padding: 12px 16px 4px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card);
}
details.section-collapsible .section-body > details.collapsible > summary {
    font-size: 15px;
    padding: 4px 0 8px;
}
/* Bivariate / trivariate key dividers nested under 2️⃣ / 3️⃣ DDA */
details.section-collapsible .section-body > details.collapsible details.collapsible {
    margin: 8px 0 12px;
    padding: 8px 12px 2px;
    border: 1px dashed var(--border);
    border-radius: 6px;
    background: #fafafa;
}
details.section-collapsible .section-body > details.collapsible details.collapsible > summary {
    font-size: 14px;
    font-weight: 600;
    padding: 2px 0 6px;
}
details.section-collapsible > .section-body > h3:first-child,
details.section-collapsible > .section-body > p:first-child {
    margin-top: 12px;
}

/* TL;DR list */
.tldr-list { padding-left: 22px; }
.tldr-list li { margin: 4px 0; }

/* Footer */
.footer { color: var(--muted); font-size: 12px; margin-top: 60px;
          border-top: 1px solid var(--border); padding-top: 12px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_png_src(path: Path) -> str | None:
    """Return a ``data:image/png;base64,...`` URI for embedding in ``<img src>``."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip():
        return None
    encoded = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _png_bytes_img_html(data: bytes, alt: str) -> str:
    """``<img>`` tag from in-memory PNG bytes."""
    if not data:
        return ""
    encoded = base64.standard_b64encode(data).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        f'alt="{_esc(alt)}" loading="lazy"/>'
    )


def _figure_img_html(path: Path) -> str:
    """``<img>`` tag with the PNG inlined as a data URI (self-contained HTML)."""
    src = _embed_png_src(path)
    if src is None:
        return ""
    return f'<img src="{src}" alt="{_esc(prettify_caption(path.stem))}" loading="lazy"/>'


def _esc(x: Any) -> str:
    """HTML-escape an arbitrary value (None / NaN -> empty string)."""
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return _html.escape(str(x))


def _parse_author_list(author: str) -> list[str]:
    """Split a comma-separated author string into individual names."""
    if not author or not author.strip():
        return []
    names: list[str] = []
    for part in author.split(","):
        name = part.strip()
        if name.lower().startswith("and "):
            name = name[4:].strip()
        if name:
            names.append(name)
    return names


def _format_authors(author: str) -> str:
    """Format author names as a comma-separated byline."""
    names = _parse_author_list(author)
    if not names:
        return ""
    return ", ".join(names)


def human_pool_df(val: Any) -> str:
    """Pooled Rubin df for display (large / non-finite → ∞)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not math.isfinite(x) or x >= 9999:
        return "∞"
    if x == int(x):
        return str(int(x))
    return f"{x:.1f}"


def human_p(p: Any) -> str:
    """Format a p-value for display.

    Already-formatted strings (e.g. ``"<0.001"`` from ``cleaning.format_table_for_csv``)
    are passed through unchanged. Numeric values follow the same rule:
    ``p < 0.001`` -> ``"<0.001"``; otherwise 3 decimal places.
    """
    if p is None:
        return ""
    if isinstance(p, str):
        return p
    try:
        v = float(p)
    except (TypeError, ValueError):
        return _esc(p)
    if not math.isfinite(v):
        return ""
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def _coerce_p(p: Any) -> float | None:
    """Best-effort numeric p-value parser; handles ``"<0.001"`` strings."""
    if p is None:
        return None
    if isinstance(p, (int, float)):
        v = float(p)
        return v if math.isfinite(v) else None
    s = str(p).strip()
    if not s:
        return None
    if s.startswith("<"):
        # Treat "<0.001" as 0.0005 (well below alpha); good enough for tiering.
        try:
            return float(s[1:]) / 2
        except ValueError:
            return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _coerce_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def classify_significance(p: Any, p_fdr: Any, *,
                          fdr_alpha: float = 0.05,
                          nominal_alpha: float = 0.05) -> str:
    """Return one of ``"sig-fdr"`` / ``"sig-nominal"`` / ``"sig-none"``."""
    p_num = _coerce_p(p)
    fdr_num = _coerce_p(p_fdr)
    if fdr_num is not None and fdr_num < fdr_alpha:
        return "sig-fdr"
    if p_num is not None and p_num < nominal_alpha:
        return "sig-nominal"
    return "sig-none"


def classify_or_direction(or_val: Any, ci_lo: Any, ci_hi: Any) -> str:
    """Return ``"or-risk"`` / ``"or-protective"`` / ``"or-neutral"``."""
    o = _coerce_float(or_val); lo = _coerce_float(ci_lo); hi = _coerce_float(ci_hi)
    if o is None or lo is None or hi is None:
        return "or-neutral"
    if lo > 1.0 and o > 1.0:
        return "or-risk"
    if hi < 1.0 and o < 1.0:
        return "or-protective"
    return "or-neutral"


def classify_missing(pct: Any, thr: MissingThresholds = MissingThresholds()) -> str:
    v = _coerce_float(pct)
    if v is None:
        return "missing-low"
    if v >= thr.high:     return "missing-severe"
    if v >= thr.moderate: return "missing-high"
    if v >= thr.low:      return "missing-medium"
    return "missing-low"


def warning_box(msg: str, severe: bool = False) -> str:
    cls = "warning-box severe" if severe else "warning-box"
    emoji = "🚨" if severe else "⚠️"
    return f'<div class="{cls}">{emoji} {_esc(msg)}</div>'


def info_box(msg: str) -> str:
    return f'<div class="info-box">ℹ️ {_esc(msg)}</div>'


# Plausibility floors for measured tumour size (spec 5.6): values below these
# are almost certainly unit or transcription errors, not real surgical lesions.
_PLAUSIBILITY_FLOORS: dict[str, tuple[float, str, str]] = {
    "max_diameter_cm": (0.5, "cm", "maximum diameter"),
    "tumor_volume": (0.5, "cm³", "tumour volume"),
}


def data_quality_warnings(dda_continuous: pd.DataFrame | None) -> list[str]:
    """Messages for continuous minima below hard plausibility floors."""
    if dda_continuous is None or dda_continuous.empty:
        return []
    if not {"column", "min"} <= set(dda_continuous.columns):
        return []
    msgs: list[str] = []
    # Handle duplicate column rows by taking the smallest min across all duplicates.
    mins = pd.to_numeric(
        dda_continuous.set_index("column")["min"], errors="coerce"
    ).groupby(level=0).min()
    for col, (floor, unit, label) in _PLAUSIBILITY_FLOORS.items():
        if col not in mins.index:
            continue
        mn = mins.loc[col]
        if pd.notna(mn) and mn < floor:
            msgs.append(
                f"Data-quality note: smallest recorded {label} is {mn:g} {unit} "
                f"— implausibly small; verify the source records before "
                f"publication.")
    return msgs


def table_to_html(df: pd.DataFrame, *, row_class_fn=None,
                  max_rows: int | None = None,
                  index: bool = False,
                  safe_html_cols: Iterable[str] = (),
                  nowrap_cols: Iterable[str] = ()) -> str:
    """Render a DataFrame to HTML with optional per-row CSS class function.

    Parameters
    ----------
    row_class_fn : callable, optional
        Receives the row's Series and returns a CSS class string
        (or ``""``). Applied AFTER any truncation by ``max_rows``.
    safe_html_cols : iterable of column names
        Cells in these columns are emitted verbatim (NOT HTML-escaped).
        Use this for pre-built ``<span class='badge ...'>`` snippets.
        Any column not listed is still escaped — default-safe.
    nowrap_cols : iterable of column names
        Cells in these columns use ``white-space: nowrap`` (no line wrap).
    """
    if df is None or df.empty:
        return '<p class="muted"><em>(empty table)</em></p>'
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    cols = list(df.columns)
    safe_set = set(safe_html_cols)
    nowrap_set = set(nowrap_cols)
    head = "".join(
        f'<th class="nowrap">{_esc(c)}</th>' if c in nowrap_set else f"<th>{_esc(c)}</th>"
        for c in cols
    )
    if index:
        head = f"<th>{_esc(df.index.name or '')}</th>" + head

    body_rows = []
    for idx, row in df.iterrows():
        cls = row_class_fn(row) if row_class_fn else ""
        cls_attr = f' class="{cls}"' if cls else ""
        cells = "".join(
            # Pre-built HTML (badges) passes through verbatim; everything else
            # is escaped to keep the document safe even with weird data.
            (
                f'<td class="nowrap">{row[c] if c in safe_set else _esc(row[c])}</td>'
                if c in nowrap_set
                else f"<td>{row[c] if c in safe_set else _esc(row[c])}</td>"
            )
            for c in cols
        )
        if index:
            cells = f"<td><strong>{_esc(idx)}</strong></td>" + cells
        body_rows.append(f"<tr{cls_attr}>{cells}</tr>")
    body = "".join(body_rows)
    return (f'<div class="table-wrap"><table class="report">'
            f'<thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def svg_grid(svg_paths: Iterable[Path], max_n: int | None = None) -> str:
    """Render an HTML grid of SVG figures embedded as base64 data URIs."""
    paths = [p for p in svg_paths if p.exists()]
    if max_n is not None:
        paths = paths[:max_n]
    if not paths:
        return '<p class="muted"><em>(no figures available)</em></p>'
    cards = []
    for p in paths:
        img = _figure_img_html(p)
        if not img:
            continue
        cards.append(
            f'<div class="figure-card">'
            f'{img}'
            f'<div class="caption">{_esc(prettify_caption(p.stem))}</div>'
            f'</div>'
        )
    if not cards:
        return '<p class="muted"><em>(no figures available)</em></p>'
    return f'<div class="figure-grid">{"".join(cards)}</div>'


def details_block(summary: str, inner_html: str, *, open: bool = False) -> str:
    open_attr = " open" if open else ""
    return (f'<details class="collapsible"{open_attr}>'
            f'<summary>{_esc(summary)}</summary>{inner_html}</details>')


def section_block(title: str, inner_html: str, *, open: bool = False) -> str:
    """Wrap a major report section in a top-level collapsible dropdown."""
    open_attr = " open" if open else ""
    return (
        f'<section class="report-section">'
        f'<details class="section-collapsible"{open_attr}>'
        f'<summary>{_esc(title)}</summary>'
        f'<div class="section-body">{inner_html}</div>'
        f'</details>'
        f'</section>'
    )


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

@dataclass
class Artifacts:
    """All artifacts discovered under ``output_root``.

    Each table is loaded lazily as a DataFrame (or ``None`` if missing).
    Figures are stored as lists of ``Path`` objects so render functions can
    decide which subset to embed.
    """
    output_root: Path

    # Cleaning / schema
    cleaning_summary: pd.DataFrame | None = None
    cleaning_log: pd.DataFrame | None = None
    derivation_log: pd.DataFrame | None = None
    schema_coercion: pd.DataFrame | None = None
    schema_summary: pd.DataFrame | None = None

    # DDA
    dda_overall: pd.DataFrame | None = None
    dda_continuous: pd.DataFrame | None = None
    dda_categorical: pd.DataFrame | None = None
    dda_binary: pd.DataFrame | None = None
    dda_datetime: pd.DataFrame | None = None
    dda_id_text: pd.DataFrame | None = None
    dda_figures: list[Path] = field(default_factory=list)
    dda_bivariate_figures: list[Path] = field(default_factory=list)
    dda_trivariate_figures: list[Path] = field(default_factory=list)
    dda_derived_columns: frozenset[str] = field(default_factory=frozenset)

    # Missingness
    missingness_summary: pd.DataFrame | None = None
    top_missing: pd.DataFrame | None = None
    missingness_figures: list[Path] = field(default_factory=list)
    mice_manifest: dict[str, Any] | None = None

    # EDA
    associations: pd.DataFrame | None = None
    diagnostic_accuracy: pd.DataFrame | None = None
    eda_paper_tables: pd.DataFrame | None = None
    eda_derived_columns: frozenset[str] = field(default_factory=frozenset)
    eda_excluded_columns: frozenset[str] = field(default_factory=frozenset)
    hidden_parent_columns: frozenset[str] = field(default_factory=frozenset)
    hidden_parent_replacements: dict[str, list[str]] = field(
        default_factory=dict)
    known_derived_columns: frozenset[str] = field(default_factory=frozenset)
    derived_sources: dict[str, list[str]] = field(default_factory=dict)
    eda_figures: list[Path] = field(default_factory=list)

    # Inferential
    inferential_summary: pd.DataFrame | None = None
    inferential_cases: pd.DataFrame | None = None
    inferential_multivariable: dict[str, pd.DataFrame] = field(default_factory=dict)
    inferential_vif: dict[str, pd.DataFrame] = field(default_factory=dict)
    inferential_model_titles: dict[str, str] = field(default_factory=dict)
    inferential_model_links: dict[str, str] = field(default_factory=dict)
    inferential_model_experimental: dict[str, bool] = field(default_factory=dict)
    inferential_figures: list[Path] = field(default_factory=list)

    # Marker panel
    panel_marker: pd.DataFrame | None = None
    panel_marker_reading_view: pd.DataFrame | None = None
    panel_figures: list[Path] = field(default_factory=list)

    # Warnings accumulated during load (rendered in appendix)
    warnings: list[str] = field(default_factory=list)


def _maybe_read_csv(p: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Read a CSV if it exists; record (non-fatal) warning otherwise."""
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:  # pragma: no cover - defensive
        warnings.append(f"Failed to read {p.name}: {e}")
        return None


def load_artifacts(cfg: ReportConfig) -> Artifacts:
    """Discover every CSV / SVG under ``cfg.output_root``.

    Missing files are recorded as warnings but never raise. The function is
    pipeline-agnostic: directories that don't exist are simply skipped.
    """
    root = cfg.output_root
    art = Artifacts(output_root=root)

    if not root.exists():
        art.warnings.append(f"output_root '{root}' does not exist.")
        return art

    # Cleaning
    cleaning_dir = root / "cleaning"
    art.cleaning_summary = _maybe_read_csv(cleaning_dir / "cleaning_summary.csv", art.warnings)
    art.cleaning_log     = _maybe_read_csv(cleaning_dir / "cleaning_log.csv", art.warnings)
    art.derivation_log   = _maybe_read_csv(cleaning_dir / "derivation_log.csv", art.warnings)
    art.schema_coercion  = _maybe_read_csv(cleaning_dir / "schema_coercion.csv", art.warnings)
    dda_derived_tbl = _maybe_read_csv(
        cleaning_dir / "dda_derived_columns.csv", art.warnings,
    )
    if dda_derived_tbl is not None and "column" in dda_derived_tbl.columns:
        art.dda_derived_columns = frozenset(
            str(c) for c in dda_derived_tbl["column"].dropna().tolist()
        )
    eda_derived_tbl = _maybe_read_csv(
        cleaning_dir / "eda_derived_columns.csv", art.warnings,
    )
    if eda_derived_tbl is not None and "column" in eda_derived_tbl.columns:
        art.eda_derived_columns = frozenset(
            str(c) for c in eda_derived_tbl["column"].dropna().tolist()
        )
    eda_excl_tbl = _maybe_read_csv(
        cleaning_dir / "eda_excluded_columns.csv", art.warnings,
    )
    if eda_excl_tbl is not None and "column" in eda_excl_tbl.columns:
        art.eda_excluded_columns = frozenset(
            str(c) for c in eda_excl_tbl["column"].dropna().tolist()
        )
    hidden_parent_tbl = _maybe_read_csv(
        cleaning_dir / "hidden_parent_columns.csv", art.warnings,
    )
    if hidden_parent_tbl is not None and "column" in hidden_parent_tbl.columns:
        art.hidden_parent_columns = frozenset(
            str(c) for c in hidden_parent_tbl["column"].dropna().tolist()
        )
    # Which flag replaced which parent, so the table can say so by name rather
    # than leaving the reader to notice that "Sex" is simply absent.
    deriv_log = _maybe_read_csv(cleaning_dir / "derivation_log.csv", art.warnings)
    if deriv_log is not None and {"derivation", "source"} <= set(deriv_log.columns):
        # Every column this pipeline derived, from the log rather than from the
        # EDA list — two sources for one fact, so they can be checked against
        # each other instead of agreeing by construction.
        art.known_derived_columns = frozenset(
            str(d).strip() for d in deriv_log["derivation"].dropna()
        )
        for _, drow in deriv_log.iterrows():
            name = str(drow["derivation"]).strip()
            sources = [s.strip() for s in str(drow.get("source", "")).split(",")
                       if s.strip()]
            # A derivation that lists itself among its sources is repairing that
            # column using the others as context — "edema volume, given whether
            # any edema was recorded" — not restating them. Only a column that
            # is a pure function of another can double-count it, so the
            # self-referential ones carry no sources for this purpose.
            if name in sources:
                art.derived_sources[name] = []
                continue
            art.derived_sources[name] = [s for s in sources if s != name]
        for _, row in deriv_log.iterrows():
            parent = str(row["source"]).strip()
            if parent in art.hidden_parent_columns:
                art.hidden_parent_replacements.setdefault(parent, []).append(
                    str(row["derivation"]).strip()
                )

    # Schema
    if cfg.schema_path and cfg.schema_path.exists():
        art.schema_summary = _load_schema_any(cfg.schema_path, art.warnings)
    else:
        art.schema_summary = _maybe_read_csv(root / "schema" / "schema_summary.csv", art.warnings)

    # DDA
    dda_tab = root / "dda" / "tables"
    art.dda_overall     = _maybe_read_csv(dda_tab / "dda_overall.csv", art.warnings)
    art.dda_continuous  = _maybe_read_csv(dda_tab / "dda_continuous.csv", art.warnings)
    art.dda_categorical = _maybe_read_csv(dda_tab / "dda_categorical.csv", art.warnings)
    art.dda_binary      = _maybe_read_csv(dda_tab / "dda_binary.csv", art.warnings)
    art.dda_datetime    = _maybe_read_csv(dda_tab / "dda_datetime.csv", art.warnings)
    art.dda_id_text     = _maybe_read_csv(dda_tab / "dda_id_text.csv", art.warnings)
    dda_fig = root / "dda" / "figures"
    if dda_fig.exists():
        art.dda_figures = sorted(dda_fig.glob("*.png"))
    dda_biv = root / "dda" / "figures_bivariate"
    if dda_biv.exists():
        art.dda_bivariate_figures = sorted(dda_biv.glob("*.png"))
    dda_tri = root / "dda" / "figures_trivariate"
    if dda_tri.exists():
        art.dda_trivariate_figures = sorted(dda_tri.glob("*.png"))

    # Missingness
    miss_tab = root / "missingness" / "tables"
    art.missingness_summary = _maybe_read_csv(miss_tab / "missingness_summary.csv", art.warnings)
    art.top_missing         = _maybe_read_csv(miss_tab / "top_missing.csv", art.warnings)
    # Fall back to flat layout (older runs)
    if art.missingness_summary is None:
        art.missingness_summary = _maybe_read_csv(root / "missingness" / "missing_per_column.csv", art.warnings)
    miss_fig = root / "missingness" / "figures"
    if miss_fig.exists():
        art.missingness_figures = sorted(miss_fig.glob("*.png"))
    else:
        # Older layout: figures dropped directly in missingness/
        flat = root / "missingness"
        if flat.exists():
            art.missingness_figures = sorted(flat.glob("*.png"))
    # Formal-MICE manifest (engine + R/package versions, m, max_iter, seed).
    mice_manifest_path = root / "missingness" / "mice" / "manifest.json"
    if mice_manifest_path.exists():
        try:
            art.mice_manifest = json.loads(mice_manifest_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - defensive
            art.warnings.append(f"Failed to read MICE manifest: {e}")

    # EDA
    art.associations = _maybe_read_csv(root / "eda" / "tables" / "associations.csv", art.warnings)
    art.diagnostic_accuracy = _maybe_read_csv(
        root / "eda" / "tables" / "diagnostic_accuracy.csv", art.warnings,
    )
    art.eda_paper_tables = _maybe_read_csv(
        root / "eda" / "tables" / "eda_paper_tables.csv", art.warnings,
    )
    eda_fig = root / "eda" / "figures"
    if eda_fig.exists():
        art.eda_figures = sorted(eda_fig.glob("*.png"))

    # Inferential
    inf_tab = root / "inferential" / "tables"
    art.inferential_summary = _maybe_read_csv(inf_tab / "inferential_summary.csv", art.warnings)
    art.inferential_cases = _maybe_read_csv(inf_tab / "multivariable_cases.csv", art.warnings)
    known_targets: set[str] = set()
    if art.inferential_cases is not None and not art.inferential_cases.empty:
        known_targets = set(art.inferential_cases["target"].astype(str))
    if art.inferential_summary is not None and not art.inferential_summary.empty:
        known_targets |= set(art.inferential_summary["target"].astype(str))
    if inf_tab.exists():
        if art.inferential_cases is not None and not art.inferential_cases.empty:
            for _, row in art.inferential_cases.iterrows():
                t = str(row["target"])
                mid = str(row.get("model_id", "") or "")
                title = str(row.get("model_title", "") or "")
                link = str(row.get("model_link", "") or "")
                key = model_key(t, mid)
                if title:
                    art.inferential_model_titles[key] = title
                if link:
                    art.inferential_model_links[key] = link
                if "experimental" in row and pd.notna(row["experimental"]):
                    art.inferential_model_experimental[key] = _parse_bool(row["experimental"])
        for f in sorted(inf_tab.glob("*__multivariable.csv")):
            base = f.stem.replace("__multivariable", "")
            target, model_id = parse_artifact_base(base, known_targets)
            key = model_key(target, model_id)
            art.inferential_multivariable[key] = pd.read_csv(f)
        for f in sorted(inf_tab.glob("*__vif.csv")):
            base = f.stem.replace("__vif", "")
            target, model_id = parse_artifact_base(base, known_targets)
            key = model_key(target, model_id)
            art.inferential_vif[key] = pd.read_csv(f)
    inf_fig = root / "inferential" / "figures"
    if inf_fig.exists():
        art.inferential_figures = sorted(inf_fig.glob("*.png"))

    # Marker panel
    panel_tab = root / "panel" / "tables"
    art.panel_marker = _maybe_read_csv(panel_tab / "01_marker_panel.csv", art.warnings)
    art.panel_marker_reading_view = _maybe_read_csv(
        panel_tab / "02_marker_panel_reading_view.csv", art.warnings,
    )
    panel_fig = root / "panel" / "figures"
    if panel_fig.exists():
        art.panel_figures = sorted(panel_fig.glob("*.png"))

    return art


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "t"}


def _legacy_experimental_from_model_id(model_id: str) -> bool:
    """Fallback for older runs whose ``multivariable_cases.csv`` lacks ``experimental``."""
    mid = str(model_id or "")
    if mid in {"experimental", "experimental_model"}:
        return True
    return mid.startswith("experimental_")


def _inferential_model_is_experimental(
    model_key_str: str,
    experimental_flags: dict[str, bool],
) -> bool:
    if model_key_str in experimental_flags:
        return experimental_flags[model_key_str]
    _, model_id = parse_model_key(model_key_str)
    return _legacy_experimental_from_model_id(model_id)


def _inferential_model_sort_key(
    model_key_str: str,
    experimental_flags: dict[str, bool],
) -> tuple[int, str]:
    """Literature variants first; experimental always last within a target."""
    return (
        1 if _inferential_model_is_experimental(model_key_str, experimental_flags) else 0,
        model_key_str,
    )


def _sort_inferential_model_keys(
    keys: list[str],
    experimental_flags: dict[str, bool],
) -> list[str]:
    return sorted(keys, key=lambda k: _inferential_model_sort_key(k, experimental_flags))


def _load_schema_any(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    """Accept JSON (dict of ColSpec-like) or CSV with one row per column."""
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for name, spec in data.items():
                if isinstance(spec, dict):
                    rows.append({"column": name, **spec})
                else:
                    rows.append({"column": name, "kind": str(spec)})
            return pd.DataFrame(rows)
        return pd.read_csv(path)
    except Exception as e:
        warnings.append(f"Failed to load schema from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _usable_link(value) -> str:
    """A URL, or "" — because a blank CSV cell does not arrive as a blank string.

    ``pandas`` reads an empty cell as NaN and ``str(NaN)`` is ``"nan"``, which
    is a perfectly good truthy string. The two experimental models have no
    published source, so they rendered ``<a href="nan">source</a>``: a dead
    relative URL under a word that promises a citation.
    """
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "<na>", "nat") else text


def _inferential_model_heading(
    title: str = "",
    *,
    link: str = "",
    model_id: str = "",
) -> str:
    """Section heading for one multivariable model variant."""
    link = _usable_link(link)
    if title and link:
        return (
            f'<h4>📐 {_esc(title)} · '
            f'<a href="{_esc(link)}" target="_blank" rel="noopener noreferrer">source</a></h4>'
        )
    if title:
        return f"<h4>📐 {_esc(title)}</h4>"
    if model_id:
        return f"<h4>📐 Model <code>{_esc(model_id)}</code></h4>"
    return ""


def _published_or(term: dict) -> str:
    """``2.94 (1.15–7.48)``, or as much of it as the paper printed."""
    o, lo, hi = (_coerce_float(term.get(k)) for k in ("or", "ci_lo", "ci_hi"))
    if o is None:
        return ""
    if lo is None or hi is None:
        return f"{o:.2f}"
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


def _published_model_block(model_id: str) -> str:
    """The source paper's own multivariable model, above the one we fitted.

    Reproducing a published model is only meaningful if the reader can see what
    was published. Without this the report shows our odds ratios for predictors
    the paper chose, with no way to tell whether we agreed with it.
    """
    published = published_models.published_model(model_id)
    if not published:
        return ""
    terms = published.get("terms") or []
    if not terms:
        return ""

    table = table_to_html(pd.DataFrame([
        {
            "Variable in the paper": t.get("variable", ""),
            "What it means": t.get("meaning", ""),
            "Published aOR (95% CI)": _published_or(t),
            "p": human_p(t.get("p")),
            "Column used here": t.get("column", ""),
        }
        for t in terms
    ]), nowrap_cols=("Published aOR (95% CI)", "p"))

    notes: list[str] = []
    for key, label in (("citation", ""), ("outcome", "Outcome"), ("cohort", "Source cohort")):
        val = str(published.get(key) or "").strip()
        if not val:
            continue
        notes.append(_esc(val) if not label else f"<strong>{label}:</strong> {_esc(val)}")
    perf = str(published.get("performance") or "").strip()
    notes.append(
        f"<strong>Reported performance:</strong> {_esc(perf)}" if perf
        else "<strong>Reported performance:</strong> none — the paper publishes no "
             "AUC or c-statistic for this model, so there is nothing to compare our "
             "validated AUC against."
    )
    note_html = "".join(f'<p class="muted">{n}</p>' for n in notes)

    return details_block(
        "📖 The published model",
        "<p>What the source paper actually fitted, quoted as printed. Odds ratios "
        "below are theirs, not ours — read them against the forest plot further "
        "down, which is the same model refitted on this cohort.</p>"
        + table + note_html,
        open=True,
    )


# Per-variant performance figures, in the order a reader should meet them:
# can it rank patients → is the number right → is acting on it worthwhile.
_PERFORMANCE_FIGURE_ORDER = ("roc", "calibration", "decision_curve")


def _render_model_performance(stem: str, art: Artifacts) -> str:
    """Collapsible ROC / calibration / decision-curve block for one variant."""
    figs = [
        p
        for suffix in _PERFORMANCE_FIGURE_ORDER
        for p in art.inferential_figures
        if p.stem == f"{stem}__{suffix}"
    ]
    if not figs:
        return ""
    return details_block(
        "📈 Model performance",
        '<p>Bootstrap internal validation on the development sample. '
        '<strong>ROC</strong> shows whether the model ranks patients correctly; '
        '<strong>calibration</strong> whether a predicted risk matches the '
        'observed rate; the <strong>decision curve</strong> whether acting on '
        'the model beats treating everyone or no one. All three are apparent '
        '(in-sample) and therefore optimistic — the optimism-corrected '
        'statistics are quoted on each figure.</p>'
        + svg_grid(figs),
    )


def _render_model_comparison(target: str, art: Artifacts) -> str:
    """Across-variant comparison figure shown once, above the individual models."""
    fig = next(
        (p for p in art.inferential_figures
         if p.stem == f"{target}__model_comparison"),
        None,
    )
    if fig is None:
        return ""
    return (
        '<p>Every variant on the same three axes. The gap between the hollow '
        '(apparent) and filled (optimism-corrected) markers is how much of each '
        "model's performance was overfitting.</p>"
        '<div class="figure-card">'
        + _figure_img_html(fig)
        + f'<div class="caption">{_esc(prettify_caption(fig.stem))}</div>'
        + "</div>"
    )


def _inferential_target_meta(art: Artifacts, target: str, *, model_key_name: str) -> str:
    """One-line EPV / sample-size summary for a multivariable target × model."""
    if art.inferential_cases is None or art.inferential_cases.empty:
        return ""
    if "target" not in art.inferential_cases.columns:
        return ""
    sub = art.inferential_cases[art.inferential_cases["target"].astype(str) == target]
    _, model_id = parse_model_key(model_key_name)
    if "model_id" in sub.columns:
        sub = sub.loc[sub["model_id"].astype(str).fillna("") == (model_id or "")]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    epv = _coerce_float(row.get("epv"))
    if epv is None:
        return ""
    events = _to_int_or_none(row.get("n_outcome_events"))
    params = _to_int_or_none(row.get("n_design_columns"))
    n = _to_int_or_none(row.get("n_complete_cases"))
    return _epv_gauge_html(epv, events, params, n)


# Events-per-variable interpretation thresholds.
_EPV_STABLE = 10.0      # >= 10: adequately powered (Peduzzi 1996 rule of thumb)
_EPV_BORDERLINE = 5.0   # 5-10: usable but interpret with caution


def _epv_gauge_html(
    epv: float,
    events: int | None,
    params: int | None,
    n: int | None,
) -> str:
    """Visual EPV stability gauge with a fixed threshold marker at EPV = 10."""
    if epv >= _EPV_STABLE:
        pill_cls, pill_txt, color = "stable", "Stable model", "var(--green)"
    elif epv >= _EPV_BORDERLINE:
        pill_cls, pill_txt, color = "borderline", "Borderline power", "var(--yellow)"
    else:
        pill_cls, pill_txt, color = "unstable", "Underpowered", "var(--red)"

    # Scale so the EPV = 10 threshold and the actual value both stay on-bar.
    scale = max(20.0, epv * 1.15)
    fill_pct = max(0.0, min(epv / scale, 1.0)) * 100
    thr_pct = _EPV_STABLE / scale * 100
    epv_disp = int(epv) if epv == int(epv) else round(epv, 1)

    detail = []
    if events is not None and params is not None:
        detail.append(f"{events} outcome events / {params} model parameters")
    if n is not None:
        detail.append(f"N = {n} complete cases")
    detail.append("threshold for a stable model: EPV \u2265 10")
    detail_html = " &middot; ".join(_esc(d) for d in detail)

    return (
        '<div class="epv-card">'
        f'<div><div class="epv-value" style="color:{color}">{epv_disp}</div>'
        '<div class="epv-key">Events / variable</div></div>'
        '<div class="epv-gauge">'
        f'<div class="epv-fill" style="width:{fill_pct:.1f}%;background:{color}"></div>'
        f'<div class="epv-thr" style="left:{thr_pct:.1f}%"></div>'
        f'<div class="epv-thr-label" style="left:{thr_pct:.1f}%">EPV 10</div>'
        '</div>'
        f'<span class="epv-pill {pill_cls}">{pill_txt}</span>'
        f'<div class="epv-detail">{detail_html}</div>'
        '</div>'
    )


def render_header(cfg: ReportConfig, art: Artifacts) -> str:
    """🧾 Top-of-report dashboard with headline counts."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Dataset shape from DDA overall if present (coerce to int for display —
    # the CSV may store these as floats due to pandas type inference).
    n_rows = n_cols = None
    if art.dda_overall is not None and not art.dda_overall.empty:
        row = art.dda_overall.iloc[0]
        n_rows = _to_int_or_none(row.get("n_rows"))
        n_cols = _to_int_or_none(row.get("n_cols"))

    n_preds_screened = (len(art.associations.drop_duplicates("predictor"))
                        if art.associations is not None and not art.associations.empty
                        and "predictor" in art.associations.columns else 0)
    n_tests = (len(art.associations) if art.associations is not None else 0)
    n_models = len(art.inferential_multivariable)

    # Stages completed (presence-based)
    stages = []
    if art.cleaning_summary is not None: stages.append("cleaning")
    if art.schema_summary is not None:   stages.append("schema")
    if any(t is not None for t in [art.dda_continuous, art.dda_categorical,
                                    art.dda_binary, art.dda_datetime]): stages.append("DDA")
    if art.missingness_summary is not None: stages.append("missingness")
    if art.associations is not None:        stages.append("EDA")
    if n_models > 0:                        stages.append("inferential")

    def card(label: str, value: Any) -> str:
        return (f'<div class="card"><div class="label">{_esc(label)}</div>'
                f'<div class="value">{_esc(value)}</div></div>')

    cards = [
        card("Generated", now),
        card("Rows", n_rows if n_rows is not None else "—"),
        card("Columns", n_cols if n_cols is not None else "—"),
        card("Targets", len(cfg.targets) or "—"),
        card("Predictors screened", n_preds_screened or "—"),
        card("EDA tests", n_tests or "—"),
        card("Inferential models", n_models or "—"),
        card("Bootstrap model validation", f"{analysis.BOOTSTRAP_RESAMPLES} resamples"),
    ]
    targets_html = ", ".join(f"<span class='badge target'>🎯 {_esc(t)}</span>"
                             for t in cfg.targets) or "<em>(none specified)</em>"
    stages_html = ", ".join(f"<code>{_esc(s)}</code>" for s in stages) or "<em>(none detected)</em>"

    blurb = ("This report summarizes automated data cleaning, schema profiling, "
             "descriptive data analysis, missingness assessment, exploratory "
             "association screening, and multivariable modelling.")

    authors_html = _format_authors(cfg.author)
    title_block = (
        f'<div class="report-title-block">'
        f'<h1>🧾 {_esc(cfg.title)}</h1>'
    )
    if authors_html:
        title_block += f'<p class="report-authors">{_esc(authors_html)}</p>'
    title_block += '</div>'

    return (
        f'<section class="report-section">'
        f'{title_block}'
        f'<p class="muted">{blurb}</p>'
        f'<div class="cards">{"".join(cards)}</div>'
        f'<p><strong>Targets:</strong> {targets_html}</p>'
        f'<p><strong>Stages detected:</strong> {stages_html}</p>'
        f'</section>'
    )


def _fmt_count(v: Any) -> Any:
    """Show whole-number counts as ints (397.0 -> 397); leave blanks/text alone."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    f = _coerce_float(v)
    if f is None:
        return v
    return int(f) if f == int(f) else f


def _format_count_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_fmt_count)
    return df


def _cleaning_provenance(art: Artifacts) -> str:
    """Provenance strip: dataset shape at each hand-off point."""
    summary = art.cleaning_summary
    if summary is None or summary.empty or "step" not in summary.columns:
        return ""

    def _step(name: str):
        hit = summary[summary["step"] == name]
        return hit.iloc[0] if not hit.empty else None

    def _shape(row) -> str:
        if row is None:
            return "—"
        n_rows, n_cols = _fmt_count(row.get("n_rows")), _fmt_count(row.get("n_columns"))
        if n_rows == "" and n_cols == "":
            return "—"
        return f"{n_rows} rows × {n_cols} columns"

    schema_row = _step("apply_schema")
    items = [
        ("Raw export", _shape(_step("raw_data"))),
        ("After schema", _shape(schema_row)),
        ("Analysed cohort", _shape(_step("final"))),
        ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    detail = "" if schema_row is None else str(schema_row.get("detail", "") or "").strip()
    if detail:
        items.insert(2, ("Schema step", detail))
    return table_to_html(pd.DataFrame(items, columns=["Item", "Value"]))


_COHORT_COLUMNS = ["#", "Criterion", "Rule", "n before", "n excluded", "n remaining"]


def _cohort_flow_table(art: Artifacts) -> str:
    """Inclusion/exclusion criteria in applied order, led by the duplicate audit."""
    summary = art.cleaning_summary
    has_summary = summary is not None and "step" in summary.columns

    def _step(name: str):
        if not has_summary:
            return None
        hit = summary[summary["step"] == name]
        return hit.iloc[0] if not hit.empty else None

    rows: list[dict] = []
    dup = _step("duplicate_audit")
    if dup is not None:
        n = _fmt_count(dup.get("n_rows"))
        rows.append({
            "#": "—", "Criterion": "Duplicate ID audit",
            "Rule": str(dup.get("detail", "") or ""),
            "n before": n, "n excluded": 0, "n remaining": n,
        })

    # n_before/n_dropped are only kept in the cleaning log — the summary carries
    # n_remaining alone — so read the log first and fall back when it is absent.
    drops: list[dict] = []
    log = art.cleaning_log
    if log is not None and "step" in log.columns:
        drops = [
            {"name": r.get("reason"), "rule": r.get("criterion"),
             "before": r.get("n_before"), "excluded": r.get("n_dropped"),
             "remaining": r.get("n_remaining")}
            for _, r in log[log["step"] == "drop_rows"].iterrows()
        ]
    if not drops and has_summary:
        drops = [
            {"name": r.get("detail"), "rule": r.get("criterion"),
             "before": None, "excluded": None, "remaining": r.get("n_rows")}
            for _, r in summary[summary["step"] == "drop_rows"].iterrows()
        ]

    for i, d in enumerate(drops, start=1):
        rows.append({
            "#": i,
            "Criterion": str(d["name"] or ""),
            "Rule": str(d["rule"] or ""),
            "n before": _fmt_count(d["before"]),
            "n excluded": _fmt_count(d["excluded"]),
            "n remaining": _fmt_count(d["remaining"]),
        })

    if not rows:
        return ""

    raw, final = _step("raw_data"), _step("final")
    n_raw, n_final = _to_int_or_none(
        None if raw is None else raw.get("n_rows")), _to_int_or_none(
        None if final is None else final.get("n_rows"))
    if n_raw is not None and n_final is not None:
        rows.append({
            "#": "", "Criterion": "Analysed cohort", "Rule": "",
            "n before": n_raw, "n excluded": n_raw - n_final, "n remaining": n_final,
        })
    return table_to_html(pd.DataFrame(rows, columns=_COHORT_COLUMNS))


# (log column, report heading) for the derived / recoded variable tables.
_DERIVED_COLUMNS = [
    ("derivation", "New column"), ("source", "Derived from"), ("rule", "Rule"),
    ("kind", "Kind"), ("rows_missing", "n missing"), ("reason", "Source"),
]
_RECODED_COLUMNS = [
    ("derivation", "Column"), ("rule", "Rule"), ("kind", "Kind"),
    ("rows_nonmissing", "n non-missing"), ("rows_missing", "n missing"),
]


def _derived_tables(log: pd.DataFrame | None) -> str:
    """New columns (``added ColSpec``), then in-place recodes (``updated ColSpec``)."""
    if log is None or log.empty or "derivation" not in log.columns:
        return ""
    action = log.get("schema_action", pd.Series("", index=log.index)).astype(str)

    def _project(sub: pd.DataFrame, spec: list[tuple[str, str]]) -> pd.DataFrame:
        pairs = [(c, label) for c, label in spec if c in sub.columns]
        out = sub[[c for c, _ in pairs]].copy()
        out.columns = [label for _, label in pairs]
        return _format_count_cols(out, ["n missing", "n non-missing"])

    parts: list[str] = []
    added = log[action.str.startswith("added ColSpec")]
    if not added.empty:
        parts.append("<h3>Derived variables</h3>")
        parts.append("<p class='muted'>Columns computed from the cleaned cohort. "
                     "<em>Rule</em> is the definition; <em>Source</em> cites the "
                     "study a cutoff was taken from.</p>")
        parts.append(table_to_html(_project(added, _DERIVED_COLUMNS), max_rows=200))

    updated = log[action.str.startswith("updated ColSpec")]
    if not updated.empty:
        parts.append("<h3>Recoded variables</h3>")
        parts.append("<p class='muted'>Existing columns rewritten in place "
                     "(e.g. structural zeros); no new column is created.</p>")
        parts.append(table_to_html(_project(updated, _RECODED_COLUMNS), max_rows=200))
    return "".join(parts)


def render_cleaning(cfg: ReportConfig, art: Artifacts) -> str:
    """🧹 Cleaning story."""
    blurb = ("The dataset was cleaned using a schema-driven process: declared "
             "null markers were applied, replacements were performed, data "
             "types were coerced, and skipped variables were excluded where "
             "appropriate.")
    body = [f'<p>{blurb}</p>']

    has_coercion = art.schema_coercion is not None and not art.schema_coercion.empty
    if (art.cleaning_summary is None and art.cleaning_log is None
            and art.derivation_log is None and not has_coercion):
        body.append(warning_box(
            "No saved cleaning summary was found. Cleaning may have been "
            "performed, but no cleaning audit table was exported."))
        return section_block("🧹 Cleaning story", "".join(body))

    provenance = _cleaning_provenance(art)
    if provenance:
        body.append("<h3>Provenance</h3>")
        body.append(provenance)

    cohort = _cohort_flow_table(art)
    if cohort:
        body.append("<h3>Inclusion / exclusion criteria</h3>")
        body.append("<p class='muted'>Criteria applied in the order shown; each "
                    "exclusion count is conditional on the criteria above it.</p>")
        body.append(cohort)

    body.append(_derived_tables(art.derivation_log))

    if has_coercion:
        coer = collapse_coercion_audit_rows(art.schema_coercion.copy())
        if "n" in coer.columns:
            coer = _format_count_cols(coer, ["n", "n_after"] if "n_after" in coer.columns else ["n"])
        n_rows = len(coer)
        body.append(details_block(
            f"Coerced value audit ({n_rows})",
            "<p>Value-level changes from schema application "
            "(<code>replace</code>, <code>nulls</code>, dtype coercion). "
            "Missing results show as <code>(missing)</code>. "
            "Datetime columns are summarized by format style "
            "(e.g. <code>DD.MM.YYYY.</code> → <code>YYYY-MM-DD 00:00:00</code>), "
            "not one row per timestamp. "
            "ID columns keep only losses to <code>(missing)</code>; "
            "other id string coercions are folded into one "
            "<code>(various)</code> → <code>(string)</code> row. "
            "Numeric format-only changes are grouped by style "
            "(e.g. <code>leading-zero integer (e.g. 01)</code> → "
            "<code>integer (e.g. 1)</code>, "
            "<code>trailing-zero decimal (e.g. 1.10)</code> → "
            "<code>dot decimal (e.g. 1.1)</code>); "
            "losses to <code>(missing)</code> stay listed. "
            "<code>n_after</code> is a running non-null count within each column "
            "(starts at pre-coercion non-nulls; only drops on losses to "
            "<code>(missing)</code>).</p>"
            + table_to_html(coer, max_rows=500),
        ))

    return section_block("🧹 Cleaning story", "".join(body))


_DDA_KIND_DISPLAY = {
    "nominal": "Nominal",
    "ordinal": "Ordinal",
}


def _dda_kind_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalize Nominal/Ordinal in the kind column for report tables only."""
    if "kind" not in df.columns:
        return df
    out = df.copy()
    out["kind"] = out["kind"].map(
        lambda k: _DDA_KIND_DISPLAY.get(str(k), k)
    )
    return out


def _dda_continuous_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """Display copy of continuous DDA stats (max 2 decimal places)."""
    out = format_table_for_display(df)
    # ``mode`` is numeric here but classified as ``default`` (3 dp) in cleaning.
    if "mode" in df.columns and pd.api.types.is_numeric_dtype(df["mode"]):
        out["mode"] = df["mode"].map(
            lambda v: format_number(v, "central") if pd.notna(v) else ""
        )
    return out


def _group_dda_bivariate_figures(
    paths: list[Path],
) -> dict[str, list[Path]]:
    """Group ``{x}__by__{partner}.svg`` paths by bivariate dict key ``x``."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        stem = p.stem
        if "__by__" in stem:
            x_key = stem.split("__by__", 1)[0]
        else:
            x_key = stem
        groups.setdefault(x_key, []).append(p)
    return {k: groups[k] for k in sorted(groups)}


def _group_dda_trivariate_figures(
    paths: list[Path],
) -> dict[str, list[Path]]:
    """Group ``{x}__vs__{y}__by__{g}.svg`` paths by pair key ``{x}__vs__{y}``."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        stem = p.stem
        if "__by__" in stem:
            pair_key = stem.split("__by__", 1)[0]
        else:
            pair_key = stem
        groups.setdefault(pair_key, []).append(p)
    return {k: groups[k] for k in sorted(groups)}


def _dda_stem_uses_hidden(stem: str, hidden: frozenset[str]) -> bool:
    """True if a DDA figure stem references any hidden-parent column name."""
    parts = stem.replace("__by__", "__").replace("__vs__", "__").split("__")
    return any(part in hidden for part in parts)


def _dda_native_derived_tables(
    tbl: pd.DataFrame,
    derived_cols: frozenset[str],
) -> str:
    """Nested Native / Derived collapsibles for one DDA datatype table."""
    if "column" not in tbl.columns or not derived_cols:
        return table_to_html(tbl)

    is_derived = tbl["column"].astype(str).isin(derived_cols)
    if not bool(is_derived.any()):
        return table_to_html(tbl)

    native = tbl.loc[~is_derived]
    derived = tbl.loc[is_derived]
    parts: list[str] = []
    if not native.empty:
        parts.append(details_block(
            f"🌱 Native ({len(native)})",
            table_to_html(native),
        ))
    if not derived.empty:
        parts.append(details_block(
            f"🧩 Derived ({len(derived)})",
            table_to_html(derived),
        ))
    return "".join(parts) if parts else table_to_html(tbl)


def render_dda(cfg: ReportConfig, art: Artifacts) -> str:
    """📊 DDA story: univariate tables/figures, then bi-/trivariate plots."""
    body = [
        '<p>Descriptive pass before association testing — no p-values appear '
        'in this section. <strong>Univariate</strong> tables and figures '
        'summarize each column on its own; <strong>bivariate</strong> plots '
        'show selected pairs (grouped by the dict key / x column); '
        '<strong>trivariate</strong> plots compare selected pairs across '
        'prespecified groups. Percentages carry 95% Wilson confidence '
        'intervals and their denominators; trends are LOESS smoothers, not '
        'fitted models.</p>',
    ]

    # --- 1️⃣ Univariate ---
    uni: list[str] = [
        '<p>Tables describe distribution shape and balance; figures show the '
        'same information visually. Continuous columns get a histogram with '
        'an aligned box plot and raw observations; categorical columns get '
        'percentages with Wilson intervals and <code>k/n</code>.</p>',
    ]

    sections = [
        ("📏 Continuous / count variables",
         "Summarized using median, mean, trimmed mean, spread, skewness, "
         "kurtosis, outlier-sensitive quantiles, and missingness.",
         art.dda_continuous),
        ("🏷️ Categorical / ordinal variables",
         "Summarized using dominant class, rarest class, class imbalance, "
         "Shannon entropy, and normalized balance.",
         art.dda_categorical),
        ("✅ Binary variables",
         "Same schema as categorical: dominant class, balance, missingness.",
         art.dda_binary),
        ("🕒 Datetime variables",
         "Range, span in days, and missingness.",
         art.dda_datetime),
        ("🪪 ID / text variables",
         "Listed for completeness; excluded from statistical screening.",
         art.dda_id_text),
    ]
    derived_cols = art.dda_derived_columns
    hidden_parents = art.hidden_parent_columns
    for heading, blurb, tbl in sections:
        if tbl is not None and not tbl.empty and hidden_parents and "column" in tbl.columns:
            tbl = tbl[~tbl["column"].astype(str).isin(hidden_parents)].copy()
        if tbl is None or tbl.empty:
            inner = (
                f"<p>{blurb}</p>"
                '<p class="muted"><em>(no variables of this kind)</em></p>'
            )
            uni.append(details_block(heading, inner))
            continue

        display_tbl = (
            _dda_continuous_for_report(tbl)
            if tbl is art.dda_continuous
            else _dda_kind_for_display(tbl)
        )
        n = len(display_tbl)
        parts = [f"<p>{blurb}</p>", _dda_native_derived_tables(display_tbl, derived_cols)]
        if tbl is art.dda_continuous:
            for msg in data_quality_warnings(art.dda_continuous):
                parts.append(warning_box(msg))
        uni.append(details_block(f"{heading} ({n})", "".join(parts)))

    if art.dda_figures:
        figs = art.dda_figures
        if hidden_parents:
            figs = [
                p for p in figs
                if p.stem.split("__", 1)[0] not in hidden_parents
            ]
        uni.append(details_block(
            f"🖼️ DDA figures ({len(figs)})",
            svg_grid(figs),
        ))

    body.append(details_block(
        "1️⃣ DDA - univariate",
        "".join(uni),
    ))

    # --- 2️⃣ Bivariate (nested dropdown per dict key / x column) ---
    if art.dda_bivariate_figures:
        biv_figs = art.dda_bivariate_figures
        if hidden_parents:
            biv_figs = [
                p for p in biv_figs
                if not _dda_stem_uses_hidden(p.stem, hidden_parents)
            ]
        groups = _group_dda_bivariate_figures(biv_figs)
        biv_parts = [
            '<p>One figure per pair from '
            '<code>{x_col: [partner, …]}</code>. '
            'Continuous×continuous: scatter with a LOESS trend; '
            'continuous×categorical: raincloud (density + box + raw points) '
            'per level; categorical×categorical: percentages within each '
            'x level with Wilson intervals. '
            'Open a key below to browse that x column’s plots.</p>',
        ]
        for x_key, figs in groups.items():
            label = prettify_label(x_key)
            biv_parts.append(details_block(
                f"🔑 {label} ({len(figs)})",
                svg_grid(figs),
            ))
        biv_inner = "".join(biv_parts)
    else:
        biv_inner = (
            '<p class="muted"><em>(no bivariate figures — run '
            '<code>run_dda_bivariate</code> with a '
            '<code>{x_col: [partner, …]}</code> dict)</em></p>'
        )
    body.append(details_block(
        f"2️⃣ DDA - bivariate ({len(biv_figs) if art.dda_bivariate_figures else 0})",
        biv_inner,
    ))

    # --- 3️⃣ Trivariate (nested dropdown per (x, y) pair) ---
    if art.dda_trivariate_figures:
        tri_figs = art.dda_trivariate_figures
        if hidden_parents:
            tri_figs = [
                p for p in tri_figs
                if not _dda_stem_uses_hidden(p.stem, hidden_parents)
            ]
        groups = _group_dda_trivariate_figures(tri_figs)
        tri_parts = [
            '<p>One figure per triple from '
            '<code>{(x, y): [group, …]}</code> (SciencePlots). '
            'Continuous×continuous: scatter with one LOESS trend per group; '
            'continuous×categorical: dodged box with its raw points alongside; '
            'categorical×categorical: within-x percentages with Wilson '
            'intervals, panelled by group on a shared 0–100 axis. '
            'Open a pair below to browse its group plots.</p>',
        ]
        for pair_key, figs in groups.items():
            if "__vs__" in pair_key:
                x_raw, y_raw = pair_key.split("__vs__", 1)
                label = f"{prettify_label(x_raw)} vs {prettify_label(y_raw)}"
            else:
                label = prettify_label(pair_key)
            tri_parts.append(details_block(
                f"🔑 {label} ({len(figs)})",
                svg_grid(figs),
            ))
        tri_inner = "".join(tri_parts)
    else:
        tri_inner = (
            '<p class="muted"><em>(no trivariate figures — run '
            '<code>run_dda_trivariate</code> with a '
            '<code>{(x, y): [group, …]}</code> dict)</em></p>'
        )
    body.append(details_block(
        f"3️⃣ DDA - trivariate ({len(tri_figs) if art.dda_trivariate_figures else 0})",
        tri_inner,
    ))

    return section_block("📊 Descriptive Data Analysis (DDA)", "".join(body))


def _mice_engine_block(art: Artifacts) -> str:
    """Compact table of the imputation engine + R / package versions.

    Reads ``output/missingness/mice/manifest.json`` so the report records the
    exact R, mice, and jsonlite versions used for the run (reproducibility).
    """
    m = art.mice_manifest
    if not m:
        return ""

    def _val(key: str, default: str = "—") -> str:
        v = m.get(key)
        return _esc(str(v)) if v not in (None, "") else default

    sha = m.get("input_sha256")
    sha_short = f"{str(sha)[:12]}…" if sha else "—"
    rows = [
        ("Engine", f"{_val('engine', 'R mice')} — {_val('method')}"),
        ("R version", _val("r_version")),
        ("mice version", _val("mice_version")),
        ("jsonlite version", _val("jsonlite_version")),
        ("Completed datasets (m)", _val("m")),
        ("Iterations (max_iter)", _val("max_iter")),
        ("Seed", _val("seed")),
        ("Rubin pooling supported", _val("rubin_pooling_supported")),
        ("Logged events", _val("logged_events_count")),
        ("Input SHA-256", sha_short),
    ]
    cells = "".join(
        f"<tr><th style='text-align:left;white-space:nowrap'>{label}</th>"
        f"<td>{value}</td></tr>"
        for label, value in rows
    )
    return (
        '<h3>Imputation engine &amp; versions</h3>'
        '<div class="info-box">ℹ️ Recorded automatically from the MICE run '
        '(<code>r_session.json</code> → manifest) for reproducibility.</div>'
        f'<div class="table-wrap"><table class="report">{cells}</table></div>'
    )


def render_missingness(cfg: ReportConfig, art: Artifacts) -> str:
    """🕳️ Missingness story."""
    body = [
        '<p>Missingness was assessed per variable and globally. Formal '
        'mixed-type multiple imputation (MICE) generated m completed datasets '
        'via <code>missingness_resolution.proper_mice_impute()</code>, which '
        'runs one <code>mice()</code> fully-conditional-specification chain in '
        'R (continuous/count → PMM, binary → logistic, nominal → polytomous, '
        'ordinal → proportional-odds). Between-imputation uncertainty is '
        'preserved and pooled with Rubin\u2019s rules in the inferential '
        'stage. Dtypes are restored and every frame is validated (including '
        'Pandera) before use. Variables with high missingness should be '
        'interpreted cautiously, and imputation assumes MAR conditional on the '
        'included predictors. A random-forest chained method '
        '(<code>rf_chained_impute()</code>) is retained only as a labelled '
        'sensitivity analysis and does not support Rubin pooling.</p>',
    ]
    body.append(_mice_engine_block(art))
    if art.missingness_summary is None and not art.missingness_figures:
        body.append(warning_box("No saved missingness artifacts were found."))
        return section_block("🕳️ Missingness story", "".join(body))

    if art.missingness_summary is not None and not art.missingness_summary.empty:
        body.append("<h3>Missingness per variable</h3>")
        body.append(table_to_html(
            art.missingness_summary,
            row_class_fn=lambda r: classify_missing(
                r.get("missing_pct", r.get("pct_missing")), cfg.missing),
            max_rows=200,
        ))

    if art.top_missing is not None and not art.top_missing.empty:
        body.append("<h3>Top missing</h3>")
        body.append(table_to_html(
            art.top_missing,
            row_class_fn=lambda r: classify_missing(
                r.get("missing_pct", r.get("pct_missing")), cfg.missing),
        ))

    if art.missingness_figures:
        body.append("<h3>Patterns</h3>")
        body.append(svg_grid(art.missingness_figures))

    return section_block("🕳️ Missingness story", "".join(body))


def _render_eda_heatmap_overview(
    df: pd.DataFrame,
    cfg: ReportConfig,
    art: Artifacts,
    *,
    target_order: Sequence[str],
) -> str:
    """Embed the seaborn association heatmap (saved artifact or on-the-fly)."""
    from eda import association_heatmap_svg, heatmap_uncorrelated_predictors

    excluded = heatmap_uncorrelated_predictors(
        df, target_order=target_order, fdr_alpha=cfg.fdr_alpha,
    )

    heatmap_path = next(
        (p for p in art.eda_figures if p.stem == "association_heatmap"),
        None,
    )
    if heatmap_path is not None and heatmap_path.exists():
        img = _figure_img_html(heatmap_path)
    else:
        data = association_heatmap_svg(
            df, target_order=target_order, fdr_alpha=cfg.fdr_alpha,
        )
        if not data:
            if excluded:
                items = "".join(f"<li>{_esc(p)}</li>" for p in excluded)
                return (
                    warning_box("Not enough association data for a heatmap overview.")
                    + "<p><strong>Predictors not FDR-significant for any target</strong> "
                    "(omitted from heatmap):</p>"
                    f"<ul class='muted'>{items}</ul>"
                )
            return warning_box("Not enough association data for a heatmap overview.")
        src = f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
        img = (
            '<img src="'
            f'{src}" alt="EDA target × predictor heatmap overview" loading="lazy"/>'
        )

    parts = [
        "<p>Target × predictor matrix from EDA screening (wide seaborn heatmap, "
        "1″ square cells). Only predictors FDR-significant for at least one target "
        "appear in the matrix (strongest left); others are listed below. "
        "Only FDR-significant cells show effect values (marked with *); "
        "non-significant cells are colour-only. <strong>Hatched</strong> cells "
        "carry a magnitude-only effect size (Cramér's V, ε²) whose colour "
        "encodes strength, not direction; grey cells were not tested. "
        f"FDR threshold α = {cfg.fdr_alpha:g}.</p>",
        '<div class="figure-card eda-heatmap-overview">',
        img,
        "</div>",
    ]
    if excluded:
        items = "".join(f"<li>{_esc(p)}</li>" for p in excluded)
        parts.append(
            "<p><strong>Predictors not FDR-significant for any target</strong> "
            "(omitted from heatmap):</p>"
            f"<ul class='muted'>{items}</ul>"
        )
    return "".join(parts)


def _eda_grade_labels(target: str) -> tuple[str, str]:
    """Column labels for outcome-negative / outcome-positive strata."""
    if str(target) == "high_grade":
        return "WHO Grade 1", "WHO Grade 2–3"
    return f"Not {_esc(target)}", _esc(target)


def _eda_cell(val: Any) -> str:
    """Display cell text; missing / NaN / 'nan' → blank."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return "" if not s or s.lower() == "nan" else s


def _eda_paper_display_table(
    rows: pd.DataFrame,
    *,
    table_kind: str,
    target: str,
) -> pd.DataFrame:
    """Format paper rows into the display columns for one datatype block."""
    g1, g23 = _eda_grade_labels(target)
    # Display family: dichotomous / multi-level categorical / interval-ratio
    family = {
        "binary": "dichotomous",
        "dichotomous": "dichotomous",
        "nominal": "categorical",
        "ordinal": "categorical",
        "categorical": "categorical",  # legacy CSV rows
        "continuous": "interval",
        "count": "interval",
    }.get(str(table_kind), "interval")
    out_rows: list[dict[str, str]] = []
    for _, r in rows.iterrows():
        role = _eda_cell(r.get("row_role")) or "variable"
        pred = _eda_cell(r.get("predictor"))
        level = _eda_cell(r.get("level"))
        if role in ("level", "reference"):
            label = f"  {level}" + (" (ref)" if role == "reference" else "")
        else:
            label = prettify_label(pred)

        if family == "dichotomous":
            out_rows.append({
                "Variable": label,
                f"{g1} n/N (%)": _eda_cell(r.get("grade1")),
                f"{g23} n/N (%)": _eda_cell(r.get("grade23")),
                "OR (95% CI)": _eda_cell(r.get("effect")),
                "FDR-p": _eda_cell(r.get("p_fdr")),
            })
        elif family == "categorical":
            # Level rows show their own BH q, never the raw Wald p — the column
            # is headed FDR-p and must mean that on every row in it.
            p_cell = _eda_cell(r.get("p_fdr")) or _eda_cell(r.get("q_level"))
            out_rows.append({
                "Variable": label,
                f"{g1} n/N (%)": _eda_cell(r.get("grade1")),
                f"{g23} n/N (%)": _eda_cell(r.get("grade23")),
                "OR (95% CI)": _eda_cell(r.get("effect")),
                "FDR-p": p_cell,
            })
        else:  # interval / ratio
            out_rows.append({
                "Variable": label,
                f"{g1} median [IQR]": _eda_cell(r.get("grade1")),
                f"{g23} median [IQR]": _eda_cell(r.get("grade23")),
                "OR per SD (95% CI)": _eda_cell(r.get("effect")),
                "AUC (95% CI)": _eda_cell(r.get("auc")),
                "FDR-p": _eda_cell(r.get("p_fdr")),
            })
    return pd.DataFrame(out_rows)


def _eda_stacked_table_html(parts: list[tuple[str, pd.DataFrame]]) -> str:
    """One HTML table: shaded datatype divider rows + written column headers."""
    parts = [(h, df) for h, df in parts if df is not None and not df.empty]
    if not parts:
        return '<p class="muted"><em>(empty table)</em></p>'
    n_cols = max(len(df.columns) for _, df in parts)
    body_rows: list[str] = []
    for heading, df in parts:
        cols = list(df.columns)
        pad = n_cols - len(cols)
        body_rows.append(
            f'<tr class="eda-kind-divider">'
            f'<th colspan="{n_cols}">{_esc(heading)}</th></tr>'
        )
        header_cells = "".join(f"<th>{_esc(c)}</th>" for c in cols)
        if pad:
            header_cells += f'<th colspan="{pad}"></th>'
        body_rows.append(f'<tr class="eda-col-header">{header_cells}</tr>')
        for _, row in df.iterrows():
            cells = "".join(f"<td>{_esc(row[c])}</td>" for c in cols)
            if pad:
                cells += f'<td colspan="{pad}"></td>'
            body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap eda-paper-stack"><table class="report">'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )


_EDA_KIND_SECTIONS: list[tuple[str, str, frozenset[str]]] = [
    ("nominal", "Nominal", frozenset({"nominal", "categorical"})),
    ("ordinal", "Ordinal", frozenset({"ordinal"})),
    ("continuous", "Interval/Ratio", frozenset({"continuous", "count"})),
    ("binary", "Dichotomous", frozenset({"binary", "dichotomous"})),
]


def _eda_datatype_parts(
    chunk: pd.DataFrame,
    *,
    target: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Build stacked datatype sections for one origin chunk; skip empties."""
    parts: list[tuple[str, pd.DataFrame]] = []
    for display_key, heading, kind_keys in _EDA_KIND_SECTIONS:
        kind_rows = chunk[chunk["table_kind"].astype(str).isin(kind_keys)]
        if kind_rows.empty:
            continue
        # Keep variable order; categorical levels follow their parent
        disp = _eda_paper_display_table(
            kind_rows, table_kind=display_key, target=target,
        )
        parts.append((heading, disp))
    return parts


def _render_eda_native_derived_block(
    paper: pd.DataFrame,
    *,
    target: str,
    derived_cols: frozenset[str],
    n_fdr_family: int,
    n_derived_family: int = 0,
    excluded_cols: frozenset[str] = frozenset(),
    hidden_parents: frozenset[str] = frozenset(),
    hidden_replacements: dict[str, list[str]] | None = None,
    derived_sources: dict[str, list[str]] | None = None,
    only: str | None = None,
) -> str:
    """One origin's stacked table plus the footnote both origins share.

    ``only`` selects "native" or "derived". The caller renders each into its own
    fold beside that origin's forest, so a reader never has to hold a table open
    in one place and its plot in another — and never sees a native q ranked
    against a derived one, which is the whole reason the two are corrected
    apart.
    """
    if paper is None or paper.empty:
        return warning_box(
            "No paper-style EDA tables found. Re-run "
            "<code>screen_associations</code> to write "
            "<code>eda/tables/eda_paper_tables.csv</code>."
        )

    sub = paper[paper["target"].astype(str) == str(target)].copy()
    if excluded_cols:
        sub = sub[~sub["predictor"].astype(str).isin(excluded_cols)]
    if sub.empty:
        return warning_box(
            f"No paper-style EDA rows for target <code>{_esc(target)}</code>."
        )

    # A hidden parent that reached this table would be corrected alongside the
    # flag that replaced it — the same information counted twice in one family.
    leaked = sorted(set(sub["predictor"].astype(str)) & set(hidden_parents))
    if leaked:
        raise ValueError(
            f"Hidden parent(s) reached the EDA table: {', '.join(leaked)}. They "
            "were dropped in favour of their derived flags, so counting them "
            "here would put the same information in the family twice."
        )

    is_derived = sub["predictor"].astype(str).isin(derived_cols)
    # Whether a flag counts as derived comes from eda_derived_columns.csv; the
    # derivation log is written by a different step and knows the same fact
    # independently. Where they disagree a derived flag is corrected in the
    # native family alongside the column it restates — which is exactly how
    # multiple_meningiomas ended up with no q at all.
    native_names = set(sub.loc[~is_derived, "predictor"].astype(str))
    doubled = sorted(
        f"{flag} (restates {src})"
        for flag in native_names
        for src in (derived_sources or {}).get(flag, [])
        if src in native_names
    )
    if doubled:
        raise ValueError(
            "Derived flag(s) corrected in the native family alongside the "
            f"column they restate: {'; '.join(doubled)}. That tests the same "
            "information twice and moves every native q. A flag whose parent "
            "is hidden is fine — it *is* the native variable then; this is only "
            "about a flag sitting next to its own source."
        )

    blocks: list[str] = []
    for key, mask in (("native", ~is_derived), ("derived", is_derived)):
        if only is not None and key != only:
            continue
        chunk = sub.loc[mask]
        if chunk.empty:
            continue
        parts = _eda_datatype_parts(chunk, target=target)
        if not parts:
            continue
        # Exactly one <table> per origin — datatypes are divider rows, not tables.
        blocks.append(_eda_stacked_table_html(parts))

    # Named here rather than left implicit: a per-SD odds ratio means nothing
    # until the reader knows which scale that SD is measured on.
    # Level rows are their own multiplicity family — say so, or the reader
    # compares a level q with its parent's q and they are not comparable.
    # Survives a frame without the column, and maps a missing role to "variable"
    # rather than to the string "nan".
    role = (
        sub["row_role"].astype("string").fillna("variable")
        if "row_role" in sub.columns
        else pd.Series("variable", index=sub.index)
    )
    if "q_level" in sub.columns:
        has_q = sub["q_level"].map(_eda_cell).astype(bool)
    else:
        has_q = pd.Series(False, index=sub.index)
    n_level_native = int((role.eq("level") & ~is_derived & has_q).sum())
    n_level_derived = int((role.eq("level") & is_derived & has_q).sum())
    level_note = ""
    if n_level_native or n_level_derived:
        counts = f"{n_level_native} native"
        if n_level_derived:
            counts += f" and {n_level_derived} derived"
        level_note = (
            " Level FDR-p: indented level rows carry their own "
            f"Benjamini–Hochberg across the {counts} level-vs-reference "
            "comparisons, native and derived corrected separately. A level q "
            "belongs to that family, not to the variable family above it."
        )

    # A predictor in neither BH family shows an empty FDR-p cell and quietly
    # pulls the section heading away from the family size printed below it.
    # Name it instead of leaving the blank to be read as "not significant".
    q_col = (
        sub["p_fdr"] if "p_fdr" in sub.columns
        else pd.Series(pd.NA, index=sub.index)
    )
    q_blank = q_col.isna() | (q_col.astype("string").fillna("").str.strip() == "")
    uncorrected = sorted({
        str(p) for p in sub.loc[(role == "variable") & q_blank, "predictor"]
    })
    uncorrected_note = (
        " Shown without FDR correction (in neither multiplicity family): "
        + ", ".join(_esc(prettify_label(p)) for p in uncorrected) + "."
    ) if uncorrected else ""

    # Named, not merely absent. A reader who knows sex was recorded needs to be
    # told it is here as "Male" rather than left to conclude it was not analysed.
    replacements = hidden_replacements or {}
    dropped_note = ""
    if hidden_parents:
        pairs = []
        for parent in sorted(hidden_parents):
            flags = replacements.get(parent) or []
            if flags:
                named = ", ".join(prettify_label(f) for f in sorted(flags))
                pairs.append(f"{prettify_label(parent)} (as {named})")
            else:
                pairs.append(prettify_label(parent))
        dropped_note = (
            " Replaced by derived flags and therefore in neither multiplicity "
            "family: " + _esc("; ".join(pairs)) + "."
        )

    scale_note = scale_footnote(sorted(set(sub["predictor"].astype(str))))
    has_auc = ("auc" in sub.columns
               and sub["auc"].astype("string").fillna("").str.strip().ne("").any())

    # AJNR house style: one "Note:—" paragraph, then the abbreviation list.
    # Written out rather than carried over from the previous version, because
    # three things it described have changed — continuous odds ratios are now
    # standardised on a declared scale, the AUC interval is DeLong rather than
    # a bootstrap, and native and derived variables are corrected in separate
    # families instead of one.
    note = [
        "<strong>Note:&mdash;</strong>Data are median [IQR] for continuous "
        "variables and n/N (%) for binary variables. Odds ratios for "
        "continuous variables are per 1-SD increase; for binary variables they "
        "compare the finding present against absent."
    ]
    if scale_note:
        note.append(_esc(scale_note))
    if has_auc:
        note.append(
            "AUC is reported for continuous variables only, as the DeLong "
            "estimate with a 95% CI computed on the logit scale."
        )
    note.append(
        "P values are corrected for multiple comparisons by the "
        "Benjamini&ndash;Hochberg false discovery rate procedure. "
        "<strong>Native and derived variables form separate families</strong>: "
        f"{n_fdr_family} native variables in this table and "
        f"{n_derived_family} derived variables in the Derived table, corrected "
        "independently. A derived variable is a measurement already in the "
        "native family with a cut-point applied to it, so correcting the two "
        "together would test the same information twice and shift every native "
        "value."
    )
    if level_note:
        note.append(level_note.strip())
    if dropped_note:
        note.append(dropped_note.strip())
    if uncorrected_note:
        note.append(uncorrected_note.strip())
    note.append(
        "Denominators vary because of missing data; each variable is analysed "
        "on its own complete cases. Blank cells indicate not applicable."
    )
    note.append(
        "<strong>These values are not portable: adding or removing any "
        "variable requires the whole of its table to be recomputed.</strong> A "
        "Benjamini&ndash;Hochberg value is the raw P multiplied by the family "
        "size over the variable's rank, so it depends on which other variables "
        "share its table and is not a per-variable constant."
    )
    abbreviations = (
        "AUC indicates area under the receiver operating characteristic curve; "
        "CI, confidence interval; FDR, false discovery rate; IQR, "
        "interquartile range; SD, standard deviation."
    )
    footnote = (
        "<p class='muted'><small>" + " ".join(note) + "</small></p>"
        "<p class='muted'><small>" + abbreviations + "</small></p>"
    )
    return "".join(blocks) + footnote


def _render_univariate_or_forest(
    paper: pd.DataFrame,
    *,
    target: str,
    excluded_cols: frozenset[str] | set[str] = frozenset(),
    include: frozenset[str] | set[str] | None = None,
    caption: str = "Unadjusted odds ratios (95% CI)",
    alt: str | None = None,
) -> str:
    """Embed the unadjusted-OR forest built from the paper-table effect column."""
    if paper is None or paper.empty:
        return ""
    try:
        from eda_paper_tables import (
            draw_univariate_or_forest,
            univariate_or_forest_data,
        )
        from plot_style import figure_to_png_bytes
    except Exception:
        return ""

    plot_df = univariate_or_forest_data(
        paper, target=target, excluded=excluded_cols, include=include,
    )
    if plot_df.empty:
        return ""
    try:
        fig, _ax = draw_univariate_or_forest(plot_df)
        png = figure_to_png_bytes(fig, tight_layout=False, kind="halftone")
    except Exception:
        return ""
    img = _png_bytes_img_html(png, alt or f"{target} univariate forest")
    if not img:
        return ""
    return (
        f'<div class="figure-card">{img}'
        f'<div class="caption">{_esc(caption)}</div>'
        f'</div>'
    )


def render_eda(cfg: ReportConfig, art: Artifacts) -> str:
    """🔍 EDA story — per-target paper tables (native / derived) + figures."""
    body: list[str] = []
    if art.associations is None or art.associations.empty:
        body.append(warning_box("No EDA associations table was found."))
        return section_block("🔍 Exploratory association screening (EDA)", "".join(body))

    body.append(
        '<p>Each predictor was screened against each target using a '
        'test matched to both outcome and predictor types (binary, continuous, '
        'ordinal, or nominal). '
        'p-values are corrected per target using Benjamini–Hochberg FDR.</p>'
    )

    df = art.associations.copy()
    for col in ["target", "predictor", "kind", "test", "effect_label",
                "effect", "p", "p_fdr", "n_used", "auc_univariate"]:
        if col not in df.columns:
            df[col] = np.nan

    excluded_cols = art.eda_excluded_columns | art.hidden_parent_columns
    if excluded_cols:
        df = df[~df["predictor"].astype(str).isin(excluded_cols)].copy()

    targets_in_data = list(df["target"].dropna().unique())
    order = [t for t in cfg.targets if t in targets_in_data]
    order += [t for t in targets_in_data if t not in order]

    body.append(details_block(
        "🥵 Heatmap Overview",
        _render_eda_heatmap_overview(df, cfg, art, target_order=order),
    ))

    derived_cols = art.eda_derived_columns
    paper = art.eda_paper_tables

    for target in order:
        sub = df[df["target"] == target].copy()
        if sub.empty:
            continue

        target_body: list[str] = []
        pred = sub["predictor"].astype(str)
        is_derived = pred.isin(derived_cols)
        if "in_fdr_family" in sub.columns:
            n_fdr_family = int(
                (sub["in_fdr_family"].fillna(True).astype(bool) & ~is_derived).sum()
            )
        else:
            n_fdr_family = int((~is_derived).sum())
        n_derived_family = int(is_derived.sum())

        native_sub = sub.loc[~is_derived].copy()
        native_sub["_p_num"] = native_sub["p_fdr"].apply(_coerce_p)
        native_sub["_eff_abs"] = native_sub["effect"].apply(
            lambda v: abs(_coerce_float(v)) if _coerce_float(v) is not None else -1)
        native_sub = native_sub.sort_values(
            ["_p_num", "_eff_abs"], ascending=[True, False], na_position="last",
        )
        n_fdr = int(native_sub.apply(
            lambda r: classify_significance(
                r.get("p"), r.get("p_fdr"),
                fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha,
            ) == "sig-fdr",
            axis=1,
        ).sum()) if not native_sub.empty else 0
        if n_fdr > 0:
            top = native_sub.iloc[0]
            line = (f"For target <code>{_esc(target)}</code>, "
                    f"<strong>{n_fdr}</strong> native predictor"
                    f"{'s' if n_fdr > 1 else ''} survived FDR correction. "
                    f"Strongest exploratory association: "
                    f"<code>{_esc(top['predictor'])}</code> "
                    f"({_esc(top['effect_label'])} = {_esc(top['effect'])}, "
                    f"FDR p = {_esc(human_p(top['p_fdr']))}).")
        else:
            line = (f"No native predictors survived FDR correction for "
                    f"<code>{_esc(target)}</code>.")
        target_body.append(f"<p>{line}</p>")

        paper_df = paper if paper is not None else pd.DataFrame()

        if paper_df.empty or "predictor" not in paper_df.columns:
            paper_preds: set[str] = set()
        elif "target" in paper_df.columns:
            tmask = paper_df["target"].astype(str) == str(target)
            paper_preds = set(paper_df.loc[tmask, "predictor"].astype(str))
        else:
            paper_preds = set(paper_df["predictor"].astype(str))
        native_preds = paper_preds - set(derived_cols)
        derived_preds = paper_preds & set(derived_cols)

        # One fold per origin, each carrying its own table *and* its own forest.
        # Splitting them across separate folds asked the reader to hold a table
        # open in one place and read its plot in another, and put a native q
        # next to a derived one in the process — the exact comparison the two
        # families are corrected apart to prevent.
        for key, title, preds, caption in (
            ("native", "🌱 Native", native_preds,
             "Native unadjusted odds ratios (95% CI)"),
            ("derived", "🧩 Derived", derived_preds,
             "Derived unadjusted odds ratios (95% CI)"),
        ):
            table_html = _render_eda_native_derived_block(
                paper_df,
                target=str(target),
                derived_cols=derived_cols,
                n_fdr_family=n_fdr_family,
                n_derived_family=n_derived_family,
                excluded_cols=excluded_cols,
                hidden_parents=art.hidden_parent_columns,
                hidden_replacements=art.hidden_parent_replacements,
                derived_sources=art.derived_sources,
                only=key,
            )
            forest = _render_univariate_or_forest(
                paper_df,
                target=str(target),
                excluded_cols=excluded_cols,
                include=preds,
                caption=caption,
                alt=f"{target} {key} univariate forest",
            )
            if not preds and not forest:
                continue
            target_body.append(details_block(
                f"{title} ({len(preds)})",
                table_html + forest,
                open=(key == "native"),
            ))

        # output/panel/ carries no target column — it is built for the primary
        # target, so it hangs under that target and no other.
        if target == order[0]:
            binary_block = _binary_marker_block(art)
            if binary_block:
                target_body.append(binary_block)

        prefix = f"{target}__"
        figs = [
            p for p in art.eda_figures
            if p.stem.startswith(prefix)
            and p.stem[len(prefix):] not in excluded_cols
            and p.stem[len(prefix):] not in {
                "univariate_forest", "univariate_forest_derived",
            }
        ]
        if figs:
            target_body.append(details_block(
                f"🖼️ EDA figures for {target} ({len(figs)})",
                svg_grid(figs)))

        body.append(
            f'<details class="collapsible">'
            f'<summary>🎯 Target: <code>{_esc(target)}</code></summary>'
            f'{"".join(target_body)}</details>'
        )

    return section_block("🔍 Exploratory association screening (EDA)", "".join(body))


def render_inferential(cfg: ReportConfig, art: Artifacts) -> str:
    """🧮 Multivariable / inferential modelling."""
    body: list[str] = []
    if not art.inferential_multivariable and (art.inferential_summary is None
                                               or art.inferential_summary.empty):
        body.append(warning_box("No multivariable model artifacts were found."))
        return section_block("🧮 Multivariable modelling", "".join(body))

    body.append(
        '<p>Multivariable logistic regression was fitted for each target and '
        'each predictor-set variant you defined. Predictors were encoded '
        'according to schema type, continuous/count variables were standardized '
        'so their odds ratios are per 1 SD, '
        'nominal variables were one-hot encoded, and high-VIF predictors were '
        'pruned. Estimates were pooled across the formal mixed-type MICE '
        'datasets with Rubin\u2019s rules.</p>'
        f'<p>{_esc(_MULTIVARIABLE_SCALE_NOTE)}</p>'
        '<p>Missing values — including binary imaging signs — were imputed '
        'within the MICE chain (binary signs via logistic regression) under a '
        'MAR assumption conditional on the included predictors, so patients '
        'are retained and imputation uncertainty propagates into the pooled '
        'confidence intervals. If a sign\u2019s missingness is likely '
        'informative (MNAR), interpret it with a separate sensitivity '
        'analysis.</p>'
    )

    by_target: dict[str, list[str]] = {}
    for key in art.inferential_multivariable:
        target, _ = parse_model_key(key)
        by_target.setdefault(target, []).append(key)

    targets = ([t for t in cfg.targets if t in by_target]
               + [t for t in by_target if t not in cfg.targets])

    for target in targets:
        model_keys = _sort_inferential_model_keys(
            by_target.get(target, []),
            art.inferential_model_experimental,
        )
        body.append(f"<h3>🎯 Target: <code>{_esc(target)}</code></h3>")
        body.append(_render_model_comparison(target, art))

        literature_keys = [
            mkey for mkey in model_keys
            if not _inferential_model_is_experimental(mkey, art.inferential_model_experimental)
        ]
        experimental_keys = [
            mkey for mkey in model_keys
            if _inferential_model_is_experimental(mkey, art.inferential_model_experimental)
        ]

        def _render_model_blocks(keys: list[str]) -> str:
            blocks: list[str] = []
            for mkey in keys:
                _, model_id = parse_model_key(mkey)
                title = art.inferential_model_titles.get(mkey, "")
                link = art.inferential_model_links.get(mkey, "")
                heading = _inferential_model_heading(title, link=link, model_id=model_id)
                if heading:
                    blocks.append(heading)

                blocks.append(_published_model_block(model_id))

                meta = _inferential_target_meta(art, target, model_key_name=mkey)
                if meta:
                    blocks.append(meta)

                stem = artifact_base(target, model_id)
                forest = [p for p in art.inferential_figures if p.stem == f"{stem}__forest"]
                if forest:
                    blocks.append(svg_grid(forest))

                if mkey in art.inferential_vif:
                    blocks.append(details_block(
                        "🔢 VIF diagnostics",
                        table_to_html(art.inferential_vif[mkey])))

                blocks.append(_render_model_performance(stem, art))

                tbl = art.inferential_multivariable[mkey].copy()
                tbl = tbl.drop(columns=["model_title"], errors="ignore")
                if "model_id" in tbl.columns:
                    tbl = tbl[["model_id"] + [c for c in tbl.columns if c != "model_id"]]
                col_or  = _first_present(tbl, ["or", "OR", "odds_ratio"])
                col_lo  = _first_present(tbl, ["or_ci_lo", "ci_lo", "lower"])
                col_hi  = _first_present(tbl, ["or_ci_hi", "ci_hi", "upper"])
                col_p   = _first_present(tbl, ["p", "pvalue", "p_value"])
                col_pred = _first_present(tbl, ["predictor_col", "predictor", "term"])

                def _row_cls(r, _or=col_or, _lo=col_lo, _hi=col_hi):
                    if _or and _lo and _hi:
                        return classify_or_direction(r.get(_or), r.get(_lo), r.get(_hi))
                    return ""

                if col_p:
                    tbl["_p_num"] = tbl[col_p].apply(_coerce_p)
                if col_or:
                    tbl["_eff_abs"] = tbl[col_or].apply(
                        lambda v: abs(math.log(v)) if (o := _coerce_float(v)) and o > 0 else -1)
                sort_cols = [c for c in ("_p_num", "_eff_abs", col_pred) if c and c in tbl.columns]
                if sort_cols:
                    tbl = tbl.sort_values(
                        sort_cols,
                        ascending=[True, False, True][: len(sort_cols)],
                        na_position="last",
                    )
                tbl = tbl.drop(columns=[c for c in ("_p_num", "_eff_abs") if c in tbl.columns])

                if col_p and col_p in tbl.columns:
                    tbl[col_p] = tbl[col_p].apply(human_p)
                if "df" in tbl.columns:
                    tbl["df"] = tbl["df"].apply(human_pool_df)

                nowrap = ("model_id",) if "model_id" in tbl.columns else ()
                blocks.append(table_to_html(tbl, row_class_fn=_row_cls, nowrap_cols=nowrap))
                blocks.append(_render_inferential_interpretation(
                    target, tbl, col_pred, col_or, col_lo, col_hi, col_p))
            return "".join(blocks)

        if literature_keys:
            body.append(details_block(
                "📚 Literature-based models",
                _render_model_blocks(literature_keys),
            ))
        if experimental_keys:
            body.append(details_block(
                "🧪 Experimental models",
                _render_model_blocks(experimental_keys),
            ))

    return section_block("🧮 Multivariable modelling", "".join(body))


# ---------------------------------------------------------------------------
# Marker panel
# ---------------------------------------------------------------------------

def _lead(text: str) -> str:
    return f'<p class="lead">{text}</p>'


def _num(value: Any, digits: int = 2) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{v:.{digits}f}"


def _int(value: Any) -> str:
    v = _coerce_float(value)
    return "—" if v is None else f"{int(round(v))}"


def _table(df: pd.DataFrame) -> str:
    return table_to_html(df)


def _panel_figure(art: Artifacts, stem: str) -> str:
    """One panel figure by filename stem, or nothing if it was not written."""
    for path in art.panel_figures:
        if path.stem == stem:
            img = _figure_img_html(path)
            return f'<div class="figure-card">{img}</div>' if img else ""
    return ""


def _panel_aim_one(art: Artifacts) -> str:
    view = art.panel_marker_reading_view
    if view is None or view.empty:
        return warning_box("No marker table was found.")
    top = art.panel_marker.iloc[0] if art.panel_marker is not None and \
        not art.panel_marker.empty else None
    lead = ""
    if top is not None and not bool(top.get("chance_overlap")):
        lead = _lead(
            f"<strong>{_esc(top['label'])}</strong> argues hardest for high grade: "
            f"seeing it makes high grade {_num(top['lr_pos'], 1)}× more likely "
            f"({_num(top['lr_pos_lo'], 1)}–{_num(top['lr_pos_hi'], 1)}). "
            f"It is present in {_int(top['present_n'])} of "
            f"{_int(top['n_used'])} scans and flags "
            f"{_int(top['catches'])} of the {_int(top['n_high_grade'])} "
            "high-grade tumours."
        )
    origins = (view["origin"].astype(str) if "origin" in view.columns
               else pd.Series("native", index=view.index))
    shown = view.drop(columns=["origin"], errors="ignore")
    parts = [lead]
    groups = (
        ("native", "🌱 Native", "a", "recorded directly"),
        ("derived", "🧩 Derived", "b", "dichotomised from a measurement"),
    )
    for key, title, letter, gloss in groups:
        rows = shown[origins.eq(key)]
        if rows.empty:
            continue
        parts.append(
            f"<h3><strong>Table 4{letter}.</strong> {title} — diagnostic "
            "performance of individual variables for WHO grade 2–3 meningioma "
            f"({gloss}, n = {len(rows)})</h3>"
        )
        parts.append(_table(rows))
        parts.append(_panel_figure(art, f"lr_forest_{key}"))
        parts.append(_panel_forest_caption(art, group=title, letter=letter))
    parts.append(_panel_table_footnotes(art))
    return "".join(p for p in parts if p)


def _panel_forest_caption(art: Artifacts, *, group: str = "",
                          letter: str = "") -> str:
    """The forest plot's own caption.

    A figure is read on its own — lifted into a slide, a poster or a reviewer's
    PDF viewer without the table above it — so its abbreviations are spelled
    out here rather than borrowed from the table footnote.
    """
    if art.panel_marker is None or art.panel_marker.empty:
        return ""
    return (
        f"<p class='caption'><strong>Figure 3{letter}.</strong> "
        + (f"{_esc(group)}. " if group else "")
        + "Positive likelihood "
        "ratio for each variable, with 95% confidence interval, on a "
        "logarithmic axis. A ratio of 1 means the finding leaves the "
        "probability of WHO grade 2–3 unchanged; variables whose interval "
        "crosses 1 are shaded and drawn in grey. Values repeat in the "
        "right-hand column. "
        "ADC = apparent diffusion coefficient; CI = confidence interval; "
        "DWI = diffusion-weighted imaging; LR+ = positive likelihood ratio; "
        "T1/T2 = T1- and T2-weighted imaging.</p>"
    )


def _panel_prevalence(panel) -> tuple[float, int, int] | None:
    """Outcome rate, from the variable with the largest denominator.

    Each row has its own complete cases, so there is no single cohort n in
    this table; the most completely measured variable is the closest thing to
    it, and it is quoted with its own n so the reader can check it.
    """
    if panel is None or panel.empty:
        return None
    if not {"n_used", "n_high_grade"}.issubset(panel.columns):
        return None
    rows = panel.dropna(subset=["n_used", "n_high_grade"])
    if rows.empty:
        return None
    row = rows.loc[rows["n_used"].idxmax()]
    n, events = int(row["n_used"]), int(row["n_high_grade"])
    return (events / n, events, n) if n else None


def _panel_table_footnotes(art: Artifacts) -> str:
    """The table's own footnote block — it has to stand on its own.

    Radiology journals expect a table to be readable lifted out of the paper,
    so every abbreviation is defined here even though the running text defines
    them too, and the sort rule is stated rather than inferred.
    """
    panel = art.panel_marker
    corrected = (panel is not None and not panel.empty
                 and "continuity_corrected" in panel.columns
                 and bool(panel["continuity_corrected"].any()))
    n_rows = 0 if panel is None or panel.empty else len(panel)
    if panel is not None and not panel.empty and "origin" in panel.columns:
        origins = panel["origin"].astype(str)
        n_native = int(origins.eq("native").sum())
        n_derived = int(origins.eq("derived").sum())
    else:
        n_native, n_derived = n_rows, 0
    # A flag whose parent was hidden counts as native: it replaced that column
    # outright, so nothing in the table restates anything. Named, because a
    # reader who knows sex was recorded needs to be told it is here as "Male".
    replaced_note = ""
    if art.hidden_parent_replacements:
        pairs = "; ".join(
            f"{prettify_label(parent)} (as "
            + ", ".join(prettify_label(f) for f in sorted(flags)) + ")"
            for parent, flags in sorted(art.hidden_parent_replacements.items())
            if flags
        )
        if pairs:
            replaced_note = (
                "Variables replaced outright by a derived flag are counted as "
                "native, because the column they were cut from is not in this "
                f"table for them to restate: {_esc(pairs)}. "
            )

    lines = [
        "Variables are sorted by LR+ in descending order. "
        "LR+ with 95% CI crossing 1.0 indicates no significant discriminative "
        "value.",
        "FDR p is the Benjamini–Hochberg adjusted p for the χ² test of "
        "association between the finding and the outcome. It is a different "
        "statistic from LR+ and can disagree with it: a variable may survive "
        "correction while its likelihood ratio interval still crosses 1, and "
        "the reverse. Read the interval for discriminative value and FDR p for "
        "whether the association survives testing several variables at once.",
        replaced_note
        + f"{_esc('Native and derived variables are corrected separately')}: "
        f"Benjamini–Hochberg runs across the {n_native} native variables in "
        f"Table 4a and, independently, across the {n_derived} derived "
        "variables in Table 4b. Derived variables do not enter the native "
        "family, because each one is a measurement already in that family with "
        "a cut-point applied to it — correcting the two together would test "
        "the same information twice and shift every native q. A q from one "
        "table is a rank within its own family and is not comparable with a q "
        "from the other.",
        "<strong>These q values are not portable. Adding or removing any "
        "variable means recomputing the whole table it sits in.</strong> A "
        "Benjamini–Hochberg q is the raw p multiplied by the family size over "
        f"the row's rank, so it depends on which other variables are in the "
        f"table with it. Table 4a currently has {n_native} members and Table 4b "
        f"has {n_derived}; change either membership and every q in that table "
        "moves. They are not per-variable constants that can be carried into a "
        "table with different members.",
    ]
    prev = _panel_prevalence(panel)
    if prev is not None:
        rate, events, n = prev
        lines.append(
            f"Cohort prevalence of WHO grade 2–3 is {rate:.1%} ({events}/{n}). "
            "LR+, sensitivity and specificity do not depend on it; PPV and NPV "
            "do, and apply only to a population with this grade 2–3 rate."
        )
    lines.append(
        "n/N (%) = patients with the finding present / patients assessed for "
        "it (percentage); each variable is scored on its own complete cases, "
        "so denominators differ between rows. LR+ = positive likelihood ratio, "
        "the number of times more often a finding is present in a WHO grade "
        "2–3 tumour than in a grade 1 one. NPV = negative predictive value, "
        "the proportion of tumours without the finding that are grade 1; "
        "PPV = positive predictive value, the proportion of tumours with the "
        "finding that are grade 2–3."
    )
    if corrected:
        lines.append(
            "* estimate calculated with a continuity correction because one "
            "cell of the 2×2 table was empty.")
    lines.append(
        "ADC = apparent diffusion coefficient; CI = confidence interval; "
        "DWI = diffusion-weighted imaging; LR+ = positive likelihood ratio; "
        "NPV = negative predictive value; PPV = positive predictive value; "
        "Sens = sensitivity; Spec = specificity; T1/T2 = T1- and T2-weighted "
        "imaging.")
    return ("<p class='muted'><small>"
            + "<br>".join(lines)
            + "</small></p>")


def _binary_marker_block(art: Artifacts) -> str:
    """The binary-marker panel, as one dropdown inside its EDA target.

    Returns "" when ``output/panel/`` was never written: the target body then
    simply has one dropdown fewer, rather than a warning about artifacts the
    reader did not ask for.
    """
    view = art.panel_marker_reading_view
    if art.panel_marker is None or view is None or view.empty:
        return ""
    return details_block("⋈ Binary-extended", _panel_aim_one(art))


def _to_int_or_none(x: Any) -> int | None:
    v = _coerce_float(x)
    return int(v) if v is not None else None


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _render_inferential_interpretation(target: str, tbl: pd.DataFrame,
                                       col_pred: str | None, col_or: str | None,
                                       col_lo: str | None, col_hi: str | None,
                                       col_p: str | None) -> str:
    if not all([col_pred, col_or, col_lo, col_hi]):
        return ""
    lines = []
    for _, r in tbl.iterrows():
        o  = _coerce_float(r.get(col_or))
        lo = _coerce_float(r.get(col_lo))
        hi = _coerce_float(r.get(col_hi))
        if o is None or lo is None or hi is None:
            continue
        pred = _esc(r.get(col_pred))
        p_str = _esc(r.get(col_p)) if col_p else ""
        if lo > 1.0:
            lines.append(f"<li>🔴 <code>{pred}</code> was associated with "
                         f"<strong>higher</strong> odds of <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}"
                         + (f", p={p_str}" if p_str else "") + ").</li>")
        elif hi < 1.0:
            lines.append(f"<li>🔵 <code>{pred}</code> was associated with "
                         f"<strong>lower</strong> odds of <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}"
                         + (f", p={p_str}" if p_str else "") + ").</li>")
        else:
            lines.append(f"<li>⚪ <code>{pred}</code> did not show a stable "
                         f"independent association with <code>{_esc(target)}</code> "
                         f"(OR={o:.2f}, 95% CI {lo:.2f}–{hi:.2f}; CI crosses 1).</li>")
    if not lines:
        return ""
    return details_block(
        "💡 Interpretation", "<ul>" + "".join(lines) + "</ul>")


# ---------------------------------------------------------------------------
# Appendix — runtime environment
# ---------------------------------------------------------------------------

_IMPORT_TO_DIST: dict[str, str] = {
    "IPython": "ipython",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


def _repo_root() -> Path:
    """meningioma-atypier project root (parent of heavy_machinery/)."""
    return Path(__file__).resolve().parent.parent.parent


def _collect_local_module_names(repo_root: Path) -> set[str]:
    """Top-level names of project .py modules (excluded from third-party scan)."""
    names: set[str] = set()
    for path in repo_root.rglob("*.py"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.name == "__init__.py":
            names.add(path.parent.name)
        else:
            names.add(path.stem)
    return names


def _collect_imported_top_level_modules(
    repo_root: Path,
    *,
    local: set[str],
) -> set[str]:
    """Third-party top-level modules imported anywhere under repo_root."""
    stdlib = getattr(sys, "stdlib_module_names", set())
    found: set[str] = set()
    for path in repo_root.rglob("*.py"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return {m for m in found if m not in stdlib and m not in local}


def _parse_requirements_packages(requirements_path: Path) -> set[str]:
    """Distribution names declared in requirements.txt (comments/blank lines skipped)."""
    if not requirements_path.is_file():
        return set()
    names: set[str] = set()
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for sep in (">=", "<=", "==", "!=", "~=", "<", ">", ";"):
            if sep in line:
                line = line.split(sep, 1)[0]
        name = line.strip()
        if name:
            names.add(name)
    return names


@functools.lru_cache(maxsize=1)
def _packages_distributions() -> dict[str, list[str]]:
    """Import-name -> distribution map, built once.

    ``importlib.metadata.packages_distributions()`` walks every path entry and
    reads the metadata of every installed distribution — about half a second in
    this environment. It was being rebuilt from scratch for each import name the
    appendix looked up, which is where the environment table spent its time.
    The interpreter's installed set cannot change mid-run, so one build is enough.
    """
    return importlib.metadata.packages_distributions()


def _distribution_name(import_name: str) -> str:
    """Map a top-level import name to its PyPI distribution name."""
    mapped = _packages_distributions().get(import_name)
    if mapped:
        return mapped[0]
    return _IMPORT_TO_DIST.get(import_name, import_name)


def _discovered_distribution_names() -> list[str]:
    """Union of requirements.txt and third-party imports across the whole repo."""
    repo = _repo_root()
    local = _collect_local_module_names(repo)
    imports = _collect_imported_top_level_modules(repo, local=local)
    req = _parse_requirements_packages(repo / "requirements.txt")
    dists = set(req)
    dists.update(_distribution_name(imp) for imp in imports)
    return sorted(dists, key=str.lower)


def _total_memory_gb() -> float | None:
    """Best-effort total RAM (GiB); stdlib only, no extra dependencies."""
    if sys.platform == "win32":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        except Exception:
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            return None
    return None


def _run_command(args: Sequence[str], *, timeout: float = 6.0) -> str | None:
    """Run a command and return stripped stdout, or None on failure."""
    try:
        out = subprocess.check_output(
            list(args),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        text = out.strip()
        return text or None
    except Exception:
        return None


def _win_processor_description() -> str | None:
    ps = (
        "$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "if (-not $p) { exit 1 }; "
        "$name = ($p.Name -replace '\\s+', ' ').Trim(); "
        "$cores = $p.NumberOfCores; $threads = $p.NumberOfLogicalProcessors; "
        "Write-Output ($name + ' — ' + $cores + ' cores, ' + $threads + ' threads')"
    )
    return _run_command(["powershell", "-NoProfile", "-Command", ps])


def _win_graphics_descriptions() -> list[str]:
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -and $_.Name -notmatch 'Microsoft Basic' } | "
        "ForEach-Object { "
        "  $n = ($_.Name -replace '\\s+', ' ').Trim(); "
        "  $parts = @($n); "
        "  if ($_.AdapterRAM -and [uint64]$_.AdapterRAM -gt 0) { "
        "    $gb = [math]::Round([uint64]$_.AdapterRAM / 1GB, 1); "
        "    $parts += ($gb.ToString() + ' GB VRAM'); "
        "  }; "
        "  if ($_.DriverVersion) { $parts += ('driver ' + $_.DriverVersion); }; "
        "  Write-Output ($parts -join ', ') "
        "}"
    )
    raw = _run_command(["powershell", "-NoProfile", "-Command", ps])
    if not raw:
        return []
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def _linux_processor_description() -> str | None:
    try:
        model: str | None = None
        logical_ids: set[str] = set()
        physical_ids: set[tuple[str, str]] = set()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            core_id = socket_id = None
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("processor"):
                    logical_ids.add(line.split(":", 1)[1].strip())
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif line.startswith("physical id"):
                    socket_id = line.split(":", 1)[1].strip()
                    if core_id is not None:
                        physical_ids.add((socket_id, core_id))
                        core_id = None
        if not model:
            return None
        threads = len(logical_ids) or os.cpu_count()
        cores = len(physical_ids) if physical_ids else None
        if cores and threads:
            return f"{model} — {cores} cores, {threads} threads"
        if threads:
            return f"{model} — {threads} threads"
        return model
    except Exception:
        return None


def _linux_graphics_descriptions() -> list[str]:
    raw = _run_command(["lspci"], timeout=4.0)
    if not raw:
        return []
    gpus: list[str] = []
    for line in raw.splitlines():
        if any(tag in line for tag in ("VGA compatible controller", "3D controller", "Display controller")):
            name = line.split(":", 2)[-1].strip()
            if name and name not in gpus:
                gpus.append(name)
    return gpus


def _mac_processor_description() -> str | None:
    brand = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not brand:
        return None
    cores = os.cpu_count()
    if cores:
        return f"{brand} — {cores} logical cores"
    return brand


def _mac_graphics_descriptions() -> list[str]:
    raw = _run_command(["system_profiler", "SPDisplaysDataType"], timeout=12.0)
    if not raw:
        return []
    gpus: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            if current:
                gpus.append(", ".join(current))
            current = [stripped.split(":", 1)[1].strip()]
        elif stripped.startswith("VRAM (") and current:
            current.append(stripped.split(":", 1)[1].strip() + " VRAM")
        elif stripped == "" and current:
            gpus.append(", ".join(current))
            current = []
    if current:
        gpus.append(", ".join(current))
    return gpus


def _nvidia_smi_descriptions() -> list[str]:
    raw = _run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return []
    gpus: list[str] = []
    for line in raw.splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        extras: list[str] = []
        if len(parts) > 1 and parts[1]:
            try:
                mib = float(parts[1])
                extras.append(f"{mib / 1024:.1f} GB VRAM")
            except ValueError:
                extras.append(f"{parts[1]} MB VRAM")
        if len(parts) > 2 and parts[2]:
            extras.append(f"driver {parts[2]}")
        gpus.append(f"{name}, {', '.join(extras)}" if extras else name)
    return gpus


def _processor_description() -> str:
    desc: str | None = None
    if sys.platform == "win32":
        desc = _win_processor_description()
    elif sys.platform == "darwin":
        desc = _mac_processor_description()
    elif sys.platform.startswith("linux"):
        desc = _linux_processor_description()
    if desc:
        return desc
    proc = platform.processor()
    cores = os.cpu_count()
    if proc and cores:
        return f"{proc} — {cores} logical cores"
    return proc or (f"{cores} logical cores" if cores else "—")


def _graphics_descriptions() -> list[str]:
    gpus = _nvidia_smi_descriptions()
    if gpus:
        return gpus
    if sys.platform == "win32":
        gpus = _win_graphics_descriptions()
    elif sys.platform == "darwin":
        gpus = _mac_graphics_descriptions()
    elif sys.platform.startswith("linux"):
        gpus = _linux_graphics_descriptions()
    else:
        gpus = []
    return [g for g in gpus if g]


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        import_name = distribution_name.replace("-", "_")
        try:
            mod = importlib.import_module(import_name)
        except ImportError:
            return "not installed"
        ver = getattr(mod, "__version__", None)
        return str(ver) if ver else "unknown"


def _system_specs_rows() -> list[dict[str, str]]:
    rows = [
        {
            "item": "Python",
            "value": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro} ({platform.python_implementation()})"
            ),
        },
        {"item": "Operating system", "value": platform.platform()},
        {"item": "Architecture", "value": platform.machine()},
        {"item": "Processor", "value": _processor_description()},
    ]
    gpus = _graphics_descriptions()
    if gpus:
        rows.append({
            "item": "Graphics" if len(gpus) == 1 else "Graphics (GPUs)",
            "value": "; ".join(gpus),
        })
    mem_gb = _total_memory_gb()
    if mem_gb is not None:
        rows.append({"item": "RAM", "value": f"{mem_gb:.1f} GB"})
    rows.append({"item": "Report generated", "value": platform.node() or "—"})
    return rows


def _package_version_rows() -> list[dict[str, str]]:
    return [
        {"package": dist, "version": _package_version(dist)}
        for dist in _discovered_distribution_names()
    ]


def _r_module_rows(art: Artifacts) -> list[tuple[str, str]]:
    """R interpreter + package versions, taken from the formal-MICE manifest.

    R packages are not pip-discoverable, so their versions are recorded by
    ``scripts/run_mice.R`` into the manifest at imputation time.
    """
    m = art.mice_manifest or {}
    r_ver = m.get("r_version")
    if r_ver:
        # "R version 4.6.1 (2026-06-24)" -> "4.6.1 (2026-06-24)"
        r_ver = str(r_ver).replace("R version ", "")
    return [
        ("R", r_ver or "not recorded (run formal MICE)"),
        ("mice", str(m.get("mice_version") or "not recorded")),
        ("jsonlite", str(m.get("jsonlite_version") or "not recorded")),
    ]


def _grouped_versions_table(groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Render package versions grouped by language, child rows indented."""
    parts = ['<div class="table-wrap"><table class="report">',
             "<tr><th>Module</th><th>Version</th></tr>"]
    for header, rows in groups:
        parts.append(
            f'<tr><th colspan="2" style="text-align:left">{_esc(header)}</th></tr>'
        )
        for name, ver in rows:
            parts.append(
                f'<tr><td style="padding-left:1.8em">{_esc(name)}</td>'
                f"<td>{_esc(ver)}</td></tr>"
            )
    parts.append("</table></div>")
    return "".join(parts)


def _render_environment_appendix(art: Artifacts) -> str:
    """Computer specs + Python and R module versions for reproducibility."""
    py_rows = [(r["package"], r["version"]) for r in _package_version_rows()]
    versions = _grouped_versions_table([
        ("🐍 Python modules", py_rows),
        ("📊 R modules (formal MICE engine)", _r_module_rows(art)),
    ])
    return (
        "<p>Environment at the time this report was built (not read from "
        "saved artifacts).</p>"
        "<h4>Computer / runtime</h4>"
        f"{table_to_html(pd.DataFrame(_system_specs_rows()))}"
        "<h4>Package versions</h4>"
        "<p>Python libraries are declared in <code>requirements.txt</code> or "
        "imported anywhere in the repo. R modules run the formal MICE engine "
        "(<code>scripts/run_mice.R</code>) and their versions are recorded in "
        "the MICE manifest at imputation time.</p>"
        f"{versions}"
    )


def render_appendix(cfg: ReportConfig, art: Artifacts) -> str:
    """📎 Appendix — warnings and runtime environment."""
    body = ['<h2>📎 Appendix</h2>']

    if art.warnings:
        body.append("<h3>Warnings during artifact load</h3>")
        body.append("<ul>" + "".join(f"<li>{_esc(w)}</li>" for w in art.warnings)
                    + "</ul>")

    body.append(details_block(
        "🖥️ Environment & package versions",
        _render_environment_appendix(art),
    ))

    return f'<section class="report-section">{"".join(body)}</section>'


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_report(cfg: ReportConfig) -> str:
    """Assemble the full HTML document and return it as a string."""
    art = load_artifacts(cfg)

    sections = [
        render_header(cfg, art),
        render_cleaning(cfg, art),
        render_dda(cfg, art),
        render_missingness(cfg, art),
        render_eda(cfg, art),
        render_inferential(cfg, art),
        render_appendix(cfg, art),
    ]

    footer = (
        f'<div class="footer">Generated {datetime.now().isoformat(timespec="seconds")} '
        f'· output root: <code>{_esc(cfg.output_root)}</code></div>'
    )

    body = "".join(sections) + footer
    return _wrap_html(cfg.title, body)


def _wrap_html(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{_esc(title)}</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<style>{_CSS}</style>'
        '</head><body><div class="container">'
        f'{body}'
        '</div></body></html>'
    )


def write_html(html: str, out_path: Path) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True, type=Path,
                   help="Root directory containing dda/, eda/, inferential/, etc.")
    p.add_argument("--targets", nargs="*", default=[],
                   help="Names of target columns (used for ordering / role tags).")
    p.add_argument("--schema", type=Path, default=None,
                   help="Optional schema CSV or JSON to render the schema section.")
    p.add_argument("--title", default="Research Data Analysis Report")
    p.add_argument("--author", default="")
    p.add_argument("--out", type=Path, default=None,
                   help="Output HTML path. Defaults to <output-root>/report/report.html")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = ReportConfig(
        output_root=args.output_root.expanduser().resolve(),
        title=args.title,
        author=args.author,
        targets=tuple(args.targets),
        schema_path=args.schema.expanduser().resolve() if args.schema else None,
    )
    out_path = args.out or (cfg.output_root / "report" / "report.html")
    html = build_report(cfg)
    written = write_html(html, out_path)
    print(f"Report written: {written} (self-contained; figures embedded inline)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
