"""Assemble ``output/report/report.html`` from existing pipeline artifacts.

Reads CSV/SVG under ``output/`` (no refitting). Collapsible major sections;
multivariable uses nested literature / experimental dropdowns per target.
CLI: ``python report.py --output-root output``.
"""

from __future__ import annotations

import argparse
import ast
import base64
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

from cleaning import format_number, format_table_for_display
from plot_style import prettify_caption


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

/* Figure grid */
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px,1fr));
               gap: 14px; margin: 12px 0 18px; }
.figure-card { border: 1px solid var(--border); border-radius: 8px;
               padding: 8px; background: #fff; }
.figure-card img, .figure-card object {
    width: 100%; height: auto; display: block;
}
.figure-card .caption { font-size: 12px; color: var(--muted);
                        margin-top: 4px; text-align: center;
                        word-break: break-word; }

/* Collapsible details */
details.collapsible { margin: 8px 0 14px; }
details.collapsible > summary {
    cursor: pointer; font-weight: 600; padding: 6px 0;
    color: var(--accent);
}
details.collapsible.glossary-block {
    margin: 24px 0 8px;
    padding: 10px 14px 6px;
    border: none;
    border-top: 1px dashed var(--border);
    border-radius: 0;
    background: transparent;
}
details.collapsible.glossary-block > summary {
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    font-size: 13px;
    font-weight: 500;
    font-style: italic;
    letter-spacing: 0.01em;
    padding: 4px 0 6px;
    color: #5b6b7c;
}
details.collapsible.glossary-block[open] > summary {
    color: #4a5568;
}
details.collapsible.glossary-block h4 {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 14px 0 6px;
}
details.collapsible.glossary-block .stat-decoder dt {
    font-size: 12px;
    font-weight: 600;
    margin-top: 10px;
    color: #4b5563;
}
details.collapsible.glossary-block .stat-decoder dd {
    font-size: 12.5px;
    color: #6b7280;
    line-height: 1.45;
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
details.section-collapsible .section-body > details.collapsible:not(.glossary-block) {
    margin: 12px 0 16px;
    padding: 12px 16px 4px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card);
}
details.section-collapsible .section-body > details.collapsible:not(.glossary-block) > summary {
    font-size: 15px;
    padding: 4px 0 8px;
}
details.section-collapsible > .section-body > h3:first-child,
details.section-collapsible > .section-body > p:first-child {
    margin-top: 12px;
}

/* TL;DR list */
.tldr-list { padding-left: 22px; }
.tldr-list li { margin: 4px 0; }

/* Glossary dropdowns (DDA / EDA / missingness / inferential) */
.stat-decoder dt { font-weight: 600; margin-top: 12px; }
.stat-decoder dd { margin-left: 0; color: #374151; }

/* Footer */
.footer { color: var(--muted); font-size: 12px; margin-top: 60px;
          border-top: 1px solid var(--border); padding-top: 12px; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_svg_src(path: Path) -> str | None:
    """Return a ``data:image/svg+xml;base64,...`` URI for embedding in ``<img src>``."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip():
        return None
    encoded = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _figure_img_html(path: Path) -> str:
    """``<img>`` tag with the SVG inlined as a data URI (self-contained HTML)."""
    src = _embed_svg_src(path)
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


# Strength tier wording -------------------------------------------------------

_STRENGTH_WORDING = {
    "strong":   ("🟢", "strong",   "Large statistical signal; worth serious attention, but still needs clinical context."),
    "moderate": ("🟡", "moderate", "Visible statistical signal; worth attention and hypothesis generation."),
    "weak":     ("🔴", "weak",     "Small statistical signal; probably not enough alone to guide clinical decisions."),
    "none":     ("⚪", "none",     "No useful statistical pattern detected."),
}


def effect_badge(effect: Any, kind: str = "corr",
                 thr: EffectThresholds = EffectThresholds()) -> str:
    """Return inline HTML badge for an effect magnitude.

    Parameters
    ----------
    effect : numeric
        Spearman rho / rank-biserial r / Cramer's V / odds ratio.
    kind : {"corr", "or"}
        ``"corr"`` thresholds use ``|effect|`` directly; ``"or"`` thresholds
        use ``|log(OR)|`` so risk and protective directions share one scale.
    """
    tier = _strength_tier(effect, kind, thr)
    emoji, label, _ = _STRENGTH_WORDING[tier]
    return f'<span class="badge effect-{tier}">{emoji} {label}</span>'


def _strength_tier(effect: Any, kind: str, thr: EffectThresholds) -> str:
    e = _coerce_float(effect)
    if e is None:
        return "none"
    if kind == "or":
        if e <= 0:
            return "none"
        mag = abs(math.log(e))
        if mag >= thr.or_strong:   return "strong"
        if mag >= thr.or_moderate: return "moderate"
        if mag >= thr.or_weak:     return "weak"
        return "none"
    # corr-like
    mag = abs(e)
    if mag >= thr.corr_strong:   return "strong"
    if mag >= thr.corr_moderate: return "moderate"
    if mag >= thr.corr_weak:     return "weak"
    return "none"


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


def glossary_block(inner_html: str, *, open: bool = False) -> str:
    """Section-end metrics glossary (DDA / EDA / missingness / inferential)."""
    open_attr = " open" if open else ""
    return (
        f'<details class="collapsible glossary-block"{open_attr}>'
        f'<summary>{_esc("📖 What do these metrics mean?")}</summary>'
        f'{inner_html}</details>'
    )


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
    schema_summary: pd.DataFrame | None = None

    # DDA
    dda_overall: pd.DataFrame | None = None
    dda_continuous: pd.DataFrame | None = None
    dda_categorical: pd.DataFrame | None = None
    dda_binary: pd.DataFrame | None = None
    dda_datetime: pd.DataFrame | None = None
    dda_id_text: pd.DataFrame | None = None
    dda_figures: list[Path] = field(default_factory=list)

    # Missingness
    missingness_summary: pd.DataFrame | None = None
    top_missing: pd.DataFrame | None = None
    missingness_figures: list[Path] = field(default_factory=list)
    mice_manifest: dict[str, Any] | None = None

    # EDA
    associations: pd.DataFrame | None = None
    diagnostic_accuracy: pd.DataFrame | None = None
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
        art.dda_figures = sorted(dda_fig.glob("*.svg"))

    # Missingness
    miss_tab = root / "missingness" / "tables"
    art.missingness_summary = _maybe_read_csv(miss_tab / "missingness_summary.csv", art.warnings)
    art.top_missing         = _maybe_read_csv(miss_tab / "top_missing.csv", art.warnings)
    # Fall back to flat layout (older runs)
    if art.missingness_summary is None:
        art.missingness_summary = _maybe_read_csv(root / "missingness" / "missing_per_column.csv", art.warnings)
    miss_fig = root / "missingness" / "figures"
    if miss_fig.exists():
        art.missingness_figures = sorted(miss_fig.glob("*.svg"))
    else:
        # Older layout: figures dropped directly in missingness/
        flat = root / "missingness"
        if flat.exists():
            art.missingness_figures = sorted(flat.glob("*.svg"))
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
    eda_fig = root / "eda" / "figures"
    if eda_fig.exists():
        art.eda_figures = sorted(eda_fig.glob("*.svg"))

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
        art.inferential_figures = sorted(inf_fig.glob("*.svg"))

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

def _inferential_model_heading(
    title: str = "",
    *,
    link: str = "",
    model_id: str = "",
) -> str:
    """Section heading for one multivariable model variant."""
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


# Per-operation column layout for the cleaning log. Each `step` is rendered as
# its own table showing only the columns relevant to that operation.
_CLEANING_LOG_GROUPS = [
    ("apply_schema", "Schema coercions",
     ["column", "action", "kind", "reason"]),
    ("drop_rows", "Dropped rows",
     ["reason", "criterion", "n_before", "n_dropped", "n_remaining"]),
]


def _render_cleaning_log_tables(log: pd.DataFrame) -> str:
    """Split the cleaning log by `step` into separate per-operation tables."""
    if "step" not in log.columns:
        return table_to_html(log, max_rows=200)

    def _drop_blank_cols(df: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in df.columns
                if not df[c].isna().all()
                and not (df[c].astype(str).str.strip() == "").all()]
        return df[keep]

    parts: list[str] = []
    handled: set[str] = set()
    for step_val, label, cols in _CLEANING_LOG_GROUPS:
        sub = log[log["step"] == step_val]
        if sub.empty:
            continue
        handled.add(step_val)
        sub = _drop_blank_cols(sub[[c for c in cols if c in sub.columns]])
        sub = _format_count_cols(sub, ["n_before", "n_dropped", "n_remaining"])
        parts.append(f"<h4>{_esc(label)}</h4>")
        parts.append(table_to_html(sub, max_rows=200))

    others = log[~log["step"].isin(handled)]
    if not others.empty:
        parts.append("<h4>Other steps</h4>")
        parts.append(table_to_html(_drop_blank_cols(others), max_rows=200))

    return "".join(parts) if parts else table_to_html(log, max_rows=200)


def _summary_with_derived_step(
    summary: pd.DataFrame,
    derivation_log: pd.DataFrame | None,
) -> pd.DataFrame:
    """Insert a ``derived`` row into the cleaning summary naming the new columns."""
    if (derivation_log is None or derivation_log.empty
            or "derivation" not in derivation_log.columns):
        return summary
    # Source pipeline may already bake in the 'derived' row — don't duplicate it.
    if "step" in summary.columns and (summary["step"] == "derived").any():
        return summary

    created = derivation_log
    if "schema_action" in created.columns:
        mask = created["schema_action"].astype(str).str.startswith("added ColSpec")
        created = created[mask]
    cols = [str(c) for c in created["derivation"].tolist() if str(c).strip()]
    if not cols:
        return summary

    new_row = {c: "" for c in summary.columns}
    if "step" in new_row:
        new_row["step"] = "derived"
    if "detail" in new_row:
        new_row["detail"] = f"added {len(cols)} column(s): " + ", ".join(cols)
    if "n_rows" in new_row and "step" in summary.columns:
        fin = summary[summary["step"] == "final"]
        if not fin.empty:
            new_row["n_rows"] = fin.iloc[0]["n_rows"]
    if "n_dropped" in new_row:
        new_row["n_dropped"] = 0

    new_df = pd.DataFrame([new_row], columns=summary.columns)
    # Place 'derived' just before the 'final' row when present.
    if "step" in summary.columns and (summary["step"] == "final").any():
        pos = summary.index.get_loc(summary.index[summary["step"] == "final"][0])
        return pd.concat(
            [summary.iloc[:pos], new_df, summary.iloc[pos:]], ignore_index=True)
    return pd.concat([summary, new_df], ignore_index=True)


def render_cleaning(cfg: ReportConfig, art: Artifacts) -> str:
    """🧹 Cleaning story."""
    blurb = ("The dataset was cleaned using a schema-driven process: declared "
             "null markers were applied, replacements were performed, data "
             "types were coerced, and skipped variables were excluded where "
             "appropriate.")
    body = [f'<p>{blurb}</p>']

    if (art.cleaning_summary is None and art.cleaning_log is None
            and art.derivation_log is None):
        body.append(warning_box(
            "No saved cleaning summary was found. Cleaning may have been "
            "performed, but no cleaning audit table was exported."))
        return section_block("🧹 Cleaning story", "".join(body))

    if art.cleaning_summary is not None and not art.cleaning_summary.empty:
        body.append("<h3>Summary</h3>")
        summary = _summary_with_derived_step(
            art.cleaning_summary, art.derivation_log)
        summary = _format_count_cols(
            summary, ["n_rows", "n_dropped", "n_rows_before"])
        body.append(table_to_html(summary))

    has_log = art.cleaning_log is not None and not art.cleaning_log.empty
    has_deriv = art.derivation_log is not None and not art.derivation_log.empty
    if has_log or has_deriv:
        log_html = (_render_cleaning_log_tables(art.cleaning_log) if has_log else "")
        if has_deriv:
            log_html += (
                "<h4>Derived variables</h4>"
                "<p>New columns computed from existing ones "
                "(binning, custom transforms).</p>"
                + table_to_html(art.derivation_log, max_rows=200)
            )
        body.append(details_block("📜 Full cleaning log", log_html))
    return section_block("🧹 Cleaning story", "".join(body))


def render_schema(cfg: ReportConfig, art: Artifacts) -> str:
    """🧬 Schema story with kind badges."""
    blurb = ("Variables were classified by analytical role. Continuous/count "
             "variables were treated numerically, ordinal variables preserved "
             "ordering, nominal variables were treated as unordered categories, "
             "and ID/text/skip variables were excluded from statistical "
             "screening where appropriate.")
    body = [f'<p>{blurb}</p>']

    if art.schema_summary is None or art.schema_summary.empty:
        body.append(warning_box("No schema artifact was found."))
        return section_block("🧬 Schema story", "".join(body))

    sch = art.schema_summary.copy()

    # Mark targets visually
    target_set = set(cfg.targets)
    if "column" in sch.columns:
        sch["role"] = sch["column"].apply(
            lambda c: "🎯 target" if c in target_set else "")

    # Kind -> emoji
    kind_emoji = {
        "continuous": "🔵 continuous", "count": "🔵 count",
        "ordinal": "🟣 ordinal", "binary": "🟢 binary",
        "nominal": "🟡 nominal", "datetime": "🕒 datetime",
        "id": "⚪ id", "text": "⚪ text", "skip": "⚪ skip",
    }
    if "kind" in sch.columns:
        sch["kind"] = sch["kind"].map(lambda k: kind_emoji.get(str(k), str(k)))

    # Collapse long level lists (> 10 values) behind a small expander.
    if "levels" in sch.columns:
        sch["levels"] = sch["levels"].map(_format_levels_cell)

    def _schema_row_cls(r) -> str:
        return "schema-skip" if str(r.get("keep")).strip().lower() in (
            "false", "0") else ""

    body.append(table_to_html(
        sch, max_rows=400, row_class_fn=_schema_row_cls,
        safe_html_cols=["levels"]))
    return section_block("🧬 Schema story", "".join(body))


def _format_levels_cell(v: Any) -> str:
    """Comma-join level lists; collapse behind an expander when > 10 values."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return ""
    levels = v
    if isinstance(v, str):
        try:
            levels = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return _esc(v)
    if not isinstance(levels, (list, tuple)):
        return _esc(str(v))
    text = ", ".join(str(x) for x in levels)
    if len(levels) > 10:
        preview = ", ".join(str(x) for x in levels[:3])
        return (f'<details class="collapsible"><summary>{_esc(preview)} … '
                f'({len(levels)} values)</summary>{_esc(text)}</details>')
    return _esc(text)


def _dda_continuous_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """Display copy of continuous DDA stats (max 2 decimal places)."""
    out = format_table_for_display(df)
    # ``mode`` is numeric here but classified as ``default`` (3 dp) in cleaning.
    if "mode" in df.columns and pd.api.types.is_numeric_dtype(df["mode"]):
        out["mode"] = df["mode"].map(
            lambda v: format_number(v, "central") if pd.notna(v) else ""
        )
    return out


def render_dda(cfg: ReportConfig, art: Artifacts) -> str:
    """📊 DDA story with per-kind subsections."""
    body = [
        '<p>This section summarizes each variable on its own, before any '
        'association testing. Tables describe distribution shape and balance; '
        'figures show the same information visually.</p>',
    ]

    # Dataset overview
    if art.dda_overall is not None and not art.dda_overall.empty:
        body.append("<h3>📦 Dataset overview</h3>")
        body.append(table_to_html(art.dda_overall))

    sections = [
        ("📏 Continuous / count variables",
         "Summarized using median, mean, trimmed mean, spread, skewness, "
         "kurtosis, outlier-sensitive quantiles, and missingness.",
         art.dda_continuous,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🏷️ Categorical / ordinal variables",
         "Summarized using dominant class, rarest class, class imbalance, "
         "Shannon entropy, and normalized balance.",
         art.dda_categorical,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("✅ Binary variables",
         "Same schema as categorical: dominant class, balance, missingness.",
         art.dda_binary,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🕒 Datetime variables",
         "Range, span in days, and missingness.",
         art.dda_datetime,
         lambda r: classify_missing(r.get("missing_pct"), cfg.missing)),
        ("🪪 ID / text variables",
         "Listed for completeness; excluded from statistical screening.",
         art.dda_id_text,
         None),
    ]
    for heading, blurb, tbl, row_fn in sections:
        body.append(f"<h3>{heading}</h3>")
        body.append(f"<p>{blurb}</p>")
        if tbl is None or tbl.empty:
            body.append('<p class="muted"><em>(no variables of this kind)</em></p>')
        else:
            display_tbl = (
                _dda_continuous_for_report(tbl)
                if tbl is art.dda_continuous
                else tbl
            )
            body.append(table_to_html(display_tbl, row_class_fn=row_fn))

    # Figures (collapsed by default; usually many)
    if art.dda_figures:
        grid_html = svg_grid(art.dda_figures)
        body.append(details_block(f"🖼️ DDA figures ({len(art.dda_figures)})",
                                  grid_html))

    # Glossary at the end so a clinician can look up any column they just saw.
    body.append(glossary_block(_dda_glossary()))

    return section_block("📊 Descriptive Data Analysis (DDA)", "".join(body))


# Every metric emitted across the DDA tables, grouped by where it appears.
_DDA_GLOSSARY_GROUPS = [
    ("Shared across tables", [
        ("column", "Variable name."),
        ("kind", "Variable type (continuous, count, ordinal, nominal, binary, "
                 "datetime, id, text)."),
        ("ordered", "Whether the categories have a meaningful order "
                    "(ordinal = yes, nominal = no)."),
        ("n", "Number of non-missing observations."),
        ("n_unique", "Number of distinct values."),
        ("missing_pct", "Percentage of missing values for this variable."),
    ]),
    ("Dataset overview", [
        ("n_rows", "Total rows (patients) in the dataset."),
        ("n_cols", "Total number of columns."),
        ("n_cols_analysed", "Columns included in statistical screening."),
        ("missing_cells_pct", "Percentage of all cells in the dataset that "
                              "are missing."),
    ]),
    ("Continuous / count variables", [
        ("min", "Smallest observed value."),
        ("max", "Largest observed value."),
        ("p_5th", "5th percentile — 5% of values fall below this."),
        ("p_95th", "95th percentile — 5% of values fall above this."),
        ("median", "Middle value (50th percentile); robust to outliers."),
        ("mean", "Arithmetic average."),
        ("trimmed_mean",
         "Average after removing the lowest 5% and highest 5% of values "
         "(10% symmetric trim). 📐 One giant tumor can't drag the number up — "
         "you still use most of the cohort. Compare to mean to see outlier pull."),
        ("mode", "Most frequent value."),
        ("std",
         "Typical distance from the average. 📏 Large std = values spread wide; "
         "near zero = everyone clustered together. Shows how much patients differ "
         "on this variable."),
        ("cv",
         "Relative spread: std ÷ |mean| (think of it as a %). 📊 Compares "
         "variability across columns with different units — e.g. age vs tumor "
         "volume. High cv = values bounce around a lot vs the typical level."),
        ("iqr", "Interquartile range (Q3 − Q1); middle-50% spread."),
        ("skewness",
         "Which way the long tail points. 0 ≈ symmetric. ➡️ Positive = a few "
         "very high values (mean often above median). ⬅️ Negative = a few very "
         "low values. Flags when the average is a poor 'typical patient' summary."),
        ("kurtosis",
         "How heavy the tails are vs a normal bell curve (0 = normal-like). "
         "📈 High = more extreme outliers than expected; low = flatter, fewer "
         "surprises. Early heads-up before choosing parametric vs non-parametric tests."),
    ]),
    ("Categorical / ordinal variables", [
        ("first_mode", "Most common category."),
        ("first_mode_pct", "Share of the most common category."),
        ("second_mode", "Second most common category."),
        ("second_mode_pct", "Share of the second most common category."),
        ("rarest", "Least common category."),
        ("rarest_pct", "Share of the least common category."),
        ("max_class_imbalance",
         "How lopsided categories are: most-common count ÷ rarest count. "
         "⚖️ 1 = perfectly even; 10 = the biggest group is 10× the smallest. "
         "High values warn that one category may dominate plots and models."),
        ("median_category", "Middle category by rank order (ordinal only)."),
        ("balance", "Normalized Shannon entropy (0–1). Closer to 1 = more "
                    "evenly distributed."),
        ("entropy_bin",
         "Shannon entropy in bits — how mixed the categories are. 🎲 0 = "
         "everyone in one bucket; higher = more categories contributing roughly "
         "evenly. Helps spot variables with rich variety vs near-constant labels."),
    ]),
    ("Binary variables", [
        ("mode", "More common of the two values."),
        ("mode_pct", "Share of the more common value."),
        ("rarest", "Less common of the two values."),
        ("rarest_pct", "Share of the less common value."),
    ]),
    ("Datetime variables", [
        ("min", "Earliest timestamp."),
        ("max", "Latest timestamp."),
        ("span_days", "Number of days between the earliest and latest value."),
    ]),
]


def _dda_glossary() -> str:
    parts: list[str] = []
    for title, items in _DDA_GLOSSARY_GROUPS:
        dt = "".join(f"<dt><code>{_esc(k)}</code></dt><dd>{_esc(v)}</dd>"
                     for k, v in items)
        parts.append(f"<h4>{_esc(title)}</h4><dl class=\"stat-decoder\">{dt}</dl>")
    return "".join(parts)


# Every metric shown in EDA association + diagnostic-accuracy tables.
_EDA_GLOSSARY_GROUPS = [
    ("Association table columns", [
        ("predictor",
         "The MRI or clinical variable being screened against the target."),
        ("kind",
         "Predictor type from the schema (continuous, count, binary, ordinal, "
         "nominal, etc.). The test is picked to match this type."),
        ("n_used",
         "Patients with both target and predictor recorded for this test. 📏 "
         "Low n_used = shakier result; missing data shrinks the sample."),
    ]),
    ("Statistical tests", [
        ("mann_whitney_u",
         "Compares a numeric predictor between outcome-present vs outcome-absent "
         "groups using ranks, not raw values. 🔬 Chosen over a t-test because "
         "tumor size, ADC, etc. are often skewed with outliers — ranks are more "
         "trustworthy."),
        ("mann_whitney_u_days",
         "Same Mann–Whitney idea applied to time gaps (days since earliest MRI "
         "date). Used when the predictor is a date."),
        ("spearman",
         "Measures whether higher values of one variable tend to pair with higher "
         "(or lower) values of another, using ranks. 📈 Chosen over Pearson "
         "correlation because it does not assume a straight-line relationship "
         "or a bell-curve shape."),
        ("chi2",
         "Tests whether category counts differ across outcome groups (e.g. skull "
         "base vs convex location by grade). 🗂️ Standard for larger tables. "
         "We switch to Fisher's exact when a 2×2 table has very small expected "
         "counts (< 5)."),
        ("fisher_exact",
         "Exact test for 2×2 category tables with sparse cells. 🎯 More reliable "
         "than χ² when few patients fall in a box — common for rare imaging "
         "findings."),
        ("kruskal_wallis",
         "Checks whether a numeric predictor differs across several outcome "
         "groups. 📊 Non-parametric alternative to one-way ANOVA — we use it "
         "when the outcome has 3+ categories and the predictor is not normally "
         "distributed."),
    ]),
    ("Effect sizes", [
        ("rank_biserial_r",
         "Signed rank difference between two groups (−1 to +1). ➡️ Positive = "
         "higher values in the outcome-present group; negative = lower. Shows "
         "direction and strength beyond the p-value."),
        ("spearman_rho",
         "Spearman correlation (−1 to +1). ➡️ Positive = both variables rise "
         "together; negative = one rises as the other falls. 0 ≈ no monotonic "
         "link."),
        ("cramers_v",
         "How tightly two categorical variables are linked (0 = none, 1 = perfect). "
         "🔗 Helps judge whether a location or margin category meaningfully "
         "tracks the outcome, not just whether χ² is significant."),
        ("epsilon_sq",
         "Kruskal–Wallis effect size — share of overall spread explained by "
         "group membership. 📐 Small ε² = groups overlap a lot; larger = clearer "
         "separation."),
        ("effect",
         "The numeric effect size for that row (one of the labels above). Separate "
         "from p: a big sample can make a tiny effect 'significant.'"),
        ("strength",
         "Badge classifying |effect| as strong / moderate / weak / none. 🚦 "
         "Quick visual on whether the signal is clinically worth a closer look, "
         "not just statistically detectable."),
    ]),
    ("P-values and significance", [
        ("p",
         "Nominal p-value — surprise if there were truly no association. ⚠️ "
         "Screening dozens of predictors inflates false positives; do not rely "
         "on raw p alone."),
        ("p_fdr",
         "FDR-adjusted p (Benjamini–Hochberg), recalculated within each target. "
         "🎲 Without correction, screening dozens of predictors at p<0.05 means "
         "you expect several false positives by luck alone — 'significant' rows "
         "that are not real associations. "
         "📐 Strict alternative (Bonferroni): divide α by the number of tests "
         "(e.g. 0.05 ÷ 30 ≈ 0.0017) or multiply each p by 30 — very conservative, "
         "but here it would often wipe out every hit and miss real MRI signals "
         "in an exploratory screen. "
         "🛡️ FDR is the middle path: among the predictors you flag, it caps "
         "the expected share that are false discoveries — shortlist candidates "
         "without treating every nominal p as trustworthy."),
        ("significance",
         "Row badge: 🟢 FDR-significant, 🟡 nominally significant only, ⚪ not "
         "significant. Green rows survived multiple-testing correction."),
        ("auc_univariate",
         "One-predictor discrimination for binary targets (0.5 = coin flip, "
         "1.0 = perfect). 🎯 Only for binary/continuous/count predictors. If "
         "raw AUC < 0.5 we report 1 − AUC so direction does not hide strength."),
    ]),
    ("Diagnostic accuracy table", [
        ("Sensitivity",
         "Among patients with the outcome, what share had the imaging feature "
         "present? 🔍 High = few false negatives. Low = the feature often misses "
         "cases."),
        ("Specificity",
         "Among patients without the outcome, what share did not have the feature? "
         "✅ High = few false positives. Low = the feature flags many healthy "
         "patients."),
        ("PPV",
         "Positive predictive value — if the feature is present, what share "
         "actually have the outcome? 🎯 Tells you how much to trust a positive "
         "finding (depends on how common the outcome is)."),
        ("NPV",
         "Negative predictive value — if the feature is absent, what share truly "
         "lack the outcome? 🛡️ High NPV = a negative finding is reassuring."),
        ("Accuracy",
         "Overall correct calls (present/absent vs outcome) as a %. 📊 Easy to "
         "read but misleading when the outcome is rare — check sensitivity and "
         "specificity too."),
        ("AUC",
         "In this table: (sensitivity + specificity) / 2 for a binary imaging "
         "feature. 📐 Simple univariate separation score (not full ROC). Matches "
         "the Upreti et al. Table 3 style."),
        ("Wilson 95% CI",
         "Confidence interval for each % in brackets. 📏 Narrow = precise; wide "
         "= few events or small sample — interpret cautiously."),
    ]),
]


def _eda_glossary() -> str:
    parts: list[str] = []
    for title, items in _EDA_GLOSSARY_GROUPS:
        dt = "".join(f"<dt><code>{_esc(k)}</code></dt><dd>{_esc(v)}</dd>"
                     for k, v in items)
        parts.append(f"<h4>{_esc(title)}</h4><dl class=\"stat-decoder\">{dt}</dl>")
    return "".join(parts)


# Missingness audit + formal MICE imputation (missingness_resolution.py).
_MICE_GLOSSARY_GROUPS = [
    ("Missingness audit", [
        ("missing_pct / pct_missing",
         "Share of empty cells per column. High values flag variables that may "
         "be unreliable in screening or models."),
        ("co-missingness (Jaccard)",
         "How often two columns are missing on the same patients (0 = never "
         "together, 1 = always together). Reveals structural gaps — e.g. ADC "
         "missing when DWI was not performed."),
    ]),
    ("Formal MICE settings (R mice)", [
        ("m",
         "Number of completed datasets (20 for publication; 3 for fast "
         "iteration). All m share one mice() chain so uncertainty is preserved."),
        ("max_iter (maxit)",
         "Fully-conditional-specification cycles through the incomplete "
         "variables until the chain stabilises (20 publication; 5 quick)."),
        ("seed",
         "Single RNG seed passed to mice() for reproducible draws."),
        ("method by variable type",
         "continuous/count → pmm, binary → logreg, nominal → polyreg, "
         "ordinal → polr. Each incomplete variable gets a model matched to its "
         "declared kind — recorded in methods.csv."),
        ("analysis_outcome",
         "The observed outcome (e.g. high_grade) predicts missing predictors "
         "but is never imputed; its source column is excluded as a duplicate. "
         "Not valid for deployment where the outcome is unknown."),
        ("derived_dependencies",
         "Parent→child map (e.g. multiple_meningiomas←meningioma_count). "
         "Non-outcome derived columns are dropped before R and recreated from "
         "their imputed sources to avoid contradictions."),
    ]),
    ("Imputation engine", [
        ("R mice via subprocess",
         "proper_mice_impute() writes input.csv + mice_spec.json, calls "
         "Rscript scripts/run_mice.R, then reloads the completed datasets. No "
         "rpy2; the notebook experience is unchanged."),
        ("predictor matrix",
         "Built explicitly in R (predictor_matrix.csv): row id, IDs, text, "
         "datetime, skipped, derived, and excluded columns are zeroed so they "
         "never silently drive imputations."),
        ("proper multiple imputation",
         "The manifest marks proper_multiple_imputation=True and "
         "rubin_pooling_supported=True only when the R run and all validation "
         "succeed — required before Rubin pooling."),
        ("dtype restoration + validation",
         "Each completed frame is recast to the original cohort (Categorical "
         "levels/order, nullable Float64/Int64/boolean), then checked for row "
         "identity, unchanged observed cells, filled missing cells, legal "
         "categories, and derived consistency. Pandera runs on every frame."),
        ("imputed_cell_variation.csv",
         "Per originally-missing cell, how the imputed value varies across the "
         "m draws (mean/sd or level counts). A diagnostic, not a confidence "
         "interval."),
    ]),
    ("Sensitivity method — RF chained (not formal MICE)", [
        ("rf_chained_impute()",
         "Random-forest IterativeImputer with post-hoc Bernoulli sampling for "
         "binary cells (legacy alias: mice_impute). Manifest marks "
         "proper_multiple_imputation=False — Rubin pooling is NOT supported."),
        ("joblib parallel / OS-aware",
         "RF draws run in parallel (loky), capped per OS/CPU and macOS "
         "battery. Used only for the optional sensitivity analysis."),
    ]),
    ("Python / R modules", [
        ("missingness_resolution.py",
         "Project module: missingness audit, formal R-mice orchestration, RF "
         "sensitivity, simple_impute screening, structural-missing handling."),
        ("R: mice / jsonlite",
         "mice runs the mixed-type FCS chain; jsonlite reads the spec. "
         "Install once: install.packages(c(\"mice\", \"jsonlite\"))."),
        ("pandas / numpy",
         "Exchange-file I/O, missing-cell counts, dtype restoration, "
         "cell-variation diagnostics."),
        ("scikit-learn / joblib",
         "IterativeImputer + RandomForestRegressor for the RF sensitivity "
         "method only (parallel via loky)."),
        ("matplotlib / seaborn",
         "Missing-% bar chart and co-missingness heatmap figures."),
        ("subprocess / hashlib / shutil (stdlib)",
         "Launch Rscript, hash the exchange input, manage the run directory."),
    ]),
]


def _missingness_glossary() -> str:
    parts: list[str] = []
    for title, items in _MICE_GLOSSARY_GROUPS:
        dt = "".join(f"<dt><code>{_esc(k)}</code></dt><dd>{_esc(v)}</dd>"
                     for k, v in items)
        parts.append(f"<h4>{_esc(title)}</h4><dl class=\"stat-decoder\">{dt}</dl>")
    return "".join(parts)


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
        body.append(glossary_block(_missingness_glossary()))
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

    body.append(glossary_block(_missingness_glossary()))

    return section_block("🕳️ Missingness story", "".join(body))


def render_eda(cfg: ReportConfig, art: Artifacts) -> str:
    """🔍 EDA story — per-target, color-coded, with badges."""
    body: list[str] = []
    if art.associations is None or art.associations.empty:
        body.append(warning_box("No EDA associations table was found."))
        body.append(glossary_block(_eda_glossary()))
        return section_block("🔍 Exploratory association screening (EDA)", "".join(body))

    body.append(
        '<p>Each predictor was screened against each target using a '
        'test matched to both outcome and predictor types (binary, continuous, '
        'ordinal, or nominal). '
        'p-values are corrected per target using Benjamini–Hochberg FDR.</p>'
    )

    df = art.associations.copy()
    # Ensure expected columns exist
    for col in ["target", "predictor", "kind", "test", "effect_label",
                "effect", "p", "p_fdr", "n_used", "auc_univariate"]:
        if col not in df.columns:
            df[col] = np.nan

    targets_in_data = list(df["target"].dropna().unique())
    # Render in the order user listed, then any extras
    order = [t for t in cfg.targets if t in targets_in_data]
    order += [t for t in targets_in_data if t not in order]

    for target in order:
        sub = df[df["target"] == target].copy()
        if sub.empty:
            continue

        body.append(f"<h3>🎯 Target: <code>{_esc(target)}</code></h3>")

        # Sort by FDR ascending, then |effect| descending
        sub["_p_num"] = sub["p_fdr"].apply(_coerce_p)
        sub["_eff_abs"] = sub["effect"].apply(
            lambda v: abs(_coerce_float(v)) if _coerce_float(v) is not None else -1)
        sub = sub.sort_values(["_p_num", "_eff_abs"],
                              ascending=[True, False], na_position="last")

        # Strength badge + significance row class
        def _row_class(r):
            tier = _strength_tier(r.get("effect"), "corr", cfg.effect)
            sig = classify_significance(
                r.get("p"), r.get("p_fdr"),
                fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)
            return sig  # row tint via significance only; strength via badge column

        sub["strength"] = sub["effect"].apply(
            lambda v: effect_badge(v, "corr", cfg.effect))
        sub["significance"] = sub.apply(
            lambda r: {"sig-fdr": "🟢 FDR-sig",
                       "sig-nominal": "🟡 nominal",
                       "sig-none": "⚪ ns"}[
                classify_significance(
                    r.get("p"), r.get("p_fdr"),
                    fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)],
            axis=1)

        # Mini summary line
        n_fdr = (sub["significance"] == "🟢 FDR-sig").sum()
        if n_fdr > 0:
            top = sub.iloc[0]
            line = (f"For target <code>{_esc(target)}</code>, "
                    f"<strong>{n_fdr}</strong> predictor"
                    f"{'s' if n_fdr > 1 else ''} survived FDR correction. "
                    f"Strongest exploratory association: "
                    f"<code>{_esc(top['predictor'])}</code> "
                    f"({_esc(top['effect_label'])} = {_esc(top['effect'])}, "
                    f"FDR p = {_esc(top['p_fdr'])}).")
        else:
            line = (f"No predictors survived FDR correction for "
                    f"<code>{_esc(target)}</code>. Any nominal findings below "
                    f"are exploratory only.")
        body.append(f"<p>{line}</p>")

        display_cols = ["predictor", "kind", "test", "effect_label", "effect",
                        "auc_univariate", "p", "p_fdr", "significance", "strength", "n_used"]
        display_cols = [c for c in display_cols if c in sub.columns]
        interp_df = sub.copy()
        if "auc_univariate" in sub.columns:
            sub["auc_univariate"] = sub["auc_univariate"].apply(
                lambda v: "" if _coerce_float(v) is None else f"{_coerce_float(v):.3f}",
            )
        sub["p"] = sub["p"].apply(human_p)
        sub["p_fdr"] = sub["p_fdr"].apply(human_p)
        body.append(table_to_html(
            sub[display_cols], row_class_fn=_row_class,
            # 'strength' is a pre-built <span> badge — don't HTML-escape it.
            safe_html_cols=("strength",),
        ))
        body.append(_render_diagnostic_accuracy(target, art, cfg))

        body.append(_render_eda_interpretation(target, interp_df, cfg))

        # Figures for this target
        figs = [p for p in art.eda_figures if p.stem.startswith(f"{target}__")]
        if figs:
            body.append(details_block(
                f"🖼️ EDA figures for {target} ({len(figs)})",
                svg_grid(figs)))

    body.append(glossary_block(_eda_glossary()))

    return section_block("🔍 Exploratory association screening (EDA)", "".join(body))


# Every metric shown in multivariable tables, EPV gauge, VIF, and forest plots.
_INFERENTIAL_GLOSSARY_GROUPS = [
    ("Sample size & power", [
        ("epv",
         "Events per variable: outcome events ÷ model parameters. 📏 Rule of thumb "
         "≥ 10 = stable; 5–10 = borderline; < 5 = underpowered. Flags when too many "
         "predictors chase too few high-grade cases."),
        ("n_complete_cases",
         "Patients available to the model. 🧩 After formal MICE every retained "
         "predictor is filled, so this is effectively the full cohort; the "
         "complete-case preview (pre-imputation) is shown only as a stability "
         "sanity check."),
        ("n_outcome_events",
         "Count of positive outcomes among those patients. 🎯 Drives EPV and how "
         "trustworthy each adjusted OR is."),
        ("n_design_columns",
         "Predictor columns in the final design matrix after encoding and VIF pruning. "
         "📐 Nominal variables expand to several columns; each counts toward EPV."),
    ]),
    ("Adjusted OR table", [
        ("predictor_col",
         "Predictor name in the fitted model (one-hot levels show as "
         "variable__category)."),
        ("risk",
         "% with the outcome. Ex: 20 high-grade / 100 patients → 20% risk. 🏥"),
        ("odds",
         "Cases ÷ non-cases. Ex: 20 high-grade / 80 not → 0.25. 🎲 Models use "
         "odds, not %."),
        ("or",
         "Odds in one group ÷ odds in another (others held fixed). Ex: edema "
         "12/18 = 0.67 vs no edema 8/62 = 0.13 → OR ≈ 5. 🔢 OR 2 = odds double; "
         "not always risk double when outcome is common."),
        ("or_ci_lo / or_ci_hi",
         "95% confidence interval for the OR. 📏 CI entirely above 1 → likely risk "
         "factor; entirely below 1 → likely protective; crossing 1 → not clearly "
         "different from no effect."),
        ("coef",
         "Log-odds coefficient before exponentiating. Used internally; OR is the "
         "clinician-facing scale."),
        ("se",
         "Standard error of the pooled coefficient. Smaller = more precise estimate."),
        ("p",
         "Two-sided p-value from the pooled estimate (Rubin + Barnard–Rubin df). ⚠️ "
         "Use with the CI — a small p with a wide CI still means uncertainty."),
        ("df",
         "Pooled degrees of freedom for the t-test (Barnard–Rubin). Corrects for "
         "multiple imputation; large values shown as ∞."),
        ("n_models",
         "Imputed datasets where this coefficient converged. Lower than m = 10 means "
         "some fits failed — interpret that row cautiously."),
        ("model_id",
         "Short label for this literature or custom predictor-set variant."),
    ]),
    ("Collinearity (VIF)", [
        ("vif",
         "Variance inflation factor = 1 / (1 − R²). 🔗 VIF > 5 means the predictor "
         "overlaps heavily with others (e.g. necrosis + heterogeneous enhancement). "
         "Highest-VIF columns are dropped iteratively until all ≤ 5 — clearer ORs than "
         "keeping redundant MRI signs."),
        ("predictor",
         "Design-matrix column checked for VIF (in the VIF diagnostics dropdown)."),
    ]),
    ("Forest plot", [
        ("Adjusted OR (log scale)",
         "Dot = pooled OR; whiskers = 95% CI. 📊 Log axis keeps large and small ORs "
         "readable on one chart. Vertical dashed line at OR = 1 is the null — no "
         "independent association."),
    ]),
    ("Encoding & pooling methods", [
        ("z-score",
         "Continuous/count predictors rescaled to (x − mean) / SD. 📐 OR becomes 'per "
         "1 SD increase' — comparable across tumor volume, ADC, etc. Raw units would "
         "mix incomparable scales on one forest plot."),
        ("one-hot (drop-first)",
         "Nominal categories → 0/1 columns vs a reference level. 🗂️ Avoids treating "
         "location labels as ordered numbers."),
        ("ordinal codes",
         "Ordered categories kept as numeric ranks (not one-hot). Preserves "
         "grade-like ordering without exploding column count."),
        ("logistic regression (Logit)",
         "Binary outcome model: linear combination of predictors → probability via "
         "logistic curve. 📈 Standard for adjusted ORs in clinical papers — transparent "
         "vs black-box ML that hides independent effects."),
        ("formal MICE + Rubin pooling",
         "Model fit on each of m datasets from one mixed-type mice() chain "
         "(proper_mice_impute); coefficients averaged and SEs combined (within "
         "+ between imputation variance). 🎲 Valid inference with missing data — "
         "single imputation or complete-case-only would ignore uncertainty or "
         "drop patients."),
        ("Barnard–Rubin df",
         "Small-sample correction for pooled p-values and CIs when m is modest. "
         "Preferable to a plain z-test after MI, which can be anti-conservative."),
        ("binary imaging signs (logreg)",
         "Missing binary MRI signs are imputed by logistic regression within the "
         "MICE chain (MAR conditional on the other predictors), so patients are "
         "not dropped. 🛡️ Imputation uncertainty propagates through Rubin pooling; "
         "if missingness is informative (MNAR), add a sensitivity analysis."),
    ]),
]


def _inferential_glossary() -> str:
    parts: list[str] = []
    for title, items in _INFERENTIAL_GLOSSARY_GROUPS:
        dt = "".join(f"<dt><code>{_esc(k)}</code></dt><dd>{_esc(v)}</dd>"
                     for k, v in items)
        parts.append(f"<h4>{_esc(title)}</h4><dl class=\"stat-decoder\">{dt}</dl>")
    return "".join(parts)


def render_inferential(cfg: ReportConfig, art: Artifacts) -> str:
    """🧮 Multivariable / inferential modelling."""
    body: list[str] = []
    if not art.inferential_multivariable and (art.inferential_summary is None
                                               or art.inferential_summary.empty):
        body.append(warning_box("No multivariable model artifacts were found."))
        body.append(glossary_block(_inferential_glossary()))
        return section_block("🧮 Multivariable modelling", "".join(body))

    body.append(
        '<p>Multivariable logistic regression was fitted for each target and '
        'each predictor-set variant you defined. Predictors were encoded '
        'according to schema type, continuous/count variables were standardized, '
        'nominal variables were one-hot encoded, and high-VIF predictors were '
        'pruned. Estimates were pooled across the formal mixed-type MICE '
        'datasets with Rubin\u2019s rules.</p>'
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

    body.append(glossary_block(_inferential_glossary()))

    return section_block("🧮 Multivariable modelling", "".join(body))


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


_DIAGNOSTIC_LABELS: dict[str, str] = {
    "perifocal_edema": "Peritumoral Edema",
    "sinus_invasion": "Skull Invasion",
    "tumor_location_non_skull_base": "Non-skull-base Location",
    "tumor_margin_irregular": "Irregular Tumor Margin",
    "sex_male": "Male Sex",
    "cortical_destruction": "Cortical Destruction",
    "heterogeneous_enhancement": "Heterogeneous Enhancement",
    "cystic_component": "Cystic Component",
    "mass_effect": "Mass Effect",
    "dural_tail": "Dural Tail",
    "capsular_enhancement": "Capsular Enhancement",
    "brain_invasion": "Brain Invasion",
    "t1_hypointensity": "T1 Hypointensity",
    "t2_hyperintensity": "T2 Hyperintensity",
    "dwi_hyperintensity": "DWI Hyperintensity",
    "transfalcine_extension": "Transfalcine Extension",
    "hist_necrosis": "Histologic Necrosis",
    "progesterone_pos": "Progesterone Positive",
}


def _diagnostic_predictor_label(name: str) -> str:
    if name in _DIAGNOSTIC_LABELS:
        return _DIAGNOSTIC_LABELS[name]
    return " ".join(part.capitalize() for part in str(name).split("_"))


def _format_pct_ci(point: Any, lo: Any, hi: Any) -> str:
    """Paper-style ``70.1% [62.2–77.2]``."""
    p = _coerce_float(point)
    if p is None:
        return ""
    pct = p * 100.0
    lo_f = _coerce_float(lo)
    hi_f = _coerce_float(hi)
    if lo_f is None or hi_f is None:
        return f"{pct:.1f}%"
    return f"{pct:.1f}% [{lo_f * 100.0:.1f}–{hi_f * 100.0:.1f}]"


def _format_diagnostic_auc(v: Any) -> str:
    f = _coerce_float(v)
    if f is None:
        return ""
    return f"{f:.3f}"


def _build_diagnostic_display_df(sub: pd.DataFrame, cfg: ReportConfig) -> pd.DataFrame:
    """Paper-style Table 3 layout (Upreti et al.)."""
    evaluated = sub[sub["note"].fillna("").eq("")].copy()
    if evaluated.empty:
        return pd.DataFrame()

    evaluated["_p_num"] = evaluated["p_fdr"].apply(_coerce_p)
    evaluated = evaluated.sort_values(
        ["_p_num", "sensitivity"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns="_p_num")

    rows = []
    for _, r in evaluated.iterrows():
        sig = classify_significance(
            r.get("p"), r.get("p_fdr"),
            fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha,
        )
        label = _diagnostic_predictor_label(str(r["predictor"]))
        if sig == "sig-fdr":
            label = f"{label}*"
        rows.append({
            "Imaging feature": label,
            "Sensitivity (95% CI)": _format_pct_ci(
                r.get("sensitivity"), r.get("sensitivity_lo"), r.get("sensitivity_hi")),
            "Specificity (95% CI)": _format_pct_ci(
                r.get("specificity"), r.get("specificity_lo"), r.get("specificity_hi")),
            "PPV (95% CI)": _format_pct_ci(
                r.get("PPV"), r.get("PPV_lo"), r.get("PPV_hi")),
            "NPV (95% CI)": _format_pct_ci(
                r.get("NPV"), r.get("NPV_lo"), r.get("NPV_hi")),
            "Accuracy": _format_pct_ci(
                r.get("accuracy"), r.get("accuracy_lo"), r.get("accuracy_hi")),
            "AUC": _format_diagnostic_auc(r.get("AUC")),
            "_sig": sig,
            "_p": r.get("p"),
            "_p_fdr": r.get("p_fdr"),
            "note": r.get("note", ""),
        })
    return pd.DataFrame(rows)


def _diagnostic_table_html(disp: pd.DataFrame, cfg: ReportConfig) -> str:
    if disp.empty:
        return '<p class="muted"><em>(no evaluable binary predictors)</em></p>'

    show = disp.drop(columns=[c for c in disp.columns if c.startswith("_")], errors="ignore")
    sig_classes = disp["_sig"].astype(str).tolist()
    counter = [0]

    def _class_fn(_r: pd.Series) -> str:
        i = counter[0]
        counter[0] += 1
        return sig_classes[i] if i < len(sig_classes) else ""

    html = table_to_html(show, row_class_fn=_class_fn)
    return html.replace(
        '<table class="report">',
        '<table class="report diagnostic-accuracy">',
        1,
    )


def _render_diagnostic_accuracy(target: str, art: Artifacts,
                                cfg: ReportConfig) -> str:
    """Collapsible univariate diagnostic accuracy table (one target)."""
    if art.diagnostic_accuracy is None or art.diagnostic_accuracy.empty:
        return details_block(
            "Like in that research: univariate diagnostic accuracy",
            warning_box("No diagnostic accuracy table was found."),
        )

    sub = art.diagnostic_accuracy[
        art.diagnostic_accuracy["target"].astype(str) == str(target)
    ].copy()
    if sub.empty:
        return details_block(
            "Like in that research: univariate diagnostic accuracy",
            warning_box(
                f"No diagnostic accuracy rows for target <code>{_esc(target)}</code>."
            ),
        )

    disp = _build_diagnostic_display_df(sub, cfg)
    skipped = sub[sub["note"].fillna("").ne("")]

    parts = [
        "<p>Univariate diagnostic-test performance of each binary imaging feature "
        "(present vs absent) against the target outcome, formatted like "
        "Upreti et al. Table 3. Wilson 95% confidence intervals are shown in "
        "brackets. Rows are colour-coded by significance "
        "(<span class='badge' style='background:var(--green-bg)'>green</span> "
        "= FDR-significant, "
        "<span class='badge' style='background:var(--yellow-bg)'>yellow</span> "
        "= nominally significant only). "
        "An asterisk (*) marks FDR-significant features.</p>",
        _diagnostic_table_html(disp, cfg),
        (
            "<p class='muted'><small>"
            "* imaging feature with FDR p &lt; "
            f"{cfg.fdr_alpha:g}. "
            "PPV = positive predictive value; NPV = negative predictive value; "
            "CI = confidence interval; AUC = (sensitivity + specificity) / 2 "
            "for a binary predictor.</small></p>"
        ),
    ]
    if not skipped.empty and "predictor" in skipped.columns:
        skip_lines = "".join(
            f"<li><code>{_esc(r['predictor'])}</code>: "
            f"{_esc(r.get('note', ''))}</li>"
            for _, r in skipped.iterrows()
        )
        parts.append(
            "<p><strong>Skipped predictors</strong></p>"
            f"<ul class='muted'>{skip_lines}</ul>"
        )

    return details_block(
        "Like in that research: univariate diagnostic accuracy", "".join(parts))


def _eda_direction_phrase(r: pd.Series, target: str) -> str:
    """Plain-language wording for one EDA association row (Spearman sign from rho)."""
    pred = _esc(r.get("predictor"))
    test = str(r.get("test") or "")
    eff = _coerce_float(r.get("effect"))
    if test == "spearman" and eff is not None:
        if eff > 0:
            return (f"Higher <code>{pred}</code> is associated with a higher rate of "
                    f"<code>{_esc(target)}</code>")
        if eff < 0:
            return (f"Higher <code>{pred}</code> is associated with a lower rate of "
                    f"<code>{_esc(target)}</code>")
    return (f"<code>{pred}</code> is associated with "
            f"<code>{_esc(target)}</code> (see figure for group differences)")


def _render_eda_interpretation(target: str, sub: pd.DataFrame,
                               cfg: ReportConfig) -> str:
    """Plain-English bullets for univariate EDA rows (one target)."""
    lines: list[str] = []
    for _, r in sub.iterrows():
        test = str(r.get("test") or "")
        if test == "skip":
            continue
        pred = _esc(r.get("predictor"))
        eff_label = _esc(r.get("effect_label"))
        eff = _esc(r.get("effect"))
        sig = classify_significance(
            r.get("p"), r.get("p_fdr"),
            fdr_alpha=cfg.fdr_alpha, nominal_alpha=cfg.nominal_alpha)
        tier = _strength_tier(r.get("effect"), "corr", cfg.effect)
        _, strength_word, strength_hint = _STRENGTH_WORDING[tier]
        p_fdr_str = _esc(human_p(r.get("p_fdr")))
        p_str = _esc(human_p(r.get("p")))
        phrase = _eda_direction_phrase(r, target)

        if sig == "sig-fdr":
            sig_note = f"FDR p = {p_fdr_str}"
            bullet = "🟢"
        elif sig == "sig-nominal":
            sig_note = f"nominal p = {p_str}; FDR p = {p_fdr_str} (exploratory only)"
            bullet = "🟡"
        else:
            sig_note = f"FDR p = {p_fdr_str}"
            bullet = "⚪"

        if sig == "sig-none":
            lines.append(
                f"<li>{bullet} <code>{pred}</code>: no clear marginal association "
                f"({sig_note}; {eff_label} = {eff}).</li>")
            continue

        lines.append(
            f"<li>{bullet} {phrase} "
            f"({eff_label} = {eff}, {sig_note}; <em>{strength_word}</em> effect). "
            f"{strength_hint}</li>")

    if not lines:
        return ""
    caveat = (
        "<li><em>Univariate screening only — effects are not adjusted for other "
        "predictors. Use the multivariable section to judge independent "
        "associations.</em></li>"
    )
    return details_block(
        "💡 Interpretation", "<ul>" + "".join(lines) + caveat + "</ul>")


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
    return Path(__file__).resolve().parent.parent


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


def _distribution_name(import_name: str) -> str:
    """Map a top-level import name to its PyPI distribution name."""
    mapped = importlib.metadata.packages_distributions().get(import_name)
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
        render_schema(cfg, art),
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
