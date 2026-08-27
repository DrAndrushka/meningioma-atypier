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
    predictor_label,
)

from cleaning import (
    collapse_coercion_audit_rows,
    format_number,
    format_table_for_display,
)
from plot_style import prettify_caption, prettify_label, read_figure_legend


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
    --bg: #F7DDB8;          /* warm golden-apricot page ground; tables and figures stay white */
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

/* Headline dashboard — one tile per number, grouped by colour band. */
.dash { display: grid; gap: 10px; margin: 16px 0 6px;
        grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }
.stat { position: relative; display: flex; flex-direction: column; gap: 3px;
        min-height: 108px; padding: 15px 14px 13px; background: #fff;
        border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.stat::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
                background: var(--tint, var(--accent)); }
.stat .ico { font-size: 15px; line-height: 1; }
.stat .k { font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em;
           text-transform: uppercase; color: var(--muted); }
.stat .v { margin-top: auto; font-size: 25px; font-weight: 700; line-height: 1.1;
           font-variant-numeric: tabular-nums; word-break: break-word; }
.stat .v.sm { font-size: 14.5px; font-weight: 600; line-height: 1.35; }
/* The "before cleaning" figure rides with its own number, not in a second tile. */
.stat .was { display: inline-block; margin-left: 6px; padding: 1px 7px;
             border-radius: 999px; background: var(--grey-bg); color: var(--muted);
             font-size: 11px; font-weight: 600; vertical-align: middle;
             font-variant-numeric: tabular-nums; }
.stat .tgt { display: inline-block; margin: 2px 4px 0 0; padding: 2px 8px;
             border-radius: 999px; background: var(--blue-bg); color: var(--blue);
             font-size: 12px; font-weight: 600; }

/* Tables */
.table-wrap { width: 100%; overflow-x: auto; margin: 8px 0 14px;
              background: #fff; border-radius: 6px; }
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
/* Legend as text, not pixels: title over the figure, the two explanations
   under it — plain words first, then the journal's Note:—. */
.figure-card .figure-title { font-size: 13.5px; font-weight: 600; color: var(--fg);
                             line-height: 1.4; margin: 2px 0 8px; }
.figure-card .figure-plain { font-size: 13px; color: var(--fg);
                             line-height: 1.55; margin: 10px 0 0; }
.figure-card .figure-note { font-size: 12px; color: var(--muted);
                            line-height: 1.55; margin: 8px 0 0;
                            padding-top: 8px; border-top: 1px solid var(--border); }
/* The Methods paragraph for the two overview figures — boxed, so it reads as
   prose belonging to the manuscript rather than as another figure caption. */
.method-statement { border: 1px solid var(--border); border-radius: 6px;
                    padding: 12px 14px; margin: 14px 0 18px; }
.method-statement-title { font-size: 12px; font-weight: 600;
                          letter-spacing: 0.02em; margin-bottom: 6px; }
.method-statement p { font-size: 13px; color: var(--fg); line-height: 1.55;
                      margin: 0; }
/* A batch of same-shaped figures is explained once, above the grid. */
.grid-plain { font-size: 13px; color: var(--fg); line-height: 1.55;
              margin: 6px 0 10px; }

/* One model reads top to bottom as numbered steps, not nested dropdowns. */
.model-group { font-size: 17px; margin: 30px 0 10px; }
.model-step { margin: 0 0 20px; }
.model-step > h5 { font-size: 14px; font-weight: 600; color: var(--fg);
                   margin: 20px 0 8px; display: flex; align-items: center;
                   gap: 8px; }
.model-step:first-child > h5 { margin-top: 4px; }
.step-n { display: inline-flex; align-items: center; justify-content: center;
          width: 20px; height: 20px; border-radius: 50%; flex: 0 0 auto;
          background: var(--accent); color: #fff; font-size: 11px;
          font-weight: 700; }
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


def figure_card(path: Path, *, caption: str | None = None,
                extra_class: str = "") -> str:
    """One figure with its legend as text — title above it, note below it.

    Both are read from the figure's ``.legend.json`` sidecar rather than drawn
    into the image, so they stay selectable, searchable, and re-wrap to the
    reader's window instead of being fixed pixels at one width.

    A figure with no sidecar keeps the name-derived grey caption it has always
    had, so this degrades to the old behaviour rather than to a blank card.
    """
    img = _figure_img_html(path)
    if not img:
        return ""
    legend = read_figure_legend(path)
    title = legend.get("title", "")
    plain, note = legend.get("plain", ""), legend.get("note", "")
    head = f'<div class="figure-title">{_esc(title)}</div>' if title else ""
    foot = ""
    if not title:
        text = caption if caption is not None else prettify_caption(path.stem)
        if text:                      # caption="" means the caller wants none
            foot = f'<div class="caption">{_esc(text)}</div>'
    # Plain words first, then the journal's Note:— — the reader meets what the
    # picture shows before the vocabulary a reviewer needs.
    if plain:
        foot += f'<p class="figure-plain">{_esc(plain)}</p>'
    if note:
        foot += f'<p class="figure-note">{_esc(note)}</p>'
    cls = ("figure-card " + extra_class).strip()
    return f'<div class="{cls}">{head}{img}{foot}</div>'


def svg_grid(svg_paths: Iterable[Path], max_n: int | None = None,
             *, plain: str = "") -> str:
    """An HTML grid of figures embedded as base64 data URIs.

    ``plain`` is the one-or-two-sentence "how to read these" line, printed once
    above the grid rather than repeated under every tile. A batch of thirty
    figures drawn the same way needs the explanation once; repeating it thirty
    times would bury the figures in boilerplate.
    """
    paths = [p for p in svg_paths if p.exists()]
    if max_n is not None:
        paths = paths[:max_n]
    if not paths:
        return '<p class="muted"><em>(no figures available)</em></p>'
    cards = [c for c in (figure_card(p) for p in paths) if c]
    if not cards:
        return '<p class="muted"><em>(no figures available)</em></p>'
    intro = f'<p class="grid-plain">{_esc(plain)}</p>' if plain else ""
    return f'{intro}<div class="figure-grid">{"".join(cards)}</div>'


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
    model_vs_single: pd.DataFrame | None = None
    model_overview: pd.DataFrame | None = None
    single_reference: pd.DataFrame | None = None
    top_selection: pd.DataFrame | None = None

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
    # The cleaning phase writes the per-column counts under tables/ as
    # missing_per_column.csv; older runs put the same file flat in
    # missingness/. Both are tried, or the section reports no artifacts while
    # the file sits on disk one directory away.
    for fallback in (miss_tab / "missing_per_column.csv",
                     root / "missingness" / "missing_per_column.csv"):
        if art.missingness_summary is not None:
            break
        art.missingness_summary = _maybe_read_csv(fallback, art.warnings)
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
    art.model_vs_single = _maybe_read_csv(inf_tab / "model_vs_single_auc.csv", art.warnings)
    art.model_overview = _maybe_read_csv(inf_tab / "model_overview.csv", art.warnings)
    art.single_reference = _maybe_read_csv(inf_tab / "single_predictor_reference.csv", art.warnings)
    art.top_selection = _maybe_read_csv(inf_tab / "top_variable_selection.csv", art.warnings)

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


def _model_steps(steps: Sequence[tuple[str, str]]) -> str:
    """Lay one model out as numbered steps rather than nested dropdowns.

    Only steps that actually have content are numbered, so a model with no
    published source or no collinearity table reads 1-2-3 rather than leaving
    gaps where a reader would look for something missing.
    """
    out: list[str] = []
    n = 0
    for heading, inner in steps:
        if not inner:
            continue
        n += 1
        out.append(
            f'<div class="model-step"><h5><span class="step-n">{n}</span>'
            f'{_esc(heading)}</h5>{inner}</div>'
        )
    return "".join(out)


def _published_or(term: dict) -> str:
    """``2.94 (1.15–7.48)``, or as much of it as the paper printed."""
    o, lo, hi = (_coerce_float(term.get(k)) for k in ("or", "ci_lo", "ci_hi"))
    if o is None:
        return ""
    if lo is None or hi is None:
        return f"{o:.2f}"
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


def _beta_se(coef: Any, se: Any) -> str:
    """``0.96 (0.37)`` — the log-odds coefficient with its standard error."""
    c, s = _coerce_float(coef), _coerce_float(se)
    if c is None:
        return ""
    return f"{c:.2f} ({s:.2f})" if s is not None else f"{c:.2f}"


def _or_ci(o: Any, lo: Any, hi: Any) -> str:
    """``2.60 (1.26–5.38)`` — same shape as the published table above it."""
    ov, l, h = (_coerce_float(x) for x in (o, lo, hi))
    if ov is None:
        return ""
    return f"{ov:.2f} ({l:.2f}–{h:.2f})" if l is not None and h is not None else f"{ov:.2f}"


def _model_level_line(tbl: pd.DataFrame) -> str:
    """Facts that were repeated on every row, stated once.

    The intercept, the imputation count and the pooled degrees of freedom are
    properties of the model, not of any predictor. As columns they cost seven
    cells per row and told the reader nothing new after the first one.
    """
    bits: list[str] = []
    if "intercept_coef" in tbl.columns and len(tbl):
        ic = _coerce_float(tbl["intercept_coef"].iloc[0])
        io = _coerce_float(tbl["intercept_or"].iloc[0]) if "intercept_or" in tbl.columns else None
        if ic is not None:
            bits.append(f"Intercept {ic:.2f}" + (f" (OR {io:.3f})" if io is not None else ""))
    if "n_models" in tbl.columns and len(tbl):
        n = _to_int_or_none(tbl["n_models"].iloc[0])
        if n:
            bits.append(f"pooled across {n} imputations")
    if "df" in tbl.columns and len(tbl):
        dfs = {human_pool_df(v) for v in tbl["df"] if str(v).strip()}
        if len(dfs) == 1:
            bits.append(f"Rubin df {dfs.pop()}")
    return f'<p class="muted">{" &middot; ".join(_esc(b) for b in bits)}</p>' if bits else ""


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

    rows = [
        {
            "Variable in the paper": t.get("variable", ""),
            "What it means": t.get("meaning", ""),
            "Published aOR (95% CI)": _published_or(t),
            "Published β": t.get("beta", ""),
            "p": human_p(t.get("p")),
            "Column used here": t.get("column", ""),
        }
        for t in terms
    ]
    # A β column is only informative when at least one term in this model
    # actually published one (Zhang 2020) — otherwise every cell is blank
    # and the column is pure noise next to the OR column.
    if not any(r["Published β"] not in (None, "") for r in rows):
        for r in rows:
            del r["Published β"]

    table = table_to_html(pd.DataFrame(rows), nowrap_cols=("Published aOR (95% CI)", "p"))

    note = str(published.get("surrogate_note") or "").strip()
    surrogate_html = warning_box(note) if note else ""

    # A note about how this paper sits against ANOTHER paper in the same set.
    # Rendered as information rather than a warning: nothing here questions the
    # refit's validity, it flags that two correctly-transcribed papers point
    # opposite ways on one variable, which a reader comparing their tables
    # would otherwise have to notice unaided.
    cross = str(published.get("cross_reference") or "").strip()
    cross_html = (f'<p class="muted"><strong>Against another paper here:</strong> '
                  f'{_esc(cross)}</p>') if cross else ""

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

    return surrogate_html + table + note_html + cross_html


def _selection_n_resamples(art: Artifacts) -> int | None:
    """The variable-selection audit's own resample count, read from
    ``top_variable_selection.csv``'s own ``resample_selection_total`` column
    — never ``model_vs_single_auc.csv``'s ``n_resamples``.

    These are two different bootstrap loops with two different drop rules —
    patient resamples that kept both outcome classes, versus selection
    resamples whose selector returned a non-empty pick — that happen to both
    equal ``analysis.BOOTSTRAP_RESAMPLES`` today. Sourcing one block's
    denominator from the other table is exactly the "two numbers, one name"
    pattern that has already caused separate defects on this branch, so this
    reads its own column even though the values agree right now. Returns
    ``None`` if the column is missing, empty, or does not agree on a single
    value (e.g. no model in this run was data-selected).
    """
    tbl = art.top_selection
    if tbl is None or tbl.empty or "resample_selection_total" not in tbl.columns:
        return None
    vals = {v for v in (_to_int_or_none(x) for x in tbl["resample_selection_total"])
            if v is not None}
    return vals.pop() if len(vals) == 1 else None


def _fmt3(x: Any) -> str:
    """Fixed three-decimal formatting (never drops trailing zeros).

    Unlike ``format_number`` — which trims trailing zeros and collapses an
    exact 0.000 to the bare integer ``0`` — this keeps a column's precision
    uniform so it reads as one measurement scale next to a neighbouring
    column, and keeps an exact zero legible as a value rather than looking
    like a blank cell.
    """
    v = _coerce_float(x)
    return "" if v is None else f"{v:.3f}"


def _model_vs_single_block(model_id: str, art: Artifacts) -> str:
    """Does this combination beat each single predictor it is built from?

    Two different quantities sit side by side here on purpose. ``ΔAUC
    corrected`` is the optimism-corrected point estimate — both AUCs it is
    built from already account for the model having seen its own data. The
    confidence interval belongs to a *different* column, ``ΔAUC
    apparent`` — the raw, uncorrected gap, resampled across patients to see
    how far it moves by chance. Pairing the corrected delta with this
    interval would silently misstate its uncertainty, so the columns are
    named and separated to make that pairing impossible to make by accident.
    """
    tbl = art.model_vs_single
    if tbl is None or tbl.empty or "model_id" not in tbl.columns:
        return ""
    sub = tbl[tbl["model_id"].astype(str) == str(model_id)]
    if sub.empty:
        return ""

    def _apparent_ci(r: pd.Series) -> str:
        app = _coerce_float(r.get("delta_auc_apparent"))
        lo = _coerce_float(r.get("delta_ci_lo"))
        hi = _coerce_float(r.get("delta_ci_hi"))
        if app is None or lo is None or hi is None:
            return ""
        return f"{app:.3f} ({lo:.3f}–{hi:.3f})"

    rows = pd.DataFrame([{
        "Single predictor": r.get("single", ""),
        "Model AUC (corrected)": format_number(r.get("auc_model_corrected")),
        "Single AUC (corrected)": _fmt3(r.get("auc_single_corrected")),
        "ΔAUC corrected": _fmt3(r.get("delta_auc_corrected")),
        "ΔAUC apparent (95% CI)": _apparent_ci(r),
        "p (D2)": human_p(r.get("d2_p")),
    } for _, r in sub.iterrows()])


    return table_to_html(rows, nowrap_cols=(
        "Model AUC (corrected)", "Single AUC (corrected)",
        "ΔAUC corrected", "ΔAUC apparent (95% CI)", "p (D2)"))


def _method_statement_block(art: Artifacts) -> str:
    """The Methods paragraph for the two overview figures, written from the run.

    Sits with the figures rather than in a distant Methods file because it is
    the sentence a reader needs at the moment they ask "what was compared
    against what?" — and because a paragraph typed by hand beside numbers
    generated by code is a paragraph that goes stale on the next run.

    Every value here is read back from what the pipeline actually did: the VIF
    threshold from config, the drops and their strongest correlates from the
    selection audit, the resample count from config.
    """
    from heavy_machinery.config import load as _load
    try:
        analysis = _load("analysis")
        vif_max = float(analysis.SELECTION_VIF_MAX)
        resamples = int(analysis.BOOTSTRAP_RESAMPLES)
    except Exception:  # pragma: no cover - config absent
        return ""
    tbl = art.top_selection
    drops: list[str] = []
    if tbl is not None and not tbl.empty and "reason" in tbl.columns:
        for _, r in tbl.iterrows():
            reason = str(r.get("reason") or "")
            if not reason.startswith("VIF="):
                continue
            name = prettify_label(str(r.get("variable") or ""))
            vif = _coerce_float(r.get("vif"))
            partner = str(r.get("partner") or "").strip()
            rho = _coerce_float(r.get("rho"))
            bits = [] if vif is None else [f"VIF {vif:.1f}"]
            if partner and rho is not None:
                bits.append(f"rho {rho:.2f} with {prettify_label(partner)}")
            drops.append(f"{name} ({'; '.join(bits)})" if bits else name)
    removed = (f", which removed {_join_and(drops)}" if drops else "")
    text = (
        f"Candidate predictors were thinned by variance inflation factor "
        f"(VIF &gt; {vif_max:.1f}, applied iteratively and without reference to "
        f"the outcome){removed}. Discrimination was then estimated over "
        f"{resamples} bootstrap resamples, with selection re-run inside every "
        f"resample for the data-selected models."
    )
    return (
        '<div class="method-statement">'
        '<div class="method-statement-title">Method</div>'
        f'<p>{text}</p></div>'
    )


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _selection_audit_block(art: Artifacts) -> str:
    """Which candidates were considered, kept, and dropped — and why.

    ``Discrimination`` is ``max(AUC, 1−AUC)`` with a ↓ marker on
    protective variables (raw AUC < 0.5) — without it a protective variable
    such as ``adc_value`` (raw AUC 0.370) reads as nearly useless instead of
    the second-best predictor it actually is. ``Selected in resamples`` is
    ``resample_selection_count`` — how often the bootstrap re-picked each
    candidate across the selection procedure's bootstrap resamples; it is
    the evidence for how stable the chosen six are, and some of that
    evidence is not reassuring.
    """
    tbl = art.top_selection
    if tbl is None or tbl.empty:
        return ""

    def _discrimination_cell(r: pd.Series) -> str:
        disc = _coerce_float(r.get("discrimination"))
        if disc is None:
            return ""
        raw_auc = _coerce_float(r.get("auc"))
        marker = " ↓" if raw_auc is not None and raw_auc < 0.5 else ""
        return f"{disc:.3f}{marker}"

    def _reason_cell(r: pd.Series) -> str:
        v = r.get("reason")
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        s = str(v).strip()
        return "" if s.lower() == "nan" else s

    def _kept_cell(r: pd.Series) -> str:
        v = r.get("kept")
        # Tri-state, not boolean: a candidate the full-cohort walk never
        # reached (it won a resample but ranked below the k-th pick, or
        # lower — see model_comparison.run_comparison_stage) has no verdict
        # at all, and that is a third thing, not the same as "considered and
        # dropped". ``None``/NaN must render blank here, not "—": plain
        # ``bool(v)`` would silently read NaN as truthy and mislabel a
        # never-evaluated row "✅ kept".
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return "✅" if bool(v) else "—"

    def _count_cell(r: pd.Series) -> str:
        n = _to_int_or_none(r.get("resample_selection_count"))
        return "" if n is None else str(n)

    n_res = _selection_n_resamples(art)
    count_header = (f"Selected in resamples (of {n_res})" if n_res is not None
                     else "Selected in resamples")

    rows = pd.DataFrame([{
        "Variable": r.get("variable", ""),
        "Discrimination": _discrimination_cell(r),
        "Kept": _kept_cell(r),
        count_header: _count_cell(r),
        "Why dropped": _reason_cell(r),
    } for _, r in tbl.iterrows()])

    return details_block(
        "\U0001f50e How these variables were chosen",
        table_to_html(rows, nowrap_cols=(
            "Discrimination", "Kept", count_header)),
    )


# Per-variant performance figures, in the order a reader should meet them:
# can it rank patients → is the number right → is acting on it worthwhile.
_PERFORMANCE_FIGURE_ORDER = ("roc", "calibration", "decision_curve")


def _render_model_performance(stem: str, art: Artifacts) -> str:
    """ROC / calibration / decision-curve figures for one variant."""
    figs = [
        p
        for suffix in _PERFORMANCE_FIGURE_ORDER
        for p in art.inferential_figures
        if p.stem == f"{stem}__{suffix}"
    ]
    if not figs:
        return ""
    return svg_grid(figs)


def _signed3(x: Any) -> str:
    """``+0.063`` / ``-0.051`` — a difference always carries its sign.

    Three decimals like :func:`_fmt3`, so a delta column lines up with the AUC
    columns beside it, but signed: the whole question a delta answers is which
    way it points, and a bare ``0.051`` makes the reader hunt for that.
    """
    v = _coerce_float(x)
    return "" if v is None else f"{v:+.3f}"


def _delta_cell(apparent: Any, lo: Any, hi: Any) -> str:
    """``+0.080 (+0.031 to +0.132)`` — the APPARENT delta and its interval.

    Same shape as :func:`_or_ci` uses for odds ratios, so the two read alike.
    Both numbers are on one scale on purpose. The interval is the
    patient-resampling spread of the uncorrected difference, so the estimate
    it belongs to is the uncorrected difference — never the optimism-corrected
    one, which is a different number and lives in its own column (see
    :func:`_model_overview_block`, and :func:`_model_vs_single_block` for the
    same split in the per-model tables).
    """
    a = _coerce_float(apparent)
    if a is None:
        return ""
    l, h = _coerce_float(lo), _coerce_float(hi)
    if l is None or h is None:
        return _signed3(a)
    return f"{_signed3(a)} ({_signed3(l)} to {_signed3(h)})"


def _model_overview_block(art: Artifacts) -> str:
    """Every model, both comparators, at the top of the section.

    The per-model tables further down each answer "did combining THESE
    variables help?" — a model against its own best ingredient, which is the
    comparison Zhang 2020 and Peng 2021 published. That question is answered
    once per model, in eleven separate folds, which makes the models hard to
    read against each other and cannot answer the other obvious question: is a
    given model worth more than the single strongest variable in the cohort?
    Most models do not contain that variable, so nothing below has a row for
    it. This table carries both, one row per model.
    """
    tbl = art.model_overview
    if tbl is None or tbl.empty:
        return ""
    def _named(v: Any) -> str:
        """A real name, or "". Missing arrives as NaN from a CSV and as None
        from an in-memory frame; both must render as an empty cell, never as
        the strings "nan" or "None"."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("", "nan", "none") else s

    ref = next((s for s in (_named(r) for r in tbl.get("reference", [])) if s), "")
    ref_name = prettify_label(ref) if ref else "reference"
    # Each comparator gets a corrected column and an apparent-with-CI column,
    # never one cell holding a corrected estimate next to an apparent
    # interval. See the prose below and _model_vs_single_block.
    own_corr, own_app = "Δ vs its best single — corrected (95% CI)", \
        "Δ vs its best single — apparent (95% CI)"
    ref_corr, ref_app = f"Δ vs {ref_name} — corrected (95% CI)", \
        f"Δ vs {ref_name} — apparent (95% CI)"
    rows = pd.DataFrame([{
        "Model": prettify_label(str(r["model_id"])),
        "Predictors": _fmt_count(r.get("n_predictors")),
        "AUC (corrected)": _fmt3(r.get("auc_corrected")),
        "Its best single": (prettify_label(_named(r.get("best_own_single")))
                            if _named(r.get("best_own_single")) else ""),
        own_corr: _delta_cell(
            r.get("delta_own_corrected"),
            r.get("delta_own_ci_lo_corrected"), r.get("delta_own_ci_hi_corrected")),
        own_app: _delta_cell(
            r.get("delta_own_apparent"), r.get("delta_own_ci_lo"), r.get("delta_own_ci_hi")),
        ref_corr: _delta_cell(
            r.get("delta_ref_corrected"),
            r.get("delta_ref_ci_lo_corrected"), r.get("delta_ref_ci_hi_corrected")),
        ref_app: _delta_cell(
            r.get("delta_ref_apparent"), r.get("delta_ref_ci_lo"), r.get("delta_ref_ci_hi")),
    } for _, r in tbl.iterrows()])
    return details_block(
        "📋 All models at a glance — is one variable as good as the whole model?",
        table_to_html(rows, nowrap_cols=(
            "Predictors", "AUC (corrected)",
            own_corr, own_app, ref_corr, ref_app)),
        open=True,
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
    return figure_card(fig) + _render_model_performance_overview(target, art)


def _render_model_performance_overview(target: str, art: Artifacts) -> str:
    """Where every variant lands, then what its extra predictors bought.

    Two cards, in that order, because they are two questions and the single
    plate that answered both at once could not be read by the radiologists it
    was written for. Discrimination first — a reader has to know where a model
    lands before "it gained 0.03" means anything — then the gain against the
    two comparators.

    Either may be absent (a run with one model draws neither), so the cards are
    collected rather than assumed.
    """
    cards = []
    for suffix in ("model_discrimination", "model_gain"):
        fig = next(
            (p for p in art.inferential_figures
             if p.stem == f"{target}__{suffix}"),
            None,
        )
        if fig is not None:
            cards.append(figure_card(fig))
    return "".join(cards)


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


def _cleaning_shape(art: Artifacts, step: str, column: str) -> int | None:
    """One number off the cleaning summary, or None when it was not written."""
    tbl = art.cleaning_summary
    if tbl is None or tbl.empty or not {"step", column}.issubset(tbl.columns):
        return None
    hit = tbl.loc[tbl["step"].astype(str) == step, column]
    return next((v for v in (_to_int_or_none(x) for x in hit) if v is not None), None)


def _source_files(art: Artifacts) -> str:
    """The raw export(s) this run was built from, as recorded by the cleaner."""
    tbl = art.cleaning_summary
    if tbl is None or tbl.empty or not {"step", "criterion"}.issubset(tbl.columns):
        return ""
    hit = tbl.loc[tbl["step"].astype(str) == "raw_data", "criterion"]
    for v in hit:
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return ""


def _schema_value_counts(art: Artifacts) -> tuple[int, int]:
    """(values the schema rewrote, how many of those became missing).

    Counted over the ``n`` column, because one audit row stands for n values,
    not for one.
    """
    tbl = art.schema_coercion
    if tbl is None or tbl.empty or "n" not in tbl.columns:
        return 0, 0
    n = pd.to_numeric(tbl["n"], errors="coerce").fillna(0)
    total = int(n.sum())
    if "value_after" not in tbl.columns:
        return total, 0
    lost = tbl["value_after"].astype(str).str.strip().eq("(missing)")
    return total, int(n[lost].sum())


def _missing_cell_count(cfg: ReportConfig, art: Artifacts) -> int | None:
    """Missing cells in the analysed cohort, before imputation."""
    for tbl in (art.missingness_summary,
                _maybe_read_csv(cfg.output_root / "missingness" / "tables"
                                / "missing_per_column.csv", art.warnings)):
        if tbl is not None and not tbl.empty and "n_missing" in tbl.columns:
            return int(pd.to_numeric(tbl["n_missing"], errors="coerce").fillna(0).sum())
    return None


def _imputation_label(art: Artifacts) -> str:
    """``MICE (R mice) · 20 draws``, or an em dash when nothing imputed."""
    mf = art.mice_manifest or {}
    if not mf:
        return "—"
    engine = str(mf.get("engine") or "").strip() or "engine not recorded"
    draws = mf.get("m") or len(mf.get("frames") or []) or None
    return f"MICE ({engine})" + (f" · {draws} draws" if draws else "")


def _resample_label(art: Artifacts) -> str:
    """The resample count actually used, not merely the one requested."""
    seen = {v for v in (
        [_to_int_or_none(x) for x in art.model_vs_single["n_resamples"]]
        if art.model_vs_single is not None and not art.model_vs_single.empty
        and "n_resamples" in art.model_vs_single.columns else []
    ) + [_selection_n_resamples(art)] if v is not None}
    if len(seen) == 1:
        return f"{seen.pop():,}"
    return f"{analysis.BOOTSTRAP_RESAMPLES:,}"


def _stat(icon: str, label: str, value: Any, *, tint: str,
          was: Any = None, small: bool = False, html: bool = False) -> str:
    """One dashboard tile. ``was`` rides beside the value as a small pill."""
    shown = value if html else _esc("—" if value in (None, "") else value)
    pill = "" if was in (None, "") else f'<span class="was">was {_esc(was)}</span>'
    cls = "v sm" if small else "v"
    return (
        f'<div class="stat" style="--tint:{tint}">'
        f'<div class="ico">{icon}</div>'
        f'<div class="k">{_esc(label)}</div>'
        f'<div class="{cls}">{shown}{pill}</div>'
        f'</div>'
    )


def render_header(cfg: ReportConfig, art: Artifacts) -> str:
    """🧾 Top-of-report dashboard: where the data came from and what was done to it."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows_after = _cleaning_shape(art, "final", "n_rows")
    rows_before = _cleaning_shape(art, "raw_data", "n_rows")
    cols_after = _cleaning_shape(art, "final", "n_columns")
    cols_before = _cleaning_shape(art, "raw_data", "n_columns")
    replaced, replaced_nan = _schema_value_counts(art)
    n_missing = _missing_cell_count(cfg, art)
    n_models = len(art.inferential_multivariable or {})

    # Colour bands, so the eye groups the eleven numbers into four ideas:
    # where it came from, how big it is, what cleaning did, how it was modelled.
    WHERE, SIZE, CLEAN, MODEL = (
        "var(--muted)", "var(--green)", "var(--orange)", "var(--accent)")

    targets_html = "".join(
        f'<span class="tgt">{_esc(t)}</span>' for t in cfg.targets
    ) or "—"

    tiles = [
        _stat("🕒", "Generated", now, tint=WHERE, small=True),
        _stat("📄", "Source file", _source_files(art) or "not recorded",
              tint=WHERE, small=True),
        _stat("📊", "Rows", f"{rows_after:,}" if rows_after else None,
              was=f"{rows_before:,}" if rows_before else None, tint=SIZE),
        _stat("🧾", "Columns", cols_after, was=cols_before, tint=SIZE),
        _stat("🔧", "Values rewritten by schema", f"{replaced:,}" if replaced else None,
              tint=CLEAN),
        _stat("🕳️", "…of those, made missing", f"{replaced_nan:,}" if replaced_nan else 0,
              tint=CLEAN),
        _stat("❓", "Missing values", f"{n_missing:,}" if n_missing is not None else None,
              tint=CLEAN),
        _stat("🧩", "Imputation", _imputation_label(art), tint=MODEL, small=True),
        _stat("🔁", "Bootstrap resamples", _resample_label(art), tint=MODEL),
        _stat("🧮", "Multivariable models", n_models or None, tint=MODEL),
        _stat("🎯", "Targets", targets_html, tint=MODEL, small=True, html=True),
    ]

    authors_html = _format_authors(cfg.author)
    title_block = f'<div class="report-title-block"><h1>🧾 {_esc(cfg.title)}</h1>'
    if authors_html:
        title_block += f'<p class="report-authors">{_esc(authors_html)}</p>'
    title_block += "</div>"

    return (
        f'<section class="report-section">'
        f'{title_block}'
        f'<div class="dash">{"".join(tiles)}</div>'
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
        parts.append(table_to_html(_project(added, _DERIVED_COLUMNS), max_rows=200))

    updated = log[action.str.startswith("updated ColSpec")]
    if not updated.empty:
        parts.append("<h3>Recoded variables</h3>")
        parts.append(table_to_html(_project(updated, _RECODED_COLUMNS), max_rows=200))
    return "".join(parts)


def render_cleaning(cfg: ReportConfig, art: Artifacts) -> str:
    """🧹 Cleaning story."""
    body: list[str] = []

    has_coercion = art.schema_coercion is not None and not art.schema_coercion.empty
    if (art.cleaning_summary is None and art.cleaning_log is None
            and art.derivation_log is None and not has_coercion):
        body.append(warning_box(
            "No saved cleaning summary was found. Cleaning may have been "
            "performed, but no cleaning audit table was exported."))
        return section_block("🧹 Cleaning story", "".join(body))

    cohort = _cohort_flow_table(art)
    if cohort:
        body.append("<h3>Inclusion / exclusion criteria</h3>")
        body.append(cohort)

    body.append(_derived_tables(art.derivation_log))

    if has_coercion:
        coer = collapse_coercion_audit_rows(art.schema_coercion.copy())
        if "n" in coer.columns:
            coer = _format_count_cols(coer, ["n", "n_after"] if "n_after" in coer.columns else ["n"])
        n_rows = len(coer)
        body.append(details_block(
            f"Coerced value audit ({n_rows})",
            table_to_html(coer, max_rows=500),
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
    body: list[str] = []

    # --- 1️⃣ Univariate ---
    uni: list[str] = []

    sections = [
        ("📏 Continuous / count variables", art.dda_continuous),
        ("🏷️ Categorical / ordinal variables", art.dda_categorical),
        ("✅ Binary variables", art.dda_binary),
        ("🕒 Datetime variables", art.dda_datetime),
        ("🪪 ID / text variables", art.dda_id_text),
    ]
    derived_cols = art.dda_derived_columns
    hidden_parents = art.hidden_parent_columns
    for heading, tbl in sections:
        if tbl is not None and not tbl.empty and hidden_parents and "column" in tbl.columns:
            tbl = tbl[~tbl["column"].astype(str).isin(hidden_parents)].copy()
        if tbl is None or tbl.empty:
            inner = '<p class="muted"><em>(no variables of this kind)</em></p>'
            uni.append(details_block(heading, inner))
            continue

        display_tbl = (
            _dda_continuous_for_report(tbl)
            if tbl is art.dda_continuous
            else _dda_kind_for_display(tbl)
        )
        n = len(display_tbl)
        parts = [_dda_native_derived_tables(display_tbl, derived_cols)]
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
            svg_grid(figs, plain=(
                "One picture per column, showing what its values look like. "
                "Nothing is compared against outcome here.")),
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
        biv_parts: list[str] = []
        for x_key, figs in groups.items():
            label = prettify_label(x_key)
            biv_parts.append(details_block(
                f"🔑 {label} ({len(figs)})",
                svg_grid(figs, plain=(
                    "Each picture puts two columns side by side, to show "
                    "whether they move together.")),
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
                svg_grid(figs, plain=(
                    "The same pairs again, split by group, to see if the "
                    "pattern holds in each one.")),
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


def _largest_missing(art: Artifacts) -> tuple[str, float] | None:
    """The most incomplete column and its percentage, or None if unknown."""
    tbl = art.missingness_summary
    if tbl is None or tbl.empty:
        return None
    pct_col = _first_present(tbl, ["pct_missing", "missing_pct"])
    name_col = _first_present(tbl, ["column", "variable", "name"])
    if pct_col is None or name_col is None:
        return None
    ranked = tbl.dropna(subset=[pct_col]).sort_values(pct_col, ascending=False)
    if ranked.empty:
        return None
    top = ranked.iloc[0]
    pct = _coerce_float(top[pct_col])
    return (str(top[name_col]), pct) if pct is not None else None


def _m_rule_of_thumb_row(art: Artifacts, m_value: Any) -> tuple[str, str] | None:
    """m against the rule of thumb, as a row for the engine table.

    White, Royston & Wood (2011): run at least as many imputations as the
    percentage of cases missing on the most incomplete variable. Checked here
    rather than asserted, because m is set in the cleaning notebook and the
    missingness it has to cover is only known after the data are read.
    """
    top = _largest_missing(art)
    n_imp = _to_int_or_none(m_value)
    if top is None or n_imp is None:
        return None
    name, pct = top
    mark = "✅ met" if n_imp >= pct else "⚠️ below the rule"
    return ("m ≥ largest % missing",
            f"m = {n_imp} vs {pct:.1f}% missing in "
            f"<code>{_esc(name)}</code> — {mark}")


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
    rule_row = _m_rule_of_thumb_row(art, m.get("m"))
    if rule_row is not None:
        rows.insert(5, rule_row)
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
        '<p class="footnote">Rule of thumb: run at least as many imputations '
        'as the percentage of cases missing on the most incomplete variable. '
        'Fewer than that and the pooled estimate still visibly depends on '
        'which way the gaps happened to be filled.</p>'
    )


def render_missingness(cfg: ReportConfig, art: Artifacts) -> str:
    """🕳️ Missingness story."""
    body = [
        '<p>Missingness was assessed per variable and globally. Formal '
        'mixed-type multiple imputation (MICE) generated m completed datasets '
        'via <code>missingness_resolution.proper_mice_impute()</code>, which '
        'runs one <code>mice()</code> fully-conditional-specification chain in '
        'R (continuous/count → PMM, binary → logistic, nominal → polytomous, '
        'ordinal → proportional-odds). PMM fills a gap by copying a value '
        'already observed for that column, so an imputed measurement is never '
        'a new number: this preserves the plausible range but piles the '
        'filled-in values onto the observed ones, which is why no cut-point '
        'anywhere in this project is searched for in imputed data. '
        'Between-imputation uncertainty is '
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

    if art.top_missing is not None and not art.top_missing.empty:
        body.append("<h3>Top missing</h3>")
        body.append(table_to_html(
            art.top_missing,
            row_class_fn=lambda r: classify_missing(
                r.get("missing_pct", r.get("pct_missing")), cfg.missing),
        ))

    if art.missingness_figures:
        body.append("<h3>Patterns</h3>")
        # No shared intro line. Both pattern figures carry their own
        # plain-words reading in a legend sidecar, and a sentence above the
        # grid would either repeat them or, as the retired one did, go on
        # describing figures that are no longer drawn.
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
        # The saved artifact carries a legend sidecar, so it goes through
        # figure_card rather than being wrapped by hand here.
        card = figure_card(heatmap_path, extra_class="eda-heatmap-overview")
        if card:
            parts = [card]
            if excluded:
                items = "".join(f"<li>{_esc(p)}</li>" for p in excluded)
                parts.append(
                    "<p><strong>Predictors not FDR-significant for any "
                    "target</strong> (omitted from heatmap):</p>"
                    f"<ul class='muted'>{items}</ul>"
                )
            return "".join(parts)
        img = ""
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

    return "".join(blocks)


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
                svg_grid(figs, plain=(
                    "One picture per predictor, comparing the two outcome "
                    "groups. The line under each gives the test and its "
                    "p-value."))))

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
        body.append(_model_overview_block(art))
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
                link = _usable_link(art.inferential_model_links.get(mkey, ""))
                summary = f"📐 {title}" if title else f"📐 Model {model_id}"

                # The link is its own line rather than part of the published
                # block: a model can carry a source URL without this repo
                # holding a transcription of what the paper fitted, and losing
                # the link in that case is how it silently went missing.
                published = _published_model_block(model_id)
                if link:
                    published += (
                        f'<p class="muted">📄 <a href="{_esc(link)}" '
                        'target="_blank" rel="noopener noreferrer">'
                        "Read the source paper</a></p>"
                    )
                meta = _inferential_target_meta(art, target, model_key_name=mkey)
                stem = artifact_base(target, model_id)
                forest = [p for p in art.inferential_figures if p.stem == f"{stem}__forest"]
                vif = (table_to_html(art.inferential_vif[mkey])
                       if mkey in art.inferential_vif else "")

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
                # Positional reset so ``display``'s default 0..n-1 index lines up
                # with ``tbl``'s row order below — ``_row_cls`` still classifies
                # off the numeric OR/CI columns, which only live on ``tbl``.
                tbl = tbl.reset_index(drop=True)

                # The numeric ``tbl`` stays as-is for the interpretation block
                # below; the display copy is what the reader sees.
                display = pd.DataFrame({
                    "Predictor": [predictor_label(r) for _, r in tbl.iterrows()],
                    "β (SE)": [_beta_se(r.get("coef"), r.get("se"))
                               for _, r in tbl.iterrows()],
                    "OR (95% CI)": [_or_ci(r.get(col_or), r.get(col_lo), r.get(col_hi))
                                    for _, r in tbl.iterrows()],
                    "P": [human_p(r.get(col_p)) for _, r in tbl.iterrows()],
                })
                # One model, read top to bottom: what was published, whether
                # this cohort can carry it, what came out, then the checks.
                blocks.append(details_block(summary, _model_steps([
                    ("The source paper", published),
                    ("Whether this cohort can carry the model", meta),
                    ("The model refit here",
                     _model_level_line(tbl) + table_to_html(
                         display, row_class_fn=lambda r: _row_cls(tbl.loc[r.name]),
                         nowrap_cols=("β (SE)", "OR (95% CI)", "P"))),
                    ("The same result as a plot",
                     svg_grid(forest) if forest else ""),
                    ("Collinearity check", vif),
                    ("How well it performs", _render_model_performance(stem, art)),
                    ("Was the combination worth it?",
                     _model_vs_single_block(model_id, art)),
                ])))
            return "".join(blocks)

        # The group is a heading, not a dropdown: one click should reach a
        # model, not a box holding seven more boxes.
        # Directly above the per-model sections: the two overview figures end
        # here, and this is the paragraph that says how they were built.
        body.append(_method_statement_block(art))
        if literature_keys:
            body.append('<h4 class="model-group">📚 Literature-based models</h4>')
            body.append(_render_model_blocks(literature_keys))
        if experimental_keys:
            body.append('<h4 class="model-group">🧪 Experimental models</h4>')
            body.append(_render_model_blocks(experimental_keys))

    body.append(_selection_audit_block(art))

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
            return figure_card(path, caption="")
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
        # Rule-in and rule-out are two columns of one table now, so the
        # section carries one figure per family rather than a pair.
        parts.append(_panel_figure(art, f"lr_table_{key}"))
    parts.append(_panel_table_footnotes(art))
    return "".join(p for p in parts if p)


def _panel_table_footnotes(art: Artifacts) -> str:
    """The table's own footnote block — it has to stand on its own.

    Radiology journals expect a table to be readable lifted out of the paper,
    so every abbreviation is defined here even though the running text defines
    them too, and the sort rule is stated rather than inferred.
    """
    panel = art.panel_marker
    # Either ratio can need the correction, and they fire on different rows —
    # an asterisk in the LR− column with no note under the table would be a
    # mark the reader cannot look up.
    corrected = panel is not None and not panel.empty and any(
        col in panel.columns and bool(panel[col].any())
        for col in ("continuity_corrected", "lr_neg_corrected"))

    # Family sizes, so the note can say what the correction was actually
    # spread across rather than leaving a reader to count the rows.
    counts: dict[str, int] = {}
    if panel is not None and not panel.empty and "origin" in panel.columns:
        counts = panel["origin"].astype(str).value_counts().to_dict()
    native_n, derived_n = counts.get("native", 0), counts.get("derived", 0)
    if native_n and derived_n:
        family = (f"within its own family ({native_n} native and {derived_n} "
                  "derived signs, corrected separately, so a q in one table is "
                  "not comparable with a q in the other)")
    elif native_n or derived_n:
        family = f"across the {native_n or derived_n} signs in this table"
    else:
        family = "within its own family"

    lines: list[str] = []
    # AJNR wants a table to be readable lifted out of the paper, so every
    # column carries its expansion here — including the three (PPV, NPV, LR−)
    # the manuscript's own Table 2 drops and this one keeps.
    lines.append(
        "<strong>Note:—</strong>LR+ indicates positive likelihood ratio; LR−, "
        "negative likelihood ratio; PPV, positive predictive value; NPV, "
        "negative predictive value; n/N (%), patients with the sign present "
        "out of those in whom it was recorded. Sensitivity, specificity, PPV, "
        "NPV and both likelihood ratios carry Wilson or Katz 95% CIs and were "
        "calculated on observed (non-imputed) cases — imputing them would "
        "report the performance of a sign nobody looked at, which is why N "
        "varies between rows. FDR p is the p value after Benjamini–Hochberg "
        f"adjustment for multiple comparisons {family}; it is not the raw p "
        "value. Rows are ordered by LR+, descending.")
    if corrected:
        lines.append(
            "* estimate calculated with a continuity correction because one "
            "cell of the 2×2 table was empty.")
    # The bands are a reading convention, not a test: they tell a reader what
    # a number in the LR+ column is worth without the table having to grade
    # any row itself.
    lines.append(
        "LR+ bands, by the usual convention: 2–5 shifts probability a little, "
        "5–10 moderately, above 10 enough to decide. The mirror for ruling "
        "out, LR−: below 0.1 is a large shift, 0.1–0.2 moderate, 0.2–0.5 "
        "small, 0.5–1 minimal (Jaeschke R, Guyatt GH, Sackett DL. "
        "Users' Guides to the Medical Literature. III. How to use an article "
        "about a diagnostic test. <em>JAMA</em> 1994;271:703–707).")
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
