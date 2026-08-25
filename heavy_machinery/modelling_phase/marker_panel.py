"""Which MRI markers, and do they combine? — the two study aims, in one place.

The report's EDA section already scores every marker on its own, and
:mod:`combinations` already knows how to compare a combined rule against a
single one. What is missing is a place where those two answers sit side by side
on one patient set, in the report that actually gets read.

Nothing here re-implements an estimator that already exists. The combination
machinery is :mod:`combinations`, reached through a nine-line adapter. The only new statistic is the positive likelihood ratio,
which is what turns "most specific" into a question with a defensible answer —
a sign that is never seen is perfectly specific and perfectly useless.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

import combinations as cb
import plot_style as ps
from cleaning import format_table_for_csv
from diagnostic_accuracy import binary_diagnostic_metrics
from eda import benjamini_hochberg
from marker_rules import format_pct_ci

# Imported, not re-declared, so this table cannot drift from the OR forest and
# the cut-point figures a reader sees a few pages earlier.
try:
    import ajnr_format as afmt
    import ajnr_style as aj
except ModuleNotFoundError:  # pragma: no cover - cutpoint_phase not on sys.path
    from heavy_machinery.config import load as _load_config  # noqa: F401
    import ajnr_format as afmt
    import ajnr_style as aj

_Z95 = 1.959963984540054

BINARY_KINDS = ("binary", "derived_binary")


def likelihood_ratio_positive(tp: int, fp: int, fn: int, tn: int) -> dict:
    """How much more likely this sign is in a high-grade tumor than a benign one.

    ``LR+ = sensitivity / (1 - specificity)``. An LR+ of 10 means seeing the
    sign makes high grade ten times more likely; an LR+ of 1 means it says
    nothing. The interval is Katz's, computed on the log scale because a ratio
    bounded below by zero and unbounded above is not symmetric.

    A zero in the flagged column makes the ratio infinite and its interval
    undefined. Half a patient is added to every cell in that case
    (Haldane-Anscombe) and ``continuity_corrected`` says so, because the
    resulting interval is honestly enormous and the reader should see why.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
           "chance_overlap": False, "continuity_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = tp == 0 or fp == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    sens = a / (a + c)
    fpr = b / (b + d)
    if fpr <= 0 or sens <= 0:
        return nan

    lr = sens / fpr
    se = math.sqrt(1.0 / a - 1.0 / (a + c) + 1.0 / b - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_pos": float(lr),
        "lr_pos_lo": float(lo),
        "lr_pos_hi": float(hi),
        "chance_overlap": bool(lo <= 1.0 <= hi),
        "continuity_corrected": bool(corrected),
    }


def likelihood_ratio_negative(tp: int, fp: int, fn: int, tn: int) -> dict:
    """What *not* seeing the sign is worth — the other half of the 2×2.

    ``LR- = (1 - sensitivity) / specificity``. An LR- of 0.1 means the absence
    of the sign makes high grade ten times *less* likely; an LR- of 1 means
    absence says nothing. Same Katz interval on the log scale as LR+, with the
    miss column standing where the hit column stood.

    The continuity correction fires on ``FN == 0`` or ``TN == 0`` — a different
    row than the LR+ one, which is why it reports its own flag instead of
    sharing ``continuity_corrected``.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    nan = {"lr_neg": np.nan, "lr_neg_lo": np.nan, "lr_neg_hi": np.nan,
           "lr_neg_corrected": False}
    if (tp + fn) == 0 or (fp + tn) == 0:
        return nan

    corrected = fn == 0 or tn == 0
    a, b, c, d = (tp + 0.5, fp + 0.5, fn + 0.5, tn + 0.5) if corrected else (tp, fp, fn, tn)

    miss = c / (a + c)
    spec = d / (b + d)
    if spec <= 0 or miss <= 0:
        return nan

    lr = miss / spec
    se = math.sqrt(1.0 / c - 1.0 / (a + c) + 1.0 / d - 1.0 / (b + d))
    lo = math.exp(math.log(lr) - _Z95 * se)
    hi = math.exp(math.log(lr) + _Z95 * se)
    return {
        "lr_neg": float(lr),
        "lr_neg_lo": float(lo),
        "lr_neg_hi": float(hi),
        "lr_neg_corrected": bool(corrected),
    }


class BinaryMarker(NamedTuple):
    """A yes/no MRI sign, shaped like a ``CutPoint`` so ``combinations`` accepts it.

    :mod:`combinations` touches exactly four members of a cut-point — ``col``,
    ``label``, ``short_label`` and ``flag`` — so supplying those is enough to
    run its whole rule machinery on markers that were never continuous. This is
    why ``combinations.py`` needs no change: the cut-point estimators and this
    section's are the same code, and cannot drift apart.
    """

    col: str
    label: str

    @property
    def short_label(self) -> str:
        """No cut-point to name, so the short form is the label itself."""
        return self.label

    def flag(self, df: pd.DataFrame) -> pd.Series:
        return df[self.col].astype("boolean")


def markers_from_diagnostic_accuracy(
    table: pd.DataFrame,
    *,
    target: str,
    exclude: Collection[str] = (),
) -> list[BinaryMarker]:
    """The marker panel, read from the EDA table rather than hard-coded.

    Whatever is activated or dropped in the cleaning notebook's ``DERIVATIONS``
    flows through here without an edit, which is the point: the panel cannot
    silently fall out of step with the cohort it describes.

    ``exclude`` is the caller's, and belongs in the notebook. The accuracy
    table carries non-imaging predictors too — ``sex_male`` is
    ``derived_binary`` and would otherwise walk into a section about MRI signs.
    """
    if table is None or table.empty:
        return []
    drop = set(exclude) | {target}
    rows = table[
        (table["target"].astype(str) == str(target))
        & (table["kind"].astype(str).isin(BINARY_KINDS))
        & (~table["predictor"].astype(str).isin(drop))
    ]
    return [
        BinaryMarker(str(r["predictor"]), ps.prettify_label(str(r["predictor"])))
        for _, r in rows.iterrows()
    ]


_PANEL_COLUMNS = [
    "marker", "label", "n_used", "present_n", "catches", "n_high_grade",
    "TP", "FP", "FN", "TN",
    "sensitivity", "sensitivity_lo", "sensitivity_hi",
    "specificity", "specificity_lo", "specificity_hi",
    "PPV", "PPV_lo", "PPV_hi", "NPV", "NPV_lo", "NPV_hi",
    "AUC", "OR", "OR_lo", "OR_hi",
    "lr_pos", "lr_pos_lo", "lr_pos_hi", "chance_overlap", "continuity_corrected",
    "lr_neg", "lr_neg_lo", "lr_neg_hi", "lr_neg_corrected",
    "p", "p_fdr", "test", "origin",
]

# A derived flag is a native column with a line drawn through it —
# ``adc_value_le0.72`` is ``adc_value`` dichotomised. Correcting the two in one
# family counts the same information twice and shifts every native q, so the
# panel is split and each half is corrected on its own.
NATIVE, DERIVED = "native", "derived"


def classify_origin(
    markers: Collection[str],
    *,
    derivation_sources: dict[str, str],
    hidden_parents: Collection[str],
) -> dict[str, str]:
    """Which side of the panel each marker belongs to, and why.

    The test is not "was this column derived" but **is the column it restates
    still in the table**:

    *Parent hidden* — the flag replaced its parent outright. Nothing in the
    table restates anything, so the flag *is* the native variable and is
    corrected with the others. ``male`` replaced ``sex``; there is no ``sex``
    row left for it to duplicate.

    *Parent present* — the flag and the measurement it was cut from are both
    here. ``adc_value_le0.72`` is ``adc_value`` with a line drawn through it,
    and correcting them together tests one thing twice. The flag goes to the
    derived table and takes no part in the native family.

    Driving this off ``hide_parent`` rather than off a hand-kept list is what
    keeps the two in step: change the derivation and the panel follows.
    """
    hidden = {str(h) for h in hidden_parents}
    out: dict[str, str] = {}
    for marker in (str(m) for m in markers):
        parent = derivation_sources.get(marker)
        out[marker] = (
            DERIVED if parent and parent not in hidden else NATIVE
        )
    return out


def _hidden_parents(output_root: Path | str) -> frozenset[str]:
    """Columns a derivation replaced outright, from the cleaning handoff."""
    path = Path(output_root) / "cleaning" / "hidden_parent_columns.csv"
    if not path.exists():
        return frozenset()
    tbl = pd.read_csv(path)
    if "column" not in tbl.columns:
        return frozenset()
    return frozenset(str(c) for c in tbl["column"].dropna())


def _derivation_parents(output_root: Path | str) -> dict[str, str]:
    """Each derived column's single source, from the cleaning derivation log.

    Only single-source derivations count. A column computed from two others
    (``edema_index``) restates neither of them on its own, and one that lists
    itself is a repair rule rather than a restatement.
    """
    path = Path(output_root) / "cleaning" / "derivation_log.csv"
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    if not {"derivation", "source"} <= set(log.columns):
        return {}
    out: dict[str, str] = {}
    for _, row in log.iterrows():
        name = str(row["derivation"]).strip()
        sources = [s.strip() for s in str(row.get("source", "")).split(",")
                   if s.strip()]
        if len(sources) == 1 and sources[0] != name:
            out[name] = sources[0]
    return out


def marker_panel_table(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    origin_by_marker: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Every marker on its own, ranked by how hard a positive finding argues.

    Ranked by LR+, but ``catches`` is what keeps the ranking honest: the most
    specific sign in a cohort is usually the one nobody ever sees, and a table
    sorted on specificity alone puts it first. Markers whose interval covers 1
    are sorted to the bottom instead of being given a rank they have not
    earned.
    """
    if not markers:
        return pd.DataFrame(columns=_PANEL_COLUMNS)

    rows = []
    for marker in markers:
        row = binary_diagnostic_metrics(
            df, target, marker.col, predictor_series=marker.flag(df),
        )
        row["marker"] = marker.col
        row["label"] = marker.label
        tp, fp = row.get("TP"), row.get("FP")
        fn, tn = row.get("FN"), row.get("TN")
        if any(pd.isna(v) for v in (tp, fp, fn, tn)):
            row.update({"lr_pos": np.nan, "lr_pos_lo": np.nan, "lr_pos_hi": np.nan,
                        "chance_overlap": False, "continuity_corrected": False})
            row.update({"lr_neg": np.nan, "lr_neg_lo": np.nan,
                        "lr_neg_hi": np.nan, "lr_neg_corrected": False})
            row.update({"present_n": 0, "catches": 0, "n_high_grade": 0})
        else:
            row.update(likelihood_ratio_positive(tp, fp, fn, tn))
            row.update(likelihood_ratio_negative(tp, fp, fn, tn))
            row["present_n"] = int(tp) + int(fp)
            row["catches"] = int(tp)
            row["n_high_grade"] = int(tp) + int(fn)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["chance_overlap"] = out["chance_overlap"].astype(bool)
    # Corrected across this table's own rows, so the footnote's claim about
    # which family the q-values belong to is true by construction rather than
    # by a comment. Borrowing them from the EDA screen would silently break
    # the moment a marker is excluded here but not there.
    origins = origin_by_marker or {}
    out["origin"] = out["marker"].astype(str).map(
        lambda m: origins.get(m, NATIVE))
    # One BH per origin, not one across the panel. Each q is therefore a rank
    # within its own family, and the two are not comparable with each other —
    # which the table footnote has to say, because nothing in the column does.
    out["p_fdr"] = np.nan
    if "p" in out.columns:
        for origin in (NATIVE, DERIVED):
            mask = out["origin"] == origin
            if mask.any():
                out.loc[mask, "p_fdr"] = benjamini_hochberg(
                    out.loc[mask, "p"]).values
        # Stated, then checked. A BH q is m/rank × p, so if a derived row had
        # slipped into the native family every native q would be wrong by the
        # ratio of the two family sizes — a silent, uniform inflation that no
        # single number on the page would look wrong.
        native_mask = out["origin"] == NATIVE
        if native_mask.any():
            expected = benjamini_hochberg(out.loc[native_mask, "p"]).values
            if not np.allclose(out.loc[native_mask, "p_fdr"].to_numpy(float),
                               np.asarray(expected, dtype=float),
                               equal_nan=True):
                raise ValueError(
                    "The native FDR family is not the native rows alone.")
    out = out.sort_values(
        ["chance_overlap", "lr_pos"], ascending=[True, False], kind="mergesort",
    ).reset_index(drop=True)
    cols = [c for c in _PANEL_COLUMNS if c in out.columns]
    return out[cols + [c for c in out.columns if c not in cols]]


def _format_lr(row: pd.Series, *, key: str = "lr_pos",
               corrected_key: str = "continuity_corrected") -> str:
    """``2.79 (1.68–4.63)`` — always the number, never a verdict.

    An interval covering 1 is left to speak for itself. Printing a phrase in
    place of the estimate substitutes our reading for the reader's, and a
    journal table is expected to carry the number either way.

    Two decimals, because one is not enough to tell the reader what the
    footnote asks them to check: at one decimal, mass effect (1.04–1.21,
    excludes 1) and dural tail (0.99–1.21, crosses it) both print as
    ``1.1 (1.0–1.2)``.
    """
    if pd.isna(row.get(key)):
        return "—"
    text = f"{row[key]:.2f} ({row[f'{key}_lo']:.2f}–{row[f'{key}_hi']:.2f})"
    return text + "*" if bool(row.get(corrected_key)) else text


def _format_q(value) -> str:
    """``0.003`` / ``<0.001`` — an adjusted p, never in scientific notation."""
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _format_present(row: pd.Series) -> str:
    """``59/309 (19%)`` — the epidemiological convention, prevalence scannable."""
    present, used = int(row["present_n"]), int(row["n_used"])
    pct = 100.0 * present / used if used else 0.0
    return f"{present}/{used} ({pct:.0f}%)"


def marker_panel_reading_view(panel: pd.DataFrame) -> pd.DataFrame:
    """The aim-1 table in the columns a clinician reads.

    Ordered by LR+ descending — a single stated sort rule, rather than the
    panel's own "informative first, then the rest", so the table's footnote
    can describe the order in one line.

    Predictive values sit next to sensitivity and specificity in the order a
    radiology table is read, and carry the cohort's own prevalence with them:
    they answer "the sign is there, now what?" where LR+ answers "how much
    does seeing it move the odds?".
    """
    if panel is None or panel.empty:
        return pd.DataFrame(columns=[
            "Variable", "n/N (%)",
            "Sens (95% CI)", "Spec (95% CI)",
            "PPV (95% CI)", "NPV (95% CI)", "FDR p", "LR+ (95% CI)",
            "LR− (95% CI)",
        ])
    ranked = panel.sort_values("lr_pos", ascending=False, na_position="last",
                               kind="mergesort")
    return pd.DataFrame({
        "Variable": ranked["label"],
        "n/N (%)": [_format_present(r) for _, r in ranked.iterrows()],
        "Sens (95% CI)": [format_pct_ci(r, "sensitivity") for _, r in ranked.iterrows()],
        "Spec (95% CI)": [format_pct_ci(r, "specificity") for _, r in ranked.iterrows()],
        "PPV (95% CI)": [format_pct_ci(r, "PPV") for _, r in ranked.iterrows()],
        "NPV (95% CI)": [format_pct_ci(r, "NPV") for _, r in ranked.iterrows()],
        "FDR p": [_format_q(r.get("p_fdr")) for _, r in ranked.iterrows()],
        "LR+ (95% CI)": [_format_lr(r) for _, r in ranked.iterrows()],
        # Beside LR+, because the two answer opposite questions about the same
        # row and a reader deciding whether a negative scan is reassuring
        # should not have to compute one from the other.
        "LR− (95% CI)": [
            _format_lr(r, key="lr_neg", corrected_key="lr_neg_corrected")
            for _, r in ranked.iterrows()
        ],
        # Carried so the report can split the table without re-deriving which
        # side each row belongs to. Dropped before display.
        "origin": (ranked["origin"] if "origin" in ranked.columns
                   else pd.Series(NATIVE, index=ranked.index)),
    }).reset_index(drop=True)


LR_TABLE_LEGEND = (
    "One row per finding, strongest first. Left: what seeing the finding is worth — the further right, the more it argues for a high-grade tumour. Right: what its absence is worth — the further left, the more it argues against one. A bar touching the line means that reading tells you nothing. The two halves have their own scales, so a position in one is not a distance in the other.")

LR_TABLE_LEGEND_DERIVED = (
    "The cut points, read the same way as the findings figure. They sit apart because each one restates a measurement that is still in the table, so they take no part in that figure's multiplicity correction and cannot be ranked against it.")


def _empty_forest(message: str) -> plt.Figure:
    """A figure that says why it is empty, rather than an axis with no rows."""
    fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM, aspect=0.5))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    return fig


# Reciprocal pairs, not powers of ten. Over the range a likelihood ratio
# actually occupies, decades give two or three labels and leave the eye nothing
# to interpolate against; and pairing 0.5 with 2, 0.35 with 3, 0.2 with 5 is
# what lets a reader see a ratio and its reciprocal as the same distance from
# "tells you nothing".
_LR_TICKS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0,
             10.0, 20.0, 50.0)


def _lr_ticks(lo: float, hi: float, *, most: int = 8) -> list[float]:
    """The ladder rungs inside ``lo``–``hi``, thinned outward from 1."""
    inside = [t for t in _LR_TICKS if lo <= t <= hi]
    if len(inside) > most:
        anchor = inside.index(1.0) if 1.0 in inside else 0
        inside = [t for i, t in enumerate(inside) if (i - anchor) % 2 == 0]
    return inside


def _lr_limits(values) -> tuple[float, float]:
    """Limits that hold every bound given, plus 1 and a little air.

    1 is forced in whatever the data does: the null line is drawn there, and a
    panel of cut points can easily have every interval above it.
    """
    usable = [1.0]
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            usable.append(f)
    lo, hi = min(usable), max(usable)
    pad = (hi / lo) ** 0.045
    lo, hi = lo / pad, hi * pad
    # Extra air on a side the null itself bounds: every interval above 1 puts
    # the dashed line on the spine, where it reads as the edge of the panel
    # rather than as the value a reader is comparing against.
    if min(usable) == 1.0:
        lo /= 1.12
    if max(usable) == 1.0:
        hi *= 1.12
    return lo, hi


def _text_width(fig: plt.Figure, text: str, size: float) -> float:
    """How wide a string will actually set, in inches.

    Measured rather than estimated because the label column is sized to it: a
    guess that runs short puts a finding's name into the forest beside it, and
    one that runs long spends the inches the intervals need.
    """
    try:
        renderer = fig.canvas.get_renderer()
        artist = fig.text(0, 0, text, fontsize=size)
        width = artist.get_window_extent(renderer=renderer).width / fig.dpi
        artist.remove()
        return float(width)
    except Exception:  # pragma: no cover - backend without a usable renderer
        return 0.55 * size / 72.0 * len(text)


def lr_table_figure(panel: pd.DataFrame) -> plt.Figure:
    """Both likelihood ratios for every finding, as one table with two forests.

    A row is a finding and the table answers two questions about it: what
    seeing it is worth (LR+, left) and what not seeing it is worth (LR−,
    right). Both are drawn *and* printed, so nothing has to be measured off an
    axis, and the pair sits on one line — the version that split them into two
    figures made the reader hold a rank in their head while turning the page.

    Two columns rather than two series on one axis. The two answer opposite
    questions and their intervals overlap heavily around 1; on a shared axis
    they interleave until neither is readable, whereas a column is something
    the eye can scan on its own. That is what recovers the finding the split
    was protecting: mass effect is eleventh by LR+ and the strongest rule-out
    of all, and its square is visibly the leftmost in the LR− column.

    Each column carries its own scale. LR− never leaves a narrow band around
    1, so forcing it onto the LR+ range would spend half the column on empty
    paper. The cost is that a position in one column is not a distance in the
    other, which the legend says.

    Rows sort by LR+, descending. One order is the price of one table, and it
    is the order the panel table itself is written in.
    """
    if len(panel) == 0 or not {"lr_pos", "lr_neg"} <= set(panel.columns):
        return _empty_forest("No marker has an estimable likelihood ratio")
    rows = panel[panel["lr_pos"].notna() & panel["lr_neg"].notna()]
    if rows.empty:
        return _empty_forest("No marker has an estimable likelihood ratio")
    rows = rows.sort_values("lr_pos", ascending=False).reset_index(drop=True)

    small = plt.rcParams["xtick.labelsize"]
    body = plt.rcParams["ytick.labelsize"]
    columns = (
        ("lr_pos", "When present", "argues for high grade \u2192", "LR+ (95% CI)", True),
        ("lr_neg", "When absent", "\u2190 argues against high grade", "LR\u2212 (95% CI)", False),
    )
    cells = {key: [afmt.fmt_est_ci(r[key], r[f"{key}_lo"], r[f"{key}_hi"], 2)
                   for _, r in rows.iterrows()]
             for key, *_ in columns}

    W = ps.FIG_WIDTH_DOUBLE
    X_NAME, EDGE, GAP, SPLIT = 0.06, 0.06, 0.10, 0.14
    ROW_H, TOP_PAD, HDR_H = 0.245, 0.104, 0.47
    AXIS_H, LEGEND_H = 0.30, 0.50

    top_in = TOP_PAD + HDR_H
    ys = [top_in + ROW_H * (i + 0.5) for i in range(len(rows))]
    bot_in = top_in + ROW_H * len(rows)
    height = bot_in + AXIS_H + LEGEND_H

    fig = plt.figure(figsize=(W, height))
    fig.patch.set_facecolor("white")

    # Columns sized to the widest string each actually holds, so the inches
    # left over go to the intervals. Clamped so that no label, however long,
    # can squeeze a forest down to a stub.
    name_w = min(1.9, max(0.8, max(
        _text_width(fig, str(v), body) for v in rows["label"]) + 0.04))
    val_w = min(0.95, max(0.5, max(
        [_text_width(fig, v, small) for vs in cells.values() for v in vs]
        + [_text_width(fig, c[3], small) for c in columns]) + 0.03))
    ax_w = (W - EDGE - X_NAME - name_w - 0.08 - 2 * GAP - 2 * val_w - SPLIT) / 2

    x0 = X_NAME + name_w + 0.08
    AX_A = (x0, x0 + ax_w)
    X_A_R = AX_A[1] + GAP + val_w
    AX_B = (X_A_R + SPLIT, X_A_R + SPLIT + ax_w)
    X_B_R = AX_B[1] + GAP + val_w

    def _rect(x0_, y0_, x1_, y1_):
        return (x0_ / W, 1 - y1_ / height, (x1_ - x0_) / W, (y1_ - y0_) / height)

    ax_a = fig.add_axes(_rect(AX_A[0], top_in, AX_A[1], bot_in))
    ax_b = fig.add_axes(_rect(AX_B[0], top_in, AX_B[1], bot_in))
    axes = {"lr_pos": ax_a, "lr_neg": ax_b}
    limits = {key: _lr_limits(
        list(rows[f"{key}_lo"]) + list(rows[f"{key}_hi"]) + list(rows[key]))
        for key, *_ in columns}
    for key, ax in axes.items():
        # y is inches from the top of the figure, inverted, so a row's position
        # is the same number in the axes and in the text columns beside them.
        ax.set_ylim(bot_in, top_in)
        ax.set_xscale("log")
        ax.set_xlim(*limits[key])
        ax.set_yticks([])
        ax.set_facecolor("none")
        ax.grid(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    def _fx(x_in: float) -> float:
        """An inch position on the sheet, as a fraction of the left axis."""
        return (x_in - AX_A[0]) / (AX_A[1] - AX_A[0])

    def _text(x_in, y_in, s, *, ha="left", size=None, weight="normal",
              style="normal", color=aj.INK):
        ax_a.text(_fx(x_in), y_in, s, transform=ax_a.get_yaxis_transform(),
                  ha=ha, va="center", fontsize=size or small, fontweight=weight,
                  fontstyle=style, color=color, clip_on=False, zorder=4)

    for i, y0 in enumerate(ys):
        if i % 2 == 0:
            fig.add_artist(Rectangle(
                (X_NAME / W, 1 - (y0 + ROW_H / 2) / height),
                (X_B_R - X_NAME) / W, ROW_H / height, transform=fig.transFigure,
                facecolor=aj.ROW_BAND, alpha=aj.ROW_BAND_ALPHA, linewidth=0,
                zorder=0))

    # --- header: the column name, then what a direction on it means ---------
    hdr_y = TOP_PAD + HDR_H * 0.24
    sub_y = TOP_PAD + HDR_H * 0.58
    _text(X_NAME, hdr_y, "Finding", weight="bold")
    for (key, head, gloss, value_head, _), x_r in ((columns[0], X_A_R),
                                                   (columns[1], X_B_R)):
        ax = axes[key]
        mid = (ax.get_position().x0 + ax.get_position().x1) / 2 * W
        _text(mid, hdr_y, head, ha="center", weight="bold")
        _text(mid, sub_y, gloss, ha="center", size=small * 0.9,
              style="italic", color="#5A5A5A")
        _text(x_r, hdr_y, value_head, ha="right", weight="bold")
    fig.add_artist(Line2D([X_NAME / W, X_B_R / W],
                          [1 - (top_in - 0.055) / height] * 2,
                          transform=fig.transFigure, color=aj.INK,
                          linewidth=1.0, zorder=3))

    # --- the rows -----------------------------------------------------------
    for i, (y0, (_, r)) in enumerate(zip(ys, rows.iterrows())):
        _text(X_NAME, y0, str(r["label"]), size=body)
        for (key, _, _, _, filled), x_r in ((columns[0], X_A_R),
                                            (columns[1], X_B_R)):
            ax = axes[key]
            est, lo, hi = r[key], r[f"{key}_lo"], r[f"{key}_hi"]
            # Grey is per column, not per row: a finding can be conclusive seen
            # and worth nothing unseen, and that is the point of printing both.
            #
            # Tested on the rounded bounds, which is what the reader is given.
            # An upper bound of 0.9996 prints as 1.00 and excludes the null by
            # the raw number, so full ink beside a printed "0.86-1.00" reads as
            # the figure contradicting its own table.
            colour = (aj.REFERENCE if round(lo, 2) <= 1.0 <= round(hi, 2)
                      else aj.INK)
            ax.plot([lo, hi], [y0, y0], color=colour, linewidth=1.0, zorder=2)
            for x_end in (lo, hi):
                ax.plot([x_end, x_end], [y0 - 0.028, y0 + 0.028], color=colour,
                        linewidth=1.0, zorder=2)
            ax.plot([est], [y0], marker=aj.MARKER, markersize=aj.MARKER_SIZE * 0.86,
                    markerfacecolor=colour if filled else "white",
                    markeredgecolor=colour, markeredgewidth=0.9,
                    linestyle="none", zorder=3)
            _text(x_r, y0, cells[key][i], ha="right", color=colour)

    for key, ax in axes.items():
        ax.axvline(1.0, zorder=1, **aj.NULL_LINE)
        ax.spines["bottom"].set_position(("outward", 4))
        ticks = _lr_ticks(*limits[key])
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        # Log minor ticks off, not just unlabelled: inside a single decade
        # matplotlib labels them, and "4 × 10⁻¹" beside "0.35" is two notations
        # for the same axis.
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=small, length=3.2, pad=2)

    rule_y = bot_in + AXIS_H - 0.03
    fig.add_artist(Line2D([X_NAME / W, X_B_R / W], [1 - rule_y / height] * 2,
                          transform=fig.transFigure, color=aj.INK,
                          linewidth=1.0, zorder=3))

    def _mark(filled: bool) -> Line2D:
        return Line2D([], [], linestyle="none", marker=aj.MARKER,
                      markerfacecolor=aj.INK if filled else "white",
                      markeredgecolor=aj.INK, markeredgewidth=0.9,
                      markersize=aj.MARKER_SIZE * 0.86)

    fig.legend([_mark(True), _mark(False), Line2D([], [], **aj.NULL_LINE)],
               ["Finding present (LR+)", "Finding absent (LR\u2212)",
                "No information (LR = 1)"],
               loc="upper left",
               bbox_to_anchor=(X_NAME / W, 1 - (rule_y + 0.11) / height),
               bbox_transform=fig.transFigure, ncol=3, frameon=False,
               fontsize=small, handletextpad=0.5, columnspacing=1.8,
               borderaxespad=0.0, borderpad=0.0)
    return fig


def usable_markers(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> tuple[list[BinaryMarker], list[dict]]:
    """Markers that vary. A column that is always the same answers nothing.

    An all-absent sign has no true positives and an undefined likelihood ratio;
    an always-present one has no true negatives. Both would enter the rule
    search and produce cells of zeros, so they are dropped here with the reason
    recorded rather than silently.
    """
    kept: list[BinaryMarker] = []
    dropped: list[dict] = []
    y = df[target].astype("boolean")
    for marker in markers:
        flags = marker.flag(df)[y.notna()]
        n_true = int((flags == True).sum())   # noqa: E712 - nullable boolean
        n_false = int((flags == False).sum())  # noqa: E712
        if n_true == 0:
            dropped.append({"marker": marker.col,
                            "reason": "never present in this cohort"})
        elif n_false == 0:
            dropped.append({"marker": marker.col,
                            "reason": "always present in this cohort"})
        else:
            kept.append(marker)
    return kept, dropped


def shared_cohort_frame(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The patients every rule is scored on — threshold-phase logic, reused."""
    if not markers:
        return df.iloc[0:0].copy()
    return cb.shared_cohort(df, markers, target)


def shared_cohort_audit(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    dropped: Sequence[dict] = (),
) -> pd.DataFrame:
    """What the shared set cost, per marker, so the denominator is auditable.

    A restriction to complete cases is defensible; a restriction nobody can
    check is not. This is the table that lets a reader see the loss is one or
    two measurements rather than the marker panel as a whole.
    """
    shared = shared_cohort_frame(df, markers, target)
    y = df[target].astype("boolean")
    rows: list[dict] = [
        {"item": "Patients in the cohort", "value": int(len(df)), "note": ""},
        {"item": "Patients in the shared set", "value": int(len(shared)),
         "note": "every marker observed and the outcome known"},
        {"item": "High grade in the shared set",
         "value": int(shared[target].astype("boolean").sum()) if len(shared) else 0,
         "note": ""},
        {"item": "Markers required", "value": int(len(markers)),
         "note": ", ".join(m.col for m in markers)},
    ]
    for marker in markers:
        missing = int((marker.flag(df).isna() & y.notna()).sum())
        rows.append({"item": marker.col, "value": missing,
                     "note": "patients this marker alone was missing for"})
    for entry in dropped:
        rows.append({"item": entry["marker"], "value": 0,
                     "note": f"excluded — {entry['reason']}"})
    return pd.DataFrame(rows)


def count_score(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """Observed high-grade rate at each number of signs present.

    The answer to the study aim that involves no choosing. Every other
    comparison in this section picks a winner and then has to pay for having
    picked it; this one asks a question with no winner in it, so the number it
    produces is the number.
    """
    return cb.count_score_table(df, markers, target, complete_only=True)


def count_thresholds(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
) -> pd.DataFrame:
    """The count used as a test: "at least one sign", "at least two", …"""
    return cb.count_threshold_table(df, markers, target, complete_only=True)


MAX_SUBTITLE_MARKERS = 5


def count_score_figure(
    counts: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    *,
    prevalence: float | None = None,
) -> plt.Figure:
    """The combination figure, drawn on markers instead of cut-points.

    The subtitle lists the criteria by name, which works for the three or four
    cut-points :mod:`combinations` feeds it and does not work here: sixteen
    marker names is a 200-character run-on line, and matplotlib gives up on the
    layout rather than wrapping it. Past
    :data:`MAX_SUBTITLE_MARKERS` the names are dropped and the axis label —
    "Criteria met (of 16)" — carries the count instead. The marker names are
    not lost; they are the rows of the aim-1 table on the same page.
    """
    counts = counts.copy()
    counts.attrs.setdefault("k", len(markers))
    listed = tuple(markers) if len(markers) <= MAX_SUBTITLE_MARKERS else ()
    return cb.count_score_figure(counts, cutpoints=listed, prevalence=prevalence)


MIN_HEADLINE_N = 10

_HEADLINE_COLUMNS = [
    "k_markers", "min_n", "n_bins_usable", "direction",
    "low_count", "low_n", "low_risk",
    "high_count", "high_n", "high_risk", "note",
]


def count_headline(
    counts: pd.DataFrame,
    *,
    k: int | None = None,
    min_n: int = MIN_HEADLINE_N,
) -> pd.DataFrame:
    """The two bins the headline sentence is allowed to quote, and which way it went.

    The count-score table has a row for every possible count, and the rows at
    the two ends are almost always the thinnest: on this cohort the highest
    occupied bin holds a single patient, whose outcome sets that bin's "risk"
    to 0% or 100% with nothing behind it. A sentence built from the first and
    last *occupied* rows therefore reports two coin flips and calls it a trend.

    So the endpoints are chosen from bins with at least ``min_n`` patients, and
    the direction is measured rather than assumed. ``direction`` is
    ``"rises"``, ``"falls"`` or ``"flat"``, and the renderer picks its wording
    from that instead of hard-coding "rises" — a sentence that is true only
    when the data cooperates is not a finding, it is a hope.

    If no two bins clear ``min_n`` the floor is relaxed to "occupied at all"
    and ``note`` says so, because a thin honest sentence beats no sentence.
    """
    if counts is None or counts.empty or "n" not in counts.columns:
        return pd.DataFrame(columns=_HEADLINE_COLUMNS)

    ordered = counts.sort_values("n_criteria_met", kind="mergesort")
    occupied = ordered[(ordered["n"] > 0) & ordered["risk"].notna()]

    note = ""
    used_min = int(min_n)
    usable = occupied[occupied["n"] >= used_min]
    if len(usable) < 2:
        usable = occupied
        used_min = 1
        note = (f"no two counts reached {int(min_n)} patients — the endpoints "
                "below rest on whichever counts were occupied at all")
    if len(usable) < 2:
        return pd.DataFrame(columns=_HEADLINE_COLUMNS)

    if k is None:
        k = int(counts.attrs.get("k", int(ordered["n_criteria_met"].max())))
    low, high = usable.iloc[0], usable.iloc[-1]
    low_risk, high_risk = float(low["risk"]), float(high["risk"])
    direction = (
        "rises" if high_risk > low_risk
        else "falls" if high_risk < low_risk
        else "flat"
    )
    return pd.DataFrame([{
        "k_markers": int(k),
        "min_n": used_min,
        "n_bins_usable": int(len(usable)),
        "direction": direction,
        "low_count": int(low["n_criteria_met"]),
        "low_n": int(low["n"]),
        "low_risk": low_risk,
        "high_count": int(high["n_criteria_met"]),
        "high_n": int(high["n"]),
        "high_risk": high_risk,
        "note": note,
    }])


DEFAULT_SEED = 20260801
DEFAULT_N_BOOT = 500


def selection_correction(
    df: pd.DataFrame,
    markers: Sequence[BinaryMarker],
    target: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> pd.DataFrame:
    """What the winner's advantage is worth once you pay for having picked it.

    With a dozen markers there are hundreds of candidate rules, and the best of
    hundreds beats the best single marker on the data that chose it even when
    no rule is genuinely better. :func:`combinations.bootstrap_best_rule`
    measures that gap by rebuilding the whole menu on each resample, taking its
    winner, and scoring that same rule back on the original cohort.

    It is run **twice**, with the same budget and seed. Correcting only the
    combination and comparing it against an uncorrected single is the error
    recorded in ``CHANGES.md``: it reported a gain of +0.008 that was really
    +0.050, because choosing the best of the singles costs almost exactly as
    much as choosing the best of the combinations.

    Both sides come out of one resample loop. They always used the same seed,
    so they were already visiting the same resamples; the second run was only
    re-scoring the same menu to look at a different part of it.
    """
    sides = {
        "best single": ("single",),
        "best combination": ("and", "or", "count"),
    }
    results = cb.bootstrap_best_rules(
        df, markers, target, sides=sides, n_boot=n_boot, seed=seed,
        max_size=max_size,
    )
    rows = [
        {
            "side": label,
            "best_rule": results[label].get("best_rule", ""),
            "J_apparent": results[label].get("J_apparent", np.nan),
            "optimism": results[label].get("optimism", np.nan),
            "J_corrected": results[label].get("J_corrected", np.nan),
            "winner_stability": results[label].get("winner_stability", np.nan),
            "n_bootstrap": results[label].get("n_bootstrap", 0),
        }
        for label in sides
    ]
    out = pd.DataFrame(rows)
    gain = float(out.loc[1, "J_corrected"] - out.loc[0, "J_corrected"])
    gain_apparent = float(out.loc[1, "J_apparent"] - out.loc[0, "J_apparent"])
    out["gain_apparent"] = gain_apparent
    out["gain_corrected"] = gain
    out["correction_effect"] = _correction_effect(gain_apparent, gain)
    return out


def _correction_effect(gain_apparent: float, gain_corrected: float) -> str:
    """Which way correcting both sides moved the gap — measured, not assumed.

    The intuition is that correction shrinks a gap, and often it does. It is
    not a law. Correction subtracts each side's *own* selection optimism, and
    on this cohort the best-of-16-singles side pays more of it than the
    best-of-210-combinations side, so the corrected gap is the **larger** one.
    Prose that asserts "the uncorrected gap is larger" is therefore a claim
    about the data, and belongs in a column where it can be wrong out loud.
    """
    if not (np.isfinite(gain_apparent) and np.isfinite(gain_corrected)):
        return ""
    if gain_corrected > gain_apparent:
        return "widens"
    if gain_corrected < gain_apparent:
        return "narrows"
    return "unchanged"


TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"


def _write_table(tables: dict, root: Path, stem: str, frame: pd.DataFrame) -> None:
    tables[stem] = frame
    (root / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    format_table_for_csv(frame).to_csv(
        root / TABLES_DIRNAME / f"{stem}.csv", index=False,
    )


def count_score_panel(
    df: pd.DataFrame,
    signs: Sequence[str],
    target: str,
) -> dict[str, pd.DataFrame]:
    """Does counting several signs beat using the best one alone? — NOT YET BUILT.

    The dose-response argument. A radiologist checks each sign in ``signs`` and
    counts how many are present; this asks whether risk climbs with that count.
    The shape of the answer is what makes it defensible: if the signs were all
    restating one underlying thing, risk would jump once and then flatten, so a
    **monotone rise is the evidence that combining helps** and a flat or
    wandering curve is the evidence that it does not.

    ``signs`` is passed in, never mined. Picking the best-scoring combination
    out of the data and then reporting how well it scores is the objection a
    committee raises first, and a pre-specified list is the answer to it. The
    existing ``selection_correction`` stays alongside as the guard for the
    "best single vs best combination" claim, which *is* selected and so has to
    carry its optimism correction and its winner stability.

    Planned outputs, all on one patient set — complete cases across every sign,
    because a count compared across different groups of patients is not a
    comparison:

    ``risk_by_count``   one row per k: n, events, risk with a Wilson interval.
    ``rule_by_count``   "≥ k of n signs" as a rule: Se, Sp, PPV, NPV, LR+, J.
    ``figure``          risk against k, greyscale, points with intervals rather
                        than bars, and an at-risk row under the axis instead of
                        floating labels that collide.

    Two things to settle before this is trusted:

    * **Collinear members double-count.** Tumor volume and max diameter
      correlate at rho ~= 0.92 in this cohort, edema volume and edema index at
      ~= 0.91. A count over all five cut-points rises partly for arithmetic
      reasons, so either the list de-duplicates or the caveat is printed.
    * **Sparse counts.** The k = 0 and k = n bins are often a handful of
      patients; a minimum n has to be set and the dropped bins named, not
      silently omitted.
    """
    raise NotImplementedError(
        "count_score_panel is a placeholder — populate COUNT_SCORE_SIGNS in "
        "the modelling notebook and implement this before calling it."
    )


def run_marker_panel(
    df: pd.DataFrame,
    *,
    target: str,
    accuracy_table: pd.DataFrame,
    output_root: Path | str,
    exclude: Collection[str] = (),
    derived_cols: Collection[str] = (),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    max_size: int = 2,
) -> dict[str, pd.DataFrame]:
    """Compute the whole panel and write it to ``output/panel/``.

    Two patient sets on purpose. The aim-1 marker table uses each marker's own
    complete cases, because it ranks markers and each row stands alone. Every
    aim-2 comparison uses the shared set, because a Youden J compared across
    two different groups of patients is not a comparison.

    Nothing here renders. ``report.py`` reads what this writes.
    """
    root = Path(output_root) / "panel"
    tables: dict[str, pd.DataFrame] = {}

    markers = markers_from_diagnostic_accuracy(
        accuracy_table, target=target, exclude=exclude,
    )
    markers = [m for m in markers if m.col in df.columns]
    kept, dropped = usable_markers(df, markers, target)

    hidden = frozenset(derived_cols) if derived_cols else _hidden_parents(output_root)
    origins = classify_origin(
        [m.col for m in kept],
        derivation_sources=_derivation_parents(output_root),
        hidden_parents=hidden,
    )
    panel = marker_panel_table(df, kept, target, origin_by_marker=origins)
    _write_table(tables, root, "01_marker_panel", panel)
    _write_table(tables, root, "02_marker_panel_reading_view",
                 marker_panel_reading_view(panel))

    shared = shared_cohort_frame(df, kept, target)
    _write_table(tables, root, "03_shared_cohort",
                 shared_cohort_audit(df, kept, target, dropped))

    empty = pd.DataFrame()
    if len(kept) >= 2 and not shared.empty:
        correction = selection_correction(
            shared, kept, target, n_boot=n_boot, seed=seed, max_size=max_size,
        )
        counts = count_score(shared, kept, target)
        _write_table(tables, root, "07_count_score", counts)
        _write_table(tables, root, "08_count_thresholds",
                     count_thresholds(shared, kept, target))
        _write_table(tables, root, "09_selection_correction", correction)
        _write_table(tables, root, "12_count_headline",
                     count_headline(counts, k=len(kept)))
    else:
        for stem in ("07_count_score", "08_count_thresholds",
                     "09_selection_correction", "12_count_headline"):
            _write_table(tables, root, stem, empty)
        counts = empty

    fig_dir = root / FIGURES_DIRNAME
    # One table per family, and no combined one. The two families are corrected
    # separately, so a figure holding both invites the reader to rank a derived
    # cut-point against a native sign as though one q had ordered them.
    legends = {NATIVE: LR_TABLE_LEGEND, DERIVED: LR_TABLE_LEGEND_DERIVED}
    for origin in (NATIVE, DERIVED):
        part = panel[panel["origin"] == origin] if "origin" in panel else panel
        if part.empty:
            continue
        # The forests are scaled and ordered by the rows they are given, so a
        # derived row reaching the native table would move the axis and the
        # ranking, not just add a line. Checked here because the filter above
        # is the only thing keeping them apart.
        if "origin" in part.columns and not (part["origin"] == origin).all():
            raise ValueError(
                f"The {origin} table was handed rows from the other family.")
        _fig = lr_table_figure(part.reset_index(drop=True))
        ps.set_figure_legend(_fig, plain=legends[origin])
        ps.save_figure(_fig, fig_dir / f"lr_table_{origin}",
                       tight_layout=False, kind="halftone")
    prevalence = (
        float(shared[target].astype("boolean").mean()) if len(shared) else None
    )
    if len(counts):
        ps.save_figure(count_score_figure(counts, kept, prevalence=prevalence),
                       fig_dir / "count_score")
    else:
        fig, ax = plt.subplots(figsize=ps.figure_size(ps.FIG_WIDTH_MEDIUM,
                                                      aspect=0.4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Not enough markers for a combination",
                ha="center", va="center", transform=ax.transAxes)
        ps.save_figure(fig, fig_dir / "count_score")

    return tables
