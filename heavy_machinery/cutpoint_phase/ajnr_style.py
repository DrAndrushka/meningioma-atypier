"""AJNR figure style — one place that decides how every figure in this phase looks.

Two rules drive everything here.

**Nothing is distinguished by colour alone.** Series are separated by *shade*
and *line style* together, so a reader with any form of colour blindness, a
greyscale printout, and a photocopy of that printout all carry the same
information. This is stricter than the journal requires; it costs nothing and it
removes a whole category of reviewer objection.

**The export rules are assertions, not intentions.** Width, resolution, minimum
type size and filename hygiene are checked in :func:`save_figure` and raise on
violation. A figure that silently ships at 7 pt type is found by the production
editor weeks later; one that refuses to save is found now.

Two formats come off the same figure object, so they cannot drift apart: TIFF
for submission (lossless, which matters for thin curves and small type) and PNG
for the HTML report and for reuse. EPS is deliberately absent — the backend
flattens transparency, which corrupts every confidence band in this phase.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- the contract ----------------------------------------------------------
WIDTH_IN = 7.0            # two-column; the journal floor is 4.0
DPI = 600                 # combination halftone
MIN_FONT_PT = 8.0
MAX_WIDTH_IN = 7.5        # beyond this a figure risks being set broadside
FORMATS = ("tif", "png")

# Filenames and metadata must not identify the authors or the equipment.
FORBIDDEN_IN_NAMES = ("pskus", "riga", "latvia", "siemens", "philips", "ge_",
                      "zaguzovs")

# --- the palette -----------------------------------------------------------
# Anchored on the values the existing univariate OR forest already uses, so a
# reader moving between that figure and these does not have to relearn what
# black and grey mean. INK, MUTED and ROW_BAND are the same three values.
#
# The five shades are spaced for even perceived lightness so adjacent series
# stay separable in print, and each is paired one-to-one with a line style:
# either cue alone is enough to tell two curves apart.
INK = "#000000"
MUTED = "#7F7F7F"         # a result whose interval crosses the null
SHADES = ("#000000", "#4D4D4D", "#7F7F7F", "#A6A6A6", "#D9D9D9")
DASHES = ("-", (0, (4, 1.4)), (0, (1, 1.2)), (0, (5, 1.2, 1, 1.2)),
          (0, (3, 1, 1, 1, 1, 1)))

BAND = "#000000"          # confidence bands: ink at low opacity, never a hue
BAND_ALPHA = 0.13
ROW_BAND = "#D9D9D9"      # alternating row shading in forest-style figures
ROW_BAND_ALPHA = 0.35
REFERENCE = "#9A9A9A"     # soft guides: cohort-average lines, knot positions
RUG = "#000000"
RUG_ALPHA = 0.35

# The null line (OR = 1, AUC = 0.5) is black and dashed, not grey — it is a
# value being tested against, not a background guide.
NULL_LINE = {"color": INK, "linewidth": 0.8, "linestyle": "--"}
MARKER = "s"
MARKER_SIZE = 4.5
CI_LINEWIDTH = 1.1

# House rule: no annotation arrows. They rarely land where they are aimed once
# a figure is resized for the page, and the caption is where an explanation
# belongs anyway.


def series_style(i: int, *, lw: float = 1.4) -> dict:
    """Shade **and** line style for series ``i`` — never one without the other."""
    return {"color": SHADES[i % len(SHADES)],
            "linestyle": DASHES[i % len(DASHES)],
            "linewidth": lw}


def point_style(*, muted: bool = False) -> dict:
    """Square marker and interval colour, matching the existing OR forest.

    ``muted`` is the same convention that figure uses: an estimate whose
    interval crosses the null is drawn in grey rather than dropped, so the
    reader sees what was tested and not only what survived.
    """
    colour = MUTED if muted else INK
    return {"color": colour, "marker": MARKER, "markersize": MARKER_SIZE,
            "linestyle": "none"}


def apply_style() -> None:
    """Set the rcParams every figure in this phase is drawn under.

    Applied locally by the figure builders rather than globally on import, so
    importing this module never changes how some other phase's figures look.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "image.cmap": "Greys",
    })


class FigureContractError(Exception):
    """A figure violates a rule the journal or this phase enforces."""


def _check_type_sizes(fig) -> None:
    """No rendered text below the minimum, at final size.

    Figures are saved at the size they are drawn, so a point size here is a
    point size on the page — no scaling factor to reason about.
    """
    small = sorted({
        round(float(t.get_fontsize()), 2)
        for t in fig.findobj(mpl.text.Text)
        if t.get_text().strip() and float(t.get_fontsize()) < MIN_FONT_PT})
    if small:
        raise FigureContractError(
            f"Text at {small} pt is below the {MIN_FONT_PT} pt minimum.")


def _check_name(stem: Path) -> None:
    lowered = stem.name.lower()
    hits = [bad for bad in FORBIDDEN_IN_NAMES if bad in lowered]
    if hits:
        raise FigureContractError(
            f"Filename contains identifying text {hits} — figures are "
            "submitted with the manuscript blinded.")


def _check_size(fig) -> None:
    width, _ = fig.get_size_inches()
    if width > MAX_WIDTH_IN:
        raise FigureContractError(
            f"Figure is {width:.2f} in wide; over {MAX_WIDTH_IN} in it risks "
            "being set broadside.")


def save_figure(fig, stem: Path | str, *, formats: tuple[str, ...] = FORMATS,
                dpi: int = DPI, tight: bool = True) -> list[Path]:
    """Write one figure to every required format, refusing to ship a bad one.

    ``stem`` carries no suffix — the formats supply it, which is what keeps the
    TIFF and the PNG provably the same picture.
    """
    stem = Path(stem)
    _check_name(stem)
    _check_size(fig)
    _check_type_sizes(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)

    extra = {"bbox_inches": "tight"} if tight else {}
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        if fmt == "tif":
            fig.savefig(path, dpi=dpi,
                        pil_kwargs={"compression": "tiff_lzw"}, **extra)
        elif fmt == "png":
            # Empty Software key: matplotlib otherwise stamps its version, and
            # the rule is that no tooling or site detail rides along in the file.
            fig.savefig(path, dpi=dpi, metadata={"Software": ""}, **extra)
        else:
            raise FigureContractError(
                f"Format {fmt!r} is not one this phase ships ({FORMATS}).")
        written.append(path)
    return written


def new_figure(ncols: int = 1, nrows: int = 1, *, height_in: float = 3.4,
               width_in: float = WIDTH_IN, **kwargs):
    """A figure at the journal's width, with the phase style already applied."""
    apply_style()
    return plt.subplots(nrows, ncols, figsize=(width_in, height_in), **kwargs)
