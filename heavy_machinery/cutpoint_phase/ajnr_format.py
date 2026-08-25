"""Number formatting for the manuscript — one place, so nothing can disagree.

Every number printed in a table or a legend passes through here. The point is
not tidiness: a value rounded to two decimals in a table and three in a figure
caption is a discrepancy a reviewer will find, and the only reliable way to
prevent it is to have one function that decides.

The journal's conventions, which differ from the defaults everywhere:

**P values carry no leading zero** — ``.03``, not ``0.03``. Below .001 they are
reported as ``<.001`` rather than as a number, because a P value that small is
not measuring anything the sample size can support. Between .001 and .01, three
decimals; above, two.

**Estimates keep their leading zero** — ``0.72``, not ``.72``. Only P values
drop it, and applying the P-value rule to an odds ratio is a common slip.

**Intervals use an en dash**, not a hyphen, and no spaces around it — except
where a bound can be negative. ``-0.089-0.056`` reads as one range with a stray
sign, so a signed quantity uses a true minus sign and the word "to"
(:func:`fmt_signed`, :func:`fmt_signed_ci`). These are separate functions rather
than a flag on :func:`fmt_est`, because every existing caller prints a quantity
that cannot go below zero and must keep printing exactly what it prints today.

A blank cell is written as an em dash, never as ``nan`` or an empty string: a
reader needs to see that the cell was considered and has no value, not wonder
whether the table lost it.
"""
from __future__ import annotations

import math

BLANK = "—"        # em dash: considered, no value
EN_DASH = "–"
MINUS = "\u2212"   # not the ASCII hyphen: that is a word-break, not a sign


def _missing(value) -> bool:
    try:
        return value is None or not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def fmt_p(p) -> str:
    """P value in the journal's style: no leading zero, ``<.001`` at the floor."""
    if _missing(p):
        return BLANK
    p = float(p)
    if p < 0.001:
        return "<.001"
    if p < 0.01:
        return f"{p:.3f}".lstrip("0")
    if p > 0.99:
        return ">.99"
    return f"{p:.2f}".lstrip("0")


def fmt_est(value, decimals: int = 2) -> str:
    """An estimate, keeping its leading zero — the rule P values do not follow."""
    return BLANK if _missing(value) else f"{float(value):.{decimals}f}"


def fmt_ci(lo, hi, decimals: int = 2) -> str:
    """A bare interval, en dash, no spaces: ``0.61–0.74``."""
    if _missing(lo) or _missing(hi):
        return BLANK
    return f"{fmt_est(lo, decimals)}{EN_DASH}{fmt_est(hi, decimals)}"


def fmt_est_ci(value, lo, hi, decimals: int = 2) -> str:
    """An estimate with its interval in parentheses: ``0.68 (0.61–0.74)``."""
    if _missing(value):
        return BLANK
    interval = fmt_ci(lo, hi, decimals)
    return fmt_est(value, decimals) if interval == BLANK else \
        f"{fmt_est(value, decimals)} ({interval})"


def fmt_pct(fraction, decimals: int = 0) -> str:
    """A share written as a percentage: ``0.93`` becomes ``93%``."""
    if _missing(fraction):
        return BLANK
    return f"{float(fraction) * 100:.{decimals}f}%"


def fmt_ratio(value, decimals: int = 2) -> str:
    """A multiple of something, written so the unit is unmistakable: ``0.93×``."""
    return BLANK if _missing(value) else f"{float(value):.{decimals}f}×"


def fmt_value(value, decimals: int) -> str:
    """A measurement in its own units, at the precision that measurement prints.

    ``:g`` rather than a fixed width, so ``15.1`` does not become ``15.100`` and
    ``0.0617`` keeps the digits that distinguish it.
    """
    if _missing(value):
        return BLANK
    return f"{round(float(value), decimals):g}"


def fmt_span(lo, hi, decimals: int) -> str:
    """A range in native units: ``0.69–0.85``."""
    if _missing(lo) or _missing(hi):
        return BLANK
    return f"{fmt_value(lo, decimals)}{EN_DASH}{fmt_value(hi, decimals)}"


def join_names(names) -> str:
    """``A``, ``A and B``, ``A, B and C`` — a list that reads as a sentence.

    Every lead line under a table is prose, and ``", ".join`` produces "for
    tumor volume, max diameter", which stops a reader mid-sentence to work out
    whether a third item went missing.
    """
    names = [str(n) for n in names]
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def yes_no(flag) -> str:
    """A graded verdict, or a blank where the rule could not be applied."""
    return BLANK if flag is None else ("Yes" if bool(flag) else "No")


def fmt_signed(value, decimals: int = 2) -> str:
    """A quantity that can be negative, with a true minus sign.

    ``-0.05`` set with a hyphen is a typographic error the eye reads as a dash;
    at 7 pt in a figure it is also barely visible. U+2212 is the character the
    glyph was designed for and is present in every font this project uses.
    """
    if _missing(value):
        return BLANK
    text = f"{abs(float(value)):.{decimals}f}"
    return f"{MINUS}{text}" if float(value) < 0 else text


def fmt_signed_ci(value, lo, hi, decimals: int = 2, *, separator: str | None = None) -> str:
    """A signed estimate with its interval: ``0.06 (0.01 to 0.10)``.

    ``separator`` overrides the choice of "to" versus an en dash. Pass one when
    formatting a whole column, so that a column holding any negative bound reads
    the same way in every row rather than switching form partway down.
    """
    if _missing(value):
        return BLANK
    if _missing(lo) or _missing(hi):
        return fmt_signed(value, decimals)
    if separator is None:
        separator = " to " if min(float(lo), float(hi)) < 0 else EN_DASH
    return (f"{fmt_signed(value, decimals)} ("
            f"{fmt_signed(lo, decimals)}{separator}{fmt_signed(hi, decimals)})")


def interval_separator(values) -> str:
    """The separator a whole column should use, given every bound in it."""
    for v in values:
        try:
            if float(v) < 0:
                return " to "
        except (TypeError, ValueError):
            continue
    return EN_DASH
