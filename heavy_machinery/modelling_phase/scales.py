"""The one place that decides which measurements are analysed on a log scale.

Three phases standardise continuous measurements before fitting — modelling,
thresholds and cut-points — and every one of them must use the same scale. When
they do not, the same cohort produces two different odds ratios for the same
variable and the reader has no way to tell which is which. That is what this
module exists to prevent: the set is declared once, here, and the other phases
read it rather than restating it.

``log1p`` rather than ``log`` because edema volume and edema index are
legitimately zero for about a third of this cohort. ``log(0)`` is undefined, so
plain log would silently drop exactly the patients whose *absence* of edema is
the finding.

Which measurements, and why:

``tumor_volume``      spans three orders of magnitude. One raw SD is a step no
                      patient in the middle of the cohort ever takes, and a
                      model fitted on the raw scale spends its slope on a
                      handful of very large tumours.
``edema_volume_cm3``  same spread, plus a third of the cohort at exactly zero.
``edema_index``       a ratio of the two, so it inherits both problems.

``max_diameter_cm`` and ``adc_value`` are not here on purpose. A diameter is
already close to the cube root of a volume, and ADC spans a factor of two
across the whole cohort; neither needs the transform, and applying it would
make their odds ratios harder to read for no gain.

The odds ratio for a logged measurement is *per 1 SD on the log scale*. That is
not a cosmetic detail — it must be said wherever such an odds ratio is printed,
which is what :func:`scale_footnote` is for.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Sample SD (divide by n-1), not population SD. One convention for the whole
# pipeline, so a measurement cannot read 1.85 in one table and 1.86 in another.
SD_DDOF = 1

LOG1P_COLUMNS: frozenset[str] = frozenset({
    "tumor_volume",
    "edema_volume_cm3",
    "edema_index",
})

# The quantitative measurements the manuscript reports. For these the scale is a
# pipeline-wide decision and every phase must agree, so the phases check
# themselves against this set at import or validation time. Anything outside it
# — an exploratory metric, a test fixture — may choose its own scale locally,
# because no second phase is publishing a number for it.
GOVERNED_COLUMNS: frozenset[str] = LOG1P_COLUMNS | frozenset({
    "max_diameter_cm",
    "adc_value",
})


def is_log_scaled(col: str) -> bool:
    """True when ``col`` is standardised on the ``log1p`` scale."""
    return str(col) in LOG1P_COLUMNS


def disagreements(flags: dict[str, bool]) -> list[str]:
    """Governed columns whose local ``log_x`` contradicts this module.

    ``flags`` maps a column to the scale the caller has declared for it. Columns
    outside :data:`GOVERNED_COLUMNS` are ignored rather than reported.
    """
    return sorted(
        col for col, log_x in flags.items()
        if str(col) in GOVERNED_COLUMNS and bool(log_x) != is_log_scaled(col)
    )


def to_model_scale(x, log_x: bool) -> np.ndarray:
    """Apply the declared transform, without centring or scaling.

    Negative values would make ``log1p`` undefined. Every measurement declared
    here is a volume or a ratio of volumes and cannot be negative, so a negative
    value means the column is not what it claims to be; it becomes NaN rather
    than a silent complex or error.
    """
    u = np.asarray(x, dtype=float)
    if not log_x:
        return u
    out = np.where(u > -1.0, u, np.nan)
    return np.log1p(out)


def z_params(x, log_x: bool) -> tuple[float, float]:
    """Mean and SD on the modelling scale, ignoring missing values.

    Returned separately because the multivariable model has to store them: a
    risk calculator converts a patient's raw value into model units with them,
    and for a logged measurement that conversion has to log first.
    """
    u = to_model_scale(x, log_x)
    if not np.any(np.isfinite(u)):
        return float("nan"), float("nan")
    return float(np.nanmean(u)), float(np.nanstd(u, ddof=SD_DDOF))


def standardise(x, log_x: bool) -> np.ndarray:
    """Centre and scale so one unit is one SD, log-transforming where declared."""
    u = to_model_scale(x, log_x)
    mu, sd = z_params(x, log_x)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(u)
    return (u - mu) / sd


def scale_footnote(cols: Iterable[str]) -> str:
    """The sentence that must accompany any table of per-SD odds ratios.

    Empty when none of ``cols`` is logged, so a table that happens to contain no
    logged measurement does not carry an explanation of something absent.
    """
    logged = [str(c) for c in cols if is_log_scaled(c)]
    if not logged:
        return ""
    from plot_style import prettify_label

    labels = [prettify_label(c) for c in dict.fromkeys(logged)]
    if len(labels) == 1:
        named = labels[0]
        verb = "was"
    else:
        named = ", ".join(labels[:-1]) + f" and {labels[-1]}"
        verb = "were"
    return (
        f"{named} {verb} log-transformed (log1p) before standardisation because "
        f"{'it is' if len(labels) == 1 else 'they are'} strongly right-skewed, "
        f"so {'its' if len(labels) == 1 else 'their'} odds ratio is per 1 SD on "
        f"the log scale. Odds ratios for the remaining measurements are per 1 SD "
        f"in their own clinical units. log1p rather than log because edema is "
        f"legitimately zero in about a third of this cohort."
    )
