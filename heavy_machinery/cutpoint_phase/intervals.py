"""Confidence intervals for proportions — Wilson, never Wald.

A confidence interval is the range a number would plausibly fall in if the study
were repeated on a fresh set of patients.

The textbook formula (Wald) is ``p ± 1.96·√[p(1−p)/n]``. It is symmetric, which
is exactly wrong near the ends: several specificities in this phase are 0.90 or
above, and a symmetric interval around 0.95 runs past 1.0 — a specificity of
"up to 102%". Wilson solves for the interval instead of approximating it, so it
can never leave [0, 1] and it stays honest when the count is small.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def wilson_ci(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n``.

    Returns ``(nan, nan)`` for an empty denominator rather than raising: a
    measurement can legitimately have no patients in a stratum, and that is a
    blank cell in a table, not a broken run.
    """
    if n <= 0:
        return float("nan"), float("nan")
    k = float(successes)
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def proportion_row(successes: int, n: int, *, alpha: float = 0.05) -> dict[str, float]:
    """A proportion with its interval, in the shape every table here expects."""
    lo, hi = wilson_ci(successes, n, alpha=alpha)
    return {
        "n": int(n),
        "events": int(successes),
        "estimate": float(successes) / n if n else float("nan"),
        "lo": lo,
        "hi": hi,
    }
