"""Step 5c — which measurements carry into cut-point derivation, and as what.

Two questions have now been asked of every measurement, and they fail
independently:

*Does it separate the grades?* (step 3, AUC) — if the interval includes 0.50,
the measurement has not been shown to carry information at all. There is
nothing to cut. It stops here.

*Does risk bend?* (steps 4-5) — if not, there is no biological boundary. But a
measurement can separate the grades perfectly well without having one: risk
simply rises the whole way.

Confusing those two is how a paper overclaims. A measurement with no bend still
earns a cut-point — it just earns a different *sentence*:

``threshold``        a real bend, located inside the data, on both scales. The
                     number marks a change in the biology.
``threshold (scale-dependent)``
                     a real bend in clinical units that disappears on the log
                     scale. Quotable, but the caveat travels with it.
``operating point``  no bend. The number is a chosen place to stand on a smooth
                     slope — the way "hypertension starts at 140/90" is a
                     convention, not a cliff. Fully reportable, as long as it is
                     never called a threshold.

The claim is attached here, once, and carried into every table downstream, so
the wording cannot drift between a figure caption and the Results text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLAIM_THRESHOLD = "threshold"
CLAIM_THRESHOLD_SCALED = "threshold (scale-dependent)"
CLAIM_OPERATING_POINT = "operating point"

# The lower end of the AUC interval must clear a coin flip. Judged on the
# interval rather than the point estimate: an AUC of 0.58 whose interval runs
# from 0.49 is not evidence of separation, it is evidence of a small study.
SEPARATION_FLOOR = 0.50


def carry_forward(separation: pd.DataFrame, bend: pd.DataFrame) -> pd.DataFrame:
    """Join step 3's separation with step 5's bend and decide what each row is.

    Returns every row, including the ones that stop here — a measurement that
    was tested and dropped is a result, and a table that silently omits it
    leaves the reader unable to tell testing from selective reporting.
    """
    sep = separation.set_index(["col", "stratum"])
    bnd = bend.set_index(["col", "stratum"])
    rows = []
    for key, s in sep.iterrows():
        b = bnd.loc[key] if key in bnd.index else None
        separates = bool(np.isfinite(s["auc_lo"]) and s["auc_lo"] > SEPARATION_FLOOR)
        quotable = bool(b is not None and b["quotable"])
        agree = bool(b is not None and b["scales_agree"])

        if not separates:
            claim, reason = "", "interval includes 0.50 — no separation shown"
        elif not quotable:
            claim, reason = CLAIM_OPERATING_POINT, (
                "separates the grades, but risk climbs smoothly — a chosen "
                "place to stand, not a boundary")
        elif not agree:
            claim, reason = CLAIM_THRESHOLD_SCALED, (
                "bends in clinical units but not on the log scale — quotable "
                "with the caveat attached")
        else:
            claim, reason = CLAIM_THRESHOLD, (
                "bends on both scales, inside the data — the number marks a "
                "change in the biology")

        rows.append({
            "measurement": s["measurement"],
            "col": key[0],
            "stratum": key[1],
            "n": s["n"],
            "auc": s["auc"],
            "auc_lo": s["auc_lo"],
            "separates": separates,
            "bend_is_real": bool(b["bend_is_real"]) if b is not None else False,
            "knee_quotable": quotable,
            "scales_agree": agree,
            "carries_forward": separates,
            "claim": claim,
            "reason": reason,
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["carries_forward", "auc"],
                             ascending=[False, False]).reset_index(drop=True)


def eligible(table: pd.DataFrame) -> pd.DataFrame:
    """Only the rows a cut-point may be derived for."""
    return table[table["carries_forward"]].reset_index(drop=True)


def describe_eligibility(table: pd.DataFrame) -> str:
    """One line: what goes on, as what, and what stops here."""
    kept = table[table["carries_forward"]]
    dropped = table[~table["carries_forward"]]
    if kept.empty:
        return "Nothing carries forward — no measurement separates the grades."
    by_claim = kept.groupby("claim")["measurement"].apply(list)
    parts = [f"{claim}: {', '.join(names)}"
             for claim, names in by_claim.items()]
    line = f"Carried into cut-point derivation — {'; '.join(parts)}."
    if not dropped.empty:
        line += (f" Stops here: {', '.join(dropped['measurement'])} "
                 "(interval includes 0.50).")
    return line
