"""Step 10 — all ten side by side, and the two pairs that might be the same thing.

Nine of the previous steps looked at one measurement at a time. This one puts
every version of every measurement on one axis so they can be ranked against
each other: five raw numbers and five yes/no answers, one row each.

**Ranking is not the same as choosing.** Two AUCs a hundredth apart are not
meaningfully different in 352 patients, and a table sorted by AUC invites the
reader to treat first place as a winner. So the ranking carries a second
question for the pairs where it actually matters.

**The two pairs.** Tumour volume and maximum diameter are both size; edema
volume and edema index are both edema. Within each pair the two measurements are
close enough that a reader will ask whether the study needs both — and they are
measured on the same patients, so their AUCs cannot be compared by looking at
whether the intervals overlap. Overlapping intervals routinely hide a real
difference when two scores are correlated, and DeLong's paired test is what
answers it properly.

**Only two comparisons are made, and they were named in advance.** Comparing all
forty-five pairs would guarantee something looked significant. The pairs are
declared here, corrected together for multiplicity, and the correction is
applied to a family of two rather than to whatever happened to be interesting.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from accuracy import flag
from eda import benjamini_hochberg
from measurements import MEASUREMENTS_BY_COL, Measurement, stratum_mask
from separation import auc_with_ci, delong_compare, oriented_score

OUTCOME = "high_grade"

FORM_CONTINUOUS = "raw number"
FORM_BINARY = "cut-point"

# Named before looking. Each pair is two ways of measuring one thing, where a
# reader will reasonably ask whether the paper needs both.
PRE_SPECIFIED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("tumor_volume", "max_diameter_cm", "both measure size"),
    ("edema_volume_cm3", "edema_index", "both measure edema"),
)


def ranked_table(df: pd.DataFrame, eligible: pd.DataFrame,
                 cutpoints: dict[str, float], *,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Every measurement in both forms, one row each, best first.

    Denominators differ between rows because the measurements have different
    amounts of missing data, so ``n`` is carried on every row. A rank read
    without it would compare a result from 309 patients against one from 352 as
    though they were the same study.
    """
    rows = []
    for _, row in eligible.iterrows():
        m: Measurement = MEASUREMENTS_BY_COL[row["col"]]
        sub = df.loc[stratum_mask(df, m, row["stratum"])]
        x = pd.to_numeric(sub[m.col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[OUTCOME], errors="coerce").to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok].astype(int)

        base = {"measurement": row["measurement"], "col": m.col,
                "stratum": row["stratum"], "claim": row.get("claim", ""),
                "n": int(y.size), "n_high": int(y.sum())}
        rows.append({**base, "form": FORM_CONTINUOUS, "cutoff": np.nan,
                     "label": f"{row['measurement']} ({FORM_CONTINUOUS})",
                     **auc_with_ci(y, x, m.direction, alpha=alpha)})
        if m.col in cutpoints:
            cutoff = cutpoints[m.col]
            binary = flag(x, cutoff, m.direction).astype(float)
            rows.append({**base, "form": FORM_BINARY,
                         "cutoff": m.round(cutoff),
                         "label": m.rule_text(cutoff),
                         **auc_with_ci(y, binary, "higher", alpha=alpha)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("auc", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def compare_pair(df: pd.DataFrame, col_a: str, col_b: str, *,
                 form: str = FORM_CONTINUOUS,
                 cutpoints: dict[str, float] | None = None,
                 alpha: float = 0.05) -> dict[str, object]:
    """DeLong comparison of two measurements on the patients who have both.

    Restricting to complete pairs is what makes the test paired at all. Scoring
    each measurement on its own patients and then differencing would compare two
    different cohorts and call the result a within-patient contrast.
    """
    m_a, m_b = MEASUREMENTS_BY_COL[col_a], MEASUREMENTS_BY_COL[col_b]
    x_a = pd.to_numeric(df[col_a], errors="coerce").to_numpy()
    x_b = pd.to_numeric(df[col_b], errors="coerce").to_numpy()
    y = pd.to_numeric(df[OUTCOME], errors="coerce").to_numpy()
    ok = np.isfinite(x_a) & np.isfinite(x_b) & np.isfinite(y)
    x_a, x_b, y = x_a[ok], x_b[ok], y[ok].astype(int)

    out = {"a": m_a.label, "b": m_b.label, "col_a": col_a, "col_b": col_b,
           "form": form, "n_both": int(y.size), "n_high": int(y.sum())}
    if y.size < 20 or y.sum() < 5 or (y.size - y.sum()) < 5:
        out.update({"auc_a": np.nan, "auc_b": np.nan, "difference": np.nan,
                    "difference_lo": np.nan, "difference_hi": np.nan,
                    "p": np.nan})
        return out

    if form == FORM_BINARY:
        cutpoints = cutpoints or {}
        if col_a not in cutpoints or col_b not in cutpoints:
            out.update({"auc_a": np.nan, "auc_b": np.nan, "difference": np.nan,
                        "difference_lo": np.nan, "difference_hi": np.nan,
                        "p": np.nan})
            return out
        score_a = flag(x_a, cutpoints[col_a], m_a.direction).astype(float)
        score_b = flag(x_b, cutpoints[col_b], m_b.direction).astype(float)
    else:
        score_a = oriented_score(x_a, m_a.direction)
        score_b = oriented_score(x_b, m_b.direction)

    out.update({k: v for k, v in
                delong_compare(y, score_a, score_b, alpha=alpha).items()
                if k != "correlated"})
    return out


def pairwise_table(df: pd.DataFrame, *,
                   pairs: Sequence[tuple[str, str, str]] = PRE_SPECIFIED_PAIRS,
                   forms: Sequence[str] = (FORM_CONTINUOUS, FORM_BINARY),
                   cutpoints: dict[str, float] | None = None,
                   alpha: float = 0.05) -> pd.DataFrame:
    """The pre-specified comparisons, corrected together for multiplicity.

    The correction is Benjamini-Hochberg over this family only — the same
    procedure the EDA tables use, imported rather than reimplemented so the two
    cannot drift apart.
    """
    rows = []
    for col_a, col_b, because in pairs:
        for form in forms:
            entry = compare_pair(df, col_a, col_b, form=form,
                                 cutpoints=cutpoints, alpha=alpha)
            entry["because"] = because
            rows.append(entry)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_fdr"] = benjamini_hochberg(table["p"]).to_numpy()
    table["distinguishable"] = table["p_fdr"] < alpha
    return table


def describe_ranking(table: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    """One line: what leads, and whether the close pairs are actually different."""
    if table.empty:
        return "Nothing to rank."
    best = table.iloc[0]
    line = (f"Best discrimination: {best['label']}, AUC {best['auc']:.2f} "
            f"(95% CI {best['auc_lo']:.2f}-{best['auc_hi']:.2f}), "
            f"n={int(best['n'])}.")
    if pairwise.empty:
        return line
    scored = pairwise.dropna(subset=["p_fdr"])
    same = scored[~scored["distinguishable"]]
    different = scored[scored["distinguishable"]]
    if not same.empty:
        named = "; ".join(f"{r['a']} vs {r['b']} as {r['form']}"
                          for _, r in same.iterrows())
        line += (f" Not distinguishable on these patients, so the paper does "
                 f"not need both — {named}.")
    if not different.empty:
        named = "; ".join(
            f"{r['a']} beats {r['b']} by {abs(r['difference']):.3f}"
            if r["difference"] > 0 else
            f"{r['b']} beats {r['a']} by {abs(r['difference']):.3f}"
            for _, r in different.iterrows())
        line += f" Genuinely different — {named}."
    return line
