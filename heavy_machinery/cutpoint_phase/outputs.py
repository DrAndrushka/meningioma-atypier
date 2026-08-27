"""The last step: write everything the manuscript needs, from tables already computed.

Nothing here calculates. Every number arrives from the notebook, already
estimated and already formatted, and this module only decides where it lands —
which is the point. A writer that recomputes is a second analysis, free to
disagree with the first, and the disagreement would surface as two documents
quoting different values for the same quantity.

Three destinations from one set of objects:

``tables/*.docx``   what the journal receives. Borders stripped, width asserted,
                    body type, a ``Note:—`` block under each.
``figures/*.tif``   what the journal receives. 600 dpi, LZW, Arial, greyscale.
``figures/*.png``   the same figure objects, for the HTML and for reuse.
``*.html``          the proof sheet: dashboard, tables, figures, all in one file
                    with images inlined, so it can be emailed or archived whole.
``manifest.json``   what produced them: commit, library versions, seeds, the
                    cohort's hash and a hash of every file above.

The thumbnails on the dashboard are drawn only for measurements that failed a
condition. On a card that met everything the curves would decorate a verdict the
numbers already carry; on one that failed they are the fastest way to see why.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import ajnr_style as aj
import docx_tables as dx
import figures as fg
import loading as ld
import manifest as mf
import manuscript_tables as mt
import report_html as rh

# plot_style lives in modelling_phase; importing heavy_machinery.config is what
# puts every phase directory on sys.path, exactly as performance_plots does.
try:
    from plot_style import prune_embedded_figures
except ModuleNotFoundError:  # pragma: no cover - path not yet extended
    from heavy_machinery.config import load as _load_config  # noqa: F401
    from plot_style import prune_embedded_figures
from measurements import MEASUREMENTS_BY_COL

FIGURE_TITLE = ("Figure 1. Discrimination, the flatness of the optimum, and the "
                "resulting uncertainty in each cut-point")
FIGURE_CAPTION = (
    "A, receiver operating characteristic curves for the five measurements, "
    "with the published cut-point marked on each. B, Youden J across every "
    "candidate cut-point, plotted against the cut-point's position in the "
    "cohort so that measurements in different units share one axis; broad "
    "plateaus indicate that many candidate values score almost identically. "
    "C, each cut-point with its bootstrap 95% interval on the same percentile "
    "axis, annotated in native units; grey indicates an interval wider than "
    "the measurement's own interquartile range.")
FIGURE_PLAIN = (
    "One line per measurement. The higher a curve bulges, the better it "
    "separates the two grades; a flat peak in B means the exact cut-point "
    "barely matters."
)
FIGURE2_TITLE = "Figure 2. Decision-curve analysis of the five cut-points"
FIGURE2_PLAIN = (
    "Whichever line is highest is the best thing to do at that level of "
    "concern."
)
FIGURE2_CAPTION = (
    "A, net benefit of each cut-point across threshold probabilities, with the "
    "treat-all strategy (dashed) and the treat-none strategy (net benefit 0 at "
    "every threshold). A cut-point is worth following only where its line lies "
    "above both. B, the difference in net benefit between using the "
    "measurement as a continuous risk estimate and using its cut-point; above "
    "zero the number does better than the yes/no rule. Each line in B is drawn "
    "only over thresholds at which at least one of the two strategies beats "
    "treating all and treating none, since outside that range the continuous "
    "model assigns every patient, or no patient, a risk above the threshold "
    "and the comparison reduces to one against a strategy needing no "
    "measurement. Both panels are "
    "corrected for optimism over 1000 patient-level bootstrap resamples with "
    "the cut-point re-derived and the model refitted within each.")
# Counted from the criteria themselves: this paragraph said "four conditions"
# for a run that tested five, and no test caught it.
DASHBOARD_LEAD = (
    f"Each measurement was tested against {mt.CRITERIA_TOTAL} criteria in "
    "order — an association with grade at all, a bend in risk, that bend "
    "surviving a change of scale, a breakpoint that survives correction for "
    "having searched every candidate, and a break that pays for the parameters "
    f"it adds. A measurement meeting all {mt.CRITERIA_TOTAL} is shown with its "
    "estimated breakpoint; one that does not is shown with its derived "
    "cut-point, where it stopped, and the curves that show why — the same "
    "patients plotted against the measurement and against its logarithm.")


def build(df: pd.DataFrame, *, eligible: pd.DataFrame, cutpoints: dict,
          fits: dict, separation: pd.DataFrame, nonlinearity: pd.DataFrame,
          wobble: pd.DataFrame, agreement: pd.DataFrame,
          dichotomy: pd.DataFrame, imputation: pd.DataFrame,
          segmented: pd.DataFrame, presence: pd.DataFrame,
          decision: pd.DataFrame, curves: dict,
          output_root: Path | str = "output") -> dict[str, object]:
    """Write the tables, the figures, the HTML and the manifest."""
    out = Path(output_root) / "cutpoints"

    t1 = mt.table_one(eligible, dichotomy=dichotomy, nonlinearity=nonlinearity,
                      segmented=segmented)
    s1 = mt.supplemental_s1(eligible, agreement=agreement, wobble=wobble,
                            imputation=imputation)
    s2 = mt.supplemental_s2(eligible, dichotomy=dichotomy)
    s3 = mt.supplemental_s3(presence)
    s4 = mt.supplemental_s4(decision)

    verdicts = mt.threshold_verdicts(
        eligible, dichotomy=dichotomy, nonlinearity=nonlinearity,
        segmented=segmented, wobble=wobble, presence=presence)
    by_col = {r["col"]: r for _, r in segmented.iterrows()}
    for entry in verdicts:
        if entry["supported"]:
            continue
        entry["thumbnail"] = fg.verdict_thumbnail(
            df, MEASUREMENTS_BY_COL[entry["col"]], fits, by_col.get(entry["col"]))

    figure, _ = fg.main_figure(df, eligible, separation, wobble, cutpoints)
    figure_paths = aj.save_figure(figure, out / "figures" / "fig_1_cutpoints",
                                  tight=False)
    png = next(p for p in figure_paths if p.suffix == ".png")

    order = [r["col"] for _, r in decision.iterrows()] if not decision.empty else []
    figure2, _ = fg.decision_figure(curves, order)
    figure2_paths = aj.save_figure(figure2, out / "figures" / "fig_2_decision",
                                   tight=False)
    png2 = next(p for p in figure2_paths if p.suffix == ".png")

    table_paths = [
        dx.write_tables(out / "tables" / "table_1.docx", [dict(
            frame=t1, title=mt.T1_TITLE, note=mt.t1_footnote(),
            index_header="Measurement", column_widths=mt.T1_WIDTHS)]),
        dx.write_tables(out / "tables" / "supplemental_tables.docx", [
            dict(frame=s1, title=mt.S1_TITLE, note=mt.s1_footnote(),
                 index_header="Measurement"),
            dict(frame=s2, title=mt.S2_TITLE, note=mt.s2_footnote(),
                 index_header="Measurement"),
            dict(frame=s3, title=mt.S3_TITLE, note=mt.s3_footnote(),
                 index_header="Measurement"),
            dict(frame=s4, title=mt.S4_TITLE, note=mt.s4_footnote(),
                 index_header="Measurement")]),
    ]

    facts = ld.cohort_facts(df)
    html_path = rh.write(out / "cutpoint_report.html", [
        rh.dashboard_section(verdicts, lead=DASHBOARD_LEAD),
        rh.table_section(mt.T1_TITLE, t1, note=mt.t1_footnote(),
                         index_header="Measurement", lead=mt.describe_t1(t1)),
        rh.figure_section(FIGURE_TITLE, png, caption=FIGURE_CAPTION,
                          plain=FIGURE_PLAIN),
        rh.table_section(mt.S1_TITLE, s1, note=mt.s1_footnote(),
                         index_header="Measurement", lead=mt.describe_s1(s1)),
        rh.table_section(mt.S2_TITLE, s2, note=mt.s2_footnote(),
                         index_header="Measurement", lead=mt.describe_s2(s2)),
        rh.table_section(mt.S3_TITLE, s3, note=mt.s3_footnote(),
                         index_header="Measurement",
                         lead=mt.describe_s3(presence)),
        rh.table_section(mt.S4_TITLE, s4, note=mt.s4_footnote(),
                         index_header="Measurement",
                         lead=mt.describe_s4(decision)),
        rh.figure_section(FIGURE2_TITLE, png2, caption=FIGURE2_CAPTION,
                          plain=FIGURE2_PLAIN),
    ], title="Meningioma cut-points — manuscript tables and figures",
       subtitle=(f"{facts['patients']} patients, {facts['high_grade']} WHO "
                 f"grade 2–3 ({float(facts['prevalence']):.1%}). Proof sheet: "
                 "the journal receives the .docx tables and the TIFF figures; "
                 "this page is for reading them together."))

    # Both figures ride inside the page as base64, so the PNGs beside it are a
    # second copy. The TIFs stay: those are what the journal receives, and the
    # page does not contain them.
    pruned, freed, _kept = prune_embedded_figures(
        html_path, roots=[out / "figures"])
    if pruned:
        print(f"Pruned {pruned} embedded figure(s), "
              f"{freed / 1048576:.1f} MB reclaimed")
    # Filtered before the manifest is built, so it never records a file that
    # was deleted a moment later.
    written = [p for p in [*table_paths, *figure_paths, *figure2_paths, html_path]
               if p.exists()]
    # Read off the imputation step rather than restated: m is a property of the
    # cleaning run this phase happened to load, not a setting this phase owns.
    draws = (int(imputation["n_draws"].max())
             if imputation is not None and not imputation.empty
             and "n_draws" in imputation else None)
    record = mf.build(df, cutpoints=cutpoints, facts=facts, written=written,
                      output_root=output_root, n_imputations=draws)
    manifest_path = mf.write(record, out / "manifest.json")

    return {"tables": {"table_1": t1, "s1": s1, "s2": s2, "s3": s3, "s4": s4},
            "verdicts": verdicts,
            "table_paths": table_paths,
            "figure_paths": [*figure_paths, *figure2_paths],
            "html": html_path,
            "manifest": record,
            "manifest_path": manifest_path,
            "audits": {p.name: dx.audit(p) for p in table_paths}}


def describe(result: dict) -> str:
    """What was written and whether the Word files meet the journal's rules."""
    lines = []
    for path in result["table_paths"]:
        audit = result["audits"][path.name]
        ok = (audit["visible_borders"] == 0
              and audit["max_width_in"] <= dx.MAX_WIDTH_IN
              and audit["font_sizes_pt"] == [dx.BODY_PT]
              and not audit["landscape"] and audit["has_note"])
        lines.append(
            f"{'✅' if ok else '❌'} {path}  "
            f"({audit['tables']} table(s), {audit['max_width_in']:.2f} in, "
            f"borders {audit['visible_borders']})")
    for path in result["figure_paths"]:
        lines.append(f"✅ {path}")
    lines.append(f"✅ {result['html']}")
    lines.append(f"✅ {result['manifest_path']}")
    lines.append(mf.describe(result["manifest"]))
    supported = [v["measurement"] for v in result["verdicts"] if v["supported"]]
    lines.append(
        f"\nThreshold supported: {', '.join(supported) if supported else 'none'}."
        f" Not supported: "
        f"{', '.join(v['measurement'] for v in result['verdicts'] if not v['supported'])}.")
    return "\n".join(lines)
