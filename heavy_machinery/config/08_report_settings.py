"""§08 — build and write HTML report."""
from __future__ import annotations

from pathlib import Path

from dda import plot_distribution_by_year
from report import ReportConfig, build_report, write_html


def run_report(
    df,
    *,
    output_root: Path,
    report_title: str,
    report_author: str,
    report_path: Path,
    focus_predictor: str | None,
    focus_reference_level,
    analysis_years: list[int] | None,
    year_column: str,
    eda_targets: list,
) -> Path:
    title = report_title
    if analysis_years is not None:
        title += f" ({analysis_years} cohort)"

    if analysis_years is None and focus_predictor and year_column in df.columns:
        by_year = plot_distribution_by_year(
            df, focus_predictor, year_column, output_root / "dda" / "figures"
        )
        if by_year:
            print(f"Focus by-year figure: {by_year}")

    cfg = ReportConfig(
        output_root=output_root,
        title=title,
        author=report_author,
        targets=tuple(eda_targets),
        focus_predictor=focus_predictor,
        focus_reference_level=focus_reference_level,
        year_column=year_column,
    )
    write_html(build_report(cfg), report_path)
    print(f"Report written: {report_path.resolve()}")
    return report_path
