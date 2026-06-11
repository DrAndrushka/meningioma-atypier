"""Step 08 — call build_report and write report.html. Adds years to title if filtered."""
from __future__ import annotations

from pathlib import Path

from report import ReportConfig, build_report, write_html


def run_report(
    *,
    output_root: Path,
    report_title: str,
    report_author: str,
    report_path: Path,
    analysis_years: list[int] | None,
    eda_targets: list,
) -> Path:
    title = report_title
    if analysis_years is not None:
        title += f" ({analysis_years} cohort)"

    cfg = ReportConfig(
        output_root=output_root,
        title=title,
        author=report_author,
        targets=tuple(eda_targets),
    )
    write_html(build_report(cfg), report_path)
    print(f"Report written: {report_path.resolve()}")
    return report_path
