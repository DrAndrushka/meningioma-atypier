"""Build HTML report and print output artifact summary.

Wraps ``report.build_report`` / ``write_html``. Appends ``ANALYSIS_YEARS`` to the
title when the cohort was year-filtered.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from plot_style import prune_embedded_figures
from report import ReportConfig, build_report, write_html


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def print_output_summary(output_root: Path | str) -> None:
    """Print emoji summary of artifacts under ``output_root``."""
    root = Path(output_root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)

    section_meta = {
        "cleaning": ("🧹", "Cleaning"),
        "schema": ("📋", "Schema"),
        "datasets": ("💾", "Model datasets"),
        "dda": ("📊", "DDA"),
        "missingness": ("🕳️", "Missingness"),
        "eda": ("🔬", "EDA"),
        "inferential": ("🧮", "Multivariable"),
        "report": ("🧾", "Report"),
    }
    section_order = list(section_meta) + ["_other"]

    by_section: dict[str, dict] = defaultdict(
        lambda: {"bytes": 0, "subs": defaultdict(list), "exts": defaultdict(int)},
    )
    for path in files:
        rel = path.relative_to(root)
        top = rel.parts[0] if len(rel.parts) > 1 else "_root"
        sub = rel.parts[1] if len(rel.parts) > 2 else "."
        bucket = by_section[top]
        bucket["bytes"] += path.stat().st_size
        bucket["subs"][sub].append(path)
        bucket["exts"][path.suffix.lower() or "(none)"] += 1

    print(f"\n📦 Pipeline outputs — {root.resolve()}")
    print("═" * 72)
    print(f"📁 {len(files)} files · {_human_bytes(total_bytes)} total\n")

    for top in sorted(
        by_section,
        key=lambda name: section_order.index(name) if name in section_order else len(section_order),
    ):
        info = by_section[top]
        emoji, label = section_meta.get(top, ("📁", top.replace("_", " ").title()))
        n_files = sum(len(paths) for paths in info["subs"].values())
        ext_bits = ", ".join(
            f"{count} {ext.lstrip('.') or 'file'}" for ext, count in sorted(info["exts"].items())
        )
        sub_bits = ", ".join(
            f"{sub}: {len(paths)}" for sub, paths in sorted(info["subs"].items()) if sub != "."
        )
        tail = f" — {sub_bits}" if sub_bits else ""
        print(
            f"{emoji} {label:<22} {n_files:>4} files · {_human_bytes(info['bytes']):>8}  "
            f"({ext_bits}){tail}"
        )

    print("\n── ✅ Key artifacts ──")
    key_artifacts = [
        ("🧾", "report/report.html"),
        ("💾", "datasets/unimputed_df.parquet"),
        ("💾", "datasets/mice_imputed_df.parquet"),
        ("💾", "datasets/simple_imputed_df.parquet"),
        ("💾", "datasets/manifest.json"),
        ("🧮", "inferential/tables/inferential_summary.csv"),
        ("🧮", "inferential/tables/multivariable_cases.csv"),
        ("🔬", "eda/tables/associations.csv"),
        ("🔬", "eda/tables/diagnostic_accuracy.csv"),
        ("🕳️", "missingness/mice/manifest.json"),
    ]
    for emoji, rel in key_artifacts:
        artifact = root / rel
        if artifact.exists():
            print(f"  {emoji} {rel} ({_human_bytes(artifact.stat().st_size)})")

    print("\n── 📄 Tables, datasets & report (non-figure files) ──")
    for artifact in files:
        rel = artifact.relative_to(root)
        if "figures" in rel.parts:
            continue
        if artifact.suffix.lower() not in {".csv", ".json", ".parquet", ".html"}:
            continue
        print(f"  • {rel} ({_human_bytes(artifact.stat().st_size)})")

    print("\n── 🖼️ Figure counts ──")
    figure_dirs = sorted({artifact.parent for artifact in files if artifact.suffix.lower() == ".png"})
    for fig_dir in figure_dirs:
        figs = sorted(fig_dir.glob("*.png"))
        print(
            f"  • {fig_dir.relative_to(root)} — {len(figs)} png · "
            f"{_human_bytes(sum(p.stat().st_size for p in figs))}"
        )


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

    # The report carries every figure inside it, so the PNGs on disk are a
    # second copy. Only files whose bytes are provably in the page are removed;
    # TIFs are never touched, so ATYPIER_FIGURES=submission still delivers them.
    n, freed, kept = prune_embedded_figures(report_path, roots=[output_root])
    if n:
        print(f"Pruned {n} embedded figure(s), {freed / 1048576:.1f} MB reclaimed")
    stray = [p for p in kept if "figures" in p.parts]
    if stray:
        print(f"  kept {len(stray)} figure(s) not embedded in this report:")
        for p in stray:
            print(f"    · {p.relative_to(output_root)}")
    return report_path
