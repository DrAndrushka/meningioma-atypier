"""Re-run the marker panel and prove it still matches the recorded baseline.

``output/panel/baseline_2026-08-04/`` holds the thirteen CSVs the pre-numpy
code produced on the real cohort at full budget — the run that took 102
minutes. Any change to :mod:`combinations`, :mod:`rule_matrix` or
:mod:`marker_panel` that moves one of those numbers is a change to a published
result, and this script is how you find that out in under a minute.

    python scripts/check_panel_against_baseline.py

Exits 0 and prints the wall time if every table matches, 1 otherwise.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), *(str(REPO / p) for p in (
    "heavy_machinery", "heavy_machinery/cleaning_phase",
    "heavy_machinery/modelling_phase", "heavy_machinery/threshold_phase"))]

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import pandas as pd                    # noqa: E402

from dataset_handoff import load_modelling_handoff        # noqa: E402
from missingness_resolution import load_imputed_frames    # noqa: E402
from marker_panel import run_marker_panel                 # noqa: E402
from model_calculator import load_model_artifact          # noqa: E402

TARGET = "high_grade"
NON_IMAGING = {"sex_male", "hist_necrosis", "progesterone_pos"}
BASELINE = REPO / "output" / "panel" / "baseline_2026-08-04"


def main() -> int:
    root = REPO / "output"
    if not BASELINE.is_dir():
        print(f"❌ no baseline at {BASELINE}")
        return 1

    df, _, _ = load_modelling_handoff(root)
    accuracy = pd.read_csv(root / "eda" / "tables" / "diagnostic_accuracy.csv")

    # Loaded exactly as the notebook's §04.5 cell does. Without them, tables 10
    # and 13 come out empty and the comparison proves nothing.
    art_dir = root / "inferential" / "model_artifacts"
    artifacts = {
        p.stem.replace("_model", "").replace(f"{TARGET}_", ""): load_model_artifact(p)
        for p in sorted(art_dir.glob("*_model.json"))
    } if art_dir.is_dir() else {}
    print(f"{len(artifacts)} model artifacts loaded")

    scratch = Path(tempfile.mkdtemp(prefix="panel-check-"))
    keep_scratch = False
    try:
        start = time.perf_counter()
        run_marker_panel(
            df, target=TARGET, accuracy_table=accuracy, output_root=scratch,
            exclude=NON_IMAGING, artifacts=artifacts,
            draws=load_imputed_frames(root),
        )
        elapsed = time.perf_counter() - start
        fresh = scratch / "panel" / "tables"

        failures = []
        expected = sorted(p.name for p in BASELINE.glob("*.csv"))
        for name in expected:
            new = fresh / name
            if not new.is_file():
                failures.append(f"{name}: not produced")
                continue
            if new.read_bytes() != (BASELINE / name).read_bytes():
                failures.append(f"{name}: differs from baseline")

        print(f"\n{len(expected) - len(failures)}/{len(expected)} tables identical")
        print(f"run_marker_panel: {elapsed:.1f}s")
        if failures:
            print("\n❌ " + "\n❌ ".join(failures))
            print(f"\nfresh tables kept for inspection: {fresh}")
            keep_scratch = True
            return 1
        print("✅ every table matches the baseline")
        return 0
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
