"""What produced these numbers — the record a reviewer would have to ask for.

A manuscript table is a claim about a computation. Six months after submission,
"why does Table 1 say .009 when I get .011" is a question somebody will ask, and
the honest answers are all facts about the run rather than about the statistics:
a different commit, a different scipy, a re-exported cohort, a seed nobody wrote
down. This module writes those facts down, once, beside the outputs.

Four kinds of thing go in:

**The code.** Git commit, whether the tree was dirty when it ran, the Python
version and the version of every library whose numerical output the tables
depend on. A dirty tree is recorded rather than refused — refusing would stop
work in the middle of an ordinary afternoon — but it is recorded loudly, because
a commit hash means nothing if the files did not match it.

**The choices.** Seeds, resample counts, the criteria that were swept, the
significance levels, which measurements are analysed on a log scale. These live
in half a dozen modules by design, each next to the code that uses it; the
manifest is the one place they appear together, which is what makes it possible
to check that they agree.

**The input.** The cohort file, its size, and a SHA-256 of its bytes. Patient
data never leaves this machine, and a hash is not patient data — it is the one
thing that can prove two runs read the same rows without disclosing any of them.

**The output.** Every file written, with size and hash. :func:`verify` re-reads
them later and says which have changed, so "are the tables in the submission the
ones this run produced" stops being a question of memory.

No number in here is computed by this module. It reads what the run already
decided, which is the only way a provenance record can be trusted: one that
recomputes anything is capable of disagreeing with the thing it documents.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_VERSION = 1

# Libraries whose version can move a printed digit. Deliberately not "everything
# pip has installed": a list nobody can read is a list nobody checks, and a
# change in a plotting backend does not move a P value.
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn",
                    "matplotlib", "docx")

# Reported as the package is named on PyPI, not as it is imported.
_DISTRIBUTION_NAMES = {"docx": "python-docx", "scikit-learn": "scikit-learn"}

CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    """Hash a file without reading it whole into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path | str) -> dict[str, object]:
    """Size and hash of one file, or a record saying it was not there."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True,
            "bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _package_versions() -> dict[str, str]:
    """Installed version of each tracked library, or 'not installed'."""
    from importlib import metadata
    out = {}
    for name in TRACKED_PACKAGES:
        dist = _DISTRIBUTION_NAMES.get(name, name)
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def _git(*args: str) -> str | None:
    """One git command, or None outside a repository."""
    try:
        result = subprocess.run(("git",) + args, capture_output=True,
                                text=True, timeout=10,
                                cwd=str(Path(__file__).resolve().parent))
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_state() -> dict[str, object]:
    """Commit, branch and whether the working tree matched it."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "uncommitted_files": (0 if not status else
                              len([ln for ln in status.splitlines() if ln])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }


def settings() -> dict[str, object]:
    """Every seed, count and threshold the phase was run with.

    Imported from the modules that own them rather than restated, so a manifest
    cannot record a seed the run did not use. That is the entire discipline here
    and it is worth the awkward import block.
    """
    import criteria as cr
    import eligibility as el
    import manuscript_tables as mt
    import nonlinearity as nl
    import segmented as sg
    import wobble as wb
    from bend_location import BOUNDARY_PERCENTILE
    from collinearity import TOGETHER
    from scales import LOG1P_COLUMNS

    return {
        "seed": wb.SEED,
        "bootstrap_resamples": wb.N_BOOTSTRAP,
        "bootstrap_min_held_out": wb.MIN_HELD_OUT,
        "spline_knots": nl.DEFAULT_N_KNOTS,
        "spline_knot_quantiles": list(nl.KNOT_QUANTILES[nl.DEFAULT_N_KNOTS]),
        "criteria_swept": list(cr.OPTIMUM_SEEKING),
        "criteria_pre_specified": list(cr.PRE_SPECIFIED),
        "headline_criterion": cr.YOUDEN,
        "separation_floor": el.SEPARATION_FLOOR,
        "boundary_percentile": BOUNDARY_PERCENTILE,
        "collinearity_flag_rho": TOGETHER,
        "segmented_deviance_cutoff": float(sg.DEVIANCE_CUTOFF),
        "log1p_columns": sorted(LOG1P_COLUMNS),
        "grading_thresholds": {
            "spread_limit_x_iqr": mt.SPREAD_LIMIT,
            "bootstrap_limit_x_iqr": mt.BOOTSTRAP_LIMIT,
            "retained_floor": mt.RETAINED_FLOOR,
            "loss_alpha": mt.LOSS_ALPHA,
            "bend_alpha": mt.BEND_ALPHA,
            "break_alpha": mt.BREAK_ALPHA,
        },
    }


def build(df: pd.DataFrame, *, cutpoints: dict, facts: dict,
          written: list[Path | str], output_root: Path | str = "output",
          n_imputations: int | None = None) -> dict[str, object]:
    """Assemble the manifest for one run. Writes nothing."""
    root = Path(output_root)
    cohort = root / "datasets" / "unimputed_df.parquet"
    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": "cutpoint",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code_state(),
        "settings": {**settings(), "n_imputations": n_imputations},
        "cohort": {
            "source": file_record(cohort),
            "patients": int(facts.get("patients", len(df))),
            "high_grade": int(facts.get("high_grade", 0)),
            "prevalence": float(facts.get("prevalence", float("nan"))),
            "columns": int(df.shape[1]),
        },
        "frozen_cutpoints": {k: float(v) for k, v in sorted(cutpoints.items())},
        "outputs": [file_record(p) for p in written],
    }


def write(manifest: dict, path: Path | str) -> Path:
    """Write the manifest as indented JSON. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def verify(path: Path | str) -> list[str]:
    """Re-hash everything a manifest lists. Returns one line per disagreement.

    An empty list means the files on disk are the files this run wrote. It is
    the cheapest possible answer to "is the figure in the submission the one the
    numbers came from", and the answer stops being available the moment nobody
    records the hashes.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    records = list(manifest.get("outputs", []))
    source = manifest.get("cohort", {}).get("source")
    if source:
        records.append(source)
    for record in records:
        if not record.get("exists"):
            continue
        target = Path(record["path"])
        if not target.exists():
            problems.append(f"missing now: {target}")
        elif _sha256(target) != record.get("sha256"):
            problems.append(f"changed since the run: {target}")
    return problems


def describe(manifest: dict) -> str:
    """The three lines worth printing in a notebook."""
    code = manifest["code"]
    commit = (code["commit"] or "no git")[:8]
    dirty = ("" if not code.get("dirty") else
             f", {code['uncommitted_files']} uncommitted file(s)")
    cohort = manifest["cohort"]
    return (
        f"📌 manifest: commit {commit}{dirty}, Python {code['python']}, "
        f"seed {manifest['settings']['seed']}, "
        f"{manifest['settings']['bootstrap_resamples']} resamples\n"
        f"   cohort {cohort['patients']} patients "
        f"({cohort['high_grade']} high grade), "
        f"sha256 {(cohort['source'].get('sha256') or '—')[:8]}\n"
        f"   {len(manifest['outputs'])} output file(s) hashed")
