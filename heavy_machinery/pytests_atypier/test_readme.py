"""The README has to keep describing the repo that exists.

Every claim here was wrong at least once: paths that moved, a test count three
revisions old, and section numbers pointing at notebook sections that had been
renumbered away. Prose does not fail loudly on its own, so it is checked.

Only paths and numbers are checked. Whether a section reference points at the
*right* section is still a human's job — this catches the ones that point at
nothing at all.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

EXTENSIONS = (
    ".py", ".ipynb", ".json", ".csv", ".xlsx", ".R", ".html", ".docx",
    ".tif", ".png", ".parquet", ".ini", ".md", ".txt", ".toml",
)
#: A span carrying any of these describes a shape of filename, not a file.
PLACEHOLDER = re.compile(r"[*<>{}…]")
#: A command line mentions a path but is not one.
COMMAND = re.compile(r"^(python|pip|streamlit|jupyter|Rscript|cd|git|nbstripout|ATYPIER_)")

#: The README writes paths the way a reader needs them — ``modelling_phase/eda.py``
#: and a bare ``plot_style.py`` both mean a file under ``heavy_machinery/``.
SEARCH_ROOTS = [
    ROOT,
    ROOT / "heavy_machinery",
    ROOT / "heavy_machinery" / "cleaning_phase",
    ROOT / "heavy_machinery" / "modelling_phase",
    ROOT / "heavy_machinery" / "cutpoint_phase",
    ROOT / "heavy_machinery" / "config",
    ROOT / "heavy_machinery" / "scripts",
    ROOT / "heavy_machinery" / "pytests_atypier",
]

NOTEBOOKS = {
    "manuscript": ROOT / "meningioma-manuscript.ipynb",
    "cutpoint": ROOT / "meningioma-cutpoints.ipynb",
    "modelling": ROOT / "meningioma-modelling.ipynb",
    "cleaning": ROOT / "meningioma-cleaning.ipynb",
}


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pipeline_source() -> str:
    """Every line of pipeline source, so a filename can be proved to be written."""
    parts = []
    for path in (ROOT / "heavy_machinery").rglob("*"):
        if path.suffix in (".py", ".R") and "__pycache__" not in path.parts:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _path_spans(text: str) -> list[str]:
    spans = []
    for raw in re.findall(r"`([^`\n]+)`", text):
        span = raw.strip()
        if PLACEHOLDER.search(span) or COMMAND.match(span) or span.startswith("."):
            continue
        if span.endswith("/") or span.endswith(EXTENSIONS):
            spans.append(span)
    return sorted(set(spans))


def _resolves(span: str) -> bool:
    rel = span.rstrip("/")
    return any((root / rel).exists() for root in SEARCH_ROOTS)


def _is_written_by_pipeline(span: str, source: str) -> bool:
    """A file a run creates. Absent from a given checkout is not drift.

    Proved rather than assumed: the basename has to appear in the pipeline
    source. A filename nothing writes any more still fails.
    """
    if span.startswith("output/"):
        span = span[len("output/"):]
    basename = span.rstrip("/").split("/")[-1]
    return bool(basename) and basename in source


def _notebook_sections() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for name, path in NOTEBOOKS.items():
        numbers: set[str] = set()
        if path.exists():
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                head = "".join(cell["source"]).strip().split("\n")[0]
                if m := re.match(r"#+\s*(?:Step\s+)?(\d+(?:\.\d+)?)", head):
                    numbers.add(m.group(1).lstrip("0") or "0")
        found[name] = numbers
    return found


def test_every_path_in_the_readme_exists_or_is_written_by_a_run(readme, pipeline_source):
    dead = [
        span for span in _path_spans(readme)
        if not _resolves(span) and not _is_written_by_pipeline(span, pipeline_source)
    ]
    assert not dead, (
        "README names paths that exist nowhere and nothing writes:\n  "
        + "\n  ".join(dead)
    )


def test_every_notebook_section_the_readme_cites_exists(readme):
    """``§06`` was cited for a notebook whose sections stop at 05.

    Which notebook a reference belongs to is decided by the nearest name before
    it, so "modelling §06" is checked against the modelling notebook rather than
    against whichever notebook happens to own a §06.
    """
    sections = _notebook_sections()
    problems = []
    for m in re.finditer(r"§\s*(\d+(?:\.\d+)?)", readme):
        number = m.group(1).lstrip("0") or "0"
        before = readme[max(0, m.start() - 250):m.start()].lower()
        owner = max(
            ((name, before.rfind(name)) for name in NOTEBOOKS),
            key=lambda pair: pair[1],
        )
        if owner[1] == -1:
            if not any(number in s for s in sections.values()):
                problems.append(f"§{m.group(1)} — no notebook has it")
            continue
        name = owner[0]
        if number not in sections[name]:
            have = ", ".join(sorted(sections[name], key=float)) or "none"
            problems.append(f"§{m.group(1)} cited for {name}, which has: {have}")

    assert not problems, "README cites notebook sections that do not exist:\n  " + "\n  ".join(
        sorted(set(problems))
    )


def test_the_test_count_in_the_readme_is_current(readme, request):
    """Counted from this run's own collection, not from a second pytest.

    Skipped on a partial run, where the count is meaningless.
    """
    collected = len(request.session.items)
    if collected < 100:
        pytest.skip("run the whole suite for a meaningful count")

    claimed = [int(n) for n in re.findall(r"(\d{3,4})\s+(?:automated\s+|pytest\s+)*tests", readme)]
    assert claimed, f"README states no test count; this run collects {collected}"
    wrong = sorted({n for n in claimed if n != collected})
    assert not wrong, f"README claims {wrong} tests; this run collects {collected}"
