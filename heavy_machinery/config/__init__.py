"""Notebook config modules — one Python file per pipeline section (01–08).

Usage from notebooks at the repo root (``meningioma-atypier/``)::

    from heavy_machinery.config import load
    load("07_analysis").resolve_eda(...)

01 cohort load · 02 renames · 03 schema overrides · 04 row filters ·
05 missingness policy · 06 derivations · 07 EDA / model variants · 08 report

Importing this package also prepends ``cleaning_phase/`` and ``modelling_phase/``
to ``sys.path`` so notebooks can use ``from schema_infer import …`` etc.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_HEAVY_MACHINERY = _CONFIG_DIR.parent

for _phase in ("cleaning_phase", "modelling_phase"):
    _path = str(_HEAVY_MACHINERY / _phase)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def load(stem: str):
    path = _CONFIG_DIR / f"{stem}.py"
    if not path.exists():
        raise ImportError(f"No config module: {path}")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod
