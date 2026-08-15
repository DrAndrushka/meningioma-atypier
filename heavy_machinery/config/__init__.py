"""Notebook config modules — one Python file per pipeline concern.

Usage from notebooks at the repo root (``meningioma-atypier/``)::

    from heavy_machinery.config import load
    load("analysis").resolve_eda(...)

cohort · column_rename_map · schema_overrides · row_filters ·
missingness · derivations · analysis · report_settings

Importing this package prepends ``cleaning_phase/``, ``modelling_phase/`` and
``cutpoint_phase/`` to ``sys.path`` so library modules can use flat sibling
imports (``from schema_infer import ColSpec``) and tests can reuse the same
layout.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_HEAVY_MACHINERY = _CONFIG_DIR.parent

for _phase in ("cleaning_phase", "modelling_phase", "cutpoint_phase"):
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
