"""Notebook config modules — one Python file per pipeline section (01–08).

Usage from notebooks in ``heavy_machinery/``::

    from config import load
    load("07_analysis").resolve_eda(...)

01 cohort load · 02 renames · 03 schema overrides · 04 row filters ·
05 missingness policy · 06 derivations · 07 EDA / model variants · 08 report
"""
import importlib.util
import sys
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent


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
