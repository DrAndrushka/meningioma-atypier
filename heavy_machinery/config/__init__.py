"""Numbered pipeline steps (01–08). Import with config.load('01_cohort')."""
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
