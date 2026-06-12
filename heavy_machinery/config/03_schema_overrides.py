"""Step 03 — merge notebook ``schema_overrides`` and write ``schema_summary.csv``."""
from __future__ import annotations

from pathlib import Path

from schema_infer import export_schema_summary, schema_summary


def apply_schema_overrides(schema, schema_overrides, output_root: Path) -> None:
    schema.update(schema_overrides)
    schema_summary(schema)
    export_schema_summary(schema, output_root)
