"""Export target for the threshold phase — ``output/thresholds/``.

Its own folder, separate from ``output/eda`` and ``output/report``, for a
reason that is not tidiness: nothing in the threshold phase may be consumed by
the cleaning or modelling pipeline. A cut-point estimated here has already seen
the outcome column, so feeding it back into the handoff would leak the answer
into the predictors. Keeping the exports in a folder no other phase reads makes
that separation structural rather than a rule someone has to remember.

Layout::

    output/thresholds/
        figures/*.svg     one per figure, SciencePlots-styled like the rest
        tables/*.csv      one per table, numbers rounded for reading
        manifest.json     what this run wrote, with the settings that produced it

``ThresholdArtifacts`` always displays inline; writing is what ``enabled``
toggles. That way the notebook reads the same whether or not it is exporting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from cleaning import format_table_for_csv, format_table_for_display
import plot_style as ps

FIGURES_DIRNAME = "figures"
TABLES_DIRNAME = "tables"
MANIFEST_NAME = "manifest.json"


@dataclass
class ThresholdArtifacts:
    """Display-and-optionally-write sink for the threshold notebook.

    Parameters
    ----------
    root      : the phase's own output folder, ``output/thresholds`` by default.
    enabled   : write to disk as well as display. Off keeps a run read-only.
    context   : free-form settings recorded in the manifest (cohort size,
                bootstrap count, seed) so an exported figure can be traced back
                to the run that produced it.
    """

    root: Path = Path("output") / "thresholds"
    enabled: bool = True
    context: dict[str, Any] = field(default_factory=dict)
    _entries: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def figure_dir(self) -> Path:
        return self.root / FIGURES_DIRNAME

    @property
    def table_dir(self) -> Path:
        return self.root / TABLES_DIRNAME

    # ---- writing ---------------------------------------------------------
    def figure(
        self,
        fig: plt.Figure,
        name: str,
        *,
        caption: str = "",
        tight_layout: bool = True,
    ) -> plt.Figure:
        """Render inline, then write the SVG when enabled."""
        display(fig)
        path = self.figure_dir / _with_suffix(name, ".svg")
        if self.enabled:
            self.figure_dir.mkdir(parents=True, exist_ok=True)
            ps.save_figure(fig, path, tight_layout=tight_layout)  # closes the figure
            self._record("figure", path, caption=caption)
        else:
            plt.close(fig)
        return fig

    def table(
        self,
        table: pd.DataFrame,
        name: str,
        *,
        caption: str = "",
        formatted: bool = True,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Display inline, then write the CSV when enabled.

        ``formatted`` controls the *display* copy only. The CSV always goes
        through the pipeline's shared rounding rules, and the returned frame is
        the untouched original so callers can keep computing with it.
        """
        shown = table if max_rows is None else table.head(int(max_rows))
        display(format_table_for_display(shown) if formatted else shown.fillna(""))
        if max_rows is not None and len(table) > max_rows:
            print(f"… {len(table) - max_rows} more rows (all of them in the CSV)")

        path = self.table_dir / _with_suffix(name, ".csv")
        if self.enabled:
            self.table_dir.mkdir(parents=True, exist_ok=True)
            format_table_for_csv(table).to_csv(path, index=False)
            self._record("table", path, caption=caption, rows=len(table),
                         columns=list(map(str, table.columns)))
            print(f"💾 {path} — {len(table)} rows")
        return table

    def note(self, text: str) -> None:
        """Record a finding in the manifest without producing a file."""
        self._entries.append({"kind": "note", "text": text})

    def _record(self, kind: str, path: Path, **extra: Any) -> None:
        entry = {
            "kind": kind,
            "path": str(path),
            "name": path.name,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        entry.update({k: v for k, v in extra.items() if v not in ("", None)})
        self._entries.append(entry)

    # ---- manifest --------------------------------------------------------
    def write_manifest(self) -> Path | None:
        """Everything this run produced, plus the settings that produced it."""
        if not self.enabled:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / MANIFEST_NAME
        payload = {
            "phase": "thresholds",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "context": _jsonable(self.context),
            "figures": [e for e in self._entries if e["kind"] == "figure"],
            "tables": [e for e in self._entries if e["kind"] == "table"],
            "notes": [e["text"] for e in self._entries if e["kind"] == "note"],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    def written(self) -> pd.DataFrame:
        """What this run wrote — the end-of-notebook receipt."""
        files = [e for e in self._entries if e["kind"] in ("figure", "table")]
        if not files:
            return pd.DataFrame(columns=["kind", "name", "path", "caption"])
        out = pd.DataFrame(files)
        keep = [c for c in ("kind", "name", "rows", "caption", "path") if c in out.columns]
        return out[keep]

    def stale_files(self) -> list[Path]:
        """Files sitting in the output folder that this run did **not** write.

        Left in place rather than deleted — a leftover from a previous
        configuration is worth seeing before it is thrown away, and this module
        has no business removing files it did not create.
        """
        written = {Path(e["path"]).resolve()
                   for e in self._entries if e["kind"] in ("figure", "table")}
        found: list[Path] = []
        for folder, pattern in ((self.figure_dir, "*.svg"), (self.table_dir, "*.csv")):
            if folder.exists():
                found += [p for p in sorted(folder.glob(pattern))
                          if p.resolve() not in written]
        return found


def _with_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(suffix) else f"{name}{suffix}"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion so numpy/pandas scalars survive json.dumps."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
