"""One self-contained HTML page holding every table and figure for the manuscript.

Self-contained means exactly that: images are embedded as base64, styles are
inline, nothing is fetched. The file can be emailed, opened on a machine that
has never seen this repository, and archived alongside the submission without
losing half of itself.

The page is a **proof sheet, not a deliverable**. The journal receives the
``.docx`` tables and the TIFF figures; this is where the author reads everything
in one place before deciding what goes where. So it shows the same numbers, from
the same objects, with the same footnotes — never a prettier variant, because a
second formatting path is a second set of numbers waiting to disagree.

Tables that are agreed appear in full. Tables still being designed appear as a
labelled placeholder with their intended columns listed, so the shape of the
finished document is visible while it is still incomplete — an absent table
reads as an oversight, a placeholder reads as a decision not yet made.
"""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1.5rem 5rem; background: #ffffff; color: #111;
       font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
.sub { color: #555; margin: 0 0 2.5rem; font-size: .95rem; }
section { margin: 0 0 3rem; }
h2 { font-size: 1.05rem; margin: 0 0 .9rem; font-weight: 650; }
.scroll { overflow-x: auto; margin: 0 0 .8rem; }
table { border-collapse: collapse; width: 100%; font-size: .87rem; }
th, td { text-align: left; padding: .5rem .7rem; vertical-align: top;
         white-space: nowrap; }
thead th { border-bottom: 1.5px solid #111; font-weight: 650; }
tbody tr:nth-child(even) { background: #f6f6f6; }
tbody th { font-weight: 500; }
td.yes { font-weight: 650; }
td.no { color: #7f7f7f; }
.note { color: #444; font-size: .82rem; line-height: 1.55; margin: 0;
        padding-left: 0; max-width: 62rem; }
.lead { margin: 0 0 .9rem; color: #333; font-size: .92rem; }
.placeholder { border: 1px dashed #b0b0b0; border-radius: 6px; padding: 1.1rem 1.3rem;
               background: #fafafa; color: #444; }
.placeholder ul { margin: .5rem 0 0; padding-left: 1.2rem; font-size: .87rem; }
.placeholder .tag { display: inline-block; font-size: .72rem; letter-spacing: .06em;
                    text-transform: uppercase; color: #7f7f7f; margin-bottom: .5rem; }
.dash { margin: 0 0 3rem; }
.dash h3 { font-size: .78rem; letter-spacing: .07em; text-transform: uppercase;
           color: #666; margin: 0 0 .7rem; font-weight: 650; }
.dash + .dash { margin-top: -1.6rem; }
.cards { display: grid; gap: .8rem;
         grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); }
.card { border: 1px solid #d5d5d5; border-radius: 8px; padding: .95rem 1.1rem;
        background: #fff; }
.card.out { background: #fafafa; border-style: dashed; }
.card .name { font-weight: 650; font-size: .95rem; margin: 0 0 .35rem; }
.card .value { font-size: 1.25rem; font-variant-numeric: tabular-nums;
               margin: 0 0 .1rem; }
.card.out .value { color: #7f7f7f; font-size: 1.05rem; }
.card .ci { color: #555; font-size: .8rem; margin: 0 0 .55rem; }
.card .criteria { font-size: .82rem; font-weight: 650; margin: 0 0 .3rem; }
.card .criteria.met { color: #111; }
.card .criteria.unmet { color: #6b6b6b; }
.card .why { color: #333; font-size: .82rem; line-height: 1.45; margin: 0 0 .35rem; }
.card .works { color: #111; font-size: .84rem; line-height: 1.45; margin: 0;
               font-weight: 500; }
.card .spark { width: 100%; height: auto; display: block; margin: .7rem 0 .15rem; }
.card .sparklab { color: #999; font-size: .66rem; margin: 0; text-align: center;
                  letter-spacing: .02em; }
.card .detail { color: #777; font-size: .74rem; margin: .45rem 0 0;
                font-variant-numeric: tabular-nums; }
.card .cav-detail { color: #666; font-size: .74rem;
                    font-variant-numeric: tabular-nums; }
.card .caveat { margin: .55rem 0 0; padding: .5rem .65rem; border-radius: 5px;
                background: #f1f1f1; color: #333; font-size: .8rem;
                line-height: 1.45; }
figure { margin: 0; }
figure img { width: 100%; height: auto; display: block; }
footer { border-top: 1px solid #ddd; margin-top: 3rem; padding-top: 1rem;
         color: #666; font-size: .8rem; }
@media (prefers-color-scheme: dark) {
  body { background: #ffffff; color: #111; }
}
"""


def _escape(value) -> str:
    return html.escape("" if value is None else str(value))


def render_table(frame: pd.DataFrame, *, index_header: str = "") -> str:
    """A DataFrame of already-formatted strings as an HTML table.

    Cells reading exactly Yes or No are given a class so a graded column can be
    weighted without the renderer having to know which column is graded.
    """
    head = "".join(f"<th>{_escape(c)}</th>" for c in frame.columns)
    rows = []
    for label, row in frame.iterrows():
        cells = []
        for value in row:
            text = _escape(value)
            css = {"Yes": ' class="yes"', "No": ' class="no"'}.get(str(value), "")
            cells.append(f"<td{css}>{text}</td>")
        rows.append(f"<tr><th>{_escape(label)}</th>{''.join(cells)}</tr>")
    return (f'<div class="scroll"><table><thead><tr>'
            f'<th>{_escape(index_header)}</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def table_section(title: str, frame: pd.DataFrame, *, note: str = "",
                  lead: str = "", index_header: str = "") -> str:
    """One finished table: its name, an optional lead line, the grid, its note."""
    parts = [f"<h2>{_escape(title)}</h2>"]
    if lead:
        parts.append(f'<p class="lead">{_escape(lead)}</p>')
    parts.append(render_table(frame, index_header=index_header))
    if note:
        parts.append(f'<p class="note">{_escape(note)}</p>')
    return f"<section>{''.join(parts)}</section>"


def placeholder_section(title: str, *, columns: Sequence[str],
                        reason: str = "") -> str:
    """A table not yet built, shown as its intended shape.

    An absent table reads as an oversight; a placeholder listing its columns
    reads as a decision still open, which is what it is.
    """
    items = "".join(f"<li>{_escape(c)}</li>" for c in columns)
    body = [f'<span class="tag">Not yet built</span>',
            f"<strong>{_escape(title)}</strong>"]
    if reason:
        body.append(f"<p class=\"lead\" style=\"margin:.6rem 0 0\">{_escape(reason)}</p>")
    body.append(f"<ul>{items}</ul>")
    return (f"<section><h2>{_escape(title)}</h2>"
            f'<div class="placeholder">{"".join(body)}</div></section>')


def _card(entry: dict) -> str:
    """One measurement's verdict: the number, what it met, how it works, a picture.

    A fixed order, because a reader scanning five cards needs the same thing in
    the same place every time: the value, how many conditions it met, which ones
    it failed, whether it works as a number or only as a rule, and a picture of
    the curve the verdict is about.
    """
    supported = bool(entry.get("supported"))
    value = entry.get("breakpoint") if supported else entry.get("cutpoint")
    parts = [f'<p class="name">{_escape(entry.get("measurement"))}</p>',
             f'<p class="value">{_escape(value)}</p>']
    if supported and entry.get("breakpoint_ci"):
        parts.append(f'<p class="ci">95% CI {_escape(entry["breakpoint_ci"])}</p>')

    if entry.get("criteria_line"):
        css = "met" if supported else "unmet"
        parts.append(f'<p class="criteria {css}">'
                     f'{_escape(entry["criteria_line"])}</p>')
    if entry.get("reason"):
        parts.append(f'<p class="why">{_escape(entry["reason"])}</p>')
    if entry.get("works"):
        parts.append(f'<p class="works">{_escape(entry["works"])}</p>')
    if entry.get("thumbnail"):
        parts.append(f'<img class="spark" alt="risk curve in clinical units and '
                     f'on a log scale" '
                     f'src="data:image/png;base64,{entry["thumbnail"]}">')
        parts.append('<p class="sparklab">risk vs value &nbsp;·&nbsp; '
                     'the same fit on a log scale</p>')
    if entry.get("detail"):
        parts.append(f'<p class="detail">{_escape(entry["detail"])}</p>')
    css = "card" if supported else "card out"
    return f'<div class="{css}">{"".join(parts)}</div>'


def dashboard_section(entries: Sequence[dict], *,
                      supported_heading: str = "Threshold supported",
                      unsupported_heading: str = "Threshold not supported",
                      lead: str = "") -> str:
    """The verdict summary that opens the page.

    Two groups, each measurement carrying the reason for its verdict rather than
    only the verdict. "Not supported" covers four different failures, and a
    reader shown only the word cannot tell which one they are looking at.

    A supported measurement shows its **breakpoint**; an unsupported one shows
    its **cut-point**, because there is no defensible breakpoint to show and
    printing one anyway would be the overclaim this page exists to prevent.
    """
    entries = list(entries)
    supported = [e for e in entries if e.get("supported")]
    rest = [e for e in entries if not e.get("supported")]
    blocks = []
    if lead:
        blocks.append(f'<p class="lead">{_escape(lead)}</p>')
    for heading, group in ((supported_heading, supported),
                           (unsupported_heading, rest)):
        if not group:
            continue
        cards = "".join(_card(e) for e in group)
        blocks.append(f'<div class="dash"><h3>{_escape(heading)} '
                      f'({len(group)})</h3><div class="cards">{cards}</div></div>')
    return f"<section>{''.join(blocks)}</section>"


def figure_section(title: str, image: Path | str, *, caption: str = "") -> str:
    """A figure embedded as base64, so the page stays one file."""
    path = Path(image)
    if not path.exists():
        return placeholder_section(title, columns=[f"expected at {path}"],
                                   reason="Figure not found on disk.")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lstrip(".").lower()
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    parts = [f"<h2>{_escape(title)}</h2>",
             f'<figure><img alt="{_escape(title)}" '
             f'src="data:{mime};base64,{data}"></figure>']
    if caption:
        parts.append(f'<p class="note">{_escape(caption)}</p>')
    return f"<section>{''.join(parts)}</section>"


def build(sections: Iterable[str], *, title: str, subtitle: str = "",
          generated_at: str | None = None) -> str:
    """Assemble the page. ``generated_at`` is injectable so a test can pin it."""
    stamp = generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<h1>{_escape(title)}</h1>"
        + (f'<p class="sub">{_escape(subtitle)}</p>' if subtitle else "")
        + "".join(sections)
        + f"<footer>Generated {_escape(stamp)}. Every number on this page comes "
          "from the same objects that produce the Word tables and the TIFF "
          "figures; this page reformats nothing.</footer>"
          "</main></body></html>")


def write(path: Path | str, sections: Iterable[str], *, title: str,
          subtitle: str = "", generated_at: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(sections, title=title, subtitle=subtitle,
                          generated_at=generated_at), encoding="utf-8")
    return path
