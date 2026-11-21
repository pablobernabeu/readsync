"""Export session data for analysis.

readsync records and synchronises, and leaves analysis to established tools.
This module writes the decrypted, verified event log to a CSV with one row per
event, and the interest areas of a set of passages to a CSV with one row per
word. The second file carries the geometry that the session used for its word
and region events, so fixations detected later, from the tracker's own data
file, can be assigned to the same boxes.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .stimuli import StimulusItem
from .storage import EventLog
from .text import FixedWidthLayout, Passage, Region, interest_areas

__all__ = ["events_to_csv", "interest_area_rows", "interest_areas_to_csv", "log_to_csv"]

INTEREST_AREA_COLUMNS = (
    "passage", "word", "text", "line", "x1", "y1", "x2", "y2", "region", "layer", "role",
)


def _columns(events: Sequence[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for event in events:
        for key in event:
            seen.setdefault(key, None)
    # Put the common keys first for readability.
    preferred = [
        k
        for k in ("type", "t", "tracker_t", "passage", "word", "x", "y", "valid")
        if k in seen
    ]
    rest = [k for k in seen if k not in preferred]
    return preferred + rest


def events_to_csv(events: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of event dicts to a tidy CSV and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = _columns(events)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    return out


def log_to_csv(log: EventLog, path: str | Path) -> Path:
    """Decrypt and verify ``log``, then write it to CSV.

    Raises :class:`readsync.security.DecryptionError` or
    :class:`readsync.storage.IntegrityError` if the log fails verification, so a
    tampered log halts the export.
    """
    return events_to_csv(log.events(), path)


def interest_area_rows(
    items: Sequence[StimulusItem | Passage], layout: FixedWidthLayout | None = None
) -> list[dict[str, Any]]:
    """One row per word: its passage, index, text, line and box, and its region.

    ``layout`` must be the one the session used, which defaults to
    :class:`readsync.text.FixedWidthLayout` in both places. Words outside every
    annotated region have empty region fields, and a word inside several
    overlapping regions lists them all, joined by ``|`` in annotation order.
    ``line`` counts from 0, as ``word`` does.
    """
    layout = layout or FixedWidthLayout()
    rows: list[dict[str, Any]] = []
    for item in items:
        passage = item.passage if isinstance(item, StimulusItem) else item
        regions: list[Region] = item.regions if isinstance(item, StimulusItem) else []
        for area in interest_areas(passage.words, layout):
            here = [r for r in regions if r.contains_word(area.word.index)]
            rows.append(
                {
                    "passage": passage.id,
                    "word": area.word.index,
                    "text": area.word.text,
                    "line": (area.y1 - layout.y0) // layout.line_height,
                    "x1": area.x1,
                    "y1": area.y1,
                    "x2": area.x2,
                    "y2": area.y2,
                    "region": "|".join(r.id for r in here),
                    "layer": "|".join(r.layer for r in here),
                    "role": "|".join(r.role for r in here),
                }
            )
    return rows


def interest_areas_to_csv(
    items: Sequence[StimulusItem | Passage],
    path: str | Path,
    layout: FixedWidthLayout | None = None,
) -> Path:
    """Write the word interest areas of ``items`` to a CSV and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INTEREST_AREA_COLUMNS))
        writer.writeheader()
        writer.writerows(interest_area_rows(items, layout))
    return out
