"""Export session data to open formats for downstream analysis.

readsync records and synchronises; it does not analyse. Analysis is done with
established open tools (Eyekit, popEye, PupEyes and the eye-tracking and EEG
packages). This module writes the decrypted, verified event log to a tidy CSV
that those tools, or any data-analysis environment, can read. Keeping the record
and the analysis separate avoids duplicating tested software.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .storage import EventLog

__all__ = ["events_to_csv", "log_to_csv"]


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

    Raises :class:`readsync.storage.IntegrityError` if the log fails its chain
    check, so a tampered record halts the export.
    """
    return events_to_csv(log.events(), path)
