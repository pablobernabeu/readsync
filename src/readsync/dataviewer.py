"""EyeLink Data Viewer integration: trial messages and interest-area sets.

SR Research's Data Viewer reconstructs the structure of a recording from
messages in the EDF file. A trial runs from a ``TRIALID`` message to a
``TRIAL_RESULT`` message, ``!V TRIAL_VAR`` messages attach variables to it, and
``!V IAREA FILE`` names an interest-area set to load for it. This module builds
those messages, and the interest-area set itself, from the word interest areas
that the session uses for its word and region events. A recording opened in
Data Viewer, or exported from it, then carries the same word boxes as the
session log.

The set is written in the tab-separated form that Data Viewer saves: one
``RECTANGLE`` line per area with an integer id, the left, top, right and bottom
coordinates in screen pixels and a label. Ids are the word index plus one,
because Data Viewer numbers interest areas from 1, and the label is the word as
displayed. The boxes are exactly the session's, so the space between two words
belongs to neither, as it does in the session's own word events.

The functions are pure apart from :func:`write_ias`, so the message protocol is
tested without a tracker. :class:`readsync.trackers.EyeLinkTracker` sends the
messages through the link.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .markers import EYELINK_MESSAGE_LIMIT
from .text import InterestArea

__all__ = [
    "ias_text",
    "trial_end_messages",
    "trial_start_messages",
    "write_ias",
]

_UNSAFE_VALUE = re.compile(r"\s+")


def _value(value: object) -> str:
    """One whitespace-free token, so a message splits into fields unambiguously."""
    return _UNSAFE_VALUE.sub("_", str(value))


def ias_text(areas: Sequence[InterestArea]) -> str:
    """Return the interest-area set for ``areas`` as Data Viewer reads it."""
    lines = ["# readsync word interest areas: id = word index + 1, label = word"]
    for area in areas:
        fields = [
            "RECTANGLE",
            str(area.word.index + 1),
            str(area.x1),
            str(area.y1),
            str(area.x2),
            str(area.y2),
            area.word.text,
        ]
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


def write_ias(areas: Sequence[InterestArea], path: str | Path) -> Path:
    """Write the interest-area set for ``areas`` to ``path`` and return it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ias_text(areas), encoding="utf-8")
    return out


def trial_start_messages(trial_id: str, ias_file: str | None = None) -> list[str]:
    """Messages that open a trial and, where given, attach its interest areas.

    ``ias_file`` is the set's file name as Data Viewer should resolve it. SR
    Research's demonstration scripts write the set beside the EDF file and give
    the bare file name.
    """
    messages = [f"TRIALID {_value(trial_id)}"]
    if ias_file is not None:
        messages.append(f"!V IAREA FILE {_value(ias_file)}")
    return [message[:EYELINK_MESSAGE_LIMIT] for message in messages]


def trial_end_messages(variables: Mapping[str, object] | None = None) -> list[str]:
    """Messages that attach trial variables and close the trial.

    Variables whose value is ``None`` are left out. ``TRIAL_RESULT 0`` closes
    the trial without a response code, since responses are kept in the log.
    """
    messages = [
        f"!V TRIAL_VAR {_value(name)} {_value(value)}"
        for name, value in (variables or {}).items()
        if value is not None
    ]
    messages.append("TRIAL_RESULT 0")
    return [message[:EYELINK_MESSAGE_LIMIT] for message in messages]
