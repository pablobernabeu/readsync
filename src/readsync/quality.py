"""Per-session data-quality report.

Recording quality should be a reported outcome, above all when sessions run
outside a fixed laboratory: every session should leave a record of how much
gaze was captured, how much was lost, and what the calibration and drift
checks said. This module computes that report from the verified event log. It
summarises the polled link stream and the logged checks; the tracker's own
data file remains the full-rate record and is quality-checked in analysis with
the established tools.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .storage import EventLog

__all__ = [
    "PassageQuality",
    "QualityReport",
    "log_quality",
    "quality_to_json",
    "session_quality",
]


@dataclass(frozen=True)
class PassageQuality:
    """Quality figures for one passage's recording.

    ``complete`` is false for a passage whose offset never arrived, which
    happens when a session crashed or was aborted mid-passage; its figures then
    cover the samples recorded up to that point.
    """

    passage: str
    duration: float
    n_samples: int
    n_invalid: int
    invalid_proportion: float
    observed_rate_hz: float
    largest_gap: float
    complete: bool = True


@dataclass(frozen=True)
class QualityReport:
    """Quality figures for a whole session.

    ``observed_rate_hz`` is the polled sample rate over the recorded passages.
    ``sample_loss`` compares it with ``expected_rate_hz`` where one was given.
    ``calibration`` and ``drift_errors`` collect the logged tracker checks, and
    ``n_responses``/``n_correct`` summarise the comprehension record.
    """

    participant: str | None
    n_passages: int
    n_samples: int
    invalid_proportion: float
    observed_rate_hz: float
    expected_rate_hz: float | None
    sample_loss: float | None
    largest_gap: float
    calibration: list[str]
    drift_errors: list[float | None]
    n_responses: int
    n_correct: int
    passages: list[PassageQuality]


def _passage_quality(
    passage: str, gaze: list[dict[str, Any]], *, complete: bool = True
) -> PassageQuality:
    times = [float(e["t"]) for e in gaze]
    duration = max(times) - min(times) if len(times) > 1 else 0.0
    invalid = sum(1 for e in gaze if not e.get("valid", True))
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    return PassageQuality(
        passage=passage,
        duration=round(duration, 3),
        n_samples=len(gaze),
        n_invalid=invalid,
        invalid_proportion=round(invalid / len(gaze), 4) if gaze else 0.0,
        observed_rate_hz=round((len(gaze) - 1) / duration, 1) if duration > 0 else 0.0,
        largest_gap=round(max(gaps), 4) if gaps else 0.0,
        complete=complete,
    )


def session_quality(
    events: Sequence[dict[str, Any]], *, expected_rate_hz: float | None = None
) -> QualityReport:
    """Compute a :class:`QualityReport` from a session's events.

    ``events`` is the verified list from :meth:`readsync.storage.EventLog.events`
    or :func:`log_quality`. Gaze samples are grouped by the passage being read,
    using the passage onset and offset events, so between-passage intervals do
    not count as loss. A passage whose offset never arrived, from a crashed or
    aborted session, is still reported, marked incomplete. ``expected_rate_hz``
    must be a positive rate when given.
    """
    if expected_rate_hz is not None and expected_rate_hz <= 0:
        raise ValueError("expected_rate_hz must be positive when given")
    participant: str | None = None
    calibration: list[str] = []
    drift_errors: list[float | None] = []
    n_responses = 0
    n_correct = 0
    passages: list[PassageQuality] = []
    current: str | None = None
    gaze: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "session_start":
            participant = str(event.get("participant"))
        elif kind == "calibration":
            calibration.append(str(event.get("detail")))
        elif kind == "drift_check":
            error = event.get("error")
            drift_errors.append(float(error) if error is not None else None)
        elif kind == "passage_onset":
            current = str(event.get("passage"))
            gaze = []
        elif kind == "gaze" and current is not None:
            gaze.append(event)
        elif kind == "passage_offset" and current is not None:
            passages.append(_passage_quality(current, gaze))
            current, gaze = None, []
        elif kind == "response":
            n_responses += 1
            n_correct += bool(event.get("correct"))
    if current is not None:
        passages.append(_passage_quality(current, gaze, complete=False))
    n_samples = sum(p.n_samples for p in passages)
    invalid = sum(p.n_invalid for p in passages)
    # The polled rate is intervals over time, counted only within passages
    # that recorded more than one sample, so a single-sample passage cannot
    # zero the session rate.
    timed = [p for p in passages if p.duration > 0]
    total_duration = sum(p.duration for p in timed)
    intervals = sum(p.n_samples - 1 for p in timed)
    observed = intervals / total_duration if total_duration > 0 else 0.0
    loss = None
    if expected_rate_hz is not None:
        loss = max(0.0, 1.0 - observed / expected_rate_hz)
    return QualityReport(
        participant=participant,
        n_passages=len(passages),
        n_samples=n_samples,
        invalid_proportion=round(invalid / n_samples, 4) if n_samples else 0.0,
        observed_rate_hz=round(observed, 1),
        expected_rate_hz=expected_rate_hz,
        sample_loss=round(loss, 4) if loss is not None else None,
        largest_gap=max((p.largest_gap for p in passages), default=0.0),
        calibration=calibration,
        drift_errors=drift_errors,
        n_responses=n_responses,
        n_correct=n_correct,
        passages=passages,
    )


def quality_to_json(report: QualityReport, path: str | Path) -> Path:
    """Write ``report`` as JSON beside the session's exports and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out


def log_quality(
    log: EventLog, *, expected_rate_hz: float | None = None
) -> QualityReport:
    """Decrypt and verify ``log``, then compute its quality report.

    Raises :class:`readsync.storage.IntegrityError` if the log fails its chain
    check, as :func:`readsync.export.log_to_csv` does.
    """
    return session_quality(log.events(), expected_rate_hz=expected_rate_hz)
