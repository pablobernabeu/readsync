"""Per-session data-quality report.

Every session should leave a record of how much gaze was captured, how much
was lost and what the calibration and drift checks reported, which are among
the items that the minimal reporting guideline for eye-tracking research asks
studies to report (Dunn et al., 2024, https://doi.org/10.3758/s13428-023-02187-1).
This module computes that report from the verified event log. It summarises the
gaze polled during each passage and the logged checks. The tracker's own data
file remains the full-rate record.
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

    ``duration`` runs from the passage onset to its offset, so samples missing
    at the start or end of a passage count as lost. ``observed_rate_hz`` is the
    number of sample intervals over that duration, and ``sample_loss`` is the
    share of the samples expected at ``expected_rate_hz`` that did not arrive.
    ``largest_gap`` includes the stretches from the onset to the first sample
    and from the last sample to the offset. ``n_stale`` counts samples that
    repeat the previous sample's tracker time, which happens when the tracker
    stops delivering new samples. ``complete`` is false for a passage whose
    offset never arrived, after a crash or an abort, and its figures then run
    to the last sample recorded.
    """

    passage: str
    duration: float
    n_samples: int
    n_invalid: int
    invalid_proportion: float
    observed_rate_hz: float
    sample_loss: float | None
    largest_gap: float
    n_stale: int
    complete: bool = True


@dataclass(frozen=True)
class QualityReport:
    """Quality figures for a whole session.

    ``status`` is the closing ``session_end`` status, ``completed`` or
    ``aborted``, or ``None`` when the log has no closing event, as after a crash
    that left the log cut short. ``n_planned`` is the number of passages the
    session set out to present. ``observed_rate_hz`` is the polled sample rate
    over the recorded passages, and ``sample_loss`` compares it with
    ``expected_rate_hz`` where one was given. Because the session polls once per
    display frame, this measures the polled stream, while the tracker's data
    file holds the full-rate record. ``calibration`` and ``drift_errors``
    collect the logged tracker checks, and ``n_responses``/``n_correct``
    summarise the comprehension record.
    """

    participant: str | None
    session: str | None
    status: str | None
    n_planned: int | None
    n_passages: int
    n_samples: int
    invalid_proportion: float
    observed_rate_hz: float
    expected_rate_hz: float | None
    sample_loss: float | None
    largest_gap: float
    n_stale: int
    calibration: list[str]
    drift_errors: list[float | None]
    n_responses: int
    n_correct: int
    passages: list[PassageQuality]


def _expected_samples(duration: float, rate_hz: float) -> float:
    """Samples expected over ``duration`` when the first falls at the onset."""
    return duration * rate_hz + 1.0


def _passage_quality(
    passage: str,
    onset: float,
    gaze: list[dict[str, Any]],
    *,
    offset: float | None,
    expected_rate_hz: float | None,
) -> PassageQuality:
    times = [float(e["t"]) for e in gaze]
    end = offset if offset is not None else (times[-1] if times else onset)
    duration = max(0.0, end - onset)
    bounds = [onset, *times, end]
    gaps = [b - a for a, b in zip(bounds, bounds[1:], strict=False)]
    invalid = sum(1 for e in gaze if not e.get("valid", True))
    tracker_times = [e.get("tracker_t") for e in gaze]
    stale = sum(
        1
        for a, b in zip(tracker_times, tracker_times[1:], strict=False)
        if a is not None and a == b
    )
    loss = None
    if expected_rate_hz is not None:
        loss = max(0.0, 1.0 - len(gaze) / _expected_samples(duration, expected_rate_hz))
    return PassageQuality(
        passage=passage,
        duration=round(duration, 3),
        n_samples=len(gaze),
        n_invalid=invalid,
        invalid_proportion=round(invalid / len(gaze), 4) if gaze else 0.0,
        observed_rate_hz=round((len(gaze) - 1) / duration, 1)
        if duration > 0 and len(gaze) > 1
        else 0.0,
        sample_loss=round(loss, 4) if loss is not None else None,
        largest_gap=round(max(gaps), 4) if gaps else 0.0,
        n_stale=stale,
        complete=offset is not None,
    )


def session_quality(
    events: Sequence[dict[str, Any]], *, expected_rate_hz: float | None = None
) -> QualityReport:
    """Compute a :class:`QualityReport` from a session's events.

    ``events`` is the verified list from :meth:`readsync.storage.EventLog.events`
    or :func:`log_quality`. Gaze samples are grouped by the passage being read,
    using the passage onset and offset events, so between-passage intervals do
    not count as loss, while missing samples anywhere within a passage do. A
    passage whose offset never arrived, from a crashed or aborted session, is
    still reported, marked incomplete. ``expected_rate_hz`` must be a positive
    rate when given. The session polls the tracker once per display frame, so
    pass the display's refresh rate, not the tracker's sampling rate.
    """
    if expected_rate_hz is not None and expected_rate_hz <= 0:
        raise ValueError("expected_rate_hz must be positive when given")
    participant: str | None = None
    session: str | None = None
    status: str | None = None
    n_planned: int | None = None
    calibration: list[str] = []
    drift_errors: list[float | None] = []
    n_responses = 0
    n_correct = 0
    passages: list[PassageQuality] = []
    current: str | None = None
    onset = 0.0
    gaze: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("type")
        if kind == "session_start":
            participant = str(event.get("participant"))
            session = event.get("session")
            planned = event.get("planned")
            n_planned = len(planned) if isinstance(planned, list) else None
        elif kind == "session_end":
            status = event.get("status")
        elif kind == "calibration":
            calibration.append(str(event.get("detail")))
        elif kind == "drift_check":
            error = event.get("error")
            drift_errors.append(float(error) if error is not None else None)
        elif kind == "passage_onset":
            current = str(event.get("passage"))
            onset = float(event.get("t", 0.0))
            gaze = []
        elif kind == "gaze" and current is not None:
            gaze.append(event)
        elif kind == "passage_offset" and current is not None:
            passages.append(
                _passage_quality(
                    current,
                    onset,
                    gaze,
                    offset=float(event.get("t", onset)),
                    expected_rate_hz=expected_rate_hz,
                )
            )
            current, gaze = None, []
        elif kind == "response":
            n_responses += 1
            n_correct += bool(event.get("correct"))
    if current is not None:
        passages.append(
            _passage_quality(current, onset, gaze, offset=None, expected_rate_hz=expected_rate_hz)
        )
    n_samples = sum(p.n_samples for p in passages)
    invalid = sum(p.n_invalid for p in passages)
    # Every passage counts towards the session figures, including one that
    # recorded no samples, so a passage with no data shows up as loss.
    total_duration = sum(p.duration for p in passages)
    intervals = sum(max(p.n_samples - 1, 0) for p in passages)
    observed = intervals / total_duration if total_duration > 0 else 0.0
    loss = None
    if expected_rate_hz is not None and passages:
        expected = sum(_expected_samples(p.duration, expected_rate_hz) for p in passages)
        loss = max(0.0, 1.0 - n_samples / expected)
    return QualityReport(
        participant=participant,
        session=session,
        status=status,
        n_planned=n_planned,
        n_passages=len(passages),
        n_samples=n_samples,
        invalid_proportion=round(invalid / n_samples, 4) if n_samples else 0.0,
        observed_rate_hz=round(observed, 1),
        expected_rate_hz=expected_rate_hz,
        sample_loss=round(loss, 4) if loss is not None else None,
        largest_gap=max((p.largest_gap for p in passages), default=0.0),
        n_stale=sum(p.n_stale for p in passages),
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

    Raises the same errors as :func:`readsync.export.log_to_csv` if the log
    fails verification.
    """
    return session_quality(log.events(), expected_rate_hz=expected_rate_hz)
