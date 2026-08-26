"""Orchestration of an offline reading session.

``ReadingSession`` ties the pieces together: it presents passages, polls the
tracker for gaze, sends markers at word, region and passage boundaries, asks the
comprehension questions and writes everything to the encrypted event log. The
whole run happens inside a ``NetworkGuard`` so that no data leave the machine
while a participant is being recorded.

Presentation is real-time and behind a ``Presenter`` protocol. Each frame, the
session calls ``presenter.tick``, which draws the current passage, returns the
session clock time and reports whether the reader has asked to move on.
``PsychoPyPresenter`` (see ``readsync.presenters``) runs sessions on a real
screen; ``HeadlessPresenter`` advances a virtual clock with no display, which is
what lets the session be tested without PsychoPy or a screen.

Every event in the log is stamped on the presenter's session clock, so within
one log ``t`` has a single time base. Gaze events additionally carry the
tracker's own timestamp as ``tracker_t`` for cross-checking against the
tracker's data file, which keeps its own clock through the marker messages.

Questions and drift checks are optional capabilities, discovered on the
collaborators at run time: a presenter that implements ``ask`` (and, for the
information-seeking regime, ``show_prompt``) collects responses, and a tracker
that implements ``drift_correct`` is checked before each passage. Backends
without these methods run exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .markers import Marker, MarkerSink, NullMarkerSink
from .security import NetworkGuard
from .stimuli import Question, StimulusItem
from .storage import EventLog
from .text import (
    FixedWidthLayout,
    InterestArea,
    Passage,
    Region,
    interest_areas,
    locate,
    region_at,
)
from .trackers import Tracker

__all__ = [
    "Presenter",
    "HeadlessPresenter",
    "QuestionResponse",
    "ReadingSession",
    "SessionResult",
]


@runtime_checkable
class Presenter(Protocol):
    """Shows a passage frame by frame until the reader advances.

    ``start_passage`` prepares a passage and its interest areas for display.
    ``tick`` renders one frame and returns ``(t, finished)``, where ``t`` is the
    time in seconds from the start of the session and ``finished`` is true once
    the reader has signalled that they have finished the passage. ``end_passage``
    clears the display. ``close`` releases any resources.

    Two further methods are optional and looked up at run time, outside this
    protocol. ``ask(question)`` presents a yes/no comprehension
    question and returns a :class:`QuestionResponse`. ``show_prompt(text)``
    displays a question before its passage, for the information-seeking regime,
    and returns the time at which the reader moved on.
    """

    def start_passage(self, passage: Passage, areas: list[InterestArea]) -> None: ...
    def tick(self) -> tuple[float, bool]: ...
    def end_passage(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class QuestionResponse:
    """A reader's answer to one question. ``onset`` is when the question
    appeared and ``response_time`` when the answer was given, both in seconds
    from session start."""

    onset: float
    response_time: float
    response: bool


class HeadlessPresenter:
    """A no-display presenter driven by a virtual clock.

    Each passage is shown for ``seconds_per_passage`` of virtual time, advanced in
    steps of ``1 / rate_hz``. The clock runs continuously across passages, so
    event times are monotonic. Deterministic, so tests are stable. Questions take
    ``seconds_per_question`` each and are answered from ``scripted_responses`` by
    question id, defaulting to yes, so response handling is testable end to end.
    """

    def __init__(
        self,
        *,
        seconds_per_passage: float = 2.0,
        rate_hz: int = 60,
        seconds_per_question: float = 1.0,
        scripted_responses: dict[str, bool] | None = None,
    ) -> None:
        self.seconds_per_passage = seconds_per_passage
        self.rate_hz = rate_hz
        self.seconds_per_question = seconds_per_question
        self.scripted_responses = dict(scripted_responses or {})
        self._clock = 0.0
        self._frames = 0
        self._i = 0

    def start_passage(self, passage: Passage, areas: list[InterestArea]) -> None:
        self._frames = max(1, int(self.seconds_per_passage * self.rate_hz))
        self._i = 0

    def tick(self) -> tuple[float, bool]:
        self._clock += 1.0 / self.rate_hz
        self._i += 1
        return self._clock, self._i >= self._frames

    def end_passage(self) -> None:
        return None

    def show_prompt(self, text: str) -> float:
        self._clock += self.seconds_per_question
        return self._clock

    def ask(self, question: Question) -> QuestionResponse:
        onset = self._clock
        self._clock += self.seconds_per_question
        response = self.scripted_responses.get(question.id, True)
        return QuestionResponse(onset=onset, response_time=self._clock, response=response)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class SessionResult:
    """Summary of a completed session."""

    participant: str
    n_passages: int
    n_events: int
    log_path: str


class ReadingSession:
    """Run a reading session and record it to an encrypted log.

    Parameters
    ----------
    participant:
        A pseudonym, not a direct identifier. Pseudonymise upstream with
        :func:`readsync.security.pseudonymise`.
    tracker, presenter, log:
        Injected collaborators, so the session is testable without hardware.
    marker_sink:
        Where word, region and passage markers are sent. Defaults to in-memory.
    layout:
        The fixed-width layout used to compute word interest areas. The presenter
        draws each word at the matching position, so the on-screen geometry and
        the analysis interest areas agree.
    offline:
        When true (the default), the run is wrapped in a ``NetworkGuard``.
    """

    def __init__(
        self,
        *,
        participant: str,
        tracker: Tracker,
        presenter: Presenter,
        log: EventLog,
        marker_sink: MarkerSink | None = None,
        layout: FixedWidthLayout | None = None,
        offline: bool = True,
    ) -> None:
        self.participant = participant
        self.tracker = tracker
        self.presenter = presenter
        self.log = log
        self.marker_sink = marker_sink or NullMarkerSink()
        self.layout = layout or FixedWidthLayout()
        self.offline = offline
        self._last_calibration: str | None = None

    def _emit(self, marker: Marker, event: dict[str, object]) -> None:
        self.marker_sink.send(marker)
        self.log.append(event)

    def _region_event(self, kind: str, t: float, passage_id: str, region: Region) -> None:
        where = {"passage": passage_id, "region": region.id, "layer": region.layer}
        self._emit(Marker(kind, t, where), {"type": kind, "t": t, **where})

    def _run_passage(self, passage: Passage, regions: list[Region]) -> None:
        areas = interest_areas(passage.words, self.layout)
        self.presenter.start_passage(passage, areas)
        current_word: int | None = None
        current_region: Region | None = None
        onset_logged = False
        try:
            while True:
                t, finished = self.presenter.tick()
                if not onset_logged:
                    self._emit(
                        Marker("passage_onset", t, {"passage": passage.id}),
                        {"type": "passage_onset", "t": t, "passage": passage.id},
                    )
                    onset_logged = True
                sample = self.tracker.poll(t)
                if sample is not None:
                    self.log.append(
                        {
                            "type": "gaze",
                            "t": t,
                            "tracker_t": sample.t,
                            "x": round(sample.x, 1),
                            "y": round(sample.y, 1),
                            "valid": sample.valid,
                        }
                    )
                    area = locate(areas, sample.x, sample.y) if sample.valid else None
                    word_index = area.word.index if area is not None else None
                    # A sample that resolves to no word, a blink or gaze in the
                    # space between words, holds the current word and region
                    # instead of ending the visit, so enter and exit markers
                    # fire on genuine transitions only.
                    if word_index is not None:
                        if word_index != current_word:
                            where = {"passage": passage.id, "word": word_index}
                            self._emit(
                                Marker("word_enter", t, where),
                                {"type": "word_enter", "t": t, **where},
                            )
                            current_word = word_index
                        region = region_at(regions, word_index)
                        if region is not current_region:
                            if current_region is not None:
                                self._region_event(
                                    "region_exit", t, passage.id, current_region
                                )
                            if region is not None:
                                self._region_event("region_enter", t, passage.id, region)
                            current_region = region
                if finished:
                    if current_region is not None:
                        self._region_event("region_exit", t, passage.id, current_region)
                    self._emit(
                        Marker("passage_offset", t, {"passage": passage.id}),
                        {"type": "passage_offset", "t": t, "passage": passage.id},
                    )
                    break
        finally:
            self.presenter.end_passage()

    def _drift_check(self, passage_id: str) -> None:
        drift = getattr(self.tracker, "drift_correct", None)
        if drift is None:
            return
        error = drift()
        self.log.append({"type": "drift_check", "passage": passage_id, "error": error})
        # A drift check can escalate into a full recalibration; when the
        # tracker's calibration message changed, record the fresh outcome so
        # the quality report shows every calibration, not only the first.
        message = getattr(self.tracker, "last_calibration_message", None)
        if message is not None and str(message) != self._last_calibration:
            self._last_calibration = str(message)
            self.log.append({"type": "calibration", "detail": self._last_calibration})

    def _show_prompts(self, item: StimulusItem) -> None:
        show = getattr(self.presenter, "show_prompt", None)
        if show is None:
            return
        for question in item.questions:
            t = float(show(question.text))
            where = {"passage": item.passage.id, "question": question.id}
            self._emit(Marker("prompt", t, where), {"type": "prompt", "t": t, **where})

    def _ask_questions(self, item: StimulusItem) -> None:
        ask = getattr(self.presenter, "ask", None)
        if ask is None:
            return
        for question in item.questions:
            result = ask(question)
            where = {"passage": item.passage.id, "question": question.id}
            self._emit(
                Marker("response", result.response_time, where),
                {
                    "type": "response",
                    "t": result.response_time,
                    "onset": result.onset,
                    **where,
                    "kind": question.kind,
                    "region": question.region,
                    "response": result.response,
                    "correct": result.response == question.answer,
                },
            )

    def _run(self, items: list[StimulusItem]) -> SessionResult:
        self.tracker.connect()
        self.tracker.calibrate()
        calibration = getattr(self.tracker, "last_calibration_message", None)
        if calibration is not None:
            self._last_calibration = str(calibration)
            self.log.append({"type": "calibration", "detail": self._last_calibration})
        self.tracker.start_recording()
        self.log.append({"type": "session_start", "participant": self.participant})
        try:
            for item in items:
                self._drift_check(item.passage.id)
                if item.question_position == "before":
                    self._show_prompts(item)
                self._run_passage(item.passage, item.regions)
                self._ask_questions(item)
        finally:
            self.tracker.stop_recording()
            self.log.append({"type": "session_end", "participant": self.participant})
            self.tracker.close()
            self.presenter.close()
        return SessionResult(
            participant=self.participant,
            n_passages=len(items),
            n_events=len(self.log),
            log_path=str(self.log.path),
        )

    def run(self, passages: list[Passage]) -> SessionResult:
        """Run the session over plain passages, offline by default."""
        return self.run_items([StimulusItem(passage=passage) for passage in passages])

    def run_items(self, items: list[StimulusItem]) -> SessionResult:
        """Run the session over annotated stimulus items, offline by default.

        Items carry the region annotation and the comprehension questions, so
        this is the entry point for an experiment loaded with
        :func:`readsync.stimuli.load_stimulus_set`. Region enter and exit
        markers are emitted as gaze moves through annotated regions. Where the
        presenter implements ``ask``, each item's questions are asked after its
        passage (or shown before it, under the information-seeking regime) and
        every response is scored and logged; with a presenter that cannot ask,
        the questions are skipped and no response events appear.
        """
        if self.offline:
            with NetworkGuard():
                return self._run(items)
        return self._run(items)
