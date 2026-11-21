"""Running a reading session.

``ReadingSession`` ties the pieces together. It presents passages, polls the
tracker for gaze, sends markers at word, region and passage boundaries, asks the
comprehension questions and writes everything to the encrypted event log. By
default, the run happens inside a ``NetworkGuard``, which blocks outbound
connections from Python code while a participant is recorded (see SECURITY.md
for what it does not cover).

Presentation sits behind a ``Presenter`` protocol. On each frame, the session
calls ``presenter.tick``, which draws the current passage, returns the session
time and reports whether the reader has asked to move on. ``PsychoPyPresenter``
(see ``readsync.presenters``) runs sessions on a screen, and ``HeadlessPresenter``
advances a virtual clock with no display, which lets the session be tested
without PsychoPy.

Events are stamped with ``t`` on the presenter's session clock, so one log has a
single time base. Gaze events also carry the sample's time on the tracker's own
clock as ``tracker_t``, for cross-checking against the tracker's data file.

Each passage is one tracker trial. Recording starts at the passage and stops at
its end, and the drift check runs before recording starts, while the tracker is
idle, which is the order SR Research's trial structure requires.

Questions, drift checks and trial messages are optional capabilities,
discovered on the collaborators at run time. A presenter that implements
``ask`` (and, for the information-seeking regime, ``show_prompt``) collects
responses. A tracker that implements ``drift_correct`` is checked before each
passage, and one that implements ``begin_trial`` and ``end_trial`` is told
where each trial starts and ends, which is how the EyeLink data file receives
its trial and interest-area messages. Backends without these methods run
without them.
"""

from __future__ import annotations

import uuid
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

    Three further methods are optional and looked up at run time, outside this
    protocol. ``now()`` returns the session clock, so that events outside a
    passage, such as calibration and the session boundaries, are stamped on the
    same clock. ``ask(question)`` presents a yes/no comprehension question and
    returns a :class:`QuestionResponse`. ``show_prompt(text)`` displays a
    question before its passage, for the information-seeking regime, and
    returns the time at which the reader moved on.
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

    def now(self) -> float:
        return self._clock

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
    """Summary of a completed session.

    ``log_head`` is the number of records in the log and the chain hash of the
    last one. Storing it apart from the log lets a later reader confirm that the
    log is complete and unaltered.
    """

    participant: str
    session_id: str
    n_passages: int
    n_events: int
    log_path: str
    log_head: tuple[int, str]


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

    def _region_event(
        self, kind: str, t: float, passage_id: str, region: Region, **detail: object
    ) -> None:
        where = {"passage": passage_id, "region": region.id, "layer": region.layer}
        self._emit(Marker(kind, t, where), {"type": kind, "t": t, **where, **detail})

    def _run_passage(
        self, passage: Passage, regions: list[Region], areas: list[InterestArea]
    ) -> None:
        self.presenter.start_passage(passage, areas)
        current_word: int | None = None
        # The regions that contain the current word, in annotation order.
        # Tracking all of them lets nested and overlapping regions, for example
        # a target word inside a sentence-level region, each get their events.
        active: list[Region] = []
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
                    # A sample that resolves to no word, during a blink, in the
                    # space between words or off the text, holds the current
                    # word and regions, so enter and exit events fire only when
                    # the gaze lands on another word.
                    if word_index is not None and word_index != current_word:
                        where = {"passage": passage.id, "word": word_index}
                        self._emit(
                            Marker("word_enter", t, where),
                            {"type": "word_enter", "t": t, **where, "tracker_t": sample.t},
                        )
                        current_word = word_index
                        here = [r for r in regions if r.contains_word(word_index)]
                        for region in active:
                            if region not in here:
                                self._region_event(
                                    "region_exit", t, passage.id, region,
                                    reason="gaze", tracker_t=sample.t,
                                )
                        for region in here:
                            if region not in active:
                                self._region_event(
                                    "region_enter", t, passage.id, region, tracker_t=sample.t
                                )
                        active = here
                if finished:
                    for region in active:
                        self._region_event(
                            "region_exit", t, passage.id, region, reason="passage_end"
                        )
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
        event: dict[str, object] = {
            "type": "drift_check", "t": self._now(), "passage": passage_id, "error": error
        }
        # A tracker may report the check's status, non-zero when it was aborted
        # or escalated into a recalibration. Without one, a check that reports
        # no error is taken to have been aborted or escalated.
        status = getattr(self.tracker, "last_drift_status", None)
        if status is not None:
            event["status"] = status
        self.log.append(event)
        escalated = status != 0 if status is not None else error is None
        # The calibration outcome is logged after an escalated check and whenever
        # the tracker's message changes, so every recalibration shows in the
        # quality report even when its message repeats an earlier one.
        message = getattr(self.tracker, "last_calibration_message", None)
        if message is not None and (escalated or str(message) != self._last_calibration):
            self._last_calibration = str(message)
            self.log.append(
                {"type": "calibration", "t": self._now(), "detail": self._last_calibration}
            )

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

    def _now(self) -> float | None:
        """The presenter's session clock, where the presenter exposes one."""
        now = getattr(self.presenter, "now", None)
        return float(now()) if now is not None else None

    def _lay_out(self, items: list[StimulusItem]) -> list[list[InterestArea]]:
        """Lay out every passage before the tracker connects.

        A word too long for a line, or a passage that does not fit the
        presenter's window, then fails before the participant is calibrated.
        """
        size = getattr(self.presenter, "size", None)
        layouts = []
        for item in items:
            areas = interest_areas(item.passage.words, self.layout)
            if size is not None:
                for area in areas:
                    if area.x1 < 0 or area.y1 < 0 or area.x2 > size[0] or area.y2 > size[1]:
                        raise ValueError(
                            f"passage {item.passage.id!r}: the word {area.word.text!r} falls "
                            f"outside the {size[0]}x{size[1]} window"
                        )
            layouts.append(areas)
        return layouts

    def _describe(self) -> dict[str, object]:
        """The settings that the tracker and presenter report, for the record."""
        settings: dict[str, object] = {}
        for name, part in (("tracker", self.tracker), ("presenter", self.presenter)):
            describe = getattr(part, "describe", None)
            if describe is not None:
                settings[name] = describe()
        return settings

    def _run_item(self, item: StimulusItem, areas: list[InterestArea]) -> None:
        """Run one item as one tracker trial.

        The order follows SR Research's trial structure: the trial is opened,
        the drift check runs while the tracker is idle, recording covers the
        passage only, and the trial is closed after the questions.
        """
        passage = item.passage
        begin_trial = getattr(self.tracker, "begin_trial", None)
        if begin_trial is not None:
            begin_trial(passage.id, areas)
        if item.question_position == "before":
            self._show_prompts(item)
        self._drift_check(passage.id)
        self.tracker.start_recording()
        try:
            self._run_passage(passage, item.regions, areas)
        finally:
            self.tracker.stop_recording()
            self.log.sync()
        self._ask_questions(item)
        end_trial = getattr(self.tracker, "end_trial", None)
        if end_trial is not None:
            end_trial({"passage": passage.id, "condition": item.condition})

    def _run(self, items: list[StimulusItem]) -> SessionResult:
        if len(self.log):
            raise ValueError(
                f"the log at {self.log.path} already holds {len(self.log)} records; "
                "give each session its own log file"
            )
        tracker_size = getattr(self.tracker, "screen_size", None)
        window_size = getattr(self.presenter, "size", None)
        if (
            tracker_size is not None
            and window_size is not None
            and tuple(tracker_size) != tuple(window_size)
        ):
            raise ValueError(
                f"the tracker reports gaze in a {tracker_size[0]}x{tracker_size[1]} "
                f"frame but the presenter's window is {window_size[0]}x{window_size[1]}, "
                "so gaze would not line up with the words"
            )
        layouts = self._lay_out(items)
        session_id = uuid.uuid4().hex
        end: dict[str, object] = {"status": "aborted"}
        started = False
        try:
            # Reading the clock opens a presenter's window, so its size is
            # checked before the tracker connects.
            self._now()
            self.tracker.connect()
            self.tracker.calibrate()
            calibration = getattr(self.tracker, "last_calibration_message", None)
            if calibration is not None:
                self._last_calibration = str(calibration)
                self.log.append(
                    {"type": "calibration", "t": self._now(), "detail": self._last_calibration}
                )
            self.log.append(
                {
                    "type": "session_start",
                    "t": self._now(),
                    "participant": self.participant,
                    "session": session_id,
                    "planned": [item.passage.id for item in items],
                    **self._describe(),
                }
            )
            started = True
            for item, areas in zip(items, layouts, strict=True):
                self._run_item(item, areas)
            end = {"status": "completed"}
        except BaseException as exc:
            end["error"] = type(exc).__name__
            raise
        finally:
            # Each step runs even if an earlier one fails, so the tracker's data
            # file is closed and received whatever happened to the log.
            try:
                if started:
                    self.log.append(
                        {
                            "type": "session_end",
                            "t": self._now(),
                            "participant": self.participant,
                            "session": session_id,
                            **end,
                        }
                    )
                self.log.sync()
            finally:
                try:
                    self.tracker.close()
                finally:
                    self.presenter.close()
        return SessionResult(
            participant=self.participant,
            session_id=session_id,
            n_passages=len(items),
            n_events=len(self.log),
            log_path=str(self.log.path),
            log_head=self.log.head,
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
        passage, and under the information-seeking regime also shown before it,
        and every response is scored and logged. With a presenter that cannot
        ask, the questions are skipped and no response events appear.
        """
        if self.offline:
            with NetworkGuard():
                return self._run(items)
        return self._run(items)
