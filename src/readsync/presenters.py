"""PsychoPy presenter for running reading sessions on a real screen.

``PsychoPyPresenter`` implements the :class:`readsync.session.Presenter` protocol
over a PsychoPy window. Each word is drawn at the centre of its interest area, so
what the reader sees and what the analysis treats as a word region are the same
geometry. The reader advances with a keypress.

PsychoPy is large and platform sensitive, so it is an optional dependency
(``pip install "readsync[present]"``) and is imported lazily. This module
therefore imports without PsychoPy or a display; only constructing and running a
presenter needs them. Because it needs a screen, it is exercised in the lab and
in the example script, and continuous integration covers the pure parts.

A note on geometry. PsychoPy pixel coordinates have their origin at the centre of
the window with the y axis pointing up, whereas interest areas use a top-left
origin with y pointing down. :meth:`_to_psychopy` converts between them. For the
on-screen word boxes to match the interest areas exactly, set the layout's
``char_width`` and ``line_height`` from the chosen monospaced font's metrics; the
defaults are a reasonable starting point that should be calibrated once per
display and font.
"""

from __future__ import annotations

from typing import Any

from .session import QuestionResponse
from .stimuli import Question
from .text import InterestArea, Passage

__all__ = ["PsychoPyPresenter"]


class PsychoPyPresenter:
    """Present reading passages in a PsychoPy window.

    Parameters
    ----------
    size:
        Window size in pixels. Must match the screen size used to compute the
        interest areas, so positions line up.
    fullscreen:
        Run full screen. Use ``False`` for development on a windowed display.
    font:
        A monospaced font, so word widths match the fixed-width layout.
    advance_keys:
        Keys that end a passage. Defaults to the space bar.
    background, text_color:
        Colours, named or as PsychoPy colour values.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1920, 1080),
        fullscreen: bool = True,
        font: str = "Consolas",
        advance_keys: tuple[str, ...] = ("space",),
        background: str = "white",
        text_color: str = "black",
        monitor: str = "testMonitor",
    ) -> None:
        self.size = size
        self.fullscreen = fullscreen
        self.font = font
        self.advance_keys = advance_keys
        self.background = background
        self.text_color = text_color
        self.monitor = monitor
        self._win: Any = None
        self._clock: Any = None
        self._keyboard: Any = None
        self._visual: Any = None
        self._stims: list[Any] = []

    def _ensure_window(self) -> None:
        if self._win is not None:
            return
        try:
            from psychopy import core, visual
            from psychopy.hardware import keyboard
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise RuntimeError(
                "PsychoPy is not installed. Install the 'present' extra "
                '(pip install "readsync[present]") to run on a screen.'
            ) from exc
        self._visual = visual
        self._win = visual.Window(
            size=list(self.size),
            units="pix",
            fullscr=self.fullscreen,
            color=self.background,
            monitor=self.monitor,
            allowGUI=not self.fullscreen,
        )
        self._clock = core.Clock()
        self._keyboard = keyboard.Keyboard()

    def _to_psychopy(self, area: InterestArea) -> tuple[float, float]:
        """Convert an interest-area centre to PsychoPy pixel coordinates."""
        width, height = self.size
        cx = (area.x1 + area.x2) / 2
        cy = (area.y1 + area.y2) / 2
        return cx - width / 2, height / 2 - cy

    def start_passage(  # pragma: no cover
        self, passage: Passage, areas: list[InterestArea]
    ) -> None:
        self._ensure_window()
        self._stims = [
            self._visual.TextStim(
                self._win,
                text=area.word.text,
                pos=self._to_psychopy(area),
                height=(area.y2 - area.y1) * 0.7,
                font=self.font,
                color=self.text_color,
                anchorHoriz="center",
                anchorVert="center",
            )
            for area in areas
        ]
        self._keyboard.clearEvents()

    def tick(self) -> tuple[float, bool]:  # pragma: no cover
        for stim in self._stims:
            stim.draw()
        self._win.flip()
        t = float(self._clock.getTime())
        keys = self._keyboard.getKeys(list(self.advance_keys), waitRelease=False)
        return t, len(keys) > 0

    def end_passage(self) -> None:  # pragma: no cover
        self._stims = []
        if self._win is not None:
            self._win.flip()

    def _text_screen(self, text: str) -> None:  # pragma: no cover
        self._ensure_window()
        stim = self._visual.TextStim(
            self._win,
            text=text,
            font=self.font,
            color=self.text_color,
            wrapWidth=self.size[0] * 0.8,
        )
        stim.draw()
        self._win.flip()
        self._keyboard.clearEvents()

    def show_prompt(self, text: str) -> float:  # pragma: no cover
        """Show a question before its passage and wait for the advance key."""
        self._text_screen(text)
        while not self._keyboard.getKeys(list(self.advance_keys), waitRelease=False):
            self._win.flip()
        return float(self._clock.getTime())

    def ask(self, question: Question) -> QuestionResponse:  # pragma: no cover
        """Present a yes/no question and collect the keyed response.

        ``Y`` answers yes and ``N`` no. The onset is the first frame on which
        the question was drawn, and the response time the moment the key came
        in, both on the same session clock as the reading record.
        """
        self._text_screen(f"{question.text}\n\nY = yes        N = no")
        onset = float(self._clock.getTime())
        while True:
            keys = self._keyboard.getKeys(["y", "n"], waitRelease=False)
            if keys:
                key = keys[0]
                return QuestionResponse(
                    onset=onset,
                    response_time=float(self._clock.getTime()),
                    response=key.name == "y",
                )
            self._win.flip()

    def close(self) -> None:  # pragma: no cover
        if self._win is not None:
            self._win.close()
            self._win = None
