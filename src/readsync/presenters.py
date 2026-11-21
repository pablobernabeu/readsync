"""PsychoPy presenter for running reading sessions on a real screen.

``PsychoPyPresenter`` implements the :class:`readsync.session.Presenter` protocol
over a PsychoPy window. Each word is drawn centred on its interest area, and the
reader advances with a keypress.

PsychoPy is large and platform-sensitive, so it is an optional dependency
(``pip install "readsync[present]"``) imported only when a window is opened.
This module therefore imports without PsychoPy or a display. The drawing code
needs a screen, so it runs in the example script, and the tests cover the
coordinate conversion.

PsychoPy pixel coordinates have their origin at the centre of
the window with the y axis pointing up, whereas interest areas use a top-left
origin with y pointing down. :meth:`_to_psychopy` converts between them. The
glyphs fill the interest areas only when the layout's ``char_width`` and
``line_height`` match the metrics of the monospaced font at the size drawn, so
set them from the font once per display. The defaults are a starting point.
"""

from __future__ import annotations

from typing import Any

from .session import QuestionResponse
from .stimuli import Question
from .text import FixedWidthLayout, InterestArea, Passage

__all__ = ["PsychoPyPresenter"]


class PsychoPyPresenter:
    """Present reading passages in a PsychoPy window.

    Parameters
    ----------
    size:
        Window size in pixels. Interest areas are in this window's pixels, so with
        an eye-tracker the window must be full screen at the display's
        resolution, and the tracker's ``screen_size`` must be the same. Opening a
        window of any other size raises ``RuntimeError``.
    fullscreen:
        Run full screen. Use ``False`` for development on a windowed display.
    font:
        A monospaced font, so word widths match the fixed-width layout.
    letter_height:
        The text height in pixels. When ``None``, it is 0.7 of the layout's line
        height. Use :meth:`check_layout` to confirm that the font's character
        width at this height matches the layout's ``char_width``.
    advance_keys:
        Keys that end a passage. Defaults to the space bar.
    background, text_color:
        Colours, named or as PsychoPy colour values.
    monitor:
        The name of the PsychoPy monitor profile.
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
        letter_height: float | None = None,
    ) -> None:
        self.size = size
        self.fullscreen = fullscreen
        self.font = font
        self.advance_keys = advance_keys
        self.background = background
        self.text_color = text_color
        self.monitor = monitor
        self.letter_height = letter_height
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
            # On high-density (Retina) displays, a pixel then stays one screen
            # pixel, so the window's size matches the requested size.
            useRetina=False,
        )
        actual = tuple(int(round(v)) for v in self._win.size)
        if actual != tuple(self.size):
            self._win.close()
            self._win = None
            raise RuntimeError(
                f"the window opened at {actual[0]}x{actual[1]} pixels, not the "
                f"requested {self.size[0]}x{self.size[1]}, so words would be drawn "
                "away from their interest areas. Set size to the display's resolution."
            )
        self._clock = core.Clock()
        self._keyboard = keyboard.Keyboard()

    def _to_psychopy(self, area: InterestArea) -> tuple[float, float]:
        """Convert an interest-area centre to PsychoPy pixel coordinates."""
        width, height = self.size
        cx = (area.x1 + area.x2) / 2
        cy = (area.y1 + area.y2) / 2
        return cx - width / 2, height / 2 - cy

    def start_passage(
        self, passage: Passage, areas: list[InterestArea]
    ) -> None:
        self._ensure_window()
        self._stims = [
            self._visual.TextStim(
                self._win,
                text=area.word.text,
                pos=self._to_psychopy(area),
                height=self._letter_height(area.y2 - area.y1),
                font=self.font,
                color=self.text_color,
                anchorHoriz="center",
                anchorVert="center",
            )
            for area in areas
        ]
        self._keyboard.clearEvents()

    def tick(self) -> tuple[float, bool]:
        for stim in self._stims:
            stim.draw()
        self._win.flip()
        t = float(self._clock.getTime())
        keys = self._keyboard.getKeys(list(self.advance_keys), waitRelease=False)
        return t, len(keys) > 0

    def end_passage(self) -> None:
        self._stims = []
        if self._win is not None:
            self._win.flip()

    def describe(self) -> dict[str, object]:
        """The settings the session records with its start."""
        return {
            "presenter": "psychopy",
            "size": list(self.size),
            "fullscreen": self.fullscreen,
            "font": self.font,
            "letter_height": self.letter_height,
        }

    def _letter_height(self, line_height: float) -> float:
        return self.letter_height if self.letter_height is not None else 0.7 * line_height

    def check_layout(self, layout: FixedWidthLayout, *, tolerance: float = 0.05) -> float:
        """Measure the drawn character width and compare it with the layout.

        Renders runs of a narrow and a wide letter in the presenter's font at its
        letter height and returns the mean width of one character in pixels.
        Raises ``ValueError`` when either width differs from ``layout.char_width``
        by more than ``tolerance`` as a fraction, because the words would then not
        fill their interest areas, or the font is not monospaced. Call it once
        per display before a session.
        """
        self._ensure_window()
        widths = []
        # A narrow and a wide letter: in a proportional font, or a fallback font
        # substituted for a missing one, their widths differ.
        for letter in ("i", "M"):
            sample = letter * 40
            stim = self._visual.TextStim(
                self._win,
                text=sample,
                font=self.font,
                height=self._letter_height(layout.line_height),
            )
            widths.append(float(stim.boundingBox[0]) / len(sample))
        for width in widths:
            if abs(width - layout.char_width) > tolerance * layout.char_width:
                raise ValueError(
                    f"{self.font} at this letter height draws characters "
                    f"{min(widths):.1f} to {max(widths):.1f} px wide, but the layout "
                    f"assumes {layout.char_width} px. Use a monospaced font, or adjust "
                    "letter_height or the layout's char_width."
                )
        return sum(widths) / len(widths)

    @property
    def window(self) -> Any:
        """The PsychoPy window, opened on first use.

        Pass it to an EyeLink calibration graphics environment so that
        calibration is drawn in the same window as the passages.
        """
        self._ensure_window()
        return self._win

    def now(self) -> float:
        """The session clock, which starts when the window opens."""
        self._ensure_window()
        return float(self._clock.getTime())

    def _text_screen(self, text: str, on_flip: Any = None) -> Any:
        """Draw a text screen, flip it on and return the stimulus.

        ``flip`` clears the back buffer, so callers redraw the returned stimulus
        before every further flip to keep the text on screen. ``on_flip`` is
        called at the flip that shows the text.
        """
        self._ensure_window()
        stim = self._visual.TextStim(
            self._win,
            text=text,
            font=self.font,
            color=self.text_color,
            wrapWidth=self.size[0] * 0.8,
        )
        stim.draw()
        if on_flip is not None:
            self._win.callOnFlip(on_flip)
        self._win.flip()
        self._keyboard.clearEvents()
        return stim

    def show_prompt(self, text: str) -> float:
        """Show a question before its passage and wait for the advance key."""
        stim = self._text_screen(text)
        while not self._keyboard.getKeys(list(self.advance_keys), waitRelease=False):
            stim.draw()
            self._win.flip()
        return float(self._clock.getTime())

    def ask(self, question: Question) -> QuestionResponse:
        """Present a yes/no question and collect the keyed response.

        ``Y`` answers yes and ``N`` no. The onset is the time just after the
        question was first flipped on screen. The keyboard's clock is reset at
        that flip, so the response time is the onset plus the key's own reaction
        time, with no wait for the next frame. Both are on the same session clock
        as the reading record.
        """
        self._ensure_window()
        stim = self._text_screen(
            f"{question.text}\n\nY = yes        N = no", on_flip=self._keyboard.clock.reset
        )
        onset = float(self._clock.getTime())
        while True:
            keys = self._keyboard.getKeys(["y", "n"], waitRelease=False)
            if keys:
                key = keys[0]
                return QuestionResponse(
                    onset=onset,
                    response_time=onset + float(key.rt),
                    response=key.name == "y",
                )
            stim.draw()
            self._win.flip()

    def close(self) -> None:
        if self._win is not None:
            self._win.close()
            self._win = None
