"""Tests for the PsychoPy presenter that do not need a display.

The drawing path needs PsychoPy and a screen and is exercised in the lab and in
the example script. The coordinate conversion and the lazy-import behaviour are
pure logic and are tested here.
"""

from __future__ import annotations

import importlib.util

import pytest

from readsync.presenters import PsychoPyPresenter
from readsync.text import InterestArea, Passage, Word

_HAS_PSYCHOPY = importlib.util.find_spec("psychopy") is not None


def test_constructs_without_psychopy() -> None:
    presenter = PsychoPyPresenter(size=(1000, 800), fullscreen=False)
    assert presenter.size == (1000, 800)


def test_coordinate_conversion_flips_and_centres() -> None:
    presenter = PsychoPyPresenter(size=(1000, 800))
    word = Word(text="x", index=0, char_start=0, char_end=1)
    area = InterestArea(word=word, x1=400, y1=300, x2=420, y2=340)  # centre (410, 320)
    px, py = presenter._to_psychopy(area)
    assert px == 410 - 500  # x: centre origin
    assert py == 400 - 320  # y: flipped about the middle


@pytest.mark.skipif(_HAS_PSYCHOPY, reason="PsychoPy is installed; cannot open a window in CI")
def test_start_passage_without_psychopy_raises_clearly() -> None:
    presenter = PsychoPyPresenter(fullscreen=False)
    with pytest.raises(RuntimeError, match="PsychoPy is not installed"):
        presenter.start_passage(Passage(id="p", text="hello world"), [])


def test_a_window_of_another_size_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests import _fake_psychopy

    _fake_psychopy.install(monkeypatch, actual_size=(2560, 1440))
    presenter = PsychoPyPresenter(size=(1920, 1080))
    with pytest.raises(RuntimeError, match="2560x1440"):
        presenter.start_passage(Passage(id="p", text="hello"), [])


def test_question_text_stays_drawn_on_every_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    from readsync.stimuli import Question
    from tests import _fake_psychopy

    _fake_psychopy.install(monkeypatch)
    presenter = PsychoPyPresenter(size=(800, 600), fullscreen=False)
    presenter._ensure_window()
    presenter._clock.now = 5.0
    presenter._keyboard.script = [[], [], [_fake_psychopy.KeyPress("n", 0.75)]]
    response = presenter.ask(Question(id="q", text="Was it red?", answer=False))
    events = presenter._win.events
    flips = [i for i, event in enumerate(events) if event == "flip"]
    assert flips and all(events[i - 1].startswith("draw:Was it red?") for i in flips)
    assert (response.onset, response.response_time, response.response) == (5.0, 5.75, False)


def test_check_layout_compares_the_drawn_width_with_the_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from readsync.text import FixedWidthLayout
    from tests import _fake_psychopy

    _fake_psychopy.install(monkeypatch)
    presenter = PsychoPyPresenter(size=(800, 600), fullscreen=False)
    # The fake font is 0.6 of the letter height wide: 0.7 * 40 * 0.6 = 16.8 px.
    assert presenter.check_layout(FixedWidthLayout(char_width=17, line_height=40)) == (
        pytest.approx(16.8)
    )
    with pytest.raises(ValueError, match="layout assumes 12 px"):
        presenter.check_layout(FixedWidthLayout(char_width=12, line_height=40))


def test_check_layout_detects_a_proportional_font(monkeypatch: pytest.MonkeyPatch) -> None:
    from readsync.text import FixedWidthLayout
    from tests import _fake_psychopy

    class _Proportional(_fake_psychopy.TextStim):
        def __init__(self, win: object, *, text: str, height: float = 20.0, **kw: object) -> None:
            super().__init__(win, text=text, height=height, **kw)  # type: ignore[arg-type]
            scale = 0.4 if text.startswith("i") else 0.8
            self.boundingBox = (len(text) * height * scale, height)

    _fake_psychopy.install(monkeypatch)
    import sys

    monkeypatch.setattr(sys.modules["psychopy.visual"], "TextStim", _Proportional)
    presenter = PsychoPyPresenter(size=(800, 600), fullscreen=False)
    with pytest.raises(ValueError, match="monospaced"):
        presenter.check_layout(FixedWidthLayout(char_width=17, line_height=40))
