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
