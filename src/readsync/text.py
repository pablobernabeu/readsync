"""Reading stimuli and word-level interest areas.

Reading research measures eye movements on individual words and short regions,
so a stimulus is not just a string. It is a sequence of words, each with a
character span in the text and a bounding box on the screen. Those bounding
boxes are the interest areas against which fixations are later assigned to words
by tools such as Eyekit or popEye. This module produces them deterministically
from a fixed-width layout, which is the standard choice for reading studies
because it makes word boundaries exact.

Coordinates are in pixels with the origin at the top-left of the text block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Word",
    "InterestArea",
    "FixedWidthLayout",
    "Passage",
    "Region",
    "region_at",
    "tokenise",
    "interest_areas",
    "locate",
]

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Word:
    """A whitespace-delimited token with its character span in the source text."""

    text: str
    index: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Region:
    """A contiguous span of words annotated with the sub-process it loads.

    Reading designs manipulate multi-word regions, not only single words: a
    region is built to load one layer of reading, and a matched comparison
    region controls for length and position. ``start`` and ``end`` are word
    indices into the passage, ``end`` exclusive. ``layer`` names the
    sub-process the region loads; the label is free text, agreed per study
    (the bundled sets use ``decoding``, ``vocabulary`` and ``integration``).
    ``role`` separates a ``target`` region from its matched ``comparison``.
    """

    id: str
    start: int
    end: int
    layer: str
    role: str = "target"

    def contains_word(self, word_index: int) -> bool:
        return self.start <= word_index < self.end


def region_at(regions: list[Region], word_index: int) -> Region | None:
    """Return the first region containing ``word_index``, or ``None``."""
    for region in regions:
        if region.contains_word(word_index):
            return region
    return None


@dataclass(frozen=True)
class InterestArea:
    """A word's bounding box on screen, in pixels."""

    word: Word
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


@dataclass(frozen=True)
class FixedWidthLayout:
    """A monospaced layout. ``char_width`` and ``line_height`` are in pixels.

    ``max_chars_per_line`` wraps the text at whole words, as a reading display
    does. ``x0`` and ``y0`` place the top-left of the text block.
    """

    char_width: int = 16
    line_height: int = 40
    max_chars_per_line: int = 80
    x0: int = 100
    y0: int = 100


@dataclass
class Passage:
    """A reading passage with derived words and interest areas."""

    id: str
    text: str
    words: list[Word] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.words:
            self.words = tokenise(self.text)


def tokenise(text: str) -> list[Word]:
    """Split ``text`` into whitespace-delimited words with character spans."""
    return [
        Word(text=m.group(), index=i, char_start=m.start(), char_end=m.end())
        for i, m in enumerate(_WORD_RE.finditer(text))
    ]


def interest_areas(words: list[Word], layout: FixedWidthLayout) -> list[InterestArea]:
    """Lay the words out left to right with word wrapping and return one interest
    area per word.

    The algorithm mirrors a simple reading display: words are placed with a
    single space between them, and a word that would overflow the line starts a
    new line. Because the layout is fixed-width, each word's box is exact.
    """
    areas: list[InterestArea] = []
    col = 0
    row = 0
    for word in words:
        length = len(word.text)
        if col > 0 and col + 1 + length > layout.max_chars_per_line:
            row += 1
            col = 0
        if col > 0:
            col += 1  # the space before the word
        x1 = layout.x0 + col * layout.char_width
        y1 = layout.y0 + row * layout.line_height
        x2 = x1 + length * layout.char_width
        y2 = y1 + layout.line_height
        areas.append(InterestArea(word=word, x1=x1, y1=y1, x2=x2, y2=y2))
        col += length
    return areas


def locate(areas: list[InterestArea], x: float, y: float) -> InterestArea | None:
    """Return the interest area containing point ``(x, y)``, or ``None``."""
    for area in areas:
        if area.contains(x, y):
            return area
    return None
