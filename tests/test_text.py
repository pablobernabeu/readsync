"""Tests for tokenisation and word-level interest areas."""

from __future__ import annotations

from readsync.text import FixedWidthLayout, Passage, interest_areas, locate, tokenise


def test_tokenise_records_character_spans() -> None:
    words = tokenise("the cat sat")
    assert [w.text for w in words] == ["the", "cat", "sat"]
    assert (words[1].char_start, words[1].char_end) == (4, 7)
    assert [w.index for w in words] == [0, 1, 2]


def test_interest_areas_are_non_overlapping_on_a_line() -> None:
    layout = FixedWidthLayout(char_width=10, line_height=30, max_chars_per_line=80, x0=0, y0=0)
    words = tokenise("the cat sat")
    areas = interest_areas(words, layout)
    assert areas[0].x1 == 0 and areas[0].x2 == 30  # "the" is 3 chars * 10 px
    assert areas[1].x1 == 40  # after "the" (30) plus one space (10)
    # boxes on the same line do not overlap
    assert areas[0].x2 <= areas[1].x1


def test_interest_areas_wrap_to_new_line() -> None:
    layout = FixedWidthLayout(char_width=10, line_height=30, max_chars_per_line=7, x0=0, y0=0)
    words = tokenise("alpha beta gamma")
    areas = interest_areas(words, layout)
    # "alpha" (5) fits; "beta" would overflow 7-char line, so it wraps down
    assert areas[0].y1 == 0
    assert areas[1].y1 == 30


def test_locate_finds_the_word_under_a_point() -> None:
    layout = FixedWidthLayout(char_width=10, line_height=30, max_chars_per_line=80, x0=0, y0=0)
    passage = Passage(id="p", text="the cat sat")
    areas = interest_areas(passage.words, layout)
    found = locate(areas, 45, 10)  # inside "cat" (40..70)
    assert found is not None and found.word.text == "cat"
    assert locate(areas, 1000, 1000) is None
