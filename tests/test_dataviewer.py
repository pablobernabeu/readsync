"""Tests for the EyeLink Data Viewer trial messages and interest-area sets."""

from __future__ import annotations

from pathlib import Path

from readsync.dataviewer import (
    ias_text,
    trial_end_messages,
    trial_start_messages,
    write_ias,
)
from readsync.markers import EYELINK_MESSAGE_LIMIT
from readsync.text import FixedWidthLayout, InterestArea, interest_areas, tokenise


def _areas() -> list[InterestArea]:
    layout = FixedWidthLayout(char_width=10, line_height=30, max_chars_per_line=12, x0=0, y0=0)
    return interest_areas(tokenise("The cat sat on the mat."), layout)


def test_ias_text_has_one_tab_separated_rectangle_per_word() -> None:
    lines = ias_text(_areas()).splitlines()
    assert lines[0].startswith("#")
    rows = [line.split("\t") for line in lines[1:]]
    assert len(rows) == 6
    assert rows[0] == ["RECTANGLE", "1", "0", "0", "30", "30", "The"]
    # 'on' wraps to the second line at 12 characters per line.
    assert rows[3] == ["RECTANGLE", "4", "0", "30", "20", "60", "on"]
    assert [row[1] for row in rows] == ["1", "2", "3", "4", "5", "6"]


def test_write_ias_creates_the_file(tmp_path: Path) -> None:
    out = write_ias(_areas(), tmp_path / "sets" / "p1.ias")
    assert out.read_text(encoding="utf-8") == ias_text(_areas())


def test_trial_start_messages_open_the_trial_and_name_the_set() -> None:
    assert trial_start_messages("p1", "p1.ias") == ["TRIALID p1", "!V IAREA FILE p1.ias"]
    assert trial_start_messages("p1") == ["TRIALID p1"]


def test_trial_end_messages_attach_variables_and_close_the_trial() -> None:
    messages = trial_end_messages({"passage": "p1", "condition": "high", "pair": None})
    assert messages == [
        "!V TRIAL_VAR passage p1",
        "!V TRIAL_VAR condition high",
        "TRIAL_RESULT 0",
    ]
    assert trial_end_messages() == ["TRIAL_RESULT 0"]


def test_trial_messages_keep_one_token_per_field_and_respect_the_cap() -> None:
    assert trial_end_messages({"cond ition": "two words"})[0] == "!V TRIAL_VAR cond_ition two_words"
    long = trial_start_messages("x" * 300)[0]
    assert len(long) == EYELINK_MESSAGE_LIMIT
