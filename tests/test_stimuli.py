"""Tests for the stimulus loader, including the bundled annotated sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from readsync.stimuli import load_passages, load_stimulus_set

_STIMULI = Path(__file__).resolve().parent.parent / "stimuli"
_CONTROLLED = _STIMULI / "controlled_freq_en.json"
_LENGTH = _STIMULI / "controlled_length_en.json"
_NATURAL = _STIMULI / "naturalistic_en.json"


def test_loads_list_form(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps([{"id": "a", "text": "one two"}, {"id": "b", "text": "three"}]),
        encoding="utf-8",
    )
    passages = load_passages(path)
    assert [p.id for p in passages] == ["a", "b"]
    assert passages[0].words[1].text == "two"


def test_loads_object_passages_form(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"language": "en", "passages": [{"id": "x", "text": "hello world"}]}),
        encoding="utf-8",
    )
    passages = load_passages(path)
    assert len(passages) == 1 and passages[0].id == "x"


def test_malformed_item_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object with 'id' and 'text'"):
        load_passages(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_passages(tmp_path / "absent.json")


def test_plain_list_loads_as_unannotated_set(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"id": "a", "text": "one two three"}]), encoding="utf-8")
    stim = load_stimulus_set(path)
    assert len(stim.items) == 1
    assert stim.items[0].condition is None
    assert stim.items[0].target_index is None


def test_controlled_set_is_a_clean_frequency_manipulation() -> None:
    stim = load_stimulus_set(_CONTROLLED)
    assert stim.license == "CC-BY-4.0"
    assert stim.citation  # provenance is recorded
    highs = [i for i in stim.items if i.condition == "high"]
    lows = [i for i in stim.items if i.condition == "low"]
    assert len(highs) == len(lows) == 12

    # Every target is annotated, sits inside the sentence, and is frequency-typed.
    for item in highs:
        assert item.target_index is not None
        assert item.properties["zipf"] >= 4.5
        assert item.word_zipf[item.target_index] == item.properties["zipf"]
    for item in lows:
        assert item.properties["zipf"] <= 3.6

    # Matched pairs share target position and target length, differing in frequency.
    by_pair: dict[str, dict[str, object]] = {}
    for item in stim.items:
        by_pair.setdefault(str(item.properties["pair"]), {})[item.condition] = item
    assert len(by_pair) == 12
    for cells in by_pair.values():
        hi, lo = cells["high"], cells["low"]
        assert hi.target_index == lo.target_index
        assert hi.properties["length"] == lo.properties["length"]
        assert hi.properties["zipf"] > lo.properties["zipf"]


def test_length_set_is_a_clean_length_manipulation() -> None:
    stim = load_stimulus_set(_LENGTH)
    assert stim.name.lower().startswith("controlled word-length")
    shorts = [i for i in stim.items if i.condition == "short"]
    longs = [i for i in stim.items if i.condition == "long"]
    assert len(shorts) == len(longs) >= 8

    by_pair: dict[str, dict[str, object]] = {}
    for item in stim.items:
        by_pair.setdefault(str(item.properties["pair"]), {})[item.condition] = item
    for cells in by_pair.values():
        short, long = cells["short"], cells["long"]
        # Length differs by at least three letters; frequency stays matched.
        assert long.properties["length"] - short.properties["length"] >= 3
        assert abs(short.properties["zipf"] - long.properties["zipf"]) <= 0.6


def test_naturalistic_set_has_per_word_frequencies_and_questions() -> None:
    stim = load_stimulus_set(_NATURAL)
    assert len(stim.items) == 6
    for item in stim.items:
        assert item.condition is None
        assert len(item.word_zipf) == len(item.passage.words)
        assert item.question and item.question.endswith("?")
        assert isinstance(item.answer, bool)


def test_load_passages_reads_the_annotated_controlled_set() -> None:
    passages = load_passages(_CONTROLLED)
    assert len(passages) == 24
    assert all(p.words for p in passages)


def _regions_item(**extra: object) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "r1",
        "text": "the cat sat on the mat",
        "regions": [
            {"id": "target", "start": 1, "end": 3, "layer": "decoding"},
            {"id": "control", "start": 4, "end": 6, "layer": "decoding", "role": "comparison"},
        ],
        "questions": [
            {"id": "q1", "text": "Did the cat sit?", "answer": True,
             "kind": "literal", "region": "target"},
        ],
    }
    item.update(extra)
    return item


def test_regions_and_questions_load_and_validate(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([_regions_item()]), encoding="utf-8")
    item = load_stimulus_set(path).items[0]
    assert [r.id for r in item.regions] == ["target", "control"]
    assert item.regions[0].layer == "decoding"
    assert item.regions[1].role == "comparison"
    assert item.questions[0].region == "target"
    assert item.questions[0].answer is True
    # backward-compatible mirrors
    assert item.question == "Did the cat sit?"
    assert item.answer is True
    assert item.question_position == "after"


def test_region_beyond_the_passage_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    bad = _regions_item(regions=[{"id": "r", "start": 4, "end": 9, "layer": "decoding"}])
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="spans words"):
        load_stimulus_set(path)


def test_question_probing_unknown_region_raises(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    bad = _regions_item(
        questions=[{"id": "q1", "text": "x?", "answer": False, "region": "absent"}]
    )
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown region"):
        load_stimulus_set(path)


def test_single_question_form_becomes_a_question_list(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps([{"id": "a", "text": "one two", "question": "Two words?", "answer": True}]),
        encoding="utf-8",
    )
    item = load_stimulus_set(path).items[0]
    assert len(item.questions) == 1
    assert item.questions[0].id == "a-q1"
    assert item.questions[0].kind == "literal"


def test_question_position_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps([_regions_item(question_position="sideways")]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="question_position"):
        load_stimulus_set(path)


def test_duplicate_region_id_raises(tmp_path: Path) -> None:
    bad = _regions_item(regions=[
        {"id": "r", "start": 0, "end": 1, "layer": "decoding"},
        {"id": "r", "start": 2, "end": 3, "layer": "decoding"},
    ])
    path = tmp_path / "s.json"
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate region id"):
        load_stimulus_set(path)


def test_duplicate_question_id_raises(tmp_path: Path) -> None:
    bad = _regions_item(questions=[
        {"id": "q", "text": "a?", "answer": True},
        {"id": "q", "text": "b?", "answer": False},
    ])
    path = tmp_path / "s.json"
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question id"):
        load_stimulus_set(path)


def test_non_boolean_answer_raises(tmp_path: Path) -> None:
    # bool("no") is True, so a string answer must be refused, not coerced.
    bad = _regions_item(questions=[{"id": "q1", "text": "a?", "answer": "no"}])
    path = tmp_path / "s.json"
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean answer"):
        load_stimulus_set(path)


def test_unknown_question_kind_raises(tmp_path: Path) -> None:
    bad = _regions_item(questions=[
        {"id": "q1", "text": "a?", "answer": True, "kind": "literall"},
    ])
    path = tmp_path / "s.json"
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        load_stimulus_set(path)


def test_id_with_whitespace_raises(tmp_path: Path) -> None:
    # Ids travel in key=value marker messages, so whitespace would corrupt them.
    bad = _regions_item(regions=[{"id": "r 1", "start": 0, "end": 1, "layer": "decoding"}])
    path = tmp_path / "s.json"
    path.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ValueError, match="whitespace"):
        load_stimulus_set(path)
