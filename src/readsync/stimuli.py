"""Loading reading stimuli from a stimulus file.

A study keeps its reading materials in a small data file rather than in code, so
they can be reviewed, versioned and swapped without touching the program. This
module reads such a file into structured objects.

The format is JSON with study metadata (name, language, licence, source, citation
and a design note) and a list of items. Each item has an ``id`` and ``text`` and
may carry experimental annotation: a ``condition`` label, a ``target`` word with
its index and properties, per-word frequencies under ``words``, multi-word
``regions`` typed by the layer of reading they load, and comprehension
``questions`` tied to those regions. A simpler file that holds only a list of
``{"id", "text"}`` objects, or an object with a ``"passages"`` key, is also
accepted, so plain passage lists load too.

Regions and questions are validated at load, so a span that runs past the end of
the passage, a duplicate or malformed identifier, or a question probing an
unknown region is caught before a session starts. Identifiers may not contain
whitespace or ``=``, because they travel verbatim in the ``key=value`` marker
messages sent to the eye-tracker and the EEG stream.

The materials are the experimenter's responsibility. A controlled reading study
needs items matched on length, frequency and other properties. The bundled sets
are described in stimuli/README.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text import Passage, Region

__all__ = ["Question", "StimulusItem", "StimulusSet", "load_stimulus_set", "load_passages"]


@dataclass(frozen=True)
class Question:
    """A yes/no comprehension question about a passage or one of its regions.

    ``answer`` is the correct response. ``kind`` records whether the question is
    ``"literal"`` or ``"inferential"``, and ``region`` names the annotated region
    it probes, where the design ties comprehension to a specific span rather than
    to the passage as a whole.
    """

    id: str
    text: str
    answer: bool
    kind: str = "literal"
    region: str | None = None


@dataclass(frozen=True)
class StimulusItem:
    """One reading item and its experimental annotation.

    ``passage`` holds the text, words and interest-area geometry. ``condition`` is
    the design cell, for example ``"high"`` or ``"low"`` frequency, or ``"short"``
    or ``"long"``. ``target_index`` points at the manipulated word within
    ``passage.words``. ``properties`` carries item-level detail such as the target
    frequency and length, and ``word_zipf`` maps each word index to its Zipf
    frequency where annotated. ``regions`` lists the annotated multi-word spans,
    each typed by the layer of reading it loads, and ``questions`` the
    comprehension checks tied to them. ``question_position`` is ``"after"`` by
    default; ``"before"`` shows the question text before the passage, the
    information-seeking regime of the naturalistic corpora. ``question`` and
    ``answer`` mirror the first question for backward compatibility.
    """

    passage: Passage
    condition: str | None = None
    target_index: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    word_zipf: dict[int, float] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    question_position: str = "after"
    question: str | None = None
    answer: bool | None = None


@dataclass(frozen=True)
class StimulusSet:
    """A named set of reading items with its provenance."""

    name: str
    language: str
    license: str
    source: str
    citation: list[str]
    design: str
    items: list[StimulusItem]

    def passages(self) -> list[Passage]:
        """The items' passages, in order, for feeding to a session."""
        return [item.passage for item in self.items]


def _items(data: Any) -> list[dict[str, Any]]:
    """Pull the item list from either a metadata object or a bare list."""
    if isinstance(data, dict):
        raw = data.get("items", data.get("passages"))
        if raw is None:
            raise ValueError("stimulus object must have an 'items' or 'passages' list")
    else:
        raw = data
    items: Iterable[object] = raw
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise ValueError(
                f"item {index} must be an object with 'id' and 'text' fields"
            )
        out.append(item)
    return out


def _check_id(kind: str, value: str, passage_id: str) -> str:
    """Refuse empty identifiers and ones that would corrupt marker messages."""
    if not value or re.search(r"[\s=]", value):
        raise ValueError(
            f"item {passage_id!r}: {kind} id {value!r} is empty or contains "
            "whitespace or '=', which the marker message format cannot carry"
        )
    return value


def _parse_regions(item: dict[str, Any], passage: Passage) -> list[Region]:
    """Parse and validate an item's ``regions`` list against its word count.

    Regions may overlap; where they do, gaze is attributed to the first region
    listed (see :func:`readsync.text.region_at`). Ids must be unique within the
    item.
    """
    regions: list[Region] = []
    seen: set[str] = set()
    for raw in item.get("regions", []):
        region = Region(
            id=_check_id("region", str(raw["id"]), passage.id),
            start=int(raw["start"]),
            end=int(raw["end"]),
            layer=str(raw["layer"]),
            role=str(raw.get("role", "target")),
        )
        n = len(passage.words)
        if not (0 <= region.start < region.end <= n):
            raise ValueError(
                f"item {passage.id!r}: region {region.id!r} spans words "
                f"[{region.start}, {region.end}) but the passage has {n} words"
            )
        if not region.layer:
            raise ValueError(f"item {passage.id!r}: region {region.id!r} has an empty layer")
        if region.id in seen:
            raise ValueError(f"item {passage.id!r}: duplicate region id {region.id!r}")
        seen.add(region.id)
        regions.append(region)
    return regions


def _parse_questions(
    item: dict[str, Any], passage: Passage, regions: list[Region]
) -> list[Question]:
    """Parse an item's questions, accepting the single ``question``/``answer``
    pair as a one-question list, and validate any region references."""
    raw_questions = list(item.get("questions", []))
    if not raw_questions and item.get("question") is not None:
        raw_questions = [
            {"id": f"{passage.id}-q1", "text": item["question"], "answer": item.get("answer")}
        ]
    region_ids = {region.id for region in regions}
    questions: list[Question] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_questions, start=1):
        answer = raw.get("answer")
        if not isinstance(answer, bool):
            # bool(...) would turn any non-empty value, "no" included, into
            # True, so anything but a JSON boolean is refused outright.
            raise ValueError(
                f"item {passage.id!r}: question {index} needs a boolean answer, "
                f"got {answer!r}"
            )
        kind = str(raw.get("kind", "literal"))
        if kind not in ("literal", "inferential"):
            raise ValueError(
                f"item {passage.id!r}: question {index} has kind {kind!r}; "
                "use 'literal' or 'inferential'"
            )
        question = Question(
            id=_check_id("question", str(raw.get("id", f"{passage.id}-q{index}")), passage.id),
            text=str(raw["text"]),
            answer=answer,
            kind=kind,
            region=str(raw["region"]) if raw.get("region") is not None else None,
        )
        if question.region is not None and question.region not in region_ids:
            raise ValueError(
                f"item {passage.id!r}: question {question.id!r} probes unknown "
                f"region {question.region!r}"
            )
        if question.id in seen:
            raise ValueError(f"item {passage.id!r}: duplicate question id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    return questions


def load_passages(path: str | Path) -> list[Passage]:
    """Read a stimulus file and return its passages in order.

    Accepts the full annotated format and the simpler list or ``passages`` forms.
    Raises ``FileNotFoundError`` if the file is absent and ``ValueError`` if an
    item lacks an ``id`` or ``text``, so a malformed set is caught before a session
    starts.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Passage(id=str(item["id"]), text=str(item["text"])) for item in _items(data)]


def load_stimulus_set(path: str | Path) -> StimulusSet:
    """Read an annotated stimulus file into a :class:`StimulusSet`.

    Missing metadata fields default to empty values, and missing item annotation
    leaves ``condition``, ``target_index`` and the property maps empty, so a plain
    passage list still loads as a set with unannotated items.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data if isinstance(data, dict) else {}
    items: list[StimulusItem] = []
    for item in _items(data):
        target = item.get("target") or {}
        word_zipf = {
            int(w["index"]): float(w["zipf"])
            for w in item.get("words", [])
            if "index" in w and "zipf" in w
        }
        properties = {k: v for k, v in target.items() if k != "word_index"}
        if "pair" in item:
            properties["pair"] = item["pair"]
        passage = Passage(id=str(item["id"]), text=str(item["text"]))
        _check_id("item", passage.id, passage.id)
        regions = _parse_regions(item, passage)
        questions = _parse_questions(item, passage, regions)
        position = str(item.get("question_position", "after"))
        if position not in ("before", "after"):
            raise ValueError(
                f"item {passage.id!r}: question_position must be 'before' or "
                f"'after', got {position!r}"
            )
        items.append(
            StimulusItem(
                passage=passage,
                condition=item.get("condition"),
                target_index=target.get("word_index"),
                properties=properties,
                word_zipf=word_zipf,
                regions=regions,
                questions=questions,
                question_position=position,
                question=questions[0].text if questions else None,
                answer=questions[0].answer if questions else None,
            )
        )
    citation = meta.get("citation", [])
    if isinstance(citation, str):
        citation = [citation]
    return StimulusSet(
        name=str(meta.get("name", "")),
        language=str(meta.get("language", "")),
        license=str(meta.get("license", "")),
        source=str(meta.get("source", "")),
        citation=list(citation),
        design=str(meta.get("design", "")),
        items=items,
    )
