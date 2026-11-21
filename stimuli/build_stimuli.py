"""Build the readsync English stimulus sets with Zipf frequency annotations.

The script looks up every frequency, so rebuilding with the same frequency source
reproduces the JSON files.

    pip install -e ".[stimuli]"
    python stimuli/build_stimuli.py

By default, the Zipf values come from the wordfreq package, which installs from
PyPI without a licence agreement. A study that has registered SUBTLEX-UK as its
frequency norm rebuilds with the official file, which is not redistributed here:

    python stimuli/build_stimuli.py --subtlex path/to/SUBTLEX-UK.txt

The file may be comma- or tab-separated and must carry a spelling column and a
Zipf column (SUBTLEX-UK's own export does). A word missing from the frequency
source has no Zipf value: it is written as null, a pair whose target is missing
is dropped with a message, and the SUBTLEX path also prints a warning.

The script writes controlled_freq_en.json, controlled_length_en.json and
naturalistic_en.json next to itself. stimuli/README.md describes the sets, their
licence and how published corpora can be added.
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from readsync.text import tokenise  # noqa: E402

LANG = "en"

# Set by main() to either the wordfreq lookup or a SUBTLEX-UK table. A lookup
# returns None for a word the source does not contain.
_lookup: Callable[[str], float | None] | None = None
_source_note = "Zipf values from the wordfreq package."  # main() adds the version
_citation: list[str] = []  # set by main() for the chosen frequency source


def _wordfreq_lookup() -> Callable[[str], float | None]:
    from wordfreq import zipf_frequency

    def lookup(word: str) -> float | None:
        # wordfreq reports 0.0 for a word it does not know.
        value = float(zipf_frequency(word, LANG))
        return value if value > 0 else None

    return lookup


def _subtlex_lookup(path: Path) -> Callable[[str], float | None]:
    """Zipf lookup from a locally supplied SUBTLEX-UK export.

    Column names are matched case-insensitively: the spelling column is the one
    named ``spelling`` (or ``word``), and the Zipf column the first whose name
    contains ``zipf``. Missing words return None with a warning.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0]
    try:
        dialect: csv.Dialect | type[csv.Dialect] = csv.Sniffer().sniff(first, delimiters=",\t;")
    except csv.Error:
        # A header the sniffer cannot classify, a single column for example:
        # fall back on the delimiter that appears, tab first since SUBTLEX-UK
        # ships tab-separated.
        class _Fallback(csv.Dialect):
            delimiter = "\t" if "\t" in first else ","
            quotechar = '"'
            doublequote = True
            lineterminator = "\r\n"
            quoting = csv.QUOTE_MINIMAL

        dialect = _Fallback
    rows = list(csv.reader(text.splitlines(), dialect))
    header = [h.strip().lower() for h in rows[0]]
    try:
        word_col = next(i for i, h in enumerate(header) if h in ("spelling", "word"))
        zipf_col = next(i for i, h in enumerate(header) if "zipf" in h)
    except StopIteration as exc:
        raise SystemExit(
            f"{path}: could not find a spelling/word column and a Zipf column "
            f"in header {rows[0]!r}"
        ) from exc
    table = {}
    for row in rows[1:]:
        if len(row) > max(word_col, zipf_col):
            try:
                table[row[word_col].strip().lower()] = float(row[zipf_col])
            except ValueError:
                continue

    def lookup(word: str) -> float | None:
        value = table.get(word)
        if value is None:
            print(f"  warning: {word!r} not in {path.name}; no Zipf value")
        return value

    return lookup
# Zipf values use the logarithmic scale of van Heuven et al. (2014). A high target
# must reach HIGH_MIN and a low target must not exceed LOW_MAX, so the targets of
# every kept pair differ by at least HIGH_MIN - LOW_MAX. Pairs that fail either
# bound under the chosen frequency source are dropped with a message.
HIGH_MIN = 4.5  # Zipf at or above this counts as high frequency
LOW_MAX = 3.6   # Zipf at or below this counts as low frequency
# For the length set: the short and long targets must be close in frequency and
# clearly different in length.
ZIPF_MATCH = 0.6  # largest allowed Zipf gap within a length pair
MIN_LEN_DIFF = 3  # smallest allowed letter difference within a length pair

WORDFREQ_CITATION = (
    "Frequencies: Speer, R. (2022). rspeer/wordfreq: v3.0 (Version 3.0.2) "
    "[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7199437"
)
SUBTLEX_CITATION = (
    "Frequencies: SUBTLEX-UK (van Heuven, Mandera, Keuleers & Brysbaert, 2014), "
    "from a locally supplied export."
)
CITATION = [
    WORDFREQ_CITATION,
    "Zipf scale: van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. "
    "(2014). SUBTLEX-UK: A new and improved word frequency database for British "
    "English. Quarterly Journal of Experimental Psychology, 67(6), 1176-1190. "
    "https://doi.org/10.1080/17470218.2013.850521",
    "Frequency and length effects in reading: Rayner, K. (1998). Eye movements in "
    "reading and information processing: 20 years of research. Psychological "
    "Bulletin, 124(3), 372-422. https://doi.org/10.1037/0033-2909.124.3.372",
    "Frequency effect, updated review: Brysbaert, M., Mandera, P., & Keuleers, E. "
    "(2018). The word frequency effect in word processing: An updated review. "
    "Current Directions in Psychological Science, 27(1), 45-50. "
    "https://doi.org/10.1177/0963721417727521",
]

# Matched frame pairs. Each frame is identical except for the target word, and the
# high and low target words have the same length, so position, context and length
# are held constant while frequency varies. Targets sit mid-sentence, never first
# or last, to avoid sentence-boundary effects.
PAIRS = [
    {"pair": "p01",
     "hi": "The cook carefully poured the water into a large metal pan.",
     "lo": "The cook carefully poured the brine into a large metal pan."},
    {"pair": "p02",
     "hi": "We quietly watched the river from the old wooden bridge.",
     "lo": "We quietly watched the otter from the old wooden bridge."},
    {"pair": "p03",
     "hi": "The farmer slowly led the horse across the muddy field.",
     "lo": "The farmer slowly led the llama across the muddy field."},
    {"pair": "p04",
     "hi": "A tall plant stood beside the calm grey mountain lake.",
     "lo": "A tall heron stood beside the calm grey mountain lake."},
    {"pair": "p05",
     "hi": "She bought some fresh bread at the small village market.",
     "lo": "She bought some fresh swede at the small village market."},
    {"pair": "p06",
     "hi": "A small bird sat on the fence beside the open gate.",
     "lo": "A small wren sat on the fence beside the open gate."},
    {"pair": "p07",
     "hi": "They saw a tree at the far edge of the meadow.",
     "lo": "They saw a fawn at the far edge of the meadow."},
    {"pair": "p08",
     "hi": "They slowly explored the garden behind the old stone cottage.",
     "lo": "They slowly explored the cellar behind the old stone cottage."},
    {"pair": "p09",
     "hi": "The friends camped near the forest for several quiet days.",
     "lo": "The friends camped near the lagoon for several quiet days."},
    {"pair": "p10",
     "hi": "The narrow ring was made of silver and felt very cold.",
     "lo": "The narrow ring was made of cobalt and felt very cold."},
    {"pair": "p11",
     "hi": "He slowly finished the coffee before the long meeting began.",
     "lo": "He slowly finished the cognac before the long meeting began."},
    {"pair": "p12",
     "hi": "A quick rabbit darted across the narrow forest path.",
     "lo": "A quick marten darted across the narrow forest path."},
    {"pair": "p13",
     "hi": "He picked a ripe apple from the tall garden tree.",
     "lo": "He picked a ripe guava from the tall garden tree."},
    {"pair": "p14",
     "hi": "The clever monkey climbed the tall green jungle tree.",
     "lo": "The clever gibbon climbed the tall green jungle tree."},
    {"pair": "p15",
     "hi": "She planted a flower beside the low garden wall.",
     "lo": "She planted a dahlia beside the low garden wall."},
]

# Matched frame pairs for the word-length manipulation. Each frame is identical
# except for the target word, which is short in one and long in the other while
# their frequencies stay close, so word length is the property deliberately varied.
LENGTH_PAIRS = [
    {"pair": "L01",
     "short": "He wanted a good job after leaving the school.",
     "long": "He wanted a good career after leaving the school."},
    {"pair": "L02",
     "short": "They discussed his pay during the long board meeting.",
     "long": "They discussed his salary during the long board meeting."},
    {"pair": "L03",
     "short": "She left her bag beside the heavy front door.",
     "long": "She left her luggage beside the heavy front door."},
    {"pair": "L04",
     "short": "The police stopped the car near the stone bridge.",
     "long": "The police stopped the vehicle near the stone bridge."},
    {"pair": "L05",
     "short": "They decided to buy the house early last spring.",
     "long": "They decided to purchase the house early last spring."},
    {"pair": "L06",
     "short": "She wrapped the gift in bright and shiny paper.",
     "long": "She wrapped the present in bright and shiny paper."},
    {"pair": "L07",
     "short": "They planned a long trip to the rocky coast.",
     "long": "They planned a long journey to the rocky coast."},
    {"pair": "L08",
     "short": "The teacher set a hard task for the class.",
     "long": "The teacher set a hard assignment for the class."},
    {"pair": "L09",
     "short": "They agreed on a clear plan for the year.",
     "long": "They agreed on a clear strategy for the year."},
    {"pair": "L10",
     "short": "He carried the tool across the building site.",
     "long": "He carried the equipment across the building site."},
    {"pair": "L11",
     "short": "She spoke to the boss about the new plan.",
     "long": "She spoke to the manager about the new plan."},
    {"pair": "L12",
     "short": "He opened a small shop near the train station.",
     "long": "He opened a small business near the train station."},
    {"pair": "L13",
     "short": "She asked for help with the heavy wooden boxes.",
     "long": "She asked for support with the heavy wooden boxes."},
    {"pair": "L14",
     "short": "They had a long talk about the summer trip.",
     "long": "They had a long conversation about the summer trip."},
    {"pair": "L15",
     "short": "They waited outside the store before the evening show.",
     "long": "They waited outside the building before the evening show."},
    {"pair": "L16",
     "short": "She explained the aim of the new research project.",
     "long": "She explained the objective of the new research project."},
    {"pair": "L17",
     "short": "They painted the house during the warm dry summer.",
     "long": "They painted the building during the warm dry summer."},
]

# Naturalistic passages, each with a yes/no comprehension question and its answer,
# so a session can check that the reader engaged with the meaning. The questions
# give the offline comprehension measure that pairs with the online eye movements.
NATURAL = [
    {"id": "n01",
     "text": "The market opened early on Saturday. Traders arranged their fruit and "
             "bread while the first customers walked slowly between the stalls.",
     "question": "Did the market open early on Saturday?", "answer": True},
    {"id": "n02",
     "text": "A light wind moved across the lake. Two small boats drifted near the "
             "shore as the morning sun warmed the quiet water.",
     "question": "Were there three boats on the lake?", "answer": False},
    {"id": "n03",
     "text": "The library stayed open late during the exam season. Students filled "
             "every table, reading carefully and making notes for their classes.",
     "question": "Did the library close early during the exam season?", "answer": False},
    {"id": "n04",
     "text": "Rain fell steadily over the city. People hurried under the shop awnings "
             "while the buses moved slowly through the crowded evening streets.",
     "question": "Were the buses moving slowly?", "answer": True},
    {"id": "n05",
     "text": "The old farmhouse stood at the end of a long track. Sheep grazed in the "
             "fields nearby, and a thin line of smoke rose from the chimney.",
     "question": "Were sheep grazing near the farmhouse?", "answer": True},
    {"id": "n06",
     "text": "The train left the station on time. Passengers settled into their seats, "
             "watching the grey suburbs give way to open green countryside.",
     "question": "Did the train leave late?", "answer": False},
]


def zipf(token: str) -> float | None:
    """Zipf frequency of a token, with surrounding punctuation stripped.

    Returns None when the frequency source does not contain the word, so a
    missing value is never mistaken for a very low frequency.
    """
    if _lookup is None:
        raise RuntimeError("frequency source not initialised; run via main()")
    word = token.strip(string.punctuation).lower()
    value = _lookup(word) if word else None
    return round(value, 2) if value is not None else None


def word_frequencies(text: str) -> list[dict[str, object]]:
    """Per-word frequency annotation aligned with readsync tokenisation."""
    return [{"index": w.index, "text": w.text, "zipf": zipf(w.text)} for w in tokenise(text)]


def report_reuse(items: list[dict[str, object]]) -> None:
    """Print each target word that also appears in another pair's sentences.

    A reused word is seen more than once in a session, so its reading time is
    affected by repetition. The build reports it rather than dropping the pair.
    """
    def words(text: object) -> set[str]:
        return {w.text.strip(string.punctuation).lower() for w in tokenise(str(text))}

    for item in items:
        target = item["target"]
        assert isinstance(target, dict)
        word = str(target["word"]).strip(string.punctuation).lower()
        elsewhere = sorted(
            str(other["id"])
            for other in items
            if other["pair"] != item["pair"] and word in words(other["text"])
        )
        if elsewhere:
            print(f"  note: target {word!r} of {item['id']} also appears in {', '.join(elsewhere)}")


def differing_index(hi: str, lo: str) -> int:
    """Index of the single word that differs between two matched frames."""
    th, tl = tokenise(hi), tokenise(lo)
    if len(th) != len(tl):
        raise ValueError(f"frames have different word counts:\n  {hi}\n  {lo}")
    diffs = [i for i, (a, b) in enumerate(zip(th, tl, strict=True)) if a.text != b.text]
    if len(diffs) != 1:
        raise ValueError(f"frames must differ in exactly one word:\n  {hi}\n  {lo}")
    return diffs[0]


def build_controlled() -> dict[str, object]:
    items: list[dict[str, object]] = []
    for pair in PAIRS:
        idx = differing_index(pair["hi"], pair["lo"])
        high_word = tokenise(pair["hi"])[idx].text
        low_word = tokenise(pair["lo"])[idx].text
        if len(high_word) != len(low_word):
            print(f"  drop {pair['pair']}: lengths differ ({high_word}/{low_word})")
            continue
        zh, zl = zipf(high_word), zipf(low_word)
        if zh is None or zl is None:
            print(f"  drop {pair['pair']}: no frequency for {high_word}={zh} {low_word}={zl}")
            continue
        if zh < HIGH_MIN or zl > LOW_MAX:
            print(f"  drop {pair['pair']}: {high_word}={zh} {low_word}={zl} miss thresholds")
            continue
        for cond, text, word, z in (
            ("high", pair["hi"], high_word, zh),
            ("low", pair["lo"], low_word, zl),
        ):
            items.append({
                "id": f"{pair['pair']}{cond[0]}",
                "text": text,
                "condition": cond,
                "pair": pair["pair"],
                "target": {"word_index": idx, "word": word, "zipf": z, "length": len(word)},
                "regions": [{"id": "target", "start": idx, "end": idx + 1,
                             "layer": "vocabulary"}],
                "words": word_frequencies(text),
            })
    print(f"  kept {len(items) // 2} pairs ({len(items)} items)")
    report_reuse(items)
    return {
        "name": "Controlled word-frequency reading set (English)",
        "language": LANG,
        "license": "CC-BY-4.0",
        "source": f"Constructed for readsync. Target and per-word frequencies are {_source_note}",
        "citation": _citation,
        "design": "Matched frame pairs. Each pair is one sentence that differs only in "
                  "the target word, and the high and low targets have the same length, "
                  "so frequency is the property deliberately varied. Other lexical "
                  "properties of the chosen words covary with frequency, and "
                  "predictability is not normed, so cloze or language-model norms are "
                  "needed before analysis.",
        "items": items,
    }


def build_length() -> dict[str, object]:
    items: list[dict[str, object]] = []
    for pair in LENGTH_PAIRS:
        idx = differing_index(pair["short"], pair["long"])
        short_word = tokenise(pair["short"])[idx].text
        long_word = tokenise(pair["long"])[idx].text
        if len(long_word) - len(short_word) < MIN_LEN_DIFF:
            print(f"  drop {pair['pair']}: length gap < {MIN_LEN_DIFF} "
                  f"({short_word}/{long_word})")
            continue
        zs, zl = zipf(short_word), zipf(long_word)
        if zs is None or zl is None:
            print(f"  drop {pair['pair']}: no frequency for {short_word}={zs} {long_word}={zl}")
            continue
        if abs(zs - zl) > ZIPF_MATCH:
            print(f"  drop {pair['pair']}: zipf gap {abs(zs - zl):.2f} > {ZIPF_MATCH} "
                  f"({short_word}={zs}/{long_word}={zl})")
            continue
        for cond, text, word, z in (
            ("short", pair["short"], short_word, zs),
            ("long", pair["long"], long_word, zl),
        ):
            items.append({
                "id": f"{pair['pair']}{cond[0]}",
                "text": text,
                "condition": cond,
                "pair": pair["pair"],
                "target": {"word_index": idx, "word": word, "zipf": z, "length": len(word)},
                "regions": [{"id": "target", "start": idx, "end": idx + 1,
                             "layer": "decoding"}],
                "words": word_frequencies(text),
            })
    print(f"  kept {len(items) // 2} pairs ({len(items)} items)")
    report_reuse(items)
    return {
        "name": "Controlled word-length reading set (English)",
        "language": LANG,
        "license": "CC-BY-4.0",
        "source": f"Constructed for readsync. Target and per-word frequencies are {_source_note}",
        "citation": _citation,
        "design": "Matched frame pairs. Each pair is one sentence that differs only in "
                  "the target word, which is short in one member and long in the other, "
                  f"with Zipf values no more than {ZIPF_MATCH} apart, so word length is the "
                  "property deliberately varied. Syllable count and morphological "
                  "complexity covary with length, and predictability is not normed.",
        "items": items,
    }


def build_naturalistic() -> dict[str, object]:
    items = [
        {
            "id": p["id"],
            "text": p["text"],
            "question": p["question"],
            "answer": p["answer"],
            "words": word_frequencies(p["text"]),
        }
        for p in NATURAL
    ]
    return {
        "name": "Neutral naturalistic passages (English)",
        "language": LANG,
        "license": "CC-BY-4.0",
        "source": f"Constructed for readsync. Per-word frequencies are {_source_note}",
        "citation": _citation,
        "design": "Short, neutral passages of connected prose for practice and for "
                  "global reading measures, each with a yes/no comprehension question. "
                  "No lexical property is controlled.",
        "items": items,
    }


def _write(name: str, data: dict[str, object]) -> None:
    (HERE / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    global _lookup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subtlex",
        type=Path,
        default=None,
        help="path to a local SUBTLEX-UK export to use as the frequency source "
        "(default: the wordfreq package)",
    )
    args = parser.parse_args()
    global _source_note
    _lookup = _subtlex_lookup(args.subtlex) if args.subtlex else _wordfreq_lookup()
    if args.subtlex:
        _source_note = f"Zipf values from a local SUBTLEX-UK export ({args.subtlex.name})."
    else:
        from importlib.metadata import version

        _source_note = f"Zipf values from the wordfreq package (version {version('wordfreq')})."
    global _citation
    _citation = [SUBTLEX_CITATION if args.subtlex else WORDFREQ_CITATION, *CITATION[1:]]
    source = f"SUBTLEX-UK ({args.subtlex.name})" if args.subtlex else "wordfreq"
    print(f"frequency source: {source}")
    print("frequency set:")
    _write("controlled_freq_en.json", build_controlled())
    print("length set:")
    _write("controlled_length_en.json", build_length())
    naturalistic = build_naturalistic()
    _write("naturalistic_en.json", naturalistic)
    print(f"naturalistic set: {len(naturalistic['items'])} passages with questions")
    print("wrote controlled_freq_en.json, controlled_length_en.json and naturalistic_en.json")


if __name__ == "__main__":
    main()
