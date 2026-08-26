# Stimuli

Reading materials for the toolkit, with real psycholinguistic annotation. The
bundled frequencies are Zipf values from the
[wordfreq](https://pypi.org/project/wordfreq/) package, so every number is real
and the files are reproducible from the build script, never hand-entered. The
bundled sets are development stand-ins: a study that registers SUBTLEX-UK as its
frequency norm rebuilds them from a locally supplied SUBTLEX-UK export
(`python stimuli/build_stimuli.py --subtlex path/to/SUBTLEX-UK.txt`), which is
not redistributed here.

## Sets

- **`controlled_freq_en.json`** holds the controlled word-frequency reading set.
  Each pair is a matched frame sentence that is identical except for one target
  word, and the two target words have the same length, so lexical frequency is the
  only manipulated property. The target is annotated as a region with layer
  `vocabulary` (the length set's targets carry layer `decoding`). Twelve pairs
  make up the 24 items. High targets are
  Zipf 4.5 and above, low targets 3.6 and below. The frequency effect on fixation
  and gaze durations is a standard word-level signature, and the size of that
  effect separates fluent, second-language and dyslexic readers, which is what
  this set is for.
- **`controlled_length_en.json`** holds the matching word-length set. Each pair
  differs only in the target word, which is short in one member and long in the
  other while the two are matched in frequency, so word length is the only
  manipulated property. The length effect is the other standard word-level
  signature and complements the frequency set for the same diagnostic purpose.
- **`naturalistic_en.json`** holds six neutral connected-prose passages for
  practice and for global reading measures, each with a yes/no comprehension
  question and its answer, which gives the offline comprehension measure that
  pairs with the online eye movements. It is not a controlled corpus.

Predictability is not normed in the controlled sets, and cloze data would be
needed before analysis.

All three sets are original and are released under CC-BY-4.0.

## Format

JSON with study metadata (`name`, `language`, `license`, `source`, `citation`,
`design`) and an `items` list. Each item has an `id` and `text`, and may carry a
`condition`, a `target` (`word_index`, `word`, `zipf`, `length`), per-word
`words` (`index`, `text`, `zipf`), multi-word `regions` and comprehension
`questions`. A region is `{"id", "start", "end", "layer", "role"}`, with `start`
and `end` word indices (`end` exclusive) and `layer` naming the sub-process the
region loads; the label is free text agreed per study, and these sets use
`decoding`, `vocabulary` and `integration`. `role` separates a `target` region
from a matched `comparison`. A question is `{"id", "text", "answer", "kind",
"region"}`, where `kind` is `literal` or `inferential` and `region` names the
region it probes; the simpler `question`/`answer` pair still loads as a
one-question list. An item-level `question_position` of `before` shows the
question ahead of the passage, reproducing the information-seeking regime used
by naturalistic reading corpora. Spans and region references are validated at load. Load
with `readsync.load_stimulus_set` for the full annotation, or
`readsync.load_passages` for the passages alone.

## Rebuilding

```bash
pip install wordfreq
python stimuli/build_stimuli.py
```

The script writes all three JSON files. Targets are checked against the frequency
thresholds and dropped with a message if they do not separate, so the kept set is
always a clean manipulation.

## Adding recognised corpora

Published eye-tracking and reading corpora load through the same `load_passages`
or `load_stimulus_set` once converted to this format. Several are not bundled here
because of their licences:

- **Natural Stories** (Futrell et al., 2021) is CC BY-NC-SA 4.0. The
  non-commercial and share-alike terms are incompatible with this MIT repository,
  so fetch it separately under that licence if you use it.
- **GECO** (Cop et al., 2017) uses a novel whose copyright status varies by
  country, so check before redistributing the text.
- **MECO** (Siegelman et al., 2022; Kuperman et al., 2023) is the most directly
  relevant multilingual reading corpus. Obtain its materials from the project and
  convert them with attribution.
