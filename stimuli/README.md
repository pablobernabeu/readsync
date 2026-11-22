# Stimuli

Reading materials for readsync, stored as JSON with word-level frequency annotation. Every Zipf value is looked up from a frequency source when the files are built, so none is entered by hand, and `build_stimuli.py` regenerates all three files. The bundled values come from the wordfreq package (Speer, 2022), which makes the sets usable for development. A study that registers SUBTLEX-UK (van Heuven et al., 2014) as its frequency norm rebuilds them from a locally supplied export, which is not redistributed here:

```bash
python stimuli/build_stimuli.py --subtlex path/to/SUBTLEX-UK.txt
```

## Sets

`controlled_freq_en.json` is a word-frequency set. The two members of each pair are the same sentence with a different target word, and the two targets have the same number of letters, so frequency is the property deliberately varied. A high-frequency target has a Zipf value of at least 4.5 and a low-frequency target at most 3.6. Each target is annotated as a one-word region with the layer `vocabulary`.

`controlled_length_en.json` is the matching word-length set. The long target in each pair has at least three letters more than the short one, and their Zipf values are no more than 0.6 apart. In every pair, the long target also has more syllables, so the set contrasts length in letters and in syllables together. Each target is annotated as a one-word region with the layer `decoding`.

The controlled sets hold constant only what they were built to hold constant. Other properties of the chosen words, such as morphological complexity and plausibility in the frame, are not controlled, and predictability is not normed. Predictability affects reading times separately from frequency (Shain, 2024), so the targets' predictability in their frames should be normed, by cloze or with a language model, before either contrast is analysed. Some target words also occur in other pairs' sentences, which gives them extra exposure within a session, and the build prints each case. In the current sets, these are 'garden' and 'tree' in the frequency set, and 'plan' and 'building' in the length set, where 'building' is also the long target of two pairs.

The frequency and length effects on fixation durations differ between readers. Both are larger in readers with weaker verbal skills (Kuperman & Van Dyke, 2011) and in dyslexic readers (Hawelka et al., 2010), and, measured within individuals, the length effect has proved more reliable across sessions than the frequency effect (Haller et al., 2026). The sets are small, so estimates for a single reader will be noisy.

`naturalistic_en.json` holds short passages of neutral connected prose for practice and for global reading measures, each with a yes/no comprehension question and its answer. No lexical property is controlled in them, and they carry no annotated regions.

The sentences, passages and questions are original and released under CC BY 4.0, as each file states. The Zipf values come from wordfreq, whose data are distributed under CC BY-SA 4.0, and each file cites it. The bundled files were built with wordfreq 3.1.1. Its changelog records no change to the data since the cited 3.0.2 release, so the values are those of that release.

## Format

A stimulus file is JSON with study metadata (`name`, `language`, `license`, `source`, `citation` and `design`) and an `items` list. Each item has an `id` and `text`, and may carry a `condition`, a `target` (`word_index`, `word`, `zipf` and `length`), per-word `words` (`index`, `text` and `zipf`), multi-word `regions` and comprehension `questions`.

A region is `{"id", "start", "end", "layer", "role"}`, with `start` and `end` as word indices and `end` exclusive. `layer` names the component of reading that the region is built to load. It is free text agreed per study: the bundled sets use `vocabulary` and `decoding`, and a study of sentence integration might add `integration`. `role` separates a `target` region from a matched `comparison` region. The bundled sets have target regions only.

A question is `{"id", "text", "answer", "kind", "region"}`, where `kind` is `literal` or `inferential` and `region` names the region it probes. A simpler `question` and `answer` pair still loads as a one-question list. Questions are asked after the passage. An item-level `question_position` of `before` also shows them before the passage, as in the information-seeking condition of the OneStop corpus (Malmaud et al., 2020; Berzak et al., 2025), which varies the regime between participants rather than between items.

Spans and region references are checked when a file is loaded. `readsync.load_stimulus_set` returns the full annotation and `readsync.load_passages` the passages alone.

## Rebuilding

```bash
pip install -e ".[stimuli]"
python stimuli/build_stimuli.py
```

The script writes all three files, records the frequency source in each file's `source` field, with the wordfreq version or the SUBTLEX-UK file name, and cites that source. Frequency pairs whose targets miss the thresholds, and length pairs that are too close in length or too far apart in frequency, are dropped with a message. A word missing from the frequency source is written with a null Zipf value, and a pair whose target is missing is dropped. The browser demo keeps a copy of `controlled_freq_en.json` in `web-demo/public/stimuli/`, and a test fails if the two differ, so copy the rebuilt file there too.

## Adding published corpora

Published reading corpora load through the same functions once converted to this format. None is bundled here.

Natural Stories (Futrell et al., 2021) is a set of ten English stories with self-paced reading times. It is distributed under CC BY-NC-SA 4.0, whose non-commercial and share-alike terms do not fit the MIT licence of the code or the CC BY 4.0 licence of the bundled sets, so fetch it from its repository under its own terms.

GECO (Cop et al., 2017) records eye movements during the reading of Agatha Christie's novel The Mysterious Affair at Styles (Cop et al., 2015). Its bilingual participants read half of the novel in their first language, Dutch, and half in English. The novel was first published in 1920 and is in the public domain in the United States, but it remains in copyright in the United Kingdom and the European Union until the end of 2046. The Dutch translation is a separate work with its own rights, so check the position in your jurisdiction before redistributing either text.

MECO (Siegelman et al., 2022; Kuperman et al., 2023) is a multilingual eye-tracking corpus of text reading, with first-language data in thirteen languages in its first release and English second-language data from readers of twelve first languages. Each text in the first-language release is followed by four yes/no comprehension questions, which this format represents directly. Its data and materials are on OSF under CC BY 4.0, for first-language reading at https://doi.org/10.17605/OSF.IO/3527A and for English as a second language at https://doi.org/10.17605/OSF.IO/Q9H43, so they can be converted and redistributed with attribution.

## References

Berzak, Y., Malmaud, J., Shubi, O., Meiri, Y., Lion, E., & Levy, R. (2025). OneStop: A 360-participant English eye tracking dataset with different reading regimes. *Scientific Data*, *12*(1), Article 1995. https://doi.org/10.1038/s41597-025-06272-2

Cop, U., Dirix, N., Drieghe, D., & Duyck, W. (2017). Presenting GECO: An eyetracking corpus of monolingual and bilingual sentence reading. *Behavior Research Methods*, *49*(2), 602–615. https://doi.org/10.3758/s13428-016-0734-0

Cop, U., Drieghe, D., & Duyck, W. (2015). Eye movement patterns in natural reading: A comparison of monolingual and bilingual reading of a novel. *PLOS ONE*, *10*(8), Article e0134008. https://doi.org/10.1371/journal.pone.0134008

Futrell, R., Gibson, E., Tily, H. J., Blank, I., Vishnevetsky, A., Piantadosi, S. T., & Fedorenko, E. (2021). The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions. *Language Resources and Evaluation*, *55*(1), 63–77. https://doi.org/10.1007/s10579-020-09503-7

Haller, P., Ding, C., Stegenwallner-Schütz, M., Reich, D. R., Koncic, I., Makowski, S., & Jäger, L. A. (2026). Replicate me if you can: Assessing measurement reliability of individual differences in reading across measurement occasions and methods. *Cognitive Science*, *50*(1), Article e70121. https://doi.org/10.1111/cogs.70121

Hawelka, S., Gagl, B., & Wimmer, H. (2010). A dual-route perspective on eye movements of dyslexic readers. *Cognition*, *115*(3), 367–379. https://doi.org/10.1016/j.cognition.2009.11.004

Kuperman, V., Siegelman, N., Schroeder, S., Acartürk, C., Alexeeva, S., Amenta, S., Bertram, R., Bonandrini, R., Brysbaert, M., Chernova, D., Da Fonseca, S. M., Dirix, N., Duyck, W., Fella, A., Frost, R., Gattei, C. A., Kalaitzi, A., Lõo, K., Marelli, M., … Usal, K. A. (2023). Text reading in English as a second language: Evidence from the Multilingual Eye-Movements Corpus. *Studies in Second Language Acquisition*, *45*(1), 3–37. https://doi.org/10.1017/S0272263121000954

Kuperman, V., & Van Dyke, J. A. (2011). Effects of individual differences in verbal skills on eye-movement patterns during sentence reading. *Journal of Memory and Language*, *65*(1), 42–73. https://doi.org/10.1016/j.jml.2011.03.002

Malmaud, J., Levy, R., & Berzak, Y. (2020). Bridging information-seeking human gaze and machine reading comprehension. In R. Fernández & T. Linzen (Eds.), *Proceedings of the 24th Conference on Computational Natural Language Learning* (pp. 142–152). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.conll-1.11

Shain, C. (2024). Word frequency and predictability dissociate in naturalistic reading. *Open Mind*, *8*, 177–201. https://doi.org/10.1162/opmi_a_00119

Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H.-D., Alexeeva, S., Amenta, S., Bertram, R., Bonandrini, R., Brysbaert, M., Chernova, D., Da Fonseca, S. M., Dirix, N., Duyck, W., Fella, A., Frost, R., Gattei, C. A., Kalaitzi, A., Kwon, N., Lõo, K., … Kuperman, V. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). *Behavior Research Methods*, *54*(6), 2843–2863. https://doi.org/10.3758/s13428-021-01772-6

Speer, R. (2022). *rspeer/wordfreq: v3.0* (Version 3.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7199437

van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. *Quarterly Journal of Experimental Psychology*, *67*(6), 1176–1190. https://doi.org/10.1080/17470218.2013.850521
