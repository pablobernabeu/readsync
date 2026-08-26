# readsync

[![CI](https://github.com/pablobernabeu/readsync/actions/workflows/ci.yml/badge.svg)](https://github.com/pablobernabeu/readsync/actions/workflows/ci.yml)

An offline-first, secure toolkit for **synchronised reading experiments** with eye-tracking and, optionally, EEG.

`readsync` presents reading materials, records gaze from a research-grade eye-tracker, marks events so they can be aligned with an EEG record, and stores everything locally under encryption. It is built for the specific demands of reading research: word and region interest areas, comprehension questions tied to regions, precise event synchronisation, operation with no internet connection during a session, a tamper-evident data record and a per-session quality report.

## Scope

`readsync` extends the established open-source stack.

- Stimulus presentation and device control are provided by [PsychoPy](https://doi.org/10.3758/s13428-018-01193-y) and by toolboxes such as [Titta](https://doi.org/10.3758/s13428-020-01358-8). `readsync` drives them.
- Analysis of reading eye-movement data is done with tested tools such as Eyekit, popEye and [PupEyes](https://doi.org/10.3758/s13428-025-02830-z). `readsync` exports to the open formats they read.

It packages the parts that reading studies repeatedly rebuild and that no existing suite offers together. Reading materials carry word and multi-word region annotation, each region typed by the layer of reading it loads, with comprehension questions tied to those regions. Stimulus, eye-tracker and EEG share one marker stream carrying the same event detail as the encrypted log. Sessions run fully offline, leave an auditable record, and end with a data-quality report. Building only this layer follows the field's own advice on avoiding redundant tools ([Niehorster et al., 2025](https://doi.org/10.3758/s13428-024-02529-7)).

This repository is the software deliverable of the Interlexis research programme on reading in multilingual adults, which is also where the browser demo's name comes from.

## Status

Alpha. The data-handling core (security, the tamper-evident log, word and region interest areas, comprehension questions with scored responses, real-time session orchestration and the per-session quality report) is implemented and tested and runs with no hardware. The PsychoPy presenter is implemented and runs a full session on a real screen, drawing each word at its interest-area position, advancing on a keypress and collecting yes/no answers; see [`examples/run_session.py`](examples/run_session.py). The EyeLink backend sketches the SR Research integration, following the documented pylink call sequence, and its pure parts are tested. Its device calls will be completed and validated against the tracker once the laboratory is installed. A session runs end to end on screen today with the synthetic `NullTracker` in its place. The webcam strand provides a `WebcamTracker` on the same interface, a MediaPipe FaceLandmarker estimator, a per-participant calibration fit and an angular-error benchmark, and it runs end to end on a laptop webcam through [`examples/webcam_live_demo.py`](examples/webcam_live_demo.py). Webcam gaze stays a methods strand, too coarse to replace the infrared tracker for word-level reading.

## Install

```bash
pip install -e ".[dev]"          # core plus test and lint tools
pip install -e ".[present,lsl]"  # add PsychoPy and Lab Streaming Layer on the lab machine
pip install -e ".[webcam]"       # add OpenCV and MediaPipe for the webcam strand
```

Core install needs only `cryptography`, so the package runs and is testable on any machine.

## Try the webcam path on a laptop

To see gaze tracking working end to end with a built-in webcam, no eye-tracker and no PsychoPy:

```bash
pip install -e ".[webcam]"
# fetch the FaceLandmarker model once (about 3.7 MB); see models/README.md
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python examples/webcam_live_demo.py
```

Calibration runs at a couple of head positions (set with `--poses`), which is what lets the model compensate for head movement; at each one, look at each dot until it moves on. The calibration is then checked by cross-validation and offered for redo if it is poor. While reading, a green dot follows your gaze and the word under it is boxed; the dot turns amber and asks you to recalibrate once your head moves beyond the range you calibrated, and blinks hold the last gaze instead of throwing the dot. Press SPACE for the next passage, C to recalibrate without losing your place, and ESC to stop. The session records to an encrypted log and exports a CSV. Even with head-pose compensation, webcam gaze is coarse and resolves a region of a line, never a single letter, so this path serves the methods strand and cannot replace the infrared tracker.

## Browser demo

The same head-pose-aware pipeline runs fully in the browser, deployed at https://interlexis.web.app, with the gaze maths ported to JavaScript and checked against the desktop version. It is a static, client-side site for Firebase Hosting: camera frames are processed in the browser and no video, gaze or calibration data is ever uploaded. See [`web-demo/`](web-demo) for the code, the security model and deployment.

## Quick start

```python
from readsync import (
    EventLog, HeadlessPresenter, NullTracker, Passage, ReadingSession,
    new_data_key, pseudonymise,
)

key = new_data_key()                       # store this separately from the data
participant = pseudonymise("alice@uni.ox.ac.uk", key=key)

log = EventLog(f"data/{participant}.log", key=key)
session = ReadingSession(
    participant=participant,
    tracker=NullTracker(),                 # swap for EyeLinkTracker() in the lab
    presenter=HeadlessPresenter(),         # swap for the PsychoPy presenter in the lab
    log=log,
)
session.run([Passage(id="p1", text="The cat sat on the mat.")])

from readsync import log_quality, log_to_csv, quality_to_json
log_to_csv(log, f"export/{participant}.csv")   # verifies integrity, then exports
quality_to_json(log_quality(log), f"export/{participant}.quality.json")
```

The session runs inside an offline guard by default, so no data leave the machine while a participant is recorded. For annotated materials, load a stimulus set and call `session.run_items(stim.items)` instead: regions typed by reading layer emit enter and exit markers, and each item's comprehension questions are asked and scored, after the passage or, under the information-seeking regime, shown before it.

## Stimuli

Reading materials live in JSON files under [`stimuli/`](stimuli), with real psycholinguistic annotation, never placeholder text. `controlled_freq_en.json` is a controlled word-frequency set: matched frame sentences that differ only in one length-matched target word, so lexical frequency is the only thing manipulated, with the target annotated as a layer-typed region. `controlled_length_en.json` is its word-length twin, short against long targets at matched frequency. `naturalistic_en.json` holds neutral connected-prose passages for practice and global measures, each with a comprehension question. The bundled Zipf values come from [wordfreq](https://pypi.org/project/wordfreq/) as a development stand-in; a study that registers SUBTLEX-UK as its norm rebuilds the sets from a locally supplied export (see [stimuli/README.md](stimuli/README.md)). Load the passages with `load_passages`, or the full annotation with `load_stimulus_set`:

```python
from readsync import load_stimulus_set
stim = load_stimulus_set("stimuli/controlled_freq_en.json")
item = stim.items[0]
print(item.condition, item.passage.words[item.target_index].text, item.properties["zipf"])
```

The sets are reproducible from [`stimuli/build_stimuli.py`](stimuli/build_stimuli.py). See [stimuli/README.md](stimuli/README.md) for the format and for adding recognised open corpora.

## Security

Participant data are pseudonymised at source, encrypted at rest, and held in an append-only, tamper-evident log. Sessions run offline. See [SECURITY.md](SECURITY.md) for the threat model and the guarantees.

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map, the extension points for new trackers and presenters, and how `readsync` interoperates with the analysis tools.

## Licence

The code is MIT; see [LICENSE](LICENSE). The bundled stimulus sets under [`stimuli/`](stimuli) are released separately under CC-BY-4.0, stated inside the files themselves, and the vendored MediaPipe runtime and model in the browser demo are Apache-2.0.
