# readsync

[![CI](https://github.com/pablobernabeu/readsync/actions/workflows/ci.yml/badge.svg)](https://github.com/pablobernabeu/readsync/actions/workflows/ci.yml)

readsync is a Python toolkit for reading experiments that record eye movements and, optionally, EEG. It presents annotated reading materials and turns gaze into word and region events as each passage is read. The same events reach the eye-tracker's data file and the EEG recording as markers. Comprehension questions tied to the materials are asked and scored, and the session is kept in an encrypted, tamper-evident log.

## What it provides

Reading materials are JSON files. A passage can carry multi-word regions, each labelled with the component of reading it is built to load, such as decoding or vocabulary. A passage can also carry yes/no comprehension questions tied to those regions. Word interest areas come from a fixed-width layout, so each word's box on screen follows from its position in the text. The bundled sets and the format are described in [stimuli/README.md](stimuli/README.md).

`ReadingSession` runs each passage as one tracker trial. Within the trial, it performs the drift check, records the passage while emitting word and region events from the gaze, asks the questions and closes the trial. Passages are presented with PsychoPy (Peirce et al., 2019), and the EyeLink is controlled through SR Research's pylink. Trackers, presenters and marker sinks are small protocols, so another device can be added without changing the session. By default, the session runs under a guard that blocks outbound network connections from Python code.

After a session, the log can be exported as a CSV with one row per event, and the word interest areas as a CSV with one row per word and its annotated region. The EyeLink backend also opens and closes a trial in the data file for each passage and writes the passage's word boxes as an EyeLink Data Viewer interest-area set, so that Data Viewer can segment the recording by passage and load the same boxes as the session. A quality report gives, for each passage, the share of polled gaze samples that were invalid and, given the expected sampling rate, the share of expected samples that never arrived. It also lists the calibration and drift checks and the comprehension score. Fixation-level analysis is left to tools that read the EyeLink data file, such as Data Viewer or, after conversion to ASC, [Eyekit](https://github.com/jwcarr/eyekit).

The webcam module estimates gaze from a laptop camera with MediaPipe's FaceLandmarker and a calibration model that accounts for head pose, and it marks gaze invalid during blinks. It implements the same tracker protocol, so a session can run on a webcam, as in [`examples/webcam_session.py`](examples/webcam_session.py). [`examples/webcam_live_demo.py`](examples/webcam_live_demo.py) calibrates the head-aware model on a laptop and shows live gaze over passages without PsychoPy, and a browser version runs at https://interlexis.web.app, with the code in [`web-demo/`](web-demo).

Webcam gaze is coarser and noisier than infrared tracking. In a comparison with an EyeLink 1000, a webcam system erred by 1.45° on average against the EyeLink's 0.91° (Kaduk et al., 2024). In a laboratory reading study set in Consolas, one degree spanned about three letters (Rayner et al., 2010), so such an error covers several letters. Word-frequency effects have nonetheless been recovered from webcam recordings of reading in very large print (Serrano-Carot et al., 2026). The module therefore serves method development and coarse measures, and its own accuracy has not yet been measured against a reference tracker.

## Status

readsync is alpha software. The core, which covers stimulus loading, interest areas, the session, the marker protocol, the encrypted log, the exports and the quality report, runs without hardware and is covered by the test suite. The PsychoPy presenter draws passages and collects answers on a screen. Its question screen, window-size check, layout check and coordinate conversion are tested against a stand-in for PsychoPy, but the drawing of passages is not. The EyeLink backend follows SR Research's documented call sequence, and its connection, trial, recording and closing calls run in pylink's dummy mode. Calibration, the drift check, recording and Data Viewer's reading of the trial and interest-area messages have not yet been tested with an EyeLink, and the Lab Streaming Layer sink has not been tested with pylsl. The timing of markers relative to the display has not been measured with a photodiode or a hardware trigger, so a study that depends on marker timing should measure it on its own set-up.

## Install

```bash
pip install -e ".[dev]"                   # core, tests and linters
pip install -e ".[present,eyelink,lsl]"   # PsychoPy, SR Research's pylink and Lab Streaming Layer
pip install -e ".[webcam]"                # MediaPipe, for webcam gaze
```

The core depends only on `cryptography`. SR Research publishes pylink under a proprietary licence that permits educational and non-commercial use.

## Quick start

This session runs with no hardware: `NullTracker` supplies synthetic gaze that sweeps across the middle of the screen, where the layout places the text.

```python
from readsync import (
    EventLog, FixedWidthLayout, HeadlessPresenter, NullTracker, Passage, ReadingSession,
    interest_areas_to_csv, log_quality, log_to_csv, new_data_key, pseudonymise,
    quality_to_json,
)

data_key = new_data_key()   # keep both keys apart from the data
id_key = new_data_key()
participant = pseudonymise("alice@example.org", key=id_key)

layout = FixedWidthLayout(y0=520)
log = EventLog(f"data/{participant}-s1.log", key=data_key)   # one file per session
session = ReadingSession(
    participant=participant,
    tracker=NullTracker(),           # EyeLinkTracker() in the laboratory
    presenter=HeadlessPresenter(),   # PsychoPyPresenter() on a screen
    log=log,
    layout=layout,
)
passages = [Passage(id="p1", text="The cat sat on the mat.")]
session.run(passages)

log_to_csv(log, f"export/{participant}-s1.csv")   # verifies the log, then exports it
interest_areas_to_csv(passages, "export/interest_areas.csv", layout)
report = log_quality(log, expected_rate_hz=60)   # the session polls once per display frame
quality_to_json(report, f"export/{participant}-s1.quality.json")
```

For annotated materials, load a stimulus file with `load_stimulus_set` and pass its `items` to `run_items` of a new session with its own log. Regions then produce enter and exit events, and each item's questions are asked and scored after its passage. With an EyeLink and an EEG recording, run the presenter full screen at the display's resolution, give the EyeLink backend the same screen size and send the markers both to the EyeLink data file and to Lab Streaming Layer. On a 1920 by 1080 display, the defaults already agree:

```python
from readsync import (
    EyeLinkMarkerSink, EyeLinkTracker, LSLMarkerSink, MultiMarkerSink, PsychoPyPresenter,
)

tracker = EyeLinkTracker(edf_name="s01", receive_to="data/s01.edf")
eeg = LSLMarkerSink()
eeg.wait_for_recorder(timeout=60)   # select the stream in the EEG recorder first
session = ReadingSession(
    participant=participant,
    tracker=tracker,
    presenter=PsychoPyPresenter(),
    log=EventLog(f"data/{participant}-s2.log", key=data_key),
    layout=layout,
    marker_sink=MultiMarkerSink(EyeLinkMarkerSink(tracker), eeg),
)
```

[`examples/run_session.py`](examples/run_session.py) runs a session on a screen with PsychoPy and the synthetic tracker.

## Documentation

[ARCHITECTURE.md](ARCHITECTURE.md) describes the modules, the extension points and every event in the session log. [SECURITY.md](SECURITY.md) sets out what the log protects and where that protection ends. [stimuli/README.md](stimuli/README.md) covers the stimulus format and the bundled sets, and [web-demo/README.md](web-demo/README.md) the browser demo. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to run the checks.

## Citing and licence

If you use readsync, please cite the version you used. [CITATION.cff](CITATION.cff) gives the details. The code is released under the MIT Licence (see [LICENSE](LICENSE)). The bundled stimulus sets are released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), as each file states, because their Zipf values come from wordfreq's data. Their sentences, passages and questions are also released on their own under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see [stimuli/README.md](stimuli/README.md)). When the browser demo is built, it fetches MediaPipe's runtime and model, which are licensed under Apache 2.0.

## References

Kaduk, T., Goeke, C., Finger, H., & König, P. (2024). Webcam eye tracking close to laboratory standards: Comparing a new webcam-based system and the EyeLink 1000. *Behavior Research Methods*, *56*(5), 5002–5022. https://doi.org/10.3758/s13428-023-02237-8

Peirce, J., Gray, J. R., Simpson, S., MacAskill, M., Höchenberger, R., Sogo, H., Kastman, E., & Lindeløv, J. K. (2019). PsychoPy2: Experiments in behavior made easy. *Behavior Research Methods*, *51*(1), 195–203. https://doi.org/10.3758/s13428-018-01193-y

Rayner, K., Slattery, T. J., & Bélanger, N. N. (2010). Eye movements, the perceptual span, and reading speed. *Psychonomic Bulletin & Review*, *17*(6), 834–839. https://doi.org/10.3758/PBR.17.6.834

Serrano-Carot, M., Angele, B., Xu, H., & Vasilev, M. R. (2026). Webcams can be used to study eye movements during reading. *Behavior Research Methods*, *58*(12), Article 331. https://doi.org/10.3758/s13428-026-03199-3
