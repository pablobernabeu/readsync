# Architecture

`readsync` is a small set of modules behind narrow interfaces, so that the parts that need research hardware are isolated from the parts that do not, and so the whole thing runs and is tested with no device.

## Module map

| Module | Responsibility |
|---|---|
| `security` | Pseudonymisation, encryption at rest, and the offline `NetworkGuard`. |
| `storage` | `EventLog`: an encrypted, append-only, hash-chained record. |
| `text` | Reading stimuli, tokenisation, fixed-width word interest areas, and layer-typed multi-word `Region` spans. |
| `trackers` | The `Tracker` protocol, a hardware-free `NullTracker`, and an `EyeLinkTracker` sketch over the SR Research SDK, including drift correction. |
| `markers` | The `MarkerSink` protocol for synchronising stimulus, gaze and EEG, with sinks for an in-memory list, Lab Streaming Layer and the EyeLink data file. Every sink carries the marker's full `key=value` detail, so the hardware streams match the session log. |
| `session` | The `Presenter` protocol, a hardware-free `HeadlessPresenter`, and `ReadingSession`, which orchestrates presentation, polling, marking, questions and logging in real time, offline. |
| `presenters` | `PsychoPyPresenter`: presents passages on a real screen, drawing each word at its interest area, and collects yes/no question responses. |
| `export` | Writes the verified log to a tidy CSV for the analysis tools. |
| `quality` | The per-session data-quality report: sample counts and loss, invalid proportion, gaps, calibration and drift checks, and the comprehension tally. |
| `stimuli` | Loads annotated reading materials (conditions, target words, per-word frequencies, layer-typed regions, comprehension questions) from a JSON stimulus file, validating the annotation at load. |
| `webcam` | The webcam strand: head-pose-aware gaze (roll-invariant feature, FaceLandmarker head pose, a regularised head-aware model with drift detection and cross-validated calibration), a `WebcamTracker` on the `Tracker` protocol, and an angular-error benchmark. |

## Design choices

**Dependency-light core.** The data-handling core depends only on `cryptography`. PsychoPy, the EyeLink SDK and Lab Streaming Layer are optional extras and are imported lazily, so importing `readsync` never pulls in a display or a device driver. This is what makes the package installable and testable anywhere, and keeps the security-critical surface small.

**Protocol interfaces.** `Tracker`, `MarkerSink` and `Presenter` are `typing.Protocol` interfaces. The session depends on the interface, so a new backend (for example a Tobii tracker through Titta, or a PsychoPy presenter) is added by implementing the protocol, with no change to the session logic. Optional capabilities, question presentation on a presenter and drift correction on a tracker, are discovered at run time, so backends without them keep working.

**Append-only record.** A research record should be auditable, so the log is never rewritten in place. Integrity is enforced by a hash chain, so it does not depend on the file system.

## Extension points

- **Presentation.** `PsychoPyPresenter` in `presenters.py` draws each word at its interest area, so word boundaries on screen match the boxes used in analysis. The session drives it in real time through the `start_passage` / `tick` / `end_passage` protocol, where `tick` renders a frame, returns the session-clock time and reports whether the reader has advanced. A different display library is supported by implementing the same protocol. `examples/run_session.py` runs a full on-screen session with the synthetic `NullTracker` standing in for the eye-tracker.
- **A real tracker.** `EyeLinkTracker` sketches the SR Research `pylink` integration. The configuration, the calibration handshake, the link sampling and the data-file transfer follow the documented call sequence, and the pure parts (the data-file name check and the sample-to-`GazeSample` conversion) are tested. The lab build installs the SDK, supplies a calibration graphics environment and validates against the device. A Tobii backend through Titta, or any other tracker, is added by implementing the same protocol; the webcam backend in `readsync.webcam` already does so.
- **EEG.** Send markers through `LSLMarkerSink` to an EEG amplifier on the same machine. LSL keeps working during an offline session because `pylsl` binds the native `liblsl` library, whose sockets sit outside the Python socket layer the offline guard patches; the internet-facing Python paths stay blocked. `EyeLinkMarkerSink` also writes each marker into the EyeLink data file, with the same `key=value` detail as the session log, so the eye record carries the same events for alignment. The analysis of co-registered eye-tracking and EEG is done with established tools.

## The event record

Every session writes one encrypted log whose events share the presenter's session clock (`t`, in seconds). The event types are:

| Event | Fields beyond `type` and `t` | Meaning |
|---|---|---|
| `session_start`, `session_end` | `participant` | Session boundaries; a completed session always ends with `session_end`. |
| `calibration` | `detail` | The tracker's calibration or validation outcome, logged at the start and again after any mid-session recalibration. |
| `drift_check` | `passage`, `error` | The drift check before a passage; `error` in degrees, or null when unavailable. |
| `passage_onset`, `passage_offset` | `passage` | Passage boundaries. |
| `gaze` | `tracker_t`, `x`, `y`, `valid` | One polled sample; `tracker_t` is the tracker's own timestamp for cross-checking against its data file. |
| `word_enter` | `passage`, `word` | Gaze entered a new word's interest area. |
| `region_enter`, `region_exit` | `passage`, `region`, `layer` | Gaze crossed an annotated region boundary; blinks and gaze between words hold the current region. |
| `prompt` | `passage`, `question` | A question shown before its passage, under the information-seeking regime. |
| `response` | `onset`, `passage`, `question`, `kind`, `region`, `response`, `correct` | A scored answer to a comprehension question. |

The exported CSV carries the same fields as columns, common keys first.

## Interoperability

`readsync` records and synchronises. It exports to the open formats that tested tools already read (Eyekit, popEye, PupEyes for eye movements; EYE-EEG and related tools for co-registration). Building another analysis package would duplicate validated software, so the project does not.

## The webcam strand

`readsync.webcam` carries the methods contribution. `WebcamTracker` implements the same `Tracker` protocol as the infrared backend, so a session runs on a webcam with only the tracker swapped. It composes a camera (`FrameSource`) and a gaze estimator (`GazeEstimator`) behind narrow protocols, so an existing model plugs in without being reimplemented. `MediaPipeIrisEstimator` is the worked adapter over the MediaPipe FaceLandmarker model, which is fetched once into `models/`.

Accuracy and reliability come from `readsync.webcam.gaze`. The iris feature is measured in the eye's own axes, so head roll does not corrupt it, and `MediaPipeIrisEstimator.signals` also reads head yaw and pitch from the FaceLandmarker transformation matrix together with an eye-openness ratio. `HeadAwareGazeModel` fits a regularised map from those signals to the screen, so it compensates for head orientation. Calibrating at more than one head position gives the yaw and pitch terms real variation to learn from, and the model reports how far the head has moved beyond the calibrated pose range, so gaze can be flagged once it can no longer be trusted. The calibration is checked by leave-one-target-out cross-validation across poses, so a poor one is caught before any reading. Simpler `AffineCalibration` and `PolynomialCalibration` maps remain for the plain feature-to-screen path, and `readsync.webcam.benchmark` reports accuracy in degrees of visual angle against the infrared tracker.

`examples/webcam_live_demo.py` runs the whole path on a laptop with a webcam and no PsychoPy. It calibrates against a grid of targets, shows passages with a live gaze dot and the word under gaze, and records to the encrypted log. The strand is kept separate from the lab toolkit and is not a dependency of it. The evidence remains that webcam gaze is too coarse for word-level reading measures, so this path is for the benchmark and for coarse uses, not a substitute for the research-grade device.
