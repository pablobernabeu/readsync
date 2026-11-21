# Architecture

readsync is a small set of modules behind narrow interfaces. Code that needs research hardware or a display is kept apart from code that does not, so the package installs and its tests run on any machine.

## Modules

| Module | Responsibility |
|---|---|
| `security` | Pseudonymisation, encryption of log records and the offline `NetworkGuard`. |
| `storage` | `EventLog`, an encrypted, append-only, hash-chained record. |
| `text` | Passages, tokenisation, fixed-width word interest areas and multi-word `Region` spans labelled by reading layer. |
| `stimuli` | Loads annotated reading materials from JSON and checks the annotation. |
| `trackers` | The `Tracker` protocol, the synthetic `NullTracker` and the `EyeLinkTracker` backend over SR Research's pylink. |
| `markers` | The `MarkerSink` protocol and sinks for memory, Lab Streaming Layer, the EyeLink data file and several sinks at once. |
| `session` | The `Presenter` protocol, the display-free `HeadlessPresenter` and `ReadingSession`, which runs passages as tracker trials and logs every event. |
| `presenters` | `PsychoPyPresenter`, which draws each word centred on its interest area and collects yes/no answers. |
| `dataviewer` | EyeLink Data Viewer trial messages and interest-area sets. |
| `export` | CSV export of the verified log and of the word interest areas. |
| `quality` | The per-session quality report. |
| `webcam` | Webcam gaze: features from MediaPipe's FaceLandmarker, calibration models, `WebcamTracker` and angular-error helpers. |

## Design choices

The core depends only on `cryptography`. PsychoPy, pylink, Lab Streaming Layer and MediaPipe are optional extras imported only when used, so importing readsync never loads a display or a device driver, and the code that handles participant data stays small.

`Tracker`, `MarkerSink` and `Presenter` are `typing.Protocol` interfaces, and the session depends only on them. A new backend is added by implementing the protocol, with no change to the session. Optional capabilities are looked up at run time: a presenter may offer `now`, `ask` and `show_prompt`, and a tracker may offer `drift_correct`, `begin_trial`, `end_trial` and `last_calibration_message`. Backends without them run without those steps.

The log is append-only so that the record can be audited, and changes to it are detected through a hash chain rather than through file-system permissions (see [SECURITY.md](SECURITY.md)). The session syncs the log to disk after every passage, and `recover_events` returns the records before any damage left by a crash.

## Trials and timing

Each passage is one tracker trial, in the order that SR Research's trial structure expects. The session opens the trial, shows the questions first if the item uses the information-seeking regime, runs the drift check while the tracker is idle, starts recording, presents the passage, stops recording, asks the questions and closes the trial.

While a passage is shown, the session repeatedly calls the presenter's `tick`, which draws a frame, flips it and returns the session time. The first frame is stamped as the passage onset. On each frame, the session polls the tracker for its newest sample, logs it, finds the word under it and emits word and region events when the gaze moves to a new word or region. Events are therefore stamped with the time of the frame on which the sample was polled, and the gaze in the log is sampled once per frame. The tracker's data file keeps the full-rate record. The word and region events mark the first polled sample to land on a word, which may fall during a saccade, so they serve gaze-contingent control and quick summaries. Fixation-based measures, and analyses locked to fixation onsets such as fixation-related potentials, should take their fixations from the tracker's data file. Markers are sent after the flip has returned, and their latency relative to the physical change on the screen has not been measured.

## Extension points

`PsychoPyPresenter` draws each word centred on its interest area. Interest areas, gaze and the presenter's window share one pixel frame, so with an eye-tracker the window must be full screen at the display's resolution, and the tracker's `screen_size` must be the same size. The presenter refuses a window that opens at another size, and the session refuses a tracker whose frame differs from the window. The words fill their boxes only when the font's character width at the drawn letter height matches the layout's `char_width`, which `check_layout` measures, so run it once per display and adjust `letter_height` or the layout. Comprehension response times are taken from the keyboard's own timestamps, with its clock reset at the flip that shows the question. A presenter for another display library implements the same protocol: `start_passage`, `tick`, `end_passage` and `close`.

`EyeLinkTracker` drives an EyeLink through pylink. Its connection, trial, recording and closing calls run in pylink's dummy mode, and calibration, the drift check and recording still have to be validated with a tracker. `eye` chooses which eye drives the online events in a binocular recording, and the data file keeps every recorded eye. To calibrate in the experiment's own PsychoPy window, pass `graphics_factory=lambda link: EyeLinkCoreGraphicsPsychoPy(link, presenter.window)`, with the graphics class from SR Research's PsychoPy examples, which are licensed to SR Research's customers and so are not included here. Another tracker is added by implementing `Tracker`, and a Tobii backend could build on Titta (Niehorster et al., 2020). `WebcamTracker` already implements the protocol.

Markers reach an EEG recording through `LSLMarkerSink`, which publishes a Lab Streaming Layer marker stream for a recorder to save alongside the EEG. The stream opens when the sink is created, and `wait_for_recorder` confirms that a recorder has connected, because markers sent before then are not delivered. `MultiMarkerSink` sends each marker to the EyeLink data file and to that stream in turn. Lab Streaming Layer keeps working under the offline guard because pylsl binds the native liblsl library, whose sockets bypass the Python socket layer the guard patches. Co-registration toolboxes such as [EYE-EEG](https://github.com/olafdimigen/eye-eeg) align the eye-tracking and EEG records on shared integer trigger codes, which readsync's string markers do not carry yet.

## The event record

Every session writes one encrypted log, and a session refuses a log that already holds records. The events, in the order a session produces them, are listed below. `t` is the presenter's session clock in seconds, and it is `null` when the presenter offers no `now`. Events marked as markers are also sent to the marker sink as `label key=value ...`, with their identifiers.

| Event | Fields beyond `type` and `t` | Marker | Meaning |
|---|---|---|---|
| `calibration` | `detail` | no | The tracker's calibration outcome, logged after calibration and again when a drift check changes it. |
| `session_start` | `participant`, `session`, `planned` | no | Start of the session, with a random session identifier and the passages the session sets out to present. |
| `prompt` | `passage`, `question` | yes | A question shown before its passage under the information-seeking regime. `t` is when the reader moved on. |
| `drift_check` | `passage`, `error` | no | The drift check before a passage. `error` is the value reported by the tracker, or `null` when the check was aborted or escalated. |
| `passage_onset`, `passage_offset` | `passage` | yes | The first and last frames of a passage. |
| `gaze` | `tracker_t`, `x`, `y`, `valid` | no | One polled sample in screen pixels. `tracker_t` is the sample's time on the tracker's clock, which for the EyeLink is the EDF timestamp in seconds, and the session time for trackers without a clock of their own. |
| `word_enter` | `passage`, `word`, `tracker_t` | yes | A polled sample landed on a different word, given by its index. `tracker_t` is that sample's tracker time. |
| `region_enter`, `region_exit` | `passage`, `region`, `layer`, `reason`, `tracker_t` | yes | A polled sample landed on a word inside or outside an annotated region. Nested and overlapping regions each have their own events. `reason` on an exit is `gaze`, or `passage_end` when the passage ended with the gaze inside the region. Samples that fall on no word, during blinks, between words or off the text, leave the current word and regions unchanged, so dwell times computed from these events include such samples. |
| `response` | `onset`, `passage`, `question`, `kind`, `region`, `response`, `correct` | yes | A scored answer. `t` is the response time and `onset` the time the question appeared. The marker carries the passage and question only. |
| `session_end` | `participant`, `session`, `status`, `error` | no | End of the session. `status` is `completed`, or `aborted` when the session stopped on an error, whose type is given in `error`. |

`EyeLinkMarkerSink` caps each message at 128 characters, within the limits that pylink and PsychoPy's SR Research plugin document, so a marker with very long identifiers is shortened in the data file while the log keeps it in full. The event CSV has the same fields as columns, with the common keys first.

## Analysis

readsync records and synchronises, and leaves analysis to established tools. The EyeLink data file holds the full-rate gaze, the markers as messages and one trial per passage with its interest-area set, so that Data Viewer can segment the recording by passage and load the word boxes. This still has to be confirmed with a recording from a tracker. After conversion to ASC, it can be read by Python and R tools such as [Eyekit](https://github.com/jwcarr/eyekit). The interest-area CSV gives each word's box and annotated regions, so fixations detected by any tool can be assigned to the boxes the session used.

The quality report covers the items that a minimal reporting guideline for eye-tracking research asks for under calibration and data loss (Dunn et al., 2024). For each passage it gives the number of polled samples, the share that were invalid, the share of expected samples that never arrived, the longest stretch without a sample, counted from onset to offset, and the number of samples that repeated the tracker's previous time. Because the session polls once per display frame, the expected rate is the display's refresh rate. The report also lists the logged calibrations and drift checks, the comprehension score, the session's status and the number of passages it planned.

## Webcam gaze

`WebcamTracker` implements the same `Tracker` protocol as the EyeLink backend, so a session can run on a webcam with only the tracker swapped. It combines a camera (`FrameSource`) and a gaze estimator (`GazeEstimator`). `MediaPipeIrisEstimator` is the estimator over MediaPipe's FaceLandmarker model, which is fetched once into `models/`. Given a fitted `HeadAwareGazeModel`, it maps each frame's signals to the screen and reports zero confidence during blinks and once the head leaves the calibrated pose range, and `WebcamTracker` marks those samples invalid.

`readsync.webcam.gaze` holds the gaze model. The iris feature is measured in the eye's own axes, and the estimator also reads head yaw, pitch and roll from the FaceLandmarker transformation matrix and an eye-openness ratio. `HeadAwareGazeModel` fits a regularised map from the iris feature, yaw and pitch to the screen and reports how far the head is outside the range of poses seen during calibration. Calibrating at more than one head position gives the pose terms variation to learn from, and the calibration error is estimated by cross-validation that holds out each target at every pose. `AffineCalibration` and `PolynomialCalibration` map the iris feature alone. `readsync.webcam.benchmark` expresses the error between two gaze streams in degrees of visual angle.

The model has known limits that a comparison with a reference eye-tracker should quantify before it is used for measurement. Rolling the head while looking at the same point turns the eyes within the head, which changes the iris feature, and the model has no roll terms. MediaPipe gives landmark x and y as fractions of the frame's width and height, so on a non-square frame the eye geometry, and with it the blink threshold, depends on the camera's aspect ratio. The pose check looks only at rotations outside the per-axis range seen at calibration and ignores head translation, so a reader who slides sideways is not flagged. Webcam samples are stamped with the time the session polled them, and the delay from the camera's exposure to the estimate is not recorded.

[`examples/webcam_live_demo.py`](examples/webcam_live_demo.py) runs the head-aware model on a laptop without PsychoPy. It calibrates against a grid of targets at two head positions by default and then shows passages with a live gaze dot and the word under it. Everything is recorded to the encrypted log. The browser demo in [`web-demo/`](web-demo) runs the same model in JavaScript.

## References

Dunn, M. J., Alexander, R. G., Amiebenomo, O. M., Arblaster, G., Atan, D., Erichsen, J. T., Ettinger, U., Giardini, M. E., Gilchrist, I. D., Hamilton, R., Hessels, R. S., Hodgins, S., Hooge, I. T. C., Jackson, B. S., Lee, H., Macknik, S. L., Martinez-Conde, S., Mcilreavy, L., Muratori, L. M., … Sprenger, A. (2024). Minimal reporting guideline for research involving eye tracking (2023 edition). *Behavior Research Methods*, *56*(5), 4351–4357. https://doi.org/10.3758/s13428-023-02187-1

Niehorster, D. C., Andersson, R., & Nyström, M. (2020). Titta: A toolbox for creating PsychToolbox and Psychopy experiments with Tobii eye trackers. *Behavior Research Methods*, *52*(5), 1970–1979. https://doi.org/10.3758/s13428-020-01358-8
