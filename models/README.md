# Models

The webcam estimator uses the MediaPipe **FaceLandmarker** model, a single file
of about 3.7 MB, released by Google under the Apache License 2.0. It is not
committed (it is a binary weight, not source), so fetch it once into this
directory:

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The example [`../examples/webcam_live_demo.py`](../examples/webcam_live_demo.py)
looks for `models/face_landmarker.task` by default and prints this command if the
file is missing. The download happens once and online. A recording session itself
needs no network.

The canonical URL is also exposed in code as
`readsync.webcam.FACE_LANDMARKER_URL`.
