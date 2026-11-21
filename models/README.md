# Models

The webcam estimator uses MediaPipe's FaceLandmarker model, a single file of about 3.8 MB published by Google under the Apache License 2.0. The file is not tracked in the repository. Fetch it once into this directory:

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

[`../examples/webcam_live_demo.py`](../examples/webcam_live_demo.py) looks for `models/face_landmarker.task` by default and prints this command if the file is missing. Only the download needs a network connection. The URL is also available in code as `readsync.webcam.FACE_LANDMARKER_URL`, and the browser demo pins the same file by its SHA-256 in `web-demo/scripts/assets-manifest.mjs`.
