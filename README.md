# Sitting Posture Classifier (webcam demo)

A desktop application that classifies sitting posture from a live webcam into five classes
(upright, leaning forward, leaning backward, leaning left, leaning right) and alerts when a
non-upright posture is held for more than 5 seconds.

Pipeline: MediaPipe Pose Landmarker (landmarks 0-12, 39 values) -> Logistic Regression.
Three models run side by side on the same frame to show the effect of class imbalance:

| Key | Model |
|---|---|
| 1 | Natural training data |
| 2 | Leaning left and right reduced to 30 images each |
| 3 | Same reduced data with class weighting |

## Run (Windows, Python 3)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python posture_app.py            # or: python posture_app.py --camera 1
```

Sit upright in view, press **Setup check**, then press 1, 2 or 3 to switch models.

## Data

The training images and videos are not included, to protect participant privacy.
The saved models contain only model parameters. `train_demo_models.py` shows how they
were trained but cannot be run without the data.

The pose model `models/pose_landmarker_full.task` is provided by Google MediaPipe under the Apache 2.0 licence.
