# Sitting Posture Classifier (webcam demo)

A desktop application that classifies sitting posture from a live webcam into five classes
(upright, leaning forward, leaning backward, leaning left, leaning right) and alerts when a
non-upright posture is held for more than 5 seconds. It was built for a university Image
Processing and Computer Vision assignment on class imbalance.

## How it works

Webcam frame -> MediaPipe Pose Landmarker (landmarks 0-12, x, y, z = 39 values) -> Logistic Regression.
If no pose is found, the frame is tried again after contrast enhancement (CLAHE) and a black border.
The shown posture is the most common prediction over the last second.

Three models run side by side on the same frame. Only their training data differ:

| Key | Model | Training data | Held-out minority recall | Held-out upright recall |
|---|---|---|---|---|
| 1 | Natural training | 25,348 frames | 0.97 | 0.92 |
| 2 | Reduced minority | 14,898 frames, leaning left and right cut to 30 each | 0.77 | 0.93 |
| 3 | Reduced minority + class weighting | Same as 2, class weights | 0.94 | 0.85 |

Recall was measured on held-out participants with five participant folds.
Postures on screen are named from the user's own point of view.

## Run (tested on Windows with Python 3.14)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python posture_app.py            # or: python posture_app.py --camera 1

Sit upright in view and press **Setup check**. Monitoring starts when the camera position is OK.
Press 1, 2 or 3 to switch the active model.

## Files

| File | Purpose |
|---|---|
| posture_app.py | The application (window, camera, person check, alerts) |
| pose_extractor.py, preprocessing.py | MediaPipe pose detection and the contrast and border fallback |
| models/demo/*.joblib | The three trained models (model parameters only) |
| train_demo_models.py, demo_imbalance_check.py | How the models were trained and evaluated (need the data, not included) |
| Other .py files | Shared helper code imported by the files above |

## Data and limitations

The training images and videos are not included, to protect participant privacy.
The application was tested by one person in one room, so it is a demonstration, not a validated tool.
The pose model `models/pose_landmarker_full.task` is provided by Google MediaPipe under the Apache 2.0 licence.