# Sitting Posture Classifier

A desktop app that watches how you sit through your webcam and warns you when you lean
out of an upright posture for too long.

It was built for a university course in Image Processing and Computer Vision. The course
project studied **class imbalance**: what happens to a model when some postures have far
fewer training examples than others. This app lets you see that effect live.

## What the app does

1. It looks at you through the webcam and finds 13 points on your head and shoulders
   (nose, eyes, ears, mouth and shoulders).
2. From the positions of these points, a model decides which of five postures you are in:
   **upright, leaning forward, leaning backward, leaning left or leaning right**.
3. If you stay in any posture other than upright for more than **5 seconds**, the screen
   turns red and the app beeps until you change position.

Before it starts, the app asks you to sit upright for 3 seconds to check that the camera
can see you properly. If you leave the chair, it pauses by itself.

## Why there are three models

In real use, some postures happen much less often than others, so a model may see very few
examples of them during training. That is class imbalance. The app runs three models side by
side on the same camera image so you can compare them:

| Key | Model | What it was trained on |
|---|---|---|
| 1 | Normal | All 25,348 training frames |
| 2 | Imbalanced | Leaning left and leaning right cut to only 30 frames each |
| 3 | Imbalanced, with class weighting | The same reduced data, but the rare postures count more during training |

Press **1**, **2** or **3** to choose which model controls the display and the alert.
When the models give different answers for the same frame, the app shows "Conditions disagree".

**What to expect.** The models were tested on people they had never seen before. The table
shows recall, which is the share of each posture the model recognised correctly:

| Model | Leaning left and right recognised | Upright recognised |
|---|---|---|
| 1 Normal | 97% | 92% |
| 2 Imbalanced | 77% | 93% |
| 3 Imbalanced, with class weighting | 94% | 85% |

Cutting the leaning examples makes the model miss more leans (97% down to 77%). Class
weighting recovers most of them (94%), but the model then calls upright sitting a lean
more often (92% down to 85%).

## How to run it (tested on Windows with Python 3.14)

You need a computer with a webcam and Python installed.

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python posture_app.py            # if you have two cameras: python posture_app.py --camera 1

Then:
1. Sit upright facing the camera, with your head and shoulders in view.
2. Click **Setup check** and stay still for 3 seconds.
3. When it says the camera position is OK, monitoring starts. Try leaning and see what happens.

Postures on screen are named from your own point of view, so "Leaning right" means you lean
to your right.

## Technical details

- **Pose detection:** Google MediaPipe Pose Landmarker. Landmarks 0 to 12 give 13 points
  with x, y and z each, so 39 numbers per frame. If no person is found, the frame is tried
  again after contrast enhancement (CLAHE) and adding a black border.
- **Classifier:** Logistic Regression with standardised inputs (scikit-learn).
- **Smoothing:** the posture shown is the most common prediction over the last second.
- **Person check:** a frame counts only if both shoulders are visible and wide enough in
  the image, so an empty chair is not treated as a person.

| File | Purpose |
|---|---|
| posture_app.py | The app (window, camera, person check, alerts) |
| pose_extractor.py, preprocessing.py | Pose detection and the contrast and border fallback |
| models/demo/*.joblib | The three trained models (model settings only) |
| train_demo_models.py, demo_imbalance_check.py | How the models were trained and tested (need the data, not included) |
| Other .py files | Shared helper code used by the files above |

## Privacy and limitations

- The training images and videos are **not included**, to protect the participants' privacy.
  The saved models contain only numbers learned during training, not images.
- Everything runs on your own computer. The app does not record or send your video.
- It was tested by one person in one room. It is a demonstration, not a validated health tool.

The pose model `models/pose_landmarker_full.task` is provided by Google MediaPipe under the
Apache 2.0 licence.