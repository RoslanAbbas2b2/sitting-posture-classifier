"""MediaPipe Pose Landmarker wrapper (Tasks API).

mediapipe 1.1.1 no longer has the legacy mp.solutions.pose API, so the
Tasks API is used. Model: pose_landmarker_full.task (balance between the
lite and heavy models, and light enough to run in real time in the FYP).

PoseExtractor.extract() returns a (33, 5) array per image with columns
x, y, z, visibility, presence, or None when no pose is detected.
x and y are normalised to [0, 1] by image width and height.

Run this file directly to draw landmarks on one image per class:
  python pose_extractor.py   ->  data/pose_check.jpg
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("models") / "pose_landmarker_full.task"
N_LANDMARKS = 33

# Upper-body landmark indices (MediaPipe pose numbering).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
UPPER_BODY_CONNECTIONS = [
    (LEFT_EAR, LEFT_EYE), (LEFT_EYE, NOSE), (NOSE, RIGHT_EYE), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]


class PoseExtractor:
    """Detects one person's pose landmarks in a single image."""

    def __init__(self, model_path: Path = MODEL_PATH, min_detection_confidence: float = 0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Return a (33, 5) array [x, y, z, visibility, presence], or None if no pose."""
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z, lm.visibility, lm.presence]
                         for lm in result.pose_landmarks[0]], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_upper_body(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw upper-body landmarks and connections on a copy of the image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    pts = {i: (int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)) for i in range(25)}
    for a, b in UPPER_BODY_CONNECTIONS:
        cv2.line(out, pts[a], pts[b], (0, 255, 0), 3)
    for i in range(25):
        cv2.circle(out, pts[i], 6, (0, 0, 255), -1)
    return out


if __name__ == "__main__":
    import pandas as pd

    from config import RAW_ROOT
    manifest = pd.read_csv(Path("data") / "manifest.csv")
    samples = manifest.groupby("class").sample(1, random_state=42)

    tiles = []
    with PoseExtractor() as extractor:
        for _, r in samples.iterrows():
            img = cv2.imread(str(RAW_ROOT / r["path"]))
            lms = extractor.extract(img)
            if lms is None:
                print(f"{r['class']:18s} NO POSE  {r['path']}")
                tile = img
            else:
                vis = lms[[LEFT_SHOULDER, RIGHT_SHOULDER], 3].mean()
                print(f"{r['class']:18s} pose found, shoulder visibility {vis:.2f}")
                tile = draw_upper_body(img, lms)
            tile = cv2.resize(tile, (432, 288))
            cv2.putText(tile, r["class"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(tile)

    out = Path("data") / "pose_check.jpg"
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"\nSaved {out.resolve()}")