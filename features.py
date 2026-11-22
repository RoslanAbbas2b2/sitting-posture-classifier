"""Feature builders for the MediaPipe route.

LandmarkFeatures reads a landmark CSV (from extract_landmarks.py) and turns
it into a feature matrix with one row per manifest image, in manifest order.
Rows for images where no pose was detected are NaN, so the evaluator treats
them as 'no_pose'.

Only landmarks 0-12 (face and shoulders) are used: their mean visibility is
1.00, while landmarks 13-24 are mostly outside the frame (visibility 0.02-0.31).

Works for both datasets: pass the folder the image paths are relative to and a
separate image-size cache (Roboflow: RAW_ROOT and data/image_sizes.csv, the
defaults; own data: OWN_FRAMES_ROOT and data/image_sizes_own.csv).

Ablation stages:
  raw_coordinates()        stage 1: x, y, z of landmarks 0-12 (39 features)
  quality_mask()           stage 2: training images whose landmarks 0-12 all have
                           visibility >= threshold (training filter only)
  normalised_coordinates() stage 4: landmarks 0-12 in pixel units, centred on the
                           shoulder midpoint and divided by shoulder width (39)
  geometric_features()     stage 5: 12 posture measurements (GEOMETRIC_NAMES)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import OWN_FRAMES_ROOT, RAW_ROOT

IMAGE_SIZES = Path("data") / "image_sizes.csv"
OWN_IMAGE_SIZES = Path("data") / "image_sizes_own.csv"
UPPER_BODY = range(13)  # landmarks 0-12

# MediaPipe pose landmark indices. "Left" is the PERSON's left
# (the image right side when the person faces the camera).
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

GEOMETRIC_NAMES = [
    "mid_x",             # shoulder midpoint x / image width
    "mid_y",             # shoulder midpoint y / image height
    "shoulder_width",    # shoulder distance / image width
    "shoulder_tilt",     # angle of the shoulder line, degrees
    "eye_tilt",          # angle of the eye line, degrees
    "ear_tilt",          # angle of the ear line, degrees
    "head_dx",           # (nose x - shoulder midpoint x) / shoulder width
    "head_dy",           # (shoulder midpoint y - nose y) / shoulder width
    "head_dz",           # (nose z - shoulder midpoint z) / shoulder width
    "face_size",         # eye distance / image width
    "face_to_shoulder",  # eye distance / shoulder width
    "head_yaw",          # (nose-left ear - nose-right ear) / (sum of both)
]


def image_sizes(paths: np.ndarray, image_root: Path = RAW_ROOT,
                cache: Path = IMAGE_SIZES) -> pd.DataFrame:
    """Width and height in pixels of every image, read from the file header.
    Cached in `cache` (rebuilt if the image list changes)."""
    if cache.exists():
        sizes = pd.read_csv(cache)
        if np.array_equal(sizes["path"].to_numpy(), paths):
            return sizes
    rows = []
    for p in paths:
        with Image.open(Path(image_root) / p) as img:
            width, height = img.size
        rows.append({"path": p, "width": width, "height": height})
    sizes = pd.DataFrame(rows)
    sizes.to_csv(cache, index=False)
    return sizes


def _angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Angle in degrees of the line from the person's right point to their left
    point (about 0 when level; positive when the image-right point is lower)."""
    d = left - right
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


class LandmarkFeatures:
    """Builds feature matrices from one landmark CSV."""

    def __init__(self, landmark_csv: Path, image_root: Path = RAW_ROOT,
                 sizes_cache: Path = IMAGE_SIZES):
        self.table = pd.read_csv(landmark_csv)
        self.paths = self.table["path"].to_numpy()
        self.detected = self.table["detected"].to_numpy(dtype=bool)
        self.image_root = image_root
        self.sizes_cache = sizes_cache

    @classmethod
    def own(cls, landmark_csv: Path) -> "LandmarkFeatures":
        """Features for the own dataset (frames in OWN_FRAMES_ROOT)."""
        return cls(landmark_csv, OWN_FRAMES_ROOT, OWN_IMAGE_SIZES)

    def sizes(self) -> pd.DataFrame:
        return image_sizes(self.paths, self.image_root, self.sizes_cache)

    def raw_coordinates(self) -> np.ndarray:
        """Stage 1: raw normalised image coordinates x, y, z of landmarks 0-12."""
        cols = [f"lm{i}_{f}" for i in UPPER_BODY for f in ("x", "y", "z")]
        return self.table[cols].to_numpy(dtype=np.float64)

    def quality_mask(self, min_visibility: float = 0.5) -> np.ndarray:
        """Stage 2: True where a pose was detected and every landmark 0-12 has
        visibility >= min_visibility."""
        vis = self.table[[f"lm{i}_vis" for i in UPPER_BODY]].to_numpy()
        return self.detected & (np.nan_to_num(vis, nan=0.0) >= min_visibility).all(axis=1)

    def points_in_pixels(self) -> np.ndarray:
        """Landmarks 0-12 as an (images, 13, 3) array in pixel units.
        MediaPipe divides x by image width and y by image height, so x and y have
        different scales; multiplying back makes distances and angles correct.
        z uses roughly the same scale as x, so it is multiplied by the width."""
        sizes = self.sizes()
        width = sizes["width"].to_numpy(dtype=np.float64)[:, None]
        height = sizes["height"].to_numpy(dtype=np.float64)[:, None]
        pts = self.raw_coordinates().reshape(len(self.paths), len(UPPER_BODY), 3).copy()
        pts[:, :, 0] *= width
        pts[:, :, 1] *= height
        pts[:, :, 2] *= width
        return pts

    def normalised_coordinates(self) -> np.ndarray:
        """Stage 4: remove where the person sits in the frame and how large they
        appear. Centre on the shoulder midpoint, divide by shoulder width (x, y)."""
        pts = self.points_in_pixels()
        midpoint = (pts[:, LEFT_SHOULDER] + pts[:, RIGHT_SHOULDER]) / 2
        shoulder_width = np.linalg.norm(pts[:, LEFT_SHOULDER, :2] - pts[:, RIGHT_SHOULDER, :2], axis=1)
        with np.errstate(invalid="ignore"):  # rows without a pose stay NaN
            pts = (pts - midpoint[:, None, :]) / shoulder_width[:, None, None]
        return pts.reshape(len(pts), -1)

    def geometric_features(self) -> np.ndarray:
        """Stage 5: 12 interpretable posture measurements (see GEOMETRIC_NAMES)."""
        sizes = self.sizes()
        img_w = sizes["width"].to_numpy(dtype=np.float64)
        img_h = sizes["height"].to_numpy(dtype=np.float64)
        pts = self.points_in_pixels()
        xy = pts[:, :, :2]

        mid = (pts[:, LEFT_SHOULDER] + pts[:, RIGHT_SHOULDER]) / 2
        shoulder_w = np.linalg.norm(xy[:, LEFT_SHOULDER] - xy[:, RIGHT_SHOULDER], axis=1)
        eye_dist = np.linalg.norm(xy[:, LEFT_EYE] - xy[:, RIGHT_EYE], axis=1)
        nose_left_ear = np.linalg.norm(xy[:, NOSE] - xy[:, LEFT_EAR], axis=1)
        nose_right_ear = np.linalg.norm(xy[:, NOSE] - xy[:, RIGHT_EAR], axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):  # rows without a pose stay NaN
            columns = [
                mid[:, 0] / img_w,
                mid[:, 1] / img_h,
                shoulder_w / img_w,
                _angle(xy[:, LEFT_SHOULDER], xy[:, RIGHT_SHOULDER]),
                _angle(xy[:, LEFT_EYE], xy[:, RIGHT_EYE]),
                _angle(xy[:, LEFT_EAR], xy[:, RIGHT_EAR]),
                (pts[:, NOSE, 0] - mid[:, 0]) / shoulder_w,
                (mid[:, 1] - pts[:, NOSE, 1]) / shoulder_w,
                (pts[:, NOSE, 2] - mid[:, 2]) / shoulder_w,
                eye_dist / img_w,
                eye_dist / shoulder_w,
                (nose_left_ear - nose_right_ear) / (nose_left_ear + nose_right_ear),
            ]
        return np.column_stack(columns)