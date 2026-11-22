"""Image preprocessing applied before MediaPipe (ablation stage 3).

Each step has:
  apply(image)            -> preprocessed image given to MediaPipe
  to_original(landmarks)  -> landmarks mapped back to the ORIGINAL image,
                             so features from every variant mean the same thing.

PreprocessedExtraction chains the steps around a PoseExtractor. With
fallback_only=True the original image is tried first, and the steps are
used only when no pose is found.
"""
import cv2
import numpy as np

from pose_extractor import PoseExtractor


class Clahe:
    """Contrast Limited Adaptive Histogram Equalisation on the lightness
    channel (LAB colour space), so colours are not changed.
    clip_limit=2.0 and an 8x8 tile grid are common settings."""

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def to_original(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks  # image size unchanged


class PadBorder:
    """Add a black border of `fraction` x width/height on every side, so a face
    that fills the frame appears smaller and the head is fully inside."""

    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction
        self._pad = (0, 0)   # (top/bottom, left/right) of the last image
        self._size = (0, 0)  # (height, width) of the last original image

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ph, pw = int(h * self.fraction), int(w * self.fraction)
        self._pad, self._size = (ph, pw), (h, w)
        return cv2.copyMakeBorder(image, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def to_original(self, landmarks: np.ndarray) -> np.ndarray:
        """x, y are normalised by the padded size; convert back to the original size.
        z uses the same scale as x, so it is rescaled by the width ratio."""
        (ph, pw), (h, w) = self._pad, self._size
        out = landmarks.copy()
        out[:, 0] = (landmarks[:, 0] * (w + 2 * pw) - pw) / w
        out[:, 1] = (landmarks[:, 1] * (h + 2 * ph) - ph) / h
        out[:, 2] = landmarks[:, 2] * (w + 2 * pw) / w
        return out


class PreprocessedExtraction:
    """Preprocess an image, run MediaPipe, and map the landmarks back."""

    def __init__(self, steps: list, fallback_only: bool = False):
        self.steps = steps
        self.fallback_only = fallback_only

    def extract(self, extractor: PoseExtractor, image: np.ndarray) -> np.ndarray | None:
        if self.fallback_only or not self.steps:
            landmarks = extractor.extract(image)
            if landmarks is not None or not self.steps:
                return landmarks
        processed = image
        for step in self.steps:
            processed = step.apply(processed)
        landmarks = extractor.extract(processed)
        if landmarks is None:
            return None
        for step in reversed(self.steps):
            landmarks = step.to_original(landmarks)
        return landmarks