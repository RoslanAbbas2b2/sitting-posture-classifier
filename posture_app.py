"""Sitting Posture Classifier: posture classification application (objective O6).

A desktop application (PySide6) that classifies the user's sitting posture from a live webcam and
beeps when a non-upright posture is held too long. It uses the selected pipeline of the report and
lets the user switch live between three TRAINING CONDITIONS of the same classifier (Logistic Regression;
keys 1 / 2 / 3) to show the effect of class imbalance and of class weighting:
  1  Natural training                     all training data (upright : lean about 1.5 : 1), no handling
  2  Reduced minority training            leaning left/right cut to 30 images each (about 274 : 1), no handling
  3  Reduced minority + class weighting   same reduced data, class weights (selected pipeline)
The three fitted pipelines are trained by train_demo_models.py (models/demo/*.joblib).

What happens on every frame (no posture rules, no calibration inside the model):
  webcam frame -> MediaPipe Pose Landmarker (fallback: CLAHE + 25% border only if no pose)
  -> landmarks 0-12 -> x, y, z re-expressed in a centred 3:2 frame (as for the Newly Collected Dataset)
  -> 39 numbers -> active model -> 5 probabilities (sum 1) -> highest = posture
  (the other two models are applied to the same 39 numbers and shown side by side)
  -> smoothed over about 1 s -> a non-upright posture held longer than the limit -> red + beep.

Setup check: checks only the camera position (shoulders visible, body size inside the range of the
training data); it does not change the model.

Labels follow the camera view (leaning_left = the person leans towards the image left, their own
right), so the messages shown to the user flip left and right.

Run:  python posture_app.py            (camera 0)
      python posture_app.py --camera 1
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pose_extractor import LEFT_SHOULDER, RIGHT_SHOULDER, PoseExtractor
from preprocessing import Clahe, PadBorder, PreprocessedExtraction
from run_roboflow_vs_joint import TARGET_ASPECT, centre_crop_3x2

# Faster start-up for the Windows Media Foundation (MSMF) camera backend (as record_session.py).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

MODEL_DIR = Path("models") / "demo"
TRAINING_COUNTS = MODEL_DIR / "training_counts.csv"              # written by train_demo_models.py
EVALUATION = Path("results") / "demo_imbalance_check.csv"        # held-out evaluation of the same settings
MODEL_KEYS = ("model1", "model2", "model3")
UPPER_BODY = range(13)               # landmarks 0-12, the model's input
HOLD_LIMIT_S = 5.0                   # demo limit before an alert
SMOOTH_S = 1.0                       # majority vote over the last second
BEEP_REPEAT_S = 4.0                  # repeat the beep while the posture is not corrected
SETUP_S = 3.0                        # setup check: sit upright this long
# Person check (diagnose_presence*.py, 10 Nov 2026, one user, one room): genuine postures had a shoulder width of
# at least 0.284, false poses on the empty chair at most 0.168 (95th percentile). 0.25 is an initial,
# experimentally selected limit, not a universal guarantee.
MIN_SHOULDER_WIDTH = 0.25
PRESENCE_WINDOW_S = 1.0              # presence = majority of valid frames over the last second
PRESENCE_MIN_FRAMES = 3              # never decide from fewer frames (low frame rates)
RESUME_LIMIT_S = 60.0                # away longer than this: the setup check must be run again
RECHECK_S = 1.0                      # back within the limit: 1 s position recheck before monitoring resumes
# Upright shoulder width / image height of the Newly Collected Dataset participants: median 0.509,
# 25th-75th percentile 0.476-0.543 (notebook Section 6.1, camera geometry). The check accepts this range
# with a margin.
SHOULDER_RANGE = (0.44, 0.60)
PROBA_ORDER = ("upright", "leaning_left", "leaning_right", "leaning_forward", "leaning_backward")
# On screen every class is named from the USER's point of view. The dataset classes follow the camera view
# (image not mirrored): leaning_left = the person moves towards the image left = leans to their own right.
DISPLAY_NAME = {
    "upright": "Upright",
    "leaning_left": "Leaning right",
    "leaning_right": "Leaning left",
    "leaning_forward": "Leaning forward",
    "leaning_backward": "Leaning backward",
}
DISPLAY_ORDER = ("upright", "leaning_right", "leaning_left", "leaning_forward", "leaning_backward")  # left, right


def dataset_tip(cls: str) -> str:
    """Hover text linking a displayed name to the dataset class."""
    return f"Dataset class {cls} (camera view)"
NON_UPRIGHT = ("leaning_forward", "leaning_backward", "leaning_left", "leaning_right")
SHOWN_AS = {  # what the user reads (left and right flipped: labels follow the camera view)
    "upright": "Upright",
    "leaning_forward": "Leaning forward",
    "leaning_backward": "Leaning backward",
    "leaning_left": "Leaning to your right",
    "leaning_right": "Leaning to your left",
    "no_pose": "No person detected",
}
# Display names of the three training conditions (counts: models/demo/training_counts.csv; upright 8,234 vs
# 30 images per lean class in conditions 2 and 3). The saved .joblib files keep their original labels.
CONDITIONS = {
    "model1": ("Natural training", "Natural",
               "All training data (upright about 1.5 times each lean class), no imbalance handling"),
    "model2": ("Reduced minority training", "Reduced",
               "Leaning left and right cut to 30 images each (upright about 274 times larger), no handling"),
    "model3": ("Reduced minority + class weighting", "Weighted",
               "Same 30 images per lean class, class weights (selected pipeline)"),
}
CLASSIFIER = "Logistic Regression on 39 MediaPipe landmark features"
MODE_STATUS = {  # short state for the status bar (the full reason is shown next to the Setup check button)
    "preview": "Setup required",
    "checking": "Checking camera position",
    "setup_failed": "Setup check failed",
    "rechecking": "Checking position",
    "paused": "Paused · no person",
}
MODE_BADGE = {  # badge while not monitoring (a person is in view)
    "preview": "Setup required",
    "checking": "Checking camera position ...",
    "setup_failed": "Setup check failed",
    "rechecking": "Checking position ...",
    "paused": "No person detected",
}


# ----------------------------------------------------------------------------------------------
# Model side: features and prediction for one live frame
# ----------------------------------------------------------------------------------------------
def crop_to_3x2(frame: np.ndarray) -> np.ndarray:
    """Centre-crop a frame to 3:2 for DISPLAY (the features use the same crop, see features())."""
    h, w = frame.shape[:2]
    new_w = min(w, int(round(h * TARGET_ASPECT)))
    x0 = (w - new_w) // 2
    return frame[:, x0:x0 + new_w]


def features(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """39 values (x, y, z of landmarks 0-12) in a centred 3:2 frame, exactly as for the
    Newly Collected Dataset training frames (run_roboflow_vs_joint.centre_crop_3x2)."""
    coords = landmarks[list(UPPER_BODY), :3].reshape(1, -1).astype(np.float64)
    return centre_crop_3x2(coords, pd.DataFrame({"width": [width], "height": [height]}))


def shoulder_width_ratio(landmarks: np.ndarray, width: int, height: int) -> float:
    """Shoulder distance divided by image height, measured in the 3:2 frame."""
    new_w = min(width, height * TARGET_ASPECT)
    offset = (width - new_w) / 2
    xs = (landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 0] * width - offset)
    ys = landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 1] * height
    return float(np.hypot(xs[0] - xs[1], ys[0] - ys[1]) / height)


def condition_stats(models: dict) -> dict:
    """Per training condition, from saved files only (missing files give None):
      counts       training images per class (models/demo/training_counts.csv, train_demo_models.py)
      weights      class weights of "balanced" class weighting for that data: n / (5 x n_class), if the condition
                   uses class weighting (scikit-learn formula)
      lean_recall  mean recall of leaning_left and leaning_right on held-out participants, averaged over seeds
      upright_recall  recall of upright, same evaluation (results/demo_imbalance_check.csv, same setting and
                   classifier)"""
    counts = pd.read_csv(TRAINING_COUNTS, index_col=0) if TRAINING_COUNTS.exists() else None
    evaluation = pd.read_csv(EVALUATION) if EVALUATION.exists() else None
    stats = {}
    for key, spec in models.items():
        entry = {"counts": None, "weights": None, "lean_recall": None, "upright_recall": None}
        if counts is not None and key in counts:
            entry["counts"] = c = counts[key]
            if str(spec.get("model", "")).endswith("+weighted"):
                entry["weights"] = c.sum() / (len(c) * c)
        if evaluation is not None and {"setting", "model"} <= set(spec):
            rows = evaluation[(evaluation["setting"] == spec["setting"]) & (evaluation["model"] == spec["model"])]
            if len(rows):
                entry["lean_recall"] = float(((rows["recall_leaning_left"] + rows["recall_leaning_right"]) / 2).mean())
                entry["upright_recall"] = float(rows["recall_upright"].mean())
        stats[key] = entry
    return stats


def load_models() -> dict:
    """The three saved models; a clear message if train_demo_models.py has not been run."""
    models = {}
    for key in MODEL_KEYS:
        path = MODEL_DIR / f"{key}.joblib"
        if not path.exists():
            raise SystemExit(f"{path} not found · run  python train_demo_models.py  first.")
        models[key] = joblib.load(path)
    return models


@dataclass
class FrameResult:
    """Everything the window needs about one camera frame."""

    image: np.ndarray                 # RGB, 3:2 crop, landmarks drawn
    probabilities: dict | None        # active model: class -> probability; None if no pose
    all_probabilities: dict | None    # every model: model key -> {class -> probability}
    predicted: str                    # highest-probability class or "no_pose"
    shoulder_ratio: float | None      # for the setup check
    shoulders_visible: bool
    process_ms: float = 0.0           # MediaPipe + the three models, for this frame
    frame_status: str = "no_pose"     # "no_pose" | "invalid_pose" (fails the person check) | "valid"


class CameraWorker(QThread):
    """Reads the webcam and runs MediaPipe and the active model off the user-interface thread."""

    frame_ready = Signal(object)      # FrameResult
    status = Signal(str)              # progress messages while starting
    failed = Signal(str)

    def __init__(self, models: dict, camera_index: int = 0):
        super().__init__()
        self.models = models
        self.active_key = "model3"
        self.camera_index = camera_index
        self._running = True
        self._shape_warned = False
        self.frame_size = (720, 1280)        # (height, width) of the last camera frame

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def open_camera(self):
        """Try DirectShow first, then Media Foundation (as record_session.py); return the first
        backend that actually delivers a frame, or None."""
        backends = ([("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)] if sys.platform == "win32"
                    else [("default", cv2.CAP_ANY)])
        for name, backend in backends:
            self.status.emit(f"Opening camera {self.camera_index} ({name}) ...")
            capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # same size as the newly collected recordings
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, first = capture.read()
                if ok:
                    h, w = first.shape[:2]
                    print(f"Camera {self.camera_index} opened with {name} ({w} x {h}).")
                    return capture
            capture.release()
        return None

    def run(self) -> None:
        try:
            capture = self.open_camera()
            if capture is None:
                self.failed.emit(f"Camera {self.camera_index} could not be opened (tried DSHOW and MSMF). "
                                 "Close other apps using the camera, or try --camera 1.")
                return
            self.status.emit("Loading the pose model ...")
            fallback = PreprocessedExtraction([Clahe(), PadBorder(0.25)], fallback_only=True)
            with PoseExtractor() as extractor:
                self.status.emit("")
                while self._running:
                    ok, frame = capture.read()
                    if not ok:
                        self.failed.emit("The camera stopped sending images.")
                        break
                    self.frame_ready.emit(self.process(frame, extractor, fallback))
            capture.release()
        except Exception as error:  # show the reason in the window instead of failing silently
            traceback.print_exc()
            self.failed.emit(f"Error · {error}")

    def process(self, frame: np.ndarray, extractor, fallback) -> FrameResult:
        """Landmarks -> person check -> features -> probabilities for one frame (unmirrored, as in training).
        A pose counts as a person only if both shoulders are visible (>= 0.5) and the shoulder width is at
        least MIN_SHOULDER_WIDTH; otherwise the models are not run and nothing is drawn."""
        start = time.perf_counter()
        height, width = frame.shape[:2]
        self.frame_size = (height, width)
        if width < height * TARGET_ASPECT - 1:
            # Narrower than 3:2 (e.g. a 4:3 camera): the features would not match the training frames
            # (centre_crop_3x2 assumes a wide image), so nothing is classified.
            if not self._shape_warned:
                self._shape_warned = True
                self.failed.emit(f"Camera image is {width} x {height} · a widescreen image (e.g. 1280 x 720) "
                                 "is needed. Try --camera 1 or another webcam.")
            return FrameResult(cv2.cvtColor(crop_to_3x2(frame), cv2.COLOR_BGR2RGB), None, None, "no_pose", None,
                               False, (time.perf_counter() - start) * 1000, frame_status="no_pose")
        landmarks = fallback.extract(extractor, frame)   # original frame first, CLAHE + border only if no pose
        display = crop_to_3x2(frame).copy()
        if landmarks is None:
            return FrameResult(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), None, None, "no_pose", None, False,
                               (time.perf_counter() - start) * 1000, frame_status="no_pose")
        # Fallback landmarks are already mapped back to the original frame, so both paths use the same scale.
        visible = bool(landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER], 3].min() >= 0.5)
        ratio = shoulder_width_ratio(landmarks, width, height)
        if not (visible and ratio >= MIN_SHOULDER_WIDTH):
            return FrameResult(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), None, None, "no_pose", ratio, visible,
                               (time.perf_counter() - start) * 1000, frame_status="invalid_pose")

        # All three models see the SAME 39 numbers, so their outputs can be compared on one frame.
        x = features(landmarks, width, height)
        all_probabilities = {}
        for key, spec in self.models.items():
            pipeline = spec["pipeline"]
            all_probabilities[key] = dict(zip(pipeline.classes_, pipeline.predict_proba(x)[0]))
        probabilities = all_probabilities[self.active_key]
        draw_landmarks(display, landmarks, width, height)
        return FrameResult(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), probabilities, all_probabilities,
                           max(probabilities, key=probabilities.get), ratio, visible,
                           (time.perf_counter() - start) * 1000, frame_status="valid")


def draw_landmarks(display: np.ndarray, landmarks: np.ndarray, width: int, height: int) -> None:
    """Draw landmarks 0-12 and the shoulder line on the 3:2 display image (in place)."""
    new_w = display.shape[1]
    offset = (width - new_w) / 2
    points = {i: (int(landmarks[i, 0] * width - offset), int(landmarks[i, 1] * height)) for i in UPPER_BODY}
    cv2.line(display, points[LEFT_SHOULDER], points[RIGHT_SHOULDER], (60, 160, 60), 3)
    for point in points.values():
        cv2.circle(display, point, 6, (60, 160, 60), -1)


# ----------------------------------------------------------------------------------------------
# Posture state: smoothing, held times, alert
# ----------------------------------------------------------------------------------------------
@dataclass
class PostureState:
    """Turns per-frame predictions into a stable posture, held times and an alert."""

    limit_s: float = HOLD_LIMIT_S
    history: deque = field(default_factory=deque)          # (time, predicted class)
    posture: str = "no_pose"                                # smoothed posture
    since: float = field(default_factory=time.monotonic)    # start of the current posture
    alert: bool = False
    last_beep: float = 0.0
    paused_at: float | None = None                          # set while frames are not valid

    def pause(self, now: float) -> None:
        """No valid person in this frame: no vote, and the held time stops until the next valid frame."""
        if self.paused_at is None:
            self.paused_at = now

    def update(self, predicted: str, now: float) -> bool:
        """Add one prediction (valid frames only). Returns True when a beep should sound now."""
        if self.paused_at is not None:                      # continue the timers after the pause
            gap = now - self.paused_at
            self.since += gap
            self.last_beep += gap
            self.paused_at = None
        self.history.append((now, predicted))
        while self.history and now - self.history[0][0] > SMOOTH_S:
            self.history.popleft()
        smoothed = Counter(p for _, p in self.history).most_common(1)[0][0]
        if smoothed != self.posture:
            self.posture, self.since = smoothed, now

        held = now - self.since
        if self.posture in NON_UPRIGHT and held > self.limit_s:
            if not self.alert or now - self.last_beep >= BEEP_REPEAT_S:
                self.alert, self.last_beep = True, now
                return True
        else:
            # Upright (or a new lean not yet past the limit): the alert ends. The 1 s majority vote
            # above already keeps single wrong frames from switching the posture, so no extra delay.
            self.alert = False
        return False

    def held(self, posture: str, now: float) -> float:
        end = self.paused_at if self.paused_at is not None else now
        return end - self.since if posture == self.posture else 0.0


@dataclass
class PresenceState:
    """Temporally confirmed person presence: the majority of valid frames over about one second.
    The window is time-based, so it works at any frame rate, and it never decides from a single frame."""

    window_s: float = PRESENCE_WINDOW_S
    history: deque = field(default_factory=deque)          # (time, frame valid)
    started: float | None = None
    present: bool = False

    def update(self, valid: bool, now: float) -> bool:
        """Add one frame. Returns True when the confirmed state changed."""
        if self.started is None:
            self.started = now
        self.history.append((now, valid))
        while now - self.history[0][0] > self.window_s:
            self.history.popleft()
        if len(self.history) < PRESENCE_MIN_FRAMES or now - self.started < 0.8 * self.window_s:
            return False                                    # not enough evidence yet: keep the state
        share = sum(v for _, v in self.history) / len(self.history)
        new = self.present if share == 0.5 else share > 0.5  # a tie keeps the current state
        changed, self.present = new != self.present, new
        return changed


def position_verdict(ratios: list, min_frames: int = 5) -> tuple[bool, str]:
    """Camera-position check on the shoulder widths of valid frames (median within SHOULDER_RANGE)."""
    if len(ratios) < min_frames:
        return False, "No person detected"
    ratio = float(np.median(ratios))
    low, high = SHOULDER_RANGE
    if ratio < low:
        return False, f"Move closer (shoulder width {ratio:.2f})"
    if ratio > high:
        return False, f"Move back (shoulder width {ratio:.2f})"
    return True, f"Camera position OK (shoulder width {ratio:.2f})"


@dataclass
class Session:
    """When monitoring is allowed.
      preview       start: camera preview only, no classification; the user must run the setup check
      checking      setup check running (SETUP_S): shoulder widths of valid frames are collected
      setup_failed  check failed: the reason is shown, still no classification
      monitoring    check passed: classification, timers and beeps
      paused        the person left (confirmed absence): no classification
      rechecking    the person returned within RESUME_LIMIT_S: RECHECK_S position recheck
    From paused: away longer than RESUME_LIMIT_S -> preview (new setup check needed). From rechecking:
    position still OK -> monitoring, otherwise -> preview."""

    mode: str = "preview"
    message: str = "Sit upright in view, then press Setup check"
    started: float = 0.0                                    # start of the running check
    ratios: list = field(default_factory=list)
    away_since: float | None = None

    @property
    def monitoring(self) -> bool:
        return self.mode == "monitoring"

    def start_setup(self, now: float) -> None:
        self.mode, self.message, self.started, self.ratios = "checking", "Checking camera position", now, []

    def update(self, valid: bool, ratio: float | None, present: bool, now: float) -> None:
        """One frame: valid = the frame passes the person check; present = confirmed presence."""
        if self.mode in ("checking", "rechecking") and valid and ratio is not None:
            self.ratios.append(ratio)
        if self.mode == "checking":
            if now - self.started >= SETUP_S:
                ok, text = position_verdict(self.ratios)
                self.mode, self.message = ("monitoring", text) if ok else ("setup_failed", text)
        elif self.mode == "monitoring" and not present:
            self.mode, self.message, self.away_since = "paused", "Paused · no person detected", now
        elif self.mode == "paused":
            if now - self.away_since > RESUME_LIMIT_S:
                self.mode = "preview"
                self.message = f"Away longer than {RESUME_LIMIT_S:g} s · run the setup check again"
            elif present:
                self.mode, self.message, self.started, self.ratios = "rechecking", "Checking position", now, []
        elif self.mode == "rechecking":
            if not present:
                self.mode, self.message = "paused", "Paused · no person detected"
            elif now - self.started >= RECHECK_S:
                ok, text = position_verdict(self.ratios, min_frames=PRESENCE_MIN_FRAMES)
                if ok:
                    self.mode, self.message = "monitoring", "Monitoring resumed"
                else:
                    self.mode, self.message = "preview", f"{text} · run the setup check again"


def beep() -> None:
    """Short beep without blocking the window."""
    if sys.platform == "win32":
        import winsound
        threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()
    else:
        QApplication.beep()


# ----------------------------------------------------------------------------------------------
# Window (glass look: translucent rounded panels over a soft painted background)
# ----------------------------------------------------------------------------------------------
# Frosted glass: the live camera image, heavily blurred, fills the window; the panels are smoky
# translucent glass on top of it with white text (as in visionOS / iOS frosted panels).
INK, MUTED = "#ffffff", "rgba(255, 255, 255, 0.62)"
ACCENT, OK_GREEN, AMBER, RED = "#9cc2ff", "#8ce99a", "#ffc46b", "#b3261e"
GOLD = "#e3b341"
SOFT_RED = "#ff8a80"                 # red text that stays readable on dark glass
BACKDROP_EVERY = 3                   # refresh the blurred backdrop every 3rd frame (about 10 per second)

STYLE = f"""
QWidget {{ background: transparent; color: {INK}; font-family: "Segoe UI Variable Display", "Segoe UI", Arial;
          font-size: 14px; }}
QLabel#title {{ font-size: 30px; font-weight: 600; }}
QLabel#hint, QLabel#caption {{ color: {MUTED}; font-size: 12px; }}
QFrame#glass, QFrame#card {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 rgba(30, 30, 34, 0.62), stop:1 rgba(18, 18, 22, 0.72));
    border: 1px solid rgba(255, 255, 255, 0.22); border-radius: 24px; }}
QFrame#card[alert="true"] {{ border: 3px solid {RED}; }}
QLabel#cardTitle {{ font-size: 16px; font-weight: 600; padding: 2px 4px 6px 4px; }}
QLabel#statusText {{ font-size: 16px; font-weight: 600; }}
QLabel#dot {{ background: {OK_GREEN}; border-radius: 6px; }}
QLabel#dot[alert="true"] {{ background: {RED}; }}
QLabel#chip {{ background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.25);
              border-radius: 13px; padding: 4px 12px; font-size: 13px; }}
QLabel#badge {{ background: rgba(20, 20, 24, 0.45); color: white; font-size: 16px; font-weight: 600;
               padding: 6px 18px; border: 1px solid rgba(255, 255, 255, 0.35); border-radius: 17px; }}
QLabel#badge[alert="true"] {{ background: {RED}; border: 1px solid #e05a50; }}
QProgressBar {{ background: rgba(255, 255, 255, 0.16); border: none; border-radius: 4px;
               max-height: 8px; min-height: 8px; }}
QProgressBar::chunk {{ background: rgba(255, 255, 255, 0.75); border-radius: 4px; }}
QProgressBar[top="true"]::chunk {{ background: {ACCENT}; }}
QProgressBar[alert="true"]::chunk {{ background: {RED}; }}
QProgressBar#setupBar {{ max-height: 30px; min-height: 30px; border-radius: 15px; text-align: center;
                        font-size: 13px; font-weight: 600; color: {INK}; }}
QProgressBar#setupBar::chunk {{ border-radius: 15px; background: rgba(255, 255, 255, 0.32); }}
QPushButton {{ background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.25);
              border-radius: 16px; padding: 8px 18px; font-weight: 600; }}
QPushButton:hover {{ background: rgba(255, 255, 255, 0.16); }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#model {{ border-radius: 14px; padding: 6px; text-align: left; }}
QPushButton#model:checked {{ background: rgba(60, 46, 14, 0.55); border: 2px solid {GOLD}; }}
QLabel#key {{ background: rgba(255, 255, 255, 0.2); border-radius: 13px; font-weight: 700; }}
QLabel#modelTitle {{ font-weight: 600; font-size: 14px; }}
QLabel#flag {{ color: {SOFT_RED}; border: 1px solid #e05a50; background: rgba(179, 38, 30, 0.25);
              border-radius: 10px; padding: 2px 9px; font-size: 11px; font-weight: 700; }}
QLabel#sectionTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#small {{ color: {MUTED}; font-size: 11px; }}
QLabel#big {{ font-size: 22px; font-weight: 700; }}
QLabel#liveClass {{ font-size: 13px; font-weight: 700; }}
QLabel#step {{ background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.18);
              border-radius: 10px; padding: 5px 3px; font-size: 11px; color: {MUTED}; }}
QLabel#step[state="ok"] {{ background: rgba(140, 233, 154, 0.12); border: 1px solid rgba(140, 233, 154, 0.5);
                          color: {INK}; }}
QLabel#step[state="fail"] {{ background: rgba(216, 68, 58, 0.18); border: 1px solid #e05a50; color: {INK}; }}
QLabel#active {{ color: {GOLD}; border: 1px solid {GOLD}; border-radius: 10px; padding: 2px 9px;
                font-size: 11px; font-weight: 700; }}
"""


class GlassBackground(QWidget):
    """Window backdrop: the live camera image, blurred and dimmed, so the panels look frosted.
    Before the first frame arrives it is a dark grey gradient."""

    def __init__(self):
        super().__init__()
        self.backdrop: QPixmap | None = None

    def set_backdrop(self, rgb: np.ndarray) -> None:
        small = cv2.resize(rgb, (192, 128), interpolation=cv2.INTER_AREA)
        small = np.ascontiguousarray(cv2.GaussianBlur(small, (0, 0), 6))
        image = QImage(small.data, small.shape[1], small.shape[0], small.strides[0], QImage.Format_RGB888)
        self.backdrop = QPixmap.fromImage(image.copy())
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        if self.backdrop is None:
            base = QLinearGradient(0, 0, w, h)
            base.setColorAt(0, QColor("#4a4d55"))
            base.setColorAt(1, QColor("#25272c"))
            painter.fillRect(self.rect(), base)
        else:  # scale to cover the whole window, then dim so white text stays readable
            scale = max(w / self.backdrop.width(), h / self.backdrop.height())
            bw, bh = self.backdrop.width() * scale, self.backdrop.height() * scale
            painter.drawPixmap(int((w - bw) / 2), int((h - bh) / 2), int(bw), int(bh), self.backdrop)
            painter.fillRect(self.rect(), QColor(10, 10, 12, 165))
        painter.end()


def set_alert(widget: QWidget, on: bool, name: str = "alert") -> None:
    """Switch a widget between style states (alert = red, top = highlighted class)."""
    if widget.property(name) != on:
        widget.setProperty(name, on)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def card(title: str, tag: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    """A glass panel with a title (and an optional tag on the same line, so no extra height)."""
    frame = QFrame(objectName="card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 16)
    layout.setSpacing(8)
    heading = QHBoxLayout()
    heading.addWidget(QLabel(title, objectName="cardTitle"))
    heading.addStretch()
    if tag is not None:
        heading.addWidget(tag)
    layout.addLayout(heading)
    return frame, layout


# Colours per class, the same everywhere on the screen.
CLASS_COLOURS = {
    "upright": "#c9ced8",
    "leaning_left": "#e3b341",
    "leaning_right": "#ff9f5a",
    "leaning_forward": "#7fb2ff",
    "leaning_backward": "#b48cff",
}
MINORITY = ("leaning_left", "leaning_right")    # the classes reduced in conditions 2 and 3
TIMELINE_S = 60.0                               # length of the posture timeline


class StackBar(QWidget):
    """One horizontal bar split into coloured parts (class shares of the training data)."""

    def __init__(self, shares: dict, height: int = 11):
        super().__init__()
        self.shares = shares
        self.setFixedHeight(height)
        self.setToolTip("\n".join(f"{DISPLAY_NAME[cls]}  {100 * share:.1f} %" for cls, share in shares.items()))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 4, 4)
        painter.setClipPath(path)
        x = 0.0
        for cls, share in self.shares.items():
            w = share * self.width()
            painter.fillRect(int(round(x)), 0, int(round(x + w)) - int(round(x)), self.height(),
                             QColor(CLASS_COLOURS[cls]))
            x += w
        painter.end()


class ValueBar(QWidget):
    """A thin bar for a value between 0 and 1 (recall), drawn in one colour."""

    def __init__(self, value: float, colour: str):
        super().__init__()
        self.value, self.colour = value, colour
        self.setFixedHeight(6)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 26))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 3, 3)
        painter.setBrush(QColor(self.colour))
        painter.drawRoundedRect(0, 0, int(self.width() * max(0.0, min(self.value, 1.0))), self.height(), 3, 3)
        painter.end()


class RingTimer(QWidget):
    """Held time of the current posture as a ring that fills up to the alert limit."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(92, 92)
        self.seconds, self.limit, self.colour = 0.0, HOLD_LIMIT_S, OK_GREEN
        self.text = "-"

    def set_state(self, seconds: float, colour: str, text: str) -> None:
        if (seconds, colour, text) != (self.seconds, self.colour, self.text):
            self.seconds, self.colour, self.text = seconds, colour, text
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        pen = painter.pen()
        pen.setWidth(9)
        pen.setCapStyle(Qt.RoundCap)
        pen.setColor(QColor(255, 255, 255, 30))
        painter.setPen(pen)
        painter.drawEllipse(rect)
        fraction = max(0.0, min(self.seconds / self.limit, 1.0))
        if fraction > 0:
            pen.setColor(QColor(self.colour))
            painter.setPen(pen)
            painter.drawArc(rect, 90 * 16, int(-fraction * 360 * 16))
        painter.setPen(QColor(INK))
        font = painter.font()
        font.setPointSize(15)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text)
        painter.end()


class Timeline(QWidget):
    """Smoothed posture of the last TIMELINE_S seconds as a coloured strip; beeps are marked."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(58)
        self.samples: deque = deque()          # (time, class or None while not monitoring)
        self.beeps: deque = deque()
        self.now = 0.0

    def add(self, now: float, posture: str | None, beeped: bool) -> None:
        self.now = now
        self.samples.append((now, posture))
        if beeped:
            self.beeps.append(now)
        while self.samples and now - self.samples[0][0] > TIMELINE_S:
            self.samples.popleft()
        while self.beeps and now - self.beeps[0] > TIMELINE_S:
            self.beeps.popleft()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, bar_h = self.width(), 24
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, bar_h, 7, 7)
        painter.fillPath(path, QColor(255, 255, 255, 18))
        painter.setClipPath(path)
        x_of = lambda t: w * (1 - (self.now - t) / TIMELINE_S)  # noqa: E731
        samples = list(self.samples)
        for (t0, cls), (t1, _) in zip(samples, samples[1:] + [(self.now, None)]):
            if cls in CLASS_COLOURS:
                painter.fillRect(int(x_of(t0)), 0, max(1, int(x_of(t1)) - int(x_of(t0)) + 1), bar_h,
                                 QColor(CLASS_COLOURS[cls]))
        painter.setClipping(False)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(SOFT_RED))
        for t in self.beeps:
            x = int(x_of(t))
            painter.drawLine(x, 0, x, bar_h + 4)
            label_x = min(max(x - 20, 0), w - 40)          # keep the label inside the widget
            painter.drawText(label_x, bar_h + 4, 40, 14, Qt.AlignCenter, "beep")
        painter.setPen(QColor(255, 255, 255, 150))
        for i, label in enumerate(("-60 s", "-45 s", "-30 s", "-15 s", "now")):
            painter.drawText(int(i * (w - 40) / 4), bar_h + 16, 40, 14,
                             Qt.AlignLeft if i < 4 else Qt.AlignRight, label)
        painter.end()


class PipelineStep(QLabel):
    """One box of the live pipeline: green = done, red = stopped here, grey = not reached."""

    def __init__(self, title: str):
        super().__init__(objectName="step")
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setTextFormat(Qt.RichText)
        self.setMinimumWidth(62)
        self.set_state("idle", "-")

    def set_state(self, state: str, detail: str) -> None:
        colour = {"ok": OK_GREEN, "fail": SOFT_RED, "idle": MUTED}[state]
        text = f"<b style='color:{colour}'>{self.title}</b><br>{detail}"
        if text != self.text():
            self.setText(text)
            self.setProperty("state", state)
            self.style().unpolish(self)
            self.style().polish(self)


def rounded(pixmap: QPixmap, radius: float = 16) -> QPixmap:
    """The camera image with rounded corners, to match the panels."""
    out = QPixmap(pixmap.size())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return out


class MainWindow(QMainWindow):
    """One dashboard: camera with the live pipeline, current posture, class probabilities, the class imbalance
    experiment (three training conditions on the same frame) and a posture timeline."""

    def __init__(self, models: dict, camera_index: int):
        super().__init__()
        self.setWindowTitle("Sitting Posture Classifier")
        self.models = models
        self.stats = condition_stats(models)
        self.state = PostureState()
        self.presence = PresenceState()         # is a real person in front of the camera?
        self.session = Session()                # monitoring only after a passed setup check
        self.camera_error = ""                  # kept visible even when the model is switched

        self.root = root = GlassBackground()
        self.frame_count = 0
        self.last_frame = None                  # for the frames-per-second estimate
        self.fps = self.ms = 0.0
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 14, 22, 14)
        outer.setSpacing(10)
        # (the application name is shown in the window title bar, so no extra title row)

        status = QFrame(objectName="glass")
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(18, 10, 18, 10)
        self.status_dot = QLabel(objectName="dot")
        self.status_dot.setFixedSize(12, 12)
        self.status_text = QLabel(objectName="statusText")
        # Long texts must never widen the window: these labels give up width instead of asking for more.
        self.status_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_text.setMinimumWidth(240)
        self.setup_chip = QLabel("Setup check · not run", objectName="chip")
        status_row.addWidget(self.status_dot)
        status_row.addSpacing(6)
        status_row.addWidget(self.status_text, 1)          # takes the free width, never more
        self.perf_chip = QLabel("- fps", objectName="chip")
        self.perf_chip.setToolTip("Frames per second shown, and the time for MediaPipe + the three models")
        self.person_chip = QLabel("○ No person", objectName="chip")
        status_row.addWidget(self.person_chip)
        status_row.addWidget(self.perf_chip)
        status_row.addWidget(self.setup_chip)
        outer.addWidget(status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        outer.addLayout(grid, 1)
        grid.addLayout(self.build_camera_column(), 0, 0, 3, 1)
        grid.addLayout(self.build_middle_column(), 0, 1)
        grid.addWidget(self.build_experiment_card(), 0, 2, 3, 1, alignment=Qt.AlignTop)  # ends after its notes
        grid.addWidget(self.build_timeline_card(), 1, 1)       # under the middle column, fits small screens
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 5)
        grid.setRowStretch(2, 1)            # spare height goes below the timeline, not between the cards
        self.setCentralWidget(root)
        self.setStyleSheet(STYLE)

        for i, key in enumerate(MODEL_KEYS, start=1):
            QShortcut(QKeySequence(str(i)), self, activated=lambda key=key: self.select_model(key))

        self.worker = CameraWorker(models, camera_index)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status.connect(self.show_worker_status)
        self.worker.failed.connect(self.show_worker_error)
        self.select_model("model3")
        self.worker.start()

    # -- layout ---------------------------------------------------------------------------------
    def build_camera_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(14)
        self.camera_card, layout = card("Front Camera")
        self.badge = QLabel("Starting camera ...", objectName="badge")
        self.badge.setAlignment(Qt.AlignCenter)
        self.video = QLabel()
        self.video.setMinimumSize(360, 200)
        # The layout must not follow the picture's size, otherwise the image grows and shrinks between frames.
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.badge, alignment=Qt.AlignHCenter)
        layout.addWidget(self.video, 1)

        heading = QHBoxLayout()
        heading.addWidget(QLabel("Live pipeline", objectName="sectionTitle"))
        heading.addWidget(QLabel("this frame", objectName="caption"))
        heading.addStretch()
        layout.addLayout(heading)
        steps = QHBoxLayout()
        steps.setSpacing(3)
        self.steps = {}
        for i, (key, title) in enumerate((("camera", "Camera"), ("pose", "Pose"), ("person", "Person"),
                                          ("features", "Features"), ("classifier", "Classifier"),
                                          ("posture", "Posture"))):
            if i:
                arrow = QLabel("›", objectName="caption")
                steps.addWidget(arrow)
            self.steps[key] = PipelineStep(title)
            steps.addWidget(self.steps[key], 2 if key in ("classifier", "posture") else 1)   # room for long names
        self.steps["camera"].setToolTip("Webcam frame, not mirrored, centred crop for display")
        self.steps["pose"].setToolTip("MediaPipe Pose Landmarker\nFallback CLAHE + 25% border, only if no pose is found")
        self.steps["person"].setToolTip(f"Person check\nBoth shoulders visible and shoulder width at least "
                                        f"{MIN_SHOULDER_WIDTH} of the image height")
        self.steps["features"].setToolTip("x, y, z of landmarks 0-12 = 39 values")
        self.steps["classifier"].setToolTip(f"StandardScaler + {CLASSIFIER} (active training condition)\n"
                                            "Selected pipeline, best of six classifiers and a fine-tuned CNN "
                                            "(notebook Sections 4 and 5)")
        layout.addLayout(steps)
        column.addWidget(self.camera_card, 1)

        setup_box = QFrame(objectName="glass")
        setup_layout = QVBoxLayout(setup_box)
        setup_layout.setContentsMargins(18, 12, 18, 14)
        row = QHBoxLayout()
        self.setup_button = QPushButton("Setup check")
        self.setup_button.setFixedWidth(150)
        self.setup_button.clicked.connect(self.start_setup_check)
        self.setup_bar = QProgressBar(objectName="setupBar", maximum=100, textVisible=True)
        self.setup_bar.setFormat("Checking camera position ...  %p%")
        self.setup_result = QLabel("Sit upright, then press Setup check\nIt checks the camera position only",
                                   objectName="hint")
        self.setup_result.setWordWrap(True)
        row.addWidget(self.setup_button)
        row.addSpacing(8)
        row.addWidget(self.setup_bar, 1)
        row.addWidget(self.setup_result, 1)
        self.setup_bar.hide()
        setup_layout.addLayout(row)
        column.addWidget(setup_box)
        return column

    def build_middle_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(14)
        posture_card, layout = card("Current Posture")
        top = QHBoxLayout()
        self.ring = RingTimer()
        top.addWidget(self.ring)
        top.addSpacing(10)
        words = QVBoxLayout()
        words.addStretch()
        self.big_posture = QLabel("-", objectName="big")
        words.addWidget(self.big_posture)
        words.addWidget(QLabel(f"Alert and beep after {HOLD_LIMIT_S:g} s", objectName="caption"))
        words.addStretch()
        top.addLayout(words, 1)
        layout.addLayout(top)
        held = QGridLayout()
        held.setHorizontalSpacing(16)
        held.setVerticalSpacing(2)
        self.held_rows = {}
        for i, cls in enumerate(("leaning_forward", "leaning_backward", "leaning_left", "leaning_right")):
            name = QLabel(SHOWN_AS[cls], objectName="small")
            value = QLabel("0 s", objectName="small")
            value.setAlignment(Qt.AlignRight)
            bar = QProgressBar(maximum=100, textVisible=False)
            r, c = divmod(i, 2)
            held.addWidget(name, 2 * r, 2 * c)
            held.addWidget(value, 2 * r, 2 * c + 1)
            held.addWidget(bar, 2 * r + 1, 2 * c, 1, 2)
            self.held_rows[cls] = (name, value, bar)
        layout.addLayout(held)
        column.addWidget(posture_card)

        proba_card, layout = card("Class Probabilities",
                                  QLabel("Logistic Regression · active condition", objectName="caption"))
        self.proba_rows = {}
        for cls in DISPLAY_ORDER:
            row = QHBoxLayout()
            name, value = QLabel(DISPLAY_NAME[cls]), QLabel("-")
            name.setToolTip(dataset_tip(cls))
            name.setFixedWidth(130)
            value.setFont(QFont("Consolas"))
            value.setFixedWidth(40)
            value.setAlignment(Qt.AlignRight)
            bar = QProgressBar(maximum=100, textVisible=False)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {CLASS_COLOURS[cls]}; border-radius: 4px; }}")
            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(value)
            layout.addLayout(row)
            self.proba_rows[cls] = (name, value, bar)
        note = QLabel("The highest class is the posture\nShown from the user's point of view",
                      objectName="hint")
        note.setToolTip("The dataset classes follow the camera view (image not mirrored)\n"
                        "Dataset leaning_left = Leaning right here · dataset leaning_right = Leaning left here")
        layout.addWidget(note)
        column.addWidget(proba_card)
        column.addStretch()
        return column

    def condition_button(self, number: int, key: str) -> tuple[QPushButton, QLabel, QLabel, QLabel]:
        """One training condition: name, training-data composition, held-out recall and this frame's result."""
        name, _, description = CONDITIONS[key]
        stats = self.stats[key]
        button = QPushButton(objectName="model", checkable=True)
        button.setToolTip(description)
        outer = QHBoxLayout(button)
        outer.setContentsMargins(12, 8, 12, 8)
        left = QVBoxLayout()
        left.setSpacing(2)
        head = QHBoxLayout()
        badge = QLabel(str(number), objectName="key")
        badge.setFixedSize(22, 22)
        badge.setAlignment(Qt.AlignCenter)
        title = QLabel(name, objectName="modelTitle")
        active = QLabel("ACTIVE", objectName="active")
        head.addWidget(badge)
        head.addWidget(title)
        head.addStretch()
        left.addLayout(head)
        counts = stats["counts"]
        if counts is None:                        # training_counts.csv missing: plain description only
            left.addWidget(QLabel(description, objectName="caption", wordWrap=True))
        else:
            order = DISPLAY_ORDER
            if stats["weights"] is not None:
                influence = {c: float(counts[c] * stats["weights"][c]) for c in order}
                w = stats["weights"]
                caption = f"Class weights · minority ×{w['leaning_left']:.0f} · upright ×{w['upright']:.2f}"
            else:
                influence = {c: float(counts[c]) for c in order}
                cut = counts["leaning_left"] < 0.05 * counts["upright"]
                caption = (f"Training data · {int(counts.sum()):,} frames"
                           + (f" · minority cut to {int(counts['leaning_left'])} each" if cut else ""))
            total = sum(influence.values())
            left.addWidget(QLabel(caption, objectName="small"))
            left.addWidget(StackBar({c: v / total for c, v in influence.items()}))
        if stats["lean_recall"] is not None:
            tip = ("Recall on held-out participants, mean over seeds (results/demo_imbalance_check.csv)\n"
                   "minority = mean of Leaning left and Leaning right")
            for label, value, colour in (
                    ("Held-out minority recall", stats["lean_recall"],
                     GOLD if stats["lean_recall"] >= 0.9 else SOFT_RED),
                    ("Held-out upright recall", stats["upright_recall"], CLASS_COLOURS["upright"])):
                row = QHBoxLayout()
                row.setSpacing(6)
                text = QLabel(label, objectName="small")
                text.setFixedWidth(140)
                text.setToolTip(tip)
                number_label = QLabel(f"{value:.2f}", objectName="small")
                number_label.setFixedWidth(28)
                row.addWidget(text)
                row.addWidget(ValueBar(value, colour), 1)
                row.addWidget(number_label)
                left.addLayout(row)
        outer.addLayout(left, 1)

        line = QFrame()
        line.setFixedWidth(1)
        line.setStyleSheet("background: rgba(255, 255, 255, 0.18);")
        outer.addSpacing(8)
        outer.addWidget(line)
        outer.addSpacing(8)
        right = QVBoxLayout()
        right.addWidget(active, alignment=Qt.AlignLeft)
        right.addStretch()
        right.addWidget(QLabel("This frame", objectName="small"))
        live_class = QLabel("-", objectName="liveClass")
        live_p = QLabel("", objectName="small")
        right.addWidget(live_class)
        right.addWidget(live_p)
        right.addStretch()
        holder = QWidget()
        holder.setLayout(right)
        holder.setFixedWidth(118)
        outer.addWidget(holder)
        for widget in (*button.findChildren(QLabel), *button.findChildren(QWidget)):
            widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        button.setMinimumHeight(outer.sizeHint().height())    # a button does not grow with its inner layout
        self.condition_live[key] = (live_class, live_p)
        return button, active, title, badge

    def build_experiment_card(self) -> QFrame:
        self.disagree_tag = QLabel("Conditions disagree", objectName="flag")
        self.disagree_tag.hide()
        frame, layout = card("Class Imbalance Experiment", self.disagree_tag)
        legend = QLabel("&nbsp;&nbsp; ".join(
            f"<span style='color:{CLASS_COLOURS[cls]}'>■</span>&nbsp;{DISPLAY_NAME[cls]}"
            for cls in DISPLAY_ORDER),
            objectName="small", wordWrap=True)                  # wraps instead of widening the window
        layout.addWidget(legend)
        minority = QHBoxLayout()
        minority.addWidget(QLabel("Minority classes", objectName="small"))
        for cls in sorted(MINORITY, key=DISPLAY_ORDER.index):
            chip = QLabel(DISPLAY_NAME[cls], objectName="small")
            chip.setToolTip(dataset_tip(cls))
            chip.setStyleSheet(f"color: {CLASS_COLOURS[cls]}; border: 1px solid {CLASS_COLOURS[cls]}; "
                               "border-radius: 8px; padding: 0px 6px;")
            minority.addWidget(chip)
        minority.addWidget(QLabel("reduced in conditions 2 and 3", objectName="small"))
        minority.addStretch()
        layout.addLayout(minority)
        self.model_buttons, self.condition_live = {}, {}
        for i, key in enumerate(MODEL_KEYS, start=1):
            button, active, title, number = self.condition_button(i, key)
            button.clicked.connect(lambda _=False, key=key: self.select_model(key))
            layout.addWidget(button)
            self.model_buttons[key] = (button, active, title, number)
        notes = QLabel("Press 1 · 2 · 3 to switch the active condition\n"
                       "Same camera, features and classifier in all three\n"
                       "Only the training data differs", objectName="hint")
        notes.setToolTip(f"Classifier · {CLASSIFIER}\nRecall · held-out participants, mean over seeds")
        layout.addWidget(notes)
        return frame

    def build_timeline_card(self) -> QFrame:
        frame, layout = card("Posture Timeline",
                             QLabel("last 60 s · active condition", objectName="caption"))
        self.timeline = Timeline()
        layout.addWidget(self.timeline)
        return frame

    # -- behaviour ------------------------------------------------------------------------------
    def show_worker_status(self, message: str) -> None:
        if message:
            self.badge.setText(message)

    def show_worker_error(self, message: str) -> None:
        self.camera_error = message
        self.badge.setText("Camera problem")
        set_alert(self.badge, True)
        self.update_status()

    def select_model(self, key: str) -> None:
        self.worker.active_key = key
        self.state = PostureState()          # timers restart with the new model
        for k, (button, active, title, number) in self.model_buttons.items():
            on = k == key
            button.setChecked(on)
            active.setVisible(on)
            title.setStyleSheet(f"color: {GOLD};" if on else "")
            number.setStyleSheet(f"color: {GOLD}; border: 1px solid {GOLD};" if on else "")
        self.update_status()

    def update_status(self) -> None:
        number = MODEL_KEYS.index(self.worker.active_key) + 1
        name = CONDITIONS[self.worker.active_key][0].lower()
        mode = "Monitoring" if self.session.monitoring else MODE_STATUS[self.session.mode]
        self.status_text.setText(self.camera_error or f"{mode}  ·  Condition {number} · {name}")
        self.status_text.setToolTip(self.camera_error or self.session.message)
        set_alert(self.status_dot, bool(self.camera_error) or not self.session.monitoring)

    def set_setup_result(self, text: str, colour: str) -> None:
        self.setup_result.setText(text)
        self.setup_result.setStyleSheet(f"color: {colour}; font-size: 13px; font-weight: 600;")
        self.setup_chip.setText("Setup check · OK" if colour == OK_GREEN else "Setup check · not passed")
        self.setup_chip.setToolTip(text)                     # full reason on hover, short chip on screen
        self.setup_chip.setStyleSheet(f"color: {colour}; font-weight: 600;")

    def start_setup_check(self) -> None:
        self.session.start_setup(time.monotonic())
        self.state = PostureState()              # monitoring stops during the check
        self.update_status()
        self.setup_bar.setValue(0)
        self.setup_bar.show()
        self.setup_result.hide()
        self.setup_button.setEnabled(False)
        self.setup_chip.setText("Setup check · running")
        self.setup_chip.setStyleSheet("")

    def finish_setup_check(self) -> None:
        """Show the result of the setup check that the session has just finished."""
        self.setup_bar.hide()
        self.setup_result.show()
        self.setup_button.setEnabled(True)
        if self.session.monitoring:
            self.set_setup_result(f"✓ {self.session.message}", OK_GREEN)
        else:
            self.set_setup_result(self.session.message, AMBER)

    def mode_changed(self, before: str) -> None:
        """React to a change of session mode (setup result, fresh timers, status line)."""
        mode = self.session.mode
        if before == "checking":
            self.finish_setup_check()
        if mode == "monitoring" or before == "monitoring":
            self.state = PostureState()          # every monitoring period starts from zero
        if mode == "preview" and before in ("paused", "rechecking"):
            self.set_setup_result(self.session.message, AMBER)
        self.update_status()

    def show_frame(self, result: FrameResult) -> None:
        now = time.monotonic()
        self.update_speed(result, now)
        valid = result.frame_status == "valid"
        if self.presence.update(valid, now) and not self.presence.present:
            self.state = PostureState()          # confirmed absence: clean start for the returning user
        before = self.session.mode
        self.session.update(valid, result.shoulder_ratio, self.presence.present, now)
        if self.session.mode != before:
            self.mode_changed(before)
        # classify only while monitoring, with a confirmed person and a valid current frame
        monitoring = self.session.monitoring and self.presence.present and valid
        beeped = False
        if monitoring:
            beeped = self.state.update(result.predicted, now)
            if beeped:
                beep()
        else:
            self.state.pause(now)                # no new vote, timers stop, no beep
        probabilities = result.probabilities if monitoring else None
        all_probabilities = result.all_probabilities if monitoring else None
        if self.session.mode == "checking":
            progress = (now - self.session.started) / SETUP_S
            self.setup_bar.setValue(int(min(progress, 1.0) * 100))

        image = np.ascontiguousarray(result.image)
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.frame_count += 1
        if self.frame_count % BACKDROP_EVERY == 1:
            self.root.set_backdrop(image)
        self.video.setPixmap(rounded(QPixmap.fromImage(qimage.copy()).scaled(
            self.video.width(), self.video.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)))

        posture, alert = self.state.posture, self.state.alert
        held = self.state.held(posture, now)
        if not self.presence.present:
            posture = "no_pose"                  # badge: "No person detected"
        if not self.session.monitoring and self.presence.present:
            text = MODE_BADGE[self.session.mode]             # e.g. "Setup required"
        else:
            text = SHOWN_AS[posture]
        self.badge.setText(text + (f"   {held:.0f} s" if posture in NON_UPRIGHT else ""))
        self.big_posture.setText(text)
        set_alert(self.badge, alert)
        set_alert(self.camera_card, alert)
        if self.session.monitoring and posture in NON_UPRIGHT:
            self.ring.set_state(held, SOFT_RED if alert else AMBER, f"{held:.0f} s")
        elif self.session.monitoring and posture == "upright":
            self.ring.set_state(0.0, OK_GREEN, "OK")
        else:
            self.ring.set_state(0.0, MUTED, "-")
        self.person_chip.setText("● Person detected" if self.presence.present else "○ No person")

        for cls, (name, value, bar) in self.held_rows.items():
            seconds = self.state.held(cls, now)
            value.setText(f"{seconds:.0f} s")
            bar.setValue(int(min(seconds / HOLD_LIMIT_S, 1.0) * 100))
            hot = alert and cls == posture
            set_alert(bar, hot)
            name.setStyleSheet("font-weight: 700;" if hot else "")

        for cls, (name, value, bar) in self.proba_rows.items():
            p = None if probabilities is None else float(probabilities[cls])
            value.setText("-" if p is None else f"{p:.2f}")
            bar.setValue(0 if p is None else int(p * 100))
            top = monitoring and cls == result.predicted
            name.setStyleSheet(f"color: {ACCENT}; font-weight: 600;" if top else "")

        choices = {} if all_probabilities is None else {
            key: max(probs, key=probs.get) for key, probs in all_probabilities.items()}
        counts = Counter(choices.values())
        self.disagree_tag.setVisible(len(counts) > 1)
        for key, (live_class, live_p) in self.condition_live.items():
            if key not in choices:
                live_class.setText("-")
                live_class.setStyleSheet(f"color: {MUTED};")
                live_p.setText("")
                continue
            cls = choices[key]
            odd = len(counts) > 1 and counts[cls] == 1      # the condition that differs
            live_class.setText(DISPLAY_NAME[cls] + ("  ●" if odd else ""))
            live_class.setStyleSheet(f"color: {SOFT_RED if odd else CLASS_COLOURS[cls]};")
            live_p.setText(f"p = {all_probabilities[key][cls]:.2f}")

        self.update_pipeline(result, monitoring, probabilities)
        self.timeline.add(now, self.state.posture if monitoring or (self.session.monitoring and
                                                                    self.presence.present) else None, beeped)

    def update_pipeline(self, result: FrameResult, monitoring: bool, probabilities: dict | None) -> None:
        """Colour the live-pipeline boxes for this frame: green = done, red = stopped here, grey = not reached."""
        height, width = self.worker.frame_size
        status = result.frame_status
        self.steps["camera"].set_state("ok", f"{width}x{height}")
        self.steps["pose"].set_state("fail" if status == "no_pose" else "ok",
                                     "none found" if status == "no_pose" else "13 points")
        if status == "valid":
            self.steps["person"].set_state("ok", f"width {result.shoulder_ratio:.2f}")
        elif status == "invalid_pose":
            self.steps["person"].set_state("fail", "shoulders hidden" if not result.shoulders_visible
                                           else f"width {result.shoulder_ratio:.2f}")
        else:
            self.steps["person"].set_state("idle", "-")
        self.steps["features"].set_state("ok" if status == "valid" else "idle",
                                         "39 values" if status == "valid" else "-")
        if monitoring:
            self.steps["classifier"].set_state("ok", "Logistic<br>Regression")
            self.steps["posture"].set_state(
                "ok", f"{DISPLAY_NAME[result.predicted]}<br>p {probabilities[result.predicted]:.2f}")
        else:
            waiting = "not monitoring" if status == "valid" else "-"
            self.steps["classifier"].set_state("idle", waiting)
            self.steps["posture"].set_state("idle", "-")


    def update_speed(self, result: FrameResult, now: float) -> None:
        """Smoothed frames per second and processing time per frame."""
        if self.last_frame is not None:
            fps = 1.0 / max(now - self.last_frame, 1e-6)
            self.fps = fps if self.fps == 0 else 0.9 * self.fps + 0.1 * fps
        self.last_frame = now
        self.ms = result.process_ms if self.ms == 0 else 0.9 * self.ms + 0.1 * result.process_ms
        if self.frame_count % 10 == 0:
            self.perf_chip.setText(f"{self.fps:.0f} fps  ·  {self.ms:.0f} ms per frame")

    def closeEvent(self, event) -> None:
        self.worker.stop()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sitting Posture Classifier (IPCV demo).")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(load_models(), args.camera)
    # open at most as large as the usable screen area (taskbar excluded); scaling-aware via Qt
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1500, available.width()), min(920, available.height() - 40))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()