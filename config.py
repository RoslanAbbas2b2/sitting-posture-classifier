"""Shared project paths.

Every script reads its data locations from here, so if a dataset folder is
moved, only this file needs to change. None of these folders are stored in Git.
"""
from pathlib import Path

DATASETS = Path("datasets")  # not included in this repository

# Raw Roboflow export (Folder Structure, v5). Read-only.
RAW_ROOT = DATASETS / "roboflow_sitting_posture_v5_raw"

# Own recordings (record_session.py): own_videos\Pxx\Pxx_<take>.mp4,
# test takes in own_videos\test_takes\Pxx\. Contains faces: never publish.
OWN_VIDEO_ROOT = str(DATASETS / "own_videos")

# Frames taken from the own videos (extract_frames.py). Contains faces: never publish.
OWN_FRAMES_ROOT = DATASETS / "own_frames"            # training takes: own_frames\<class>\*.jpg
OWN_TEST_FRAMES_ROOT = DATASETS / "own_frames_test"  # webcam test takes: own_frames_test\<take>\*.jpg