"""Extract still frames from the own posture videos (recorded with record_session.py).

Two sets of takes, processed the same way: the first and last 2 seconds of every take are
skipped (the participant is still moving into or out of the posture), then 2 frames per second
are saved as JPG, without crop or resize.

  --set train (default)  the 9 training takes in own_videos\\Pxx\\, mapped to the 5 classes
                         -> OWN_FRAMES_ROOT\\<class>\\Pxx_<take>_<frame>.jpg, data/manifest_own.csv
  --set test             the webcam test takes (headtilt_left, headtilt_right, head_down) in
                         own_videos\\test_takes\\Pxx\\ (P31-P102). NOT training data.
                         -> OWN_TEST_FRAMES_ROOT\\<take>\\Pxx_<take>_<frame>.jpg,
                            data/manifest_own_test.csv (class = take name)
Phone takes (.mov, side and floor views) are for the FYP only and are ignored.

Each manifest is rebuilt from the frames on disk, with the same columns as data/manifest.csv
plus "take", so later scripts can read all of them the same way.

Run:
    python extract_frames.py                     training takes, all participants
    python extract_frames.py --person P01        training takes, one participant (test run)
    python extract_frames.py --set test          webcam test takes, all participants
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import OWN_FRAMES_ROOT, OWN_TEST_FRAMES_ROOT, OWN_VIDEO_ROOT

# Training take -> model class. Left/right follow the CAMERA view, as in Roboflow.
TAKE_CLASS = {
    "upright": "upright",
    "upright_turning": "upright",
    "upright_moving": "upright",
    "leaning_left": "leaning_left",
    "leaning_left_supported": "leaning_left",
    "leaning_right": "leaning_right",
    "leaning_right_supported": "leaning_right",
    "leaning_forward": "leaning_forward",
    "leaning_backward": "leaning_backward",
}
TEST_TAKES = ("headtilt_left", "headtilt_right", "head_down")
SKIP_SECONDS = 2.0      # trimmed at the start and at the end of every take
FRAMES_PER_SECOND = 2   # more frames from one clip add near-duplicates, not information
JPEG_QUALITY = 95
MANIFEST_OWN = Path("data") / "manifest_own.csv"
MANIFEST_OWN_TEST = Path("data") / "manifest_own_test.csv"
PERSON_PATTERN = re.compile(r"^P\d{2,3}$")   # P01 ... P102

# set -> (video folder, take -> output folder name, frames root, manifest)
SETS = {
    "train": (Path(OWN_VIDEO_ROOT), TAKE_CLASS, OWN_FRAMES_ROOT, MANIFEST_OWN),
    "test": (Path(OWN_VIDEO_ROOT) / "test_takes", {t: t for t in TEST_TAKES},
             OWN_TEST_FRAMES_ROOT, MANIFEST_OWN_TEST),
}


def person_key(name):
    """Sort P2 < P10 < P100 by number; other folder names go last."""
    return (0, int(name[1:])) if PERSON_PATTERN.match(name) else (1, name)


class FrameExtractor:
    """Saves evenly spaced frames of one video into one output folder."""

    def __init__(self, out_root, skip_seconds=SKIP_SECONDS, per_second=FRAMES_PER_SECOND):
        self.out_root = Path(out_root)
        self.skip_seconds = skip_seconds
        self.per_second = per_second

    def target_frames(self, n_frames, fps):
        """Frame numbers to keep: one every 1/per_second s inside the trimmed window."""
        start = self.skip_seconds
        end = n_frames / fps - self.skip_seconds
        times = np.arange(start, end, 1 / self.per_second)
        return {int(round(t * fps)) for t in times}

    def extract(self, video, person, take, folder_name):
        """Save the selected frames of one video; returns how many were saved."""
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or n_frames <= 0:
            print(f"    cannot read {video.name}")
            cap.release()
            return 0

        folder = self.out_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        # Remove frames left from an earlier run of this exact take (e.g. re-recorded video).
        for old in folder.glob(f"{person}_{take}_[0-9][0-9][0-9][0-9].jpg"):
            old.unlink()

        keep = self.target_frames(n_frames, fps)
        saved, index = 0, 0
        while keep and index <= max(keep):
            ok, frame = cap.read()        # read in order: more reliable than seeking in mp4
            if not ok:
                break
            if index in keep:
                path = folder / f"{person}_{take}_{index:04d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                saved += 1
            index += 1
        cap.release()
        return saved


def find_videos(video_root, takes, only_person=None):
    """Yield (video, person, take) for every expected take and report odd or missing files."""
    for person_dir in sorted(video_root.iterdir(), key=lambda p: person_key(p.name)):
        person = person_dir.name
        if not person_dir.is_dir() or not PERSON_PATTERN.match(person):
            continue                      # skips test_takes and recordings_log.csv
        if only_person and person != only_person:
            continue
        found = set()
        for video in sorted(person_dir.glob("*.mp4")):
            take = video.stem.removeprefix(f"{person}_")
            if not video.stem.startswith(f"{person}_") or take not in takes:
                print(f"    skipped, unexpected name: {person}\\{video.name}")
                continue
            found.add(take)
            yield video, person, take
        missing = sorted(set(takes) - found)
        if missing:
            print(f"    {person}: missing takes {missing}")


def build_manifest(frames_root, manifest_path):
    """One row per frame on disk, same columns as data/manifest.csv plus 'take'."""
    rows = []
    for path in sorted(Path(frames_root).glob("*/*.jpg")):
        name, frame = path.stem.rsplit("_", 1)        # "P01_leaning_left", "0060"
        person, take = name.split("_", 1)             # "P01", "leaning_left"
        rows.append({
            "path": str(path.relative_to(frames_root)),
            "split": "own", "class": path.parent.name, "original_name": name,
            "source": "own", "n_copies": 1, "frame": int(frame),
            "session_id": person, "person": person, "group": person, "take": take,
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="take_set", choices=list(SETS), default="train",
                        help="train: 9 training takes (default); test: webcam test takes")
    parser.add_argument("--person", help="only this participant, e.g. P01 (test run)")
    args = parser.parse_args()

    video_root, take_folders, frames_root, manifest_path = SETS[args.take_set]
    extractor = FrameExtractor(frames_root)
    per_person = {}
    for video, person, take in find_videos(video_root, take_folders, args.person):
        saved = extractor.extract(video, person, take, take_folders[take])
        per_person[person] = per_person.get(person, 0) + saved
        print(f"{video.name}: {saved} frames")

    print(f"\nExtracted {sum(per_person.values())} frames from {len(per_person)} participants.")
    manifest = build_manifest(frames_root, manifest_path)
    print(f"\n{manifest_path}: {len(manifest)} frames, {manifest['person'].nunique()} participants")
    print(manifest["class"].value_counts().to_string())


if __name__ == "__main__":
    main()