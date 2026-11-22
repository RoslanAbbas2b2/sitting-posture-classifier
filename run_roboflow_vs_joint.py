"""Step 4: Roboflow-only training (v1) vs joint training (v2), evaluated on the Newly Collected Dataset.

Question: does adding many new people to the training data improve posture classification
on people the model has never seen?

Pipeline (the best pipeline on the Roboflow Posture Dataset, unchanged): fallback MediaPipe
landmarks 0-12 (x, y, z = 39 values) -> StandardScaler -> Logistic Regression with class
weighting (models.make_model("logreg+weighted")).

Folds: the 102 participants of the Newly Collected Dataset are split into 5 folds by person,
men and women spread evenly (data/participants.csv: person, gender; seed 42); saved to
data/own_folds.csv. For each fold, the test set is every frame of that fold's participants;
frames without a pose count as wrong (as in the Roboflow experiments).

Experiments (all tested on the same frames of the Newly Collected Dataset):
  v1          Roboflow-only training, cross-dataset evaluation: one model per imbalance setting
              and seed, trained on every Roboflow Posture Dataset image with a pose and tested
              on all 102 participants; the folds only organise the reporting
  v2          Joint training: for each fold, every Roboflow image with a pose + the newly
              collected participants of the other 4 folds (frames with a pose); leaning frames
              that fail the shoulder check (own_label_checks.py, train_ok = False) left out
  v2_nocheck  Joint training without the shoulder check: as v2, but newly collected leaning
              frames that fail the shoulder check stay in training (the pose requirement and
              all other rules unchanged); separates "more people" from "cleaner labels"
  v1_nocrop   Roboflow-only training, evaluated WITHOUT the 3:2 coordinate correction: the v1
              models tested on the uncorrected 16:9 coordinates (sensitivity check)

Aspect ratio: Roboflow images are 1080 x 720 (3:2), newly collected frames 1280 x 720 (16:9).
MediaPipe x is a fraction of the image width, so newly collected coordinates are re-expressed as
if the frame had been centre-cropped to 3:2: x' = (x*W - offset) / W', z' = z*W / W' (y
unchanged). Only the numbers change; no image is cropped.

Imbalance: natural and 10:1 (leaning_left/right reduced to upright/10 in the TRAINING data,
seeds 0-4), exactly as in the Roboflow experiments. The reduction is applied to the complete
training set of each model, so in joint training the leaning frames of both datasets are reduced.

Head-tilt test: webcam headtilt_left / headtilt_right frames (72 participants, body upright)
are predicted by a model that never saw that participant (v2: the model of their fold).
Reported: share of frames predicted leaning_left or leaning_right, per participant.

Outputs in results/v1_v2/: runs.csv (every run), summary.csv (mean and SD over seeds),
per_person.csv, headtilt.csv, confusion_<variant>_<setting>.csv.

Run:  python run_roboflow_vs_joint.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

from config import OWN_FRAMES_ROOT, OWN_TEST_FRAMES_ROOT
from evaluation import CLASSES, NO_POSE, compute_metrics, imbalance_settings
from extract_frames import MANIFEST_OWN, MANIFEST_OWN_TEST, person_key
from features import OWN_IMAGE_SIZES, LandmarkFeatures
from models import make_model

DATA = Path("data")
OUT = Path("results") / "v1_v2"
PARTICIPANTS = DATA / "participants.csv"
FOLDS_CSV = DATA / "own_folds.csv"
LABEL_CHECKS = DATA / "own_label_checks.csv"
N_FOLDS = 5
FOLD_SEED = 42
MODEL = "logreg+weighted"
RATIOS = ["natural", "10"]
TARGET_ASPECT = 1080 / 720
LEANS = ("leaning_left", "leaning_right")
HEADTILT_TAKES = ("headtilt_left", "headtilt_right")
COMPARISONS = [("v2", "v1"), ("v2", "v2_nocheck"), ("v1", "v1_nocrop")]


def check_order(features: LandmarkFeatures, manifest: pd.DataFrame, name: str) -> None:
    if not np.array_equal(features.paths, manifest["path"].to_numpy()):
        raise SystemExit(f"{name}: landmark CSV and manifest are not in the same order; re-run extraction.")


def centre_crop_3x2(coords: np.ndarray, sizes: pd.DataFrame) -> np.ndarray:
    """Re-express normalised x and z as if each frame were centre-cropped to 3:2."""
    width = sizes["width"].to_numpy(dtype=float)[:, None]
    height = sizes["height"].to_numpy(dtype=float)[:, None]
    new_width = height * TARGET_ASPECT
    offset = (width - new_width) / 2
    pts = coords.reshape(len(coords), -1, 3).copy()
    pts[:, :, 0] = (pts[:, :, 0] * width - offset) / new_width
    pts[:, :, 2] = pts[:, :, 2] * width / new_width
    return pts.reshape(len(coords), -1)


def assign_folds(persons: list[str]) -> pd.DataFrame:
    """Person -> fold 0-4; men and women dealt out in turn after a seeded shuffle."""
    people = pd.read_csv(PARTICIPANTS)
    missing = sorted(set(persons) - set(people["person"]), key=person_key)
    if missing:
        raise SystemExit(f"{PARTICIPANTS} has no gender for: {missing}")
    people = people[people["person"].isin(persons)].sort_values("person")
    rng = np.random.default_rng(FOLD_SEED)
    fold, i = {}, 0
    for _, g in people.groupby("gender"):
        for person in rng.permutation(g["person"].to_numpy()):
            fold[person] = i % N_FOLDS
            i += 1
    table = people.assign(fold=people["person"].map(fold))
    table = table.iloc[sorted(range(len(table)), key=lambda k: person_key(table["person"].iat[k]))]
    table.to_csv(FOLDS_CSV, index=False)
    return table


def load_roboflow() -> tuple[np.ndarray, np.ndarray]:
    manifest = pd.read_csv(DATA / "manifest.csv")
    features = LandmarkFeatures(DATA / "landmarks_fallback.csv")
    check_order(features, manifest, "roboflow")
    return features.raw_coordinates(), manifest["class"].to_numpy(dtype=object)


def load_own(landmark_csv: Path, manifest_path: Path, image_root: Path, sizes_cache: Path):
    """Manifest, raw coordinates and 3:2-corrected coordinates of one own frame set."""
    manifest = pd.read_csv(manifest_path)
    features = LandmarkFeatures(landmark_csv, image_root, sizes_cache)
    check_order(features, manifest, str(manifest_path))
    raw = features.raw_coordinates()
    return manifest, raw, centre_crop_3x2(raw, features.sizes())


def predict(model, X: np.ndarray) -> np.ndarray:
    """Predictions for every row; rows without a pose get 'no_pose'."""
    pred = np.full(len(X), NO_POSE, dtype=object)
    available = ~np.isnan(X).any(axis=1)
    pred[available] = model.predict(X[available])
    return pred


def fit(X: np.ndarray, y: np.ndarray, imbalance):
    idx = imbalance.apply(np.arange(len(y)), y)
    return make_model(MODEL).fit(X[idx], y[idx])


def paired_test(table: pd.DataFrame, value: str, a: str, b: str) -> str:
    """Median difference a - b over participants, how many improved, Wilcoxon signed-rank p."""
    diff = (table[a] - table[b]).dropna()
    p = wilcoxon(diff).pvalue if (diff != 0).any() else float("nan")
    return (f"{a} - {b}: median {value} difference {diff.median():+.3f}, higher for "
            f"{(diff > 0).sum()} of {len(diff)} participants, Wilcoxon p = {p:.2g}")


def main() -> None:
    pd.set_option("display.width", 200)
    OUT.mkdir(parents=True, exist_ok=True)

    rf_X, rf_y = load_roboflow()
    rf_X, rf_y = rf_X[~np.isnan(rf_X).any(axis=1)], rf_y[~np.isnan(rf_X).any(axis=1)]
    own, own_raw, own_X = load_own(DATA / "landmarks_own_fallback.csv", MANIFEST_OWN,
                                   OWN_FRAMES_ROOT, OWN_IMAGE_SIZES)
    test, _, test_X = load_own(DATA / "landmarks_own_test_fallback.csv", MANIFEST_OWN_TEST,
                               OWN_TEST_FRAMES_ROOT, DATA / "image_sizes_own_test.csv")
    own_y = own["class"].to_numpy(dtype=object)

    folds = assign_folds(sorted(own["person"].unique(), key=person_key))
    print("=== Folds (participants per gender) ===")
    print(pd.crosstab(folds["fold"], folds["gender"], margins=True), "\n")
    fold_of = dict(zip(folds["person"], folds["fold"]))
    own_fold = own["person"].map(fold_of).to_numpy()
    test_fold = test["person"].map(fold_of).to_numpy()

    checks = pd.read_csv(LABEL_CHECKS).set_index("path")["train_ok"]
    own_has_pose = ~np.isnan(own_X).any(axis=1)
    if not own.loc[own_has_pose, "path"].isin(checks.index).all():
        raise SystemExit("own_label_checks.csv is out of date: re-run own_label_checks.py")
    train_ok = own["path"].map(checks).fillna(True).to_numpy(dtype=bool)

    person_rows = own.groupby("person").indices
    is_tilt = test["take"].isin(HEADTILT_TAKES).to_numpy()
    runs, per_person, headtilt = [], [], []

    for ratio in RATIOS:
        for imbalance in imbalance_settings(ratio):
            setting = imbalance.name
            print(f"Running {setting} ...")
            v1 = fit(rf_X, rf_y, imbalance)
            preds = {"v1": predict(v1, own_X), "v1_nocrop": predict(v1, own_raw)}
            tilt_preds = {"v1": predict(v1, test_X)}
            for variant, use_check in (("v2", True), ("v2_nocheck", False)):
                pred = np.empty(len(own_X), dtype=object)
                tilt = np.empty(len(test_X), dtype=object)
                for fold in range(N_FOLDS):
                    use = own_has_pose & (own_fold != fold) & (train_ok | (not use_check))
                    model = fit(np.vstack([rf_X, own_X[use]]), np.concatenate([rf_y, own_y[use]]),
                                imbalance)
                    pred[own_fold == fold] = predict(model, own_X[own_fold == fold])
                    tilt[test_fold == fold] = predict(model, test_X[test_fold == fold])
                preds[variant], tilt_preds[variant] = pred, tilt

            seed = imbalance.seed if imbalance.ratio is not None else None
            for variant, pred in preds.items():
                row = {"variant": variant, "ratio": imbalance.ratio_name, "seed": seed,
                       **compute_metrics(own_y, pred)}
                for fold in range(N_FOLDS):
                    m = own_fold == fold
                    row[f"bal_acc_fold{fold}"] = compute_metrics(own_y[m], pred[m])["balanced_accuracy"]
                runs.append(row)
                for person, idx in person_rows.items():
                    per_person.append({"variant": variant, "ratio": imbalance.ratio_name, "seed": seed,
                                       "person": person, "balanced_accuracy":
                                       compute_metrics(own_y[idx], pred[idx])["balanced_accuracy"]})
                if setting in ("natural", "10to1_s0"):
                    cm = confusion_matrix(own_y, pred, labels=CLASSES + [NO_POSE])[: len(CLASSES)]
                    pd.DataFrame(cm, index=CLASSES, columns=CLASSES + [NO_POSE]).to_csv(
                        OUT / f"confusion_{variant}_{setting}.csv")

            for variant, tilt in tilt_preds.items():
                for person, g in test[is_tilt].groupby("person"):
                    p = tilt[g.index.to_numpy()]
                    p = p[p != NO_POSE]
                    headtilt.append({"variant": variant, "ratio": imbalance.ratio_name, "seed": seed,
                                     "person": person, "frames": len(p),
                                     "share_leaning": float(np.isin(p, LEANS).mean())})

    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "runs.csv", index=False)
    metrics = ["balanced_accuracy", "macro_f1", "recall_leaning_left", "recall_leaning_right",
               "recall_upright", "recall_leaning_forward", "recall_leaning_backward"]
    summary = runs.groupby(["ratio", "variant"])[metrics].agg(["mean", "std"]).round(3)
    summary.to_csv(OUT / "summary.csv")
    print("\n=== Pooled over the 102 participants of the Newly Collected Dataset "
          "(mean and SD over seeds; natural has one run) ===")
    print("v1 = Roboflow-only training, v2 = joint training; _nocrop / _nocheck = ablations")
    print(runs.groupby(["ratio", "variant"])[metrics[:4]].mean().round(3))
    print("\nSD over seeds (10:1):")
    print(runs[runs["ratio"] != "natural"].groupby("variant")[metrics[:4]].std().round(3))
    print("\n=== Balanced accuracy per fold (mean over seeds) ===")
    fold_cols = [f"bal_acc_fold{f}" for f in range(N_FOLDS)]
    print(runs.groupby(["ratio", "variant"])[fold_cols].mean().round(3))

    per_person = pd.DataFrame(per_person)
    per_person.to_csv(OUT / "per_person.csv", index=False)
    print("\n=== Per participant (balanced accuracy, mean over seeds) ===")
    for ratio, g in per_person.groupby("ratio"):
        table = g.groupby(["person", "variant"])["balanced_accuracy"].mean().unstack()
        print(f"[{ratio}]")
        for a, b in COMPARISONS:
            print("  " + paired_test(table, "balanced accuracy", a, b))

    headtilt = pd.DataFrame(headtilt)
    headtilt.to_csv(OUT / "headtilt.csv", index=False)
    print("\n=== Head-tilt test: share of head-tilt frames predicted as leaning_left/right ===")
    for ratio, g in headtilt.groupby("ratio"):
        table = g.groupby(["person", "variant"])["share_leaning"].mean().unstack()
        print(f"[{ratio}] mean over participants: "
              + ", ".join(f"{v} {table[v].mean():.3f}" for v in table.columns)
              + f" ({len(table)} participants)")
        print("  " + paired_test(table, "share", "v2", "v1"))
        print("  " + paired_test(table, "share", "v2", "v2_nocheck"))
    print(f"\nSaved results in {OUT}")


if __name__ == "__main__":
    main()