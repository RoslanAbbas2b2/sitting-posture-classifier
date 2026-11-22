"""Which imbalance setting makes the effect of imbalance VISIBLE for the demo application?

Background: with the v2 training data (Roboflow + self-collected participants), a 10:1 ratio
leaves hundreds of leaning_left / leaning_right frames, and the notebook (Section 7) found that
the ratio then no longer matters. The demo's model switch (1 natural / 2 imbalanced / 3 imbalanced +
class weighting) is only useful if model 2 clearly misses the minority classes and model 3 recovers.

This script uses the v2 design of run_roboflow_vs_joint.py (5 person-level folds of the self-collected
participants; train on Roboflow + the other four folds, shoulder check applied; test on the held-out
fold) and compares, for each setting, Logistic Regression without handling ("logreg") and with class
weighting ("logreg+weighted"):
  natural      no reduction
  10to1        leaning_left/right reduced to upright / 10 (ControlledImbalance, as in the report)
  n<N>         leaning_left/right reduced to N images each (absolute number), e.g. n30

Only the TRAINING data are reduced; the test frames never change. Results are pooled over the 5 folds.
The chosen setting is recorded in docs/decision_log.md and used by train_demo_models.py.

Run:  python demo_imbalance_check.py                 (seed 0)
      python demo_imbalance_check.py --seeds 0 1 2   (mean over seeds)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import OWN_FRAMES_ROOT
from evaluation import ControlledImbalance, compute_metrics
from extract_frames import MANIFEST_OWN
from features import OWN_IMAGE_SIZES
from models import make_model
from run_roboflow_vs_joint import FOLDS_CSV, LABEL_CHECKS, LEANS, N_FOLDS, load_own, load_roboflow, predict

DATA = Path("data")
OUT = Path("results") / "demo_imbalance_check.csv"
ABSOLUTE_SIZES = (300, 100, 30)  # images kept per minority class
MODELS = ("logreg", "logreg+weighted")


def reduce_to(n: int, labels: np.ndarray, seed: int) -> np.ndarray:
    """Indices to keep: every non-minority image and n random images of each minority class."""
    rng = np.random.default_rng(seed)
    keep = [np.flatnonzero(~np.isin(labels, LEANS))]
    for cls in LEANS:
        idx = np.flatnonzero(labels == cls)
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def training_indices(setting: str, labels: np.ndarray, seed: int) -> np.ndarray:
    """Training rows to keep for one setting ('natural', '10to1' or 'n<N>')."""
    if setting == "natural":
        return np.arange(len(labels))
    if setting == "10to1":
        return ControlledImbalance(ratio=10, seed=seed).apply(np.arange(len(labels)), labels)
    return reduce_to(int(setting[1:]), labels, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="seeds for the random reduction")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    rf_X, rf_y = load_roboflow()
    has_pose = ~np.isnan(rf_X).any(axis=1)
    rf_X, rf_y = rf_X[has_pose], rf_y[has_pose]
    own, _, own_X = load_own(DATA / "landmarks_own_fallback.csv", MANIFEST_OWN, OWN_FRAMES_ROOT, OWN_IMAGE_SIZES)
    own_y = own["class"].to_numpy(dtype=object)

    fold_of = pd.read_csv(FOLDS_CSV).set_index("person")["fold"]
    own_fold = own["person"].map(fold_of).to_numpy()
    checks = pd.read_csv(LABEL_CHECKS).set_index("path")["train_ok"]
    train_ok = own["path"].map(checks).fillna(True).to_numpy(dtype=bool)
    own_has_pose = ~np.isnan(own_X).any(axis=1)

    settings = ["natural", "10to1", *[f"n{n}" for n in ABSOLUTE_SIZES]]
    rows = []
    for setting in settings:
        for seed in ([None] if setting == "natural" else args.seeds):
            for name in MODELS:
                pred = np.empty(len(own_X), dtype=object)
                minority_counts = []
                for fold in range(N_FOLDS):
                    use = own_has_pose & (own_fold != fold) & train_ok
                    X = np.vstack([rf_X, own_X[use]])
                    y = np.concatenate([rf_y, own_y[use]])
                    keep = training_indices(setting, y, seed or 0)
                    minority_counts.append(int(np.isin(y[keep], LEANS).sum() / 2))
                    model = make_model(name).fit(X[keep], y[keep])
                    test = own_fold == fold
                    pred[test] = predict(model, own_X[test])
                m = compute_metrics(own_y, pred)
                rows.append({"setting": setting, "seed": seed, "model": name,
                             "minority_per_class": int(np.mean(minority_counts)),
                             "balanced_accuracy": m["balanced_accuracy"],
                             "recall_leaning_left": m["recall_leaning_left"],
                             "recall_leaning_right": m["recall_leaning_right"],
                             "recall_upright": m["recall_upright"]})
            print(f"done: {setting}" + ("" if seed is None else f" seed {seed}"))

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    summary = (table.groupby(["setting", "model"], sort=False)
               [["minority_per_class", "balanced_accuracy", "recall_leaning_left",
                 "recall_leaning_right", "recall_upright"]].mean().round(3))
    print("\n=== Pooled over the 5 self-collected folds (mean over seeds) ===")
    print(summary)

    print("\n=== Effect of imbalance and of class weighting on minority recall (mean of left and right) ===")
    recall = summary[["recall_leaning_left", "recall_leaning_right"]].mean(axis=1).unstack("model")
    natural = recall.loc["natural", "logreg"]
    recall["drop_without_handling"] = (natural - recall["logreg"]).round(3)
    recall["recovered_by_weighting"] = (recall["logreg+weighted"] - recall["logreg"]).round(3)
    print(recall.round(3))
    print("\nPick the setting with a clear drop without handling AND a clear recovery with weighting.")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
