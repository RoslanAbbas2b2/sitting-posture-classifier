"""Train and save the three models used by the posture application (posture_app.py).

All three use the selected pipeline of the report (MediaPipe landmarks 0-12, x/y/z = 39 values,
StandardScaler + Logistic Regression) and the v2 training data: every Roboflow image with a pose plus
EVERY self-collected participant (3:2-corrected coordinates; leaning frames that fail the shoulder
check left out, as in v2). Only the minority reduction and the handling differ:

  model 1  natural                      all training data, no handling            ("logreg")
  model 2  imbalanced                   leaning_left/right reduced to 30 each      ("logreg")
  model 3  imbalanced + class weighting the same reduced data, class weights       ("logreg+weighted")

Why 30 images: demo_imbalance_check.py (9 Nov 2026) showed that a 10:1 ratio leaves about 680
leaning images per class in the v2 data and barely changes minority recall (0.968 -> 0.946), while
30 per class (about the number in a Roboflow 10:1 fold) lowers it to 0.760 and class weighting
restores it to 0.934. The same random reduction (seed 0) is used for models 2 and 3, so they differ
only in the class weights.

The models are trained on all participants, so they have no test data of their own: the demo is an
illustration; the evaluated numbers are those of run_roboflow_vs_joint.py and demo_imbalance_check.py.

Output: models/demo/model1.joblib, model2.joblib, model3.joblib (each a dict with the fitted
pipeline and a description) and models/demo/training_counts.csv.

Run:  python train_demo_models.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import OWN_FRAMES_ROOT
from demo_imbalance_check import training_indices
from evaluation import CLASSES
from extract_frames import MANIFEST_OWN
from features import OWN_IMAGE_SIZES
from models import make_model
from run_roboflow_vs_joint import LABEL_CHECKS, load_own, load_roboflow

DATA = Path("data")
OUT = Path("models") / "demo"
SEED = 0
DEMO_MODELS = {
    "model1": {"label": "Natural", "setting": "natural", "model": "logreg",
               "description": "All training data, no imbalance handling"},
    "model2": {"label": "Imbalanced", "setting": "n30", "model": "logreg",
               "description": "Leaning left/right reduced to 30 images each, no handling"},
    "model3": {"label": "Imbalanced + class weighting", "setting": "n30", "model": "logreg+weighted",
               "description": "Same reduced data, class weights (selected pipeline)"},
}


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Roboflow images with a pose + all self-collected frames that pass the shoulder check."""
    rf_X, rf_y = load_roboflow()
    rf_ok = ~np.isnan(rf_X).any(axis=1)
    own, _, own_X = load_own(DATA / "landmarks_own_fallback.csv", MANIFEST_OWN, OWN_FRAMES_ROOT, OWN_IMAGE_SIZES)
    checks = pd.read_csv(LABEL_CHECKS).set_index("path")["train_ok"]
    own_ok = ~np.isnan(own_X).any(axis=1) & own["path"].map(checks).fillna(True).to_numpy(dtype=bool)
    X = np.vstack([rf_X[rf_ok], own_X[own_ok]])
    y = np.concatenate([rf_y[rf_ok], own["class"].to_numpy(dtype=object)[own_ok]])
    return X, y


def main() -> None:
    X, y = load_training_data()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for key, spec in DEMO_MODELS.items():
        keep = training_indices(spec["setting"], y, SEED)
        model = make_model(spec["model"]).fit(X[keep], y[keep])
        counts[key] = pd.Series(y[keep]).value_counts().reindex(CLASSES)
        joblib.dump({**spec, "pipeline": model, "classes": list(model.classes_), "seed": SEED},
                    OUT / f"{key}.joblib")
        print(f"Saved {OUT / f'{key}.joblib'}  ({spec['label']}: {len(keep):,} training rows)")

    table = pd.DataFrame(counts)
    table.to_csv(OUT / "training_counts.csv")
    print("\nTraining images per class:")
    print(table)


if __name__ == "__main__":
    main()
