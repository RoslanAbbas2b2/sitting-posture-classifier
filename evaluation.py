"""Shared evaluation code for every experiment in this project.

Every pipeline is evaluated in exactly the same way:
  - Leave-one-group-out cross-validation over the three evaluation groups in
    data/manifest.csv: A and B each contain one visually identified person,
    Others contains several persons with few images (combined because they do
    not each cover every posture class). Train on two groups, test on the
    third (three folds). This is not strict leave-one-person-out, and person
    identity was inferred from the images (no participant IDs).
  - Optional controlled imbalance, applied to the TRAINING images of each
    fold only. Test images are never changed. Because the reduction is
    random, each ratio is repeated over several seeds and summarised as
    mean and standard deviation (summarise_repeats).
  - Optional training filter (quality filtering), also applied to the
    TRAINING images only.
  - Images with no feature vector (e.g. MediaPipe found no pose) are not
    used for training, but stay in the test set and count as wrong
    predictions (label "no_pose"). The test set is therefore the same
    4,813 images for every pipeline, so a pipeline that detects more
    poses gets credit for it.
  - Metrics are reported per fold and on the pooled out-of-fold
    predictions (every image predicted once, by a model trained without
    that image's evaluation group).
run_experiment() runs one feature set + model over several ratios and seeds;
every route (landmarks, image features, CNN) uses it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

CLASSES = ["leaning_backward", "leaning_forward", "leaning_left", "leaning_right", "upright"]
NO_POSE = "no_pose"
MANIFEST = Path("data") / "manifest.csv"
RESULTS_DIR = Path("results")
SEEDS = (0, 1, 2, 3, 4)  # seeds for the random minority reduction
REPEAT_METRICS = ["balanced_accuracy", "macro_f1", "accuracy",
                  "recall_leaning_left", "recall_leaning_right",
                  "precision_leaning_left", "precision_leaning_right"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, balanced accuracy, macro F1, and per-class precision and recall.
    Balanced accuracy is the mean recall over the five classes.
    A 'no_pose' prediction counts as a miss for the true class."""
    recall = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    precision = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
    }
    for cls, p, r in zip(CLASSES, precision, recall):
        metrics[f"precision_{cls}"] = float(p)
        metrics[f"recall_{cls}"] = float(r)
    return metrics


@dataclass
class ControlledImbalance:
    """Reduce the minority classes in a training set so that the reference
    class has `ratio` times more images than each minority class.
    ratio=None keeps the natural class distribution."""

    ratio: float | None = None
    minority_classes: tuple[str, ...] = ("leaning_left", "leaning_right")
    reference_class: str = "upright"
    seed: int = 42

    @property
    def ratio_name(self) -> str:
        return "natural" if self.ratio is None else f"{self.ratio:g}to1"

    @property
    def name(self) -> str:
        """Ratio plus seed, e.g. '10to1_s3' (natural has no random step)."""
        return self.ratio_name if self.ratio is None else f"{self.ratio_name}_s{self.seed}"

    def apply(self, train_idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Return the training indices to keep (random reduction, fixed seed)."""
        if self.ratio is None:
            return train_idx
        rng = np.random.default_rng(self.seed)
        train_labels = labels[train_idx]
        target = int((train_labels == self.reference_class).sum() / self.ratio)
        keep = [train_idx[~np.isin(train_labels, self.minority_classes)]]
        for cls in self.minority_classes:
            cls_idx = train_idx[train_labels == cls]
            if len(cls_idx) > target:
                cls_idx = rng.choice(cls_idx, size=target, replace=False)
            keep.append(cls_idx)
        return np.sort(np.concatenate(keep))


@dataclass
class ExperimentResult:
    """All outputs of one experiment: per-fold and pooled metrics, predictions."""

    name: str
    imbalance: str
    per_fold: pd.DataFrame
    pooled: dict
    paths: np.ndarray
    true_labels: np.ndarray
    predictions: np.ndarray

    def confusion(self) -> pd.DataFrame:
        """Pooled confusion matrix: rows = true class, columns = predicted (incl. no_pose)."""
        cols = CLASSES + [NO_POSE]
        cm = confusion_matrix(self.true_labels, self.predictions, labels=cols)[: len(CLASSES)]
        return pd.DataFrame(cm, index=CLASSES, columns=cols)

    def summary_line(self) -> str:
        p = self.pooled
        folds = ", ".join(f"{r.test_group}={r.balanced_accuracy:.3f}" for r in self.per_fold.itertuples())
        return (f"{self.name} [{self.imbalance}]  pooled: bal_acc={p['balanced_accuracy']:.3f} "
                f"macro_f1={p['macro_f1']:.3f} acc={p['accuracy']:.3f} | "
                f"recall left={p['recall_leaning_left']:.3f} right={p['recall_leaning_right']:.3f} | "
                f"fold bal_acc: {folds}")

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        """Write CSVs and a confusion-matrix figure, and update results/summary.csv."""
        out = results_dir / f"{self.name}__{self.imbalance}"
        out.mkdir(parents=True, exist_ok=True)
        self.per_fold.to_csv(out / "per_fold.csv", index=False)
        pd.Series(self.pooled, name="value").to_csv(out / "pooled_metrics.csv")
        self.confusion().to_csv(out / "confusion_matrix.csv")
        pd.DataFrame({"path": self.paths, "true": self.true_labels,
                      "pred": self.predictions}).to_csv(out / "predictions.csv", index=False)
        self._plot_confusion(out / "confusion_matrix.png")
        self._update_summary(results_dir / "summary.csv")
        return out

    def _plot_confusion(self, path: Path) -> None:
        cm = self.confusion()
        row_share = cm.div(cm.sum(axis=1), axis=0)  # colour = share of the true class
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(cm.index)), cm.index)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm.iat[i, j], ha="center", va="center",
                        color="white" if row_share.iat[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"{self.name} [{self.imbalance}]\n"
            "pooled leave-one-group-out cross-validation"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _update_summary(self, path: Path) -> None:
        row = {"experiment": self.name, "imbalance": self.imbalance, **self.pooled}
        for r in self.per_fold.itertuples():
            row[f"bal_acc_test_{r.test_group}"] = r.balanced_accuracy
        _upsert_row(path, row)


def _upsert_row(path: Path, row: dict) -> None:
    """Add a row to a CSV summary, replacing an older row with the same
    experiment and imbalance."""
    summary = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not summary.empty:
        summary = summary[~((summary["experiment"] == row["experiment"])
                            & (summary["imbalance"] == row["imbalance"]))]
    pd.concat([summary, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def summarise_repeats(experiment: str, ratio_name: str, results: list[ExperimentResult],
                      results_dir: Path = RESULTS_DIR) -> dict:
    """Mean and sample standard deviation of pooled metrics over repeated
    runs (one run per seed). Saved to results/summary_repeats.csv."""
    row = {"experiment": experiment, "imbalance": ratio_name, "n_runs": len(results)}
    for metric in REPEAT_METRICS:
        values = np.array([r.pooled[metric] for r in results])
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    results_dir.mkdir(exist_ok=True)
    _upsert_row(results_dir / "summary_repeats.csv", row)
    return row


class LeaveOnePersonOutEvaluator:
    """Runs one model through the three leave-one-group-out folds (test A, B or Others).

    The class keeps its historical name. The design is leave-one-group-out, not strict
    leave-one-person-out: the Others fold holds out several persons at once.
    """

    def __init__(self, manifest_path: Path = MANIFEST):
        manifest = pd.read_csv(manifest_path)
        self.paths = manifest["path"].to_numpy()
        self.labels = manifest["class"].to_numpy(dtype=object)
        self.groups = manifest["group"].to_numpy()
        self.test_groups = sorted(set(self.groups))  # ["A", "B", "Others"]

    def evaluate(self, name: str, make_model: Callable, X: np.ndarray,
                 paths: np.ndarray | None = None,
                 imbalance: ControlledImbalance | None = None,
                 train_mask: np.ndarray | None = None) -> ExperimentResult:
        """Evaluate a model.

        make_model: function that returns a NEW untrained model (one per fold).
        X: one feature row per manifest image, in manifest order. Rows that
           contain NaN are 'not available' (not trained on, predicted no_pose).
        paths: if given, checked against the manifest order.
        train_mask: if given, only rows where it is True may be used for
           training (quality filtering). Test rows are never removed.
        """
        imbalance = imbalance or ControlledImbalance()
        if len(X) != len(self.labels):
            raise ValueError("X must have one row per manifest image")
        if paths is not None and not np.array_equal(paths, self.paths):
            raise ValueError("Feature rows are not in manifest order")

        available = ~np.isnan(X).any(axis=1)
        trainable = available if train_mask is None else available & train_mask
        predictions = np.full(len(X), NO_POSE, dtype=object)
        fold_rows = []
        for test_group in self.test_groups:
            is_test = self.groups == test_group
            train_idx = imbalance.apply(np.flatnonzero(~is_test & trainable), self.labels)
            model = make_model()
            model.fit(X[train_idx], self.labels[train_idx])
            test_idx = np.flatnonzero(is_test & available)
            predictions[test_idx] = model.predict(X[test_idx])
            fold_rows.append({"test_group": test_group, "n_train": len(train_idx),
                              "n_test": int(is_test.sum()),
                              **compute_metrics(self.labels[is_test], predictions[is_test])})

        return ExperimentResult(name=name, imbalance=imbalance.name,
                                per_fold=pd.DataFrame(fold_rows),
                                pooled=compute_metrics(self.labels, predictions),
                                paths=self.paths, true_labels=self.labels,
                                predictions=predictions)


def imbalance_settings(ratio: str, seeds: tuple[int, ...] = SEEDS) -> list[ControlledImbalance]:
    """One setting for 'natural' (no random step); one per seed for any other ratio."""
    if ratio == "natural":
        return [ControlledImbalance()]
    return [ControlledImbalance(ratio=float(ratio), seed=s) for s in seeds]


def run_experiment(experiment: str, make_model: Callable, X: np.ndarray, paths: np.ndarray,
                   ratios: list[str], train_mask: np.ndarray | None = None,
                   evaluator: LeaveOnePersonOutEvaluator | None = None) -> list[dict]:
    """Evaluate one feature set + model at each ratio (repeated over seeds),
    save every run, and print and return the mean +- SD summary per ratio."""
    evaluator = evaluator or LeaveOnePersonOutEvaluator()
    summaries = []
    for ratio in ratios:
        settings = imbalance_settings(ratio)
        runs = []
        for imbalance in settings:
            result = evaluator.evaluate(experiment, make_model, X, paths=paths,
                                        imbalance=imbalance, train_mask=train_mask)
            result.save()
            runs.append(result)
        s = summarise_repeats(experiment, settings[0].ratio_name, runs)
        print(f"  {experiment} [{s['imbalance']}] runs={s['n_runs']}: "
              f"bal_acc={s['balanced_accuracy_mean']:.3f}±{s['balanced_accuracy_std']:.3f}  "
              f"macro_f1={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}  "
              f"recall left={s['recall_leaning_left_mean']:.3f}±{s['recall_leaning_left_std']:.3f}  "
              f"right={s['recall_leaning_right_mean']:.3f}±{s['recall_leaning_right_std']:.3f}")
        summaries.append(s)
    return summaries


if __name__ == "__main__":
    # Sanity check: a model that always predicts the most common training class.
    # Expected: balanced accuracy 0.20 in every fold (one of five classes always
    # correct). Results are printed, not saved.
    from sklearn.dummy import DummyClassifier

    lm = pd.read_csv(Path("data") / "landmarks_raw.csv")
    cols = [f"lm{i}_{f}" for i in range(13) for f in ("x", "y", "z")]
    evaluator = LeaveOnePersonOutEvaluator()
    for imb in (ControlledImbalance(), ControlledImbalance(ratio=10)):
        res = evaluator.evaluate("sanity_dummy", lambda: DummyClassifier(strategy="most_frequent"),
                                 lm[cols].to_numpy(), paths=lm["path"].to_numpy(), imbalance=imb)
        print(res.summary_line())
        print(res.per_fold[["test_group", "n_train", "n_test", "accuracy", "balanced_accuracy"]]
              .round(3).to_string(index=False), "\n")