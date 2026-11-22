"""Classifiers and imbalance-handling techniques shared by every route.

A model is named "<base>[+<handling>]", e.g. "svm", "svm+smote".
make_model(name) returns a NEW untrained imbalanced-learn Pipeline:
    [StandardScaler] -> [resampler] -> classifier
- Scaling is added for models that depend on feature scale, and always
  before SMOTE because SMOTE uses distances.
- Resampling runs only during fit (imblearn Pipeline), so test data is
  never resampled.

Models use scikit-learn defaults (only iteration limits raised so they
converge). No hyperparameter tuning: with three leave-one-group-out folds,
tuning without nested cross-validation would pick settings that suit the
test person.
"""
from collections.abc import Callable
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_SEED = 42  # random_state of the classifiers and samplers


@dataclass
class BaseModel:
    """A classifier factory. `build(class_weight)` returns a new classifier."""

    build: Callable
    needs_scaling: bool
    supports_class_weight: bool


BASE_MODELS = {
    "logreg": BaseModel(lambda cw: LogisticRegression(max_iter=2000, class_weight=cw,
                                                      random_state=MODEL_SEED), True, True),
    "svm": BaseModel(lambda cw: SVC(kernel="rbf", class_weight=cw,
                                    random_state=MODEL_SEED), True, True),
    "rf": BaseModel(lambda cw: RandomForestClassifier(n_jobs=-1, class_weight=cw,
                                                      random_state=MODEL_SEED), False, True),
    "hgb": BaseModel(lambda cw: HistGradientBoostingClassifier(class_weight=cw,
                                                               random_state=MODEL_SEED), False, True),
    "knn": BaseModel(lambda cw: KNeighborsClassifier(), True, False),
    "mlp": BaseModel(lambda cw: MLPClassifier(max_iter=1000, early_stopping=True,
                                              random_state=MODEL_SEED), True, False),
}

# Imbalance-handling techniques, applied inside each training fold only.
SAMPLERS = {
    "ros": lambda: RandomOverSampler(random_state=MODEL_SEED),    # copy minority images
    "smote": lambda: SMOTE(random_state=MODEL_SEED),              # synthetic minority samples
    "rus": lambda: RandomUnderSampler(random_state=MODEL_SEED),   # drop majority images
}
HANDLING = ["none", "weighted", *SAMPLERS]
MODEL_HELP = f"<{'|'.join(BASE_MODELS)}>[+{'|'.join(HANDLING[1:])}]"


def parse_model(name: str) -> tuple[str, str]:
    """'svm+smote' -> ('svm', 'smote'); 'svm' -> ('svm', 'none')."""
    base, _, handling = name.partition("+")
    handling = handling or "none"
    if base not in BASE_MODELS or handling not in HANDLING:
        raise ValueError(f"Unknown model '{name}'")
    if handling == "weighted" and not BASE_MODELS[base].supports_class_weight:
        raise ValueError(f"{base} does not support class weighting")
    return base, handling


def make_model(name: str) -> Pipeline:
    """Build a NEW untrained pipeline: [scaler] -> [sampler] -> classifier."""
    base, handling = parse_model(name)
    model = BASE_MODELS[base]
    steps = []
    if model.needs_scaling or handling == "smote":
        steps.append(("scale", StandardScaler()))
    if handling in SAMPLERS:
        steps.append(("resample", SAMPLERS[handling]()))
    steps.append(("classify", model.build("balanced" if handling == "weighted" else None)))
    return Pipeline(steps)