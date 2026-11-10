"""Cross-validated models for stage 2 (feature value) and later stages.

Folds are grouped by module presentation, so a model is never scored on a cohort
it was trained on. A random split by student would leak cohort-level information
(click norms, assessment schedule, module difficulty) and inflate results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eew.config import RANDOM_SEED

PRESENTATION = ["code_module", "code_presentation"]


def make_model(kind: str = "logistic"):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=1.0, random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model kind {kind!r}")


def presentation_groups(df: pd.DataFrame) -> np.ndarray:
    return (df["code_module"] + "_" + df["code_presentation"]).to_numpy()


def oof_predict(
    df: pd.DataFrame, features: list[str], target: str, kind: str = "logistic", n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold predicted probabilities, folds grouped by module presentation."""
    X = df[features].to_numpy(dtype=float)
    y = df[target].astype(int).to_numpy()
    groups = presentation_groups(df)
    pred = np.full(len(df), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model(kind).fit(X[train], y[train])
        pred[test] = m.predict_proba(X[test])[:, 1]
    return pred
