"""Stage 3: a landmark model of disengagement onset.

The question a tutor needs answered each week is "who is about to stop engaging?",
not "who will eventually be unregistered?" (official withdrawal lags last activity
by a median of 9 days, and by 14+ days for a third of students). So the event is
**disengagement onset**: the day of a withdrawing student's last VLE activity.

At each landmark week w (cutoff c = 7w), for every student still enrolled:

    event_h = withdraws AND c <= last_active_day < c + 7h      (default h = 4 weeks)

Students who have *already* gone permanently silent before c are labelled 0. In
deployment they cannot be told apart from students on a break, and the goal is
warning *before* silence, so flagging them counts against the model.

One pooled model is trained on all landmark weeks at once (a "landmark
super-model"), with smooth functions of the week, so it can learn how risk and the
meaning of each feature change over the term. Validation folds are grouped by
module presentation. `last_active_day` looks into the future and is used ONLY for
labels and evaluation, never as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from eew.config import RANDOM_SEED
from eew.model import presentation_groups

HORIZON_WEEKS = 4
TRAIN_WEEKS = list(range(1, 25))  # landmarks used for training
EVAL_WEEKS = [2, 4, 6, 8, 10, 12]


def add_onset_label(df: pd.DataFrame, horizon_weeks: int = HORIZON_WEEKS) -> pd.DataFrame:
    """Label: goes permanently silent (and later withdraws) within the next h weeks."""
    out = df.copy()
    c = 7 * out["week"]
    last = out["last_active_day"]
    out["onset"] = out["will_withdraw"] & (last >= c) & (last < c + 7 * horizon_weeks)
    out["already_silent"] = out["will_withdraw"] & (last < c) | (
        out["will_withdraw"] & last.isna()
    )
    out["weeks_to_onset"] = (last - c) / 7
    return out


class LandmarkModel:
    """Pooled landmark model: features + smooth week terms (+ week interactions for 'logistic')."""

    def __init__(self, kind: str = "logistic", interactions: bool = True):
        self.kind, self.interactions = kind, interactions

    def _design(self, X: np.ndarray, week: np.ndarray, fit: bool) -> np.ndarray:
        if self.kind == "gbm":
            return np.column_stack([X, week])
        if fit:
            self.spline = SplineTransformer(n_knots=4, degree=3).fit(week.reshape(-1, 1))
        W = self.spline.transform(week.reshape(-1, 1))
        parts = [X, W]
        if self.interactions:
            # Let each feature's effect drift linearly over the term.
            parts.append(X * ((week - 12) / 12).reshape(-1, 1))
        return np.column_stack(parts)

    def fit(self, X, week, y):
        D = self._design(X, week, fit=True)
        if self.kind == "gbm":
            self.est = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                l2_regularization=1.0, random_state=RANDOM_SEED,
            ).fit(D, y)
        else:
            self.est = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)
            ).fit(D, y)
        return self

    def predict_proba(self, X, week):
        return self.est.predict_proba(self._design(X, week, fit=False))[:, 1]


def oof_landmark(
    data: pd.DataFrame, features: list[str], target: str = "onset",
    kind: str = "logistic", interactions: bool = True, n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold predictions for the stacked landmark data, folds grouped by presentation."""
    X = data[features].to_numpy(dtype=float)
    week = data["week"].to_numpy(dtype=float)
    y = data[target].astype(int).to_numpy()
    groups = presentation_groups(data)
    pred = np.full(len(data), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = LandmarkModel(kind, interactions).fit(X[train], week[train], y[train])
        pred[test] = m.predict_proba(X[test], week[test])
    return pred


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted event rate by decile of predicted risk."""
    df = pd.DataFrame({"y": y.astype(float), "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
