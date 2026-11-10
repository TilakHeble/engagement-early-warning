"""Stage 3 tests: onset labels and the pooled landmark model."""

import numpy as np
import pandas as pd

from eew import landmark


def test_onset_label_windows():
    df = pd.DataFrame({
        "week": [4] * 5,  # cutoff day 28, 4-week horizon -> [28, 56)
        "will_withdraw": [True, True, True, True, False],
        "last_active_day": [30.0, 55.0, 56.0, 20.0, np.nan],
    })
    out = landmark.add_onset_label(df)
    assert out["onset"].tolist() == [True, True, False, False, False]
    assert out["already_silent"].tolist() == [False, False, False, True, False]
    assert out.loc[0, "weeks_to_onset"] == (30 - 28) / 7


def test_never_clicked_withdrawer_is_already_silent():
    df = pd.DataFrame({"week": [2], "will_withdraw": [True], "last_active_day": [np.nan]})
    out = landmark.add_onset_label(df)
    assert out["already_silent"].iloc[0] and not out["onset"].iloc[0]


def test_landmark_model_learns_signal_and_calibration_table():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    week = rng.integers(1, 25, n).astype(float)
    p = 1 / (1 + np.exp(-(-3 + 1.5 * x[:, 0])))
    y = rng.random(n) < p
    for kind in ["logistic", "gbm"]:
        m = landmark.LandmarkModel(kind).fit(x, week, y)
        pred = m.predict_proba(x, week)
        assert pred[x[:, 0] > 1].mean() > 2 * pred[x[:, 0] < -1].mean()
    t = landmark.calibration_table(y, pred, n_bins=5)
    assert len(t) == 5 and t["n"].sum() == n
