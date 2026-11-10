"""Stage 2 feature tests: correctness on a small example, and no use of future data."""

import numpy as np
import pandas as pd
import pytest

from eew import features, population

KEY = {"code_module": "AAA", "code_presentation": "2013J"}
WEEK = 4  # cutoff day 28


@pytest.fixture
def inputs():
    enr = pd.DataFrame([
        {**KEY, "id_student": i, "withdrawn": False, "withdrawal_day": np.nan,
         "failed_or_withdrew": False, "date_registration": -20, "studied_credits": 60,
         "num_of_prev_attempts": 0}
        for i in (1, 2, 3)
    ])
    daily = pd.DataFrame(
        # student 1: steady 10 clicks/day on even days from day -14 to 27
        [{**KEY, "id_student": 1, "day": d, "clicks": 10} for d in range(-14, 28, 2)]
        # student 2: active early, silent since day 10
        + [{**KEY, "id_student": 2, "day": d, "clicks": 20} for d in range(0, 11)]
        # student 3: never clicks
    )
    assessments = pd.DataFrame([
        {**KEY, "id_assessment": 100, "assessment_type": "TMA", "date": 10, "weight": 10},
        {**KEY, "id_assessment": 101, "assessment_type": "CMA", "date": 20, "weight": 10},
        {**KEY, "id_assessment": 102, "assessment_type": "TMA", "date": 40, "weight": 10},  # not due
        {**KEY, "id_assessment": 103, "assessment_type": "Exam", "date": 25, "weight": 100},  # exam
    ])
    sa = pd.DataFrame([
        {"id_assessment": 100, "id_student": 1, "date_submitted": 9, "is_banked": 0, "score": 80},
        {"id_assessment": 101, "id_student": 1, "date_submitted": 22, "is_banked": 0, "score": 60},
        {"id_assessment": 100, "id_student": 2, "date_submitted": 12, "is_banked": 0, "score": 50},
    ])
    ar = population.at_risk(population.early_warning_population(enr), WEEK)
    return daily, enr, assessments, sa, ar


def _build(daily, enr, a, sa, ar):
    return features.build_features(daily, enr, a, sa, ar, WEEK).set_index("id_student")


def test_feature_values(inputs):
    f = _build(*inputs)
    assert f.loc[1, "days_since_last_click"] == 28 - 26
    assert f.loc[2, "days_since_last_click"] == 28 - 10
    assert f.loc[3, "never_clicked"] == 1
    # Student 1 is steady (35 clicks/week before and after): ratio 0.
    assert f.loc[1, "traj_log_ratio"] == pytest.approx(0)
    # Student 2: 220 clicks over the 4 earlier weeks (55/week), then 0.
    assert f.loc[2, "traj_log_ratio"] == pytest.approx(-np.log1p(55))
    assert f.loc[2, "traj_slope4"] < 0
    assert f.loc[1, "pct_clicks_last14"] > f.loc[3, "pct_clicks_last14"]
    # Assessments due before day 28 (exam excluded): 100 and 101.
    assert (f["n_due"] == 2).all()
    assert f.loc[1, "share_missed"] == 0 and f.loc[1, "share_late"] == 0.5
    assert f.loc[2, "share_missed"] == 0.5 and f.loc[2, "mean_days_late"] == 2
    assert f.loc[3, "share_missed"] == 1
    # Scores only for submissions made >= 14 days before the cutoff (day < 14).
    assert f.loc[1, "mean_score"] == pytest.approx(0.80) and f.loc[1, "has_score"] == 1
    assert f.loc[3, "has_score"] == 0
    assert list(f.columns[-len(features.FEATURES):]) == features.FEATURES


def test_no_future_data_leaks(inputs):
    daily, enr, a, sa, ar = inputs
    before = _build(daily, enr, a, sa, ar)
    future_daily = pd.concat([daily, pd.DataFrame(
        [{**KEY, "id_student": s, "day": d, "clicks": 500} for s in (1, 2, 3) for d in (28, 30, 60)]
    )])
    future_sa = pd.concat([sa, pd.DataFrame([
        {"id_assessment": 101, "id_student": 2, "date_submitted": 28, "is_banked": 0, "score": 99},
        {"id_assessment": 102, "id_student": 1, "date_submitted": 35, "is_banked": 0, "score": 10},
    ])])
    after = _build(future_daily, enr, a, future_sa, ar)
    pd.testing.assert_frame_equal(before[features.FEATURES], after[features.FEATURES])


def test_no_demographics_in_features():
    banned = {"gender", "age_band", "region", "highest_education", "imd_band", "disability"}
    assert not banned & set(features.FEATURES)
