"""Tests for the population rules, baseline scores and capacity evaluation."""

import numpy as np
import pandas as pd

from eew import baseline, evaluate, population

KEY = {"code_module": "AAA", "code_presentation": "2013J"}


def _enrolments():
    rows = [
        # id, withdrawn, withdrawal_day, failed_or_withdrew
        (1, False, np.nan, False),
        (2, True, -5, True),      # withdrew before start -> excluded
        (3, True, np.nan, True),  # withdrawn, no date -> excluded
        (4, True, 10, True),      # withdraws in week 2
        (5, True, 40, True),      # withdraws in week 6
        (6, False, np.nan, True),  # fails
    ]
    return pd.DataFrame(
        [{**KEY, "id_student": i, "withdrawn": w, "withdrawal_day": d, "failed_or_withdrew": f}
         for i, w, d, f in rows]
    )


def test_population_excludes_pre_start_and_undated():
    pop = population.early_warning_population(_enrolments())
    assert set(pop.id_student) == {1, 4, 5, 6}
    assert population.excluded_summary(_enrolments()) == {
        "withdrawn_total": 4, "withdrawn_before_start": 1, "withdrawn_no_date": 1,
    }


def test_at_risk_drops_already_withdrawn_and_labels():
    pop = population.early_warning_population(_enrolments())
    ar = population.at_risk(pop, week=2).set_index("id_student")  # cutoff day 14
    assert set(ar.index) == {1, 5, 6}  # student 4 withdrew on day 10
    assert ar.loc[5, "will_withdraw"] and not ar.loc[6, "will_withdraw"]
    assert ar.loc[6, "needs_support"]
    assert ar.loc[5, "weeks_until_withdrawal"] == (40 - 14) / 7


def test_baseline_uses_only_data_before_cutoff():
    pop = population.early_warning_population(_enrolments())
    ar = population.at_risk(pop, week=2)
    daily = pd.DataFrame([
        {**KEY, "id_student": 1, "day": 10, "clicks": 50},
        {**KEY, "id_student": 1, "day": 14, "clicks": 999},  # on cutoff day: must be ignored
        {**KEY, "id_student": 5, "day": 3, "clicks": 5},
    ])
    out = baseline.baseline_scores(daily, ar, week=2).set_index("id_student")
    assert out.loc[1, "clicks_last7"] == 50 and out.loc[1, "clicks_cum"] == 50
    assert out.loc[5, "clicks_last7"] == 0 and out.loc[5, "clicks_cum"] == 5
    assert out.loc[6, "band"] == "none"
    assert out.loc[6, "risk_cum"] > out.loc[1, "risk_cum"]  # no clicks = higher risk


def test_flag_top_respects_capacity_with_ties():
    df = pd.DataFrame({**{k: [v] * 100 for k, v in KEY.items()}, "s": [1.0] * 100})
    assert evaluate.flag_top(df, "s", 0.10).sum() == 10


def test_capacity_metrics_perfect_score():
    df = pd.DataFrame({
        **{k: [v] * 10 for k, v in KEY.items()},
        "will_withdraw": [True] + [False] * 9,
        "weeks_until_withdrawal": [3.0] + [np.nan] * 9,
        "s": [1.0] + [0.0] * 9,
    })
    m = evaluate.capacity_metrics(df, "s", 0.10)
    assert m["recall"] == 1 and m["precision"] == 1 and m["auc"] == 1
    assert m["median_weeks_to_withdrawal"] == 3


def test_last_activity_and_silent_students():
    pop = population.early_warning_population(_enrolments())
    daily = pd.DataFrame([
        {**KEY, "id_student": 5, "day": 20, "clicks": 3},
        {**KEY, "id_student": 5, "day": 45, "clicks": 9},  # after withdrawal (day 40): ignored
    ])
    last = population.last_activity_day(daily, pop)
    ar = population.add_last_activity(population.at_risk(pop, week=2), last).set_index("id_student")
    assert ar.loc[5, "weeks_until_last_activity"] == (20 - 14) / 7
    assert pd.isna(ar.loc[1, "weeks_until_last_activity"])  # never withdrew


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    rows = []
    for p in range(12):
        for i in range(50):
            rows.append({"code_module": f"M{p}", "code_presentation": "2013J",
                         "will_withdraw": rng.random() < 0.2,
                         "weeks_until_withdrawal": 2.0, "s": rng.random()})
    df = pd.DataFrame(rows)
    m = evaluate.capacity_metrics(df, "s", 0.10, n_boot=500)
    assert m["recall_lo"] <= m["recall"] <= m["recall_hi"]
    assert m["precision_lo"] <= m["precision"] <= m["precision_hi"]
    assert 0 < m["max_recall"] <= 1


def test_paired_difference_detects_better_score():
    rng = np.random.default_rng(2)
    rows = []
    for p in range(15):
        for i in range(60):
            y = rng.random() < 0.2
            rows.append({"code_module": f"M{p}", "code_presentation": "2013J", "will_withdraw": y,
                         "good": y + rng.random() * 0.8, "noise": rng.random()})
    df = pd.DataFrame(rows)
    d = evaluate.paired_difference(df, "good", "noise", 0.10)
    assert d["d_precision_lo"] > 0 and d["p_a_not_better"] < 0.01
    same = evaluate.paired_difference(df, "noise", "noise", 0.10)
    assert same["d_precision"] == 0
