"""Stage 4 tests: weekly outreach simulation and its summary."""

import numpy as np
import pandas as pd

from eew import policy

KEY = {"code_module": "AAA", "code_presentation": "2013J"}


def _scores(n=10, weeks=(1, 2, 3)):
    # Student i has score i every week: the same top students stay highest.
    return pd.DataFrame([{**KEY, "id_student": i, "week": w, "s": float(i)}
                         for w in weeks for i in range(n)])


def test_capacity_and_cooldown():
    c = policy.simulate(_scores(), "s", capacity=0.2, cooldown_weeks=4)
    per_week = c.groupby("week").size()
    assert (per_week == 2).all()  # ceil(0.2 * 10)
    # Cooldown: the top two in week 1 are skipped in weeks 2 and 3.
    assert set(c[c.week == 1].id_student) == {9, 8}
    assert set(c[c.week == 2].id_student) == {7, 6}
    assert set(c[c.week == 3].id_student) == {5, 4}


def test_no_cooldown_recontacts():
    c = policy.simulate(_scores(), "s", capacity=0.2, cooldown_weeks=0)
    assert set(c[c.week == 3].id_student) == {9, 8}


def test_summary_counts_only_contacts_before_silence():
    pop = pd.DataFrame([{**KEY, "id_student": i, "withdrawn": i in (9, 5, 1),
                         "failed_or_withdrew": i in (9, 5, 1)} for i in range(10)])
    # Last activity: student 9 on day 20 (week 2 cutoff is day 14 -> reached in time),
    # student 5 on day 10 (contacted week 3 = day 21 -> too late), student 1 never contacted.
    idx = pd.MultiIndex.from_tuples(
        [("AAA", "2013J", 9), ("AAA", "2013J", 5), ("AAA", "2013J", 1)], names=list(KEY) + ["id_student"])
    last = pd.Series([20.0, 10.0, 15.0], index=idx, name="last_active_day")
    contacts = pd.DataFrame([{**KEY, "id_student": 9, "week": 2}, {**KEY, "id_student": 5, "week": 3}])
    s = policy.summarise(contacts, pop, last, first_week=1, last_week=3)
    assert s["withdrawers_in_window"] == 3
    assert s["reached_before_silence"] == 1 / 3
    assert s["median_lead_weeks"] == (20 - 14) / 7
    assert s["unique_students_contacted"] == 2 and s["contacts"] == 2
    assert np.isclose(s["contacts_per_100_students"], 20)


def test_paired_coverage_difference():
    idx = pd.MultiIndex.from_tuples([("M%d" % i, "2013J") for i in range(10)],
                                    names=["code_module", "code_presentation"])
    a = pd.DataFrame({"targets": 10, "reached": 6}, index=idx)
    b = pd.DataFrame({"targets": 10, "reached": 3}, index=idx)
    d = policy.paired_coverage_difference(a, b)
    assert d["difference"] == 0.3 and d["lo"] == d["hi"] == 0.3
