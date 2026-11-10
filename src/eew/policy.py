"""Stage 4: simulate a whole term of tutor outreach driven by a weekly risk score.

Stages 1-3 score single weeks. A support team works across the term: each week it
contacts the highest-risk students it has capacity for, and does not re-contact
someone it reached recently. This module replays that process on held-out
(out-of-fold) scores and asks:

  * coverage: what share of withdrawing students were contacted *while still
    engaging* (on or before their last VLE activity)?
  * lead time: how long before their last activity was the first such contact?
  * workload: total contacts, unique students contacted, and contacts per student
    reached in time.

Caveat: the simulation assumes contact does not change behaviour (outcomes are
those observed without the system). It measures who would be *reached*, not who
would be *saved*; that needs a trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eew.data import ENROLMENT_KEY

PRESENTATION = ["code_module", "code_presentation"]


def simulate(
    scores: pd.DataFrame,
    score: str,
    capacity: float,
    cooldown_weeks: int = 4,
    weeks: list[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Replay weekly outreach and return one row per contact.

    `scores` has one row per (enrolment, landmark week) for students still enrolled
    that week, with the score column. Each week, within each presentation, up to
    ceil(capacity * enrolled) students are contacted, highest score first, skipping
    anyone contacted in the previous `cooldown_weeks` weeks. Ties are broken randomly.
    """
    rng = np.random.default_rng(seed)
    df = scores if weeks is None else scores[scores["week"].isin(weeks)]
    df = df[ENROLMENT_KEY + ["week", score]].copy()
    df["_tie"] = rng.random(len(df))
    df["_key"] = df[ENROLMENT_KEY].astype(str).agg("|".join, axis=1)

    contacts = []
    for _, g in df.groupby(PRESENTATION, sort=False):
        last_contact: dict[str, int] = {}
        for week, gw in g.groupby("week", sort=True):
            n = int(np.ceil(capacity * len(gw)))
            eligible = gw[[
                (week - last_contact.get(k, -10_000)) > cooldown_weeks for k in gw["_key"]
            ]]
            chosen = eligible.sort_values([score, "_tie"], ascending=False).head(n)
            for k in chosen["_key"]:
                last_contact[k] = week
            contacts.append(chosen[ENROLMENT_KEY + ["week"]])
    return pd.concat(contacts, ignore_index=True) if contacts else pd.DataFrame(
        columns=ENROLMENT_KEY + ["week"])


def summarise(
    contacts: pd.DataFrame,
    population: pd.DataFrame,
    last_active: pd.Series,
    first_week: int,
    last_week: int,
) -> dict[str, float]:
    """Coverage, lead time and workload for one simulated term.

    Students counted for coverage: withdrawing students whose last activity falls
    within the simulated weeks (on or after the first contact opportunity, before the
    last week's cutoff + 7 days). Earlier disengagements could not be reached.
    """
    w = population[population["withdrawn"]].merge(
        last_active, left_on=ENROLMENT_KEY, right_index=True, how="left")
    start, end = 7 * first_week, 7 * last_week + 7
    targets = w[(w["last_active_day"] >= start) & (w["last_active_day"] < end)]

    c = contacts.assign(contact_day=7 * contacts["week"])
    m = targets[ENROLMENT_KEY + ["last_active_day"]].merge(c, on=ENROLMENT_KEY, how="left")
    in_time = m[m["contact_day"] <= m["last_active_day"]]
    first = in_time.groupby(ENROLMENT_KEY).agg(
        first_contact=("contact_day", "min"), last_active_day=("last_active_day", "first"))
    lead = (first["last_active_day"] - first["first_contact"]) / 7

    n_targets = len(targets)
    n_enrolled = population[ENROLMENT_KEY].drop_duplicates().shape[0]
    contacted = contacts[ENROLMENT_KEY].drop_duplicates()
    wd = contacted.merge(population[ENROLMENT_KEY + ["withdrawn", "failed_or_withdrew"]], on=ENROLMENT_KEY)
    return {
        "withdrawers_in_window": n_targets,
        "reached_before_silence": float(len(first) / max(n_targets, 1)),
        "median_lead_weeks": float(lead.median()) if len(lead) else np.nan,
        "contacts": int(len(contacts)),
        "contacts_per_100_students": float(100 * len(contacts) / max(n_enrolled, 1)),
        "unique_students_contacted": int(len(contacted)),
        "share_contacted": float(len(contacted) / max(n_enrolled, 1)),
        "contacted_who_withdraw": float(wd["withdrawn"].mean()) if len(wd) else np.nan,
        "contacted_who_fail_or_withdraw": float(wd["failed_or_withdrew"].mean()) if len(wd) else np.nan,
        "contacts_per_student_reached_in_time": float(len(contacts) / max(len(first), 1)),
    }


def reached_by_presentation(
    contacts: pd.DataFrame, population: pd.DataFrame, last_active: pd.Series,
    first_week: int, last_week: int,
) -> pd.DataFrame:
    """Per presentation: withdrawers in the window, and how many were reached before silence."""
    w = population[population["withdrawn"]].merge(
        last_active, left_on=ENROLMENT_KEY, right_index=True, how="left")
    start, end = 7 * first_week, 7 * last_week + 7
    t = w[(w["last_active_day"] >= start) & (w["last_active_day"] < end)][ENROLMENT_KEY + ["last_active_day"]]
    m = t.merge(contacts.assign(contact_day=7 * contacts["week"]), on=ENROLMENT_KEY, how="left")
    m["in_time"] = m["contact_day"] <= m["last_active_day"]
    per_student = m.groupby(ENROLMENT_KEY)["in_time"].any().reset_index()
    return per_student.groupby(PRESENTATION).agg(targets=("in_time", "size"), reached=("in_time", "sum"))


def paired_coverage_difference(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Coverage of policy a minus policy b, with a paired bootstrap over presentations."""
    j = a.join(b, lsuffix="_a", rsuffix="_b")
    arr = j[["targets_a", "reached_a", "reached_b"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s = arr[idx].sum(axis=1)
    d = (s[:, 1] - s[:, 2]) / s[:, 0]
    tot = arr.sum(0)
    return {"difference": float((tot[1] - tot[2]) / tot[0]),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
