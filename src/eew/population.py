"""Who the early-warning system is for, and what counts as a correct flag.

Rules (see docs/build_log.md for the reasoning):
  * Population: enrolments still registered at day 0. Withdrawals dated before
    day 0, or with no date, cannot be predicted from in-course behaviour, so
    they are excluded from model metrics and reported separately.
  * At week k the system can only use data from days < 7k ("the cutoff").
  * The at-risk set at week k is everyone in the population not yet withdrawn
    by the cutoff.
  * A flag at week k is correct if that student withdraws on or after the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eew.data import ENROLMENT_KEY


def cutoff_day(week: int) -> int:
    """First day the model at `week` is NOT allowed to see."""
    return 7 * week


def early_warning_population(enrolments: pd.DataFrame) -> pd.DataFrame:
    """Drop withdrawals that happened before the module started or have no date."""
    pre_start = enrolments["withdrawn"] & (
        enrolments["withdrawal_day"].isna() | (enrolments["withdrawal_day"] < 0)
    )
    return enrolments.loc[~pre_start].copy()


def excluded_summary(enrolments: pd.DataFrame) -> dict[str, int]:
    """Counts of withdrawals left out of model metrics, for transparent reporting."""
    w = enrolments[enrolments["withdrawn"]]
    return {
        "withdrawn_total": int(len(w)),
        "withdrawn_before_start": int((w["withdrawal_day"] < 0).sum()),
        "withdrawn_no_date": int(w["withdrawal_day"].isna().sum()),
    }


def at_risk(population: pd.DataFrame, week: int) -> pd.DataFrame:
    """Students still enrolled at the week-k cutoff, with the label `will_withdraw`."""
    c = cutoff_day(week)
    gone = population["withdrawn"] & (population["withdrawal_day"] < c)
    out = population.loc[
        ~gone, ENROLMENT_KEY + ["withdrawn", "withdrawal_day", "failed_or_withdrew"]
    ].copy()
    out["will_withdraw"] = out["withdrawn"]
    # Broader target: will fail or withdraw ("needs support").
    out["needs_support"] = out["failed_or_withdrew"]
    out["weeks_until_withdrawal"] = (out["withdrawal_day"] - c) / 7
    out["week"] = week
    return out


def last_activity_day(daily: pd.DataFrame, enrolments: pd.DataFrame) -> pd.Series:
    """Day of each withdrawn student's last VLE click before their official withdrawal.

    Used ONLY for evaluation (it looks into the future), never as a model input.
    Official withdrawal often lags real disengagement, so this gives an honest
    measure of how much useful warning a flag gives. NaN if the student never clicked.
    """
    w = enrolments.loc[enrolments["withdrawn"], ENROLMENT_KEY + ["withdrawal_day"]]
    d = daily.merge(w, on=ENROLMENT_KEY)
    d = d[d["day"] < d["withdrawal_day"]]
    return d.groupby(ENROLMENT_KEY)["day"].max().rename("last_active_day")


def add_last_activity(at_risk_df: pd.DataFrame, last_active: pd.Series) -> pd.DataFrame:
    """Add `weeks_until_last_activity` (from the week-k cutoff) for withdrawn students.

    Values <= 0 mean the student had already gone silent by the time of the flag.
    Students who never clicked count as silent from the start (day -inf -> negative).
    """
    out = at_risk_df.merge(last_active, left_on=ENROLMENT_KEY, right_index=True, how="left")
    c = cutoff_day(int(out["week"].iloc[0])) if len(out) else 0
    silent = out["will_withdraw"] & out["last_active_day"].isna()
    out["weeks_until_last_activity"] = (out["last_active_day"] - c) / 7
    out.loc[silent, "weeks_until_last_activity"] = -np.inf
    out.index = at_risk_df.index
    return out
