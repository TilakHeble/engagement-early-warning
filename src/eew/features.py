"""Stage 2: research-led features, computed only from data before the week-k cutoff.

Feature groups (engagement research distinguishes the level of engagement from its
pattern and direction):

  level        how much: clicks over the last 7 and 14 days, cumulative clicks
  regularity   how steady: share of days active since the start, active days in the last 14
  recency      how recent: days since the last click, never clicked at all
  trajectory   which direction: recent weekly clicks against the student's OWN earlier
               average (log ratio), and the slope over the last 4 weeks
  relative     cohort percentiles of level and regularity, because modules differ a lot
               in how many clicks are normal
  assessment   behaviour on assessments already due: missed, late, average days late,
               and scores (only scores old enough to have been released, see below)
  context      how early the student registered, study load and previous attempts

Demographics (gender, age, region, education, deprivation, disability) are never
features. They are used only in the fairness audit (stage 5).

Leakage rules:
  * Only clicks on days < cutoff.
  * An assessment is "due" if its deadline is < cutoff. It counts as submitted only
    if date_submitted < cutoff.
  * Scores are only used for submissions made at least SCORE_DELAY_DAYS before the
    cutoff, as a conservative allowance for marking time (OULAD has no release date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eew.data import ENROLMENT_KEY
from eew.population import cutoff_day

PRESENTATION = ["code_module", "code_presentation"]
SCORE_DELAY_DAYS = 14

FEATURE_GROUPS: dict[str, list[str]] = {
    "level": ["log_clicks_last7", "log_clicks_last14", "log_clicks_cum"],
    "regularity": ["active_share", "active_days_last14"],
    "recency": ["days_since_last_click", "never_clicked"],
    "trajectory": ["traj_log_ratio", "traj_slope4", "has_trajectory"],
    "relative": ["pct_clicks_last14", "pct_clicks_cum", "pct_active_share"],
    "assessment": [
        "n_due", "share_missed", "share_late", "mean_days_late",
        "mean_score", "has_score", "clicked_before_start",
    ],
    "context": ["registered_days_before", "studied_credits", "num_of_prev_attempts"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def _sum_window(daily: pd.DataFrame, start: int, end: int, col: str = "clicks") -> pd.Series:
    w = daily[(daily["day"] >= start) & (daily["day"] < end)]
    return w.groupby(ENROLMENT_KEY)[col].sum()


def _align(s: pd.Series, keys: pd.MultiIndex, fill=0.0) -> np.ndarray:
    return s.reindex(keys).fillna(fill).to_numpy(dtype=float)


def click_features(daily: pd.DataFrame, at_risk: pd.DataFrame, week: int) -> pd.DataFrame:
    c = cutoff_day(week)
    keys = pd.MultiIndex.from_frame(at_risk[ENROLMENT_KEY])
    seen = daily[daily["day"] < c]
    out = pd.DataFrame(index=at_risk.index)

    last7 = _align(_sum_window(seen, c - 7, c), keys)
    last14 = _align(_sum_window(seen, c - 14, c), keys)
    cum = _align(seen.groupby(ENROLMENT_KEY)["clicks"].sum(), keys)
    out["log_clicks_last7"] = np.log1p(last7)
    out["log_clicks_last14"] = np.log1p(last14)
    out["log_clicks_cum"] = np.log1p(cum)

    in_term = seen[seen["day"] >= 0]
    active_days = _align(in_term.groupby(ENROLMENT_KEY)["day"].nunique(), keys)
    out["active_share"] = active_days / c
    recent = seen[seen["day"] >= c - 14]
    out["active_days_last14"] = _align(recent.groupby(ENROLMENT_KEY)["day"].nunique(), keys)

    last_day = seen.groupby(ENROLMENT_KEY)["day"].max().reindex(keys).to_numpy(dtype=float)
    out["never_clicked"] = np.isnan(last_day).astype(float)
    # Never clicked: as long ago as possible (since the earliest possible activity).
    out["days_since_last_click"] = np.where(np.isnan(last_day), c + 30, c - last_day)
    out["clicked_before_start"] = (_align(_sum_window(seen, -10_000, 0), keys) > 0).astype(float)

    # Trajectory: weekly clicks in the last 2 weeks against the student's own average
    # over the weeks before that (from week -2, when the VLE opens). Needs >= 2 earlier weeks.
    earlier_weeks = (c - 14 - (-14)) / 7
    has_traj = earlier_weeks >= 2
    earlier = _align(_sum_window(seen, -14, c - 14), keys)
    recent_rate = last14 / 2
    earlier_rate = earlier / earlier_weeks if has_traj else np.zeros(len(keys))
    out["traj_log_ratio"] = (np.log1p(recent_rate) - np.log1p(earlier_rate)) if has_traj else 0.0
    out["has_trajectory"] = float(has_traj)

    # Slope of log weekly clicks over the last 4 weeks (or fewer if not yet available).
    n_weeks = int(min(4, max(1, (c + 14) // 7)))
    weekly = np.column_stack([
        np.log1p(_align(_sum_window(seen, c - 7 * (i + 1), c - 7 * i), keys))
        for i in reversed(range(n_weeks))
    ])
    if n_weeks >= 2:
        t = np.arange(n_weeks) - (n_weeks - 1) / 2
        out["traj_slope4"] = weekly @ t / (t @ t)
    else:
        out["traj_slope4"] = 0.0

    grp = at_risk[PRESENTATION].assign(
        l14=last14, cum=cum, act=out["active_share"].to_numpy()
    ).groupby(PRESENTATION)
    out["pct_clicks_last14"] = grp["l14"].rank(pct=True).to_numpy()
    out["pct_clicks_cum"] = grp["cum"].rank(pct=True).to_numpy()
    out["pct_active_share"] = grp["act"].rank(pct=True).to_numpy()
    return out


def assessment_features(
    assessments: pd.DataFrame, student_assessment: pd.DataFrame, at_risk: pd.DataFrame, week: int
) -> pd.DataFrame:
    c = cutoff_day(week)
    due = assessments[(assessments["assessment_type"] != "Exam") & (assessments["date"] < c)]
    # Every at-risk student x every assessment already due on their presentation.
    grid = at_risk[ENROLMENT_KEY].reset_index().merge(due, on=PRESENTATION)
    subs = student_assessment[student_assessment["date_submitted"] < c]
    grid = grid.merge(
        subs[["id_assessment", "id_student", "date_submitted", "score"]],
        on=["id_assessment", "id_student"], how="left",
    )
    grid["submitted"] = grid["date_submitted"].notna()
    grid["late"] = grid["submitted"] & (grid["date_submitted"] > grid["date"])
    grid["days_late"] = (grid["date_submitted"] - grid["date"]).clip(lower=0)
    score_known = grid["submitted"] & (grid["date_submitted"] < c - SCORE_DELAY_DAYS)
    grid["known_score"] = grid["score"].where(score_known)

    g = grid.groupby("index")
    out = pd.DataFrame(index=at_risk.index)
    n_due = g.size().reindex(out.index).fillna(0)
    out["n_due"] = n_due
    out["share_missed"] = (g["submitted"].apply(lambda s: (~s).mean())).reindex(out.index).fillna(0)
    out["share_late"] = g["late"].mean().reindex(out.index).fillna(0)
    out["mean_days_late"] = g["days_late"].mean().reindex(out.index).fillna(0)
    ms = g["known_score"].mean().reindex(out.index)
    out["has_score"] = ms.notna().astype(float)
    # No known score yet: use a neutral 0 and let has_score carry the information.
    out["mean_score"] = (ms.fillna(0) / 100).to_numpy()
    return out


def context_features(enrolments: pd.DataFrame, at_risk: pd.DataFrame) -> pd.DataFrame:
    ctx = at_risk[ENROLMENT_KEY].merge(
        enrolments[ENROLMENT_KEY + ["date_registration", "studied_credits", "num_of_prev_attempts"]],
        on=ENROLMENT_KEY, how="left",
    )
    out = pd.DataFrame(index=at_risk.index)
    reg = ctx["date_registration"].to_numpy(dtype=float)
    out["registered_days_before"] = np.nan_to_num(-reg, nan=np.nanmedian(-reg))
    out["studied_credits"] = ctx["studied_credits"].to_numpy(dtype=float)
    out["num_of_prev_attempts"] = ctx["num_of_prev_attempts"].to_numpy(dtype=float)
    return out


def build_features(
    daily: pd.DataFrame,
    enrolments: pd.DataFrame,
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    at_risk: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    """All stage 2 features for one prediction week, aligned to `at_risk` rows."""
    feats = pd.concat(
        [
            click_features(daily, at_risk, week),
            assessment_features(assessments, student_assessment, at_risk, week),
            context_features(enrolments, at_risk),
        ],
        axis=1,
    )
    return pd.concat([at_risk, feats[FEATURES]], axis=1)
