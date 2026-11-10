"""Stage 7 tests: warning-sign rules."""

import numpy as np
import pandas as pd

from eew.dashboard_data import warning_signs

BASE = dict(n_due=3, share_missed=0.0, share_late=0.0, has_score=0, mean_score=0.0, never_clicked=0,
            days_since_last_click=1, pct_clicks_last14=0.6, has_trajectory=1, traj_log_ratio=0.0,
            active_days_last14=8)


def signs(**kw):
    return warning_signs(pd.Series({**BASE, **kw}))


def test_no_signs_message():
    assert signs() == ["No single sign: overall pattern"]


def test_assessment_signs_first_and_capped_at_three():
    s = signs(share_missed=2 / 3, has_score=1, mean_score=0.30, days_since_last_click=15,
              share_late=1 / 3, pct_clicks_last14=0.1, traj_log_ratio=np.log(0.2), active_days_last14=1)
    assert s == ["Missed 2 of 3 assessments", "Avg score 30% (below 40%)", "Silent for 15 days"]


def test_never_clicked_replaces_inactivity_and_skips_activity_rules():
    s = signs(never_clicked=1, days_since_last_click=40, pct_clicks_last14=0.05, active_days_last14=0)
    assert s == ["No VLE activity yet"]


def test_trajectory_needs_history():
    assert "Activity halved" not in " ".join(signs(has_trajectory=0, traj_log_ratio=np.log(0.1)))
    assert "Activity halved vs earlier weeks" in " ".join(signs(traj_log_ratio=np.log(0.4)))
