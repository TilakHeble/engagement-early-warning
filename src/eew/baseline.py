"""Stage 1 baselines: descriptive engagement scores of the kind current platforms show.

Two baselines, both computed only from clicks before the week-k cutoff:

  * Engagement band (7-day): each student's clicks over the last 7 days, compared
    with others on the same module presentation. Banded as
    none (0 clicks) / low (<25th percentile of active students) / medium (25th-75th) /
    high (75th-90th) / very high (>90th). Risk score = cohort percentile of
    last-7-day clicks (lower = riskier).
  * Cumulative clicks: total clicks since the start, as a cohort percentile.

These answer "how engaged is this student right now?", which is the question a
daily engagement score answers. Stage 3 asks "how likely is this student to
withdraw?" instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eew.data import ENROLMENT_KEY
from eew.population import cutoff_day

PRESENTATION = ["code_module", "code_presentation"]
BANDS = ["none", "low", "medium", "high", "very high"]


def clicks_before(daily: pd.DataFrame, at_risk: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Total clicks per at-risk enrolment over days start <= day < end (0 if none)."""
    window = daily[(daily["day"] >= start) & (daily["day"] < end)]
    totals = window.groupby(ENROLMENT_KEY)["clicks"].sum()
    idx = pd.MultiIndex.from_frame(at_risk[ENROLMENT_KEY])
    return pd.Series(totals.reindex(idx).fillna(0).to_numpy(), index=at_risk.index)


def assign_band(last7: pd.Series, groups: pd.DataFrame) -> pd.Series:
    """Band each student's last-7-day clicks relative to active students in the same presentation."""
    band = pd.Series("none", index=last7.index, dtype=object)
    df = groups.assign(c=last7)
    for _, g in df.groupby(PRESENTATION):
        active = g["c"] > 0
        if not active.any():
            continue
        q25, q75, q90 = np.percentile(g.loc[active, "c"], [25, 75, 90])
        c = g.loc[active, "c"]
        band.loc[c.index] = np.select(
            [c < q25, c <= q75, c <= q90], ["low", "medium", "high"], default="very high"
        )
    return pd.Categorical(band, categories=BANDS, ordered=True)


def baseline_scores(daily: pd.DataFrame, at_risk: pd.DataFrame, week: int) -> pd.DataFrame:
    """Add baseline features and risk scores (higher = more at risk) for week k."""
    c = cutoff_day(week)
    out = at_risk.copy()
    out["clicks_last7"] = clicks_before(daily, out, c - 7, c)
    out["clicks_cum"] = clicks_before(daily, out, -10_000, c)
    out["band"] = assign_band(out["clicks_last7"], out[PRESENTATION])
    grp = out.groupby(PRESENTATION)
    # Risk = 1 - cohort percentile of engagement, so low engagement scores high.
    out["risk_band7"] = 1 - grp["clicks_last7"].rank(pct=True, method="average")
    out["risk_cum"] = 1 - grp["clicks_cum"].rank(pct=True, method="average")
    return out
