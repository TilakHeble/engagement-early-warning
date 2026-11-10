"""Evaluation the way a student-support team would use the system.

Headline: with capacity to contact a fixed share of each cohort (module
presentation) at week k:

  * precision: what share of contacts reach a student who needs it?
  * recall: what share of those students are reached? Also reported as a share
    of the maximum possible, because a 10% capacity cannot reach every student
    when more than 10% need help.
  * warning: weeks from the flag to official withdrawal, and to the student's
    last VLE activity. Official withdrawal often lags real disengagement, so the
    first measure overstates useful warning.

Uncertainty: 95% intervals from a bootstrap over module presentations (the
cohorts are the independent units, not individual students).

AUC is reported for comparison with the literature, but it does not reflect the
capacity limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PRESENTATION = ["code_module", "code_presentation"]


def flag_top(df: pd.DataFrame, score: str, capacity: float, seed: int = 0) -> pd.Series:
    """Flag the highest-risk `capacity` share within each presentation.

    Ties are broken randomly (fixed seed) so that a score with many ties, such as
    "0 clicks", cannot flag more students than the capacity allows.
    """
    rng = np.random.default_rng(seed)
    jitter = pd.Series(rng.random(len(df)) * 1e-9, index=df.index)
    s = df[score] + jitter
    n = df.groupby(PRESENTATION)[score].transform("size")
    rank = s.groupby([df[c] for c in PRESENTATION]).rank(ascending=False, method="first")
    return rank <= np.ceil(capacity * n)


def _per_presentation(df: pd.DataFrame, y: pd.Series, flagged: pd.Series) -> pd.DataFrame:
    """Counts per presentation, so bootstrap resamples are cheap sums."""
    return (
        pd.DataFrame({"pos": y, "flag": flagged, "tp": flagged & y})
        .groupby([df[c] for c in PRESENTATION])
        .sum()
    )


def bootstrap_ci(
    counts: pd.DataFrame, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, tuple[float, float]]:
    """Percentile intervals for recall and precision, resampling whole presentations."""
    rng = np.random.default_rng(seed)
    c = counts[["pos", "flag", "tp"]].to_numpy(dtype=float)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    sums = c[idx].sum(axis=1)  # (n_boot, 3)
    recall = sums[:, 2] / np.maximum(sums[:, 0], 1)
    precision = sums[:, 2] / np.maximum(sums[:, 1], 1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "recall": tuple(np.percentile(recall, [lo, hi])),
        "precision": tuple(np.percentile(precision, [lo, hi])),
    }


def capacity_metrics(
    df: pd.DataFrame,
    score: str,
    capacity: float,
    target: str = "will_withdraw",
    n_boot: int = 1000,
) -> dict[str, float]:
    """Precision, recall, warning time and bootstrap intervals at a fixed contact capacity.

    `target` is 'will_withdraw' (retention) or 'needs_support' (fail or withdraw).
    Warning times are only defined for withdrawals; the last-activity measures
    need a `last_active_day` column (see population.add_last_activity).
    """
    y = df[target].astype(bool)
    flagged = flag_top(df, score, capacity)
    tp = flagged & y
    base_rate = y.mean()
    precision = tp.sum() / max(flagged.sum(), 1)
    recall = tp.sum() / max(y.sum(), 1)
    max_recall = min(1.0, flagged.sum() / max(y.sum(), 1))
    out = {
        "n_at_risk": int(len(df)),
        "base_rate": float(base_rate),
        "flagged": int(flagged.sum()),
        "precision": float(precision),
        "lift": float(precision / base_rate) if base_rate else np.nan,
        "recall": float(recall),
        "max_recall": float(max_recall),
        "recall_share_of_max": float(recall / max_recall) if max_recall else np.nan,
        "auc": float(roc_auc_score(y, df[score])) if y.nunique() == 2 else np.nan,
    }
    caught = tp & df["will_withdraw"].astype(bool)
    out["median_weeks_to_withdrawal"] = float(df.loc[caught, "weeks_until_withdrawal"].median())
    if "weeks_until_last_activity" in df:
        w = df.loc[caught, "weeks_until_last_activity"]
        out["median_weeks_to_last_activity"] = float(w.median())
        out["share_caught_already_inactive"] = float((w <= 0).mean()) if len(w) else np.nan
    if n_boot:
        ci = bootstrap_ci(_per_presentation(df, y, flagged), n_boot=n_boot)
        out["recall_lo"], out["recall_hi"] = ci["recall"]
        out["precision_lo"], out["precision_hi"] = ci["precision"]
    return out


def band_rule_metrics(
    df: pd.DataFrame, bands=("none", "low"), target: str = "will_withdraw"
) -> dict[str, float]:
    """Metrics for 'flag everyone in these bands', where the number flagged is not capped."""
    y = df[target].astype(bool)
    flagged = df["band"].isin(bands)
    tp = flagged & y
    return {
        "share_flagged": float(flagged.mean()),
        "recall": float(tp.sum() / max(y.sum(), 1)),
        "precision": float(tp.sum() / max(flagged.sum(), 1)),
    }
