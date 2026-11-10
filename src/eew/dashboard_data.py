"""Stage 7: data for the tutor dashboard.

The dashboard runs on a small precomputed snapshot (dashboards/snapshot/), so it can be
deployed without the raw OULAD files. The snapshot covers the 2014 presentations
(weeks 1-24), with held-out stage 3 risk scores.

Warning signs are simple, transparent rules on the same features the model uses. They
are shown to tutors as observable signals to discuss with the student. They are NOT
an explanation of the model's internal reasoning, and the dashboard says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eew.config import PROCESSED_DIR, RESULTS_DIR, ROOT

SNAPSHOT_DIR = ROOT / "dashboards" / "snapshot"
MAX_SIGNS = 3


def warning_signs(row: pd.Series) -> list[str]:
    """Up to three plain-English signals, most important first (assessment, then activity)."""
    signs = []
    n_due = int(row.get("n_due", 0))
    if n_due > 0 and row.get("share_missed", 0) > 0:
        k = int(round(row["share_missed"] * n_due))
        signs.append(f"Missed {k} of {n_due} assessment{'s' if n_due > 1 else ''}")
    if row.get("has_score", 0) == 1 and row.get("mean_score", 1) < 0.40:
        signs.append(f"Avg score {row['mean_score'] * 100:.0f}% (below 40%)")
    if row.get("never_clicked", 0) == 1:
        signs.append("No VLE activity yet")
    elif row.get("days_since_last_click", 0) >= 10:
        signs.append(f"Silent for {int(row['days_since_last_click'])} days")
    if n_due > 0 and row.get("share_late", 0) > 0:
        k = int(round(row["share_late"] * n_due))
        signs.append(f"{k} assessment{'s' if k > 1 else ''} late")
    if row.get("never_clicked", 0) == 0 and row.get("pct_clicks_last14", 1) <= 0.20:
        signs.append("Bottom 20% for recent activity")
    if row.get("has_trajectory", 0) == 1 and row.get("traj_log_ratio", 0) <= np.log(0.5):
        signs.append("Activity halved vs earlier weeks")
    if row.get("never_clicked", 0) == 0 and row.get("active_days_last14", 14) <= 2:
        signs.append(f"Active {int(row['active_days_last14'])} of last 14 days")
    return signs[:MAX_SIGNS] or ["No single sign: overall pattern"]


def build_snapshot(out_dir: Path = SNAPSHOT_DIR) -> dict[str, Path]:
    """Write the dashboard snapshot: student-weeks for 2014 presentations, plus result tables."""
    from eew.experiments import build_landmark_data, landmark_predictions

    data = build_landmark_data()
    oof = landmark_predictions()
    key = ["code_module", "code_presentation", "id_student", "week"]
    df = data[data["code_presentation"].str.startswith("2014")].merge(
        oof[key + ["p_landmark_gbm"]], on=key, how="left")
    enr = pd.read_parquet(PROCESSED_DIR / "enrolments.parquet")[
        ["code_module", "code_presentation", "id_student", "final_result"]]
    df = df.merge(enr, on=["code_module", "code_presentation", "id_student"], how="left")
    df["risk"] = df["p_landmark_gbm"]
    df["risk_rank_pct"] = df.groupby(["code_module", "code_presentation", "week"])["risk"].rank(
        ascending=False, pct=True)
    df["warning_signs"] = df.apply(lambda r: " | ".join(warning_signs(r)), axis=1)
    keep = key + ["risk", "risk_rank_pct", "warning_signs", "onset", "already_silent", "will_withdraw",
                  "weeks_to_onset", "final_result", "n_due", "share_missed", "days_since_last_click"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"students": out_dir / "students.parquet"}
    df[keep].to_parquet(paths["students"], index=False, compression="zstd")
    for name in ["stage4_curves", "stage4_cooldowns", "stage5_group_metrics", "stage6_unseen_module",
                 "stage6_module_calibration", "stage3_metrics"]:
        src = RESULTS_DIR / f"{name}.csv"
        dst = out_dir / f"{name}.csv"
        dst.write_text(src.read_text())
        paths[name] = dst
    return paths


if __name__ == "__main__":
    for name, p in build_snapshot().items():
        print(f"{name}: {p} ({p.stat().st_size / 1e6:.2f} MB)")
