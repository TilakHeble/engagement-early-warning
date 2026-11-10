"""Stage 2 experiment: do the research-led features beat the engagement baseline?

Run with `python -m eew.experiments`. Writes results/stage2_*.csv, used by
notebooks/02_features.ipynb.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from eew import baseline as B
from eew import data as D
from eew import evaluate as E
from eew import features as F
from eew import model as M
from eew import population as P
from eew.config import CONTACT_CAPACITY, PREDICTION_WEEKS, PROCESSED_DIR, RESULTS_DIR

TARGETS = ["will_withdraw", "needs_support"]
ABLATION_WEEKS = [4, 8, 12]


def load_inputs():
    enr = pd.read_parquet(PROCESSED_DIR / "enrolments.parquet")
    daily = pd.read_parquet(PROCESSED_DIR / "daily_clicks.parquet")
    assessments = D.load_table("assessments")
    student_assessment = D.load_table("studentAssessment")
    return enr, daily, assessments, student_assessment


def week_frame(week, enr, daily, assessments, student_assessment, pop, last_active) -> pd.DataFrame:
    """At-risk students for one week, with features, the baseline score and last activity."""
    ar = P.at_risk(pop, week)
    f = F.build_features(daily, enr, assessments, student_assessment, ar, week)
    f = f.join(B.baseline_scores(daily, ar, week)[["risk_cum", "risk_band7"]])
    return P.add_last_activity(f, last_active)


def run_stage2(n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    enr, daily, a, sa = load_inputs()
    pop = P.early_warning_population(enr)
    last_active = P.last_activity_day(daily, pop)

    metrics, diffs, ablation, coefs = [], [], [], []
    for week in PREDICTION_WEEKS:
        f = week_frame(week, enr, daily, a, sa, pop, last_active)
        for target in TARGETS:
            f["p_model"] = M.oof_predict(f, F.FEATURES, target, "logistic")
            for score, name in [("risk_cum", "Baseline: cumulative clicks"), ("p_model", "Stage 2 model")]:
                metrics.append({"week": week, "target": target, "score": name,
                                **E.capacity_metrics(f, score, CONTACT_CAPACITY, target, n_boot)})
            diffs.append({"week": week, "target": target,
                          **E.paired_difference(f, "p_model", "risk_cum", CONTACT_CAPACITY, target)})

            if target == "will_withdraw" and week in ABLATION_WEEKS:
                full = E.capacity_metrics(f, "p_model", CONTACT_CAPACITY, target, n_boot=0)
                for group, cols in F.FEATURE_GROUPS.items():
                    keep = [c for c in F.FEATURES if c not in cols]
                    f["p_drop"] = M.oof_predict(f, keep, target, "logistic")
                    m = E.capacity_metrics(f, "p_drop", CONTACT_CAPACITY, target, n_boot=0)
                    ablation.append({"week": week, "dropped_group": group,
                                     "auc_loss": full["auc"] - m["auc"],
                                     "precision_loss": full["precision"] - m["precision"]})
                # Coefficients from a model fitted on all presentations (for interpretation only).
                mdl = M.make_model("logistic").fit(f[F.FEATURES].to_numpy(float), f[target].astype(int))
                lr = mdl[-1]
                for feat, coef in zip(F.FEATURES, lr.coef_[0]):
                    group = next(g for g, cols in F.FEATURE_GROUPS.items() if feat in cols)
                    coefs.append({"week": week, "feature": feat, "group": group, "std_coef": coef})

    out = {
        "metrics": pd.DataFrame(metrics),
        "differences": pd.DataFrame(diffs),
        "ablation": pd.DataFrame(ablation),
        "coefficients": pd.DataFrame(coefs),
    }
    for name, df in out.items():
        df.to_csv(RESULTS_DIR / f"stage2_{name}.csv", index=False)
    return out


if __name__ == "__main__":
    res = run_stage2()
    d = res["differences"]
    print(d.round(3).to_string())
