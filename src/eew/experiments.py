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


# --------------------------------------------------------------------------------------
# Stage 3: landmark model of disengagement onset
# --------------------------------------------------------------------------------------

def run_stage1(n_boot: int = 1000) -> pd.DataFrame:
    """Stage 1 baselines (engagement band, cumulative clicks), as in notebook 01."""
    enr, daily, _, _ = load_inputs()
    pop = P.early_warning_population(enr)
    last_active = P.last_activity_day(daily, pop)
    rows = []
    for week in PREDICTION_WEEKS:
        df = P.add_last_activity(B.baseline_scores(daily, P.at_risk(pop, week), week), last_active)
        for target in TARGETS:
            for score, name in [("risk_band7", "Engagement band (7-day)"), ("risk_cum", "Cumulative clicks")]:
                rows.append({"week": week, "target": target, "baseline": name,
                             **E.capacity_metrics(df, score, CONTACT_CAPACITY, target, n_boot)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    return out


def build_landmark_data(cache: bool = True) -> pd.DataFrame:
    """Stack landmark weeks 1-24 with features and onset labels (cached as parquet)."""
    from eew import landmark as L

    path = PROCESSED_DIR / "landmark_stage3.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    enr, daily, a, sa = load_inputs()
    pop = P.early_warning_population(enr)
    last_active = P.last_activity_day(daily, pop)
    data = pd.concat(
        [L.add_onset_label(week_frame(w, enr, daily, a, sa, pop, last_active)) for w in L.TRAIN_WEEKS],
        ignore_index=True,
    )
    data.to_parquet(path, index=False)
    return data


def run_stage3(n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    from eew import landmark as L

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data = build_landmark_data()
    no_traj = [f for f in F.FEATURES
               if f not in F.FEATURE_GROUPS["trajectory"] + F.FEATURE_GROUPS["recency"]]
    data["p_landmark_gbm"] = L.oof_landmark(data, F.FEATURES, "onset", "gbm")
    data["p_landmark_lr"] = L.oof_landmark(data, F.FEATURES, "onset", "logistic")
    data["p_gbm_no_traj"] = L.oof_landmark(data, no_traj, "onset", "gbm")

    ev = data[data["week"].isin(L.EVAL_WEEKS)].copy()
    ev["p_stage2"] = np.nan
    for w in L.EVAL_WEEKS:
        i = ev["week"] == w
        ev.loc[i, "p_stage2"] = M.oof_predict(ev[i], F.FEATURES, "will_withdraw", "logistic")

    names = {
        "risk_cum": "Baseline: cumulative clicks",
        "p_stage2": "Stage 2 model (per week, eventual withdrawal)",
        "p_landmark_lr": "Stage 3 landmark: logistic",
        "p_landmark_gbm": "Stage 3 landmark: gradient boosting",
    }
    metrics, diffs, calib = [], [], []
    for w in L.EVAL_WEEKS:
        f = ev[ev["week"] == w]
        for score, name in names.items():
            m = E.capacity_metrics(f, score, CONTACT_CAPACITY, "onset", n_boot)
            flagged = E.flag_top(f, score, CONTACT_CAPACITY)
            caught = flagged & f["onset"]
            m["median_weeks_to_onset"] = float(f.loc[caught, "weeks_to_onset"].median())
            m["share_flags_already_silent"] = float((flagged & f["already_silent"]).sum() / flagged.sum())
            metrics.append({"week": w, "score": name, **m})
        for a, b in [("p_landmark_gbm", "risk_cum"), ("p_landmark_gbm", "p_stage2"),
                     ("p_landmark_gbm", "p_landmark_lr"), ("p_landmark_gbm", "p_gbm_no_traj")]:
            diffs.append({"week": w, "model": a, "compared_with": b,
                          **E.paired_difference(f, a, b, CONTACT_CAPACITY, "onset")})
    for score in ["p_landmark_gbm", "p_landmark_lr", "p_stage2"]:
        t = L.calibration_table(ev["onset"].to_numpy(), ev[score].to_numpy())
        calib.append(t.assign(score=names.get(score, score)).reset_index())

    out = {
        "metrics": pd.DataFrame(metrics),
        "differences": pd.DataFrame(diffs),
        "calibration": pd.concat(calib, ignore_index=True),
        "event_rates": data.groupby("week").agg(
            enrolled=("onset", "size"), onset_rate=("onset", "mean"),
            already_silent_rate=("already_silent", "mean")).reset_index(),
    }
    for name, df in out.items():
        df.to_csv(RESULTS_DIR / f"stage3_{name}.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Stage 4: term-long outreach simulation
# --------------------------------------------------------------------------------------

def landmark_predictions(cache: bool = True) -> pd.DataFrame:
    """Out-of-fold stage 3 scores for every landmark week (cached)."""
    from eew import landmark as L

    path = PROCESSED_DIR / "landmark_predictions.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    data = build_landmark_data()
    data["p_landmark_gbm"] = L.oof_landmark(data, F.FEATURES, "onset", "gbm")
    keep = ["code_module", "code_presentation", "id_student", "week", "onset", "already_silent",
            "will_withdraw", "needs_support", "risk_cum", "p_landmark_gbm"]
    out = data[keep]
    out.to_parquet(path, index=False)
    return out


def run_stage4(first_week: int = 1, last_week: int = 24) -> dict[str, pd.DataFrame]:
    from eew import policy as Pol

    scores = landmark_predictions()
    scores = scores.assign(random=np.random.default_rng(0).random(len(scores)))
    enr = pd.read_parquet(PROCESSED_DIR / "enrolments.parquet")
    daily = pd.read_parquet(PROCESSED_DIR / "daily_clicks.parquet")
    pop = P.early_warning_population(enr)
    last_active = P.last_activity_day(daily, pop)
    weeks = list(range(first_week, last_week + 1))

    names = {"p_landmark_gbm": "Stage 3 model", "risk_cum": "Baseline: cumulative clicks",
             "random": "Random"}
    def run(score, capacity, cooldown):
        c = Pol.simulate(scores, score, capacity, cooldown, weeks)
        return {"score": names[score], "capacity": capacity, "cooldown_weeks": cooldown,
                **Pol.summarise(c, pop, last_active, first_week, last_week)}

    # Workload curves: each score across capacities, with a 4-week cool-down.
    curves = pd.DataFrame([run(s, cap, 4) for s in names
                           for cap in [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]])
    # Cool-down sensitivity for the model at 5% weekly capacity.
    cooldowns = pd.DataFrame([run("p_landmark_gbm", 0.05, cd) for cd in [0, 1, 2, 4, 8, 12]])
    # Paired bootstrap of coverage differences at realistic capacities.
    diffs = []
    for capacity in [0.025, 0.05, 0.10]:
        per = {s: Pol.reached_by_presentation(Pol.simulate(scores, s, capacity, 4, weeks),
                                              pop, last_active, first_week, last_week) for s in names}
        for other in ["random", "risk_cum"]:
            diffs.append({"capacity": capacity, "compared_with": names[other],
                          **Pol.paired_coverage_difference(per["p_landmark_gbm"], per[other])})
    out = {"curves": curves, "cooldowns": cooldowns, "differences": pd.DataFrame(diffs)}
    for name, df in out.items():
        df.to_csv(RESULTS_DIR / f"stage4_{name}.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Stage 5: fairness audit
# --------------------------------------------------------------------------------------

def run_stage5(n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    from eew import fairness as Fa
    from eew import landmark as L
    from eew import policy as Pol

    pr = landmark_predictions()
    enr = pd.read_parquet(PROCESSED_DIR / "enrolments.parquet")
    daily = pd.read_parquet(PROCESSED_DIR / "daily_clicks.parquet")
    groups = Fa.add_groups(enr)
    key = ["code_module", "code_presentation", "id_student"]
    ev = pr[pr["week"].isin(L.EVAL_WEEKS)].merge(groups, on=key)

    def weekly_flags(score):
        return pd.concat([E.flag_top(g, score, CONTACT_CAPACITY) for _, g in ev.groupby("week")]).reindex(ev.index)

    ev["p_recal_disability"] = Fa.group_recalibrate(ev, "p_landmark_gbm", "disability")
    f_model, f_recal = weekly_flags("p_landmark_gbm"), weekly_flags("p_recal_disability")

    refs = {"disability": "No declared disability", "imd_quintile": "3 (40-60%)",
            "age_group": "Under 35", "gender": "M", "education": "A level or equivalent"}
    metrics = pd.concat([Fa.group_metrics(ev, a, "p_landmark_gbm", f_model, n_boot=n_boot)
                         for a in Fa.AUDIT_GROUPS], ignore_index=True).assign(score="Stage 3 model")
    metrics_recal = Fa.group_metrics(ev, "disability", "p_recal_disability", f_recal, n_boot=n_boot).assign(
        score="Recalibrated by disability")
    gaps = pd.concat([Fa.gap_vs_reference(ev, a, "p_landmark_gbm", f_model, refs[a], n_boot=n_boot)
                      for a in Fa.AUDIT_GROUPS], ignore_index=True)

    # Term-long coverage by group (stage 4 policy: 5% weekly capacity, 4-week cool-down).
    pop = P.early_warning_population(enr)
    last_active = P.last_activity_day(daily, pop)
    sc = pr.merge(groups[key + ["disability"]], on=key)
    sc["p_recal_disability"] = Fa.group_recalibrate(sc, "p_landmark_gbm", "disability")
    w = pop[pop["withdrawn"]].merge(last_active, left_on=key, right_index=True)
    w = w[(w["last_active_day"] >= 7) & (w["last_active_day"] < 7 * 25)][key + ["last_active_day"]].merge(groups, on=key)
    coverage = []
    for score, name in [("p_landmark_gbm", "Stage 3 model"), ("p_recal_disability", "Recalibrated by disability")]:
        c = Pol.simulate(sc, score, 0.05, 4, list(range(1, 25)))
        m = w.merge(c.assign(contact_day=7 * c["week"]), on=key, how="left")
        m["in_time"] = m["contact_day"] <= m["last_active_day"]
        reached = m.groupby(key)["in_time"].any().reset_index().merge(groups, on=key)
        for a in Fa.AUDIT_GROUPS:
            r = reached.groupby(a)["in_time"].agg(["mean", "size"]).reset_index()
            coverage.append(r.rename(columns={a: "group", "mean": "reached_before_silence",
                                              "size": "withdrawers"}).assign(attribute=a, score=name))
    overall = pd.DataFrame([
        {"score": n, "recall": float((f & ev["onset"]).sum() / ev["onset"].sum()),
         "precision": float((f & ev["onset"]).sum() / f.sum())}
        for n, f in [("Stage 3 model", f_model), ("Recalibrated by disability", f_recal)]
    ])
    out = {"group_metrics": pd.concat([metrics, metrics_recal], ignore_index=True), "gaps": gaps,
           "term_coverage": pd.concat(coverage, ignore_index=True), "overall": overall}
    for name, df in out.items():
        df.to_csv(RESULTS_DIR / f"stage5_{name}.csv", index=False)
    return out


# --------------------------------------------------------------------------------------
# Stage 6: transfer to unseen modules and years
# --------------------------------------------------------------------------------------

def run_stage6() -> dict[str, pd.DataFrame]:
    from eew import transfer as T

    data = build_landmark_data()
    oof = landmark_predictions()
    key = ["code_module", "code_presentation", "id_student", "week"]
    data = data.merge(oof[key + ["p_landmark_gbm"]], on=key, how="left")
    data["p_unseen_module"] = T.leave_one_module_out(data, F.FEATURES)
    with_h, without_h = T.forward_in_time(data, F.FEATURES)
    data["p_2013_with_history"], data["p_2013_without_history"] = with_h, without_h

    cap = CONTACT_CAPACITY
    # 1. Unseen module vs stage 3 validation (other presentations of the same module seen).
    lomo = []
    for module, g in data.groupby("code_module"):
        for score, name in [("p_landmark_gbm", "Stage 3 validation (module seen)"),
                            ("p_unseen_module", "Module never seen")]:
            lomo.append({"module": module, "setting": name, **T.summarise(g, score, cap)})
    for score, name in [("p_landmark_gbm", "Stage 3 validation (module seen)"),
                        ("p_unseen_module", "Module never seen")]:
        lomo.append({"module": "All", "setting": name, **T.summarise(data, score, cap)})
    lomo_diff = T.paired_recall_difference(data, "p_unseen_module", "p_landmark_gbm", cap)

    # 2-3. Forward in time (2014 test), with / without the module's own 2013 history.
    test = data[data["code_presentation"].str.startswith("2014")]
    fwd = []
    for module, g in list(test.groupby("code_module")) + [("All", test)]:
        for score, name in [("p_landmark_gbm", "Stage 3 validation"),
                            ("p_2013_with_history", "Trained on 2013, incl. own module"),
                            ("p_2013_without_history", "Trained on 2013, without own module")]:
            fwd.append({"module": module, "setting": name, **T.summarise(g, score, cap)})
    has_history = test[test["code_module"] != "CCC"]  # CCC has no 2013 run
    diffs = pd.DataFrame([
        {"comparison": "Module never seen - stage 3 validation (all presentations)", **lomo_diff},
        {"comparison": "2013-trained (incl. own module) - stage 3 validation (2014)",
         **T.paired_recall_difference(test, "p_2013_with_history", "p_landmark_gbm", cap)},
        {"comparison": "Own-module history: with - without (2014, modules with a 2013 run)",
         **T.paired_recall_difference(has_history, "p_2013_with_history", "p_2013_without_history", cap)},
    ])
    out = {"unseen_module": pd.DataFrame(lomo), "forward_in_time": pd.DataFrame(fwd), "differences": diffs}
    for name, df in out.items():
        df.to_csv(RESULTS_DIR / f"stage6_{name}.csv", index=False)
    return out


def run_stage6_calibration() -> pd.DataFrame:
    """Module-level calibration in 2014: cold-start model vs one-run local recalibration."""
    from eew import transfer as T

    data = build_landmark_data()
    rec = T.local_recalibration(data, F.FEATURES, shrink_k=100)
    raw = T.local_recalibration(data, F.FEATURES, shrink_k=None)
    rec["p_local_raw"] = raw["p_local_recal"].to_numpy()
    sens = []
    for k in [None, 25, 50, 100, 200, 400]:
        r = T.local_recalibration(data, F.FEATURES, shrink_k=k)
        t = T.calibration_by_module(r, "p_local_recal")
        sens.append({"shrink_k": "none" if k is None else k, "mean_abs_log_ratio": t["abs_log_ratio"].mean(),
                     "max_abs_log_ratio": t["abs_log_ratio"].max()})
    pd.DataFrame(sens).to_csv(RESULTS_DIR / "stage6_shrinkage_sensitivity.csv", index=False)
    oof = landmark_predictions()
    key = ["code_module", "code_presentation", "id_student", "week"]
    rec = rec.merge(oof[key + ["p_landmark_gbm"]], on=key, how="left")
    tabs = []
    for score, name in [("p_landmark_gbm", "Stage 3 validation"), ("p_cold", "Cold start (module never seen)"),
                        ("p_local_raw", "One-run recalibration (raw)"),
                        ("p_local_recal", "One-run recalibration (shrunk, k=100)")]:
        tabs.append(T.calibration_by_module(rec, score).reset_index().assign(setting=name))
    out = pd.concat(tabs, ignore_index=True)
    out.to_csv(RESULTS_DIR / "stage6_module_calibration.csv", index=False)
    return out
