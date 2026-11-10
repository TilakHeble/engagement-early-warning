"""Stage 5: fairness audit of the stage 3 model and the stage 4 outreach policy.

Demographics are never model inputs (see features.py), but a model can still treat
groups differently if their VLE behaviour differs. Groups are joined back to
held-out predictions only here, for auditing.

For each group, at a fixed contact capacity:
  * flag rate                 share of the group flagged
  * miss rate (1 - recall)    share of upcoming disengagements in the group NOT flagged
  * precision                 share of the group's flags that are correct
  * calibration ratio         mean predicted risk / observed onset rate (1 = calibrated)
  * AUC                       ranking quality within the group

Intervals: 95% bootstrap over module presentations (clusters), with the same
draws used for every group so gaps between groups can be compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eew.data import ENROLMENT_KEY

PRESENTATION = ["code_module", "code_presentation"]
AUDIT_GROUPS = ["disability", "imd_quintile", "age_group", "gender", "education"]


def add_groups(enrolments: pd.DataFrame) -> pd.DataFrame:
    """Audit groupings, merging categories too small to measure reliably."""
    e = enrolments[ENROLMENT_KEY + ["disability", "imd_band", "age_band", "gender",
                                    "highest_education"]].copy()
    imd = e["imd_band"].astype("string").str.replace("%", "", regex=False)
    lower = pd.to_numeric(imd.str.split("-").str[0], errors="coerce")
    labels = {0: "1 most deprived (0-20%)", 1: "2 (20-40%)", 2: "3 (40-60%)", 3: "4 (60-80%)",
              4: "5 least deprived (80-100%)"}
    e["imd_quintile"] = (lower // 20).map(labels).fillna("Unknown")
    e["age_group"] = e["age_band"].map({"0-35": "Under 35", "35-55": "35 and over", "55<=": "35 and over"})
    e["education"] = e["highest_education"].map({
        "No Formal quals": "Below A level", "Lower Than A Level": "Below A level",
        "A Level or Equivalent": "A level or equivalent",
        "HE Qualification": "HE qualification or above",
        "Post Graduate Qualification": "HE qualification or above",
    })
    e["disability"] = e["disability"].map({"Y": "Declared disability", "N": "No declared disability"})
    return e[ENROLMENT_KEY + AUDIT_GROUPS]


def _counts(df: pd.DataFrame, group: str, flagged: pd.Series, target: str, score: str) -> pd.DataFrame:
    """Per presentation x group sums, so that bootstrap resampling is cheap."""
    y = df[target].astype(bool)
    return (
        pd.DataFrame({
            "n": 1, "pos": y, "flag": flagged, "tp": flagged & y, "pred": df[score],
            group: df[group], **{c: df[c] for c in PRESENTATION},
        })
        .groupby(PRESENTATION + [group]).sum()
    )


def group_metrics(
    df: pd.DataFrame, group: str, score: str, flagged: pd.Series, target: str = "onset",
    n_boot: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Fairness metrics by group, with clustered bootstrap intervals."""
    c = _counts(df, group, flagged, target, score)
    pres = c.index.droplevel(group).unique()
    groups = sorted(df[group].dropna().unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pres), size=(n_boot, len(pres)))

    def stats(tab: pd.DataFrame) -> pd.DataFrame:
        s = tab.groupby(level=group).sum()
        return pd.DataFrame({
            "flag_rate": s["flag"] / s["n"],
            "miss_rate": 1 - s["tp"] / s["pos"].clip(lower=1),
            "precision": s["tp"] / s["flag"].clip(lower=1),
            "base_rate": s["pos"] / s["n"],
            "calibration_ratio": (s["pred"] / s["n"]) / (s["pos"] / s["n"]).clip(lower=1e-9),
        }).reindex(groups)

    point = stats(c)
    boot = []
    by_pres = {p: c.loc[p] for p in pres}
    for d in draws:
        tab = pd.concat([by_pres[pres[i]] for i in d])
        boot.append(stats(tab).to_numpy())
    boot = np.stack(boot)
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    out = point.copy()
    for j, col in enumerate(point.columns):
        out[f"{col}_lo"], out[f"{col}_hi"] = lo[:, j], hi[:, j]

    sizes = df.groupby(group).agg(rows=(target, "size"), events=(target, "sum"))
    aucs = {g: roc_auc_score(x[target], x[score]) if x[target].nunique() == 2 else np.nan
            for g, x in df.groupby(group)}
    out = out.join(sizes).assign(auc=pd.Series(aucs))
    out.index.name = "group"
    return out.reset_index().assign(attribute=group)


def gap_vs_reference(
    df: pd.DataFrame, group: str, score: str, flagged: pd.Series, reference: str,
    target: str = "onset", n_boot: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Miss-rate and calibration gaps of each group vs a reference group, paired bootstrap."""
    c = _counts(df, group, flagged, target, score)
    pres = c.index.droplevel(group).unique()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pres), size=(n_boot, len(pres)))
    by_pres = {p: c.loc[p] for p in pres}

    def miss_cal(tab):
        s = tab.groupby(level=0).sum()
        miss = 1 - s["tp"] / s["pos"].clip(lower=1)
        cal = (s["pred"] / s["n"]) / (s["pos"] / s["n"]).clip(lower=1e-9)
        return miss, cal

    m0, c0 = miss_cal(c.droplevel(PRESENTATION))
    rows = []
    boot = [miss_cal(pd.concat([by_pres[pres[i]] for i in d])) for d in draws]
    for g in m0.index:
        if g == reference:
            continue
        dm = np.array([m[g] - m[reference] for m, _ in boot if g in m and reference in m])
        dc = np.array([cl[g] - cl[reference] for _, cl in boot if g in cl and reference in cl])
        rows.append({
            "attribute": group, "group": g, "reference": reference,
            "miss_rate_gap": m0[g] - m0[reference],
            "miss_gap_lo": np.percentile(dm, 2.5), "miss_gap_hi": np.percentile(dm, 97.5),
            "calibration_gap": c0[g] - c0[reference],
            "cal_gap_lo": np.percentile(dc, 2.5), "cal_gap_hi": np.percentile(dc, 97.5),
        })
    return pd.DataFrame(rows)


def group_recalibrate(
    df: pd.DataFrame, score: str, group: str, target: str = "onset", n_splits: int = 5
) -> np.ndarray:
    """Cross-fitted group recalibration: scale each group's odds so predicted = observed.

    The scaling factor for each group is estimated on *other* presentations (grouped
    folds) and applied to held-out ones, so the evaluation stays honest. This uses a
    protected attribute at decision time; whether that is acceptable is a policy and
    legal question for the institution, so it is reported as an option, not a default.
    """
    from sklearn.model_selection import GroupKFold

    p = df[score].clip(1e-6, 1 - 1e-6).to_numpy()
    odds = p / (1 - p)
    out = np.empty(len(df))
    pres = (df["code_module"] + "_" + df["code_presentation"]).to_numpy()
    for train, test in GroupKFold(n_splits=n_splits).split(df, groups=pres):
        tr = df.iloc[train]
        factor = (tr.groupby(group)[target].mean() / tr.groupby(group)[score].mean()).to_dict()
        f = df.iloc[test][group].map(factor).fillna(1.0).to_numpy()
        o = odds[test] * f
        out[test] = o / (1 + o)
    return out
