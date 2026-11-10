"""Stage 6: does the stage 3 model transfer to modules and years it has not seen?

Stage 3 validation held out whole presentations, but the same *modules* appeared in
training (e.g. train on BBB-2013J, test on BBB-2014J). A new university brings
modules with their own designs, click norms and assessment schedules. Three tests:

  1. Leave one module out: train on 6 modules, test on the 7th.
  2. Forward in time: train on 2013 presentations, test on 2014 (as a deployment would).
  3. Local history: for each 2014 module, compare training on 2013 data that includes
     that module's own earlier run against training on 2013 data without it.

All use the stage 3 landmark gradient-boosting model, the onset target and 10% weekly
contact capacity, evaluated at landmark weeks 2-12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eew import evaluate as E
from eew.landmark import EVAL_WEEKS, LandmarkModel

PRESENTATION = ["code_module", "code_presentation"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                target: str = "onset", kind: str = "gbm") -> np.ndarray:
    m = LandmarkModel(kind).fit(
        train[features].to_numpy(float), train["week"].to_numpy(float), train[target].astype(int).to_numpy()
    )
    return m.predict_proba(test[features].to_numpy(float), test["week"].to_numpy(float))


def weekly_counts(df: pd.DataFrame, score: str, capacity: float, target: str = "onset") -> pd.DataFrame:
    """Per presentation, summed over evaluation weeks: events, flags, hits, predicted and rows."""
    parts = []
    for _, g in df[df["week"].isin(EVAL_WEEKS)].groupby("week"):
        f = E.flag_top(g, score, capacity)
        y = g[target].astype(bool)
        parts.append(pd.DataFrame({
            **{c: g[c] for c in PRESENTATION}, "n": 1, "pos": y, "flag": f, "tp": f & y, "pred": g[score],
        }))
    return pd.concat(parts).groupby(PRESENTATION).sum()


def summarise(df: pd.DataFrame, score: str, capacity: float, target: str = "onset") -> dict[str, float]:
    ev = df[df["week"].isin(EVAL_WEEKS)]
    c = weekly_counts(df, score, capacity, target).sum()
    y = ev[target].astype(int)
    return {
        "rows": int(c["n"]), "events": int(c["pos"]),
        "recall": float(c["tp"] / max(c["pos"], 1)),
        "precision": float(c["tp"] / max(c["flag"], 1)),
        "auc": float(roc_auc_score(y, ev[score])) if y.nunique() == 2 else np.nan,
        "calibration_ratio": float(c["pred"] / max(c["pos"], 1)),
    }


def paired_recall_difference(df: pd.DataFrame, score_a: str, score_b: str, capacity: float,
                             target: str = "onset", n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Recall of a minus b over evaluation weeks, paired bootstrap over presentations."""
    a = weekly_counts(df, score_a, capacity, target)
    b = weekly_counts(df, score_b, capacity, target).reindex(a.index)
    arr = np.column_stack([a["pos"], a["tp"], b["tp"]]).astype(float)
    rng = np.random.default_rng(seed)
    s = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].sum(axis=1)
    d = (s[:, 1] - s[:, 2]) / np.maximum(s[:, 0], 1)
    tot = arr.sum(0)
    return {"difference": float((tot[1] - tot[2]) / max(tot[0], 1)),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def leave_one_module_out(data: pd.DataFrame, features: list[str]) -> pd.Series:
    """Predictions for every row from a model that never saw that row's module."""
    pred = pd.Series(np.nan, index=data.index)
    for module in sorted(data["code_module"].unique()):
        test = data["code_module"] == module
        pred[test] = fit_predict(data[~test], data[test], features)
    return pred


def forward_in_time(data: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    """2014 predictions from 2013-only training: with, and without, the module's own 2013 history."""
    train_years = data["code_presentation"].str.startswith("2013")
    test = data[~train_years]
    with_history = pd.Series(fit_predict(data[train_years], test, features), index=test.index)
    without_history = pd.Series(np.nan, index=test.index)
    for module in sorted(test["code_module"].unique()):
        tr = data[train_years & (data["code_module"] != module)]
        idx = test.index[test["code_module"] == module]
        without_history[idx] = fit_predict(tr, data.loc[idx], features)
    return with_history, without_history


def local_recalibration(
    data: pd.DataFrame, features: list[str], target: str = "onset", shrink_k: float | None = 100.0
) -> pd.DataFrame:
    """Cold-start model + one completed local run for calibration.

    For each module with a 2013 run: train on 2013 data from OTHER modules; predict the
    module's 2013 run (now complete, so outcomes are known) and its 2014 run. Scale the
    2014 odds by observed/predicted on the 2013 run. Within a presentation this keeps
    the ranking (so recall is unchanged) and corrects the probability level.
    The correction is shrunk towards 1 when the local run had few events:
    log(factor) is multiplied by events / (events + shrink_k). A run with almost no
    disengagements (e.g. GGG-2013J, 0.3% of student-weeks) otherwise gives a noisy factor.
    `shrink_k=None` applies the raw factor.
    Returns the 2014 rows with `p_cold` and `p_local_recal`.
    """
    is2013 = data["code_presentation"].str.startswith("2013")
    out = []
    for module in sorted(data.loc[is2013, "code_module"].unique()):
        train = data[is2013 & (data["code_module"] != module)]
        hist = data[is2013 & (data["code_module"] == module)]
        new = data[~is2013 & (data["code_module"] == module)]
        if new.empty:
            continue
        p_hist = fit_predict(train, pd.concat([hist, new]), features)
        ph, pn = p_hist[: len(hist)], p_hist[len(hist):]
        factor = hist[target].mean() / ph.mean()
        if shrink_k is not None:
            events = hist[target].sum()
            factor = float(np.exp(np.log(factor) * events / (events + shrink_k)))
        pn_c = np.clip(pn, 1e-6, 1 - 1e-6)
        odds = pn_c / (1 - pn_c) * factor
        out.append(new.assign(p_cold=pn, p_local_recal=odds / (1 + odds), recal_factor=factor))
    return pd.concat(out)


def calibration_by_module(df: pd.DataFrame, score: str, target: str = "onset") -> pd.DataFrame:
    """Calibration ratio (predicted / observed) per module over evaluation weeks."""
    ev = df[df["week"].isin(EVAL_WEEKS)]
    t = ev.groupby("code_module").agg(predicted=(score, "mean"), observed=(target, "mean"))
    t["ratio"] = t["predicted"] / t["observed"]
    t["abs_log_ratio"] = np.abs(np.log(t["ratio"]))
    return t
