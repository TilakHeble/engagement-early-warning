"""Stage 6 tests: the transfer splits never train on the data they predict."""

import numpy as np
import pandas as pd
import pytest

from eew import transfer


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    rows = []
    for m in ["AAA", "BBB", "CCC"]:
        for pres in ["2013J", "2014J"]:
            for i in range(150):
                for w in [2, 4]:
                    x = rng.normal()
                    rows.append({"code_module": m, "code_presentation": pres, "id_student": i, "week": w,
                                 "x": x, "onset": rng.random() < 1 / (1 + np.exp(-(-2.5 + 1.5 * x)))})
    return pd.DataFrame(rows)


def test_leave_one_module_out_excludes_module(monkeypatch, data):
    seen = []
    real = transfer.fit_predict

    def spy(train, test, features, **kw):
        assert not set(train["code_module"]) & set(test["code_module"])
        seen.append(test["code_module"].iloc[0])
        return real(train, test, features, **kw)

    monkeypatch.setattr(transfer, "fit_predict", spy)
    pred = transfer.leave_one_module_out(data, ["x"])
    assert sorted(seen) == ["AAA", "BBB", "CCC"] and pred.notna().all()


def test_forward_in_time_uses_only_2013_and_drops_own_history(monkeypatch, data):
    calls = []
    real = transfer.fit_predict

    def spy(train, test, features, **kw):
        assert train["code_presentation"].str.startswith("2013").all()
        assert test["code_presentation"].str.startswith("2014").all()
        calls.append((set(train["code_module"]), set(test["code_module"])))
        return real(train, test, features, **kw)

    monkeypatch.setattr(transfer, "fit_predict", spy)
    with_h, without_h = transfer.forward_in_time(data, ["x"])
    assert len(with_h) == len(without_h) == (data["code_presentation"] == "2014J").sum()
    # Every "without history" model excludes the module it predicts.
    for train_mods, test_mods in calls[1:]:
        assert not train_mods & test_mods


def test_summarise_and_paired_difference(data):
    data = data.assign(good=data["x"] + 3 * data["onset"], noise=np.random.default_rng(1).random(len(data)))
    s = transfer.summarise(data, "good", 0.1)
    assert s["recall"] > 0.2 and s["auc"] > 0.9
    d = transfer.paired_recall_difference(data, "good", "noise", 0.1, n_boot=300)
    assert d["lo"] > 0


def test_local_recalibration_fixes_level_keeps_ranking(data):
    # Make module BBB's true risk much higher than other modules at the same x.
    d = data.copy()
    rng = np.random.default_rng(5)
    bbb = d["code_module"] == "BBB"
    d.loc[bbb, "onset"] = rng.random(bbb.sum()) < 1 / (1 + np.exp(-(-0.5 + 1.5 * d.loc[bbb, "x"])))
    out = transfer.local_recalibration(d, ["x"], shrink_k=None)
    b = out[out["code_module"] == "BBB"]
    before = abs(np.log(b["p_cold"].mean() / b["onset"].mean()))
    after = abs(np.log(b["p_local_recal"].mean() / b["onset"].mean()))
    assert after < before
    # Ranking within the module is unchanged.
    assert (b["p_cold"].rank() == b["p_local_recal"].rank()).all()


def test_shrinkage_pulls_factor_towards_one(data):
    raw = transfer.local_recalibration(data, ["x"], shrink_k=None).groupby("code_module")["recal_factor"].first()
    shrunk = transfer.local_recalibration(data, ["x"], shrink_k=100).groupby("code_module")["recal_factor"].first()
    assert (np.abs(np.log(shrunk)) <= np.abs(np.log(raw)) + 1e-12).all()
