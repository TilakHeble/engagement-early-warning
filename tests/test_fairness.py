"""Stage 5 tests: group mappings and the fairness metrics."""

import numpy as np
import pandas as pd

from eew import fairness


def test_add_groups_mappings():
    e = pd.DataFrame({
        "code_module": "AAA", "code_presentation": "2013J", "id_student": [1, 2, 3, 4],
        "disability": ["Y", "N", "N", "N"],
        "imd_band": ["0-10%", "10-20", "90-100%", None],
        "age_band": ["0-35", "35-55", "55<=", "0-35"],
        "gender": ["F", "M", "F", "M"],
        "highest_education": ["No Formal quals", "A Level or Equivalent",
                              "Post Graduate Qualification", "Lower Than A Level"],
    })
    g = fairness.add_groups(e).set_index("id_student")
    assert g.loc[1, "imd_quintile"].startswith("1 most") and g.loc[2, "imd_quintile"].startswith("1 most")
    assert g.loc[3, "imd_quintile"].startswith("5 least") and g.loc[4, "imd_quintile"] == "Unknown"
    assert g.loc[3, "age_group"] == "35 and over" and g.loc[1, "education"] == "Below A level"
    assert g.loc[1, "disability"] == "Declared disability"


def _synthetic(seed=0):
    """Group B's events get low scores, so the model misses them more."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(20):
        for i in range(200):
            g = "B" if i % 4 == 0 else "A"
            y = rng.random() < 0.1
            s = rng.random() + (1.0 if (y and g == "A") else 0.0) + (0.3 if (y and g == "B") else 0.0)
            rows.append({"code_module": f"M{p}", "code_presentation": "2013J", "grp": g,
                         "onset": y, "score": s, "prob": 0.1})
    return pd.DataFrame(rows)


def test_group_metrics_detects_higher_miss_rate():
    df = _synthetic()
    flagged = df.groupby(["code_module", "code_presentation"])["score"].rank(ascending=False, pct=True) <= 0.1
    m = fairness.group_metrics(df, "grp", "prob", flagged, n_boot=200).set_index("group")
    assert m.loc["B", "miss_rate"] > m.loc["A", "miss_rate"] + 0.2
    assert m.loc["A", "miss_rate_lo"] <= m.loc["A", "miss_rate"] <= m.loc["A", "miss_rate_hi"]
    assert abs(m.loc["A", "calibration_ratio"] - 1) < 0.3  # prob 0.1 vs base rate ~0.1
    gaps = fairness.gap_vs_reference(df, "grp", "prob", flagged, reference="A", n_boot=200)
    row = gaps.iloc[0]
    assert row["group"] == "B" and row["miss_gap_lo"] > 0


def test_group_recalibration_fixes_underprediction():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(10):
        for i in range(400):
            g = "B" if i % 5 == 0 else "A"
            true = 0.10 if g == "B" else 0.05
            rows.append({"code_module": f"M{p}", "code_presentation": "2013J", "grp": g,
                         "onset": rng.random() < true, "prob": 0.05})  # B under-predicted
    df = pd.DataFrame(rows)
    df["recal"] = fairness.group_recalibrate(df, "prob", "grp")
    b = df[df.grp == "B"]
    assert abs(b["recal"].mean() / b["onset"].mean() - 1) < 0.25
    assert b["prob"].mean() / b["onset"].mean() < 0.7
