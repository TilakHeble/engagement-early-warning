"""Tests for OULAD loading, using a tiny synthetic copy of the real schema."""

import pandas as pd
import pytest

from eew import data


@pytest.fixture
def raw_dir(tmp_path):
    key = {"code_module": "AAA", "code_presentation": "2013J"}
    tables = {
        "courses": pd.DataFrame([{**key, "module_presentation_length": 268}]),
        "studentInfo": pd.DataFrame([
            {**key, "id_student": 1, "gender": "F", "region": "Scotland",
             "highest_education": "A Level", "imd_band": "20-30%", "age_band": "0-35",
             "num_of_prev_attempts": 0, "studied_credits": 60, "disability": "N",
             "final_result": "Pass"},
            {**key, "id_student": 2, "gender": "M", "region": "Wales",
             "highest_education": "HE Qualification", "imd_band": "?", "age_band": "35-55",
             "num_of_prev_attempts": 1, "studied_credits": 120, "disability": "Y",
             "final_result": "Withdrawn"},
        ]),
        "studentRegistration": pd.DataFrame([
            {**key, "id_student": 1, "date_registration": -30, "date_unregistration": None},
            {**key, "id_student": 2, "date_registration": -10, "date_unregistration": 45},
        ]),
        "studentVle": pd.DataFrame([
            {**key, "id_student": 1, "id_site": 10, "date": 0, "sum_click": 3},
            {**key, "id_student": 1, "id_site": 11, "date": 0, "sum_click": 2},
            {**key, "id_student": 1, "id_site": 10, "date": 1, "sum_click": 4},
            {**key, "id_student": 2, "id_site": 10, "date": 0, "sum_click": 1},
        ]),
    }
    for name, df in tables.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def test_enrolments_labels(raw_dir):
    df = data.build_enrolments(raw_dir).set_index("id_student")
    assert len(df) == 2
    assert not df.loc[1, "withdrawn"] and df.loc[2, "withdrawn"]
    assert df.loc[2, "withdrawal_day"] == 45
    assert pd.isna(df.loc[1, "withdrawal_day"])
    assert pd.isna(df.loc[2, "imd_band"])  # '?' read as missing
    assert df.loc[1, "module_presentation_length"] == 268


def test_daily_clicks_sums_sites(raw_dir):
    daily = data.build_daily_clicks(raw_dir)
    s1 = daily[daily.id_student == 1].set_index("day")["clicks"]
    assert s1.to_dict() == {0: 5, 1: 4}


def test_reads_parquet(raw_dir):
    info = pd.read_csv(raw_dir / "studentInfo.csv")
    info.to_parquet(raw_dir / "studentInfo.parquet", index=False)
    (raw_dir / "studentInfo.csv").unlink()
    df = data.build_enrolments(raw_dir).set_index("id_student")
    assert pd.isna(df.loc[2, "imd_band"])  # '?' still read as missing
    assert df.loc[2, "withdrawn"]


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download OULAD"):
        data.load_table("studentInfo", tmp_path)


def test_missing_column(raw_dir):
    df = pd.read_csv(raw_dir / "courses.csv").drop(columns="module_presentation_length")
    df.to_csv(raw_dir / "courses.csv", index=False)
    with pytest.raises(ValueError, match="module_presentation_length"):
        data.load_table("courses", raw_dir)
