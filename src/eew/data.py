"""Load and validate the Open University Learning Analytics Dataset (OULAD).

OULAD: Kuzilek, Hlosta & Zdrahal (2017), "Open University Learning Analytics
dataset", Scientific Data 4, 170171. Licence: CC BY 4.0.

All OULAD dates are integer days relative to the start of a module
presentation (day 0). Negative values are days before the start.

The unit of analysis throughout this project is an *enrolment*: one student on
one module presentation, keyed by (code_module, code_presentation, id_student).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eew.config import PROCESSED_DIR, RAW_DIR

ENROLMENT_KEY = ["code_module", "code_presentation", "id_student"]

# Expected columns for each raw OULAD table. Loading fails loudly if any are missing.
SCHEMA: dict[str, list[str]] = {
    "courses": ["code_module", "code_presentation", "module_presentation_length"],
    "assessments": [
        "code_module", "code_presentation", "id_assessment",
        "assessment_type", "date", "weight",
    ],
    "vle": [
        "id_site", "code_module", "code_presentation",
        "activity_type", "week_from", "week_to",
    ],
    "studentInfo": [
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "final_result",
    ],
    "studentRegistration": [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration",
    ],
    "studentAssessment": [
        "id_assessment", "id_student", "date_submitted", "is_banked", "score",
    ],
    "studentVle": [
        "code_module", "code_presentation", "id_student", "id_site", "date", "sum_click",
    ],
}

FINAL_RESULTS = {"Pass", "Distinction", "Fail", "Withdrawn"}


def _find(name: str, raw_dir: Path) -> Path:
    """Locate a raw table as Parquet (preferred) or CSV."""
    for ext in (".parquet", ".csv"):
        path = Path(raw_dir) / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{name}.parquet / {name}.csv not found in {raw_dir}. Download OULAD "
        f"(see README, 'Getting the data') and put the tables in {raw_dir}."
    )


def load_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one raw OULAD table (Parquet or CSV) and check it has the expected columns.

    The official CSVs mark missing values with '?', so that is treated as NA.
    """
    if name not in SCHEMA:
        raise KeyError(f"Unknown OULAD table {name!r}. Expected one of {sorted(SCHEMA)}")
    path = _find(name, raw_dir)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path).replace("?", pd.NA)
    else:
        df = pd.read_csv(path, na_values=["?"])
    missing = set(SCHEMA[name]) - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
    return df


def build_enrolments(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per enrolment with outcome labels and the withdrawal day.

    Columns added:
      - withdrawn (bool): final_result == 'Withdrawn'
      - withdrawal_day (float): date_unregistration for withdrawn students, else NA
      - failed_or_withdrew (bool): the broader 'needs support' outcome
      - module_presentation_length (int): days in the presentation
    """
    info = load_table("studentInfo", raw_dir)
    reg = load_table("studentRegistration", raw_dir)
    courses = load_table("courses", raw_dir)

    unexpected = set(info["final_result"].dropna()) - FINAL_RESULTS
    if unexpected:
        raise ValueError(f"Unexpected final_result values: {sorted(unexpected)}")

    df = (
        info.merge(reg, on=ENROLMENT_KEY, how="left", validate="one_to_one")
        .merge(courses, on=["code_module", "code_presentation"], how="left", validate="many_to_one")
    )
    df["withdrawn"] = df["final_result"].eq("Withdrawn")
    df["failed_or_withdrew"] = df["final_result"].isin(["Fail", "Withdrawn"])
    df["withdrawal_day"] = df["date_unregistration"].where(df["withdrawn"])
    df["id_student"] = df["id_student"].astype("int64")
    return df


def build_daily_clicks(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Total VLE clicks per enrolment per day (days with zero clicks are absent).

    studentVle.csv has ~10.6M rows (one per student, site and day), so it is read
    in chunks and summed down to enrolment-day level.
    """
    path = _find("studentVle", raw_dir)
    cols = ENROLMENT_KEY + ["date", "sum_click"]
    if path.suffix == ".parquet":
        chunks = [pd.read_parquet(path, columns=cols)]
    else:
        chunks = pd.read_csv(path, usecols=cols, chunksize=2_000_000)
    parts = [c.groupby(ENROLMENT_KEY + ["date"], as_index=False)["sum_click"].sum() for c in chunks]
    daily = pd.concat(parts).groupby(ENROLMENT_KEY + ["date"], as_index=False)["sum_click"].sum()
    return daily.rename(columns={"date": "day", "sum_click": "clicks"})


def build_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    """Build the processed tables used by every later step and save them as parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "enrolments": build_enrolments(raw_dir),
        "daily_clicks": build_daily_clicks(raw_dir),
    }
    paths = {}
    for name, df in outputs.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths


if __name__ == "__main__":
    for name, p in build_all().items():
        print(f"wrote {p}")
