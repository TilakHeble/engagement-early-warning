"""Project paths and shared settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Prediction points: the model at week k may only use data from days < 7 * k.
PREDICTION_WEEKS = [2, 4, 6, 8, 10, 12]

# Tutor capacity used in the headline evaluation: share of a cohort that can be
# contacted in one week.
CONTACT_CAPACITY = 0.10

RANDOM_SEED = 42
