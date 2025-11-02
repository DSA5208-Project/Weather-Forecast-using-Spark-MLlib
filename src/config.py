"""Configuration values for the weather forecasting pipeline."""
from pathlib import Path

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory locations for cached data
DATA_DIR = ROOT_DIR / "data"
RAW_ARCHIVE_PATH = DATA_DIR / "2024.tar.gz"
EXTRACTED_DIR = DATA_DIR / "global-hourly-2024"

# NOAA Global Hourly dataset URL
DATASET_URL = "https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2024.tar.gz"

# Default number of rows to sample when running locally (None loads entire dataset)
DEFAULT_ROW_LIMIT = None

# Thresholds for feature selection
CONTINUOUS_FEATURES_TO_KEEP = 8
CATEGORICAL_FEATURES_TO_KEEP = 32
