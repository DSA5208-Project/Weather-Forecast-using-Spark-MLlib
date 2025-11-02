"""
Configuration file for the Weather Forecast project.
Contains all parameters for data processing, model training, and evaluation.
"""

import os

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create directories if they don't exist
for directory in [MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
DATASET_URL = "https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2024.tar.gz"
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "2024.csv")  # After extraction

# Target variable
TARGET_COLUMN = "TMP"

# Important weather features (based on NOAA ISD documentation)
# These will be parsed from the raw data
FEATURE_COLUMNS = [
    "LATITUDE",
    "LONGITUDE", 
    "ELEVATION",
    "WND_DIRECTION",      # Wind direction angle
    "WND_SPEED",          # Wind speed rate
    "CIG_HEIGHT",         # Ceiling height dimension
    "VIS_DISTANCE",       # Visibility distance dimension
    "DEW_POINT",          # Dew point temperature
    "SLP_PRESSURE",       # Sea level pressure
    "STATION",            # Station identifier (will be encoded)
    "HOUR",               # Hour of day (derived from DATE)
    "MONTH",              # Month (derived from DATE)
    "DAY_OF_YEAR"         # Day of year (derived from DATE)
]

# Missing/invalid value indicators (from ISD documentation)
MISSING_VALUES = {
    "TMP": ["+9999", "9999"],
    "DEW": ["+9999", "9999"],
    "SLP": ["99999"],
    "WND": ["999", "9999"],
    "CIG": ["99999"],
    "VIS": ["999999"]
}

# Quality control codes to filter out (based on ISD format)
QUALITY_CONTROL_CODES = ["V020"]  # Valid codes to keep

# ==================== DATA PREPROCESSING ====================
# Train/Test split ratio
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_SEED = 42

# Missing value handling
MAX_MISSING_PERCENT = 0.3  # Drop rows with >30% missing values
FILL_STRATEGY = "median"   # Options: "mean", "median", "zero"

# Outlier detection (for temperature)
TEMP_MIN = -90.0  # Minimum valid temperature (°C)
TEMP_MAX = 60.0   # Maximum valid temperature (°C)

# Standardization
STANDARDIZE_FEATURES = True

# ==================== FEATURE SELECTION ====================
# UnivariateFeatureSelector parameters
FEATURE_SELECTION_METHOD = "fpr"  # Options: "numTopFeatures", "percentile", "fpr", "fdr", "fwe"
FEATURE_SELECTION_PARAM = 0.05    # For fpr: false positive rate threshold
NUM_TOP_FEATURES = 10             # If using "numTopFeatures" method

# Use chi2 or f_regression for feature selection
SELECTION_TEST_TYPE = "f_regression"  # Options: "chi2", "f_regression"

# ==================== MODEL CONFIGURATION ====================
# Models to train and compare
MODELS_TO_TRAIN = [
    "LinearRegression",
    "RandomForestRegressor",
    "GBTRegressor",
    "GeneralizedLinearRegression"
]

# Cross-validation parameters
NUM_FOLDS = 5
PARALLELISM = 4  # Number of parallel folds

# Hyperparameter grids for each model
HYPERPARAMETERS = {
    "LinearRegression": {
        "regParam": [0.0, 0.01, 0.1, 1.0],
        "elasticNetParam": [0.0, 0.5, 1.0]
    },
    "RandomForestRegressor": {
        "numTrees": [10, 20, 50],
        "maxDepth": [5, 10, 15],
        "minInstancesPerNode": [1, 5, 10]
    },
    "GBTRegressor": {
        "maxIter": [10, 20, 50],
        "maxDepth": [3, 5, 7],
        "stepSize": [0.1, 0.2, 0.3]
    },
    "GeneralizedLinearRegression": {
        "family": ["gaussian"],
        "link": ["identity"],
        "regParam": [0.0, 0.01, 0.1]
    }
}

# ==================== EVALUATION METRICS ====================
# Metrics to compute and display
EVALUATION_METRICS = [
    "rmse",    # Root Mean Squared Error
    "mae",     # Mean Absolute Error
    "r2",      # R² Score
    "mse"      # Mean Squared Error
]

# ==================== SPARK CONFIGURATION ====================
# Spark application settings
SPARK_APP_NAME = "Weather-Forecast-MLlib"
SPARK_MASTER = "local[*]"  # Use all available cores

# Spark configuration
SPARK_CONFIG = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "8",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

# ==================== OUTPUT SETTINGS ====================
# Model save paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models")

# Visualization settings
PLOT_DPI = 300
PLOT_FORMAT = "png"
FIGURE_SIZE = (12, 8)

# Result files
RESULTS_CSV = os.path.join(OUTPUT_DIR, "model_results.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")

# Report settings
REPORT_PATH = os.path.join(OUTPUT_DIR, "model_report.txt")

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")

# ==================== SAMPLE DATA ====================
# For testing with sample.csv
USE_SAMPLE_DATA = True  # Set to False when using full dataset
SAMPLE_SIZE_FRACTION = 0.1  # Use 10% of data for testing


# Model training configuration
SPECIFIC_MODELS = None  # List of specific models to train (None = train all)
# Example: ["LinearRegression", "RandomForestRegressor"]

print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
print(f"Models directory: {MODELS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
