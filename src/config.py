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
# Default local dataset used when running the pipeline in development/test
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "sample.csv")

# Location of the full NOAA dataset after extraction
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "2024.csv")  # After extraction

# Target variable
TARGET_COLUMN = "TMP"

# Feature classification
# Continuous features - numerical values with meaningful ranges
# These columns contain weather observation data in format: "value,quality_code[,additional_fields]"
# They will be parsed to extract the numeric value
CONTINUOUS_FEATURES = [
    # Geographic features (direct numeric values)
    "LATITUDE",           # Geographic coordinate (-90 to 90)
    "LONGITUDE",          # Geographic coordinate (-180 to 180)
    "ELEVATION",          # Height above sea level in meters
    
    # Primary weather observations (comma-separated format - need parsing)
    "TMP",                # Air temperature (scaled by 10, e.g., "+0150,5" = 15.0°C)
    "DEW",                # Dew point temperature (scaled by 10)
    "SLP",                # Sea level pressure (scaled by 10, in hPa)
    "WND",                # Wind observation (direction,quality,type,speed,quality) -> extract direction and speed
    "CIG",                # Ceiling height (in meters)
    "VIS",                # Visibility distance (in meters)
    
    # Precipitation data (AA1, AA2, AA3 fields)
    "AA1",                # Liquid precipitation occurrence (period, depth in mm, condition)
    "AA2",                # Liquid precipitation occurrence (second observation)
    "AA3",                # Liquid precipitation occurrence (third observation)
    
    # Sky cover/cloud data (GD1, GD2, GD3, GD4 fields)
    "GD1",                # Sky cover layer 1 (coverage code, base height in meters)
    "GD2",                # Sky cover layer 2
    "GD3",                # Sky cover layer 3
    "GD4",                # Sky cover layer 4
    
    # Atmospheric pressure (MA1 field)
    "MA1",                # Atmospheric pressure observation (altimeter, station pressure in hPa)
    
    # Pressure change/tendency (MD1 field)
    "MD1",                # Atmospheric pressure change (tendency code, 3hr change in hPa)
    
    # Present weather (MW1, MW2, MW3, MW4, MW5 fields)
    "MW1",                # Present weather observation 1 (condition code)
    "MW2",                # Present weather observation 2
    "MW3",                # Present weather observation 3
    "MW4",                # Present weather observation 4
    "MW5",                # Present weather observation 5
    
    # Past weather (OC1 field)
    "OC1",                # Wind gust observation (speed in m/s)
]

# Categorical features - discrete values or codes
CATEGORICAL_FEATURES = [
    # Station identifiers
    "STATION",            # Station identifier (numeric code)
    "NAME",               # Station name (string)
    "CALL_SIGN",          # Call sign identifier
    
    # Data source and quality indicators  
    "SOURCE",             # Data source code
    "REPORT_TYPE",        # Type of report (FM-15, FM-16, NSRDB, SOD, SOM, etc.)
    "QUALITY_CONTROL",    # Quality control flag
    
    # Time-derived features (will be created during preprocessing)
    "HOUR",               # Hour of day (0-23, cyclic)
    "MONTH",              # Month (1-12, cyclic)
    "DAY_OF_YEAR",        # Day of year (1-365/366)
    "SEASON",             # Season derived from month (0=Winter, 1=Spring, 2=Summer, 3=Fall)
    "TIME_OF_DAY",        # Time of day category (0=Night, 1=Morning, 2=Afternoon, 3=Evening)
    "WND_DIRECTION_BIN",  # Wind direction binned into 8 compass directions (0-7)
]

# All features combined
FEATURE_COLUMNS = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# ==================== DATA PREPROCESSING ====================
# Train/Test split ratio
TRAIN_TEST_SPLIT_RATIO = 0.7
RANDOM_SEED = 42

# Data cleaning settings
MAX_MISSING_PERCENT = 0.5  # Drop features with >50% missing values

# Outlier detection method (IQR)
IQR_MULTIPLIER = 1.5    # For IQR method

# Standardization
STANDARDIZE_CONTINUOUS = True  # Standardize continuous features only

# ==================== FEATURE SELECTION ====================
# UnivariateFeatureSelector parameters for CONTINUOUS features
# This uses F-test (ANOVA F-value) for regression
CONTINUOUS_FEATURE_SELECTION = {
    "featureType": "continuous",
    "labelType": "continuous",  # For regression tasks
    "selectionMode": "fpr",     # False Positive Rate
    "selectionThreshold": 0.05, # Keep features with p-value < 0.05
}

# Feature importance analysis
CALCULATE_FEATURE_IMPORTANCE = True
REMOVE_HIGHLY_CORRELATED = True  # Remove features with correlation > threshold
CORRELATION_THRESHOLD = 0.95     # Threshold for removing correlated features

# Skip feature selection (use all features)
SKIP_FEATURE_SELECTION = False

# ==================== MODEL CONFIGURATION ====================
# Models to train and compare
MODELS_TO_TRAIN = [
    "LinearRegression",
    "RandomForestRegressor",
    # "GBTRegressor",
    # "GeneralizedLinearRegression"
]

# Optionally restrict training to a subset of models.
# When empty, all models listed in ``MODELS_TO_TRAIN`` are used.
SPECIFIC_MODELS = []

# Cross-validation parameters
NUM_FOLDS = 5
PARALLELISM = 4  # Number of parallel folds

# Hyperparameter grids for each model
HYPERPARAMETERS = {
    "LinearRegression": {
        "regParam": [0.01, 0.1, 1.0],
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


print(f"Configuration loaded. Project root: {PROJECT_ROOT}")
print(f"Models directory: {MODELS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
