"""
Configuration file for Weather Forecast project
"""

# Data paths
DATA_PATH = "data/2024.tar.gz"  # Path to the downloaded dataset
OUTPUT_DIR = "output"
MODEL_DIR = "models"

# Data preprocessing parameters
TRAIN_TEST_SPLIT = 0.7  # 70% training, 30% testing
RANDOM_SEED = 42

# Target variable
TARGET_COLUMN = "TMP"

# Feature columns (to be determined after data exploration)
# These are common weather observation columns
FEATURE_COLUMNS = [
    "DEW",  # Dew point temperature
    "SLP",  # Sea level pressure
    "WND",  # Wind speed
    "VIS",  # Visibility
    "AA1",  # Precipitation
]

# Model parameters for hyperparameter tuning
LINEAR_REGRESSION_PARAMS = {
    "maxIter": [100, 200],
    "regParam": [0.0, 0.01, 0.1],
    "elasticNetParam": [0.0, 0.5, 1.0]
}

RANDOM_FOREST_PARAMS = {
    "numTrees": [10, 20, 50],
    "maxDepth": [5, 10, 15],
    "minInstancesPerNode": [1, 5, 10]
}

# Cross-validation parameters
CV_FOLDS = 5

# Spark configuration
SPARK_CONFIG = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "100"
}
