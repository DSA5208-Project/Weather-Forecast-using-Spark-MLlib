"""
Weather Forecast using Spark MLlib
===================================

A machine learning project for predicting air temperature from weather observation data.
"""

# Import main classes for easy access
from src.data_preprocessing import WeatherDataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator
from src.utils import create_spark_session, setup_logging, format_time
import src.config as config

# Define what's available when using "from src import *"
__all__ = [
    'WeatherDataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'create_spark_session',
    'setup_logging',
    'format_time',
    'config'
]

