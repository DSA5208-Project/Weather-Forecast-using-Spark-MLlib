"""
Utility functions for the Weather Forecast project.
Includes Spark session creation, data parsing, and helper functions.
"""

import logging
import re
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lit, split, regexp_replace
from pyspark.sql.types import FloatType, IntegerType, StringType
import src.config as config


def setup_logging():
    """
    Set up logging configuration for the project.
    """
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_spark_session(app_name=None):
    """
    Create and configure a Spark session.
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        SparkSession: Configured Spark session
    """
    logger = logging.getLogger(__name__)
    
    if app_name is None:
        app_name = config.SPARK_APP_NAME
    
    logger.info(f"Creating Spark session: {app_name}")
    
    # Build Spark session
    builder = SparkSession.builder.appName(app_name).master(config.SPARK_MASTER)
    
    # Apply configuration
    for key, value in config.SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Spark master: {spark.sparkContext.master}")
    
    return spark


def parse_weather_field(df, column_name, value_index=0, scale_factor=None, 
                       missing_values=['9999', '99999', '999999', '+9999']):
    """
    Generic parser for weather observation fields.
    Most weather fields follow the format: "value,quality_code[,additional_fields]"
    
    Args:
        df (DataFrame): Input DataFrame
        column_name (str): Name of the column to parse
        value_index (int): Index of the value in comma-separated string (default: 0)
        scale_factor (float): Divide value by this (e.g., 10 for temperature). If None, no scaling.
        missing_values (list): List of string patterns indicating missing data
    
    Returns:
        DataFrame: DataFrame with new parsed column named "{column_name}_VALUE"
    """
    output_col = f"{column_name}_VALUE"
    
    # Check if column exists in DataFrame
    if column_name not in df.columns:
        return df
    
    # Split by comma and get the value at specified index
    parts = split(col(column_name), ",")
    value = parts.getItem(value_index)
    quality_code = parts.getItem(1)  # Quality code is typically at index 1
    
    # Remove any quotes, plus signs, and whitespace
    value = regexp_replace(value, '"', '')
    value = regexp_replace(value, '\\+', '')
    value = regexp_replace(value, ' ', '')
    
    # Handle missing values - check if value matches any missing patterns or quality code is 9
    is_missing = lit(False)
    for missing_pattern in missing_values:
        is_missing = is_missing | (value == missing_pattern) | value.contains(missing_pattern)
    
    # Add quality code check - if quality code is 9, it's missing
    is_missing = is_missing | (quality_code == "9")
    
    # Parse to float and apply scaling
    parsed = when(is_missing | value.isNull() | (value == ""), None) \
             .otherwise(value.cast("float"))
    
    if scale_factor:
        parsed = parsed / scale_factor
    
    df = df.withColumn(output_col, parsed)
    
    return df