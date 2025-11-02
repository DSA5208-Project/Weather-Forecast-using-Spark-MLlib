"""
Utility functions for the Weather Forecast project.
Includes Spark session creation, data parsing, and helper functions.
"""

import logging
import re
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
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


def parse_weather_value(value_str, position, length, scale=1.0, missing_values=None):
    """
    Parse a weather value from a formatted string.
    
    The NOAA ISD format uses fixed-position fields with quality codes.
    Example: "+0150,5" means value=+0150 (scaled), quality=5
    
    Args:
        value_str (str): The string containing the value
        position (int): Starting position in the string
        length (int): Length of the value field
        scale (float): Scaling factor (e.g., 10.0 for temperature in tenths)
        missing_values (list): List of strings indicating missing values
        
    Returns:
        float: Parsed value or None if missing/invalid
    """
    if missing_values is None:
        missing_values = ["9999", "+9999", "99999", "999999"]
    
    try:
        if value_str is None or value_str == "":
            return None
        
        # Split by comma to separate value and quality code
        parts = value_str.split(",")
        if len(parts) == 0:
            return None
        
        value = parts[0].strip()
        
        # Check for missing value indicators
        if value in missing_values or value == "" or value == "null":
            return None
        
        # Convert and scale
        numeric_value = float(value) / scale
        
        return numeric_value
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_wind_data(wnd_str):
    """
    Parse wind data: direction, quality, type code, speed, quality
    Format: "140,5,N,0052,5" = direction(140), quality(5), type(N), speed(52), quality(5)
    
    Args:
        wnd_str (str): Wind data string
        
    Returns:
        tuple: (direction, speed) or (None, None)
    """
    try:
        if wnd_str is None or wnd_str == "":
            return None, None
        
        parts = wnd_str.split(",")
        if len(parts) < 5:
            return None, None
        
        # Direction (0-360 degrees)
        direction = float(parts[0])
        if direction == 999:
            direction = None
        
        # Speed (in meters per second, already scaled)
        speed = float(parts[3])
        if speed == 9999:
            speed = None
        else:
            speed = speed / 10.0  # Scale from tenths
        
        return direction, speed
    
    except (ValueError, AttributeError, IndexError):
        return None, None


def parse_ceiling_data(cig_str):
    """
    Parse ceiling height data.
    Format: "01524,5,M,N" = height(1524), quality(5), determination(M), cavok(N)
    
    Args:
        cig_str (str): Ceiling data string
        
    Returns:
        float: Ceiling height in meters or None
    """
    try:
        if cig_str is None or cig_str == "":
            return None
        
        parts = cig_str.split(",")
        if len(parts) < 1:
            return None
        
        height = float(parts[0])
        if height == 99999:
            return None
        
        return height
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_visibility_data(vis_str):
    """
    Parse visibility distance data.
    Format: "016000,5,N,1" = distance(16000), quality(5), variability(N), quality_variability(1)
    
    Args:
        vis_str (str): Visibility data string
        
    Returns:
        float: Visibility in meters or None
    """
    try:
        if vis_str is None or vis_str == "":
            return None
        
        parts = vis_str.split(",")
        if len(parts) < 1:
            return None
        
        distance = float(parts[0])
        if distance == 999999:
            return None
        
        return distance
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_temperature_data(tmp_str):
    """
    Parse temperature data.
    Format: "+0150,5" = temperature(+15.0°C), quality(5)
    Temperature is in tenths of degrees Celsius
    
    Args:
        tmp_str (str): Temperature data string
        
    Returns:
        float: Temperature in Celsius or None
    """
    return parse_weather_value(tmp_str, 0, 5, scale=10.0, missing_values=["+9999", "9999"])


def parse_dew_point_data(dew_str):
    """
    Parse dew point temperature data.
    Format same as temperature.
    
    Args:
        dew_str (str): Dew point data string
        
    Returns:
        float: Dew point in Celsius or None
    """
    return parse_weather_value(dew_str, 0, 5, scale=10.0, missing_values=["+9999", "9999"])


def parse_sea_level_pressure(slp_str):
    """
    Parse sea level pressure data.
    Format: "10038,5" = pressure(1003.8 hPa), quality(5)
    Pressure is in tenths of hectopascals
    
    Args:
        slp_str (str): Sea level pressure data string
        
    Returns:
        float: Pressure in hPa or None
    """
    return parse_weather_value(slp_str, 0, 5, scale=10.0, missing_values=["99999"])


def extract_datetime_features(date_str):
    """
    Extract hour, month, and day of year from ISO datetime string.
    Format: "2003-01-01T00:53:00"
    
    Args:
        date_str (str): ISO format datetime string
        
    Returns:
        tuple: (hour, month, day_of_year) or (None, None, None)
    """
    try:
        if date_str is None or date_str == "":
            return None, None, None
        
        # Parse ISO format datetime
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        hour = dt.hour
        month = dt.month
        day_of_year = dt.timetuple().tm_yday
        
        return hour, month, day_of_year
    
    except (ValueError, AttributeError):
        return None, None, None


def register_udfs(spark):
    """
    Register UDFs for parsing weather data in Spark SQL.
    
    Args:
        spark (SparkSession): Spark session
    """
    # Register UDFs
    spark.udf.register("parse_temperature", parse_temperature_data, FloatType())
    spark.udf.register("parse_dew_point", parse_dew_point_data, FloatType())
    spark.udf.register("parse_sea_level_pressure", parse_sea_level_pressure, FloatType())
    spark.udf.register("parse_ceiling", parse_ceiling_data, FloatType())
    spark.udf.register("parse_visibility", parse_visibility_data, FloatType())
    
    logging.getLogger(__name__).info("Registered UDFs for weather data parsing")


def calculate_statistics(df, column_name):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (DataFrame): Spark DataFrame
        column_name (str): Column name
        
    Returns:
        dict: Statistics dictionary
    """
    stats = df.select(
        col(column_name)
    ).summary("count", "mean", "stddev", "min", "max").collect()
    
    stats_dict = {row[0]: float(row[1]) if row[1] else None for row in stats}
    return stats_dict


def print_data_summary(df, logger=None):
    """
    Print a summary of the DataFrame.
    
    Args:
        df (DataFrame): Spark DataFrame
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {df.count():,}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Columns: {', '.join(df.columns)}")
    logger.info("=" * 60)


def save_results_to_csv(results_dict, output_path):
    """
    Save model results to CSV file.
    
    Args:
        results_dict (dict): Dictionary of results
        output_path (str): Output file path
    """
    import csv
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in results_dict.items():
            writer.writerow([key, value])
    
    logging.getLogger(__name__).info(f"Results saved to {output_path}")


def format_time(seconds):
    """
    Format seconds into human-readable time.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
