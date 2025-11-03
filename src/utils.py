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


# ==================== ADDITIONAL PARSING FUNCTIONS ====================

def parse_precipitation_data(precip_str):
    """
    Parse precipitation data from AA1, AA2, AA3 fields.
    Format: "01,0000,9,5" = period(hours), depth(mm), condition, quality
    
    Args:
        precip_str (str): Precipitation data string
        
    Returns:
        tuple: (period_hours, depth_mm, condition_code) or (None, None, None)
    """
    try:
        if precip_str is None or precip_str == "":
            return None, None, None
        
        parts = precip_str.split(",")
        if len(parts) < 3:
            return None, None, None
        
        # Period in hours
        period = float(parts[0])
        if period == 99:
            period = None
        
        # Depth in millimeters (already scaled)
        depth = float(parts[1])
        if depth == 9999:
            depth = None
        else:
            depth = depth / 10.0  # Scale from tenths
        
        # Condition code
        condition = int(parts[2]) if parts[2].isdigit() else None
        
        return period, depth, condition
    
    except (ValueError, AttributeError, IndexError):
        return None, None, None


def parse_sky_cover_layer(sky_str):
    """
    Parse sky cover layer data from GD1, GD2, GD3, GD4 fields.
    Format: "3,99,5,+01524,5,9" = coverage, opacity, quality, base_height, quality, cloud_type
    
    Args:
        sky_str (str): Sky cover data string
        
    Returns:
        tuple: (coverage_code, base_height_m) or (None, None)
    """
    try:
        if sky_str is None or sky_str == "":
            return None, None
        
        parts = sky_str.split(",")
        if len(parts) < 4:
            return None, None
        
        # Coverage code (0-8 indicating octas of sky coverage)
        coverage = int(parts[0]) if parts[0].isdigit() else None
        if coverage == 99:
            coverage = None
        
        # Base height in meters
        height = float(parts[3].replace("+", ""))
        if height == 99999 or height == -9999:
            height = None
        
        return coverage, height
    
    except (ValueError, AttributeError, IndexError):
        return None, None


def parse_atmospheric_pressure(press_str):
    """
    Parse atmospheric pressure observation from MA1 field.
    Format: "10041,5,99999,9" = altimeter(hPa), quality, station_pressure, quality
    
    Args:
        press_str (str): Pressure data string
        
    Returns:
        float: Atmospheric pressure in hPa or None
    """
    try:
        if press_str is None or press_str == "":
            return None
        
        parts = press_str.split(",")
        if len(parts) < 1:
            return None
        
        pressure = float(parts[0])
        if pressure == 99999:
            return None
        
        return pressure / 10.0  # Scale from tenths
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_pressure_change(change_str):
    """
    Parse atmospheric pressure change from MD1 field.
    Format: "8,1,001,1,+999,9" = tendency, quality, 3hr_change, quality, 24hr_change, quality
    
    Args:
        change_str (str): Pressure change data string
        
    Returns:
        tuple: (tendency_code, change_3hr_hPa) or (None, None)
    """
    try:
        if change_str is None or change_str == "":
            return None, None
        
        parts = change_str.split(",")
        if len(parts) < 3:
            return None, None
        
        # Tendency code (0-8)
        tendency = int(parts[0]) if parts[0].isdigit() else None
        if tendency == 9:
            tendency = None
        
        # 3-hour pressure change
        change_3hr = int(parts[2])
        if change_3hr == 999:
            change_3hr = None
        else:
            change_3hr = change_3hr / 10.0  # Scale from tenths
        
        return tendency, change_3hr
    
    except (ValueError, AttributeError, IndexError):
        return None, None


def parse_present_weather(weather_str):
    """
    Parse present weather observation from MW fields.
    Format: "10,1" = condition_code, quality
    Or: "0060,0000,02,000,0000,02,000,0000,02,000" = complex format
    
    Args:
        weather_str (str): Weather observation string
        
    Returns:
        int: Weather condition code or None
    """
    try:
        if weather_str is None or weather_str == "":
            return None
        
        parts = weather_str.split(",")
        if len(parts) < 1:
            return None
        
        # Get the first numeric code
        condition = int(parts[0])
        if condition == 99 or condition == 9999:
            return None
        
        return condition
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_sunshine_observation(sun_str):
    """
    Parse sunshine observation from GJ1, GK1 fields.
    Format: "08,99,1,99,9,99,9,99999,9,99,9,99,9" = duration and various quality codes
    
    Args:
        sun_str (str): Sunshine data string
        
    Returns:
        float: Sunshine duration in minutes or None
    """
    try:
        if sun_str is None or sun_str == "":
            return None
        
        parts = sun_str.split(",")
        if len(parts) < 1:
            return None
        
        # Duration in minutes (first field)
        duration = int(parts[0])
        if duration == 99 or duration == 9999:
            return None
        
        return float(duration)
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_wind_gust(gust_str):
    """
    Parse wind gust observation from OC1 field.
    Format: "45,5" = gust_speed, quality
    Or: "0093,1" = gust_speed (in different scale), quality
    
    Args:
        gust_str (str): Wind gust data string
        
    Returns:
        float: Wind gust speed in m/s or None
    """
    try:
        if gust_str is None or gust_str == "":
            return None
        
        parts = gust_str.split(",")
        if len(parts) < 1:
            return None
        
        gust = float(parts[0])
        if gust == 9999 or gust == 999:
            return None
        
        # Scale appropriately (some formats use tenths)
        if gust > 200:  # Likely in tenths
            gust = gust / 10.0
        
        return gust
    
    except (ValueError, AttributeError, IndexError):
        return None


def get_wind_type_code(wnd_str):
    """
    Extract wind type code from WND field.
    Format: "140,5,N,0052,5" = direction, quality, type_code, speed, quality
    
    Args:
        wnd_str (str): Wind data string
        
    Returns:
        str: Wind type code (N, C, V, etc.) or None
    """
    try:
        if wnd_str is None or wnd_str == "":
            return None
        
        parts = wnd_str.split(",")
        if len(parts) < 3:
            return None
        
        type_code = parts[2].strip()
        if type_code == "" or type_code == "9":
            return None
        
        return type_code
    
    except (ValueError, AttributeError, IndexError):
        return None


def parse_report_type(report_str):
    """
    Parse report type from REPORT_TYPE field.
    Common types: FM-15, FM-16, SOD, SOM, NSRDB
    
    Args:
        report_str (str): Report type string
        
    Returns:
        str: Simplified report type category
    """
    try:
        if report_str is None or report_str == "":
            return "UNKNOWN"
        
        report = report_str.strip().upper()
        
        # Categorize report types
        if "FM" in report:
            return "METAR"  # Meteorological Aerodrome Report
        elif "SO" in report:
            return "SUMMARY"  # Summary observation
        elif "NSRDB" in report:
            return "SOLAR"  # Solar/radiation data
        else:
            return "OTHER"
    
    except (ValueError, AttributeError):
        return "UNKNOWN"


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
