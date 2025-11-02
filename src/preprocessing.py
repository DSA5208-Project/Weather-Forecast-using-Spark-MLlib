"""Data loading and cleaning utilities for the NOAA weather dataset."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config
from .data_downloader import ensure_extracted


CSV_OPTIONS = {
    "header": "true",
    "multiLine": "false",
    "mode": "PERMISSIVE",
    "quote": '"',
    "escape": '"',
    "recursiveFileLookup": "true",
}


def _invalid_condition(column: F.Column, invalid_values: Sequence[str]) -> F.Column:
    cond = column.isNull() | (column == "")
    for value in invalid_values:
        cond = cond | (column == F.lit(value))
    return cond


def _parse_measure(column: str, value_index: int, invalid_values: Sequence[str], scale: float = 1.0) -> F.Column:
    parts = F.split(F.col(column), ",")
    raw_value = parts.getItem(value_index)
    numeric = F.trim(F.regexp_replace(raw_value, "^[+]", ""))
    cond = _invalid_condition(numeric, invalid_values)
    return F.when(cond, None).otherwise(numeric.cast("double") / scale)


def load_weather_dataframe(
    spark: SparkSession,
    data_dir: Optional[str] = None,
    row_limit: Optional[int] = config.DEFAULT_ROW_LIMIT,
) -> DataFrame:
    """Load the NOAA dataset into a Spark DataFrame."""
    if data_dir:
        extract_dir = Path(data_dir)
        if not extract_dir.exists():
            raise FileNotFoundError(f"Provided data directory does not exist: {extract_dir}")
    else:
        extract_dir = ensure_extracted()
    df = spark.read.options(**CSV_OPTIONS).csv(str(extract_dir))
    if row_limit:
        df = df.limit(int(row_limit))
    return df


def prepare_observations(df: DataFrame) -> DataFrame:
    """Clean and enrich the raw dataframe ready for feature engineering."""
    df = df.withColumn("timestamp", F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss"))
    df = df.filter(df.timestamp.isNotNull())

    df = df.withColumn("air_temperature_c", _parse_measure("TMP", 0, ["9999", "99999", "999999"], scale=10.0))
    df = df.withColumn("temp_quality", F.split(F.col("TMP"), ",").getItem(1))
    df = df.withColumn("dew_point_c", _parse_measure("DEW", 0, ["9999", "99999", "999999"], scale=10.0))
    df = df.withColumn("dew_quality", F.split(F.col("DEW"), ",").getItem(1))
    df = df.withColumn("sea_level_pressure_hpa", _parse_measure("SLP", 0, ["9999", "99999", "999999"], scale=10.0))
    df = df.withColumn("slp_quality", F.split(F.col("SLP"), ",").getItem(1))

    df = df.withColumn("wind_direction_deg", _parse_measure("WND", 0, ["999", "9999", "99999"], scale=1.0))
    df = df.withColumn("wind_speed_ms", _parse_measure("WND", 3, ["9999", "99999"], scale=10.0))
    df = df.withColumn("wind_speed_quality", F.split(F.col("WND"), ",").getItem(4))

    df = df.withColumn("visibility_m", _parse_measure("VIS", 0, ["9999", "99999", "999999"], scale=10.0))
    df = df.withColumn("visibility_quality", F.split(F.col("VIS"), ",").getItem(1))

    df = df.withColumn("call_sign_clean", F.when(F.col("CALL_SIGN").isin("99999", "999999"), None).otherwise(F.col("CALL_SIGN")))
    df = df.withColumn("report_type_clean", F.coalesce(F.col("REPORT_TYPE"), F.lit("UNKNOWN")))
    df = df.withColumn("quality_control_clean", F.coalesce(F.col("QUALITY_CONTROL"), F.lit("UNKNOWN")))

    df = df.withColumn("station_lat", F.col("LATITUDE").cast("double"))
    df = df.withColumn("station_lon", F.col("LONGITUDE").cast("double"))
    df = df.withColumn("station_elevation", F.col("ELEVATION").cast("double"))

    valid_quality = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    df = df.filter(F.col("temp_quality").isin(*valid_quality))
    df = df.filter(F.col("dew_quality").isin(*valid_quality))
    df = df.filter(F.col("slp_quality").isin(*valid_quality))
    df = df.filter(F.col("wind_speed_quality").isin(*valid_quality))
    df = df.filter(F.col("visibility_quality").isin(*valid_quality))

    df = df.filter(
        df.air_temperature_c.isNotNull()
        & df.dew_point_c.isNotNull()
        & df.sea_level_pressure_hpa.isNotNull()
        & df.wind_speed_ms.isNotNull()
        & df.wind_direction_deg.isNotNull()
        & df.visibility_m.isNotNull()
        & df.station_lat.isNotNull()
        & df.station_lon.isNotNull()
        & df.station_elevation.isNotNull()
    )

    df = df.withColumn("wind_direction_rad", F.radians(F.col("wind_direction_deg")))
    df = df.withColumn("wind_u_component", F.col("wind_speed_ms") * F.cos(F.col("wind_direction_rad")))
    df = df.withColumn("wind_v_component", F.col("wind_speed_ms") * F.sin(F.col("wind_direction_rad")))

    df = df.withColumn("hour_of_day", F.hour(F.col("timestamp")))
    df = df.withColumn("day_of_year", F.dayofyear(F.col("timestamp")))

    df = df.dropna(subset=["wind_u_component", "wind_v_component"])

    df = df.withColumnRenamed("air_temperature_c", "label")
    return df


CONTINUOUS_FEATURE_COLUMNS: List[str] = [
    "dew_point_c",
    "sea_level_pressure_hpa",
    "wind_speed_ms",
    "wind_u_component",
    "wind_v_component",
    "visibility_m",
    "station_lat",
    "station_lon",
    "station_elevation",
    "hour_of_day",
    "day_of_year",
]

CATEGORICAL_FEATURE_COLUMNS: List[str] = [
    "report_type_clean",
    "quality_control_clean",
    "call_sign_clean",
    "SOURCE",
]
