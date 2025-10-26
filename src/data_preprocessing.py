"""
Data preprocessing module for weather forecast project
Handles data loading, cleaning, and feature engineering
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, regexp_extract, trim
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
import config


def create_spark_session(app_name="WeatherForecast"):
    """
    Create and configure Spark session
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession object
    """
    builder = SparkSession.builder.appName(app_name)
    
    # Apply Spark configurations
    for key, value in config.SPARK_CONFIG.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def load_data(spark, file_path):
    """
    Load weather data from CSV file
    
    Args:
        spark: SparkSession object
        file_path: Path to the weather data file
        
    Returns:
        DataFrame with loaded data
    """
    print(f"Loading data from {file_path}...")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Initial data shape: {df.count()} rows, {len(df.columns)} columns")
    return df


def parse_weather_value(column_name):
    """
    Parse weather observation values from the format 'value,quality_code'
    Example: '+0023,1' -> 23
    
    Args:
        column_name: Name of the column to parse
        
    Returns:
        Column expression with parsed numeric value
    """
    # Extract the numeric value before the comma
    return regexp_extract(col(column_name), r'^([+-]?\d+)', 1).cast(DoubleType())


def clean_data(df):
    """
    Clean and preprocess weather data
    - Remove invalid values (9999, +9999, etc. indicate missing data)
    - Parse weather observation format
    - Handle missing values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\nCleaning data...")
    
    # Parse temperature column (TMP format: +0023,1)
    if "TMP" in df.columns:
        df = df.withColumn("TMP_raw", col("TMP"))
        df = df.withColumn("TMP", parse_weather_value("TMP") / 10.0)  # Convert to actual temperature
        # Remove invalid temperature values
        df = df.filter((col("TMP") != 999.9) & (col("TMP").isNotNull()))
    
    # Parse dew point (DEW format: +0018,1)
    if "DEW" in df.columns:
        df = df.withColumn("DEW", parse_weather_value("DEW") / 10.0)
        df = df.withColumn("DEW", when(col("DEW") == 999.9, None).otherwise(col("DEW")))
    
    # Parse sea level pressure (SLP format: 10134,1)
    if "SLP" in df.columns:
        df = df.withColumn("SLP", parse_weather_value("SLP") / 10.0)
        df = df.withColumn("SLP", when(col("SLP") == 9999.9, None).otherwise(col("SLP")))
    
    # Parse wind speed (WND format: 070,1,N,0046,1)
    if "WND" in df.columns:
        df = df.withColumn("WND_speed", regexp_extract(col("WND"), r',N,(\d+),', 1).cast(DoubleType()) / 10.0)
        df = df.withColumn("WND_speed", when(col("WND_speed") == 999.9, None).otherwise(col("WND_speed")))
        df = df.withColumn("WND_direction", regexp_extract(col("WND"), r'^(\d+),', 1).cast(DoubleType()))
        df = df.withColumn("WND_direction", when(col("WND_direction") == 999, None).otherwise(col("WND_direction")))
    
    # Parse visibility (VIS format: 016000,1,9,9)
    if "VIS" in df.columns:
        df = df.withColumn("VIS", regexp_extract(col("VIS"), r'^(\d+),', 1).cast(DoubleType()))
        df = df.withColumn("VIS", when(col("VIS") == 999999, None).otherwise(col("VIS")))
    
    # Parse precipitation (AA1 format: 01,0000,9,1)
    if "AA1" in df.columns:
        df = df.withColumn("AA1_depth", regexp_extract(col("AA1"), r',(\d+),', 1).cast(DoubleType()) / 10.0)
        df = df.withColumn("AA1_depth", when(col("AA1_depth") == 999.9, None).otherwise(col("AA1_depth")))
    
    print(f"Data after cleaning: {df.count()} rows")
    
    return df


def select_features(df):
    """
    Select relevant features for model training
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with selected features
    """
    print("\nSelecting features...")
    
    # Define available features based on what exists in the dataframe
    available_features = []
    
    feature_mapping = {
        "DEW": "DEW",
        "SLP": "SLP",
        "WND_speed": "WND",
        "WND_direction": "WND",
        "VIS": "VIS",
        "AA1_depth": "AA1"
    }
    
    for feature, original in feature_mapping.items():
        if feature in df.columns:
            available_features.append(feature)
    
    # Always include the target
    selected_columns = ["TMP"] + available_features
    
    # Filter to only include rows with no missing values in selected columns
    df_selected = df.select(selected_columns)
    df_selected = df_selected.dropna()
    
    print(f"Selected features: {available_features}")
    print(f"Data after feature selection: {df_selected.count()} rows")
    
    return df_selected, available_features


def prepare_features(df, feature_columns):
    """
    Prepare features for machine learning
    - Assemble feature vectors
    - Standardize features
    
    Args:
        df: DataFrame with selected features
        feature_columns: List of feature column names
        
    Returns:
        DataFrame with assembled and scaled features
    """
    print("\nPreparing features for ML...")
    
    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features_raw"
    )
    df = assembler.transform(df)
    
    # Standardize features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Select only necessary columns
    df = df.select("TMP", "features")
    
    print(f"Features prepared. Final dataset: {df.count()} rows")
    
    return df, scaler_model


def split_data(df, train_ratio=0.7, seed=42):
    """
    Split data into training and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion of data for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"\nSplitting data: {train_ratio*100}% train, {(1-train_ratio)*100}% test")
    
    train_df, test_df = df.randomSplit([train_ratio, 1-train_ratio], seed=seed)
    
    print(f"Training set: {train_df.count()} rows")
    print(f"Test set: {test_df.count()} rows")
    
    return train_df, test_df


def preprocess_pipeline(spark, data_path):
    """
    Complete preprocessing pipeline
    
    Args:
        spark: SparkSession object
        data_path: Path to raw data
        
    Returns:
        Tuple of (train_df, test_df, feature_columns, scaler_model)
    """
    # Load data
    df = load_data(spark, data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Select features
    df, feature_columns = select_features(df)
    
    # Prepare features
    df, scaler_model = prepare_features(df, feature_columns)
    
    # Split data
    train_df, test_df = split_data(
        df,
        train_ratio=config.TRAIN_TEST_SPLIT,
        seed=config.RANDOM_SEED
    )
    
    return train_df, test_df, feature_columns, scaler_model


if __name__ == "__main__":
    # Test preprocessing
    spark = create_spark_session()
    
    # You need to specify the correct path to your data
    data_path = "data/2024/*.csv"  # Adjust based on your data location
    
    try:
        train_df, test_df, feature_columns, scaler = preprocess_pipeline(spark, data_path)
        print("\n" + "="*50)
        print("Preprocessing completed successfully!")
        print("="*50)
        
        # Show sample data
        print("\nSample training data:")
        train_df.show(5)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    finally:
        spark.stop()
