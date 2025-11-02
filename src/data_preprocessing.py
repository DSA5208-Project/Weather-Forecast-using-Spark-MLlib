"""
Data Preprocessing Module for Weather Forecast Project
Handles loading, cleaning, and preprocessing of NOAA weather data.
"""

import logging
import os
import requests
import tarfile
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, isnan, count, udf, lit, hour, month, dayofyear, to_timestamp
)
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

import src.config as config
from src.utils import (
    parse_temperature_data, parse_dew_point_data, parse_sea_level_pressure,
    parse_wind_data, parse_ceiling_data, parse_visibility_data,
    extract_datetime_features
)


class WeatherDataPreprocessor:
    """
    Preprocessor for NOAA Global Hourly Weather Data.
    Handles data loading, parsing, cleaning, and feature engineering.
    """
    
    def __init__(self, spark):
        """
        Initialize the preprocessor.
        
        Args:
            spark (SparkSession): Active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.scaler_model = None
        self.feature_columns = []
        
    def load_data(self):
        """
        Load weather data from CSV file. Downloads data from DATASET_URL if not exists.
        
        Args:
            file_path (str): Path to CSV file. If None, uses config.
            
        Returns:
            DataFrame: Loaded Spark DataFrame
        """
        # Check if data exists, if not, download it
        if not os.path.exists(config.RAW_DATA_PATH):
            self.logger.info(f"Data file not found at {config.RAW_DATA_PATH}")
            self._download_and_extract_data()
        
        self.logger.info(f"Loading data from: {config.RAW_DATA_PATH}")
        
        try:
            df = self.spark.read.csv(
                config.RAW_DATA_PATH,
                header=True,
                inferSchema=True,
                escape='"',
                multiLine=True
            )
            
            initial_count = df.count()
            self.logger.info(f"Loaded {initial_count:,} records")
            self.logger.info(f"Columns: {', '.join(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _download_and_extract_data(self):
        """
        Download and extract weather data from DATASET_URL.
        Downloads the tar.gz file and extracts the CSV.
        """
        self.logger.info(f"Downloading data from {config.DATASET_URL}...")
        
        try:
            # Download the tar.gz file
            tar_path = config.RAW_DATA_PATH.replace('.csv', '.tar.gz')
            
            response = requests.get(config.DATASET_URL, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0
            
            self.logger.info(f"Downloading to {tar_path}...")
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (block_size * 1000) == 0:  # Log every ~8MB
                                self.logger.info(f"Download progress: {progress:.1f}%")
            
            self.logger.info(f"Download complete. Extracting archive...")
            
            # Extract the tar.gz file
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find the CSV file in the archive
                csv_members = [m for m in tar.getmembers() if m.name.endswith('.csv')]
                
                if not csv_members:
                    raise ValueError("No CSV file found in the archive")
                
                # Extract the first CSV file
                csv_member = csv_members[0]
                self.logger.info(f"Extracting {csv_member.name}...")
                
                tar.extract(csv_member, path=os.path.dirname(config.RAW_DATA_PATH))
                
                # Rename extracted file to expected name if different
                extracted_path = os.path.join(os.path.dirname(config.RAW_DATA_PATH), csv_member.name)
                if extracted_path != config.RAW_DATA_PATH:
                    os.rename(extracted_path, config.RAW_DATA_PATH)
            
            # Clean up the tar.gz file
            os.remove(tar_path)
            
            self.logger.info(f"Data successfully downloaded and extracted to {config.RAW_DATA_PATH}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading data: {e}")
            raise
        except tarfile.TarError as e:
            self.logger.error(f"Error extracting archive: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during download/extraction: {e}")
            raise
    
    def parse_weather_columns(self, df):
        """
        Parse complex weather data columns into numeric features.
        
        Args:
            df (DataFrame): Raw data DataFrame
            
        Returns:
            DataFrame: DataFrame with parsed columns
        """
        self.logger.info("Parsing weather data columns...")
        
        # Register UDFs for parsing
        parse_temp_udf = udf(parse_temperature_data, FloatType())
        parse_dew_udf = udf(parse_dew_point_data, FloatType())
        parse_slp_udf = udf(parse_sea_level_pressure, FloatType())
        parse_ceiling_udf = udf(parse_ceiling_data, FloatType())
        parse_vis_udf = udf(parse_visibility_data, FloatType())
        
        # UDF for parsing wind data (returns tuple, need to extract separately)
        def parse_wind_direction(wnd_str):
            direction, _ = parse_wind_data(wnd_str)
            return direction
        
        def parse_wind_speed(wnd_str):
            _, speed = parse_wind_data(wnd_str)
            return speed
        
        parse_wnd_dir_udf = udf(parse_wind_direction, FloatType())
        parse_wnd_speed_udf = udf(parse_wind_speed, FloatType())
        
        # Parse columns
        df = df.withColumn("TMP_VALUE", parse_temp_udf(col("TMP")))
        df = df.withColumn("DEW_POINT", parse_dew_udf(col("DEW")))
        df = df.withColumn("SLP_PRESSURE", parse_slp_udf(col("SLP")))
        df = df.withColumn("WND_DIRECTION", parse_wnd_dir_udf(col("WND")))
        df = df.withColumn("WND_SPEED", parse_wnd_speed_udf(col("WND")))
        df = df.withColumn("CIG_HEIGHT", parse_ceiling_udf(col("CIG")))
        df = df.withColumn("VIS_DISTANCE", parse_vis_udf(col("VIS")))
        
        # Extract datetime features
        df = df.withColumn("HOUR", hour(to_timestamp(col("DATE"))))
        df = df.withColumn("MONTH", month(to_timestamp(col("DATE"))))
        df = df.withColumn("DAY_OF_YEAR", dayofyear(to_timestamp(col("DATE"))))
        
        self.logger.info("Weather columns parsed successfully")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing and invalid values.
        
        Args:
            df (DataFrame): DataFrame with parsed columns
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        self.logger.info("Handling missing values...")
        
        initial_count = df.count()
        
        # Drop rows where target variable is missing
        df = df.filter(col("TMP_VALUE").isNotNull())
        
        # Apply temperature range filter
        df = df.filter(
            (col("TMP_VALUE") >= config.TEMP_MIN) &
            (col("TMP_VALUE") <= config.TEMP_MAX)
        )
        
        after_target_count = df.count()
        self.logger.info(f"Removed {initial_count - after_target_count:,} rows with invalid target values")
        
        # Calculate missing percentage for each row
        feature_cols = [
            "LATITUDE", "LONGITUDE", "ELEVATION",
            "WND_DIRECTION", "WND_SPEED", "CIG_HEIGHT",
            "VIS_DISTANCE", "DEW_POINT", "SLP_PRESSURE",
            "HOUR", "MONTH", "DAY_OF_YEAR"
        ]
        
        # Drop rows with too many missing features
        for col_name in feature_cols:
            if col_name in df.columns:
                # Count nulls before
                null_count_before = df.filter(col(col_name).isNull()).count()
                
                # For continuous features, fill with median or mean
                if config.FILL_STRATEGY == "median":
                    # Get median (approximate using percentile)
                    median_val = df.approxQuantile(col_name, [0.5], 0.01)
                    if median_val and median_val[0] is not None:
                        df = df.fillna({col_name: median_val[0]})
                elif config.FILL_STRATEGY == "mean":
                    mean_val = df.select(col(col_name)).agg({col_name: "mean"}).collect()[0][0]
                    if mean_val is not None:
                        df = df.fillna({col_name: mean_val})
                else:
                    df = df.fillna({col_name: 0.0})
                
                null_count_after = df.filter(col(col_name).isNull()).count()
                self.logger.info(
                    f"Column {col_name}: filled {null_count_before - null_count_after:,} missing values"
                )
        
        final_count = df.count()
        self.logger.info(f"Final dataset size: {final_count:,} rows")
        
        return df
    
    def remove_outliers(self, df):
        """
        Remove statistical outliers using IQR method for key features.
        
        Args:
            df (DataFrame): DataFrame
            
        Returns:
            DataFrame: DataFrame without outliers
        """
        self.logger.info("Removing outliers...")
        
        initial_count = df.count()
        
        # For each numeric feature, calculate IQR and remove outliers
        numeric_features = ["DEW_POINT", "SLP_PRESSURE", "WND_SPEED", "VIS_DISTANCE"]
        
        for feature in numeric_features:
            if feature in df.columns:
                # Calculate Q1, Q3, and IQR
                quantiles = df.approxQuantile(feature, [0.25, 0.75], 0.01)
                if len(quantiles) == 2 and quantiles[0] is not None and quantiles[1] is not None:
                    q1, q3 = quantiles
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter outliers
                    df = df.filter(
                        (col(feature) >= lower_bound) & (col(feature) <= upper_bound)
                    )
        
        final_count = df.count()
        removed = initial_count - final_count
        self.logger.info(f"Removed {removed:,} outlier rows ({removed/initial_count*100:.2f}%)")
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features (e.g., STATION).
        
        Args:
            df (DataFrame): DataFrame
            
        Returns:
            DataFrame: DataFrame with encoded categorical features
        """
        self.logger.info("Encoding categorical features...")
        
        # Encode STATION as numeric index
        if "STATION" in df.columns:
            indexer = StringIndexer(inputCol="STATION", outputCol="STATION_INDEX")
            df = indexer.fit(df).transform(df)
            self.logger.info("Encoded STATION column")
        
        return df
    
    def create_feature_vector(self, df):
        """
        Assemble features into a single vector column.
        
        Args:
            df (DataFrame): DataFrame with individual feature columns
            
        Returns:
            DataFrame: DataFrame with 'features' vector column
        """
        self.logger.info("Creating feature vector...")
        
        # Define feature columns
        self.feature_columns = [
            "LATITUDE", "LONGITUDE", "ELEVATION",
            "WND_DIRECTION", "WND_SPEED",
            "CIG_HEIGHT", "VIS_DISTANCE",
            "DEW_POINT", "SLP_PRESSURE",
            "STATION_INDEX",
            "HOUR", "MONTH", "DAY_OF_YEAR"
        ]
        
        # Filter only existing columns
        available_features = [col_name for col_name in self.feature_columns if col_name in df.columns]
        
        self.logger.info(f"Using {len(available_features)} features: {', '.join(available_features)}")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=available_features,
            outputCol="features_raw"
        )
        
        df = assembler.transform(df)
        
        return df, available_features
    
    def standardize_features(self, df):
        """
        Standardize features using StandardScaler.
        
        Args:
            df (DataFrame): DataFrame with 'features_raw' column
            
        Returns:
            DataFrame: DataFrame with standardized 'features' column
        """
        if not config.STANDARDIZE_FEATURES:
            self.logger.info("Skipping feature standardization (disabled in config)")
            df = df.withColumnRenamed("features_raw", "features")
            return df
        
        self.logger.info("Standardizing features...")
        
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        self.scaler_model = scaler.fit(df)
        df = self.scaler_model.transform(df)
        
        self.logger.info("Features standardized successfully")
        
        return df
    
    def prepare_dataset(self, df):
        """
        Complete preprocessing pipeline: parse, clean, encode, and assemble features.
        
        Args:
            df (DataFrame): Raw data DataFrame
            
        Returns:
            DataFrame: Preprocessed DataFrame ready for modeling
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING DATA PREPROCESSING")
        self.logger.info("=" * 60)
        
        # Step 1: Parse weather columns
        df = self.parse_weather_columns(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove outliers
        df = self.remove_outliers(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Create feature vector
        df, feature_names = self.create_feature_vector(df)
        
        # Step 6: Standardize features
        df = self.standardize_features(df)
        
        # Step 7: Select final columns
        df = df.select(
            "features",
            col("TMP_VALUE").alias("label"),
            *feature_names
        )
        
        # Cache the result for efficiency
        df = df.cache()
        
        final_count = df.count()
        self.logger.info("=" * 60)
        self.logger.info(f"PREPROCESSING COMPLETE: {final_count:,} records")
        self.logger.info("=" * 60)
        
        return df
    
    def split_data(self, df, train_ratio=None):
        """
        Split data into training and testing sets.
        
        Args:
            df (DataFrame): Preprocessed DataFrame
            train_ratio (float): Training set ratio (default from config)
            
        Returns:
            tuple: (train_df, test_df)
        """
        if train_ratio is None:
            train_ratio = config.TRAIN_TEST_SPLIT_RATIO
        
        self.logger.info(f"Splitting data: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test")
        
        train_df, test_df = df.randomSplit(
            [train_ratio, 1 - train_ratio],
            seed=config.RANDOM_SEED
        )
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        self.logger.info(f"Training set: {train_count:,} records")
        self.logger.info(f"Test set: {test_count:,} records")
        
        return train_df, test_df
