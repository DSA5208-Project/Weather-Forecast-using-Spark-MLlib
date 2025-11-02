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
    col, when, isnan, count, udf, lit, hour, month, dayofyear, to_timestamp,
    mean as _mean, stddev as _stddev, abs as _abs, sum as _sum
)
from pyspark.sql.types import FloatType, IntegerType, StringType
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
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
        self.continuous_features = []
        self.categorical_features = []
        self.all_features = []
        
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
    
    def create_derived_categorical_features(self, df):
        """
        Create derived categorical features from continuous ones.
        
        Args:
            df (DataFrame): DataFrame with parsed columns
            
        Returns:
            DataFrame: DataFrame with additional categorical features
        """
        self.logger.info("Creating derived categorical features...")
        
        # 1. Bin wind direction into 8 compass directions
        # N (337.5-22.5), NE (22.5-67.5), E (67.5-112.5), SE (112.5-157.5),
        # S (157.5-202.5), SW (202.5-247.5), W (247.5-292.5), NW (292.5-337.5)
        df = df.withColumn("WND_DIRECTION_BIN",
            when(col("WND_DIRECTION").isNull(), None)
            .when((col("WND_DIRECTION") >= 337.5) | (col("WND_DIRECTION") < 22.5), 0)  # N
            .when((col("WND_DIRECTION") >= 22.5) & (col("WND_DIRECTION") < 67.5), 1)   # NE
            .when((col("WND_DIRECTION") >= 67.5) & (col("WND_DIRECTION") < 112.5), 2)  # E
            .when((col("WND_DIRECTION") >= 112.5) & (col("WND_DIRECTION") < 157.5), 3) # SE
            .when((col("WND_DIRECTION") >= 157.5) & (col("WND_DIRECTION") < 202.5), 4) # S
            .when((col("WND_DIRECTION") >= 202.5) & (col("WND_DIRECTION") < 247.5), 5) # SW
            .when((col("WND_DIRECTION") >= 247.5) & (col("WND_DIRECTION") < 292.5), 6) # W
            .otherwise(7)  # NW
        )
        
        # 2. Create season from month
        # Winter: 12, 1, 2 -> 0
        # Spring: 3, 4, 5 -> 1
        # Summer: 6, 7, 8 -> 2
        # Fall: 9, 10, 11 -> 3
        df = df.withColumn("SEASON",
            when(col("MONTH").isNull(), None)
            .when(col("MONTH").isin([12, 1, 2]), 0)
            .when(col("MONTH").isin([3, 4, 5]), 1)
            .when(col("MONTH").isin([6, 7, 8]), 2)
            .otherwise(3)
        )
        
        # 3. Create time of day category
        # Night: 0-5 -> 0
        # Morning: 6-11 -> 1
        # Afternoon: 12-17 -> 2
        # Evening: 18-23 -> 3
        df = df.withColumn("TIME_OF_DAY",
            when(col("HOUR").isNull(), None)
            .when(col("HOUR") < 6, 0)
            .when(col("HOUR") < 12, 1)
            .when(col("HOUR") < 18, 2)
            .otherwise(3)
        )
        
        self.logger.info("Derived categorical features created:")
        self.logger.info("  - WND_DIRECTION_BIN: 8 compass directions (0-7)")
        self.logger.info("  - SEASON: 4 seasons (0-3)")
        self.logger.info("  - TIME_OF_DAY: 4 time periods (0-3)")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing and invalid values with comprehensive cleaning.
        Drop features with too many missing values.
        
        Args:
            df (DataFrame): DataFrame with parsed columns
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        self.logger.info("Handling missing values...")
        
        total_count = df.count()
        columns_to_drop = []
        
        # Step 1: Calculate missing percentage for each column and identify columns to drop
        self.logger.info("\nAnalyzing missing values:")
        for column in df.columns:
            # Count null and NaN values
            missing_count = df.filter(col(column).isNull() | isnan(col(column))).count()
            missing_percent = (missing_count / total_count) * 100
            
            if missing_percent > 0:
                self.logger.info(f"  {column}: {missing_count:,} missing ({missing_percent:.2f}%)")
            
            # Mark columns with > MAX_MISSING_PERCENT for removal
            if missing_percent > config.MAX_MISSING_PERCENT * 100:
                columns_to_drop.append(column)
                self.logger.info(f"    -> Will be DROPPED (exceeds {config.MAX_MISSING_PERCENT*100}% threshold)")
        
        # Step 2: Drop columns with too many missing values
        if columns_to_drop:
            self.logger.info(f"\nDropping {len(columns_to_drop)} columns with >{config.MAX_MISSING_PERCENT*100}% missing values:")
            for col_name in columns_to_drop:
                self.logger.info(f"  - {col_name}")
            df = df.drop(*columns_to_drop)
        else:
            self.logger.info(f"\nNo columns exceed the {config.MAX_MISSING_PERCENT*100}% missing value threshold")
        
        # Step 3: Fill remaining missing values with reasonable strategies
        self.logger.info("\nFilling remaining missing values:")
        
        # Get list of continuous and categorical features that still exist
        remaining_continuous = [f for f in config.CONTINUOUS_FEATURES if f in df.columns and f not in columns_to_drop]
        remaining_categorical = [f for f in config.CATEGORICAL_FEATURES if f in df.columns and f not in columns_to_drop]

        self.continuous_features = remaining_continuous
        self.categorical_features = remaining_categorical
        
        # Fill continuous features with mean
        for feature in remaining_continuous:
            # Calculate mean (excluding nulls)
            mean_value = df.select(_mean(col(feature))).first()[0]
            
            if mean_value is not None:
                before_count = df.filter(col(feature).isNull()).count()
                if before_count > 0:
                    df = df.withColumn(feature, 
                        when(col(feature).isNull(), mean_value).otherwise(col(feature))
                    )
                    self.logger.info(f"  {feature}: filled {before_count:,} missing values with mean ({mean_value:.2f})")
        
        # Fill categorical features with mode (most frequent value)
        for feature in remaining_categorical:
            # Get mode (most frequent value)
            mode_row = df.groupBy(feature).count().orderBy(col("count").desc()).first()
            
            if mode_row is not None and mode_row[0] is not None:
                mode_value = mode_row[0]
                before_count = df.filter(col(feature).isNull()).count()
                if before_count > 0:
                    df = df.withColumn(feature,
                        when(col(feature).isNull(), mode_value).otherwise(col(feature))
                    )
                    self.logger.info(f"  {feature}: filled {before_count:,} missing values with mode ({mode_value})")
        
        # Drop any remaining rows with null values in target
        critical_columns = [config.TARGET_COLUMN]  # Target variable must not be null
        before_count = df.count()
        df = df.dropna(subset=critical_columns)
        after_count = df.count()
        
        if before_count - after_count > 0:
            self.logger.info(f"\nDropped {before_count - after_count:,} rows with missing target variable")
        
        self.logger.info(f"\nMissing value handling complete. Final record count: {after_count:,}")
        
        return df
    
    def remove_outliers(self, df):
        """
        Remove statistical outliers using IQR method.
        
        Args:
            df (DataFrame): DataFrame
            
        Returns:
            DataFrame: DataFrame without outliers
        """
        self.logger.info("Removing outliers using IQR method...")
        
        initial_count = df.count()
        
        # Apply IQR outlier detection to continuous features only
        for feature in self.continuous_features:
            # IQR method
            quantiles = df.approxQuantile(feature, [0.25, 0.75], 0.01)
            if len(quantiles) == 2 and quantiles[0] is not None and quantiles[1] is not None:
                q1, q3 = quantiles
                iqr = q3 - q1
                lower_bound = q1 - config.IQR_MULTIPLIER * iqr
                upper_bound = q3 + config.IQR_MULTIPLIER * iqr
                
                before_count = df.count()
                df = df.filter(
                    (col(feature) >= lower_bound) & (col(feature) <= upper_bound)
                )
                after_count = df.count()
                removed = before_count - after_count
                
                if removed > 0:
                    self.logger.info(
                        f"  {feature}: removed {removed:,} outliers "
                        f"(bounds: {lower_bound:.2f} to {upper_bound:.2f})"
                    )
        
        final_count = df.count()
        total_removed = initial_count - final_count
        self.logger.info(
            f"Total outliers removed: {total_removed:,} ({total_removed/initial_count*100:.2f}%)"
        )
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using StringIndexer and OneHotEncoder.
        
        Args:
            df (DataFrame): DataFrame
            
        Returns:
            DataFrame: DataFrame with one-hot encoded categorical features
        """
        self.logger.info("Encoding categorical features with one-hot encoding...")

        # Create indexers and encoders for each categorical feature
        indexers = []
        encoders = []
        encoded_col_names = []
        
        for cat_col in self.categorical_features:
            # Skip if the column is already numeric and doesn't need indexing
            # (e.g., already binned features like WND_DIRECTION_BIN, SEASON, TIME_OF_DAY)
            if cat_col in ['WND_DIRECTION_BIN', 'SEASON', 'TIME_OF_DAY', 'HOUR', 'MONTH']:
                # These are already numeric, just need one-hot encoding
                indexer_output = f"{cat_col}_indexed"
                encoder_output = f"{cat_col}_encoded"
                
                # Cast to double if needed for OneHotEncoder
                df = df.withColumn(indexer_output, col(cat_col).cast("double"))
                
                # Create OneHotEncoder
                encoder = OneHotEncoder(
                    inputCols=[indexer_output],
                    outputCols=[encoder_output],
                    dropLast=True  # Drop last category to avoid multicollinearity
                )
                encoders.append(encoder)
                encoded_col_names.append(encoder_output)
            else:
                # For string categorical features (like STATION), use StringIndexer first
                indexer_output = f"{cat_col}_indexed"
                encoder_output = f"{cat_col}_encoded"
                
                # StringIndexer: converts strings to numeric indices
                indexer = StringIndexer(
                    inputCol=cat_col,
                    outputCol=indexer_output,
                    handleInvalid="keep"  # Keep invalid values as a separate category
                )
                indexers.append(indexer)
                
                # OneHotEncoder: converts indices to binary vectors
                encoder = OneHotEncoder(
                    inputCols=[indexer_output],
                    outputCols=[encoder_output],
                    dropLast=True
                )
                encoders.append(encoder)
                encoded_col_names.append(encoder_output)
        
        # Create and fit pipeline
        if indexers:
            pipeline_stages = indexers + encoders
        else:
            pipeline_stages = encoders
        
        if pipeline_stages:
            pipeline = Pipeline(stages=pipeline_stages)
            pipeline_model = pipeline.fit(df)
            df = pipeline_model.transform(df)

            # Drop original and indexed columns, keep only encoded ones
            cols_to_drop = self.categorical_features + [f"{col}_indexed" for col in self.categorical_features]
            df = df.drop(*cols_to_drop)

            self.logger.info(f"Successfully encoded {len(self.categorical_features)} categorical features:")
            for original, encoded in zip(self.categorical_features, encoded_col_names):
                self.logger.info(f"  {original} -> {encoded}")
        
        # update the encoded column names for later use
        self.categorical_features = encoded_col_names
        
        return df
    
    
    def standardize_features(self, df):
        """
        Standardize CONTINUOUS features only using StandardScaler.
        Categorical features are kept as-is and concatenated afterward.
        
        Args:
            df (DataFrame): DataFrame with features
            
        Returns:
            DataFrame: DataFrame with standardized 'features' column
        """
        if not config.STANDARDIZE_CONTINUOUS:
            self.logger.info("Skipping feature standardization (disabled in config)")
            df = df.withColumnRenamed("features_raw", "features")
            return df
        
        self.logger.info("Standardizing continuous features only...")
        
        # Step 1: Create separate vectors for continuous and categorical features
        continuous_assembler = VectorAssembler(
            inputCols=self.continuous_features,
            outputCol="continuous_features",
            handleInvalid="skip"  # Skip rows with invalid values
        )
        df = continuous_assembler.transform(df)
        
        # Step 2: Standardize only continuous features
        continuous_scaler = StandardScaler(
            inputCol="continuous_features",
            outputCol="continuous_features_scaled",
            withStd=True,
            withMean=True
        )
        self.scaler_model = continuous_scaler.fit(df)
        df = self.scaler_model.transform(df)
        
        # Step 3: Create separate vector for categorical features
        categorical_assembler = VectorAssembler(
            inputCols=self.categorical_features,
            outputCol="categorical_features",
            handleInvalid="skip"
        )
        df = categorical_assembler.transform(df)
        
        # Step 4: Concatenate scaled continuous and unscaled categorical features
        final_assembler = VectorAssembler(
            inputCols=["continuous_features_scaled", "categorical_features"],
            outputCol="features",
            handleInvalid="skip"
        )
        df = final_assembler.transform(df)
        
        self.logger.info(f"Standardized {len(self.continuous_features)} continuous features")
        self.logger.info(f"Kept {len(self.categorical_features)} categorical features unscaled")
        
        # Clean up intermediate columns
        df = df.drop("continuous_features", "continuous_features_scaled", "categorical_features")
        
        self.logger.info("Feature standardization complete (continuous only)")
        
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
        
        # Parse weather columns
        df = self.parse_weather_columns(df)
        
        # Create derived categorical features
        df = self.create_derived_categorical_features(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Remove outliers
        df = self.remove_outliers(df)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Standardize features (continuous features only)
        df = self.standardize_features(df)
        
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
