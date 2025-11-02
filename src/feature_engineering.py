"""
Feature Engineering and Selection Module
Uses UnivariateFeatureSelector with proper handling of categorical and continuous features.
"""

import logging
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml.feature import UnivariateFeatureSelector, VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType

import src.config as config


class FeatureEngineer:
    """
    Handles feature selection using UnivariateFeatureSelector.
    Treats categorical and continuous features differently based on their types.
    """
    
    def __init__(self, spark):
        """
        Initialize feature engineer.
        
        Args:
            spark (SparkSession): Active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.continuous_selector_model = None
        self.categorical_selector_model = None
        self.selected_continuous_features = []
        self.selected_categorical_features = []
        self.selected_continuous_indices = []
        self.feature_scores = {}
        self.continuous_feature_names = []
        self.categorical_feature_names = []

    def select_features(self, df, continuous_feature_names, categorical_feature_names):
        """
        Perform univariate feature selection separately for continuous and categorical features.
        
        UnivariateFeatureSelector in Spark MLlib uses:
        - F-test (ANOVA) for continuous features with continuous label (regression)
        - Chi-squared test for categorical features with categorical label (classification)
        - F-test for categorical features with continuous label (regression)
        
        Args:
            df (DataFrame): DataFrame with features and label
            continuous_feature_names (list): List of continuous feature names
            categorical_feature_names (list): List of categorical feature names
        Returns:
            DataFrame: DataFrame with selected features
        """
        self.logger.info("=" * 60)
        self.logger.info("PERFORMING TYPE-AWARE UNIVARIATE FEATURE SELECTION")
        self.logger.info("=" * 60)
        
        self.continuous_feature_names = continuous_feature_names
        self.categorical_feature_names = categorical_feature_names
        
        # Total number of features
        total_features = len(continuous_feature_names) + len(categorical_feature_names)
        self.logger.info(f"Total features: {total_features}")
        self.logger.info(f"  - Continuous: {len(continuous_feature_names)}")
        self.logger.info(f"  - Categorical: {len(categorical_feature_names)}")
        
        # Create separate feature vectors for continuous and categorical features
        df = self._create_separate_feature_vectors(df, continuous_feature_names, categorical_feature_names)
        
        # Select continuous features
        if len(continuous_feature_names) > 0:
            df = self._select_continuous_features(df)
        else:
            self.logger.info("No continuous features to select")
            self.selected_continuous_indices = []
        
        # Combine selected features
        df = self._combine_final_features(df)
        
        total_selected = len(self.selected_continuous_indices)
        self.logger.info("=" * 60)
        self.logger.info(f"FEATURE SELECTION COMPLETE")
        self.logger.info(f"Selected {total_selected} out of {total_features} features")
        self.logger.info(f"  - Continuous: {len(self.selected_continuous_indices)}/{len(continuous_feature_names)}")
        self.logger.info(f"  - Reduction: {(1 - total_selected/total_features)*100:.1f}%")
        self.logger.info("=" * 60)
        
        return df
    
    def _create_separate_feature_vectors(self, df, continuous_names, categorical_names):
        """Create separate feature vectors for continuous and categorical features."""
        self.logger.info("\nCreating separate feature vectors...")
        
        # Create continuous features vector
        if len(continuous_names) > 0:
            cont_assembler = VectorAssembler(
                inputCols=continuous_names,
                outputCol="continuous_features_vector",
                handleInvalid="skip"
            )
            df = cont_assembler.transform(df)

            # Step 2: Standardize only continuous features
            continuous_scaler = StandardScaler(
                inputCol="continuous_features_vector",
                outputCol="continuous_features",
                withStd=True,
                withMean=True
            )
            self.scaler_model = continuous_scaler.fit(df)
            df = self.scaler_model.transform(df)
            self.logger.info(f"  Created continuous_features vector: {len(continuous_names)} features")
        
        # Create categorical features vector
        if len(categorical_names) > 0:
            cat_assembler = VectorAssembler(
                inputCols=categorical_names,
                outputCol="categorical_features",
                handleInvalid="skip"
            )
            df = cat_assembler.transform(df)
            self.logger.info(f"  Created categorical_features vector: {len(categorical_names)} features")
        
        return df
    def _select_continuous_features(self, df):
        """Select continuous features using F-test for regression."""
        self.logger.info("\n" + "-" * 60)
        self.logger.info("SELECTING CONTINUOUS FEATURES")
        self.logger.info("-" * 60)
        
        config_params = config.CONTINUOUS_FEATURE_SELECTION
        
        selector = UnivariateFeatureSelector(
            featuresCol="continuous_features",
            outputCol="selected_continuous_features",
            labelCol="label",
            selectionMode=config_params["selectionMode"],
        )

        if config_params.get("featureType"):
            selector = selector.setFeatureType(config_params["featureType"])
        if config_params.get("labelType"):
            selector = selector.setLabelType(config_params["labelType"])
        if config_params.get("selectionThreshold") is not None:
            selector = selector.setSelectionThreshold(config_params["selectionThreshold"])
        
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Feature type: {config_params['featureType']}")
        self.logger.info(f"  - Label type: {config_params['labelType']}")
        self.logger.info(f"  - Selection mode: {config_params['selectionMode']}")
        self.logger.info(f"  - Threshold: {config_params['selectionThreshold']}")
        
        # Fit the selector
        self.continuous_selector_model = selector.fit(df)
        
        # Get selected feature indices
        self.selected_continuous_indices = self.continuous_selector_model.selectedFeatures
        num_selected = len(self.selected_continuous_indices)
        
        self.logger.info(f"\nSelected {num_selected} continuous features")
        self.logger.info(f"Selected indices: {sorted(self.selected_continuous_indices)}")
        
        # Get feature names
        self.selected_continuous_features = [
            self.continuous_feature_names[i] for i in self.selected_continuous_indices
        ]
        self.logger.info(f"Selected features: {', '.join(self.selected_continuous_features)}")
        
        # Transform data
        df = self.continuous_selector_model.transform(df)
        
        return df
    
    def _combine_final_features(self, df):
        """Combine selected continuous and categorical features into final feature vector."""
        self.logger.info("\nCombining selected features...")
        
        cols_to_assemble = []
        
        if len(self.selected_continuous_indices) > 0:
            cols_to_assemble.append("selected_continuous_features")
        
        cols_to_assemble.append("categorical_features")
        
        if len(cols_to_assemble) == 0:
            self.logger.warning("No features selected! Using original features.")
            return df
        
        # Use VectorAssembler to combine the selected feature vectors
        final_assembler = VectorAssembler(
            inputCols=cols_to_assemble,
            outputCol="features",
            handleInvalid="skip"
        )
        
        df = final_assembler.transform(df)
        
        self.logger.info(f"Combined features into final 'features' vector")
        
        return df
