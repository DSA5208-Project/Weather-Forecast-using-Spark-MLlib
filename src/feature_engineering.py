"""
Feature Engineering and Selection Module
Uses UnivariateFeatureSelector for automated feature selection.
"""

import logging
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml.feature import UnivariateFeatureSelector, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col

import src.config as config


class FeatureEngineer:
    """
    Handles feature selection using UnivariateFeatureSelector.
    """
    
    def __init__(self, spark):
        """
        Initialize feature engineer.
        
        Args:
            spark (SparkSession): Active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.selector_model = None
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features_univariate(self, df, feature_col="features", label_col="label"):
        """
        Perform univariate feature selection using statistical tests.
        
        UnivariateFeatureSelector supports:
        - featureType="continuous" for regression (F-test)
        - featureType="categorical" for classification (chi-squared)
        
        Selection modes:
        - numTopFeatures: Select top K features
        - percentile: Select top X% of features
        - fpr: False Positive Rate test
        - fdr: False Discovery Rate test
        - fwe: Family-Wise Error rate test
        
        Args:
            df (DataFrame): DataFrame with features and label
            feature_col (str): Name of features column
            label_col (str): Name of label column
            
        Returns:
            DataFrame: DataFrame with selected features
        """
        self.logger.info("=" * 60)
        self.logger.info("PERFORMING UNIVARIATE FEATURE SELECTION")
        self.logger.info("=" * 60)
        
        # Get original feature count
        original_feature_count = df.select(feature_col).first()[0].size
        self.logger.info(f"Original number of features: {original_feature_count}")
        
        # Configure UnivariateFeatureSelector
        selector = UnivariateFeatureSelector(
            featuresCol=feature_col,
            outputCol="selectedFeatures",
            labelCol=label_col,
            featureType="continuous",  # For regression tasks
            selectionMode=config.FEATURE_SELECTION_METHOD
        )
        
        # Set selection parameter based on mode
        if config.FEATURE_SELECTION_METHOD == "numTopFeatures":
            num_features = min(config.NUM_TOP_FEATURES, original_feature_count)
            selector.setSelectionThreshold(num_features)
            self.logger.info(f"Selection mode: numTopFeatures (K={num_features})")
            
        elif config.FEATURE_SELECTION_METHOD == "percentile":
            percentile = config.FEATURE_SELECTION_PARAM
            selector.setSelectionThreshold(percentile)
            self.logger.info(f"Selection mode: percentile ({percentile*100}%)")
            
        elif config.FEATURE_SELECTION_METHOD in ["fpr", "fdr", "fwe"]:
            threshold = config.FEATURE_SELECTION_PARAM
            selector.setSelectionThreshold(threshold)
            self.logger.info(f"Selection mode: {config.FEATURE_SELECTION_METHOD} (threshold={threshold})")
        
        # Fit the selector
        self.logger.info("Fitting UnivariateFeatureSelector...")
        self.selector_model = selector.fit(df)
        
        # Get selected feature indices
        selected_indices = self.selector_model.selectedFeatures
        num_selected = len(selected_indices)
        
        self.logger.info(f"Selected {num_selected} out of {original_feature_count} features")
        self.logger.info(f"Feature reduction: {(1 - num_selected/original_feature_count)*100:.1f}%")
        self.logger.info(f"Selected feature indices: {sorted(selected_indices)}")
        
        # Transform data with selected features
        df_selected = self.selector_model.transform(df)
        
        # Rename column for consistency
        df_selected = df_selected.withColumnRenamed("selectedFeatures", "features")
        
        self.logger.info("Feature selection completed successfully")
        self.logger.info("=" * 60)
        
        return df_selected
    
    def get_feature_importance_scores(self, df, feature_names, feature_col="features", label_col="label"):
        """
        Calculate feature importance scores using correlation analysis.
        
        Args:
            df (DataFrame): DataFrame with features and label
            feature_names (list): List of feature names
            feature_col (str): Name of features column
            label_col (str): Name of label column
            
        Returns:
            dict: Feature importance scores
        """
        self.logger.info("Calculating feature importance scores...")
        
        # Calculate correlation between each feature and target
        feature_scores = {}
        
        for idx, feature_name in enumerate(feature_names):
            if feature_name in df.columns:
                # Calculate Pearson correlation
                corr_result = df.stat.corr(feature_name, label_col)
                feature_scores[feature_name] = abs(corr_result) if corr_result is not None else 0.0
        
        # Sort by importance
        feature_scores = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info("Top 10 features by correlation:")
        for i, (feature, score) in enumerate(list(feature_scores.items())[:10], 1):
            self.logger.info(f"  {i}. {feature}: {score:.4f}")
        
        self.feature_scores = feature_scores
        return feature_scores
    
    def save_feature_importance(self, output_path=None):
        """
        Save feature importance scores to CSV file.
        
        Args:
            output_path (str): Output file path
        """
        if output_path is None:
            output_path = config.FEATURE_IMPORTANCE_CSV
        
        if not self.feature_scores:
            self.logger.warning("No feature scores available to save")
            return
        
        # Convert to pandas and save
        df_importance = pd.DataFrame([
            {"Feature": feature, "Importance": score}
            for feature, score in self.feature_scores.items()
        ])
        
        df_importance.to_csv(output_path, index=False)
        self.logger.info(f"Feature importance scores saved to: {output_path}")
    
    def create_interaction_features(self, df, feature_pairs=None):
        """
        Create interaction features (product of feature pairs).
        This can capture non-linear relationships.
        
        Args:
            df (DataFrame): Input DataFrame
            feature_pairs (list): List of tuples specifying feature pairs
            
        Returns:
            DataFrame: DataFrame with additional interaction features
        """
        if feature_pairs is None:
            # Default interactions for weather data
            feature_pairs = [
                ("WND_SPEED", "WND_DIRECTION"),  # Wind characteristics
                ("DEW_POINT", "SLP_PRESSURE"),   # Atmospheric conditions
                ("HOUR", "MONTH"),                # Time-based patterns
            ]
        
        self.logger.info(f"Creating {len(feature_pairs)} interaction features...")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_X_{feat2}"
                df = df.withColumn(interaction_name, col(feat1) * col(feat2))
                self.logger.info(f"Created interaction feature: {interaction_name}")
        
        return df
    
    def create_polynomial_features(self, df, degree=2, features=None):
        """
        Create polynomial features for selected columns.
        
        Args:
            df (DataFrame): Input DataFrame
            degree (int): Polynomial degree (2 for quadratic, 3 for cubic)
            features (list): List of feature names to create polynomials for
            
        Returns:
            DataFrame: DataFrame with polynomial features
        """
        if features is None:
            # Default polynomial features for key weather variables
            features = ["WND_SPEED", "DEW_POINT", "SLP_PRESSURE"]
        
        self.logger.info(f"Creating degree-{degree} polynomial features...")
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    poly_name = f"{feature}_POW{d}"
                    df = df.withColumn(poly_name, col(feature) ** d)
                    self.logger.info(f"Created polynomial feature: {poly_name}")
        
        return df
    
    def get_correlation_matrix(self, df, feature_col="features"):
        """
        Calculate correlation matrix for features.
        Useful for identifying multicollinearity.
        
        Args:
            df (DataFrame): DataFrame with features
            feature_col (str): Name of features vector column
            
        Returns:
            DataFrame: Correlation matrix
        """
        self.logger.info("Calculating feature correlation matrix...")
        
        # Calculate Pearson correlation
        corr_matrix = Correlation.corr(df, feature_col, "pearson").head()[0]
        
        self.logger.info(f"Correlation matrix shape: {corr_matrix.numRows} x {corr_matrix.numCols}")
        
        return corr_matrix
    
    def remove_highly_correlated_features(self, df, feature_names, threshold=0.95):
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            df (DataFrame): DataFrame with individual feature columns
            feature_names (list): List of feature names
            threshold (float): Correlation threshold (default: 0.95)
            
        Returns:
            tuple: (DataFrame, list of remaining features)
        """
        self.logger.info(f"Removing features with correlation > {threshold}...")
        
        features_to_remove = set()
        
        # Calculate pairwise correlations
        for i, feat1 in enumerate(feature_names):
            if feat1 in features_to_remove or feat1 not in df.columns:
                continue
            
            for feat2 in feature_names[i+1:]:
                if feat2 in features_to_remove or feat2 not in df.columns:
                    continue
                
                # Calculate correlation
                corr = abs(df.stat.corr(feat1, feat2))
                
                if corr > threshold:
                    # Remove feature with lower correlation to target
                    corr1 = abs(df.stat.corr(feat1, "label"))
                    corr2 = abs(df.stat.corr(feat2, "label"))
                    
                    if corr1 < corr2:
                        features_to_remove.add(feat1)
                        self.logger.info(f"Removing {feat1} (corr with {feat2}: {corr:.3f})")
                    else:
                        features_to_remove.add(feat2)
                        self.logger.info(f"Removing {feat2} (corr with {feat1}: {corr:.3f})")
        
        # Keep only uncorrelated features
        remaining_features = [f for f in feature_names if f not in features_to_remove]
        
        self.logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        self.logger.info(f"Remaining features: {len(remaining_features)}")
        
        return df, remaining_features
