#!/usr/bin/env python3
"""
Main Execution Script for Weather Forecast Project
===================================================

This script orchestrates the complete machine learning pipeline:
1. Data loading and preprocessing
2. Feature engineering and selection
3. Model training with cross-validation
4. Model evaluation and comparison
5. Visualization and reporting

Usage:
    python main.py

Configuration:
    Edit src/config.py to customize:
    - Data paths and settings
    - Model parameters
    - Feature selection options
    - Output preferences

Author: Weather Forecast Team
Date: 2024
"""

import time
import sys
import logging

# Import project modules
from src import (
    WeatherDataPreprocessor,
    FeatureEngineer,
    ModelTrainer,
    ModelEvaluator,
    create_spark_session,
    setup_logging,
    format_time,
    config
)


def main():
    """
    Main execution function.
    """
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("WEATHER FORECAST USING APACHE SPARK MLLIB")
    logger.info("=" * 80)
    logger.info("")
    
    # Record start time
    start_time = time.time()
    
    try:
        # ============================================================
        # STEP 1: Initialize Spark Session
        # ============================================================
        logger.info("STEP 1: Initializing Spark Session")
        logger.info("-" * 80)
        spark = create_spark_session()
        logger.info("Spark session initialized successfully\n")
        
        # ============================================================
        # STEP 2: Load and Preprocess Data
        # ============================================================
        logger.info("STEP 2: Loading and Preprocessing Data")
        logger.info("-" * 80)
        
        preprocessor = WeatherDataPreprocessor(spark)
        
        # Load data using configuration
        data_path = config.DEFAULT_DATA_PATH
        raw_df = preprocessor.load_data(data_path)
        
        # Preprocess data
        processed_df = preprocessor.prepare_dataset(raw_df)
        
        # Split data
        train_df, test_df = preprocessor.split_data(processed_df)
        
        logger.info("Data preprocessing completed\n")
        
        # ============================================================
        # STEP 3: Feature Engineering and Selection
        # ============================================================
        if not config.SKIP_FEATURE_SELECTION:
            logger.info("STEP 3: Feature Engineering and Selection")
            logger.info("-" * 80)
            
            feature_engineer = FeatureEngineer(spark)
            
            # Calculate feature importance
            feature_scores = feature_engineer.get_feature_importance_scores(
                train_df,
                preprocessor.continuous_features,
                preprocessor.categorical_features
            )
            
            # Perform univariate feature selection (treats categorical and continuous separately)
            train_df = feature_engineer.select_features_by_type(
                train_df,
                preprocessor.continuous_features,
                preprocessor.categorical_features
            )
            test_df = feature_engineer.select_features_by_type(
                test_df,
                preprocessor.continuous_features,
                preprocessor.categorical_features
            )
            
            # Save feature importance
            feature_engineer.save_feature_importance()
            
            logger.info("Feature engineering completed\n")
        else:
            logger.info("STEP 3: Skipping feature selection (config.SKIP_FEATURE_SELECTION = True)\n")
        
        # ============================================================
        # STEP 4: Train Models
        # ============================================================
        logger.info("STEP 4: Training Models")
        logger.info("-" * 80)
        
        trainer = ModelTrainer(spark)
        
        # Use specific models if configured
        if config.SPECIFIC_MODELS:
            config.MODELS_TO_TRAIN = config.SPECIFIC_MODELS
            logger.info(f"Training custom models: {', '.join(config.SPECIFIC_MODELS)}")
        else:
            logger.info(f"Training all configured models: {', '.join(config.MODELS_TO_TRAIN)}")
        
        # Train all models
        models, training_results = trainer.train_all_models(train_df)
        
        # Save models
        trainer.save_all_models()
        
        # Print training summary
        trainer.print_training_summary()
        training_summary = trainer.get_training_summary()
        
        logger.info("Model training completed\n")
        
        # ============================================================
        # STEP 5: Evaluate Models
        # ============================================================
        logger.info("STEP 5: Evaluating Models")
        logger.info("-" * 80)
        
        evaluator = ModelEvaluator(spark)
        
        # Evaluate all models on test set
        evaluation_results, all_predictions = evaluator.evaluate_all_models(
            models, test_df
        )
        
        # Compare models
        best_model_name, comparison_df = evaluator.compare_models()
        
        # Create visualizations
        evaluator.create_all_visualizations(all_predictions, comparison_df)
        
        # Save results
        evaluator.save_results(comparison_df)
        
        # Generate report
        evaluator.generate_report(comparison_df, training_summary)
        
        logger.info("Model evaluation completed\n")
        
        # ============================================================
        # STEP 6: Final Summary
        # ============================================================
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total execution time: {format_time(total_time)}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model RMSE: {comparison_df.iloc[0]['rmse']:.4f}")
        logger.info(f"Best model R²: {comparison_df.iloc[0]['r2']:.4f}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - Models: {config.MODELS_DIR}")
        logger.info(f"  - Results: {config.RESULTS_CSV}")
        logger.info(f"  - Report: {config.REPORT_PATH}")
        logger.info(f"  - Visualizations: {config.OUTPUT_DIR}")
        logger.info(f"  - Feature importance: {config.FEATURE_IMPORTANCE_CSV}")
        logger.info("=" * 80)
        
        # Stop Spark
        spark.stop()
        logger.info("Spark session stopped")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to stop Spark if initialized
        try:
            spark.stop()
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
