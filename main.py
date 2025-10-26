"""
Main script for Weather Forecast using Spark MLlib
Orchestrates the entire pipeline: data preprocessing, model training, and evaluation
"""

import argparse
import os
import sys
from datetime import datetime
import json

# Import custom modules
from src.data_preprocessing import create_spark_session, preprocess_pipeline
from src.train_model import (
    train_linear_regression,
    train_random_forest,
    train_gradient_boosted_trees,
    save_model,
    get_cv_metrics
)
from src.evaluate_model import (
    evaluate_model,
    plot_predictions,
    plot_cv_results,
    compare_models,
    save_metrics_report
)
from src import config


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [config.OUTPUT_DIR, config.MODEL_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def log_experiment(metrics_list, training_times, feature_columns, output_dir):
    """
    Save experiment configuration and results
    
    Args:
        metrics_list: List of model metrics
        training_times: Dictionary of training times
        feature_columns: List of feature names
        output_dir: Directory to save log
    """
    experiment_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "train_test_split": config.TRAIN_TEST_SPLIT,
            "random_seed": config.RANDOM_SEED,
            "cv_folds": config.CV_FOLDS,
            "features": feature_columns
        },
        "models": {},
        "training_times": training_times
    }
    
    # Add model metrics
    for metrics in metrics_list:
        model_name = metrics['model_name']
        experiment_log["models"][model_name] = {
            "rmse": metrics['rmse'],
            "mae": metrics['mae'],
            "r2": metrics['r2'],
            "mse": metrics['mse']
        }
    
    # Find best model
    best_model = min(metrics_list, key=lambda x: x['rmse'])
    experiment_log["best_model"] = best_model['model_name']
    experiment_log["best_rmse"] = best_model['rmse']
    
    # Save to JSON
    log_path = os.path.join(output_dir, "experiment_log.json")
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=4)
    
    print(f"\nExperiment log saved to: {log_path}")


def main(args):
    """
    Main execution function
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*70)
    print("WEATHER FORECAST USING SPARK MLLIB")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Setup
    setup_directories()
    
    # Create Spark session
    print("Initializing Spark session...")
    spark = create_spark_session("WeatherForecast_Project")
    
    try:
        # ===== Data Preprocessing =====
        print("\n" + "#"*70)
        print("# STEP 1: DATA PREPROCESSING")
        print("#"*70)
        
        train_df, test_df, feature_columns, scaler = preprocess_pipeline(
            spark,
            args.data_path
        )
        
        # Cache data for faster access
        train_df.cache()
        test_df.cache()
        
        print(f"\nFeatures used: {feature_columns}")
        
        # ===== Model Training =====
        print("\n" + "#"*70)
        print("# STEP 2: MODEL TRAINING")
        print("#"*70)
        
        models = {}
        cv_models = {}
        training_times = {}
        
        # Train Linear Regression
        if not args.skip_lr:
            lr_model, lr_cv, lr_time = train_linear_regression(
                train_df,
                cv_folds=config.CV_FOLDS
            )
            models['Linear_Regression'] = lr_model
            cv_models['Linear_Regression'] = lr_cv
            training_times['Linear_Regression'] = lr_time
            
            # Save model
            save_model(lr_model, "linear_regression_model", config.MODEL_DIR)
        
        # Train Random Forest
        if not args.skip_rf:
            rf_model, rf_cv, rf_time = train_random_forest(
                train_df,
                cv_folds=config.CV_FOLDS
            )
            models['Random_Forest'] = rf_model
            cv_models['Random_Forest'] = rf_cv
            training_times['Random_Forest'] = rf_time
            
            # Save model
            save_model(rf_model, "random_forest_model", config.MODEL_DIR)
        
        # Train Gradient Boosted Trees (optional)
        if args.include_gbt:
            gbt_model, gbt_cv, gbt_time = train_gradient_boosted_trees(
                train_df,
                cv_folds=config.CV_FOLDS
            )
            models['Gradient_Boosted_Trees'] = gbt_model
            cv_models['Gradient_Boosted_Trees'] = gbt_cv
            training_times['Gradient_Boosted_Trees'] = gbt_time
            
            # Save model
            save_model(gbt_model, "gbt_model", config.MODEL_DIR)
        
        # ===== Model Evaluation =====
        print("\n" + "#"*70)
        print("# STEP 3: MODEL EVALUATION")
        print("#"*70)
        
        metrics_list = []
        
        for model_name, model in models.items():
            # Evaluate model
            metrics, predictions = evaluate_model(model, test_df, model_name)
            metrics_list.append(metrics)
            
            # Generate plots
            plot_predictions(predictions, model_name, config.OUTPUT_DIR)
            
            # Plot CV results
            cv_metrics = get_cv_metrics(cv_models[model_name])
            plot_cv_results(cv_metrics, [], model_name, config.OUTPUT_DIR)
        
        # ===== Model Comparison =====
        print("\n" + "#"*70)
        print("# STEP 4: MODEL COMPARISON")
        print("#"*70)
        
        compare_models(metrics_list, config.OUTPUT_DIR)
        save_metrics_report(metrics_list, config.OUTPUT_DIR)
        
        # ===== Summary =====
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        for model_name in training_times:
            print(f"\n{model_name}:")
            print(f"  Training time: {training_times[model_name]:.2f} seconds")
            
            # Find corresponding metrics
            model_metrics = next(m for m in metrics_list if m['model_name'] == model_name)
            print(f"  RMSE: {model_metrics['rmse']:.4f}")
            print(f"  R²:   {model_metrics['r2']:.4f}")
        
        # Best model
        best_model = min(metrics_list, key=lambda x: x['rmse'])
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model['model_name']}")
        print(f"RMSE: {best_model['rmse']:.4f}")
        print(f"MAE:  {best_model['mae']:.4f}")
        print(f"R²:   {best_model['r2']:.4f}")
        print("="*70)
        
        # Save experiment log
        log_experiment(metrics_list, training_times, feature_columns, config.OUTPUT_DIR)
        
        print(f"\nAll results saved to: {config.OUTPUT_DIR}")
        print(f"Models saved to: {config.MODEL_DIR}")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print("ERROR OCCURRED")
        print('!'*70)
        print(f"\n{str(e)}\n")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    finally:
        # Cleanup
        train_df.unpersist()
        test_df.unpersist()
        spark.stop()
        print("\nSpark session stopped.")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weather Forecast using Spark MLlib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all default models
  python main.py --data-path data/2024/*.csv
  
  # Skip Linear Regression
  python main.py --data-path data/2024/*.csv --skip-lr
  
  # Include Gradient Boosted Trees
  python main.py --data-path data/2024/*.csv --include-gbt
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/2024/*.csv',
        help='Path to weather data files (default: data/2024/*.csv)'
    )
    
    parser.add_argument(
        '--skip-lr',
        action='store_true',
        help='Skip Linear Regression model training'
    )
    
    parser.add_argument(
        '--skip-rf',
        action='store_true',
        help='Skip Random Forest model training'
    )
    
    parser.add_argument(
        '--include-gbt',
        action='store_true',
        help='Include Gradient Boosted Trees model (slower)'
    )
    
    args = parser.parse_args()
    
    # Validate data path
    if not os.path.exists(args.data_path.split('/*')[0]):
        print(f"Error: Data path does not exist: {args.data_path}")
        print("\nPlease download the data from:")
        print("https://www.ncei.noaa.gov/data/global-hourly/archive/csv/")
        print("\nExtract the files to the 'data/2024/' directory")
        sys.exit(1)
    
    sys.exit(main(args))
