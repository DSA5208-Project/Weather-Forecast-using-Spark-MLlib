"""
Model Evaluation Module
Computes metrics, generates visualizations, and creates reports.
"""

import logging
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

import src.config as config


class ModelEvaluator:
    """
    Evaluates trained models and generates comprehensive reports.
    """
    
    def __init__(self, spark):
        """
        Initialize evaluator.
        
        Args:
            spark (SparkSession): Active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
        plt.rcParams['font.size'] = 10
        
    def evaluate_model(self, model, test_df, model_name):
        """
        Evaluate a model on test data using multiple metrics.
        
        Args:
            model: Trained Spark ML model
            test_df (DataFrame): Test data
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Cache predictions for multiple metric calculations
        predictions.cache()
        
        # Calculate metrics
        metrics = {}
        
        for metric_name in config.EVALUATION_METRICS:
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName=metric_name
            )
            metric_value = evaluator.evaluate(predictions)
            metrics[metric_name] = metric_value
            self.logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return predictions
    
    def evaluate_all_models(self, models, test_df):
        """
        Evaluate all models on test data.
        
        Args:
            models (dict): Dictionary of trained models
            test_df (DataFrame): Test data
            
        Returns:
            dict: Evaluation results for all models
        """
        self.logger.info("=" * 60)
        self.logger.info("EVALUATING ALL MODELS ON TEST SET")
        self.logger.info("=" * 60)
        
        all_predictions = {}
        
        for model_name, model in models.items():
            predictions = self.evaluate_model(model, test_df, model_name)
            all_predictions[model_name] = predictions
        
        self.logger.info("=" * 60)

        return all_predictions
    
    def compare_models(self):
        """
        Compare all models and identify the best one.
        
        Returns:
            tuple: (best_model_name, comparison_df)
        """
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available")
            return None, None
        
        self.logger.info("Comparing model performance...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.evaluation_results.items():
            row = {"Model": model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values("rmse")
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]["Model"]
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("=" * 60)
        self.logger.info("\n" + comparison_df.to_string(index=False))
        self.logger.info("=" * 60)
        self.logger.info(f"BEST MODEL: {best_model_name}")
        self.logger.info("=" * 60)
        
        return best_model_name, comparison_df
    
    def plot_predictions_vs_actual(self, predictions_df, model_name, output_dir=None):
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            predictions_df (DataFrame): Spark DataFrame with predictions
            model_name (str): Name of the model
            output_dir (str): Output directory for plots
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        self.logger.info(f"Creating predictions vs actual plot for {model_name}...")
        
        # Sample data for plotting (to avoid memory issues)
        sample_size = min(10000, predictions_df.count())
        sample_df = predictions_df.sample(False, sample_size / predictions_df.count(), seed=42)
        
        # Convert to pandas
        plot_data = sample_df.select("label", "prediction").toPandas()
        
        # Create plot
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        
        # Scatter plot
        ax.scatter(plot_data["label"], plot_data["prediction"], 
                  alpha=0.5, s=10, label="Predictions")
        
        # Perfect prediction line
        min_val = min(plot_data["label"].min(), plot_data["prediction"].min())
        max_val = max(plot_data["label"].max(), plot_data["prediction"].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label="Perfect Prediction")
        
        # Labels and title
        ax.set_xlabel("Actual Temperature (°C)", fontsize=12)
        ax.set_ylabel("Predicted Temperature (°C)", fontsize=12)
        ax.set_title(f"Predictions vs Actual - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(output_dir, f"predictions_vs_actual_{model_name}.{config.PLOT_FORMAT}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plot saved: {plot_path}")
    
    def plot_residuals(self, predictions_df, model_name, output_dir=None):
        """
        Create residual plot.
        
        Args:
            predictions_df (DataFrame): Spark DataFrame with predictions
            model_name (str): Name of the model
            output_dir (str): Output directory for plots
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        self.logger.info(f"Creating residual plot for {model_name}...")
        
        # Add residuals column
        predictions_with_residuals = predictions_df.withColumn(
            "residual", col("prediction") - col("label")
        )
        
        # Sample data
        sample_size = min(10000, predictions_with_residuals.count())
        sample_df = predictions_with_residuals.sample(
            False, sample_size / predictions_with_residuals.count(), seed=42
        )
        
        # Convert to pandas
        plot_data = sample_df.select("prediction", "residual").toPandas()
        
        # Create plot
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        
        # Scatter plot
        ax.scatter(plot_data["prediction"], plot_data["residual"], 
                  alpha=0.5, s=10)
        
        # Zero line
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        # Labels and title
        ax.set_xlabel("Predicted Temperature (°C)", fontsize=12)
        ax.set_ylabel("Residuals (Predicted - Actual)", fontsize=12)
        ax.set_title(f"Residual Plot - {model_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(output_dir, f"residuals_{model_name}.{config.PLOT_FORMAT}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plot saved: {plot_path}")
    
    def plot_error_distribution(self, predictions_df, model_name, output_dir=None):
        """
        Create histogram of prediction errors.
        
        Args:
            predictions_df (DataFrame): Spark DataFrame with predictions
            model_name (str): Name of the model
            output_dir (str): Output directory for plots
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        self.logger.info(f"Creating error distribution plot for {model_name}...")
        
        # Calculate errors
        predictions_with_error = predictions_df.withColumn(
            "error", col("prediction") - col("label")
        )
        
        # Convert to pandas
        errors = predictions_with_error.select("error").toPandas()["error"]
        
        # Create plot
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        
        # Histogram
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        
        # Mean line
        mean_error = errors.mean()
        ax.axvline(x=mean_error, color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_error:.3f}°C')
        
        # Labels and title
        ax.set_xlabel("Prediction Error (°C)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Error Distribution - {model_name}", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save plot
        plot_path = os.path.join(output_dir, f"error_distribution_{model_name}.{config.PLOT_FORMAT}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plot saved: {plot_path}")
    
    def plot_model_comparison(self, comparison_df, output_dir=None):
        """
        Create bar plot comparing all models.
        
        Args:
            comparison_df (DataFrame): Pandas DataFrame with comparison results
            output_dir (str): Output directory for plots
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIR
        
        self.logger.info("Creating model comparison plot...")
        
        # Create subplots for each metric
        metrics = [m for m in config.EVALUATION_METRICS if m in comparison_df.columns]
        num_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 6))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Sort by metric value
            sorted_df = comparison_df.sort_values(metric)
            
            # Create bar plot
            bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
            
            # Color the best model differently
            bars[0].set_color('green')
            
            # Customize plot
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df["Model"], rotation=45, ha='right')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f"Model Comparison - {metric.upper()}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Save plot
        plot_path = os.path.join(output_dir, f"model_comparison.{config.PLOT_FORMAT}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plot saved: {plot_path}")
    
    def create_all_visualizations(self, all_predictions, comparison_df):
        """
        Create all visualizations for all models.
        
        Args:
            all_predictions (dict): Dictionary of predictions for each model
            comparison_df (DataFrame): Model comparison DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("CREATING VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        # Model comparison plot
        self.plot_model_comparison(comparison_df)
        
        # Individual model plots
        for model_name, predictions_df in all_predictions.items():
            self.plot_predictions_vs_actual(predictions_df, model_name)
            self.plot_residuals(predictions_df, model_name)
            self.plot_error_distribution(predictions_df, model_name)
        
        self.logger.info("All visualizations created successfully")
        self.logger.info("=" * 60)
    
    def save_results(self, comparison_df, output_path=None):
        """
        Save evaluation results to CSV.
        
        Args:
            comparison_df (DataFrame): Model comparison DataFrame
            output_path (str): Output file path
        """
        if output_path is None:
            output_path = config.RESULTS_CSV
        
        comparison_df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to: {output_path}")
    
    def generate_report(self, comparison_df, training_summary, output_path=None):
        """
        Generate a comprehensive text report.
        
        Args:
            comparison_df (DataFrame): Model comparison DataFrame
            training_summary (dict): Training summary dictionary
            output_path (str): Output file path
        """
        if output_path is None:
            output_path = config.REPORT_PATH
        
        self.logger.info("Generating comprehensive report...")
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WEATHER FORECAST MODEL TRAINING AND EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write("1. DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Source: {config.DATASET_URL}\n")
            f.write(f"Target Variable: {config.TARGET_COLUMN} (Air Temperature)\n")
            f.write(f"Train/Test Split: {config.TRAIN_TEST_SPLIT_RATIO*100:.0f}% / "
                   f"{(1-config.TRAIN_TEST_SPLIT_RATIO)*100:.0f}%\n")
            f.write(f"Random Seed: {config.RANDOM_SEED}\n\n")
            
            # Preprocessing info
            f.write("2. DATA PREPROCESSING\n")
            f.write("-" * 80 + "\n")
            temp_min = getattr(config, "TEMP_MIN", None)
            temp_max = getattr(config, "TEMP_MAX", None)
            if temp_min is not None and temp_max is not None:
                f.write(f"Temperature Range Filter: {temp_min}°C to {temp_max}°C\n")
            else:
                f.write("Temperature Range Filter: Not specified in configuration\n")

            fill_strategy_desc = (
                f"Drop columns with >{int(config.MAX_MISSING_PERCENT * 100)}% missing, "
                "mean/mode imputation for remaining values"
            )
            f.write(f"Missing Value Strategy: {fill_strategy_desc}\n")

            standardize_flag = getattr(
                config,
                "STANDARDIZE_FEATURES",
                getattr(config, "STANDARDIZE_CONTINUOUS", False),
            )
            f.write(f"Feature Standardization: {'Enabled' if standardize_flag else 'Disabled'}\n")

            if getattr(config, "SKIP_FEATURE_SELECTION", False):
                feature_selection_desc = "Skipped"
                feature_selection_param = "N/A"
            else:
                fs_cfg = getattr(config, "CONTINUOUS_FEATURE_SELECTION", {}) or {}
                mode = fs_cfg.get("selectionMode", "N/A")
                threshold = fs_cfg.get("selectionThreshold", "N/A")
                feature_selection_desc = f"Univariate selector (mode: {mode})"
                feature_selection_param = f"Threshold: {threshold}"

            f.write(f"Feature Selection Method: {feature_selection_desc}\n")
            f.write(f"Feature Selection Parameter: {feature_selection_param}\n\n")
            
            # Training info
            f.write("3. MODEL TRAINING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Models Trained: {training_summary['num_models_trained']}\n")
            f.write(f"Cross-Validation Folds: {config.NUM_FOLDS}\n")
            f.write(f"Best Model: {training_summary['best_model']}\n\n")
            
            # Model comparison
            f.write("4. MODEL COMPARISON (TEST SET PERFORMANCE)\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Individual model details
            f.write("5. DETAILED MODEL RESULTS\n")
            f.write("-" * 80 + "\n")
            for model_name, results in training_summary["model_results"].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Cross-Validation RMSE: {results['best_cv_rmse']:.4f}\n")
                f.write(f"  Training Time: {results['training_time']:.2f} seconds\n")
                f.write(f"  Best Parameters:\n")
                for param, value in results["best_params"].items():
                    f.write(f"    {param}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Report saved to: {output_path}")
