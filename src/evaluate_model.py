"""
Model evaluation module for weather forecast project
Evaluates trained models and generates performance metrics
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as sql_abs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def evaluate_model(model, test_df, model_name="Model"):
    """
    Evaluate a trained model on test data
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print('='*60)
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Calculate various metrics
    evaluator_rmse = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    evaluator_mae = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="mae"
    )
    
    evaluator_r2 = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="r2"
    )
    
    evaluator_mse = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="mse"
    )
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mse = evaluator_mse.evaluate(predictions)
    
    metrics = {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse
    }
    
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MSE:  {mse:.4f}")
    
    return metrics, predictions


def plot_predictions(predictions, model_name, output_dir):
    """
    Create visualization of predictions vs actual values
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    print(f"\nGenerating plots for {model_name}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to Pandas for plotting (sample if too large)
    sample_size = min(10000, predictions.count())
    pred_pd = predictions.select("TMP", "prediction").sample(
        fraction=sample_size/predictions.count(),
        seed=42
    ).toPandas()
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(pred_pd['TMP'], pred_pd['prediction'], alpha=0.5, s=10)
    
    # Add perfect prediction line
    min_val = min(pred_pd['TMP'].min(), pred_pd['prediction'].min())
    max_val = max(pred_pd['TMP'].max(), pred_pd['prediction'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax1.set_title('Predicted vs Actual Temperature', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = pred_pd['TMP'] - pred_pd['prediction']
    ax2.scatter(pred_pd['prediction'], residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Residuals (°C)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Residuals (°C)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = axes[1, 1]
    errors = np.abs(residuals)
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xlabel('Absolute Error (°C)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{model_name}_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def plot_cv_results(cv_metrics, param_names, model_name, output_dir):
    """
    Plot cross-validation results
    
    Args:
        cv_metrics: List of CV metrics
        param_names: Names of parameters
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    print(f"\nGenerating CV plots for {model_name}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cv_metrics)), cv_metrics, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Parameter Combination', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Cross-Validation Results - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = np.argmin(cv_metrics)
    plt.plot(best_idx, cv_metrics[best_idx], 'r*', markersize=20, label=f'Best (RMSE={cv_metrics[best_idx]:.4f})')
    plt.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{model_name}_cv_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"CV plot saved to: {plot_path}")
    plt.close()


def compare_models(metrics_list, output_dir):
    """
    Create comparison visualizations for multiple models
    
    Args:
        metrics_list: List of metric dictionaries
        output_dir: Directory to save plots
    """
    print("\nGenerating model comparison plots...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['rmse', 'mae', 'r2', 'mse']
    titles = ['Root Mean Squared Error', 'Mean Absolute Error', 'R² Score', 'Mean Squared Error']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        bars = ax.bar(df['model_name'], df[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Rotate x labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def save_metrics_report(metrics_list, output_dir):
    """
    Save metrics to a text file
    
    Args:
        metrics_list: List of metric dictionaries
        output_dir: Directory to save report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, "metrics_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        for metrics in metrics_list:
            f.write(f"\n{metrics['model_name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"MAE:  {metrics['mae']:.6f}\n")
            f.write(f"R²:   {metrics['r2']:.6f}\n")
            f.write(f"MSE:  {metrics['mse']:.6f}\n")
        
        # Find best model
        best_model = min(metrics_list, key=lambda x: x['rmse'])
        f.write("\n" + "="*60 + "\n")
        f.write(f"BEST MODEL: {best_model['model_name']}\n")
        f.write(f"RMSE: {best_model['rmse']:.6f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nMetrics report saved to: {report_path}")


if __name__ == "__main__":
    # Test evaluation
    from data_preprocessing import create_spark_session, preprocess_pipeline
    from train_model import train_linear_regression, train_random_forest
    import config
    
    spark = create_spark_session()
    
    try:
        # Load and preprocess data
        data_path = "data/2024/*.csv"
        train_df, test_df, feature_columns, scaler = preprocess_pipeline(spark, data_path)
        
        # Train models
        lr_model, lr_cv, _ = train_linear_regression(train_df, cv_folds=3)
        rf_model, rf_cv, _ = train_random_forest(train_df, cv_folds=3)
        
        # Evaluate models
        lr_metrics, lr_pred = evaluate_model(lr_model, test_df, "Linear Regression")
        rf_metrics, rf_pred = evaluate_model(rf_model, test_df, "Random Forest")
        
        # Generate visualizations
        plot_predictions(lr_pred, "Linear_Regression", config.OUTPUT_DIR)
        plot_predictions(rf_pred, "Random_Forest", config.OUTPUT_DIR)
        
        # Compare models
        compare_models([lr_metrics, rf_metrics], config.OUTPUT_DIR)
        
        # Save report
        save_metrics_report([lr_metrics, rf_metrics], config.OUTPUT_DIR)
        
        print("\n" + "="*60)
        print("Evaluation completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
