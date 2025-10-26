"""
Example script for loading and making predictions with trained models
"""

from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
from data_preprocessing import create_spark_session, preprocess_pipeline
import config
import os


def load_model(model_path, model_type="linear_regression"):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('linear_regression' or 'random_forest')
        
    Returns:
        Loaded model
    """
    if model_type == "linear_regression":
        return LinearRegressionModel.load(model_path)
    elif model_type == "random_forest":
        return RandomForestRegressionModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def make_predictions(model, data_df):
    """
    Make predictions using a trained model
    
    Args:
        model: Trained model
        data_df: DataFrame with features
        
    Returns:
        DataFrame with predictions
    """
    predictions = model.transform(data_df)
    return predictions


def main():
    """
    Example of loading a model and making predictions
    """
    print("Loading trained model...")
    
    # Create Spark session
    spark = create_spark_session("ModelInference")
    
    try:
        # Load the best model (replace with your model path)
        model_path = os.path.join(config.MODEL_DIR, "random_forest_model")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train a model first by running: python main.py")
            return
        
        model = load_model(model_path, model_type="random_forest")
        print(f"Model loaded from: {model_path}")
        
        # Load new data for prediction
        print("\nLoading data for prediction...")
        data_path = "data/2024/*.csv"  # Adjust as needed
        
        # Preprocess data (same as training)
        train_df, test_df, feature_columns, scaler = preprocess_pipeline(
            spark,
            data_path
        )
        
        # Make predictions on test set
        print("\nMaking predictions...")
        predictions = make_predictions(model, test_df)
        
        # Show sample predictions
        print("\nSample predictions:")
        predictions.select("TMP", "prediction").show(20)
        
        # Calculate prediction statistics
        from pyspark.sql.functions import avg, stddev, min as spark_min, max as spark_max
        
        stats = predictions.select(
            avg("TMP").alias("avg_actual"),
            avg("prediction").alias("avg_predicted"),
            stddev("TMP").alias("std_actual"),
            stddev("prediction").alias("std_predicted"),
            spark_min("TMP").alias("min_actual"),
            spark_max("TMP").alias("max_actual")
        ).collect()[0]
        
        print("\nPrediction Statistics:")
        print(f"Average actual temperature: {stats['avg_actual']:.2f}°C")
        print(f"Average predicted temperature: {stats['avg_predicted']:.2f}°C")
        print(f"Temperature range: {stats['min_actual']:.2f}°C to {stats['max_actual']:.2f}°C")
        
        # Save predictions to CSV
        output_path = "output/predictions.csv"
        print(f"\nSaving predictions to: {output_path}")
        predictions.select("TMP", "prediction").coalesce(1).write.csv(
            output_path,
            header=True,
            mode="overwrite"
        )
        print("Predictions saved!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
