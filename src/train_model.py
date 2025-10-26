"""
Model training module for weather forecast project
Implements multiple ML models with hyperparameter tuning
"""

from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import config
import time
from datetime import datetime


def train_linear_regression(train_df, cv_folds=5):
    """
    Train Linear Regression model with cross-validation
    
    Args:
        train_df: Training DataFrame
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (best_model, cv_model, training_time)
    """
    print("\n" + "="*60)
    print("Training Linear Regression Model")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="TMP",
        predictionCol="prediction"
    )
    
    # Create parameter grid
    param_grid = ParamGridBuilder()
    
    for max_iter in config.LINEAR_REGRESSION_PARAMS["maxIter"]:
        param_grid = param_grid.addGrid(lr.maxIter, [max_iter])
    
    for reg_param in config.LINEAR_REGRESSION_PARAMS["regParam"]:
        param_grid = param_grid.addGrid(lr.regParam, [reg_param])
    
    for elastic_net in config.LINEAR_REGRESSION_PARAMS["elasticNetParam"]:
        param_grid = param_grid.addGrid(lr.elasticNetParam, [elastic_net])
    
    param_grid = param_grid.build()
    
    print(f"Testing {len(param_grid)} parameter combinations")
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        seed=config.RANDOM_SEED
    )
    
    # Train model
    print("Starting cross-validation...")
    cv_model = cv.fit(train_df)
    
    training_time = time.time() - start_time
    
    # Get best model
    best_model = cv_model.bestModel
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("\nBest model parameters:")
    print(f"  Max iterations: {best_model.getMaxIter()}")
    print(f"  Regularization: {best_model.getRegParam()}")
    print(f"  Elastic Net: {best_model.getElasticNetParam()}")
    
    return best_model, cv_model, training_time


def train_random_forest(train_df, cv_folds=5):
    """
    Train Random Forest model with cross-validation
    
    Args:
        train_df: Training DataFrame
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (best_model, cv_model, training_time)
    """
    print("\n" + "="*60)
    print("Training Random Forest Model")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize model
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="TMP",
        predictionCol="prediction",
        seed=config.RANDOM_SEED
    )
    
    # Create parameter grid
    param_grid = ParamGridBuilder()
    
    for num_trees in config.RANDOM_FOREST_PARAMS["numTrees"]:
        param_grid = param_grid.addGrid(rf.numTrees, [num_trees])
    
    for max_depth in config.RANDOM_FOREST_PARAMS["maxDepth"]:
        param_grid = param_grid.addGrid(rf.maxDepth, [max_depth])
    
    for min_instances in config.RANDOM_FOREST_PARAMS["minInstancesPerNode"]:
        param_grid = param_grid.addGrid(rf.minInstancesPerNode, [min_instances])
    
    param_grid = param_grid.build()
    
    print(f"Testing {len(param_grid)} parameter combinations")
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        seed=config.RANDOM_SEED
    )
    
    # Train model
    print("Starting cross-validation...")
    cv_model = cv.fit(train_df)
    
    training_time = time.time() - start_time
    
    # Get best model
    best_model = cv_model.bestModel
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("\nBest model parameters:")
    print(f"  Number of trees: {best_model.getNumTrees}")
    print(f"  Max depth: {best_model.getMaxDepth()}")
    print(f"  Min instances per node: {best_model.getMinInstancesPerNode()}")
    
    return best_model, cv_model, training_time


def train_gradient_boosted_trees(train_df, cv_folds=5):
    """
    Train Gradient Boosted Trees model with cross-validation
    
    Args:
        train_df: Training DataFrame
        cv_folds: Number of cross-validation folds
        
    Returns:
        Tuple of (best_model, cv_model, training_time)
    """
    print("\n" + "="*60)
    print("Training Gradient Boosted Trees Model")
    print("="*60)
    
    start_time = time.time()
    
    # Initialize model
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="TMP",
        predictionCol="prediction",
        seed=config.RANDOM_SEED
    )
    
    # Create parameter grid (smaller for GBT as it's computationally expensive)
    param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 10]) \
        .addGrid(gbt.maxIter, [10, 20]) \
        .addGrid(gbt.stepSize, [0.1, 0.2]) \
        .build()
    
    print(f"Testing {len(param_grid)} parameter combinations")
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        labelCol="TMP",
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        seed=config.RANDOM_SEED
    )
    
    # Train model
    print("Starting cross-validation...")
    cv_model = cv.fit(train_df)
    
    training_time = time.time() - start_time
    
    # Get best model
    best_model = cv_model.bestModel
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("\nBest model parameters:")
    print(f"  Max depth: {best_model.getMaxDepth()}")
    print(f"  Max iterations: {best_model.getMaxIter()}")
    print(f"  Step size: {best_model.getStepSize()}")
    
    return best_model, cv_model, training_time


def save_model(model, model_name, output_dir):
    """
    Save trained model to disk
    
    Args:
        model: Trained model to save
        model_name: Name for the saved model
        output_dir: Directory to save the model
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_path = os.path.join(output_dir, model_name)
    
    try:
        model.write().overwrite().save(model_path)
        print(f"\nModel saved to: {model_path}")
    except Exception as e:
        print(f"\nError saving model: {e}")


def get_cv_metrics(cv_model):
    """
    Extract cross-validation metrics
    
    Args:
        cv_model: Fitted CrossValidator model
        
    Returns:
        List of average metrics for each parameter combination
    """
    return cv_model.avgMetrics


if __name__ == "__main__":
    # Test model training
    from data_preprocessing import create_spark_session, preprocess_pipeline
    
    spark = create_spark_session()
    
    try:
        # Load and preprocess data
        data_path = "data/2024/*.csv"
        train_df, test_df, feature_columns, scaler = preprocess_pipeline(spark, data_path)
        
        # Train models
        lr_model, lr_cv, lr_time = train_linear_regression(train_df, cv_folds=config.CV_FOLDS)
        rf_model, rf_cv, rf_time = train_random_forest(train_df, cv_folds=config.CV_FOLDS)
        
        # Save models
        save_model(lr_model, "linear_regression_model", config.MODEL_DIR)
        save_model(rf_model, "random_forest_model", config.MODEL_DIR)
        
        print("\n" + "="*60)
        print("Model training completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
