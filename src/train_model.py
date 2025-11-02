"""
Model Training Module
Implements multiple regression models with cross-validation for hyperparameter tuning.
"""

import logging
import time
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor,
    GBTRegressor, GeneralizedLinearRegression
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import src.config as config


class ModelTrainer:
    """
    Trains and tunes multiple regression models for temperature prediction.
    """
    
    def __init__(self, spark):
        """
        Initialize model trainer.
        
        Args:
            spark (SparkSession): Active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        
    def create_model(self, model_name):
        """
        Create a regression model instance.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Estimator: Spark ML model instance
        """
        self.logger.info(f"Creating model: {model_name}")
        
        if model_name == "LinearRegression":
            model = LinearRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction"
            )
        
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=config.RANDOM_SEED
            )
        
        elif model_name == "GBTRegressor":
            model = GBTRegressor(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=config.RANDOM_SEED
            )
        
        elif model_name == "GeneralizedLinearRegression":
            model = GeneralizedLinearRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction"
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def create_param_grid(self, model, model_name):
        """
        Create parameter grid for hyperparameter tuning.
        
        Args:
            model: Spark ML model instance
            model_name (str): Name of the model
            
        Returns:
            ParamGrid: Parameter grid for cross-validation
        """
        self.logger.info(f"Creating parameter grid for {model_name}")
        
        builder = ParamGridBuilder()
        
        if model_name not in config.HYPERPARAMETERS:
            self.logger.warning(f"No hyperparameters defined for {model_name}, using defaults")
            return builder.build()
        
        params = config.HYPERPARAMETERS[model_name]
        
        # Add parameters to grid
        if model_name == "LinearRegression":
            if "regParam" in params:
                builder.addGrid(model.regParam, params["regParam"])
            if "elasticNetParam" in params:
                builder.addGrid(model.elasticNetParam, params["elasticNetParam"])
        
        elif model_name == "RandomForestRegressor":
            if "numTrees" in params:
                builder.addGrid(model.numTrees, params["numTrees"])
            if "maxDepth" in params:
                builder.addGrid(model.maxDepth, params["maxDepth"])
            if "minInstancesPerNode" in params:
                builder.addGrid(model.minInstancesPerNode, params["minInstancesPerNode"])
        
        elif model_name == "GBTRegressor":
            if "maxIter" in params:
                builder.addGrid(model.maxIter, params["maxIter"])
            if "maxDepth" in params:
                builder.addGrid(model.maxDepth, params["maxDepth"])
            if "stepSize" in params:
                builder.addGrid(model.stepSize, params["stepSize"])
        
        elif model_name == "GeneralizedLinearRegression":
            if "family" in params:
                builder.addGrid(model.family, params["family"])
            if "link" in params:
                builder.addGrid(model.link, params["link"])
            if "regParam" in params:
                builder.addGrid(model.regParam, params["regParam"])
        
        param_grid = builder.build()
        self.logger.info(f"Created parameter grid with {len(param_grid)} combinations")
        
        return param_grid
    
    def train_model_with_cv(self, train_df, model_name):
        """
        Train a model with cross-validation for hyperparameter tuning.
        
        Args:
            train_df (DataFrame): Training data
            model_name (str): Name of the model to train
            
        Returns:
            tuple: (best_model, cv_results, training_time)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING MODEL: {model_name}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create model
        model = self.create_model(model_name)
        
        # Create parameter grid
        param_grid = self.create_param_grid(model, model_name)
        
        # Create evaluator (using RMSE as primary metric)
        evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        # Create cross-validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=config.NUM_FOLDS,
            parallelism=config.PARALLELISM,
            seed=config.RANDOM_SEED
        )
        
        self.logger.info(f"Starting {config.NUM_FOLDS}-fold cross-validation...")
        self.logger.info(f"Testing {len(param_grid)} parameter combinations")
        
        # Fit model
        cv_model = cv.fit(train_df)
        
        training_time = time.time() - start_time
        
        # Get best model and metrics
        best_model = cv_model.bestModel
        avg_metrics = cv_model.avgMetrics
        
        # Log results
        best_rmse = min(avg_metrics)
        best_params_idx = avg_metrics.index(best_rmse)
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best cross-validation RMSE: {best_rmse:.4f}")
        self.logger.info(f"Best parameters (combination #{best_params_idx + 1}):")
        
        # Extract and log best parameters
        best_params = param_grid[best_params_idx]
        for param, value in best_params.items():
            self.logger.info(f"  {param.name}: {value}")
        
        # Store results
        cv_results = {
            "model_name": model_name,
            "best_cv_rmse": best_rmse,
            "all_cv_rmse": avg_metrics,
            "best_params": {param.name: value for param, value in best_params.items()},
            "training_time": training_time,
            "num_param_combinations": len(param_grid)
        }
        
        self.logger.info("=" * 60)
        
        return best_model, cv_results, training_time
    
    def train_all_models(self, train_df):
        """
        Train all models specified in config.
        
        Args:
            train_df (DataFrame): Training data
            
        Returns:
            dict: Dictionary of trained models and results
        """
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING {len(config.MODELS_TO_TRAIN)} MODELS")
        self.logger.info("=" * 60)
        
        total_start_time = time.time()
        
        for model_name in config.MODELS_TO_TRAIN:
            try:
                # Train model with cross-validation
                best_model, cv_results, training_time = self.train_model_with_cv(
                    train_df, model_name
                )
                
                # Store model and results
                self.models[model_name] = best_model
                self.training_results[model_name] = cv_results
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        total_time = time.time() - total_start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"ALL MODELS TRAINED IN {total_time:.2f} SECONDS")
        self.logger.info("=" * 60)
        
        # Select best model based on CV RMSE
        self.select_best_model()
        
        return self.models, self.training_results
    
    def select_best_model(self):
        """
        Select the best model based on cross-validation RMSE.
        """
        if not self.training_results:
            self.logger.warning("No training results available")
            return
        
        self.logger.info("Selecting best model based on cross-validation RMSE...")
        
        best_rmse = float('inf')
        best_name = None
        
        for model_name, results in self.training_results.items():
            cv_rmse = results["best_cv_rmse"]
            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        self.logger.info("=" * 60)
        self.logger.info(f"BEST MODEL: {best_name}")
        self.logger.info(f"Cross-validation RMSE: {best_rmse:.4f}")
        self.logger.info("=" * 60)
        
        return self.best_model, self.best_model_name
    
    def save_model(self, model, model_name, path=None):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained Spark ML model
            model_name (str): Name of the model
            path (str): Save path (default from config)
        """
        if path is None:
            path = f"{config.ALL_MODELS_PATH}/{model_name}"
        
        try:
            model.write().overwrite().save(path)
            self.logger.info(f"Model saved: {path}")
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
    
    def save_all_models(self):
        """
        Save all trained models.
        """
        self.logger.info("Saving all trained models...")
        
        for model_name, model in self.models.items():
            self.save_model(model, model_name)
        
        # Save best model to special location
        if self.best_model is not None:
            self.save_model(self.best_model, self.best_model_name, config.BEST_MODEL_PATH)
            self.logger.info(f"Best model ({self.best_model_name}) saved to: {config.BEST_MODEL_PATH}")
    
    def get_training_summary(self):
        """
        Get a summary of all training results.
        
        Returns:
            dict: Training summary
        """
        summary = {
            "num_models_trained": len(self.models),
            "best_model": self.best_model_name,
            "model_results": {}
        }
        
        for model_name, results in self.training_results.items():
            summary["model_results"][model_name] = {
                "best_cv_rmse": results["best_cv_rmse"],
                "training_time": results["training_time"],
                "best_params": results["best_params"]
            }
        
        return summary
    
    def print_training_summary(self):
        """
        Print a formatted summary of training results.
        """
        self.logger.info("=" * 60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 60)
        
        # Sort models by CV RMSE
        sorted_results = sorted(
            self.training_results.items(),
            key=lambda x: x[1]["best_cv_rmse"]
        )
        
        self.logger.info(f"{'Model':<30} {'CV RMSE':<12} {'Time (s)':<10}")
        self.logger.info("-" * 60)
        
        for model_name, results in sorted_results:
            rmse = results["best_cv_rmse"]
            time_taken = results["training_time"]
            marker = "(*)" if model_name == self.best_model_name else "   "
            self.logger.info(f"{model_name:<30} {rmse:<12.4f} {time_taken:<10.2f} {marker}")
        
        self.logger.info("-" * 60)
        self.logger.info("(*) = Best model")
        self.logger.info("=" * 60)
