"""Model training utilities for the weather forecasting pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


@dataclass
class ModelResult:
    name: str
    rmse: float
    pipeline_model: PipelineModel


def _regression_evaluator() -> RegressionEvaluator:
    return RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")


def train_linear_regression(base_pipeline: Pipeline) -> Tuple[Pipeline, CrossValidator]:
    lr = LinearRegression(featuresCol="features", labelCol="label")
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )
    pipeline = Pipeline(stages=base_pipeline.getStages() + [lr])
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=_regression_evaluator(),
        numFolds=3,
        parallelism=1,
    )
    return pipeline, cv


def train_random_forest(base_pipeline: Pipeline) -> Tuple[Pipeline, CrossValidator]:
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", seed=13)
    grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [8, 12])
        .addGrid(rf.numTrees, [50, 100])
        .build()
    )
    pipeline = Pipeline(stages=base_pipeline.getStages() + [rf])
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=_regression_evaluator(),
        numFolds=3,
        parallelism=1,
    )
    return pipeline, cv


def evaluate_models(train_df, test_df, base_pipeline: Pipeline) -> Dict[str, ModelResult]:
    """Train and evaluate the regression models."""
    results: Dict[str, ModelResult] = {}
    for name, trainer in {
        "LinearRegression": train_linear_regression,
        "RandomForest": train_random_forest,
    }.items():
        _, cv = trainer(base_pipeline)
        model = cv.fit(train_df)
        predictions = model.transform(test_df)
        rmse = _regression_evaluator().evaluate(predictions)
        results[name] = ModelResult(name=name, rmse=rmse, pipeline_model=model.bestModel)
    return results
