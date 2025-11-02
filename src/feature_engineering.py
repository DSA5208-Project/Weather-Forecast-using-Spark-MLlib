"""Feature engineering pipeline for the weather forecasting task."""
from __future__ import annotations

from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    QuantileDiscretizer,
    StandardScaler,
    StringIndexer,
    UnivariateFeatureSelector,
    VectorAssembler,
)

from . import config
from .preprocessing import CATEGORICAL_FEATURE_COLUMNS, CONTINUOUS_FEATURE_COLUMNS


def build_feature_pipeline() -> Pipeline:
    """Create the Spark ML pipeline responsible for feature preparation."""
    continuous_cols: List[str] = CONTINUOUS_FEATURE_COLUMNS
    categorical_cols: List[str] = CATEGORICAL_FEATURE_COLUMNS

    label_bucketizer = QuantileDiscretizer(
        inputCol="label",
        outputCol="label_bucket",
        numBuckets=10,
        handleInvalid="keep",
        relativeError=0.01,
    )

    categorical_indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    categorical_encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in categorical_cols],
        outputCols=[f"{col}_ohe" for col in categorical_cols],
        handleInvalid="keep",
    )
    categorical_assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in categorical_cols],
        outputCol="categorical_features_vec",
    )
    categorical_selector = (
        UnivariateFeatureSelector(
            featuresCol="categorical_features_vec",
            outputCol="selected_categorical_features",
            labelCol="label_bucket",
            selectionMode="numTopFeatures",
        )
        .setFeatureType("categorical")
        .setLabelType("categorical")
        .setSelectionThreshold(config.CATEGORICAL_FEATURES_TO_KEEP)
    )

    continuous_assembler = VectorAssembler(
        inputCols=continuous_cols,
        outputCol="continuous_features_vec",
    )
    continuous_selector = (
        UnivariateFeatureSelector(
            featuresCol="continuous_features_vec",
            outputCol="selected_continuous_features",
            labelCol="label",
            selectionMode="numTopFeatures",
        )
        .setFeatureType("continuous")
        .setLabelType("continuous")
        .setSelectionThreshold(config.CONTINUOUS_FEATURES_TO_KEEP)
    )
    scaler = StandardScaler(
        inputCol="selected_continuous_features",
        outputCol="scaled_continuous_features",
        withMean=True,
        withStd=True,
    )

    final_assembler = VectorAssembler(
        inputCols=["scaled_continuous_features", "selected_categorical_features"],
        outputCol="features",
    )

    stages = [
        label_bucketizer,
        *categorical_indexers,
        categorical_encoder,
        categorical_assembler,
        categorical_selector,
        continuous_assembler,
        continuous_selector,
        scaler,
        final_assembler,
    ]
    return Pipeline(stages=stages)
