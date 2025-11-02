"""Entry point for training weather forecasting models with Spark MLlib."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from src.feature_engineering import build_feature_pipeline
from src.modeling import evaluate_models
from src.preprocessing import load_weather_dataframe, prepare_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temperature forecasting models using Spark MLlib")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing extracted NOAA CSV files. If omitted the dataset will be downloaded automatically.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of rows to load for quicker experimentation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/results.json",
        help="Path to store the evaluation summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("WeatherForecasting").getOrCreate()

    raw_df = load_weather_dataframe(spark, data_dir=args.data_dir, row_limit=args.limit)
    cleaned_df = prepare_observations(raw_df)

    train_df, test_df = cleaned_df.randomSplit([0.7, 0.3], seed=42)

    feature_pipeline = build_feature_pipeline()
    results = evaluate_models(train_df, test_df, feature_pipeline)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({name: {"rmse": result.rmse} for name, result in results.items()}, fh, indent=2)

    for name, result in results.items():
        print(f"Model: {name} | RMSE: {result.rmse:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
