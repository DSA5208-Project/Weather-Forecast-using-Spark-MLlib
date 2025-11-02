# Weather Forecast using Spark MLlib

This project builds a complete Apache Spark MLlib pipeline to predict air temperature (`TMP`) from the NOAA Global Hourly surface weather observations. The implementation follows the requirements outlined in **MLlib.pdf** and leverages the ISD format documentation to parse and clean the raw measurements.

## 📦 Dataset acquisition

The official dataset for 2024 is distributed as a compressed archive at:

```
https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2024.tar.gz
```

Running the training script automatically downloads and extracts the archive into `data/` unless the files are already present locally. To reuse an existing extraction, pass the directory path with `--data-dir`.

The repository also contains `sample.csv`, a reduced extract that illustrates the raw schema used by the real dataset.

## 🧹 Data preparation

Key cleaning and enrichment steps (see `src/preprocessing.py`) include:

* Parsing ISD composite fields (`TMP`, `DEW`, `SLP`, `WND`, `VIS`) and discarding sentinel values (`9999`, `+9999`, etc.) described in `isd-format-document.pdf`.
* Validating associated quality flags; only observations whose quality codes are 0–8 are retained.
* Normalising station metadata (latitude, longitude, elevation) and categorical attributes (report type, call sign, quality control source).
* Engineering additional predictors: wind vector components, hour-of-day, day-of-year, and deriving categorical encodings.

The label column is renamed to `label` after extraction so the Spark estimators can consume it directly.

## 🔬 Feature selection strategy

Categorical and continuous predictors are treated independently using Spark's `UnivariateFeatureSelector`:

* **Categorical:** Each field is indexed, one-hot encoded, and then filtered with `featureType="categorical"` to retain the most informative indicators.
* **Continuous:** Numerical attributes are assembled, scored with `featureType="continuous"`, and standardised before modelling.

The selected feature vectors are concatenated into a single `features` column ready for downstream estimators (see `src/feature_engineering.py`).

## 🤖 Models and evaluation

Two regression families are trained via cross-validated pipelines (`src/modeling.py`):

1. **Linear Regression** with Elastic Net regularisation (`regParam` and `elasticNetParam` tuned).
2. **Random Forest Regressor** with tuned `maxDepth` and `numTrees`.

Each model performs three-fold cross-validation on the training subset (70% of the data) and is evaluated on a held-out 30% test split using RMSE.

## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py --limit 500000  # optional row limit for quick local experiments
```

Command-line options:

* `--data-dir`: Path to a directory containing extracted NOAA CSV files (skip download).
* `--limit`: Cap the number of rows ingested (useful for development).
* `--output`: Destination for a JSON summary of the evaluation metrics (defaults to `output/results.json`).

The script prints the RMSE for each trained model and stores the results as JSON.

## 🗂️ Project structure

```
Weather-Forecast-using-Spark-MLlib/
├── main.py
├── requirements.txt
├── sample.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_downloader.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── preprocessing.py
├── MLlib.pdf
├── isd-format-document.pdf
└── ...
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
