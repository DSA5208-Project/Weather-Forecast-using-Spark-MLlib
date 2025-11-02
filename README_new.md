# Weather Forecast using Spark MLlib

A machine learning project to predict air temperature from weather observation data using Apache Spark MLlib.

## 📋 Project Overview

This project implements a complete machine learning pipeline for weather temperature prediction using Apache Spark. The system processes hourly surface weather observations from stations worldwide and builds predictive models using various regression algorithms.

**Dataset**: Global Hourly Weather Data (2024)  
**Source**: [NOAA Global Hourly Dataset](https://www.ncei.noaa.gov/data/global-hourly/archive/csv/)  
**Target Variable**: Air Temperature (TMP)  
**Documentation**: [Dataset Documentation](https://www.ncei.noaa.gov/data/global-hourly/doc/)

## 🎯 Project Objectives

- Preprocess large-scale weather data using Apache Spark
- Build and compare multiple machine learning regression models
- Perform hyperparameter tuning using cross-validation
- Evaluate model performance using standard metrics (RMSE, MAE, R²)
- Generate comprehensive visualizations and reports

## 🏗️ Project Structure

```
Weather-Forecast-using-Spark-MLlib/
│
├── main.py                       # Main execution script (entry point)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # Project license
├── .gitignore                    # Git ignore rules
├── sample.csv                    # Sample data for testing
├── MLlib.pdf                     # Project requirements document
├── isd-format-document.pdf       # Dataset format documentation
│
├── src/                          # Source code directory
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Configuration parameters
│   ├── data_preprocessing.py    # Data loading and cleaning
│   ├── feature_engineering.py   # Feature selection with UnivariateFeatureSelector
│   ├── train_model.py           # Model training with CV
│   ├── evaluate_model.py        # Model evaluation and visualization
│   └── utils.py                 # Utility functions
│
├── data/                         # Raw data (download 2024.tar.gz here)
│
├── models/                       # Trained models (auto-created)
│   ├── best_model/              # Best performing model
│   └── all_models/              # All trained models
│
└── output/                       # Results and visualizations (auto-created)
    ├── model_results.csv        # Performance metrics
    ├── feature_importance.csv   # Feature selection results
    ├── model_report.txt         # Comprehensive text report
    ├── training.log             # Execution log
    └── *.png                    # Visualization plots
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Java**: JDK 8 or 11 (required for PySpark)
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~5GB for dataset and outputs

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DSA5208-Project/Weather-Forecast-using-Spark-MLlib.git
   cd Weather-Forecast-using-Spark-MLlib
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Java installation**:
   ```bash
   java -version
   # Should show Java 8 or 11
   ```

### Download Dataset

Download the 2024 weather data:

```bash
# Create data directory
mkdir -p data

# Download the dataset (Option 1: Using wget)
wget https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2024.tar.gz -O data/2024.tar.gz

# Or (Option 2: Using curl)
curl -o data/2024.tar.gz https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2024.tar.gz

# Extract the archive
cd data
tar -xzf 2024.tar.gz
cd ..
```

**Note**: The full dataset is large (~2GB compressed). For testing, you can use the provided `sample.csv` file.

## 📊 Usage

### Quick Start (Using Sample Data)

Test the pipeline with sample data:

```bash
python main.py --use-sample
```

### Full Pipeline (Using Complete Dataset)

Run the complete pipeline with all features:

```bash
python main.py
```

### Custom Options

```bash
# Use custom data file
python main.py --data-path path/to/your/data.csv

# Train specific models only
python main.py --models LinearRegression RandomForestRegressor

# Skip feature selection (use all features)
python main.py --skip-feature-selection

# Combine options
python main.py --use-sample --models LinearRegression GBTRegressor
```

### Command-Line Arguments

- `--data-path PATH`: Path to CSV data file
- `--use-sample`: Use sample.csv for testing
- `--skip-feature-selection`: Skip feature selection step
- `--models MODEL1 MODEL2 ...`: Train specific models only

### Available Models

- `LinearRegression`: Linear regression with regularization
- `RandomForestRegressor`: Random forest ensemble
- `GBTRegressor`: Gradient-boosted trees
- `GeneralizedLinearRegression`: Generalized linear model

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Data preprocessing**: Missing value handling, outlier removal
- **Feature selection**: Method (fpr, numTopFeatures, etc.) and parameters
- **Models**: Hyperparameter grids for cross-validation
- **Training**: Number of CV folds, parallelism
- **Output**: Plot settings, file paths

Key configuration options:

```python
# Feature selection
FEATURE_SELECTION_METHOD = "fpr"  # False Positive Rate
FEATURE_SELECTION_PARAM = 0.05    # Threshold

# Training
NUM_FOLDS = 5                     # Cross-validation folds
TRAIN_TEST_SPLIT_RATIO = 0.7      # 70% train, 30% test

# Models to train
MODELS_TO_TRAIN = [
    "LinearRegression",
    "RandomForestRegressor",
    "GBTRegressor",
    "GeneralizedLinearRegression"
]
```

## 📈 Pipeline Steps

### 1. Data Preprocessing

- **Loading**: Read CSV data with Spark
- **Parsing**: Extract features from complex weather columns (WND, TMP, DEW, SLP, etc.)
- **Cleaning**: Remove invalid values (9999, 999 indicators)
- **Filtering**: Apply temperature range filters (-90°C to 60°C)
- **Imputation**: Fill missing values using median strategy
- **Outlier Removal**: Remove statistical outliers using IQR method
- **Encoding**: Convert categorical features (STATION) to numeric
- **Standardization**: Scale features using StandardScaler

### 2. Feature Engineering

- **Feature Extraction**: Parse weather observations into numeric features
  - Wind direction and speed
  - Ceiling height
  - Visibility distance
  - Dew point temperature
  - Sea level pressure
  - Geographic coordinates (latitude, longitude, elevation)
  - Temporal features (hour, month, day of year)

- **Feature Selection**: Use UnivariateFeatureSelector
  - Method: F-test for regression (f_regression)
  - Selection mode: False Positive Rate (fpr) with threshold 0.05
  - Reduces dimensionality by selecting statistically significant features

### 3. Model Training

- **Cross-Validation**: K-fold CV (default: 5 folds)
- **Hyperparameter Tuning**: Grid search over parameter combinations
- **Multiple Models**: Train and compare 4 regression algorithms
- **Best Model Selection**: Choose model with lowest CV RMSE

### 4. Model Evaluation

**Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MSE (Mean Squared Error)

**Visualizations**:
- Predictions vs Actual scatter plots
- Residual plots
- Error distribution histograms
- Model comparison bar charts

### 5. Results and Reporting

- CSV files with detailed metrics
- Comprehensive text report
- High-resolution plots (300 DPI PNG)
- Feature importance rankings
- Training logs

## 📁 Output Files

After execution, check the `output/` directory:

- **model_results.csv**: Performance metrics for all models
- **feature_importance.csv**: Feature correlation scores
- **model_report.txt**: Comprehensive text report
- **training.log**: Detailed execution log
- **predictions_vs_actual_*.png**: Scatter plots of predictions
- **residuals_*.png**: Residual analysis plots
- **error_distribution_*.png**: Error distribution histograms
- **model_comparison.png**: Bar chart comparing all models

## 🧪 Testing

Test with sample data (fast execution):

```bash
python main.py --use-sample
```

This uses `sample.csv` (~100 rows) for quick validation.

## 🐛 Troubleshooting

### Common Issues

**1. Java not found**:
```
Error: JAVA_HOME is not set
```
Solution: Install Java 8 or 11 and set JAVA_HOME environment variable

**2. Memory errors**:
```
OutOfMemoryError: Java heap space
```
Solution: Increase Spark memory in `src/config.py`:
```python
SPARK_CONFIG = {
    "spark.driver.memory": "8g",  # Increase from 4g
    "spark.executor.memory": "8g"
}
```

**3. PySpark not found**:
```
ModuleNotFoundError: No module named 'pyspark'
```
Solution: Install dependencies:
```bash
pip install -r requirements.txt
```

**4. Dataset not found**:
```
FileNotFoundError: sample.csv
```
Solution: Ensure you're running from the project root directory

## 📝 Requirements

See [MLlib.pdf](MLlib.pdf) for detailed project requirements.

### Key Requirements Met

✅ Data preprocessing with invalid value removal  
✅ Feature standardization  
✅ Train/test split (70/30)  
✅ Multiple ML models (4 algorithms)  
✅ Cross-validation for hyperparameter tuning  
✅ UnivariateFeatureSelector for feature selection  
✅ Comprehensive evaluation (RMSE, MAE, R²)  
✅ Visualizations and performance plots  
✅ Complete source code with README  
✅ Trained model persistence  
✅ Detailed report generation  

## 📚 Documentation

- **Dataset Format**: See [isd-format-document.pdf](isd-format-document.pdf)
- **Project Requirements**: See [MLlib.pdf](MLlib.pdf)
- **NOAA Documentation**: https://www.ncei.noaa.gov/data/global-hourly/doc/

## 🏆 Results

Example results (will vary based on data):

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 2.34 | 1.78 | 0.94 |
| GBT | 2.45 | 1.85 | 0.93 |
| Linear Regression | 3.12 | 2.41 | 0.88 |
| GLM | 3.15 | 2.43 | 0.87 |

## 👥 Group Members

List your group members here:
- Name 1 (Student ID)
- Name 2 (Student ID)
- Name 3 (Student ID)

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NOAA for providing the Global Hourly Weather Dataset
- Apache Spark community for MLlib
- Course instructors and TAs for guidance
