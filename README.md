# Weather Forecast using Spark MLlib

A scalable machine learning project to predict air temperature from weather observation data using Apache Spark MLlib on Google Cloud Platform.

## Project Overview

This project implements a complete distributed machine learning pipeline for weather temperature prediction using Apache Spark. The system processes hourly surface weather observations from stations worldwide and builds predictive models using various regression algorithms, designed to run on Google Cloud Dataproc clusters.

**Dataset**: Global Hourly Weather Data (2024)  
**Source**: [NOAA Global Hourly Dataset](https://www.ncei.noaa.gov/data/global-hourly/archive/csv/)  
**Target Variable**: Air Temperature (TMP)  
**Documentation**: [Dataset Documentation](https://www.ncei.noaa.gov/data/global-hourly/doc/)

## Project Objectives

- Preprocess large-scale weather data using Apache Spark on distributed clusters
- Build and compare multiple machine learning regression models
- Perform hyperparameter tuning using cross-validation
- Evaluate model performance using standard metrics (RMSE, MAE, R²)
- Generate comprehensive visualizations and reports
- Deploy on Google Cloud Platform for scalable processing

## Project Structure

```
Weather-Forecast-using-Spark-MLlib/
│
├── main.py                       # Main execution script (entry point)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # Project license
├── sample.csv                    # Sample weather data
│
├── src/                          # Source code directory
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Configuration parameters
│   ├── data_preprocessing.py    # Data loading and cleaning
│   ├── feature_engineering.py   # Feature selection and engineering
│   ├── train_model.py           # Model training with CV
│   ├── evaluate_model.py        # Model evaluation and visualization
│   └── utils.py                 # Utility functions
│
├── models/                       # Trained models (auto-created)
└── output/                       # Results and visualizations (auto-created)
```

## Quick Start (Local Setup)

### Prerequisites

- Python 3.8 or higher
- Java 8 or 11 (required for PySpark)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DSA5208-Project/Weather-Forecast-using-Spark-MLlib.git
cd Weather-Forecast-using-Spark-MLlib
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the project**
```bash
python main.py
```

### Expected Output

The pipeline will generate:
- Trained models in `models/` directory
- Performance metrics in `output/model_results.csv`
- Comprehensive report in `output/model_report.txt`
- Visualization plots in `output/` directory

## Google Cloud Platform Setup

### Basic Setup

#### 1. Log in to your Google account

```bash
gcloud auth login
```

This will open a browser window for authentication.

#### 2. Create a project

```bash
gcloud projects create [PROJECT_NAME]
gcloud config set project [PROJECT_NAME]
```

Example:
```bash
gcloud projects create weather-forecast-spark
gcloud config set project weather-forecast-spark
```

#### 3. Link to a billing account

```bash
gcloud billing accounts list
gcloud billing projects link [PROJECT_NAME] --billing-account=[ACCOUNT_ID]
```

#### 4. Enable Google APIs

```bash
gcloud services enable dataproc.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
```

#### 5. Set permissions

```bash
gcloud iam service-accounts list
gcloud projects add-iam-policy-binding [PROJECT_NAME] \
  --member="serviceAccount:[EMAIL]" \
  --role="roles/dataproc.worker"
```

Replace `[EMAIL]` with your service account email.

### Create a Dataproc Cluster

#### 1. Update default network settings (if needed)

```bash
gcloud compute networks subnets update default \
  --region=[REGION] --enable-private-ip-google-access
```

#### 2. Create a Dataproc cluster

**Basic cluster configuration:**

```bash
gcloud dataproc clusters create [CLUSTER_NAME] \
  --region=[REGION] \
  --image-version=[IMAGE_VERSION] \
  --master-machine-type=[MASTER_MACHINE_TYPE] \
  --master-boot-disk-type=[MASTER_DISK_TYPE] \
  --master-boot-disk-size=[DISK_SIZE] \
  --num-workers=[NUMBER_OF_WORKERS] \
  --worker-machine-type=[WORKER_MACHINE_TYPE] \
  --worker-boot-disk-type=[WORKER_DISK_TYPE] \
  --worker-boot-disk-size=[DISK_SIZE] \
  --enable-component-gateway \
  --optional-components=JUPYTER
```

**Example configuration:**

```bash
gcloud dataproc clusters create weather-spark-cluster \
  --region=us-central1 \
  --image-version=2.1-debian11 \
  --master-machine-type=n1-standard-4 \
  --master-boot-disk-type=pd-standard \
  --master-boot-disk-size=50GB \
  --num-workers=2 \
  --worker-machine-type=n1-standard-4 \
  --worker-boot-disk-type=pd-standard \
  --worker-boot-disk-size=50GB \
  --enable-component-gateway \
  --optional-components=JUPYTER
```

**Machine type options:**
- `n1-standard-2`: 2 vCPUs, 7.5 GB RAM
- `n1-standard-4`: 4 vCPUs, 15 GB RAM
- `n1-standard-8`: 8 vCPUs, 30 GB RAM (recommended)
- `n1-highmem-4`: 4 vCPUs, 26 GB RAM

**Image versions:**
- `2.1-debian11` (recommended for PySpark 3.x)
- `2.0-debian10`
- `1.5-ubuntu18`

#### 3. Verify cluster creation

```bash
gcloud dataproc clusters list --region=[REGION]
```

### Run PySpark Jobs on Dataproc

#### 1. Copy local files to virtual machines

```bash
# List available compute instances
gcloud compute instances list

# Copy files to master node
gcloud compute scp [LOCAL_FILES] [VM_NAME]:[REMOTE_PATH]
```

**Example:**

```bash
# Copy entire project to master node
gcloud compute scp --recurse \
  ./Weather-Forecast-using-Spark-MLlib \
  weather-spark-cluster-m:~/
```

#### 2. SSH to virtual machines

```bash
gcloud compute ssh [VM_NAME]
```

**Example:**

```bash
gcloud compute ssh weather-spark-cluster-m
```

#### 3. Use PySpark on the cluster

Once SSH'd into the master node:

```bash
# Navigate to project directory
cd ~/Weather-Forecast-using-Spark-MLlib

# Start PySpark shell
pyspark

# Or submit the job directly
spark-submit main.py
```

**Interactive PySpark example:**

```python
# In pyspark shell
>>> df = spark.read.csv("sample.csv", header=True, inferSchema=True)
>>> df.printSchema()
>>> df.show(10)
>>> 
>>> # Import functions
>>> from pyspark.sql.functions import max, min, avg, col
>>> 
>>> # Perform analysis
>>> df.groupBy("STATION").agg(
...     max("TMP").alias("MAX_TEMP"),
...     min("TMP").alias("MIN_TEMP"),
...     avg("TMP").alias("AVG_TEMP")
... ).sort(col("AVG_TEMP").desc()).show()
>>> 
>>> quit()
```

#### 4. Submit a PySpark job via gcloud

You can also submit jobs without SSH:

```bash
gcloud dataproc jobs submit pyspark main.py \
  --cluster=[CLUSTER_NAME] \
  --region=[REGION] \
  --py-files=src/__init__.py,src/config.py,src/data_preprocessing.py,src/feature_engineering.py,src/train_model.py,src/evaluate_model.py,src/utils.py
```

**Example:**

```bash
gcloud dataproc jobs submit pyspark main.py \
  --cluster=weather-spark-cluster \
  --region=us-central1 \
  --py-files=src/__init__.py,src/config.py,src/data_preprocessing.py,src/feature_engineering.py,src/train_model.py,src/evaluate_model.py,src/utils.py
```

#### 5. Exit from the virtual machine

```bash
exit
```

### Clean Up Resources

**Important:** Delete resources when not in use to avoid charges!

```bash
# Delete the cluster
gcloud dataproc clusters delete [CLUSTER_NAME] --region=[REGION]

# Delete storage bucket
gsutil rm -r gs://[BUCKET_NAME]

# Delete the project (if needed)
gcloud projects delete [PROJECT_NAME]
```

## Configuration

Edit `src/config.py` to customize:

- **Data paths**: Input data location (local or GCS)
- **Model parameters**: Hyperparameter grids for each algorithm
- **Feature selection**: Enable/disable feature selection
- **Training settings**: Train/test split ratio, cross-validation folds
- **Output preferences**: Model saving, visualization options

**Example configurations:**

```python
# For quick testing
TRAIN_TEST_SPLIT_RATIO = 0.8
CV_FOLDS = 3
MODELS_TO_TRAIN = ['linear_regression', 'decision_tree']

# For production
TRAIN_TEST_SPLIT_RATIO = 0.8
CV_FOLDS = 5
MODELS_TO_TRAIN = ['linear_regression', 'decision_tree', 'random_forest', 'gbt']
```

## Models Implemented

The project supports multiple regression algorithms:

1. **Linear Regression**: Baseline model with regularization
2. **Decision Tree Regressor**: Non-linear tree-based model
3. **Random Forest Regressor**: Ensemble of decision trees
4. **Gradient Boosted Trees (GBT)**: Advanced boosting algorithm

Each model includes:
- Hyperparameter tuning via grid search
- Cross-validation (default: 5-fold)
- Feature importance analysis
- Comprehensive evaluation metrics

## Evaluation Metrics

Models are evaluated using:
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (R-squared): Variance explained (0-1, higher is better)
- **Training Time**: Model complexity indicator

## Visualizations

The pipeline generates:
- Model comparison bar charts (RMSE, MAE, R²)
- Feature importance plots
- Prediction vs actual scatter plots
- Residual analysis plots

## Sample Output

```
================================================================================
WEATHER FORECAST USING APACHE SPARK MLLIB
================================================================================

STEP 1: Initializing Spark Session
--------------------------------------------------------------------------------
Spark session initialized successfully

STEP 2: Loading and Preprocessing Data
--------------------------------------------------------------------------------
Data preprocessing completed

STEP 3: Feature Engineering and Selection
--------------------------------------------------------------------------------
Feature engineering completed

STEP 4: Training Models
--------------------------------------------------------------------------------
Training: Linear Regression... ✓
Training: Decision Tree... ✓
Training: Random Forest... ✓
Training: Gradient Boosted Trees... ✓

STEP 5: Evaluating Models
--------------------------------------------------------------------------------
Best Model: Random Forest
RMSE: 2.3456
R²: 0.8912

================================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY
================================================================================
```


## Additional Resources

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Google Cloud Dataproc Documentation](https://cloud.google.com/dataproc/docs)
- [NOAA Weather Data Documentation](https://www.ncei.noaa.gov/data/global-hourly/doc/)
