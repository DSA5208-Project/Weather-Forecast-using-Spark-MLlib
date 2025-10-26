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
│
├── src/                          # Source code directory
│   ├── __init__.py              # Package initializer
│   ├── config.py                # Configuration parameters
│   ├── data_preprocessing.py    # Data loading and cleaning
│   ├── train_model.py           # Model training with CV
│   ├── evaluate_model.py        # Model evaluation and visualization
│   ├── predict.py               # Prediction module for new data
│   └── utils.py                 # Utility functions
│
├── models/                       # Trained models (auto-created)
│
├── output/                       # Results and visualizations (auto-created)
│
└── docs/                         # Additional documentation
```
