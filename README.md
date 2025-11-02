# Weather Forecast using Spark MLlib

A machine learning project to predict air temperature from weather observation data using Apache Spark MLlib.

## 📋 Project Overview

This project implements a complete machine learning pipeline for weather temperature prediction using Apache Spark. The system processes hourly surface weather observations from stations worldwide and builds predictive models using various regression algorithms.


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
│   └── utils.py                 # Utility functions
│   └── ...
│
├── models/                       # Trained models
│
├── output/                       # Results and visualizations
│
└── raw_dataset.csv
```