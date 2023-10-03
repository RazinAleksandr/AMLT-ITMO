# AMLT-ITMO

This repository contains the work and files related to labs by subject Advanced Machine Learning Technologies.

Labs list:
- Laboratory 1: Time series forecasting using ARIMA and LSTM models. (ts_ prefix)

## Project Structure
```
.
├── data
│   ├── interim: Contains the intermediate processed data
│   ├── metrics: Contains metrics related to models
│   ├── models: Trained models
│   ├── predicted: Models predictions 
│   ├── preprocessed: Processed data for modeling
│   ├── raw: Raw datasets and their DVC file
│   └── scalers: Contains scalers used in preprocessing
├── dvc.lock & dvc.yaml: DVC files to manage and version large datasets and ML models
├── figures: Visualizations and plots related to model comparisons
├── models: ML models for labs
├── notebooks: Jupyter notebooks data analysis, result analysis
├── params.yaml: Configuration and hyperparameter file
├── src: Source code of the project including data processing, model training, forecasting, and utilities
│   ├── data
│   │   ├── datasets.py      - Responsible for data loading and dataset creation.
│   │   └── __init__.py
│   ├── features              - Contains scripts related to feature engineering and transformation.
│   │   └── __init__.py
│   ├── models
│   │   ├── ts_forecast.py      - Script used for forecasting using trained models.
│   │   ├── train.py         - Contains the ts training routines for the models.
│   │   └── __init__.py
│   ├── scripts
│   │   ├── ts_arima_train.py- Training script for the ARIMA model.
│   │   ├── ts_preproc.py    - Script for preprocessing the time series data.
│   │   └── ts_train.py      - Training script for time series models.
│   └── utils
│       ├── base_helpers.py  - Basic helper functions used throughout the project.
│       ├── base_transforms.py - Basic transformations applied to the data.
│       ├── ts_helpers.py    - Helper functions tailored for time series processing.
│       ├── ts_metrics.py    - Metrics for evaluating time series model performance.
│       ├── ts_plots.py      - Script for generating time series visualizations.
│       ├── ts_transforms.py - Transformations specific to time series data.
│       └── __init__.py
└── setup.py: Setup file for the repository

```
