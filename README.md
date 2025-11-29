# Week-5-Assignment-Neural-Networks-for-E-Commerce-Sales-Forecasting

Objective This project applies neural network fundamentals to forecast short-horizon e-commerce sales for Zenith Electronics. The goal is to generate reliable predictions to support inventory planning, marketing campaigns, and operational decisions.

Dataset

Source: Synthetic dataset created for testing
File: data/raw/sales.csv
Columns: date, sales
Timeframe: 30 consecutive days (January 2023)
Granularity: Daily
Target: Next-day sales prediction
Workflow Overview

EDA: Time-aware analysis of trends, seasonality, and anomalies

Feature Engineering:

Calendar: day-of-week, week-of-year, month, weekend flag
Lagged targets: sales_t-1, sales_t-7
Rolling stats: 7-day and 14-day rolling mean and std
Preprocessing:

Chronological split: 70% train, 15% validation, 15% test
Imputation, One-Hot encoding, scaling via scikit-learn Pipeline
Baselines:

Naïve (last value)
Ridge regression
Model:

MLPRegressor with ReLU, early stopping, and L2 regularization
Hyperparameter tuning via TimeSeriesSplit and GridSearchCV
Evaluation:

Metrics: MAE, RMSE, R²
Plots: predicted vs actual, residual distribution
Best Model Parameters

{'model__alpha': 0.0001, 'model__hidden_layer_sizes': (64,), 'model__learning_rate_init': 0.001}


**Final Test Metrics**
- MAE: ~169.38
- RMSE: ~169.39
- R²: (value under review due to scaling issue)


**Project Structure**

cst600-week05-sales-nn-pavan/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_load.py
│   ├── features.py
│   ├── model_mlp.py
│   ├── evaluate.py
│   └── main.py
├── data/
│   ├── raw/sales.csv
│   └── processed/
└── figures/
    ├── pred_vs_actual.png
    └── residuals.png


**Risks & Limitations**
- Small synthetic dataset limits generalization
- No promo/holiday signals
- Non-stationarity and cold-start risks


**Next Steps**
- Add holiday calendars and promo flags
- Expand lag windows and rolling features
- Explore probabilistic forecasting and ensemble models

**References**
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice (2nd ed.). OTexts.
- Kaggle Datasets. (n.d.). Retrieved from https://www.kaggle.com/datasets
