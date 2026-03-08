# ATM Refill Prediction
Machine learning system for recursive forecasting ATM withdrawals and estimating
when an ATM will run out of cash.

The system uses historical ATM transaction data from Kaggle ATM Transactions

Dataset source:
https://www.kaggle.com/datasets/andrewgeorgeee/atm-transactions?resource=download

---

## Overview

ATM operators must ensure machines do not run out of cash while minimizing unnecessary refilling.

So, this project wants to:
- Predict daily ATM withdrawal
- Simulate cash depletion
- Estimate number of days until refill is required

The system works by predicting next-day withdrawal and recursively simulate the days after.

---

## Features

The final model (retrained after feature importance) uses time-series and behavioral features such as:
- Lagged withdrawals (lag_1, lag_7, lag_28)
- Momentum (pct_change_1)
- ATM capacity usage (amount_per_capacity)
- Calendar effects (Working_day, dayofweek)
- Cyclical encoding (dow_sin, dow_cos)

## Model and Evaluation

Models evaluated:
- Linear Regression
- Random Forest
- XGBoost

Final model: Random Forest (Tuned)
Best parameters:
- n_estimators: 600
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: 0.5
- max_depth: 20

Evaluation results:
- MAE : 5,875.65
- RMSE: 35,542.56
- R²  : 0.9716

## Deployment

The model is deployed using FastAPI

Endpoint:
POST /predict endpoint

Example input:
```json
{
  "lag_1": 900000,
  "pct_change_1": 0.02,
  "lag_7": 850000,
  "lag_28": 870000,
  "amount_per_capacity": 0.45,
  "Working_day": 1,
  "capacity": 2000000,
  "dayofweek": 2,
  "current_amount": 1000000
}
```

Example output:
```json
{
  "prediction_per_day": [
    676761.6335661152,
    205619.44257847941,
    103798.68652929281,
    62603.57835376092
  ],
  "amount_per_day": [
    323238.36643388483,
    117618.92385540542,
    13820.237326112605,
    -48783.34102764831
  ],
  "day_left_to_fill": 3
}
```

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- FastAPI

## Project Structure
```
atm-refill-prediction
│
├── data/
│   └── atms_data.csv
│
├── models/
│   └── rf_model.pkl
│
├── src/
│   └── predict.py
│
├── api/
│   └── main.py
│
├── notebooks/
│   └── training.ipynb
│
└── README.md
```
