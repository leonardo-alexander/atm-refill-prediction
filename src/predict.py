import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("models/rf_model.pkl")

model = bundle["model"]
features = bundle["features"]


def predict(data: dict):
    df = pd.DataFrame([data])

    if "dayofweek" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    pred_log = model.predict(df)

    return float(np.expm1(pred_log)[0])


def turnover(data: dict):
    df = pd.DataFrame([data])

    cash = df["current_amount"].iloc[0]
    predictions = []
    amount_per_day = []
    day = 0

    while cash > 0 and day < 30:
        pred = predict(df.iloc[0].to_dict())
        predictions.append(pred)
        cash -= pred
        amount_per_day.append(cash)

        if cash <= 0:
            break

        df["lag_1"] = pred
        df["amount_per_capacity"] = pred / df["capacity"]
        df["dayofweek"] = (df["dayofweek"] + 1) % 7
        df["Working_day"] = (df["dayofweek"] < 5).astype(int)

        if day >= 6:
            df["lag_7"] = predictions[day - 6]

        if day >= 27:
            df["lag_28"] = predictions[day - 27]

        day += 1

    return {
        "prediction_per_day": predictions,
        "amount_per_day": amount_per_day,
        "day_left_to_fill": day,
    }
