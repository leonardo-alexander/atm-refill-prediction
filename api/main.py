from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import turnover

app = FastAPI(title="ATM Refill Prediction API")


class ATMInput(BaseModel):
    lag_1: float
    pct_change_1: float
    lag_7: float
    lag_28: float
    amount_per_capacity: float
    Working_day: int
    capacity: float
    dayofweek: int
    current_amount: float


# Prediction sample
# {
#   "lag_1": 900000,
#   "pct_change_1": 0.02,
#   "lag_7": 850000,
#   "lag_28": 870000,
#   "amount_per_capacity": 0.45,
#   "Working_day": 1,
#   "capacity": 20000000,
#   "dayofweek": 2,
#   "current_amount: 1000000"
# }


@app.get("/")
def home():
    return {"message": "ATM Forecast API"}


@app.post("/predict")
def predict_turnover(data: ATMInput):
    return turnover(data.dict())
