from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "xgb_model_v2.pkl")

app = FastAPI()

# Load model
model = joblib.load(model_path)


# Define input schema
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.Time,
        transaction.Amount,
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28
    ]])

    proba = model.predict_proba(data)[0][1]
    prediction = int(proba >= 0.3)  # your tuned threshold

    return {
        "fraud_probability": float(proba),
        "is_fraud": prediction
    }