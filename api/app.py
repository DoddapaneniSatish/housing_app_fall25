# api/app.py
"""
FastAPI service for Telco Customer Churn prediction.
Loads the trained classification pipeline and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/telco_churn_best_model.pkl")

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="FastAPI service for predicting customer churn (Yes/No)",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    model = joblib.load(path)
    print("✓ Model loaded successfully")
    return model


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Prediction request containing customer feature dictionaries.
    """
    instances: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "tenure": 12,
                        "monthly_charges": 70.35,
                        "total_charges": 845.5,
                        "senior_citizen": 0,
                        "gender": "Female",
                        "contract_type": "Month-to-month",
                        "payment_method": "Electronic check",
                        "internet_service": "Fiber optic",
                        "partner": "Yes",
                        "dependents": "No",
                        "phone_service": "Yes",
                        "paperless_billing": "Yes",
                        "multiple_lines": "No",
                        "online_security": "No",
                        "online_backup": "Yes",
                        "device_protection": "No",
                        "tech_support": "No",
                        "streaming_tv": "Yes",
                        "streaming_movies": "Yes",
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    churn_prediction: List[int]
    churn_probability: List[float]
    count: int


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format: {e}",
        )

    # Required feature check (matches training pipeline)
    required_columns = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "senior_citizen",
        "gender",
        "contract_type",
        "payment_method",
        "internet_service",
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return PredictResponse(
        churn_prediction=[int(p) for p in preds],
        churn_probability=[float(p) for p in probs],
        count=len(preds),
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Telco Customer Churn API - Starting Up")
    print("=" * 80)
    print(f"Model path   : {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")
