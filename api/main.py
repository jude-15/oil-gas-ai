import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# =========================================================
# DATABASE
# =========================================================

from database.db_connection import engine
from database.queries import WELL_INTELLIGENCE_QUERY


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Oil & Gas AI API",
    description="Industrial AI API for well monitoring and failure analysis",
    version="1.0.0"
)


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

MODEL_PATH = BASE_DIR / "models" / "failure_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    MODEL_LOADED = True

except Exception as e:
    model = None
    scaler = None

    MODEL_LOADED = False

    print("Warning: ML model could not be loaded.")
    print(e)


# =========================================================
# INPUT DATA MODEL
# =========================================================

class SensorData(BaseModel):

    pressure_bar: float

    temperature_c: float

    vibration_mm_s: float

    flow_rate_bbl_day: float

    production_rate_bbl_day: float

    pump_speed_rpm: float


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Oil & Gas AI API is running",
        "status": "online"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "api": "online",
        "model_loaded": MODEL_LOADED
    }


# =========================================================
# GET WELLS
# =========================================================

@app.get("/wells")
def get_wells():

    df = pd.read_sql(
        WELL_INTELLIGENCE_QUERY,
        engine
    )

    return df.to_dict(
        orient="records"
    )


# =========================================================
# AI FAILURE PREDICTION
# =========================================================

@app.post("/predict")
def predict_failure(data: SensorData):

    if not MODEL_LOADED:

        return {
            "error": "ML model is not loaded"
        }

    # -----------------------------------------------------
    # Prepare sensor features
    # -----------------------------------------------------

    features = np.array([[
        data.pressure_bar,
        data.temperature_c,
        data.vibration_mm_s,
        data.flow_rate_bbl_day,
        data.production_rate_bbl_day,
        data.pump_speed_rpm
    ]])

    # -----------------------------------------------------
    # Apply the same scaler used during training
    # -----------------------------------------------------

    features_scaled = scaler.transform(
        features
    )

    # -----------------------------------------------------
    # Predict failure probability
    # -----------------------------------------------------

    probability = model.predict_proba(
        features_scaled
    )[0][1]

    # -----------------------------------------------------
    # Risk classification
    # -----------------------------------------------------

    if probability >= 0.80:

        risk_level = "HIGH"

    elif probability >= 0.40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # -----------------------------------------------------
    # Prediction result
    # -----------------------------------------------------

    return {

        "failure_probability": round(
            float(probability),
            4
        ),

        "failure_probability_percent": round(
            float(probability * 100),
            2
        ),

        "risk_level": risk_level

    }