import json
import os
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fundfirst_logistic_regression.joblib"
METADATA_PATH = MODEL_DIR / "fundfirst_model_metadata.json"

model = joblib.load(MODEL_PATH)
with METADATA_PATH.open(encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

FEATURES = metadata["features"]


def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="FundFirst API",
    version="1.0.0",
    description="Prediction API for the FundFirst Logistic Regression pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


class PredictionInput(BaseModel):
    AveragePrice: Annotated[FiniteNumber, Field(gt=0)]
    MedianAnnualPay: Annotated[FiniteNumber, Field(gt=0)]
    SavingRatio: FiniteNumber
    BaseRate: FiniteNumber


class PredictionOutput(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "FundFirst API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "features": FEATURES,
        "classes": [str(label) for label in model.classes_],
    }


@app.get("/metadata")
def model_metadata() -> dict[str, object]:
    return metadata


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput) -> PredictionOutput:
    frame = pd.DataFrame(
        [{feature: getattr(data, feature) for feature in FEATURES}],
        columns=FEATURES,
    )

    prediction = str(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]

    return PredictionOutput(
        prediction=prediction,
        probabilities={
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    )
