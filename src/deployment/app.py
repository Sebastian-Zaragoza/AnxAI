from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.utils.config import load_config


project_root = Path(__file__).resolve().parents[2]


def load_model_pipeline() -> Optional[object]:
    config = load_config()
    if not config:
        return None

    model_path = (
        project_root
        / config["paths"]["model_dir"]
        / "random_forest_anxiety_level_pipeline.pkl"
    )
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None


model_pipeline = load_model_pipeline()

app = FastAPI(
    title="AnxAI API",
    version="1.0.0",
    description=(
        "API to estimate anxiety levels based on digital behavior patterns. "
        "Includes health and prediction endpoints for Railway deployment."
    ),
    contact={"name": "AnxAI Team"},
    openapi_tags=[
        {
            "name": "health",
            "description": "API status check and model availability validation.",
        },
        {
            "name": "predictions",
            "description": "Anxiety-level prediction from digital usage metrics.",
        },
    ],
)


class PredictionPayload(BaseModel):
    daily_screen_time_min: float = Field(
        ...,
        gt=0,
        description="Daily screen time in minutes.",
        examples=[145.0],
    )
    notification_count: int = Field(
        ...,
        ge=0,
        description="Number of notifications received per day.",
        examples=[120],
    )
    social_media_time_min: float = Field(
        ...,
        ge=0,
        description="Daily social media usage in minutes.",
        examples=[95.0],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "daily_screen_time_min": 145.0,
                "notification_count": 120,
                "social_media_time_min": 95.0,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Anxiety label returned by the model.",
        examples=["Your anxiety levels are low"],
    )


@app.get("/", tags=["health"], summary="Welcome message")
def read_root():
    return {"message": "AnxAI API is running. Visit /docs to explore endpoints."}


@app.get(
    "/health",
    tags=["health"],
    summary="API status",
    description="Checks whether the API is responding and whether the model is loaded.",
)
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "docs_url": "/docs",
    }


@app.post(
    "/predictions",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Predict anxiety level",
    description=(
        "Receives digital behavior metrics and returns a low- or high-anxiety prediction."
    ),
)
def predict_habit(payload: PredictionPayload):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Train and export the model before predicting.",
        )

    data = payload.model_dump()
    input_data = pd.DataFrame([data])
    prediction = model_pipeline.predict(input_data)[0]

    result = "Your anxiety levels are low" if prediction == 1 else "Your anxiety levels are high"
    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=8000, reload=False)