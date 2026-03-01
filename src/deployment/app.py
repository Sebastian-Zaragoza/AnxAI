from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
        "This demo exposes only the prediction endpoint through Swagger Docs."
    ),
    contact={"name": "AnxAI Team"},
    openapi_tags=[
        {
            "name": "predictions",
            "description": "Anxiety-level prediction from digital usage metrics.",
        },
    ],
)

# Restrictive CORS: rejects browser requests from all cross-origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=[],
    allow_headers=[],
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


def docs_only_guard(request: Request) -> None:
    referer = request.headers.get("referer", "")
    host = request.headers.get("host", "").lower()
    referer_lower = referer.lower()

    # Demo safeguard: allows browser calls only when triggered from Swagger Docs.
    if "/docs" not in referer_lower or (host and host not in referer_lower):
        raise HTTPException(
            status_code=403,
            detail="Forbidden. Use Swagger Docs at /docs to execute this endpoint.",
        )


@app.post(
    "/predictions",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Predict anxiety level",
    description=(
        "Receives digital behavior metrics and returns a low- or high-anxiety prediction."
    ),
)
def predict_habit(
    payload: PredictionPayload,
    _: None = Depends(docs_only_guard),
):
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