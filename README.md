## AnxAI API

AnxAI is a machine-learning-powered API built with **FastAPI** and served with **Uvicorn**.
It predicts anxiety level categories from digital behavior signals.

This project is backend-only (API-first) and is ready for deployment on **Railway**.

## Project Purpose

The purpose of AnxAI is to provide a simple, documented, and testable inference service that:

- receives digital behavior features,
- runs a trained ML pipeline,
- returns an anxiety-level prediction,
- can be integrated by web/mobile clients or other backend services.

## API Endpoints

- `GET /`  
  Welcome message and quick docs pointer.

- `GET /health`  
  Service status and model availability (`model_loaded`).

- `POST /predictions`  
  Anxiety prediction based on:
  - `daily_screen_time_min`
  - `notification_count`
  - `social_media_time_min`

## Interactive Swagger Documentation

FastAPI auto-generates interactive API docs:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

Quick test flow from Swagger:

1. Open `/docs`.
2. Expand `POST /predictions`.
3. Click **Try it out**.
4. Send a sample payload.
5. Review the response and status code.

## Project Architecture

Folder structure (high-level):

```text
AnxAI/
├── config/
│   └── config.yaml              # Paths and project settings
├── data/
│   ├── raw/                     # Raw datasets (local)
│   └── processed/               # Processed datasets (local)
├── models/                      # Trained model artifacts (.pkl)
├── src/
│   ├── data/                    # Data loading/cleaning/preprocessing
│   ├── deployment/
│   │   └── app.py               # FastAPI app and endpoints
│   ├── models/                  # Train/evaluate/predict scripts
│   └── utils/                   # Shared utilities (config/logging)
├── tests/                       # API and behavior tests
├── Dockerfile                   # Container setup (Railway-compatible)
└── requirements.txt             # Python dependencies
```

## Required Resources

- Python `3.10+`
- `pip`
- A trained model file in `models/`:
  - `random_forest_anxiety_level_pipeline.pkl`
- Dependencies from `requirements.txt`

Optional but recommended:

- Virtual environment (`venv` or Conda)
- Docker (if deploying/testing in containers)

## Clone and Setup

### 1) Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AnxAI
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

This API does not require custom environment variables to run locally by default.

For deployment consistency (especially in Railway), define:

- `PORT` (provided automatically by Railway in production)
- `PYTHONPATH=.` (optional, useful for local module resolution)

PowerShell example:

```powershell
$env:PORT="8000"
$env:PYTHONPATH="."
```

## Run the API

```bash
uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/docs`

## Run Tests

```bash
python -m pytest
```

Tests cover:

- endpoint availability,
- OpenAPI/Swagger exposure,
- prediction behavior (including model-not-loaded scenarios).

## Railway Deployment Notes

You can deploy with the existing `Dockerfile` or equivalent start command:

```bash
python -m uvicorn src.deployment.app:app --host 0.0.0.0 --port $PORT
```







