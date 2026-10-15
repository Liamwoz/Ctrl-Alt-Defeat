"""HTTP API for the trained mushroom classification and CitiBike duration pipelines."""

from __future__ import annotations

import io
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Mushroom model
# ---------------------------------------------------------------------------
MUSHROOM_MODEL_PATH = Path(
    os.getenv(
        "MUSHROOM_MODEL_PATH",
        PROJECT_ROOT / "Mushroom" / "xgboost_final.joblib",
    )
)
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10_000

mushroom_model = joblib.load(MUSHROOM_MODEL_PATH)
FEATURE_COLUMNS = list(mushroom_model.feature_names_in_)
NUMERIC_COLUMNS = ["cap-diameter", "stem-height", "stem-width"]
CATEGORICAL_COLUMNS = [
    "gill-color",
    "habitat",
    "season",
    "ring-type",
    "cap-shape",
    "stem-surface",
]

# ---------------------------------------------------------------------------
# CitiBike model
# ---------------------------------------------------------------------------
CITIBIKE_MODEL_PATH = Path(
    os.getenv(
        "CITIBIKE_MODEL_PATH",
        PROJECT_ROOT
        / "NYC_Citi_Bike"
        / "NYC Citi Bike System Data"
        / "citibike_model.json",
    )
)
CITIBIKE_FEATURES_PATH = Path(
    os.getenv(
        "CITIBIKE_FEATURES_PATH",
        PROJECT_ROOT
        / "NYC_Citi_Bike"
        / "NYC Citi Bike System Data"
        / "citibike_features.json",
    )
)

citibike_model: xgb.XGBRegressor | None = None
CITIBIKE_FEATURES: list[str] | None = None
if CITIBIKE_MODEL_PATH.is_file() and CITIBIKE_FEATURES_PATH.is_file():
    citibike_model = xgb.XGBRegressor()
    citibike_model.load_model(CITIBIKE_MODEL_PATH)
    with open(CITIBIKE_FEATURES_PATH) as f:
        CITIBIKE_FEATURES = json.load(f)

_SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # Winter
    3: 1,  4: 1, 5: 1,   # Spring
    6: 2,  7: 2, 8: 2,   # Summer
    9: 3,  10: 3, 11: 3, # Fall
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MushroomInput(BaseModel):
    """The properties accepted for one mushroom."""

    cap_diameter: float | None = Field(default=None, alias="cap-diameter")
    stem_height: float | None = Field(default=None, alias="stem-height")
    stem_width: float | None = Field(default=None, alias="stem-width")
    gill_color: str | None = Field(default=None, alias="gill-color")
    habitat: str | None = None
    season: str | None = None
    ring_type: str | None = Field(default=None, alias="ring-type")
    cap_shape: str | None = Field(default=None, alias="cap-shape")
    stem_surface: str | None = Field(default=None, alias="stem-surface")

    model_config = {"populate_by_name": True}


class CitiBikeInput(BaseModel):
    """The properties accepted for one CitiBike station-day prediction."""

    start_lat: float
    start_lng: float
    trip_date: date
    trip_count: int
    casual_trip_count: int
    member_trip_count: int
    gender_0_trip_count: int  # unknown gender
    gender_1_trip_count: int  # male
    gender_2_trip_count: int  # female
    mean_birth_year: float = Field(default=1985.0)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prediction API",
    description="Mushroom classification and CitiBike trip duration prediction.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://main.d6cw6ies6ycnl.amplifyapp.com",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mushroom helpers
# ---------------------------------------------------------------------------

def _prediction_response(features: pd.DataFrame, row_number: int | None = None) -> dict[str, Any]:
    """Run the mushroom pipeline and return a frontend-friendly prediction."""
    encoded = int(mushroom_model.predict(features)[0])
    probabilities = mushroom_model.predict_proba(features)[0]
    classes = [int(value) for value in mushroom_model.classes_]
    poisonous_probability = float(probabilities[classes.index(1)])
    response: dict[str, Any] = {
        "prediction": "poisonous" if encoded == 1 else "edible",
        "is_poisonous": encoded == 1,
        "poisonous_probability": poisonous_probability,
        "edible_probability": float(1 - poisonous_probability),
    }
    if row_number is not None:
        response["row"] = row_number
    return response


def _validate_features(data: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required columns.", "missing_columns": missing},
        )
    if data.empty:
        raise HTTPException(status_code=422, detail="The CSV must contain at least one data row.")
    if len(data) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV is limited to {MAX_ROWS} rows.")
    return data[FEATURE_COLUMNS].copy()

# ---------------------------------------------------------------------------
# Shared routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mushroom_model": MUSHROOM_MODEL_PATH.name,
        "citibike_available": citibike_model is not None,
        "citibike_model": CITIBIKE_MODEL_PATH.name if citibike_model else None,
    }

# ---------------------------------------------------------------------------
# Mushroom routes
# ---------------------------------------------------------------------------

@app.get("/mushroom/schema")
def mushroom_schema() -> dict[str, Any]:
    return {
        "features": [
            {"name": column, "type": "number" if column in NUMERIC_COLUMNS else "string"}
            for column in FEATURE_COLUMNS
        ],
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "max_csv_rows": MAX_ROWS,
    }


@app.post("/mushroom/predict")
def mushroom_predict(mushroom: MushroomInput) -> dict[str, Any]:
    values = mushroom.model_dump(by_alias=True)
    features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return _prediction_response(_validate_features(features))


@app.post("/mushroom/predict/csv")
async def mushroom_predict_csv(file: UploadFile) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    contents = await file.read(MAX_CSV_BYTES + 1)
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV file is limited to 10 MB.")
    try:
        data = _validate_features(pd.read_csv(io.BytesIO(contents)))
    except pd.errors.ParserError as error:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {error}") from error

    encoded = mushroom_model.predict(data).astype(int)
    probabilities = mushroom_model.predict_proba(data)
    classes = [int(value) for value in mushroom_model.classes_]
    poisonous_index = classes.index(1)
    predictions = [
        {
            "row": index + 1,
            "prediction": "poisonous" if value == 1 else "edible",
            "is_poisonous": bool(value == 1),
            "poisonous_probability": float(probabilities[index][poisonous_index]),
        }
        for index, value in enumerate(encoded)
    ]
    return {"count": len(predictions), "predictions": predictions}

# ---------------------------------------------------------------------------
# CitiBike routes
# ---------------------------------------------------------------------------

def _require_citibike_model() -> tuple[xgb.XGBRegressor, list[str]]:
    if citibike_model is None or CITIBIKE_FEATURES is None:
        missing = [
            str(path)
            for path in (CITIBIKE_MODEL_PATH, CITIBIKE_FEATURES_PATH)
            if not path.is_file()
        ]
        raise HTTPException(
            status_code=503,
            detail=f"Citi Bike prediction is unavailable; missing required artifact(s): {', '.join(missing)}",
        )
    return citibike_model, CITIBIKE_FEATURES


@app.get("/citibike/schema")
def citibike_schema() -> dict[str, Any]:
    _, features = _require_citibike_model()
    return {
        "features": features,
        "target": "mean_trip_duration_s",
        "description": "Predicts the mean trip duration in seconds for a CitiBike station on a given day.",
    }


@app.post("/citibike/predict")
def citibike_predict(req: CitiBikeInput) -> dict[str, Any]:
    model, features = _require_citibike_model()
    d = req.trip_date
    month = d.month
    season = _SEASON_MAP[month]
    dow = d.weekday()  # 0=Mon, 6=Sun — matches training data
    is_weekend = int(dow >= 5)
    year = d.year
    tc = max(req.trip_count, 1)

    row = {
        "start_lat": req.start_lat,
        "start_lng": req.start_lng,
        "day_of_week": dow,
        "month": month,
        "season": season,
        "is_weekend": is_weekend,
        "year": year,
        "log_trip_count": float(np.log1p(req.trip_count)),
        "casual_ratio": req.casual_trip_count / tc,
        "member_ratio": req.member_trip_count / tc,
        "gender_male_ratio": req.gender_1_trip_count / tc,
        "gender_female_ratio": req.gender_2_trip_count / tc,
        "gender_unknown_ratio": req.gender_0_trip_count / tc,
        "approx_mean_age": float(year - req.mean_birth_year),
    }

    X = np.array([[row[feature] for feature in features]])
    predicted_seconds = float(model.predict(X)[0])

    return {
        "predicted_mean_duration_seconds": round(predicted_seconds, 1),
        "predicted_mean_duration_minutes": round(predicted_seconds / 60, 2),
    }
