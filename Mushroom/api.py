"""HTTP API for the trained mushroom classification pipeline."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MUSHROOM_MODEL_PATH", PROJECT_ROOT / "xgboost_final.joblib")
)
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10_000

model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = list(model.feature_names_in_)
NUMERIC_COLUMNS = ["cap-diameter", "stem-height", "stem-width"]
CATEGORICAL_COLUMNS = [
    "gill-color",
    "habitat",
    "season",
    "ring-type",
    "cap-shape",
    "stem-surface",
]


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


app = FastAPI(
    title="Mushroom Prediction API",
    description="Predicts whether mushrooms are edible or poisonous.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _prediction_response(features: pd.DataFrame, row_number: int | None = None) -> dict[str, Any]:
    """Run the saved pipeline and return a frontend-friendly prediction."""
    encoded = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = [int(value) for value in model.classes_]
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH.name}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "features": [
            {"name": column, "type": "number" if column in NUMERIC_COLUMNS else "string"}
            for column in FEATURE_COLUMNS
        ],
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "max_csv_rows": MAX_ROWS,
    }


@app.post("/predict")
def predict(mushroom: MushroomInput) -> dict[str, Any]:
    values = mushroom.model_dump(by_alias=True)
    features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return _prediction_response(_validate_features(features))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    contents = await file.read(MAX_CSV_BYTES + 1)
    if len(contents) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV file is limited to 10 MB.")
    try:
        data = _validate_features(pd.read_csv(io.BytesIO(contents)))
    except pd.errors.ParserError as error:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {error}") from error

    encoded = model.predict(data).astype(int)
    probabilities = model.predict_proba(data)
    classes = [int(value) for value in model.classes_]
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
