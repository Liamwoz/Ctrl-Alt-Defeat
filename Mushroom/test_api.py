import io

import pandas as pd
from fastapi.testclient import TestClient

from api import app, FEATURE_COLUMNS

client = TestClient(app)


def valid_row() -> dict[str, object]:
    return {
        "cap-diameter": 10.0,
        "stem-height": 5.0,
        "stem-width": 3.0,
        "gill-color": "white",
        "habitat": "woods",
        "season": "autumn",
        "ring-type": "smooth",
        "cap-shape": "convex",
        "stem-surface": "smooth",
    }


def test_schema_and_single_prediction() -> None:
    assert client.get("/schema").status_code == 200
    response = client.post("/predict", json=valid_row())
    assert response.status_code == 200
    assert response.json()["prediction"] in {"edible", "poisonous"}


def test_csv_prediction() -> None:
    csv = pd.DataFrame([valid_row(), valid_row()])[FEATURE_COLUMNS].to_csv(index=False)
    response = client.post(
        "/predict/csv",
        files={"file": ("mushrooms.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_csv_reports_missing_columns() -> None:
    csv = "cap-diameter\n10\n"
    response = client.post(
        "/predict/csv",
        files={"file": ("mushrooms.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert response.status_code == 422
    assert "missing_columns" in response.json()["detail"]
