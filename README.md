Ctrl-Alt-Defeat

Rune Lemmens
Lex Renders
Liam Wozniak

## Mushroom API

The backend in `Mushroom/api.py` serves the saved XGBoost pipeline through FastAPI.
Install the project dependencies from the repository root, then start it:

```powershell
pip install -r requirements.txt
uvicorn Mushroom.api:app --reload
```

The API provides `GET /health`, `GET /schema`, `POST /predict` for one mushroom,
and `POST /predict/csv` for a CSV upload. CSV files must contain the nine feature
columns returned by `/schema`; additional columns are ignored.

## Combined Mushroom and Citi Bike API

The root-level `main.py` serves both prediction APIs. Start it from the
repository root:

```powershell
pip install -r requirements.txt
uvicorn main:app --reload
```

Mushroom endpoints are available at `/mushroom/schema`, `/mushroom/predict`,
and `/mushroom/predict/csv`. The Citi Bike endpoints are `/citibike/schema`
and `/citibike/predict`. The mushroom model is loaded from
`Mushroom/xgboost_final.joblib` by default. Citi Bike predictions use
`citibike_model.json` and `citibike_features.json` from
`NYC_Citi_Bike/NYC Citi Bike System Data` by default. Override their locations
with `CITIBIKE_MODEL_PATH` and `CITIBIKE_FEATURES_PATH` if needed.