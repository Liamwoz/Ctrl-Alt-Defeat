Ctrl-Alt-Defeat

Pamela Barbosa
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
`NYC_Citi_Bike` by default. Override their locations
with `CITIBIKE_MODEL_PATH` and `CITIBIKE_FEATURES_PATH` if needed.

## Frontend

Run `main.py` as above, then serve the project root in another terminal:

```powershell
python -m http.server 3000
```

Open `http://localhost:3000`. The Citi Bike station picker uses Citi Bike's
public GBFS station-information feed for live station names and coordinates,
and OpenStreetMap for map tiles. An internet connection is required for the
station feed, tiles, and Leaflet map library.