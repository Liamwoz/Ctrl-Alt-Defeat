# Ctrl-Alt-Defeat

**Group members:** Pamela Barbosa, Rune Lemmens, Lex Renders, Liam Wozniak

This repository contains two machine-learning projects: mushroom edibility
classification and Citi Bike station-day trip-duration prediction. The primary
inference API is the combined FastAPI application in `main.py`; the static web
interface is in `Frontend/`.

## Requirements

- Python 3.11
- Internet access for installing packages and for the frontend's Citi Bike
  station search/map (GBFS, Leaflet, and OpenStreetMap)

Python 3.11 is required for the checked-in PyCaret 3.3.2 and
scikit-learn 1.4.2 dependencies and the saved mushroom pipeline. Python 3.13
is not supported by this dependency set.

## Setup

Run these commands from the repository root. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the API

From the repository root, with the virtual environment active:

```sh
python -m uvicorn main:app --reload
```

The API listens at `http://127.0.0.1:8000`. Keep this terminal running. The
interactive API reference is at <http://127.0.0.1:8000/docs>; the alternate
reference is at <http://127.0.0.1:8000/redoc>.

The mushroom model is loaded from `Mushroom/xgboost_final.joblib`. The Citi Bike
model and its feature list are loaded from
`NYC_Citi_Bike/citibike_model.json` and
`NYC_Citi_Bike/citibike_features.json`. The Citi Bike paths can be overridden
with `CITIBIKE_MODEL_PATH` and `CITIBIKE_FEATURES_PATH`; the mushroom path can
be overridden with `MUSHROOM_MODEL_PATH`.

### Shared endpoint

- `GET /health` — returns `"status": "ok"`, the loaded mushroom model filename,
  and whether the Citi Bike model is available.

### Mushroom endpoints

- `GET /mushroom/schema` — lists the model's nine required features, feature
  types, numeric and categorical columns, and the CSV row limit.
- `POST /mushroom/predict` — classifies one mushroom. Send a JSON object with
  these nine fields:

  | Field | Type |
  | --- | --- |
  | `cap-diameter` | number |
  | `stem-height` | number |
  | `stem-width` | number |
  | `gill-color` | string |
  | `habitat` | string |
  | `season` | string |
  | `ring-type` | string |
  | `cap-shape` | string |
  | `stem-surface` | string |

### Citi Bike endpoints

- `GET /citibike/schema` — lists the features expected by the trained model,
  the prediction target (`mean_trip_duration_s`), and a description. It returns
  HTTP 503 if the Citi Bike model artifacts are unavailable.
- `POST /citibike/predict` — predicts the mean trip duration in seconds for a
  station on a given date. Send a JSON object with:

  | Field | Type | Meaning |
  | --- | --- | --- |
  | `start_lat` | number | Station latitude |
  | `start_lng` | number | Station longitude |
  | `trip_date` | string | Date in `YYYY-MM-DD` format |
  | `trip_count` | integer | Total trips |
  | `casual_trip_count` | integer | Casual rider trips |
  | `member_trip_count` | integer | Member rider trips |
  | `gender_0_trip_count` | integer | Trips with unknown gender |
  | `gender_1_trip_count` | integer | Trips by male riders |
  | `gender_2_trip_count` | integer | Trips by female riders |
  | `mean_birth_year` | number | Mean birth year for trips with recorded birth year |

  `mean_birth_year` is optional and defaults to `1985.0`. The response contains
  `predicted_mean_duration_seconds` and
  `predicted_mean_duration_minutes`. The frontend asks for average rider age
  and converts it to a mean birth year using the selected trip year before
  calling this endpoint.


## Start the frontend

Keep the API running in its terminal. Open a **second** PowerShell terminal,
change to the repository root (the folder containing `main.py`), and start the
static file server:

```powershell
.\.venv311\Scripts\python.exe -m http.server 3000 --directory Frontend
```

Leave this terminal running and open <http://localhost:3000> in your browser.
The frontend automatically calls the local API when opened on `localhost` or
`127.0.0.1`; elsewhere it uses the configured hosted API. Keep both the API
and frontend terminals running while using the local site.

The station picker requires internet access for the live Citi Bike GBFS station feed, Leaflet, and OpenStreetMap tiles.

## Hosting

The API is hosted on an AWS EC2 server redirected to https through Cloudfront: https://d16mi61bdk4mg3.cloudfront.net // https://d16mi61bdk4mg3.cloudfront.net/docs

Our frontend is hosted on AWS Amplify available with this URL: https://main.d6cw6ies6ycnl.amplifyapp.com