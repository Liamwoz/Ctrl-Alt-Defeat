Ctrl-Alt-Defeat

Rune Lemmens
Lex Renders
Liam Wozniak

## Mushroom API

The backend in `Mushroom/api.py` serves the saved XGBoost pipeline through FastAPI.
Install its dependencies from `Mushroom/requirements-api.txt`, then start it from
the repository root:

```powershell
pip install -r Mushroom/requirements-api.txt
uvicorn Mushroom.api:app --reload
```

The API provides `GET /health`, `GET /schema`, `POST /predict` for one mushroom,
and `POST /predict/csv` for a CSV upload. CSV files must contain the nine feature
columns returned by `/schema`; additional columns are ignored.