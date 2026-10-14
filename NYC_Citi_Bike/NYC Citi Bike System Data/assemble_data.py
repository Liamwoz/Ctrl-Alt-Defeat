"""
Assemble all downloaded Citi Bike zip files (data/raw/) into one combined CSV.

Who worked on this: Rune Lemmens / Lex Renders / Liam Wozniak
What it does: unzips every archive in data/raw/ in memory, maps each CSV's
columns onto one unified schema (the Citi Bike export format changed in
Feb 2021 - see notes below), and streams the result into one combined CSV at
data/processed/citibike_combined.csv without ever loading the full dataset
into memory at once.

Known quirks of the raw files handled here:
- Yearly zips (2013-2023) contain junk (__MACOSX/, .DS_Store, empty folders)
  and, for several months, the SAME month twice: once as a single flat CSV
  and once split into chunked "_1.csv"/"_2.csv" files inside a subfolder.
  We keep only one copy per month (the flat file when available) to avoid
  double-counting trips.
- Monthly zips (2024+) are only split into part1..partN with no duplicate,
  so all parts for those are used.
- Pre-2021 files use the "classic" schema (starttime, usertype, birth year,
  gender, ...); 2021+ files use the "new" schema (started_at, rideable_type,
  member_casual, ...) and drop birth year/gender entirely. Both are mapped
  onto one unified schema below; columns that don't exist for a given era
  are left as NaN (this is a structural/MNAR gap, not random missingness -
  worth calling out explicitly in the cleaning notebook).
"""

import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).parent / "data" / "processed" / "citibike_combined.csv"
PROGRESS_LOG = OUTPUT_PATH.parent / ".assembled_zips.txt"

CHUNK_SIZE = 200_000

UNIFIED_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lng",
    "end_station_id",
    "end_station_name",
    "end_lat",
    "end_lng",
    "member_casual",
    "bike_id",
    "birth_year",
    "gender",
    "trip_duration_s",
]

OLD_SCHEMA_RENAME = {
    "starttime": "started_at",
    "stoptime": "ended_at",
    "start_station_id": "start_station_id",
    "start_station_name": "start_station_name",
    "start_station_latitude": "start_lat",
    "start_station_longitude": "start_lng",
    "end_station_id": "end_station_id",
    "end_station_name": "end_station_name",
    "end_station_latitude": "end_lat",
    "end_station_longitude": "end_lng",
    "bikeid": "bike_id",
    "birth_year": "birth_year",
    "gender": "gender",
}

USERTYPE_TO_MEMBER_CASUAL = {"subscriber": "member", "customer": "casual"}

MONTH_KEY_RE = re.compile(r"(\d{6})-citibike-tripdata")


def normalize_columns(columns: list[str]) -> list[str]:
    return [c.strip().lower().replace(" ", "_") for c in columns]


def is_junk_member(name: str) -> bool:
    base = Path(name).name
    return (
        "__MACOSX" in name
        or base.startswith("._")
        or base == ".DS_Store"
        or not base.lower().endswith(".csv")
    )


def select_members_for_zip(zf: ZipFile) -> list[str]:
    """Pick one set of CSV members per month, preferring the flat (non-split) file."""
    members = [n for n in zf.namelist() if not is_junk_member(n)]

    months: dict[str, list[str]] = {}
    for name in members:
        match = MONTH_KEY_RE.search(Path(name).name)
        key = match.group(1) if match else Path(name).name  # fallback: treat as its own group
        months.setdefault(key, []).append(name)

    chosen: list[str] = []
    for month_key, names in months.items():
        flat_candidates = [n for n in names if re.fullmatch(rf"{month_key}-citibike-tripdata\.csv", Path(n).name)]
        if flat_candidates:
            chosen.append(flat_candidates[0])  # ignore duplicate split "_1.csv"/"_2.csv" copies
        else:
            chosen.extend(names)  # no flat copy exists, so the split parts are the only copy

    return chosen


def map_chunk_to_unified(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = normalize_columns(list(chunk.columns))

    if "ride_id" in chunk.columns:
        # 2021+ schema: already close to unified, bike_id/birth_year/gender don't exist
        mapped = chunk.copy()
    elif "starttime" in chunk.columns:
        # pre-2021 schema
        mapped = chunk.rename(columns=OLD_SCHEMA_RENAME)
        mapped["member_casual"] = (
            mapped.get("usertype", pd.Series(dtype="object")).astype(str).str.lower().map(USERTYPE_TO_MEMBER_CASUAL)
        )
    else:
        raise ValueError(f"Unrecognized schema with columns: {list(chunk.columns)}")

    for col in UNIFIED_COLUMNS:
        if col not in mapped.columns:
            mapped[col] = pd.NA

    mapped["start_station_id"] = mapped["start_station_id"].astype(str)
    mapped["end_station_id"] = mapped["end_station_id"].astype(str)

    started = pd.to_datetime(mapped["started_at"], errors="coerce")
    ended = pd.to_datetime(mapped["ended_at"], errors="coerce")
    mapped["trip_duration_s"] = (ended - started).dt.total_seconds()

    return mapped[UNIFIED_COLUMNS]


def append_to_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, mode="a", header=not OUTPUT_PATH.exists(), index=False)


def load_progress() -> set[str]:
    if PROGRESS_LOG.exists():
        return set(PROGRESS_LOG.read_text().splitlines())
    return set()


def mark_done(zip_name: str) -> None:
    with open(PROGRESS_LOG, "a") as f:
        f.write(zip_name + "\n")


def main() -> None:
    zip_paths = sorted(RAW_DIR.glob("*.zip"))
    if not zip_paths:
        print(f"No zip files found in {RAW_DIR}. Run download_data.py first.")
        return

    done = load_progress()
    total_rows = 0

    for zip_path in zip_paths:
        if zip_path.name in done:
            print(f"[skip] {zip_path.name} already assembled")
            continue

        print(f"[processing] {zip_path.name}")
        with ZipFile(zip_path) as zf:
            members = select_members_for_zip(zf)
            for member in members:
                with zf.open(member) as f:
                    for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
                        unified = map_chunk_to_unified(chunk)
                        append_to_output(unified)
                        total_rows += len(unified)
                print(f"    + {member} ({total_rows:,} rows written so far)")

        mark_done(zip_path.name)

    print(f"\nDone. Combined CSV: {OUTPUT_PATH} ({total_rows:,} rows written this run)")


if __name__ == "__main__":
    main()
