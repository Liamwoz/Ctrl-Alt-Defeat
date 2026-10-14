"""
Download all main-system (non-Jersey City) NYC Citi Bike trip data zip files.

Who worked on this: Rune Lemmens / Lex Renders / Liam Wozniak
What it does: lists every object in the public `tripdata` S3 bucket, filters out
the Jersey City ("JC-") files and the index page, and downloads the remaining
yearly (2013-2023) and monthly (2024+) zip archives into data/raw/.

This is a one-off bulk download covering the ENTIRE main-system history.
Zipped size is roughly 30-40 GB, so make sure you have enough disk space and
a stable connection before running this. The script is safe to re-run: files
that are already downloaded with the correct size are skipped, partially
downloaded files are resumed, and a single file failing (timeout/connection
drop) no longer aborts the whole run - it's retried a few times and then
skipped so the rest of the files still get downloaded.
"""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BUCKET_URL = "https://s3.amazonaws.com/tripdata/"
RAW_DIR = Path(__file__).parent / "data" / "raw"

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5
# (connect timeout, read timeout) - large files need a generous read timeout since
# it resets on every chunk received, not on total transfer time
REQUEST_TIMEOUT = (10, 120)

# S3 ListBucketResult uses this XML namespace
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_main_system_files() -> list[tuple[str, int]]:
    """Return (key, size_in_bytes) for every main-system trip data zip in the bucket."""
    response = requests.get(BUCKET_URL, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    files = []
    for contents in root.findall("s3:Contents", NS):
        key = contents.find("s3:Key", NS).text
        size = int(contents.find("s3:Size", NS).text)

        if not key.endswith(".zip"):
            continue  # skip index.html etc.
        if key.startswith("JC-"):
            continue  # Jersey City is a separate bike-share system, not NYC

        files.append((key, size))

    return sorted(files, key=lambda item: item[0])


def download_file(key: str, expected_size: int) -> bool:
    """Stream-download a single file with resume + retry. Returns True on success."""
    destination = RAW_DIR / key

    if destination.exists() and destination.stat().st_size == expected_size:
        print(f"[skip] {key} already downloaded")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        current_size = destination.stat().st_size if destination.exists() else 0
        resume_from = current_size if 0 < current_size < expected_size else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        try:
            print(
                f"[download] {key} ({expected_size / 1_048_576:,.1f} MB)"
                + (f" resuming from {resume_from / 1_048_576:,.1f} MB" if resume_from else "")
                + (f" (attempt {attempt}/{MAX_RETRIES})" if attempt > 1 else "")
            )
            with requests.get(BUCKET_URL + key, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                response.raise_for_status()
                # S3 can ignore the Range header and send the full file back (status 200
                # instead of 206); only append if it actually honored our resume request,
                # otherwise overwrite so we don't duplicate bytes.
                mode = "ab" if (resume_from and response.status_code == 206) else "wb"
                with open(destination, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)

            actual_size = destination.stat().st_size
            if actual_size == expected_size:
                return True
            print(f"[warn] {key} size mismatch after download ({actual_size:,} vs {expected_size:,} bytes), retrying")
        except requests.exceptions.RequestException as exc:
            print(f"[warn] {key} failed on attempt {attempt}/{MAX_RETRIES}: {exc}")
        except OSError as exc:
            # e.g. OneDrive/antivirus briefly locking the file right after a write
            print(f"[warn] {key} file access issue on attempt {attempt}/{MAX_RETRIES}: {exc}")

        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"[error] giving up on {key} after {MAX_RETRIES} attempts")
    return False


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = list_main_system_files()
    total_gb = sum(size for _, size in files) / 1_073_741_824
    print(f"Found {len(files)} main-system files to download (~{total_gb:,.1f} GB zipped)\n")

    failed = []
    for key, size in files:
        if not download_file(key, size):
            failed.append(key)

    print("\nDone. Raw zip files are in:", RAW_DIR)
    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded, rerun this script to retry them:")
        for key in failed:
            print(" -", key)


if __name__ == "__main__":
    main()
