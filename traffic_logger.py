import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = os.environ.get("TOMTOM_API_KEY", "").strip()
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dhaka_osrm_traffic_data.csv"
FIELDS = ["Timestamp", "Day_of_Week", "Corridor_Name", "Distance_km", "Estimated_Duration_Min"]

CORRIDORS = {
    "Farmgate-Mohakhali": (23.7588, 90.3892, 23.7782, 90.4041),
    "Gabtoli-Farmgate": (23.7786, 90.3441, 23.7588, 90.3892),
    "Motijheel-Dhanmondi": (23.7331, 90.4172, 23.7578, 90.3745),
    "Mirpur10-Banani": (23.8069, 90.3687, 23.7937, 90.4066),
    "Gulistan-Uttara": (23.7233, 90.4128, 23.8759, 90.3795),
}


def make_session() -> requests.Session:
    """Retry transient TomTom/network failures without duplicating CSV rows."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_route(session: requests.Session, o_lat, o_lon, d_lat, d_lon):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
    params = {"key": API_KEY, "traffic": "true", "travelMode": "car", "routeType": "fastest"}
    response = session.get(url, params=params, timeout=(10, 45))
    response.raise_for_status()
    routes = response.json().get("routes", [])
    if not routes or "summary" not in routes[0]:
        raise ValueError("TomTom response did not contain a route summary")
    summary = routes[0]["summary"]
    return summary["lengthInMeters"] / 1000.0, summary["travelTimeInSeconds"] / 60.0


def main() -> None:
    if not API_KEY:
        sys.exit("TOMTOM_API_KEY is not set. Add it under Settings > Secrets and variables > Actions.")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    successes = 0
    failures = 0
    session = make_session()

    # Append mode is intentional: every workflow run adds one timestamped row
    # per corridor instead of replacing the historical time series.
    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(FIELDS)

        for name, (a, b, c, d) in CORRIDORS.items():
            try:
                distance, duration = get_route(session, a, b, c, d)
                row = [stamp, now.strftime("%A"), name, round(distance, 3), round(duration, 2)]
                successes += 1
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                failures += 1
                print(f"{name}: API failure ({exc.__class__.__name__}: {exc})", file=sys.stderr)
                # Preserve the timestamp/corridor so quality checks can report
                # the failed observation instead of silently losing the run.
                row = [stamp, now.strftime("%A"), name, "", ""]
            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
            time.sleep(0.2)  # avoid sending all corridor requests at once

    print(f"{stamp} logged {len(CORRIDORS)} corridors: {successes} successful, {failures} failed")
    if failures:
        print("Some rows contain blank measurements; the pipeline will exclude them during cleaning.", file=sys.stderr)
    if successes == 0:
        sys.exit("No corridor measurements succeeded; refusing to treat this run as healthy.")


if __name__ == "__main__":
    main()
