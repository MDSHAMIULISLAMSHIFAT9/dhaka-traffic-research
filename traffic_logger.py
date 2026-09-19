import csv, os, sys
from datetime import datetime, timezone
import requests

API_KEY = os.environ.get("TOMTOM_API_KEY", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dhaka_osrm_traffic_data.csv")
FIELDS = ["Timestamp", "Day_of_Week", "Corridor_Name", "Distance_km", "Estimated_Duration_Min"]

CORRIDORS = {
    "Farmgate-Mohakhali":  (23.7588, 90.3892, 23.7782, 90.4041),
    "Gabtoli-Farmgate":    (23.7786, 90.3441, 23.7588, 90.3892),
    "Motijheel-Dhanmondi": (23.7331, 90.4172, 23.7578, 90.3745),
    "Mirpur10-Banani":     (23.8069, 90.3687, 23.7937, 90.4066),
    "Gulistan-Uttara":     (23.7233, 90.4128, 23.8759, 90.3795),
}

def get_route(o_lat, o_lon, d_lat, d_lon):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
    params = {"key": API_KEY, "traffic": "true", "travelMode": "car", "routeType": "fastest"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        s = r.json()["routes"][0]["summary"]
        return s["lengthInMeters"] / 1000.0, s["travelTimeInSeconds"] / 60.0
    except Exception as exc:
        print(f"API failure: {exc.__class__.__name__}", file=sys.stderr)
        return None, None

def main():
    if not API_KEY:
        sys.exit("TOMTOM_API_KEY is not set.")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    new_file = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for name, (a, b, c, d) in CORRIDORS.items():
            dist, dur = get_route(a, b, c, d)
            w.writerow([stamp, now.strftime("%A"), name,
                        "" if dist is None else round(dist, 3),
                        "" if dur is None else round(dur, 2)])
    print(stamp, "logged", len(CORRIDORS), "corridors")

if __name__ == "__main__":
    main()
