from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
GITHUB_USER = "MDSHAMIULISLAMSHIFAT9"
REPO = "dhaka-traffic-research"
BRANCH = "main"
CSV_NAME = "dhaka_osrm_traffic_data.csv"

# Time zone in which osrm_logger.py wrote the Timestamp column.
# GitHub Actions runners use UTC, so "UTC" is correct unless your script
# explicitly wrote Dhaka time (then set "Asia/Dhaka").
LOGGER_TZ = "UTC"
LOCAL_TZ = "Asia/Dhaka"                        # UTC+6, no DST

# You can point the pipeline at a local file instead of GitHub:
#   export DATA_SOURCE=/path/to/dhaka_osrm_traffic_data.csv
DATA_SOURCE = os.environ.get("DATA_SOURCE") or (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/{CSV_NAME}"
)

# Cleaning thresholds
MAX_SPEED_KMH = 120.0     # physically implausible for urban Dhaka above this
DIST_TOL = 0.25           # drop rows whose distance deviates >25% from corridor median
EXPECTED_PER_DAY = 96     # 24 h * 4 snapshots/h per corridor

# Study design constants
DAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SLOT_ORDER = ["Morning Peak", "Midday", "Evening Peak", "Night"]
FOCUS_DAYS = ["Thursday", "Friday", "Sunday"]

# --------------------------------------------------------------------------
# 2. OUTPUT FOLDERS
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
GEN_DIR = ROOT / "paper" / "generated"
for _d in (DATA_DIR, FIG_DIR, TAB_DIR, GEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PROCESSED_CSV = DATA_DIR / "processed.csv"

# --------------------------------------------------------------------------
# 3. LOADING
# --------------------------------------------------------------------------
REQUIRED = ["Timestamp", "Day_of_Week", "Corridor_Name", "Distance_km", "Estimated_Duration_Min"]
_NA_TOKENS = {"", "none", "null", "nan", "na", "n/a", "error", "-"}


def load_raw(source: str | None = None) -> pd.DataFrame:
    """Read the raw CSV (GitHub raw URL or local path) as strings."""
    source = source or DATA_SOURCE
    if source.startswith("http"):
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")          # only needed for private repos
        if token:
            headers["Authorization"] = f"token {token}"
        resp = requests.get(source, headers=headers, timeout=60)
        resp.raise_for_status()
        buf = io.StringIO(resp.text)
    else:
        buf = source
    df = pd.read_csv(buf, dtype=str, on_bad_lines="skip", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def _blank_to_na(s: pd.Series) -> pd.Series:
    """Strip whitespace; map '', 'None', 'ERROR', 'NaN', ... to missing (None)."""
    def fix(v):
        if v is None or pd.isna(v):
            return None
        v = str(v).strip()
        return None if v.lower() in _NA_TOKENS else v
    return s.astype(object).map(fix)


# --------------------------------------------------------------------------
# 4. CLEANING  (Phase 1)
# --------------------------------------------------------------------------
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns (clean_df, report). Rules, applied in this order:
      1. drop repeated header rows (logger appended the header again)
      2. blank / 'None' / 'ERROR' API responses -> NaN, then drop incomplete rows
      3. drop zero or negative durations / distances
      4. drop exact duplicate (Timestamp, Corridor_Name) rows
      5. convert to Asia/Dhaka and RECOMPUTE Day_of_Week from local time
      6. drop rows whose distance deviates > DIST_TOL from the corridor median
      7. compute Speed_kmh = Distance_km / (Estimated_Duration_Min / 60)
      8. drop physically impossible speeds (> MAX_SPEED_KMH)
    Tail values are NOT trimmed: volatility is the object of study.
    """
    rep: dict = {"rows_raw": int(len(raw))}
    df = raw[REQUIRED].copy()
    for c in REQUIRED:
        df[c] = _blank_to_na(df[c])

    hdr = df["Timestamp"].eq("Timestamp")
    rep["repeated_header_rows"] = int(hdr.sum())
    df = df[~hdr].copy()

    for c in ("Distance_km", "Estimated_Duration_Min"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(LOGGER_TZ, ambiguous="NaT", nonexistent="NaT")
    except (TypeError, AttributeError, ValueError):
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df["Timestamp"] = ts.dt.tz_convert(LOCAL_TZ)

    need = ["Timestamp", "Corridor_Name", "Distance_km", "Estimated_Duration_Min"]
    m = df[need].isna().any(axis=1)
    rep["missing_or_unparseable_rows"] = int(m.sum())
    df = df[~m].copy()

    m = (df["Distance_km"] <= 0) | (df["Estimated_Duration_Min"] <= 0)
    rep["zero_or_negative_rows"] = int(m.sum())
    df = df[~m].copy()

    m = df.duplicated(["Timestamp", "Corridor_Name"], keep="first")
    rep["duplicate_rows"] = int(m.sum())
    df = df[~m].copy()

    csv_day = df["Day_of_Week"].fillna("").str.strip().str.title()
    df["Day_of_Week"] = df["Timestamp"].dt.day_name()
    rep["day_label_mismatch_rows"] = int((csv_day != df["Day_of_Week"]).sum())

    med = df.groupby("Corridor_Name")["Distance_km"].transform("median")
    m = (df["Distance_km"] - med).abs() / med > DIST_TOL
    rep["distance_inconsistent_rows"] = int(m.sum())
    df = df[~m].copy()

    df["Speed_kmh"] = df["Distance_km"] / (df["Estimated_Duration_Min"] / 60.0)
    m = df["Speed_kmh"] > MAX_SPEED_KMH
    rep["impossible_speed_rows"] = int(m.sum())
    df = df[~m].copy()

    df = df.sort_values(["Corridor_Name", "Timestamp"]).reset_index(drop=True)
    rep["rows_clean"] = int(len(df))
    return df, rep


# --------------------------------------------------------------------------
# 5. TIME FEATURES  (Phase 2)
# --------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Morning Peak 07:00-09:59 | Midday 10:00-15:59 | Evening Peak 16:00-19:59 | Night 20:00-06:59
    (local Dhaka time, left-closed hourly bins)
    """
    d = df.copy()
    d["Date"] = d["Timestamp"].dt.strftime("%Y-%m-%d")
    d["Hour"] = d["Timestamp"].dt.hour
    h = d["Hour"]
    slot = np.select(
        [h.between(7, 9), h.between(10, 15), h.between(16, 19)],
        ["Morning Peak", "Midday", "Evening Peak"],
        default="Night",
    )
    d["Time_Slot"] = pd.Categorical(slot, categories=SLOT_ORDER, ordered=True)
    d["Day_of_Week"] = pd.Categorical(d["Day_of_Week"], categories=DAY_ORDER, ordered=True)
    return d


def load_processed() -> pd.DataFrame:
    """Read data/processed.csv written by 02_ingest_preprocess.py."""
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Run 02_ingest_preprocess.py first (data/processed.csv not found).")
    d = pd.read_csv(PROCESSED_CSV)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], utc=True).dt.tz_convert(LOCAL_TZ)
    d["Day_of_Week"] = pd.Categorical(d["Day_of_Week"], categories=DAY_ORDER, ordered=True)
    d["Time_Slot"] = pd.Categorical(d["Time_Slot"], categories=SLOT_ORDER, ordered=True)
    return d


def corridor_ids(df: pd.DataFrame) -> dict:
    """Map corridor name -> short ID (C1 = fastest mean speed, C2 = next, ...).
    Used for compact table/figure labels; the paper's corridor table gives the names."""
    order = df.groupby("Corridor_Name")["Speed_kmh"].mean().sort_values(ascending=False).index
    return {name: f"C{i + 1}" for i, name in enumerate(order)}
