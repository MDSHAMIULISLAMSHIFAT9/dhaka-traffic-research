"""
02_ingest_preprocess.py - Phase 2: fetch from GitHub, clean, categorise time slots.

    python 02_ingest_preprocess.py

Produces:
    data/processed.csv                 analysis-ready table
    tables/volatility_by_slot.csv      mean / SD / CV per corridor x day-type x slot
"""
import pandas as pd

from common import (
    DATA_SOURCE,
    PROCESSED_CSV,
    SLOT_ORDER,
    TAB_DIR,
    add_time_features,
    clean,
    load_raw,
)


def main() -> None:
    print(f"Fetching: {DATA_SOURCE}")
    raw = load_raw()
    df, rep = clean(raw)
    print(f"raw rows {rep['rows_raw']:,} -> clean rows {rep['rows_clean']:,}")

    df = add_time_features(df)

    g = df.groupby("Corridor_Name", observed=True)
    # Speed relative to each corridor's own median: removes corridor-length/road-class effects
    df["Speed_Rel"] = df["Speed_kmh"] / g["Speed_kmh"].transform("median")
    # Travel Time Index: duration / free-flow proxy (5th percentile duration of that corridor)
    df["TTI"] = df["Estimated_Duration_Min"] / g["Estimated_Duration_Min"].transform(lambda s: s.quantile(0.05))

    keep = [
        "Timestamp", "Date", "Hour", "Day_of_Week", "Time_Slot", "Corridor_Name",
        "Distance_km", "Estimated_Duration_Min", "Speed_kmh", "Speed_Rel", "TTI",
    ]
    df = df[keep].sort_values(["Timestamp", "Corridor_Name"]).reset_index(drop=True)
    df.to_csv(PROCESSED_CSV, index=False)

    print("\nObservations per time slot:")
    print(df["Time_Slot"].value_counts().reindex(SLOT_ORDER).to_string())
    print("\nObservations per day of week:")
    print(df["Day_of_Week"].value_counts().sort_index().to_string())

    # Volatility summary (coefficient of variation) - the descriptive core of the paper
    vol = (
        df.groupby(["Corridor_Name", "Day_of_Week", "Time_Slot"], observed=True)["Estimated_Duration_Min"]
        .agg(n="size", mean_min="mean", sd_min="std", median_min="median")
        .reset_index()
    )
    vol["cv_pct"] = 100 * vol["sd_min"] / vol["mean_min"]
    vol.to_csv(TAB_DIR / "volatility_by_slot.csv", index=False)

    print(f"\nSaved {PROCESSED_CSV}  ({len(df):,} rows)")
    print(f"Saved {TAB_DIR / 'volatility_by_slot.csv'}")


if __name__ == "__main__":
    main()
