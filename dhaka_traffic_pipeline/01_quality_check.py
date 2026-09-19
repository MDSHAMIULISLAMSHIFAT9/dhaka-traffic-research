"""
01_quality_check.py - Phase 1: data monitoring & quality control.

Run any time during the 7-14 day collection window:
    python 01_quality_check.py
It pulls the CSV from GitHub, cleans it, and prints a health report.
"""
import pandas as pd

from common import DATA_DIR, EXPECTED_PER_DAY, clean, load_raw

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


def main() -> None:
    raw = load_raw()
    df, rep = clean(raw)

    print("=" * 70)
    print("1. CLEANING REPORT")
    print("=" * 70)
    for k, v in rep.items():
        print(f"  {k:32s} {v:>10,}")
    raw_n = max(rep["rows_raw"], 1)
    print(f"  {'missing/API-failure rate':32s} {100 * rep['missing_or_unparseable_rows'] / raw_n:>9.2f}%")
    if rep["day_label_mismatch_rows"] > 0.02 * max(rep["rows_clean"], 1):
        print("  !! Day_of_Week in the CSV disagrees with Asia/Dhaka local time for >2% of rows.")
        print("     Likely cause: logger wrote UTC weekday. The pipeline recomputes it, so you are safe.")

    print("\n" + "=" * 70)
    print("2. COVERAGE")
    print("=" * 70)
    if df.empty:
        raise SystemExit("No valid rows survived cleaning - inspect the raw CSV.")
    d0, d1 = df["Timestamp"].min(), df["Timestamp"].max()
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    print(f"  first snapshot : {d0}")
    print(f"  last snapshot  : {d1}")
    print(f"  calendar days  : {df['Date'].nunique()}")
    print(f"  corridors      : {df['Corridor_Name'].nunique()}")

    per = df.groupby(["Corridor_Name", "Date"]).size().rename("n").reset_index()
    per["coverage_pct"] = 100 * per["n"] / EXPECTED_PER_DAY
    by_corr = per.groupby("Corridor_Name")["coverage_pct"].agg(["mean", "min"]).round(1)
    print("\n  Coverage vs. ideal 96 snapshots/day (GitHub cron is best-effort, 60-90% is normal):")
    print(by_corr.to_string())

    gaps = (
        df.sort_values("Timestamp")
        .groupby("Corridor_Name")["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    print(f"\n  gaps > 45 min  : {int((gaps > 45).sum())}    largest gap: {gaps.max():,.0f} min")

    days_by_dow = df.groupby("Day_of_Week")["Date"].nunique().sort_values(ascending=False)
    print("\n  distinct calendar days observed per weekday (want >=2 for each weekday):")
    print(days_by_dow.to_string())

    print("\n" + "=" * 70)
    print("3. IS THE SIGNAL REAL?  (static-duration check)")
    print("=" * 70)
    g = df.groupby("Corridor_Name")["Estimated_Duration_Min"]
    st = pd.DataFrame(
        {
            "n": g.size(),
            "n_unique": g.nunique(),
            "mean_min": g.mean(),
            "sd_min": g.std(),
            "cv_pct": 100 * g.std() / g.mean(),
        }
    )
    hourly = df.groupby(["Corridor_Name", df["Timestamp"].dt.hour])["Estimated_Duration_Min"].mean().unstack()
    st["hourly_swing_pct"] = 100 * (hourly.max(axis=1) - hourly.min(axis=1)) / g.mean()
    print(st.to_string())

    static = st[(st["n_unique"] <= 2) | (st["cv_pct"] < 1.0)]
    if len(static) == st.shape[0]:
        print(
            "\n  !!! CRITICAL: every corridor returns (almost) the same duration at all hours."
            "\n      The public OSRM router uses fixed speeds per road class, not live traffic."
            "\n      An ANOVA on this data will only measure noise. See GUIDE.md, section 0."
        )
    elif len(static):
        print(f"\n  !! {len(static)} corridor(s) show no time variation: {list(static.index)}")
    else:
        print("\n  OK: durations vary within every corridor.")

    out = DATA_DIR / "clean_phase1.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved cleaned data -> {out}")


if __name__ == "__main__":
    main()
