"""
HeatGuard AI - Phase 1: Data cleaning & calendar-gap pipeline.

Input : heatguard_raw.csv  (187,387 daily rows, 7 cities, 1951-2024)
Output: data/heatguard_clean.csv  +  data/cleaning_report.json

What this script does (see PROGRESS.md, Phase 1):
  1. Parse dates, sort by (city, date), drop exact (city, date) duplicates.
  2. Detect calendar gaps and assign a `segment_id` per city. Phase 2 must compute
     lags / rolling windows *within* (city, segment_id) so nothing leaks across a gap.
     NOTE: we do NOT insert fake rows for missing dates - we only mark the gaps.
  3. temp_min: treat the 0.0 placeholder as missing, then linear-interpolate inside
     each segment (never across a gap).
  4. rain: invalidate impossible values, then fill NaN with 0.0 and add
     `rain_is_recorded` (0 = no measurement / invalid, 1 = real measurement).
  5. Validate (labels still consistent with the departure rule, no nulls left, etc).

Usage (from repo root):
    python preprocessing/data_cleaner.py
    python preprocessing/data_cleaner.py --input heatguard_raw.csv --output data/heatguard_clean.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("data_cleaner")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COLS = [
    "date", "city", "latitude", "longitude", "temp_max", "temp_min", "rain",
    "year", "month", "day_of_year", "season", "heatwave_threshold",
    "is_heatwave_day", "severity",
]
MONSOON_MONTHS = {6, 7, 8, 9}
# Rain rule: >300 mm/day OUTSIDE Jun-Sep is physically implausible for these cities
# (neighbouring days are 0 mm). Found: Kolkata & Mumbai on 2023-12-01 (~1012 mm).
# >300 mm days *inside* the monsoon (e.g. Pune 2005-07-26) are real and are kept.
RAIN_OFF_SEASON_MAX_MM = 300.0
TEMP_MIN_INTERP_LIMIT = 3  # never fill more than 3 consecutive missing days


def find_input(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in (REPO_ROOT / "heatguard_raw.csv", REPO_ROOT / "dataset" / "heatguard_raw.csv"):
        if cand.exists():
            return cand
    raise FileNotFoundError("heatguard_raw.csv not found in repo root or data/. Use --input.")


def load_and_sort(path: Path, report: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    report["rows_in"] = int(len(df))

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    dups = int(df.duplicated(["city", "date"]).sum())
    report["duplicates_dropped"] = dups
    if dups:
        df = df.drop_duplicates(["city", "date"], keep="first").reset_index(drop=True)

    df["is_heatwave_day"] = df["is_heatwave_day"].astype(int)
    return df


def add_gap_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """days_since_prev, gap_before, segment_id (per city; +1 after every gap)."""
    df["days_since_prev"] = df.groupby("city")["date"].diff().dt.days
    df["gap_before"] = (df["days_since_prev"] > 1).astype(int)
    df["segment_id"] = df.groupby("city")["gap_before"].cumsum().astype(int)

    gaps = df[df["gap_before"] == 1]
    report["calendar_gaps"] = {
        "total": int(len(gaps)),
        "by_gap_length_days": {str(int(k)): int(v) for k, v in
                               gaps["days_since_prev"].value_counts().sort_index().items()},
        "by_city": {k: int(v) for k, v in gaps.groupby("city").size().items()},
        "largest_gap_days": int(gaps["days_since_prev"].max()) if len(gaps) else 0,
    }
    return df


def clean_temp_min(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    # 0.0 is a placeholder: Bengaluru 2019-01-11 (same date where the other 6 cities are NaN)
    zero_mask = df["temp_min"] == 0.0
    report["temp_min_zero_placeholders"] = [
        {"city": r.city, "date": str(r.date.date())} for r in df[zero_mask].itertuples()
    ]
    df.loc[zero_mask, "temp_min"] = np.nan

    was_missing = df["temp_min"].isna()
    report["temp_min_missing_before"] = int(was_missing.sum())

    # interpolate strictly within (city, segment) so we never bridge a calendar gap
    df["temp_min"] = (
        df.groupby(["city", "segment_id"])["temp_min"]
        .transform(lambda s: s.interpolate(method="linear", limit=TEMP_MIN_INTERP_LIMIT,
                                           limit_area="inside"))
    )
    # segment-edge leftovers (if any): nearest value within the same segment
    still = df["temp_min"].isna()
    if still.any():
        df["temp_min"] = df.groupby(["city", "segment_id"])["temp_min"].transform(
            lambda s: s.ffill(limit=1).bfill(limit=1))

    df["temp_min_imputed"] = (was_missing & df["temp_min"].notna()).astype(int)
    report["temp_min_imputed"] = int(df["temp_min_imputed"].sum())
    report["temp_min_missing_after"] = int(df["temp_min"].isna().sum())
    return df


def clean_rain(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    invalid = (df["rain"] > RAIN_OFF_SEASON_MAX_MM) & (~df["month"].isin(MONSOON_MONTHS))
    report["rain_invalidated"] = [
        {"city": r.city, "date": str(r.date.date()), "raw_rain_mm": round(float(r.rain), 1)}
        for r in df[invalid].itertuples()
    ]
    report["rain_kept_extreme_monsoon_days"] = [
        {"city": r.city, "date": str(r.date.date()), "rain_mm": round(float(r.rain), 1)}
        for r in df[(df["rain"] > 300) & (~invalid)].itertuples()
    ]
    df.loc[invalid, "rain"] = np.nan

    df["rain_is_recorded"] = df["rain"].notna().astype(int)
    report["rain_missing_total"] = int((df["rain_is_recorded"] == 0).sum())
    report["rain_missing_by_city_pct"] = {
        k: round(float(v) * 100, 2)
        for k, v in (1 - df.groupby("city")["rain_is_recorded"].mean()).items()
    }
    df["rain"] = df["rain"].fillna(0.0)
    return df


def validate(df: pd.DataFrame, report: dict) -> None:
    """Hard checks - raise if the cleaned data breaks an assumption later phases rely on."""
    checks = {}

    checks["no_nulls_in_model_columns"] = bool(
        df[["temp_max", "temp_min", "rain", "heatwave_threshold"]].notna().all().all())
    checks["no_duplicate_city_date"] = not df.duplicated(["city", "date"]).any()
    checks["sorted_by_city_date"] = bool(
        df.equals(df.sort_values(["city", "date"]).reset_index(drop=True)))
    checks["temp_min_not_above_temp_max"] = bool((df["temp_min"] <= df["temp_max"]).all())

    # ground-truth labels must still follow the departure rule from the plan
    dep = df["temp_max"] - df["heatwave_threshold"]
    exp_sev = np.select([dep >= 4, dep >= 2, dep >= 0], ["Extreme", "Severe", "Warning"], "Normal")
    checks["is_heatwave_matches_departure_rule"] = bool(((dep >= 0).astype(int) == df["is_heatwave_day"]).all())
    sev_mismatch = int((exp_sev != df["severity"]).sum())
    checks["severity_matches_departure_rule"] = sev_mismatch == 0
    report["severity_mismatches"] = sev_mismatch

    checks["rain_non_negative"] = bool((df["rain"] >= 0).all())
    checks["rain_flag_consistent"] = bool(((df["rain_is_recorded"] == 0) | df["rain"].notna()).all())
    checks["segments_have_no_internal_gaps"] = bool(
        (df.loc[df["gap_before"] == 0, "days_since_prev"].dropna() == 1).all())

    report["validation_checks"] = checks
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")


def summarize(df: pd.DataFrame, report: dict) -> None:
    report["rows_out"] = int(len(df))
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    report["rows_by_city"] = {k: int(v) for k, v in df["city"].value_counts().sort_index().items()}
    report["severity_counts"] = {k: int(v) for k, v in df["severity"].value_counts().items()}
    report["segments_by_city"] = {k: int(v) for k, v in
                                  (df.groupby("city")["segment_id"].max() + 1).items()}
    report["columns_out"] = list(df.columns)


def clean(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict]:
    report: dict = {"input_file": str(input_path.name)}
    df = load_and_sort(input_path, report)
    df = add_gap_columns(df, report)
    df = clean_temp_min(df, report)
    df = clean_rain(df, report)
    validate(df, report)
    summarize(df, report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    # first row of each city has no previous day -> store as 0 instead of NaN
    out["days_since_prev"] = out["days_since_prev"].fillna(0).astype(int)
    out.to_csv(output_path, index=False)

    report_path = output_path.parent / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Saved %s (%d rows) and %s", output_path, len(out), report_path.name)
    return df, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="HeatGuard AI data cleaner (Phase 1)")
    ap.add_argument("--input", default=None, help="path to heatguard_raw.csv")
    ap.add_argument("--output", default=str(REPO_ROOT / "data" / "heatguard_clean.csv"))
    args = ap.parse_args()

    _, rep = clean(find_input(args.input), Path(args.output))
    print(f"\nRows in/out        : {rep['rows_in']:,} -> {rep['rows_out']:,}")
    print(f"Calendar gaps      : {rep['calendar_gaps']['total']} (largest {rep['calendar_gaps']['largest_gap_days']} days)")
    print(f"temp_min imputed   : {rep['temp_min_imputed']} (remaining NaN: {rep['temp_min_missing_after']})")
    print(f"rain invalidated   : {len(rep['rain_invalidated'])}")
    print(f"rain not recorded  : {rep['rain_missing_total']:,}")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()