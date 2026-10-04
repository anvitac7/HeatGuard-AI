"""
HeatGuard AI - Phase 2: Zero-leakage feature engineering.

Input : data/heatguard_clean.csv      (output of preprocessing/data_cleaner.py)
Output: data/model_ready_dataset.csv  (features + targets + split label)
        data/feature_schema.json      (which columns are features / targets / reference)
        data/feature_report.json      (row counts, class balance per split, leakage checks)

Rules enforced here (see PROGRESS.md Phase 2 notes):
  * Every lag / rolling feature is computed inside (city, segment_id), so no window ever
    crosses a calendar gap. Rows without a full 7-day history are dropped.
  * Features use information up to and including day t only.
  * Targets describe day t+1 and exist only if the next row is exactly 1 day later.
  * Chronological split by the *target* date: train <=2015, val 2016-2020, test 2021-2024.

Usage (from repo root):
    python preprocessing/feature_engineering.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("feature_engineering")
REPO_ROOT = Path(__file__).resolve().parent.parent

GROUP = ["city", "segment_id"]
CITIES = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai", "Pune"]
SEASONS = ["Monsoon", "Post-Monsoon", "Summer", "Winter"]
MIN_HISTORY_DAYS = 7  # longest lag / rolling window

TRAIN_END_YEAR, VAL_END_YEAR = 2015, 2020
YEAR_MIN, YEAR_MAX = 1951, 2024
SEV_ORDER = ["Normal", "Warning", "Severe", "Extreme"]

# Day-t columns that are kept for reference (persistence baseline, evaluation) but are
# NEVER model inputs - they trivially encode the answer.
REFERENCE_COLS = ["is_heatwave_day", "severity", "heatwave_threshold"]
META_COLS = ["date", "city", "season", "split", "target_date"]
TARGET_COLS = ["target_heatwave_next_day", "target_departure_next_day",
               "target_severity_next_day", "target_temp_max_next_day"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------------------------
def add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(GROUP, sort=False)

    for lag in (1, 2, 3, 7):
        df[f"temp_max_lag{lag}"] = g["temp_max"].shift(lag)
        df[f"temp_min_lag{lag}"] = g["temp_min"].shift(lag)
    for lag in (1, 3, 7):
        df[f"rain_lag{lag}"] = g["rain"].shift(lag)

    df["diurnal_temp_range"] = df["temp_max"] - df["temp_min"]
    g = df.groupby(GROUP, sort=False)  # re-group so new column is visible

    def roll(col: str, window: int, how: str) -> pd.Series:
        # window ends at day t (inclusive); full window required -> NaN near segment start
        return g[col].transform(lambda s: getattr(s.rolling(window, min_periods=window), how)())

    df["temp_max_3d_avg"] = roll("temp_max", 3, "mean")
    df["temp_max_7d_avg"] = roll("temp_max", 7, "mean")
    df["temp_max_7d_max"] = roll("temp_max", 7, "max")
    df["temp_min_3d_avg"] = roll("temp_min", 3, "mean")
    df["temp_min_7d_avg"] = roll("temp_min", 7, "mean")
    df["rain_7d_sum"] = roll("rain", 7, "sum")
    df["diurnal_range_3d_avg"] = roll("diurnal_temp_range", 3, "mean")
    return df


def add_heatwave_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive heatwave days ending today (0 if today is not a heatwave day)."""
    hw = df["is_heatwave_day"].astype(int)
    # new run id every time hw changes value or a new (city, segment) starts
    new_run = (hw != hw.shift()) | (df[GROUP] != df[GROUP].shift()).any(axis=1)
    run_id = new_run.cumsum()
    df["heatwave_streak_days"] = hw.groupby(run_id).cumsum() * hw
    return df


def add_calendar_and_spatial(df: pd.DataFrame) -> pd.DataFrame:
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["year_scaled"] = (df["year"] - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)

    for c in CITIES:
        df[f"city_{c}"] = (df["city"] == c).astype(int)
    for s in SEASONS:
        df[f"season_{s}"] = (df["season"] == s).astype(int)

    df["temp_departure_today"] = df["temp_max"] - df["heatwave_threshold"]
    return df


# ----------------------------------------------------------------------------------------
# Targets + known-future covariate
# ----------------------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", sort=False)
    nxt_date = g["date"].shift(-1)
    consecutive = (nxt_date - df["date"]).dt.days == 1  # next row must be exactly t+1

    df["target_date"] = nxt_date
    nxt_hw = g["is_heatwave_day"].shift(-1)
    nxt_temp = g["temp_max"].shift(-1)
    nxt_thr = g["heatwave_threshold"].shift(-1)
    nxt_sev = g["severity"].shift(-1)

    df["target_heatwave_next_day"] = nxt_hw.where(consecutive)
    df["target_temp_max_next_day"] = nxt_temp.where(consecutive)
    df["target_departure_next_day"] = (nxt_temp - nxt_thr).where(consecutive)
    df["target_severity_next_day"] = nxt_sev.where(consecutive)

    # IMD threshold is a fixed climatological calendar (same value for a given city and
    # day-of-year every year), so tomorrow's threshold is known today - NOT leakage.
    df["threshold_next_day"] = nxt_thr.where(consecutive)
    df["gap_to_next_threshold"] = df["temp_max"] - df["threshold_next_day"]
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    ty = df["target_date"].dt.year
    df["split"] = np.select([ty <= TRAIN_END_YEAR, ty <= VAL_END_YEAR], ["train", "val"], "test")
    return df


# ----------------------------------------------------------------------------------------
# Feature lists
# ----------------------------------------------------------------------------------------
def feature_groups() -> dict[str, list[str]]:
    return {
        "current_weather": ["temp_max", "temp_min", "rain", "diurnal_temp_range",
                            "temp_departure_today"],
        "lags": [f"temp_max_lag{l}" for l in (1, 2, 3, 7)]
                + [f"temp_min_lag{l}" for l in (1, 2, 3, 7)]
                + [f"rain_lag{l}" for l in (1, 3, 7)],
        "rolling": ["temp_max_3d_avg", "temp_max_7d_avg", "temp_max_7d_max",
                    "temp_min_3d_avg", "temp_min_7d_avg", "rain_7d_sum",
                    "diurnal_range_3d_avg"],
        "heatwave_state": ["heatwave_streak_days"],
        "calendar": ["sin_doy", "cos_doy", "sin_month", "cos_month", "year_scaled"],
        "spatial": ["latitude", "longitude"] + [f"city_{c}" for c in CITIES],
        "season_onehot": [f"season_{s}" for s in SEASONS],
        "missingness": ["rain_is_recorded"],
        # tomorrow's threshold is deterministic climatology -> legitimately known today
        "known_future": ["threshold_next_day", "gap_to_next_threshold"],
    }


def all_features() -> list[str]:
    return [c for cols in feature_groups().values() for c in cols]


# ----------------------------------------------------------------------------------------
# Leakage / correctness checks (independent of the pandas shift logic above)
# ----------------------------------------------------------------------------------------
def verify(model_df: pd.DataFrame, clean_df: pd.DataFrame, report: dict, n_sample: int = 3000) -> None:
    checks: dict[str, bool] = {}
    feats = all_features()

    # 1. forbidden raw columns are never inputs
    checks["no_forbidden_columns_in_features"] = not (set(feats) & set(REFERENCE_COLS))
    checks["no_target_columns_in_features"] = not (set(feats) & set(TARGET_COLS))

    # 2. no nulls in features or targets
    checks["no_nulls_in_features"] = bool(model_df[feats].notna().all().all())
    checks["no_nulls_in_targets"] = bool(model_df[TARGET_COLS].notna().all().all())

    # 3. recompute a random sample from raw (date, city) lookups and compare
    look = clean_df.set_index(["city", "date"])
    rng = np.random.default_rng(0)
    sample = model_df.iloc[rng.choice(len(model_df), size=min(n_sample, len(model_df)), replace=False)]
    one = pd.Timedelta(days=1)
    ok_lag = ok_roll = ok_tgt = ok_streak = True
    for r in sample.itertuples():
        c, d = r.city, r.date
        past = lambda k, col: look.at[(c, d - k * one), col]
        ok_lag &= np.isclose(r.temp_max_lag1, past(1, "temp_max"))
        ok_lag &= np.isclose(r.temp_max_lag7, past(7, "temp_max"))
        ok_lag &= np.isclose(r.rain_lag3, past(3, "rain"))
        w3 = [look.at[(c, d - k * one), "temp_max"] for k in range(3)]
        w7 = [look.at[(c, d - k * one), "temp_max"] for k in range(7)]
        ok_roll &= np.isclose(r.temp_max_3d_avg, np.mean(w3))
        ok_roll &= np.isclose(r.temp_max_7d_max, np.max(w7))
        ok_tgt &= np.isclose(r.target_temp_max_next_day, look.at[(c, d + one), "temp_max"])
        ok_tgt &= int(r.target_heatwave_next_day) == int(look.at[(c, d + one), "is_heatwave_day"])
        ok_tgt &= r.target_severity_next_day == look.at[(c, d + one), "severity"]
        # streak: walk back while heatwave
        k, s = 0, 0
        while (c, d - k * one) in look.index and look.at[(c, d - k * one), "is_heatwave_day"] == 1:
            s += 1; k += 1
        ok_streak &= int(r.heatwave_streak_days) == s
    checks["lags_match_independent_lookup"] = bool(ok_lag)
    checks["rolling_windows_match_independent_lookup"] = bool(ok_roll)
    checks["targets_match_next_day_lookup"] = bool(ok_tgt)
    checks["heatwave_streak_matches_walk_back"] = bool(ok_streak)

    # 4. target is exactly one day after the feature date
    checks["target_date_is_t_plus_1"] = bool(((model_df["target_date"] - model_df["date"]).dt.days == 1).all())

    # 5. chronological split, no overlap
    by = model_df.groupby("split")["target_date"].agg(["min", "max"])
    checks["split_is_chronological"] = bool(
        by.loc["train", "max"] < by.loc["val", "min"] and by.loc["val", "max"] < by.loc["test", "min"])

    # 6. departure target consistent with severity rule
    dep = model_df["target_departure_next_day"]
    exp = np.select([dep >= 4, dep >= 2, dep >= 0], ["Extreme", "Severe", "Warning"], "Normal")
    checks["departure_target_matches_severity_target"] = bool((exp == model_df["target_severity_next_day"]).all())
    checks["heatwave_target_matches_departure_target"] = bool(
        ((dep >= 0).astype(int) == model_df["target_heatwave_next_day"].astype(int)).all())

    report["leakage_and_correctness_checks"] = checks
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"Feature verification failed: {failed}")


# ----------------------------------------------------------------------------------------
def build(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    clean_df = load(input_path)
    df = clean_df.copy()
    report["rows_in"] = int(len(df))

    df = add_lags_and_rolling(df)
    df = add_heatwave_streak(df)
    df = add_calendar_and_spatial(df)
    df = add_targets(df)

    n0 = len(df)
    no_history = df[[c for c in all_features() if c not in ("threshold_next_day", "gap_to_next_threshold")]].isna().any(axis=1)
    no_target = df[TARGET_COLS].isna().any(axis=1)
    report["rows_dropped_no_7day_history"] = int((no_history & ~no_target).sum())
    report["rows_dropped_no_valid_next_day_target"] = int(no_target.sum())
    df = df[~(no_history | no_target)].reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["rows_dropped_total"] = int(n0 - len(df))

    df["target_heatwave_next_day"] = df["target_heatwave_next_day"].astype(int)
    df = assign_split(df)

    verify(df, clean_df, report)

    # ---- report
    sp = df.groupby("split").agg(rows=("city", "size"),
                                 heatwave_next_day=("target_heatwave_next_day", "sum"),
                                 first_target=("target_date", "min"), last_target=("target_date", "max"))
    sp["positive_rate_pct"] = (100 * sp["heatwave_next_day"] / sp["rows"]).round(3)
    report["split_summary"] = {
        k: {"rows": int(v.rows), "heatwave_next_day": int(v.heatwave_next_day),
            "positive_rate_pct": float(v.positive_rate_pct),
            "first_target": str(v.first_target.date()), "last_target": str(v.last_target.date())}
        for k, v in sp.iterrows()}
    report["severity_target_counts_by_split"] = {
        s: {k: int(v) for k, v in grp["target_severity_next_day"].value_counts().items()}
        for s, grp in df.groupby("split")}
    report["n_features"] = len(all_features())
    report["persistence_sanity"] = {  # P(HW tomorrow | HW today) - plan says ~63%
        "p_hw_next_given_hw_today_pct": round(float(
            100 * df.loc[df.is_heatwave_day == 1, "target_heatwave_next_day"].mean()), 2)}

    # ---- save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = META_COLS + all_features() + REFERENCE_COLS + TARGET_COLS
    out = df[out_cols].copy()
    # round float FEATURES only (removes float32 noise, shrinks file ~40%); targets and
    # reference columns stay exact so label rules keep holding
    float_feats = [c for c in all_features() if out[c].dtype == "float64"]
    out[float_feats] = out[float_feats].round(4)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["target_date"] = out["target_date"].dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)

    schema = {
        "feature_columns": all_features(),
        "feature_groups": feature_groups(),
        "target_columns": TARGET_COLS,
        "primary_target": "target_heatwave_next_day",
        "reference_columns_never_inputs": REFERENCE_COLS,
        "meta_columns": META_COLS,
        "categorical_string_columns_for_catboost": ["city", "season"],
        "split_column": "split",
        "split_rule": {"train": f"target_date <= {TRAIN_END_YEAR}",
                       "val": f"{TRAIN_END_YEAR + 1} <= target_date <= {VAL_END_YEAR}",
                       "test": f"target_date >= {VAL_END_YEAR + 1}"},
    }
    (output_path.parent / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    (output_path.parent / "feature_report.json").write_text(json.dumps(report, indent=2))
    log.info("Saved %s (%d rows x %d cols)", output_path, *out.shape)
    return df, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="HeatGuard AI feature engineering (Phase 2)")
    ap.add_argument("--input", default=str(REPO_ROOT / "data" / "heatguard_clean.csv"))
    ap.add_argument("--output", default=str(REPO_ROOT / "data" / "model_ready_dataset.csv"))
    args = ap.parse_args()
    _, rep = build(Path(args.input), Path(args.output))

    print(f"\nRows            : {rep['rows_in']:,} -> {rep['rows_out']:,} (dropped {rep['rows_dropped_total']})")
    print(f"Features        : {rep['n_features']}")
    for s in ("train", "val", "test"):
        v = rep["split_summary"][s]
        print(f"{s:<5}: {v['rows']:>7,} rows | {v['heatwave_next_day']:>5} heatwave-next-day ({v['positive_rate_pct']}%)")
    print("All leakage / correctness checks passed.")


if __name__ == "__main__":
    main()