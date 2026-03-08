#Everything related to the data: generating synthetic snapshots (useful while
#you are building the pipeline) and feature engineering on top of them.
#The synthetic schema matches the real Bicing GBFS feed so you can swap the
#data source later without changing anything else.
#
#Schema (what each row looks like):
#   station_id        int
#   station_type      str  (residential / central / university)
#   lat, lon          float
#   capacity          int  (total docks at the station)
#   timestamp         pandas Timestamp (15-min intervals)
#   bikes_available   int  (the thing we care about)
#   docks_available   int  (capacity - bikes_available)

import os
import numpy as np
import pandas as pd


DATA_CSV = "data/bicing_snapshots.csv"
INTERVAL_MIN = 15       #snapshot every 15 minutes (matches a realistic polling cadence)
HORIZON_STEPS = 1        #predict 1 step ahead = 15 minutes ahead


#Station types and how they behave in a typical day.
#Residential: empty in the morning (everyone leaves), fill up at night.
#Central: opposite pattern (filled by commuters in the morning).
#University: fills during lecture hours (9-14), empties after.
STATION_TYPES = ["residential", "central", "university"]


def _make_stations(n_stations=20, seed=42):
    #Builds a small "map" of stations. Coordinates are roughly Barcelona.
    rng = np.random.default_rng(seed)
    station_ids = np.arange(1, n_stations + 1)
    types = rng.choice(STATION_TYPES, size=n_stations, p=[0.5, 0.3, 0.2])
    #Barcelona is roughly 41.35-41.45 N, 2.10-2.20 E
    lats = 41.38 + rng.normal(0, 0.02, n_stations)
    lons = 2.15 + rng.normal(0, 0.02, n_stations)
    capacities = rng.integers(20, 35, n_stations)  #real Bicing stations: 20-35 docks
    return pd.DataFrame({
        "station_id": station_ids,
        "station_type": types,
        "lat": lats.round(5),
        "lon": lons.round(5),
        "capacity": capacities,
    })


def _daily_pattern(hour, station_type):
    #Returns a number between 0 and 1 representing how "full" the station
    #tends to be at this hour. It's a hand-crafted sine-ish function so that
    #the ML models have a real daily pattern to learn from.
    if station_type == "residential":
        #Full at night, empty during the day.
        return 0.5 + 0.4 * np.cos(2 * np.pi * (hour - 21) / 24)
    if station_type == "central":
        #Empty at night, full during the workday.
        return 0.5 + 0.4 * np.cos(2 * np.pi * (hour - 9) / 24 + np.pi)
    if station_type == "university":
        #Bell-shaped around 11-13h.
        return 0.3 + 0.5 * np.exp(-((hour - 12) ** 2) / 8)
    return 0.5


def generate_snapshots(n_stations=20, n_days=14, seed=42):
    #Creates a realistic time series of 15-min snapshots for n_stations stations
    #over n_days days. Saves the result to DATA_CSV and returns the DataFrame.

    rng = np.random.default_rng(seed)
    stations = _make_stations(n_stations=n_stations, seed=seed)

    #Build every (station, timestamp) pair.
    start = pd.Timestamp("2026-03-01 00:00:00")
    steps_per_day = 24 * 60 // INTERVAL_MIN   #96 snapshots per day
    timestamps = pd.date_range(start=start, periods=n_days * steps_per_day, freq=str(INTERVAL_MIN) + "min")

    rows = []
    for _, st in stations.iterrows():
        capacity = int(st["capacity"])
        station_type = st["station_type"]

        for t in timestamps:
            hour = t.hour + t.minute / 60.0
            is_weekend = t.dayofweek >= 5

            #Weekends flatten the daily pattern a bit.
            pattern = _daily_pattern(hour, station_type)
            if is_weekend:
                pattern = 0.5 + 0.4 * (pattern - 0.5)  #squash toward 0.5

            #Noise so the time series isn't deterministic.
            noise = rng.normal(0, 0.08)
            fill_ratio = max(0.0, min(1.0, pattern + noise))

            bikes = int(round(fill_ratio * capacity))
            docks = capacity - bikes

            rows.append({
                "station_id": int(st["station_id"]),
                "station_type": station_type,
                "lat": st["lat"],
                "lon": st["lon"],
                "capacity": capacity,
                "timestamp": t,
                "bikes_available": bikes,
                "docks_available": docks,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    return df


def load_snapshots():
    #Loads the CSV if it exists. If not, generates it first.
    if not os.path.exists(DATA_CSV):
        generate_snapshots()
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    return df


def build_features(df):
    #Turns raw snapshots into a feature matrix for supervised learning.
    #Each row represents "at time t, at station s, what will bikes_available
    #be in 15 minutes?"
    #
    #Features we add:
    # - hour of day, minute, day of week, is_weekend (time features)
    # - bikes_available_lag_1, lag_2, lag_3   (last 3 snapshots, 15/30/45 min ago)
    # - rolling_mean_1h                        (average over the last hour)
    # - station_type_one_hot                   (3 columns, residential/central/university)
    #
    #Target (y): bikes_available at t + HORIZON_STEPS  = bikes_available 15 min later

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    #Time features
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    #Lag features (per station).
    grp = df.groupby("station_id", group_keys=False)
    df["lag_1"] = grp["bikes_available"].shift(1)
    df["lag_2"] = grp["bikes_available"].shift(2)
    df["lag_3"] = grp["bikes_available"].shift(3)
    #Rolling mean of the last 4 snapshots (covers the past hour).
    df["rolling_1h"] = grp["bikes_available"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    #Target: the thing we want to predict.
    df["target"] = grp["bikes_available"].shift(-HORIZON_STEPS)

    #One-hot encode station type. Plain get_dummies keeps it beginner-friendly.
    df = pd.get_dummies(df, columns=["station_type"], prefix="type")

    #Drop rows with NaN (first few rows per station don't have lags, last
    #rows don't have a target because it's 15 min into the future).
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "hour", "minute", "dayofweek", "is_weekend",
        "capacity", "bikes_available", "docks_available",
        "lag_1", "lag_2", "lag_3", "rolling_1h",
        "type_residential", "type_central", "type_university",
    ]
    #Some one-hot columns may not exist if a type didn't appear. Handle that.
    for col in ["type_residential", "type_central", "type_university"]:
        if col not in df.columns:
            df[col] = 0
    return df, feature_cols


def split_by_time(df, test_frac=0.2):
    #CRUCIAL for time series: split by time, NOT randomly. The last test_frac
    #of the timeline is the "future" we evaluate on. If you random-split you
    #leak future info into training and get optimistic-but-wrong numbers.
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test = df[df["timestamp"] > cutoff].reset_index(drop=True)
    return train, test
