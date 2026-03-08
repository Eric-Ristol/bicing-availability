#Loads the trained model and predicts bikes_available 15 minutes ahead
#for a given station.
#
#Two ways to use it:
# 1. Import `predict_station(station_id)` from another script.
# 2. Run `python predict.py` directly and type a station id.

import os
import numpy as np
import pandas as pd
import joblib

import data


MODELS_DIR = "models"


def load_saved_model():
    #Loads the model + the feature column order. Raises if train.py
    #hasn't been run yet.
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run `python train.py` first.")

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols


def _latest_features_for_station(station_id):
    #Rebuilds the feature row for the MOST RECENT snapshot of the given
    #station so we can feed it to the model.
    df = data.load_snapshots()
    if station_id not in df["station_id"].unique():
        valid = sorted(df["station_id"].unique().tolist())
        raise ValueError("Unknown station_id " + str(station_id) +
                         ". Valid ids: " + str(valid))

    feat_df, feature_cols = data.build_features(df)
    #feat_df already dropped rows with missing lags, so the last row for this
    #station is the newest one we can predict from.
    station_rows = feat_df[feat_df["station_id"] == station_id]
    if len(station_rows) == 0:
        raise RuntimeError("Not enough history for station " + str(station_id))
    latest = station_rows.iloc[-1]
    return latest, feature_cols


def predict_station(station_id):
    #Returns a dict with the current bikes, the 15-min-ahead prediction,
    #the capacity and the timestamp. Used by main.py.
    model, feature_cols = load_saved_model()
    latest, _ = _latest_features_for_station(station_id)

    #If train.py fell back to the persistence baseline, the "model" is
    #literally the string "persistence".
    if isinstance(model, str) and model == "persistence":
        pred = float(latest["bikes_available"])
    else:
        #Use a 2-D numpy array so sklearn doesn't complain about feature names
        #(we trained on .values, so we must predict on .values too).
        X = np.array([latest[feature_cols].values], dtype=float)
        pred = float(model.predict(X)[0])

    #Clip to a sensible range. Predictions should never be negative and
    #never exceed the station capacity.
    capacity = int(latest["capacity"])
    pred = max(0.0, min(float(capacity), pred))

    return {
        "station_id": int(station_id),
        "timestamp": str(latest["timestamp"]),
        "capacity": capacity,
        "bikes_now": int(latest["bikes_available"]),
        "bikes_predicted_15min": round(pred, 1),
    }


def ask_int(prompt, minv, maxv):
    #Helper to read an int in a given range from the terminal.
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("  [ERR] Please enter a whole number.")
            continue
        if value < minv or value > maxv:
            print("  [ERR] Value must be between " + str(minv) + " and " + str(maxv) + ".")
            continue
        return value


def run_interactive():
    df = data.load_snapshots()
    ids = sorted(df["station_id"].unique().tolist())
    print("\n=== Predict bikes available in 15 minutes ===")
    print("Available station ids: " + str(ids[0]) + ".." + str(ids[-1]))
    sid = ask_int("Enter a station id: ", ids[0], ids[-1])
    result = predict_station(sid)
    print("\nStation " + str(result["station_id"]) +
          "  (capacity=" + str(result["capacity"]) + ")")
    print("Latest snapshot: " + result["timestamp"])
    print("  bikes now:       " + str(result["bikes_now"]))
    print("  bikes in 15 min: " + str(result["bikes_predicted_15min"]))


if __name__ == "__main__":
    run_interactive()
