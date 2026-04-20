#Basic pytest tests. Covers the happy path of each main function.

import os
import pandas as pd

import data
import train
import predict

def test_generate_creates_csv():
    #Generating the dataset should create the CSV file with the expected columns.
    df = data.generate_snapshots(n_stations=10, n_days=5)
    assert os.path.exists(data.DATA_CSV)
    expected = {"station_id", "station_type", "lat", "lon", "capacity",
                "timestamp", "bikes_available", "docks_available"}
    assert expected.issubset(df.columns)
    #10 stations * 5 days * 96 snapshots/day = 4800 rows.
    assert len(df) == 10 * 5 * 96

def test_generate_default_size():
    #Regenerate with the default size so later tests have enough history.
    df = data.generate_snapshots()
    assert len(df) == 20 * 14 * 96
    #bikes_available must always be within [0, capacity].
    assert (df["bikes_available"] >= 0).all()
    assert (df["bikes_available"] <= df["capacity"]).all()

def test_build_features_adds_expected_columns():
    #All feature columns should be present and contain no NaNs after the dropna.
    df = data.load_snapshots()
    feat_df, feature_cols = data.build_features(df)
    for col in feature_cols:
        assert col in feat_df.columns, "Missing feature: " + col
    assert feat_df[feature_cols].isna().sum().sum() == 0
    assert "target" in feat_df.columns
    assert feat_df["target"].isna().sum() == 0

def test_time_based_split_has_no_leakage():
    #Train must end strictly at or before the start of test. If there's
    #overlap we have a leakage bug.
    df = data.load_snapshots()
    feat_df, _ = data.build_features(df)
    train_df, test_df = data.split_by_time(feat_df, test_frac=0.2)
    assert len(train_df) > 0 and len(test_df) > 0
    assert train_df["timestamp"].max() <= test_df["timestamp"].min()

def test_training_saves_model():
    #Running the full training should produce the model + feature list + comparison.
    train.run_training()
    assert os.path.exists(os.path.join("models", "best_model.joblib"))
    assert os.path.exists(os.path.join("models", "feature_names.joblib"))
    assert os.path.exists(os.path.join("models", "comparison.csv"))

def test_training_beats_persistence_baseline():
    #The winning model should be at least as good as the persistence baseline.
    #If a proper model can't match "predict t+15 = t_now" on this much data,
    #something is wrong with the features or the training code.
    comp = pd.read_csv(os.path.join("models", "comparison.csv"))
    persistence_rmse = comp[comp["name"] == "Persistence"]["rmse"].iloc[0]
    best_rmse = comp["rmse"].min()
    assert best_rmse <= persistence_rmse + 1e-6, \
        "Best RMSE (" + str(best_rmse) + ") is worse than persistence (" + str(persistence_rmse) + ")"

def test_predict_returns_sensible_value():
    #A prediction for any known station should be between 0 and its capacity.
    df = data.load_snapshots()
    station_id = int(df["station_id"].iloc[0])
    result = predict.predict_station(station_id)
    assert 0 <= result["bikes_predicted_15min"] <= result["capacity"]
    assert result["station_id"] == station_id
