#Trains models to predict bikes_available 15 minutes into the future.

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  #headless backend so this works without a display
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

import data

MODELS_DIR = "models"
PLOTS_DIR = "plots"

def persistence_baseline(test_df):
    #Predicts "the future will look exactly like right now".
    y_true = test_df["target"].values
    y_pred = test_df["bikes_available"].values
    return y_true, y_pred

def evaluate(y_true, y_pred):
    #Standard regression metrics. RMSE is the main one for bike counts
    #users angry when a station is full/empty).
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def build_models():
    #Dict so it's easy to add/remove candidates. All models are small enough
    #to train in seconds on a laptop.
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=80, max_depth=12, random_state=42, n_jobs=-1
        ),
        #LightGBM: gradient-boosted decision trees. sklearn-compatible API so
        #it slots right in alongside the others. n_estimators=200 with early
        #stopping would be better on real data, but 150 is fine for the
        #synthetic dataset here (and trains in ~1 second).
        "LightGBM": LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,  #silence LightGBM's per-iteration output
        ),
    }

def plot_feature_importance(model, feature_cols, out_path):
    #Horizontal bar chart of LightGBM's feature importances (gain).
    #Gain measures how much each feature reduces loss when used for a split --
    #a much more meaningful metric than "split count" (which just counts how
    #often the feature was used regardless of how helpful it was).
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1])

    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("LightGBM -- feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_true_vs_pred(y_true, y_pred, title, out_path):
    #Simple scatter + diagonal. Points close to the y=x line are good
    #predictions; points far from it are the ones we care about.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual bikes_available (t+15 min)")
    ax.set_ylabel("Predicted bikes_available")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def run_training():
    #Full training flow:

    print(">>> Loading snapshots...")
    df = data.load_snapshots()
    print("    " + str(len(df)) + " rows, " + str(df["station_id"].nunique()) + " stations")

    print(">>> Building features...")
    feat_df, feature_cols = data.build_features(df)

    print(">>> Time-based split...")
    train_df, test_df = data.split_by_time(feat_df, test_frac=0.2)
    print("    train: " + str(len(train_df)) + " rows, test: " + str(len(test_df)) + " rows")
    if len(test_df) == 0:
        raise RuntimeError("Test set is empty. Generate more data first.")

    #Keep feature names as a DataFrame so LightGBM doesn't warn about
    #"feature names not found". sklearn models ignore the column names, so
    X_train = train_df[feature_cols]
    y_train = train_df["target"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values

    results = []

    #Baseline first so we always have a reference number to beat.
    print(">>> Persistence baseline")
    y_true_base, y_pred_base = persistence_baseline(test_df)
    metrics = evaluate(y_true_base, y_pred_base)
    metrics["name"] = "Persistence"
    results.append(metrics)
    print("    MAE=" + str(round(metrics["mae"], 3)) +
          "  RMSE=" + str(round(metrics["rmse"], 3)) +
          "  R2=" + str(round(metrics["r2"], 3)))

    best_name = "Persistence"
    best_rmse = metrics["rmse"]
    best_model = None
    best_preds = y_pred_base

    for name, model in build_models().items():
        print(">>> Training " + name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["name"] = name
        results.append(metrics)
        print("    MAE=" + str(round(metrics["mae"], 3)) +
              "  RMSE=" + str(round(metrics["rmse"], 3)) +
              "  R2=" + str(round(metrics["r2"], 3)))
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_model = model
            best_preds = y_pred

    #Save artifacts.
    os.makedirs(MODELS_DIR, exist_ok=True)
    comp = pd.DataFrame(results)[["name", "mae", "rmse", "r2"]]
    comp.to_csv(os.path.join(MODELS_DIR, "comparison.csv"), index=False)

    if best_model is not None:
        joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    else:
        #The baseline won. Save a tiny marker so predict.py can fall back to it.
        joblib.dump("persistence", os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names.joblib"))

    #Diagnostic plot uses the winner's predictions (or baseline if no model won).
    plot_true_vs_pred(
        np.array(y_test if best_model is not None else y_true_base),
        np.array(best_preds),
        "True vs Predicted (" + best_name + ")",
        os.path.join(PLOTS_DIR, "true_vs_pred.png"),
    )

    #Feature importance plot -- only possible when a tree model won
    #(LightGBM or RandomForest both expose .feature_importances_).
    if best_model is not None and hasattr(best_model, "feature_importances_"):
        plot_feature_importance(
            best_model,
            feature_cols,
            os.path.join(PLOTS_DIR, "feature_importance.png"),
        )
        print(">>> Feature importance saved to " + os.path.join(PLOTS_DIR, "feature_importance.png"))

    print(">>> Winner: " + best_name + "  RMSE=" + str(round(best_rmse, 3)))
    print(">>> Comparison saved to " + os.path.join(MODELS_DIR, "comparison.csv"))
    return comp

if __name__ == "__main__":
    run_training()
