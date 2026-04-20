#Main entry point. Shows a small menu so you can drive the whole project

import os
import sys

import data
import train
import predict

def print_menu():
    print("")
    print("================================================")
    print("   BICING AVAILABILITY PREDICTOR -- Main menu")
    print("================================================")
    print("  I.   Generate a synthetic dataset")
    print("  II.  Train models (baseline + LinearRegression + RandomForest)")
    print("  III. Predict bikes in 15 min for a station")
    print("  IV.  Show dataset summary")
    print("  V.   Show model comparison")
    print("  VI.  Launch API server (web demo)")
    print("  VII. Exit")
    print("------------------------------------------------")

def option_generate():
    print("\n>>> Generating synthetic snapshots (20 stations, 14 days)...")
    df = data.generate_snapshots()
    print("    " + str(len(df)) + " rows saved to " + data.DATA_CSV)

def option_train():
    print("\n>>> Training models...")
    comp = train.run_training()
    print("\nFinal comparison:")
    print(comp.to_string(index=False))

def option_predict():
    try:
        predict.run_interactive()
    except FileNotFoundError as e:
        print("  [ERR] " + str(e))
    except ValueError as e:
        print("  [ERR] " + str(e))

def option_summary():
    if not os.path.exists(data.DATA_CSV):
        print("  [ERR] No dataset yet. Pick option I first.")
        return
    df = data.load_snapshots()
    print("\nDataset summary:")
    print("  rows:      " + str(len(df)))
    print("  stations:  " + str(df["station_id"].nunique()))
    print("  timestamps:" + str(df["timestamp"].nunique()))
    print("  from:      " + str(df["timestamp"].min()))
    print("  to:        " + str(df["timestamp"].max()))
    print("  mean bikes_available: " +
          str(round(df["bikes_available"].mean(), 2)))
    print("  by station type:")
    print(df.groupby("station_type")["bikes_available"].mean().round(2).to_string())

def option_comparison():
    path = os.path.join("models", "comparison.csv")
    if not os.path.exists(path):
        print("  [ERR] No comparison yet. Pick option II first.")
        return
    import pandas as pd
    comp = pd.read_csv(path)
    print("\nModel comparison (sorted by RMSE):")
    print(comp.sort_values("rmse").to_string(index=False))

def option_api():
    print("\n>>> Starting API server at http://localhost:8000")
    print("    Open that URL in your browser to use the web demo.")
    print("    Press Ctrl+C to stop the server.\n")
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

def main():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip().upper()
        if choice == "I":
            option_generate()
        elif choice == "II":
            option_train()
        elif choice == "III":
            option_predict()
        elif choice == "IV":
            option_summary()
        elif choice == "V":
            option_comparison()
        elif choice == "VI":
            option_api()
        elif choice == "VII" or choice in ("Q", "EXIT", "QUIT"):
            print("Bye!")
            sys.exit(0)
        else:
            print("  [ERR] Unknown option. Try I, II, III, IV, V, VI or VII.")

if __name__ == "__main__":
    main()
