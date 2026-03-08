#Pulls real Bicing station data from the official GBFS feed and appends a
#snapshot to data/bicing_snapshots.csv. Meant to be run on your own laptop
#(this script needs internet access to reach the Barcelona open-data servers).
#
#Usage:
#   python fetch_live.py         #fetches once and exits
#   python fetch_live.py --loop  #keeps fetching every 15 minutes until you kill it
#
#The output CSV has the same columns as data.py's synthetic generator, so the
#training code works on either. If you only run the synthetic generator you can
#ignore this file completely.

import os
import sys
import time
import argparse

import pandas as pd
import requests


#GBFS = General Bikeshare Feed Specification. Barcelona exposes it here:
STATION_INFO_URL = "https://opendata-ajuntament.barcelona.cat/data/dataset/bicing/resource/e5adca8d-98bf-42c3-9b9c-364ef0a80494/station_information.json"
STATION_STATUS_URL = "https://opendata-ajuntament.barcelona.cat/data/dataset/bicing/resource/1b215493-9e63-4a12-8980-2d7e0fa19f85/station_status.json"

#Fallback: the canonical GBFS URLs hosted by Smou (the operator).
SMOU_STATION_INFO = "https://www.bicing.barcelona/es/get-stations"
SMOU_STATION_STATUS = "https://www.bicing.barcelona/es/get-stations"

OUT_CSV = "data/bicing_snapshots.csv"


def _fetch_json(url, timeout=10):
    #Tiny wrapper so the error message is readable if the endpoint is down.
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "bicing-predictor/1.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_station_info():
    #Returns a DataFrame with one row per station: id, name, lat, lon, capacity.
    data = _fetch_json(STATION_INFO_URL)
    #GBFS nests the list under data -> stations.
    stations = data.get("data", {}).get("stations", data.get("stations", []))
    if not stations:
        raise RuntimeError("No stations returned by the info feed.")
    rows = []
    for s in stations:
        rows.append({
            "station_id": int(s.get("station_id", s.get("id", 0))),
            "name": s.get("name", ""),
            "lat": float(s.get("lat", 0)),
            "lon": float(s.get("lon", 0)),
            "capacity": int(s.get("capacity", 0)),
        })
    return pd.DataFrame(rows)


def fetch_station_status():
    #Returns a DataFrame: id, bikes_available, docks_available, last_reported.
    data = _fetch_json(STATION_STATUS_URL)
    stations = data.get("data", {}).get("stations", data.get("stations", []))
    rows = []
    for s in stations:
        rows.append({
            "station_id": int(s.get("station_id", s.get("id", 0))),
            "bikes_available": int(s.get("num_bikes_available", 0)),
            "docks_available": int(s.get("num_docks_available", 0)),
            "last_reported": int(s.get("last_reported", 0)),
        })
    return pd.DataFrame(rows)


def fetch_one_snapshot():
    #Joins info+status and returns a DataFrame ready to append to our CSV.
    info = fetch_station_info()
    status = fetch_station_status()
    df = info.merge(status, on="station_id", how="inner")

    #Convert the Unix timestamp (last_reported) to a pandas Timestamp.
    df["timestamp"] = pd.to_datetime(df["last_reported"], unit="s").dt.round("15min")

    #We don't know station_type for real stations. Mark them as "unknown" so
    #the model can still train (the one-hot columns just become zeros).
    df["station_type"] = "unknown"

    cols = ["station_id", "station_type", "lat", "lon", "capacity",
            "timestamp", "bikes_available", "docks_available"]
    return df[cols]


def append_snapshot(df):
    #Appends the snapshot to OUT_CSV (creates the file if needed, with header).
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
    print("Wrote " + str(len(df)) + " rows to " + OUT_CSV)


def main():
    parser = argparse.ArgumentParser(description="Fetch live Bicing snapshots.")
    parser.add_argument("--loop", action="store_true",
                        help="Keep fetching every 15 minutes.")
    parser.add_argument("--interval", type=int, default=15,
                        help="Minutes between fetches when --loop is set.")
    args = parser.parse_args()

    while True:
        try:
            snap = fetch_one_snapshot()
            append_snapshot(snap)
        except Exception as e:
            #Don't crash the loop on a transient network error.
            print("[WARN] fetch failed: " + str(e))
        if not args.loop:
            break
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
