#FastAPI wrapper around the Bicing availability predictor.

import os
import sys

#Add the project root to the path so we can import predict.py / data.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import predict
import data

app = FastAPI(
    title="Bicing Availability Predictor API",
    description="Predicts bikes available at a Barcelona Bicing station 15 minutes ahead.",
    version="1.0.0",
)

#Serve static files.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#Load model once at startup.
model = None
feature_cols = None

class PredictRequest(BaseModel):
    station_id: int

class PredictResponse(BaseModel):
    station_id: int
    timestamp: str
    capacity: int
    bikes_now: int
    bikes_predicted_15min: float

@app.on_event("startup")
def load_on_startup():
    global model, feature_cols
    print("Loading trained model...")
    model, feature_cols = predict.load_saved_model()
    print("Model ready.")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/stations")
def list_stations():
    #Returns all station ids with their type, capacity, and coordinates.
    df = data.load_snapshots()
    stations = df.groupby("station_id").first().reset_index()
    result = []
    for _, row in stations.iterrows():
        result.append({
            "station_id": int(row["station_id"]),
            "station_type": row["station_type"],
            "capacity": int(row["capacity"]),
            "lat": round(float(row["lat"]), 5),
            "lon": round(float(row["lon"]), 5),
        })
    return result

@app.post("/predict", response_model=PredictResponse)
def predict_bikes(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        result = predict.predict_station(req.station_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
