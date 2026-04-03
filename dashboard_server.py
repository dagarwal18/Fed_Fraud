"""
Dashboard Server
==================
FastAPI backend serving:
  - REST endpoints for FL results and baseline data
  - WebSocket endpoint for live transaction streaming simulation
  - Static dashboard.html at root

Usage:
    python dashboard_server.py
    → Open http://localhost:8765
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

from config import PROJECT_ROOT, BANK_IDS, NN_PARAMS, NUM_FEATURES
from data_loader import load_global_test_data
from model import FraudMLP, set_parameters

app = FastAPI(title="Federated Fraud Detection Dashboard")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DASHBOARD_HTML = os.path.join(PROJECT_ROOT, "dashboard.html")


# ─── REST Endpoints ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML page."""
    with open(DASHBOARD_HTML, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/fl-results")
async def get_fl_results():
    """Return FL round-by-round results."""
    path = os.path.join(PROJECT_ROOT, "fl_results.json")
    if not os.path.exists(path):
        return {"error": "fl_results.json not found. Run run_fl.py first."}
    with open(path) as f:
        return json.load(f)


@app.get("/api/baseline")
async def get_baseline_results():
    """Return per-bank baseline results."""
    path = os.path.join(PROJECT_ROOT, "baseline_results.json")
    if not os.path.exists(path):
        return {"error": "baseline_results.json not found. Run baseline.py first."}
    with open(path) as f:
        return json.load(f)


# ─── WebSocket: Live Transaction Stream ──────────────────────────

def load_global_model():
    """Load the saved global model for inference."""
    model_path = os.path.join(MODELS_DIR, "global_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Global model not found at {model_path}. Run run_fl.py first."
        )
    device = torch.device("cpu")  # CPU for streaming inference
    model = FraudMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def build_test_rows():
    """
    Load all banks' test data row-by-row with bank labels.
    Returns a list of (bank_id, feature_vector, actual_label) tuples, shuffled.
    """
    rows = []
    for bank_id in BANK_IDS:
        bank_dir = os.path.join(PROJECT_ROOT, "banks", bank_id)
        import pandas as pd
        df = pd.read_csv(os.path.join(bank_dir, "test.csv"))
        y = df["isFraud"].values.astype(np.float32)
        X = df.drop(columns=["isFraud"]).values.astype(np.float32)
        # Get the TransactionAmt column (first feature column)
        amounts = df["TransactionAmt"].values if "TransactionAmt" in df.columns else np.zeros(len(y))

        for i in range(len(y)):
            rows.append({
                "bank": bank_id,
                "features": X[i],
                "actual": int(y[i]),
                "amount": float(amounts[i]),
            })

    random.shuffle(rows)
    return rows


@app.websocket("/ws/stream")
async def stream_transactions(websocket: WebSocket):
    """
    WebSocket endpoint that simulates live transaction scoring.
    On connect: loads the global model and streams predictions.
    """
    await websocket.accept()

    try:
        # Load model
        model, device = load_global_model()
        rows = build_test_rows()

        print(f"[Stream] Connected. Streaming {len(rows)} transactions...")

        for i, row in enumerate(rows):
            # Run inference
            with torch.no_grad():
                x = torch.tensor(row["features"], dtype=torch.float32).unsqueeze(0).to(device)
                logit = model(x).squeeze().item()
                prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid

            prediction = "FRAUD" if prob >= 0.5 else "LEGIT"
            actual_label = "FRAUD" if row["actual"] == 1 else "LEGIT"
            correct = (prediction == actual_label)

            # Determine row category for color coding
            if prediction == "FRAUD" and actual_label == "FRAUD":
                category = "true_positive"
            elif prediction == "FRAUD" and actual_label == "LEGIT":
                category = "false_positive"
            elif prediction == "LEGIT" and actual_label == "FRAUD":
                category = "false_negative"
            else:
                category = "true_negative"

            message = {
                "index": i + 1,
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "bank": row["bank"],
                "amount": round(row["amount"], 2),
                "fraud_prob": round(prob, 4),
                "prediction": prediction,
                "actual": actual_label,
                "correct": correct,
                "category": category,
            }

            await websocket.send_json(message)

            # Random sleep between 50-200ms
            await asyncio.sleep(random.uniform(0.05, 0.2))

        # Signal stream complete
        await websocket.send_json({"type": "complete", "total": len(rows)})

    except WebSocketDisconnect:
        print("[Stream] Client disconnected.")
    except Exception as e:
        print(f"[Stream] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


if __name__ == "__main__":
    print(f"\n  Dashboard: http://localhost:8765")
    print(f"  Press Ctrl+C to stop.\n")
    uvicorn.run(app, host="0.0.0.0", port=8765)
