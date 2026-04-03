"""
Flower FL Server
=================
Orchestrates federated learning across 4 bank clients using Byzantine-robust
aggregation (Krum) and centrally evaluates the global model on the global test set.

Features:
  - Krum aggregation (tolerates 1 malicious client)
  - Centralized global test evaluation after every round
  - MLflow experiment tracking (per-round metrics)
  - Model persistence (saves final global model to models/)
"""

import os
import numpy as np
import torch
import mlflow
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import TensorDataset, DataLoader

from config import (
    SERVER_ADDRESS, NUM_ROUNDS, NN_PARAMS,
    MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_AVAILABLE_CLIENTS,
    PROJECT_ROOT,
)
from utils import print_banner, FLResultTracker
from data_loader import load_global_test_data
from model import FraudMLP, get_parameters, set_parameters, evaluate_model


MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_evaluate_fn(tracker: FLResultTracker):
    """Return an evaluation function for centralized evaluation on the global test set."""
    print("Server: Loading global test data for centralized evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_X, g_y = load_global_test_data()
    g_X_t = torch.tensor(g_X, dtype=torch.float32)
    g_y_t = torch.tensor(g_y, dtype=torch.float32)
    global_test_ds = TensorDataset(g_X_t, g_y_t)
    global_testloader = DataLoader(global_test_ds, batch_size=NN_PARAMS["batch_size"] * 2)

    print(f"Server: Global test set loaded with {len(g_y)} samples (Fraud: {g_y.mean()*100:.2f}%).")

    # Track the latest parameters so we can save the final model
    latest_params = {"params": None}

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        print(f"\nServer: Starting Global Evaluation for Round {server_round}...")
        model = FraudMLP().to(device)
        set_parameters(model, parameters)

        metrics = evaluate_model(model, global_testloader, device)
        loss = metrics.pop("loss")

        print(f"  ► Round {server_round} GLOBAL TEST: AUC={metrics['auc']:.4f}, Loss={loss:.4f}")

        # Log to MLflow
        mlflow.log_metrics({
            "global_auc": metrics["auc"],
            "global_loss": loss,
            "global_pr_auc": metrics["pr_auc"],
            "global_precision": metrics["precision"],
            "global_recall": metrics["recall"],
            "global_f1": metrics["f1"],
        }, step=server_round)

        tracker.log_round(
            server_round,
            global_metrics={"auc": metrics["auc"], "loss": loss}
        )

        # Store latest params for model saving
        latest_params["params"] = parameters

        # Save model on final round
        if server_round == NUM_ROUNDS:
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = os.path.join(MODELS_DIR, "global_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Global model saved to {model_path}")
            mlflow.log_artifact(model_path)

        return loss, metrics

    return evaluate


def start_server():
    """Start the Flower FL server."""
    print_banner("Federated Fraud Detection Server (Krum)", char="█")
    print(f"  Address:    {SERVER_ADDRESS}")
    print(f"  Rounds:     {NUM_ROUNDS}")
    print(f"  Min clients: {MIN_FIT_CLIENTS}")

    tracker = FLResultTracker()

    # Set up MLflow
    mlflow.set_experiment("federated_fraud_detection")
    with mlflow.start_run(run_name="fl_training_krum"):
        mlflow.log_params({
            "strategy": "Krum",
            "num_rounds": NUM_ROUNDS,
            "num_clients": MIN_FIT_CLIENTS,
            "num_malicious_clients": 1,
            "hidden_size_1": NN_PARAMS["hidden_size_1"],
            "hidden_size_2": NN_PARAMS["hidden_size_2"],
            "learning_rate": NN_PARAMS["learning_rate"],
            "batch_size": NN_PARAMS["batch_size"],
            "local_epochs": NN_PARAMS["local_epochs"],
            "pos_weight": NN_PARAMS["pos_weight"],
        })

        strategy = fl.server.strategy.Krum(
            num_malicious_clients=1,
            num_clients_to_keep=1,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=MIN_FIT_CLIENTS,
            min_evaluate_clients=MIN_EVALUATE_CLIENTS,
            min_available_clients=MIN_AVAILABLE_CLIENTS,
            evaluate_fn=get_evaluate_fn(tracker),
        )

        fl.server.start_server(
            server_address=SERVER_ADDRESS,
            config=fl.server.ServerConfig(
                num_rounds=NUM_ROUNDS,
                round_timeout=600.0,
            ),
            strategy=strategy,
        )

    tracker.save("fl_results.json")
    tracker.print_summary()
    print_banner("Server shutdown complete")


if __name__ == "__main__":
    start_server()
