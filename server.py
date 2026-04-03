"""
Flower FL Server
=================
Orchestrates federated learning across 4 bank clients using Byzantine-robust
aggregation (Krum) and centrally evaluates the global model on the global test set.

Aggregation:
  We use the Krum strategy to protect against malicious or severely mismatched updates.
  The server centrally evaluates the aggregated model on the combined global test set.
"""

import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import TensorDataset, DataLoader

from config import (
    SERVER_ADDRESS, NUM_ROUNDS, NN_PARAMS,
    MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS, MIN_AVAILABLE_CLIENTS,
)
from utils import print_banner, FLResultTracker
from data_loader import load_global_test_data
from model import FraudMLP, set_parameters, evaluate_model


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
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        print(f"\nServer: Starting Global Evaluation for Round {server_round}...")
        model = FraudMLP().to(device)
        set_parameters(model, parameters)
        
        metrics = evaluate_model(model, global_testloader, device)
        loss = metrics.pop("loss")
        
        print(f"  ► Round {server_round} GLOBAL TEST: AUC={metrics['auc']:.4f}, Loss={loss:.4f}")
        
        tracker.log_round(
            server_round,
            global_metrics={"auc": metrics["auc"], "loss": loss}
        )
        return loss, metrics

    return evaluate


def start_server():
    """Start the Flower FL server."""
    print_banner("Federated Fraud Detection Server (Krum)", char="█")
    print(f"  Address:    {SERVER_ADDRESS}")
    print(f"  Rounds:     {NUM_ROUNDS}")
    print(f"  Min clients: {MIN_FIT_CLIENTS}")

    tracker = FLResultTracker()

    # Use Krum aggregation for Byzantine robustness.
    # We tolerate 1 malicious client, and keep 1 to average/represent the round.
    strategy = fl.server.strategy.Krum(
        num_malicious_clients=1,
        num_clients_to_keep=1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        evaluate_fn=get_evaluate_fn(tracker), # Centralized global evaluation
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
