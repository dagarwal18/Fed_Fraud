"""
Flower FL Client
=================
Each client represents one bank. It:
  1. Loads its own local data (no cross-bank access)
  2. Receives global model parameters from the server
  3. Trains locally on its own data
  4. Sends updated model back to the server
  5. Evaluates on its local validation set
"""

import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl

from config import SERVER_ADDRESS, BANK_IDS, NN_PARAMS
from data_loader import load_bank_data
from model import (
    FraudMLP, get_parameters, set_parameters, train_model, evaluate_model
)
from utils import print_metrics, print_banner


class FraudDetectionClient(fl.client.NumPyClient):
    """Flower client for one bank's PyTorch MLP fraud detection model."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FraudMLP().to(self.device)
        self.round_num = 0

        # Load this bank's data
        print_banner(f"Initializing Client: {bank_id} on {self.device}")
        data = load_bank_data(bank_id)
        
        # Convert to PyTorch DataLoaders
        X_train_t = torch.tensor(data["X_train"], dtype=torch.float32)
        y_train_t = torch.tensor(data["y_train"], dtype=torch.float32)
        train_ds = TensorDataset(X_train_t, y_train_t)
        self.trainloader = DataLoader(train_ds, batch_size=NN_PARAMS["batch_size"], shuffle=True)

        X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)
        y_val_t = torch.tensor(data["y_val"], dtype=torch.float32)
        val_ds = TensorDataset(X_val_t, y_val_t)
        self.valloader = DataLoader(val_ds, batch_size=NN_PARAMS["batch_size"] * 2)

        self.n_train = len(y_train_t)
        self.n_val = len(y_val_t)
        
        fraud_sum = y_train_t.sum().item()
        fraud_mean = y_train_t.mean().item()
        print(f"  [{bank_id}] Ready — "
              f"{self.n_train} train samples, "
              f"{fraud_sum:.0f} fraud cases "
              f"({fraud_mean * 100:.2f}%)")

    def get_parameters(self, config):
        """Return current model parameters as list of numpy arrays."""
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        """Receive global model parameters from server."""
        if parameters:
            set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        """
        Train the model on local data.
        """
        self.round_num += 1
        self.set_parameters(parameters)

        print(f"\n  [{self.bank_id}] Round {self.round_num} — Training...")

        train_model(self.model, self.trainloader, NN_PARAMS["local_epochs"], self.device)

        # Evaluate to track local metrics
        train_metrics = evaluate_model(self.model, self.trainloader, self.device)
        val_metrics = evaluate_model(self.model, self.valloader, self.device)

        print(f"  [{self.bank_id}] Train: ", end="")
        print_metrics(train_metrics)
        print(f"  [{self.bank_id}] Val:   ", end="")
        print_metrics(val_metrics)

        return (
            self.get_parameters(config={}),
            self.n_train,
            {
                "train_auc": train_metrics["auc"],
                "train_loss": train_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model on local validation data."""
        self.set_parameters(parameters)

        metrics = evaluate_model(self.model, self.valloader, self.device)

        print(f"  [{self.bank_id}] Eval:  ", end="")
        print_metrics(metrics)

        return (
            metrics["loss"],
            self.n_val,
            {
                "auc": metrics["auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            },
        )


def start_client(bank_id: str):
    """Start a Flower client for the given bank."""
    client = FraudDetectionClient(bank_id)
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <bank_id>")
        print(f"  Available banks: {BANK_IDS}")
        sys.exit(1)

    bank_id = sys.argv[1]
    if bank_id not in BANK_IDS:
        print(f"Error: Unknown bank '{bank_id}'. Choose from {BANK_IDS}")
        sys.exit(1)

    start_client(bank_id)
