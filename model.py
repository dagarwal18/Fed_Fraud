"""
Model Definition
=================
PyTorch Multilayer Perceptron (MLP) for Fraud Detection.
Includes federated learning parameter extraction tools.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score
)
from config import NN_PARAMS, NUM_FEATURES

class FraudMLP(nn.Module):
    """Simple Neural Network for tabular fraud data."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, NN_PARAMS["hidden_size_1"]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(NN_PARAMS["hidden_size_1"], NN_PARAMS["hidden_size_2"]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(NN_PARAMS["hidden_size_2"], 1),
            # Note: No Sigmoid here because we use BCEWithLogitsLoss for numerical stability
        )
        
    def forward(self, x):
        return self.net(x)

def get_parameters(net: nn.Module) -> list[np.ndarray]:
    """Extract model weights as a list of NumPy arrays (for Flower)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

import collections

def set_parameters(net: nn.Module, parameters: list[np.ndarray]):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train_model(net: nn.Module, trainloader, epochs: int, device: torch.device):
    """Train the network on the training set."""
    # Use BCEWithLogitsLoss with pos_weight for class imbalance
    pos_weight = torch.tensor([NN_PARAMS["pos_weight"]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=NN_PARAMS["learning_rate"])
    
    net.train()
    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = net(X_batch).squeeze(1) # [batch_size]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

def evaluate_model(net: nn.Module, testloader, device: torch.device) -> dict:
    """Evaluate the network on the validation/test set."""
    pos_weight = torch.tensor([NN_PARAMS["pos_weight"]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    net.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = net(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            # Apply sigmoid to get probabilities for metrics
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    y_prob = np.concatenate(all_probs)
    y = np.concatenate(all_targets)
    y_pred = (y_prob >= 0.5).astype(int)
    
    avg_loss = total_loss / len(y)
    
    if len(np.unique(y)) < 2:
        auc = 0.0
        pr_auc = 0.0
    else:
        auc = roc_auc_score(y, y_prob)
        pr_auc = average_precision_score(y, y_prob)
        
    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "loss": float(avg_loss),
    }
