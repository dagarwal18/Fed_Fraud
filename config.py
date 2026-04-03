"""
Federated Learning Configuration
=================================
Central config for the FL fraud detection system.
All hyperparameters and paths live here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANKS_DIR = os.path.join(PROJECT_ROOT, "banks")
BANK_IDS = ["bank_A", "bank_B", "bank_C", "bank_D"]

# ──────────────────────────────────────────────
# Flower / FL settings
# ──────────────────────────────────────────────
SERVER_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 10
MIN_FIT_CLIENTS = 4          # all banks must participate
MIN_EVALUATE_CLIENTS = 4
MIN_AVAILABLE_CLIENTS = 4

# ──────────────────────────────────────────────
# PyTorch MLP hyperparameters (local training)
# ──────────────────────────────────────────────
NN_PARAMS = {
    "hidden_size_1": 128,
    "hidden_size_2": 64,
    "learning_rate": 0.001,
    "batch_size": 256,
    "local_epochs": 1,          # Number of passes over local data per FL round
    "pos_weight": 20.0,         # Handling class imbalance
}

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
TARGET_COL = "isFraud"
NUM_FEATURES = 371
