"""
Data Loader
============
Loads per-bank CSV splits into numpy arrays for PyTorch training.
Each bank only accesses its own data — no cross-bank leakage.
Also provides a global test set loader for unified evaluation.
"""

import os
import numpy as np
import pandas as pd
from config import BANKS_DIR, TARGET_COL


def load_bank_data(bank_id: str) -> dict:
    """
    Load train/val/test splits for a single bank.

    Returns:
        dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
              as numpy arrays.
    """
    bank_dir = os.path.join(BANKS_DIR, bank_id)

    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(bank_dir, f"{split_name}.csv")
        df = pd.read_csv(csv_path)

        y = df[TARGET_COL].values.astype(np.float32)
        X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)

        splits[f"X_{split_name}"] = X
        splits[f"y_{split_name}"] = y

    print(f"  [{bank_id}] Loaded — train: {splits['X_train'].shape}, "
          f"val: {splits['X_val'].shape}, "
          f"fraud rate: {splits['y_train'].mean():.4f}")

    return splits


def load_global_test_data() -> tuple:
    """
    Loads and concatenates the standard test.csv from all banks 
    to create a uniform global test set for true baseline comparison.
    """
    from config import BANK_IDS
    all_X = []
    all_y = []
    
    for bank_id in BANK_IDS:
        bank_dir = os.path.join(BANKS_DIR, bank_id)
        csv_path = os.path.join(bank_dir, "test.csv")
        df = pd.read_csv(csv_path)

        y = df[TARGET_COL].values.astype(np.float32)
        X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
        
        all_X.append(X)
        all_y.append(y)
        
    global_X = np.concatenate(all_X, axis=0)
    global_y = np.concatenate(all_y, axis=0)
    
    return global_X, global_y


def load_bank_metadata(bank_id: str) -> dict:
    """Load metadata.json for a bank."""
    import json
    meta_path = os.path.join(BANKS_DIR, bank_id, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)
