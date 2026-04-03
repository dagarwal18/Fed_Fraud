"""
Baseline Comparison (PyTorch)
=============================
Trains a single-bank MLP model on each bank independently,
then compares against the federated model's performance on a 
GLOBAL test set (combined test sets of all banks).
"""

import json
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import BANK_IDS, NN_PARAMS
from data_loader import load_bank_data, load_global_test_data
from model import FraudMLP, train_model, evaluate_model
from utils import print_banner, print_metrics


def train_single_bank(bank_id: str, global_testloader: DataLoader) -> dict:
    """Train on a single bank's data, but test on the global test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bank_data(bank_id)

    X_train_t = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(data["y_train"], dtype=torch.float32)
    train_ds = TensorDataset(X_train_t, y_train_t)
    trainloader = DataLoader(train_ds, batch_size=NN_PARAMS["batch_size"], shuffle=True)

    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val_t = torch.tensor(data["y_val"], dtype=torch.float32)
    val_ds = TensorDataset(X_val_t, y_val_t)
    valloader = DataLoader(val_ds, batch_size=NN_PARAMS["batch_size"] * 2)

    model = FraudMLP().to(device)
    
    # Train for more epochs locally for the standalone baseline since 
    # the federated clients will train iteratively for 10 FL rounds.
    train_model(model, trainloader, epochs=10, device=device)

    # Local validation performance
    val_metrics = evaluate_model(model, valloader, device)
    # Global test performance
    test_metrics = evaluate_model(model, global_testloader, device)

    return {
        "bank_id": bank_id,
        "train_size": len(y_train_t),
        "fraud_rate": float(y_train_t.mean()),
        "val": val_metrics,
        "test": test_metrics,
    }


def main():
    print_banner("BASELINE: Single-Bank Model Performance (PyTorch MLP)", char="█")
    
    # 1. Load GLOBAL test set
    print("  Loading GLOBAL test set...")
    g_X, g_y = load_global_test_data()
    g_X_t = torch.tensor(g_X, dtype=torch.float32)
    g_y_t = torch.tensor(g_y, dtype=torch.float32)
    global_test_ds = TensorDataset(g_X_t, g_y_t)
    global_testloader = DataLoader(global_test_ds, batch_size=NN_PARAMS["batch_size"] * 2)
    print(f"  [Global Test] {len(g_y)} samples with {g_y.mean()*100:.2f}% fraud rate.")

    results = []
    for bank_id in BANK_IDS:
        print(f"\n  Training {bank_id} baseline...")
        result = train_single_bank(bank_id, global_testloader)
        results.append(result)

        print(f"  [{bank_id}] Local Val metrics:")
        print_metrics(result["val"], prefix="  ")
        print(f"  [{bank_id}] Global Test metrics:")
        print_metrics(result["test"], prefix="  ")

    # Summary comparison
    print_banner("BASELINE SUMMARY (Evaluated on Global Test Set)")
    print(f"  {'Bank':<8} {'Train':>7} {'Fraud%':>7} {'Local Val AUC':>13} "
          f"{'Global Test AUC':>15} {'Global F1':>9} {'Global Recall':>13}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*13} {'─'*15} {'─'*9} {'─'*13}")

    for r in results:
        print(f"  {r['bank_id']:<8} {r['train_size']:>7} "
              f"{r['fraud_rate']*100:>6.2f}% "
              f"{r['val']['auc']:>13.4f} "
              f"{r['test']['auc']:>15.4f} "
              f"{r['test']['f1']:>9.4f} "
              f"{r['test']['recall']:>13.4f}")

    # Load FL results if available
    fl_results_path = "fl_results.json"
    if os.path.exists(fl_results_path):
        print_banner("FEDERATED vs BASELINE COMPARISON")
        with open(fl_results_path) as f:
            fl_data = json.load(f)

        if fl_data:
            last_round = fl_data[-1]
            fl_auc = last_round["global"].get("auc", 0)

            avg_baseline_auc = np.mean([r["test"]["auc"] for r in results])
            best_baseline_auc = max(r["test"]["auc"] for r in results)

            print(f"  Avg baseline Global AUC (single bank):  {avg_baseline_auc:.4f}")
            print(f"  Best baseline Global AUC (single bank): {best_baseline_auc:.4f}")
            print(f"  Federated Global AUC (round {last_round['round']}):     {fl_auc:.4f}")
            print()

            if fl_auc > avg_baseline_auc:
                improvement = (fl_auc - avg_baseline_auc) / avg_baseline_auc * 100
                print(f"  ✓ Federated model improves over avg baseline by {improvement:.1f}%")
            else:
                print(f"  ⚠ Federated model did not outperform avg baseline")
    else:
        print(f"\n  No FL results found. Run 'python run_fl.py' first")
        print(f"  to compare federated vs baseline performance.")

    # Save baseline results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Baseline results saved to baseline_results.json")


if __name__ == "__main__":
    main()
