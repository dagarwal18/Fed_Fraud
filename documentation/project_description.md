# Federated Fraud Detection System — Project Description

> **Document purpose:** This document is the single, authoritative reference for the complete Federated Learning fraud detection project. It covers the motivation, architecture, data pipeline, every source file's internal workings, the aggregation strategies used, experimental results, and the current state of the repository. Reading this document alone should provide sufficient context to understand, modify, or extend the system.

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [System Architecture](#2-system-architecture)
3. [Why Federated Learning](#3-why-federated-learning)
4. [Technology Stack & Rationale](#4-technology-stack--rationale)
5. [Data Pipeline (Layer 1 — Pre-existing)](#5-data-pipeline-layer-1--pre-existing)
6. [Federated Learning System (Layer 2 — This Repo)](#6-federated-learning-system-layer-2--this-repo)
7. [Project Structure](#7-project-structure)
8. [File-by-File Reference](#8-file-by-file-reference)
9. [Federated Learning Flow Diagram](#9-federated-learning-flow-diagram)
10. [Aggregation Strategy: Krum](#10-aggregation-strategy-krum)
11. [Global Test Evaluation](#11-global-test-evaluation)
12. [Baseline Methodology](#12-baseline-methodology)
13. [Experimental Results](#13-experimental-results)
14. [Design Decisions & Trade-offs](#14-design-decisions--trade-offs)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)
16. [Current Repository State](#16-current-repository-state)

---

## 1. Problem Statement & Motivation

Financial fraud is an adversarial, cross-institutional problem. Fraudsters exploit the fact that individual banks see only their own slice of transaction data. A customer may establish a legitimate history at Bank A while simultaneously perpetrating fraud at Bank B. No single bank can detect this pattern because it requires knowledge of behaviour across institutions.

**The naive solution** — pool all banks' data centrally — is legally and ethically impossible due to:
- **Data privacy regulations** (GDPR, CCPA, India's DPDPA)
- **Competitive sensitivity** — banks will not share customer transaction logs with competitors
- **Liability** — centralised data creates a single catastrophic breach target

**The federated solution:** Each bank trains a fraud detection model locally on its own data. Only the trained model's weight updates (gradients) are sent to a central aggregation server. The server mathematically combines these updates into a single global model that benefits from the fraud patterns observed by *all* banks — without any bank's raw data ever leaving its premises.

---

## 2. System Architecture

The system is organised into three conceptual layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: MONITORING                           │
│                    (Planned — not yet implemented)                    │
│         Dashboard, drift detection, alert system                     │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│                  LAYER 2: FEDERATED LEARNING SYSTEM                  │
│                      (THIS REPOSITORY)                               │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │ Client A │    │ Client B │    │ Client C │    │ Client D │      │
│   │ (bank_A) │    │ (bank_B) │    │ (bank_C) │    │ (bank_D) │      │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘      │
│        │               │               │               │            │
│        └───────────────┬┴───────────────┘               │            │
│                        │                                │            │
│                        ▼                                │            │
│              ┌─────────────────┐                        │            │
│              │   FL Server     │◄───────────────────────┘            │
│              │ (Krum Aggreg.)  │                                     │
│              │ + Global Eval   │                                     │
│              └─────────────────┘                                     │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
┌──────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA PIPELINE                            │
│                  (Already completed externally)                      │
│   Raw IEEE-CIS Fraud Dataset → Cleaning → Feature Engineering       │
│   → Partitioning into 4 bank silos → train/val/test splits          │
└──────────────────────────────────────────────────────────────────────┘
```

### Communication Flow

All communication between clients and server uses **gRPC** over `127.0.0.1:8080` (localhost simulation). In a real deployment, each client would run on a separate machine within the bank's network, and the server would sit in a neutral trusted zone.

---

## 3. Why Federated Learning

| Approach | Data Privacy | Cross-Bank Patterns | Scalable |
|:---|:---:|:---:|:---:|
| Single bank model | ✅ | ❌ | ✅ |
| Centralised pooled model | ❌ | ✅ | ❌ |
| **Federated Learning** | **✅** | **✅** | **✅** |

Federated Learning is the only approach that simultaneously preserves data privacy, captures cross-institutional fraud patterns, and scales to any number of participants.

---

## 4. Technology Stack & Rationale

| Component | Technology | Why |
|:---|:---|:---|
| **FL Framework** | [Flower (flwr)](https://flower.dev/) v1.28 | Production-grade, framework-agnostic FL orchestration. Handles gRPC transport, client management, and provides built-in aggregation strategies (FedAvg, Krum, Trimmed Mean). |
| **Model** | PyTorch MLP | Neural network weights are continuous tensors — they can be algebraically averaged, which is the mathematical foundation of FedAvg and Krum. **XGBoost (previously used) was abandoned** because decision trees are discrete structures that cannot be averaged. |
| **Training** | PyTorch (torch) v2.11 | Industry-standard deep learning framework. Provides autograd, GPU acceleration, and the `state_dict()` API for seamless parameter extraction/injection. |
| **Metrics** | scikit-learn v1.8 | ROC-AUC, PR-AUC, precision, recall, F1 computation on numpy arrays. |
| **Data** | pandas + numpy | Efficient CSV ingestion and array manipulation. |

### Why PyTorch MLP Instead of XGBoost

The project originally used XGBoost. However, during implementation we discovered a fundamental incompatibility:

1. **FedAvg requires arithmetic averaging of model parameters.** Each client sends its weight tensor W_k to the server. The server computes W_global = Σ(n_k × W_k) / N_total. This works because neural network weights are real-valued matrices in continuous ℝ^n space.

2. **XGBoost models are collections of decision trees.** A tree is a discrete graph structure (node splits, thresholds, leaf values). You cannot "average" two trees — the result would not be a valid tree. The only XGBoost-compatible FL approach is **tree bagging** (concatenating trees from all clients), which is incompatible with robust aggregation strategies like Krum or Trimmed Mean.

3. **Krum specifically requires computing Euclidean distances between client updates** (flattened weight vectors). This is only meaningful for continuous parameters.

The PyTorch MLP was chosen as the simplest neural architecture that:
- Has continuous weights (enabling FedAvg / Krum / Trimmed Mean)
- Can handle the 371-dimensional tabular feature space
- Does not require complex architecture design (no convolutions, attention, etc.)

---

## 5. Data Pipeline (Layer 1 — Pre-existing)

The data pipeline was built separately (not part of this repository's scope). Its output is the `banks/` directory structure:

### Source Dataset
The IEEE-CIS Fraud Detection dataset (Kaggle) was used as the raw source. It contains ~590K e-commerce transactions with ~3.5% fraud rate.

### Preprocessing Steps (done externally)
1. **Cleaning:** Removed columns with >50% missing values, imputed remaining NaNs with medians
2. **Feature Engineering:** Created 371 features including:
   - Transaction amount log-transforms
   - Time-based features (hour, day-of-week, is_late_night)
   - Velocity features (tx_count_uid, time_diff_uid)
   - Ratio features (amt_z_uid, amt_ratio_uid)
   - Email domain matching
   - Frequency-encoded categorical variables
3. **Bank Partitioning:** Transactions were split into 4 Non-IID silos to simulate real banks with different customer demographics:

| Bank | Train Size | Val Size | Test Size | Fraud Rate |
|:---|---:|---:|---:|---:|
| bank_A | 15,019 | 3,219 | 3,219 | 2.27% |
| bank_B | 106,400 | 22,800 | 22,800 | 5.32% |
| bank_C | 267,371 | 57,294 | 57,294 | 2.71% |
| bank_D | 24,586 | 5,269 | 5,269 | 4.35% |

4. **Schema:** Every CSV has `isFraud` as the first column (binary target) followed by 371 numeric feature columns. All banks share the identical column schema.

### Non-IID Nature
The data partitioning is deliberately **non-IID** (non-independent and identically distributed). Different banks have:
- Different dataset sizes (15K vs 267K)
- Different fraud rates (2.27% vs 5.32%)
- Different transaction amount distributions

This realism is critical — real banks serve different customer segments. It also makes federated learning harder and more meaningful, since the federated model must generalise across these heterogeneous distributions.

---

## 6. Federated Learning System (Layer 2 — This Repo)

This is the core implementation. The FL system consists of:

1. **A Flower server** that orchestrates training rounds, aggregates model updates using Krum, and centrally evaluates the global model on the combined global test set.
2. **Four Flower clients** (one per bank) that each train a local PyTorch MLP on their own data and exchange only weight updates with the server.
3. **A baseline comparison** system that trains isolated per-bank models and evaluates them on the same global test set for fair comparison.

---

## 7. Project Structure

```
FED_FRAUD/
│
├── banks/                          # [DATA] Pre-processed per-bank datasets
│   ├── bank_A/
│   │   ├── train.csv               #   15,019 rows × 372 cols (isFraud + 371 features)
│   │   ├── val.csv                  #    3,219 rows
│   │   ├── test.csv                 #    3,219 rows
│   │   └── metadata.json           #   Bank statistics, medians, frequency maps
│   ├── bank_B/                      #   106,400 / 22,800 / 22,800 rows
│   ├── bank_C/                      #   267,371 / 57,294 / 57,294 rows
│   └── bank_D/                      #    24,586 /  5,269 /  5,269 rows
│
├── documentation/                  # [DOCS] Project documentation
│   ├── readme.md                    #   Repository overview and quickstart
│   └── project_description.md      #   This file — exhaustive technical reference
│
├── config.py                       # [CONFIG] Central configuration hub
├── data_loader.py                  # [DATA]   CSV → numpy loader, global test builder
├── model.py                        # [MODEL]  PyTorch MLP + FL parameter helpers
├── client.py                       # [FL]     Flower client implementation
├── server.py                       # [FL]     Flower server with Krum strategy
├── run_fl.py                       # [ORCH]   Single-command FL launcher
├── baseline.py                     # [EVAL]   Standalone per-bank training + comparison
├── utils.py                        # [UTIL]   Logging, metrics, result tracking
│
├── requirements.txt                # [DEPS]   Python package dependencies
├── fl_results.json                 # [OUTPUT] Round-by-round FL global test metrics
├── baseline_results.json           # [OUTPUT] Per-bank baseline global test metrics
│
├── venv/                           # [ENV]    Python virtual environment (not committed)
├── Cleaning_script/                # [LEGACY] External data cleaning scripts
├── _inspect.py                     # [DEBUG]  One-off data inspection script
└── _inspect_out.txt                # [DEBUG]  Output of inspection script
```

---

## 8. File-by-File Reference

### `config.py` — Central Configuration

**Purpose:** Single source of truth for all system parameters — paths, Flower settings, neural network hyperparameters, and data schema constants.

**Internal structure:**
```python
PROJECT_ROOT          # Absolute path to repo root (auto-detected)
BANKS_DIR             # Path to banks/ data directory
BANK_IDS              # ["bank_A", "bank_B", "bank_C", "bank_D"]

SERVER_ADDRESS        # "127.0.0.1:8080" — gRPC endpoint
NUM_ROUNDS            # 10 — federated training rounds
MIN_FIT_CLIENTS       # 4 — all banks must participate in every round
MIN_EVALUATE_CLIENTS  # 4
MIN_AVAILABLE_CLIENTS # 4

NN_PARAMS = {
    "hidden_size_1": 128,    # First hidden layer width
    "hidden_size_2": 64,     # Second hidden layer width
    "learning_rate": 0.001,  # Adam optimizer learning rate
    "batch_size": 256,       # Mini-batch size for DataLoader
    "local_epochs": 1,       # Passes over local data per FL round
    "pos_weight": 20.0,      # BCEWithLogitsLoss positive class weight
}

TARGET_COL            # "isFraud"
NUM_FEATURES          # 371
```

**Design rationale:**
- `local_epochs = 1` prevents client drift — too many local epochs cause clients to diverge from the global model, hurting aggregation quality.
- `pos_weight = 20.0` compensates for the ~3-5% fraud rate. Without this, the model would learn to predict "not fraud" for everything and still achieve 95%+ accuracy.
- `MIN_FIT_CLIENTS = 4` ensures every round aggregates knowledge from all banks. Allowing partial participation would weaken the Krum guarantee.

---

### `data_loader.py` — Data Loading

**Purpose:** Provides two functions for loading CSV data as numpy arrays.

**Functions:**

1. **`load_bank_data(bank_id: str) → dict`**
   - Reads `train.csv`, `val.csv`, `test.csv` from the specified bank directory.
   - Returns a dictionary with keys `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.
   - All values are cast to `np.float32` for PyTorch compatibility.
   - Prints a summary line with dataset shape and fraud rate.
   - **Privacy guarantee:** A client calling `load_bank_data("bank_A")` can only access bank_A's files. There is no cross-bank data access.

2. **`load_global_test_data() → (np.ndarray, np.ndarray)`**
   - Iterates over all 4 banks, reads each bank's `test.csv`, and concatenates them into a single global test set.
   - Returns `(X_global, y_global)` as numpy arrays.
   - Used by both `server.py` (centralized eval) and `baseline.py` (fair comparison).
   - Total global test set size: 3,219 + 22,800 + 57,294 + 5,269 = **88,582 samples**.

3. **`load_bank_metadata(bank_id: str) → dict`**
   - Reads the `metadata.json` file for a given bank (contains statistics, medians, frequency maps used during preprocessing).

---

### `model.py` — Neural Network & FL Helpers

**Purpose:** Defines the PyTorch model architecture and provides the serialisation interface required by Flower.

**Components:**

1. **`FraudMLP(nn.Module)`** — The neural network:
   ```
   Input (371) → Linear(371, 128) → ReLU → Dropout(0.2)
                → Linear(128, 64)  → ReLU → Dropout(0.2)
                → Linear(64, 1)    → [raw logits]
   ```
   - No sigmoid in `forward()` — `BCEWithLogitsLoss` combines sigmoid + cross-entropy for numerical stability.
   - Dropout(0.2) provides regularisation against overfitting on small bank datasets (bank_A has only 15K rows).
   - Total trainable parameters: 371×128 + 128 + 128×64 + 64 + 64×1 + 1 = **55,937**.

2. **`get_parameters(net) → list[np.ndarray]`**
   - Extracts all model weights and biases as a list of numpy arrays via `state_dict()`.
   - This is the format Flower uses to transmit parameters over gRPC.
   - For our 3-layer MLP, this returns 6 arrays: [W1, b1, W2, b2, W3, b3].

3. **`set_parameters(net, parameters)`**
   - Inverse of `get_parameters` — takes a list of numpy arrays from the server and loads them into the model.
   - Uses `collections.OrderedDict` to map parameter names to tensors.
   - `strict=True` ensures the parameter shapes match exactly.

4. **`train_model(net, trainloader, epochs, device)`**
   - Standard PyTorch training loop: zero_grad → forward → loss → backward → step.
   - Uses `BCEWithLogitsLoss(pos_weight=20.0)` for class-imbalanced binary classification.
   - Uses `Adam` optimizer with lr=0.001.

5. **`evaluate_model(net, testloader, device) → dict`**
   - Runs inference in `torch.no_grad()` mode.
   - Applies sigmoid to logits to get probability scores.
   - Computes and returns: AUC, PR-AUC, precision, recall, F1, loss.
   - Handles edge case where test set has only one class (returns AUC=0).

---

### `client.py` — Flower FL Client

**Purpose:** Implements the Flower `NumPyClient` interface. One instance represents one bank.

**Class: `FraudDetectionClient(fl.client.NumPyClient)`**

**`__init__(bank_id)`:**
- Detects GPU availability (falls back to CPU).
- Creates a fresh `FraudMLP` model on the device.
- Loads the bank's data via `load_bank_data()`.
- Converts numpy arrays to `TensorDataset` → `DataLoader` objects.
- Stores `trainloader` and `valloader` as instance attributes.
- Prints ready status with sample count and fraud percentage.

**`get_parameters(config)`:**
- Delegates to `model.get_parameters()` — returns current weights as numpy arrays.

**`set_parameters(parameters)`:**
- Delegates to `model.set_parameters()` — overwrites local model with server's global weights.

**`fit(parameters, config) → (parameters, num_examples, metrics)`:**
- **Step 1:** Receive and apply global model parameters from the server.
- **Step 2:** Train locally for `local_epochs` (1) on this bank's training data.
- **Step 3:** Evaluate on both train and val loaders for monitoring.
- **Step 4:** Return updated parameters, number of training examples (used by Krum for weighting), and local metrics dictionary.
- The `num_examples` return value is critical — the server uses it for weighted aggregation.

**`evaluate(parameters, config) → (loss, num_examples, metrics)`:**
- Receive global model, evaluate on local validation set.
- This is used for Flower's distributed evaluation (separate from the server's centralized global eval).

**`start_client(bank_id)`:**
- Entry point when `client.py` is run as a script.
- Creates the client instance and connects to the server via `fl.client.start_client()`.

**CLI usage:** `python client.py bank_A`

---

### `server.py` — Flower FL Server (Krum + Global Eval)

**Purpose:** Orchestrates the federated training process. Manages rounds, applies Krum aggregation, and evaluates the global model on the combined test set.

**`get_evaluate_fn(tracker) → Callable`:**
- Factory function that creates a closure for centralized evaluation.
- **On first call:** Loads the global test set (all 4 banks' test.csv concatenated = 88,582 samples) into a `DataLoader`.
- **Returns an `evaluate(server_round, parameters, config)` function** that:
  1. Creates a fresh `FraudMLP` on the device.
  2. Loads the provided global parameters into it.
  3. Runs `evaluate_model()` on the global test set.
  4. Logs round metrics to the `FLResultTracker`.
  5. Returns `(loss, metrics_dict)` to Flower.
- This function is called by Flower **after every aggregation round**, giving us a single authoritative metric for how well the global model generalises across all banks.

**`start_server()`:**
- Creates a `FLResultTracker` instance.
- Instantiates Flower's built-in `Krum` strategy with:
  - `num_malicious_clients=1` — tolerates 1 adversarial/outlier client
  - `num_clients_to_keep=1` — selects 1 closest client update
  - `fraction_fit=1.0` — sample all clients for training
  - `evaluate_fn=get_evaluate_fn(tracker)` — the centralized eval hook
- Starts the Flower server on `127.0.0.1:8080` with 10 rounds and 600s timeout.
- After training completes, saves results to `fl_results.json` and prints a summary.

---

### `run_fl.py` — Orchestration Script

**Purpose:** Launches the complete FL pipeline from a single terminal command.

**How it works:**
1. Detects the venv Python executable at `./venv/Scripts/python.exe`. Falls back to `sys.executable`.
2. **Starts `server.py`** as a background subprocess.
3. Waits 5 seconds for the server to bind its gRPC port.
4. **Starts 4 `client.py` processes** (one per bank) with 2-second stagger to avoid connection storms.
5. **Waits for the server process to exit** (it exits after all rounds complete).
6. **Waits for all client processes to exit** (60-second timeout, force-terminates if stuck).
7. Prints completion message.

**Why subprocesses?** The Flower framework's `start_server()` and `start_client()` are both blocking calls. Running them in the same process would require threads and complex coordination. Separate subprocesses are simpler and mirror real deployment topology.

---

### `baseline.py` — Single-Bank Baseline Comparison

**Purpose:** Establishes the single-institution performance floor to prove that federated learning adds value.

**Flow:**
1. Loads the **global test set** (all 4 banks' test.csv concatenated).
2. For each bank:
   a. Loads that bank's training data.
   b. Creates a fresh `FraudMLP`.
   c. Trains for 10 epochs (matching the FL system's 10 rounds of 1 local epoch each).
   d. Evaluates on the bank's local validation set (for sanity).
   e. Evaluates on the **global test set** (for fair comparison with FL).
3. Prints a summary table comparing all banks.
4. If `fl_results.json` exists, loads FL results and computes the improvement percentage.
5. Saves results to `baseline_results.json`.

**Why 10 epochs for baseline?** The FL system runs 10 rounds, each with 1 local epoch. So each bank's model sees its data 10 times in total during FL. The baseline matches this by training for 10 epochs directly. This ensures the comparison is fair in terms of total local compute.

**Why test on the global set?** A bank model that scores 0.90 AUC on its own test data may score only 0.40 on another bank's test data (different fraud patterns). Testing on the global set reveals how well a single-bank model generalises. This is the comparison that matters: can the federated model — which never sees any single bank's raw data — generalise better than any individual bank's model?

---

### `utils.py` — Utility Functions

**Purpose:** Shared logging, formatting, and result persistence utilities.

**Components:**

1. **`print_banner(text, char, width)`** — Prints visually distinct section headers for console output readability during long training runs.

2. **`print_metrics(metrics, prefix)`** — Formats a metrics dictionary as a pipe-delimited one-liner: `auc=0.7641 │ pr_auc=0.0961 │ precision=0.0692 │ ...`

3. **`save_round_results(results, output_path)`** — Writes a list of round dictionaries to JSON.

4. **`get_timestamp()`** — Returns current time as `YYYY-MM-DD HH:MM:SS` string.

5. **`FLResultTracker`** — Class that accumulates per-round metrics:
   - `log_round(round_num, global_metrics, client_metrics)` — appends a round entry
   - `save(path)` — persists to JSON
   - `print_summary()` — prints Round 1 vs Round N comparison with AUC delta

---

### `requirements.txt` — Dependencies

```
flwr          # Flower FL framework (server, client, strategies)
torch         # PyTorch (MLP model, training, inference)
scikit-learn  # Metrics (ROC-AUC, F1, precision, recall)
pandas        # CSV data loading
numpy         # Array operations
```

No version pins are specified — `pip` resolves the latest compatible versions. The system has been tested with: flwr 1.28.0, torch 2.11.0, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.

---

## 9. Federated Learning Flow Diagram

```
                            ┌─────────────────────┐
                            │   FL Server (Krum)   │
                            │                      │
                            │  1. Initialize       │
                            │     random model      │
                            │                      │
                            │  2. Broadcast global  │
                            │     weights to all    │
                            │     clients           │
                            └──────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Client A       │    │   Client B       │    │   Client C/D    │
    │   (bank_A)       │    │   (bank_B)       │    │   (bank_C/D)    │
    │                  │    │                  │    │                  │
    │  3. Receive       │    │  3. Receive       │    │  3. Receive       │
    │     global wts    │    │     global wts    │    │     global wts    │
    │                  │    │                  │    │                  │
    │  4. Train locally │    │  4. Train locally │    │  4. Train locally │
    │     on own data   │    │     on own data   │    │     on own data   │
    │     (1 epoch)     │    │     (1 epoch)     │    │     (1 epoch)     │
    │                  │    │                  │    │                  │
    │  5. Send updated  │    │  5. Send updated  │    │  5. Send updated  │
    │     weights back  │    │     weights back  │    │     weights back  │
    └───────┬──────────┘    └───────┬──────────┘    └───────┬──────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                            ┌─────────────────────┐
                            │   FL Server          │
                            │                      │
                            │  6. KRUM Aggregation: │
                            │     Select the client│
                            │     whose update is  │
                            │     closest to the   │
                            │     majority          │
                            │                      │
                            │  7. Global Eval:     │
                            │     Test aggregated   │
                            │     model on combined │
                            │     global test set   │
                            │     (88,582 samples)  │
                            │                      │
                            │  8. Log AUC, loss     │
                            │                      │
                            │  9. Repeat from (2)   │
                            │     for 10 rounds     │
                            └──────────────────────┘
```

### Round-by-Round Data Flow

For each of the 10 rounds:

1. **Server → Clients:** Global model parameters (6 numpy arrays, ~224KB total) sent via gRPC.
2. **Clients (parallel):** Each client independently:
   - Loads parameters into local `FraudMLP`
   - Runs 1 epoch of Adam-SGD over its training DataLoader (batch_size=256)
   - Returns updated parameters + training sample count + local metrics
3. **Server aggregation (Krum):** The server:
   - Receives 4 parameter sets
   - Computes pairwise Euclidean distances between all 4 updates
   - Selects the 1 update that minimises the sum of distances to its (4-1-1=2) closest neighbours
   - This update becomes the new global model
4. **Server evaluation:** Evaluates the new global model on the 88,582-sample global test set
5. **Loop:** Steps 1-4 repeat for 10 rounds

---

## 10. Aggregation Strategy: Krum

### What is Krum?

Krum is a **Byzantine-robust** aggregation rule designed to defend against adversarial or corrupted client updates. It was introduced by Blanchard et al. (2017) as an alternative to FedAvg that provides provable guarantees even when a fraction of clients are malicious.

### How Krum Works

Given `n` client updates {W₁, W₂, ..., Wₙ} and a tolerance of `f` malicious clients:

1. For each client update Wᵢ, compute its distance to every other update: d(Wᵢ, Wⱼ) = ‖Wᵢ − Wⱼ‖²
2. For each Wᵢ, compute the score s(Wᵢ) = sum of the (n − f − 1) smallest distances
3. Select the client with the **minimum score**: W_selected = argmin s(Wᵢ)

The intuition: a malicious client's update will be far from the honest majority. By selecting the update closest to its neighbours, Krum ignores outliers.

### Configuration in This Project

```python
fl.server.strategy.Krum(
    num_malicious_clients=1,   # Tolerate up to 1 adversarial client
    num_clients_to_keep=1,     # Select 1 best update (pure Krum, not Multi-Krum)
)
```

With 4 clients and f=1:
- Each client's score = sum of (4 − 1 − 1) = 2 closest distances
- The client whose 2 nearest neighbours are closest wins

### Why Krum Over FedAvg?

FedAvg computes a simple weighted average: W_global = Σ(nₖ × Wₖ) / N_total. This is optimal when all clients are honest and data is IID. However:
- A single poisoned client can arbitrarily shift the global model.
- With non-IID data (our case), some clients' updates may be "outliers" not due to malice but due to their unique fraud distributions.

Krum provides robustness in both cases by selecting updates that are "typical" of the majority.

---

## 11. Global Test Evaluation

### The Problem with Local Evaluation

If each bank's model is evaluated only on its own test set, the results are misleading:
- bank_A's model might score 0.81 AUC on bank_A's test data — but this says nothing about its ability to detect bank_B's fraud patterns.
- You cannot directly compare a bank_A AUC of 0.81 against a bank_B AUC of 0.90 because the test sets have different sizes, fraud rates, and difficulty levels.

### The Solution: Global Test Set

We concatenate all 4 banks' test.csv files into a single **global test set** of 88,582 samples. Both the federated model **and** each standalone baseline model are evaluated on this identical dataset.

This means:
- The federated model's AUC of **0.7641** is directly comparable to bank_A's baseline AUC of **0.4053** — both were computed on the exact same 88,582 samples.
- The comparison is unambiguous: the federated model generalises substantially better.

### Server-Side vs Client-Side Evaluation

The system performs **two types of evaluation**:

1. **Server-side centralized eval** (`evaluate_fn` in `server.py`): After each round, the server loads the global model and tests it on the full 88,582-sample global test set. This is the authoritative metric saved to `fl_results.json`.

2. **Client-side distributed eval** (`evaluate()` in `client.py`): Each client evaluates the global model on its own local validation set. This is useful for monitoring per-bank performance but is not used for the final comparison.

---

## 12. Baseline Methodology

The baseline trains 4 independent models, one per bank, with no knowledge sharing. Each model:
- Uses the same `FraudMLP` architecture (371 → 128 → 64 → 1)
- Trains for 10 epochs on its own bank's training data
- Is evaluated on the **same global test set** used by the federated model

### Why This Baseline Is Fair

| Parameter | Baseline | Federated |
|:---|:---|:---|
| Architecture | FraudMLP (55,937 params) | FraudMLP (55,937 params) |
| Total local epochs | 10 | 10 (1 per round × 10 rounds) |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss function | BCEWithLogitsLoss(pos_weight=20) | BCEWithLogitsLoss(pos_weight=20) |
| Evaluation set | Global test (88,582 samples) | Global test (88,582 samples) |

The only difference is **knowledge sharing**: federated clients receive aggregated global weights before each local epoch; baseline models train in complete isolation.

---

## 13. Experimental Results

### FL Results (fl_results.json) — Krum Aggregation, 10 Rounds

| Round | Global Test AUC | Global Test Loss |
|:---:|---:|---:|
| 0 (init) | 0.5981 | 126.53 |
| 1 | 0.3562 | 275.30 |
| 2 | 0.7628 | 8.55 |
| 3 | 0.7649 | 2.18 |
| 4 | 0.7557 | 1.26 |
| 5 | 0.3825 | 55.65 |
| 6 | 0.7599 | 1.18 |
| 7 | **0.7665** | 1.27 |
| 8 | 0.7595 | 1.30 |
| 9 | 0.7630 | 1.29 |
| 10 | **0.7641** | 1.38 |

**Observations:**
- Round 0 is the evaluation of a randomly initialised model (AUC ~0.60, near random).
- Rounds 1 and 5 show AUC dips — this is Krum selecting a client that happens to have a dominant local distribution (likely bank_A with only 15K samples and 2.27% fraud). This is expected behaviour with non-IID data.
- The model converges to a stable ~0.76 AUC by round 3 and sustains it.
- Final loss stabilises around 1.2-1.4, indicating no overfitting.

### Baseline Results (baseline_results.json) — Global Test AUC

| Bank | Train Size | Fraud Rate | Local Val AUC | Global Test AUC |
|:---|---:|---:|---:|---:|
| bank_A | 15,019 | 2.27% | 0.7695 | 0.4053 |
| bank_B | 106,400 | 5.32% | 0.7755 | 0.7630 |
| bank_C | 267,371 | 2.71% | 0.8005 | 0.7735 |
| bank_D | 24,586 | 4.35% | 0.7559 | 0.7653 |

**Key observation:** bank_A scores 0.77 on its *own* validation set but only **0.41** on the global test set. This dramatically illustrates the problem: a small bank with a low fraud rate learns a biased model that fails catastrophically on data from other banks.

### Head-to-Head Comparison

| Metric | Value |
|:---|---:|
| Average baseline AUC (global test) | **0.6768** |
| Best single-bank AUC (global test) — bank_C | **0.7735** |
| Federated model AUC (global test, round 10) | **0.7641** |
| **Improvement over average baseline** | **+12.9%** |

The federated model outperforms the average standalone bank by nearly 13%. It approaches (but does not exceed) the best single bank — this is expected because:
1. bank_C has 267K samples (6x more than the next largest), so its standalone model already captures substantial diversity.
2. Krum's selection mechanism is conservative — it protects against outliers but may not optimally blend knowledge from all clients like FedAvg would.

---

## 14. Design Decisions & Trade-offs

### Decision 1: MLP Architecture

**Chose:** 3-layer MLP (371 → 128 → 64 → 1) with ReLU and Dropout.

**Why not deeper/wider?** With non-IID federated data and only 1 local epoch per round, a complex model would overfit to each bank's local patterns before the server can aggregate. A shallow MLP has fast convergence and is robust to the parameter averaging inherent in federated learning.

**Why not a linear model?** Linear models cannot capture the non-linear interactions between features (e.g., transaction amount × time-of-day patterns in fraud).

### Decision 2: BCEWithLogitsLoss + pos_weight

**Chose:** Weighted binary cross-entropy with `pos_weight=20.0`.

**Alternatives considered:**
- Focal Loss — more complex, harder to tune in federated setting
- SMOTE oversampling — would create synthetic samples, problematic with heterogeneous bank distributions
- Class-weighted sampling — requires per-bank tuning

`pos_weight=20.0` was chosen as a simple, effective approach: it roughly corresponds to 1/fraud_rate, telling the model that missing a fraud case is 20x worse than a false alarm.

### Decision 3: Krum over FedAvg

**Chose:** Krum with 1 malicious tolerance.

**Rationale:** With 4 banks of vastly different sizes (15K to 267K), weighted FedAvg would let bank_C dominate the global model (its update carries ~65% of the total weight). Krum treats all updates equally and selects based on consensus, providing a more balanced representation.

### Decision 4: 1 Local Epoch per Round

**Chose:** `local_epochs=1` with 10 FL rounds.

**Why not more local epochs?** This is the classic "client drift" problem. With many local epochs, each client's model drifts far from the global model toward its own local optimum. When the server aggregates these divergent models, the result is poor. 1 epoch per round minimises drift while still making meaningful local progress.

### Decision 5: Subprocess-Based Orchestration

**Chose:** `run_fl.py` launches server + 4 clients as 5 separate OS processes.

**Why not threads?** Flower's `start_server()` and `start_client()` are blocking calls that internally manage their own event loops. Running them in threads within a single process would require monkey-patching and is fragile. Separate processes are clean, debuggable, and mirror actual network deployment.

---

## 15. Known Limitations & Future Work

### Current Limitations

1. **No differential privacy (DP):** Model weight updates are sent in plaintext. A sophisticated attacker could potentially reconstruct training data characteristics from these updates. Adding DP noise (e.g., `DPFedAvgAdaptive` strategy in Flower) would mitigate this.

2. **Simulated network topology:** All processes run on localhost. Real deployment would require TLS-encrypted gRPC channels and proper authentication.

3. **No model persistence:** The global model exists only in-memory during training. After `run_fl.py` completes, the trained model is lost. A production system would save the final `state_dict` to disk.

4. **Static data:** The system trains on a fixed dataset. Real fraud detection requires continuous learning as fraud patterns evolve.

5. **AUC plateau:** The model converges to ~0.76 AUC by round 3 and shows minimal improvement thereafter. This could be addressed by:
   - Learning rate scheduling
   - Increasing model capacity
   - More rounds
   - Multi-Krum (keeping >1 clients per round)

### Future Work (Layer 3 — Monitoring)

- **Dashboard:** Real-time web dashboard showing per-round AUC, per-client metrics, and convergence graphs.
- **Drift Detection:** Monitor feature distributions and model performance for concept drift.
- **Alert System:** Automated alerts when fraud rate spikes or model performance degrades.
- **Model Versioning:** Save and version global model checkpoints.

---

## 16. Current Repository State

**As of:** 2026-04-03

| Component | Status |
|:---|:---|
| Data pipeline (Layer 1) | ✅ Complete — 4 banks with train/val/test splits |
| PyTorch MLP model | ✅ Complete — 55,937 parameters, BCEWithLogitsLoss |
| Flower FL server | ✅ Complete — Krum aggregation + centralized global eval |
| Flower FL clients | ✅ Complete — 4 clients, 1 per bank |
| FL orchestrator | ✅ Complete — single-command launcher |
| Baseline comparison | ✅ Complete — global test evaluation |
| Results verified | ✅ Complete — FL outperforms avg baseline by +12.9% |
| Documentation | ✅ Complete — readme + project description |
| Monitoring (Layer 3) | ❌ Not started |

**Latest run results are saved in:**
- `fl_results.json` — 11 entries (round 0 through 10), final AUC = 0.7641
- `baseline_results.json` — 4 entries (one per bank), evaluated on global test set

The repository is in a clean, working state. Running `python run_fl.py` followed by `python baseline.py` will reproduce the documented results.
