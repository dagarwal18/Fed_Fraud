# Federated Fraud Detection System

A privacy-preserving, cross-bank fraud detection system built with **Federated Learning**. Multiple simulated banks collaboratively train a shared neural network model — without ever exchanging raw transaction data — using the [Flower](https://flower.dev/) framework and **PyTorch**.

## Overview

Financial fraud patterns often span multiple institutions. A single bank's data may not capture the full spectrum of fraudulent behaviour. This system solves that problem by allowing 4 banks to jointly train a global fraud detection model while keeping each bank's data strictly local. The server aggregates only model weight updates (not data) using the **Krum** Byzantine-robust aggregation strategy.

## Project Structure

```
FED_FRAUD/
├── banks/                   # Pre-processed per-bank datasets
│   ├── bank_A/              #   train.csv, val.csv, test.csv, metadata.json
│   ├── bank_B/
│   ├── bank_C/
│   └── bank_D/
├── documentation/           # This folder — docs and project description
│   ├── readme.md
│   └── project_description.md
├── config.py                # Central configuration (paths, FL settings, hyperparams)
├── data_loader.py           # Per-bank data loading + global test set builder
├── model.py                 # PyTorch MLP definition + FL parameter helpers
├── client.py                # Flower FL client — one per bank
├── server.py                # Flower FL server — Krum aggregation + global eval
├── run_fl.py                # Single-command orchestrator (server + 4 clients)
├── baseline.py              # Standalone single-bank training for comparison
├── utils.py                 # Logging, metrics formatting, result tracking
├── requirements.txt         # Python dependencies
├── fl_results.json          # Output: round-by-round FL global test performance
└── baseline_results.json    # Output: per-bank standalone global test performance
```

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for faster training

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Federated Learning pipeline

This single command launches the FL server and all 4 bank clients as background processes:

```bash
python run_fl.py
```

The server runs 10 rounds of Krum-aggregated federated training. After each round it centrally evaluates the global model on the **combined global test set** (all banks' test data merged). Results are saved to `fl_results.json`.

### 4. Run the Baseline comparison

This trains an independent model per bank and evaluates each on the same global test set:

```bash
python baseline.py
```

Results are saved to `baseline_results.json` and a comparison against `fl_results.json` is printed.

### 5. Interpret the results

| Metric | Value |
|:---|---:|
| Avg single-bank AUC (global test) | 0.6768 |
| Best single-bank AUC (global test) | 0.7735 |
| **Federated model AUC (global test)** | **0.7641** |
| **Improvement over avg baseline** | **+12.9%** |

The federated model outperforms the average standalone bank by ~13%, demonstrating that cross-bank knowledge fusion works without sharing raw data.

## Key Design Decisions

- **PyTorch MLP** over XGBoost — Neural network weights live in continuous space, enabling true algebraic aggregation (FedAvg, Krum, Trimmed Mean). Tree-based models cannot be averaged.
- **Krum aggregation** — Byzantine-robust strategy that tolerates 1 malicious/outlier client out of 4; selects the update closest to the bulk of clients.
- **Centralized server-side evaluation** — The server evaluates the global model on the full combined test set after every round, giving a single authoritative performance metric.
- **BCEWithLogitsLoss with pos_weight** — Handles the severe class imbalance (~3-5% fraud) without resampling.

## Further Documentation

See [project_description.md](project_description.md) for an exhaustive technical deep-dive covering architecture, data flow, file-by-file internals, and design rationale.
