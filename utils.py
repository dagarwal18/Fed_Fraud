"""
Utility Functions
==================
Logging, metrics formatting, and result tracking for the FL system.
"""

import json
import os
import time
from datetime import datetime


def print_banner(text: str, char: str = "═", width: int = 60):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print a metrics dictionary."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    line = " │ ".join(parts)
    if prefix:
        print(f"  {prefix}: {line}")
    else:
        print(f"  {line}")


def save_round_results(results: list, output_path: str = "fl_results.json"):
    """Save accumulated FL round results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class FLResultTracker:
    """Tracks metrics across FL rounds for later analysis."""

    def __init__(self):
        self.rounds = []

    def log_round(self, round_num: int, global_metrics: dict,
                  client_metrics: dict = None):
        entry = {
            "round": round_num,
            "timestamp": get_timestamp(),
            "global": global_metrics,
        }
        if client_metrics:
            entry["clients"] = client_metrics
        self.rounds.append(entry)

    def save(self, path: str = "fl_results.json"):
        save_round_results(self.rounds, path)

    def print_summary(self):
        if not self.rounds:
            print("  No rounds recorded.")
            return
        print_banner("FL Training Summary")
        print(f"  Total rounds: {len(self.rounds)}")
        first = self.rounds[0]["global"]
        last = self.rounds[-1]["global"]
        print(f"  Round 1  → AUC: {first.get('auc', 0):.4f}, "
              f"Loss: {first.get('loss', 0):.4f}")
        print(f"  Round {len(self.rounds)} → AUC: {last.get('auc', 0):.4f}, "
              f"Loss: {last.get('loss', 0):.4f}")
        auc_delta = last.get('auc', 0) - first.get('auc', 0)
        print(f"  AUC improvement: {auc_delta:+.4f}")
