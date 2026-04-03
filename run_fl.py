"""
FL Runner
==========
Single-script launcher that starts the server and all 4 clients
in separate processes. No need to open multiple terminals.

Usage:
    python run_fl.py

Note: Uses the venv Python if available.
"""

import subprocess
import sys
import time
import os

from config import BANK_IDS, NUM_ROUNDS, SERVER_ADDRESS


def main():
    print("=" * 60)
    print("  FEDERATED FRAUD DETECTION — Launch Sequence")
    print("=" * 60)
    print(f"\n  Banks:   {BANK_IDS}")
    print(f"  Rounds:  {NUM_ROUNDS}")
    print(f"  Server:  {SERVER_ADDRESS}")
    print()

    # Use venv Python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(python):
        python = sys.executable

    # 1. Start server in background (inherit stderr for live output)
    print("[1/5] Starting FL server...")
    server_proc = subprocess.Popen(
        [python, os.path.join(script_dir, "server.py")],
        cwd=script_dir,
    )
    time.sleep(5)  # Give server time to bind the port

    # 2. Start all clients
    client_procs = []
    for i, bank_id in enumerate(BANK_IDS):
        print(f"[{i+2}/5] Starting client: {bank_id}...")
        proc = subprocess.Popen(
            [python, os.path.join(script_dir, "client.py"), bank_id],
            cwd=script_dir,
        )
        client_procs.append((bank_id, proc))
        time.sleep(2)  # Stagger client connections

    print("\n" + "=" * 60)
    print("  All processes launched. Waiting for FL to complete...")
    print("  (This may take several minutes depending on data size)")
    print("=" * 60 + "\n")

    # 3. Wait for server to finish (it exits after NUM_ROUNDS)
    server_proc.wait()
    print("\n  Server process exited.")

    # 4. Wait for all clients to finish
    for bank_id, proc in client_procs:
        try:
            proc.wait(timeout=60)
            print(f"  Client {bank_id} exited.")
        except subprocess.TimeoutExpired:
            proc.terminate()
            print(f"  Client {bank_id} terminated (timeout).")

    print("\n" + "=" * 60)
    print("  FL TRAINING COMPLETE")
    print("=" * 60)
    print("\n  Check fl_results.json for detailed round-by-round metrics.")
    print("  Run 'python baseline.py' to compare against single-bank baseline.\n")


if __name__ == "__main__":
    main()
