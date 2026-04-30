"""
FedSnow federation loop.

Usage:
    python federation/run_federation.py
    python federation/run_federation.py --rounds 3       # quick test run
    python federation/run_federation.py --skip-cortex    # skip drift analysis
    python federation/run_federation.py --skip-upload    # local-only dry run
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NUM_ROUNDS, BATCH_SIZE, DATA_DIR
from clients import ALL_CLIENTS
from clients.base_client import BaseClient, FedMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_set() -> DataLoader:
    path = os.path.join(DATA_DIR, "test_set.csv")
    df   = pd.read_csv(path)
    X    = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32)
    y    = torch.tensor(df["label"].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)


def _evaluate_on_test(model: FedMLP, test_loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds    = model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total   += len(y_b)
    return correct / total


def _broadcast(clients: list, weights: OrderedDict) -> None:
    for c in clients:
        c.set_weights(weights)


def _init_weights() -> OrderedDict:
    proxy = BaseClient.__new__(BaseClient)
    proxy.model = FedMLP()
    return proxy.get_weights()


def _local_fedavg(client_metrics: list) -> OrderedDict:
    """Fallback FedAvg for --skip-upload dry runs (no Snowflake required)."""
    total = sum(m["num_samples"] for m in client_metrics)
    agg: OrderedDict = OrderedDict()
    for m in client_metrics:
        frac = m["num_samples"] / total
        for layer, arr in m["weights"].items():
            scaled = np.array(arr) * frac
            agg[layer] = agg[layer] + scaled if layer in agg else scaled
    return agg


def _l2_divergence(client_w: OrderedDict, global_w: OrderedDict) -> float:
    total = 0.0
    for layer in global_w:
        if layer in client_w:
            diff   = np.array(client_w[layer]) - np.array(global_w[layer])
            total += float(np.linalg.norm(diff))
    return total


# ---------------------------------------------------------------------------
# Main federation loop
# ---------------------------------------------------------------------------

def run(num_rounds: int, skip_cortex: bool, skip_upload: bool) -> None:
    # Lazy-import Snowflake modules only when a real run is requested
    if not skip_upload:
        from sf.upload_weights import upload_weights, upload_round_metric
        from sf.fedavg_procedure import call_fedavg
        from sf.download_global_model import fetch_global_weights

    if not skip_upload and not skip_cortex:
        from cortex.drift_analysis import run_drift_analysis

    print("=" * 60)
    print("  FedSnow — Federated Learning Simulator")
    print(f"  Rounds: {num_rounds} | Clients: {len(ALL_CLIENTS)}")
    print(f"  Mode: {'local dry-run' if skip_upload else 'Snowflake'}")
    print("=" * 60)

    clients     = [Cls() for Cls in ALL_CLIENTS]
    dataloaders = {c.client_id: c.get_dataloader() for c in clients}
    test_loader = _load_test_set()

    global_weights = _init_weights()
    history: list[tuple[int, float]] = []

    for round_id in range(1, num_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"  Round {round_id}/{num_rounds}")
        print(f"{'─'*60}")

        # 1. Broadcast global weights
        _broadcast(clients, global_weights)

        # 2. Local training on private shards
        round_metrics: list[dict] = []
        for client in clients:
            dl      = dataloaders[client.client_id]
            metrics = client.train_local(dl)
            w_json  = client.serialize_weights()

            print(
                f"  [{client.client_name:<8}] "
                f"acc={metrics['accuracy']:.4f}  "
                f"loss={metrics['loss']:.4f}  "
                f"n={metrics['num_samples']}"
            )

            # 3. Upload weights to Snowflake
            if not skip_upload:
                upload_weights(
                    round_id=round_id,
                    client_id=client.client_id,
                    client_name=client.client_name,
                    num_samples=metrics["num_samples"],
                    weights_json=w_json,
                    local_loss=metrics["loss"],
                    local_accuracy=metrics["accuracy"],
                )

            round_metrics.append({
                "client_id":      client.client_id,
                "client_name":    client.client_name,
                "local_accuracy": metrics["accuracy"],
                "local_loss":     metrics["loss"],
                "num_samples":    metrics["num_samples"],
                "weights":        client.get_weights(),
            })

        # 4 & 5. Aggregate + download global model
        if not skip_upload:
            print(f"\n  Calling FEDAVG_AGGREGATE({round_id})...")
            sp_result = call_fedavg(round_id)
            print(f"  SP result: {sp_result}")
            global_weights, _ = fetch_global_weights(round_id)
        else:
            global_weights = _local_fedavg(round_metrics)

        # 6. Evaluate global model on held-out test set
        proxy = BaseClient.__new__(BaseClient)
        proxy.model = FedMLP()
        proxy.set_weights(global_weights)
        global_accuracy = _evaluate_on_test(proxy.model, test_loader)

        print(f"\n  Global accuracy (test set): {global_accuracy:.4f}")
        history.append((round_id, global_accuracy))

        # 7. Log ROUND_METRICS
        if not skip_upload:
            for m in round_metrics:
                div = _l2_divergence(m["weights"], global_weights)
                upload_round_metric(
                    round_id=round_id,
                    client_id=m["client_id"],
                    client_name=m["client_name"],
                    local_accuracy=m["local_accuracy"],
                    local_loss=m["local_loss"],
                    global_accuracy=global_accuracy,
                    weight_divergence=div,
                    num_samples=m["num_samples"],
                )

        # 8. Cortex drift analysis
        if not skip_upload and not skip_cortex:
            print("  Running Cortex drift analysis...")
            try:
                run_drift_analysis(
                    round_id=round_id,
                    client_metrics=round_metrics,
                    global_weights=global_weights,
                )
            except Exception as exc:
                print(f"  [WARN] Cortex analysis failed: {exc}")

    # Final summary
    print(f"\n{'='*60}")
    print("  Federation complete — Round summary")
    print(f"  {'Round':<8} {'Global Accuracy'}")
    for rid, acc in history:
        bar = "█" * int(acc * 40)
        print(f"  {rid:<8} {acc:.4f}  {bar}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSnow federation loop")
    parser.add_argument(
        "--rounds", type=int, default=NUM_ROUNDS,
        help=f"Number of federation rounds (default {NUM_ROUNDS})"
    )
    parser.add_argument(
        "--skip-cortex", action="store_true",
        help="Skip Cortex drift analysis"
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Local dry run — no Snowflake calls"
    )
    args = parser.parse_args()

    run(
        num_rounds=args.rounds,
        skip_cortex=args.skip_cortex,
        skip_upload=args.skip_upload,
    )
