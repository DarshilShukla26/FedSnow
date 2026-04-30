"""
Evaluate the global federated model from Snowflake.

Usage:
    python evaluation/evaluate_global_model.py              # latest round
    python evaluation/evaluate_global_model.py --round 5   # specific round
    python evaluation/evaluate_global_model.py --compare   # round 1 vs round 10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BATCH_SIZE, DATA_DIR
from clients.base_client import FedMLP, BaseClient
from sf.download_global_model import fetch_global_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_loader() -> DataLoader:
    path = os.path.join(DATA_DIR, "test_set.csv")
    df   = pd.read_csv(path)
    X    = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32)
    y    = torch.tensor(df["label"].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)


def evaluate_model(model: FedMLP, loader: DataLoader) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b).argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_b.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred),
    }


def load_model_for_round(round_id: int | None) -> tuple[FedMLP, dict]:
    weights, meta = fetch_global_weights(round_id)
    proxy = BaseClient.__new__(BaseClient)
    proxy.model = FedMLP()
    proxy.set_weights(weights)
    return proxy.model, meta


def print_report(metrics: dict, label: str) -> None:
    cm = metrics["confusion"]
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate FedSnow global model")
    parser.add_argument("--round", type=int, default=None,
                        help="Round ID to evaluate (default: latest)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare round 1 vs round 10")
    args = parser.parse_args()

    test_loader = load_test_loader()
    print(f"Test set size: {len(test_loader.dataset)} samples")

    if args.compare:
        model1, meta1 = load_model_for_round(1)
        model10, meta10 = load_model_for_round(10)

        m1  = evaluate_model(model1, test_loader)
        m10 = evaluate_model(model10, test_loader)

        print_report(m1,  f"Round 1  (global acc={meta1['global_accuracy']:.4f})")
        print_report(m10, f"Round 10 (global acc={meta10['global_accuracy']:.4f})")

        print(f"\n{'═'*40}")
        print("  Improvement (Round 1 → Round 10)")
        print(f"{'═'*40}")
        print(f"  Accuracy  Δ: {m10['accuracy']  - m1['accuracy']:+.4f}")
        print(f"  Precision Δ: {m10['precision'] - m1['precision']:+.4f}")
        print(f"  Recall    Δ: {m10['recall']    - m1['recall']:+.4f}")
        print(f"  F1        Δ: {m10['f1']        - m1['f1']:+.4f}")
    else:
        model, meta = load_model_for_round(args.round)
        metrics     = evaluate_model(model, test_loader)
        label       = (
            f"Round {meta['round_id']} "
            f"(global acc={meta['global_accuracy']:.4f})"
        )
        print_report(metrics, label)


if __name__ == "__main__":
    main()
