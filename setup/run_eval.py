import sys, os
sys.path.insert(0, "/opt/anaconda3/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from config import BATCH_SIZE, DATA_DIR
from clients.base_client import FedMLP, BaseClient
from sf.download_global_model import fetch_global_weights


def load_test_loader():
    path = os.path.join(DATA_DIR, "test_set.csv")
    df = pd.read_csv(path)
    X = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE)


def evaluate_model(model, loader):
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


def load_model(round_id):
    weights, meta = fetch_global_weights(round_id)
    proxy = BaseClient.__new__(BaseClient)
    proxy.model = FedMLP()
    proxy.set_weights(weights)
    return proxy.model, meta


test_loader = load_test_loader()
print(f"Test set: {len(test_loader.dataset)} samples\n")

results = {}
for rid in [1, 3, 5, 7, 10]:
    model, meta = load_model(rid)
    m = evaluate_model(model, test_loader)
    results[rid] = m
    print(f"Round {rid:>2} | acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}")

# Detailed report for round 1 and round 10
for rid, label in [(1, "Round 1 (start)"), (10, "Round 10 (final)")]:
    m = results[rid]
    cm = m["confusion"]
    print(f"\n{'─'*45}")
    print(f"  {label}")
    print(f"{'─'*45}")
    print(f"  Accuracy : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}")
    print(f"  F1 Score : {m['f1']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

m1  = results[1]
m10 = results[10]
print(f"\n{'═'*45}")
print(f"  Improvement: Round 1 → Round 10")
print(f"{'═'*45}")
print(f"  Accuracy  Δ: {m10['accuracy']  - m1['accuracy']:+.4f}  ({(m10['accuracy']-m1['accuracy'])*100:+.1f}%)")
print(f"  Precision Δ: {m10['precision'] - m1['precision']:+.4f}")
print(f"  Recall    Δ: {m10['recall']    - m1['recall']:+.4f}")
print(f"  F1        Δ: {m10['f1']        - m1['f1']:+.4f}")
print(f"{'═'*45}")
