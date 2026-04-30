"""
Generate 4 non-IID data shards for the FedSnow clients.
Each shard simulates a private dataset that never leaves the client.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CLIENTS, TOTAL_SAMPLES, TEST_SAMPLES, DATA_DIR


def make_noniid_shard(X: np.ndarray, y: np.ndarray, class_0_ratio: float,
                      n_samples: int, rng: np.random.Generator) -> tuple:
    """Draw n_samples from (X, y) with a target class-0 ratio."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0 = int(n_samples * class_0_ratio)
    n1 = n_samples - n0

    chosen0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
    chosen1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
    chosen  = np.concatenate([chosen0, chosen1])
    rng.shuffle(chosen)

    return X[chosen], y[chosen]


def generate_shards(seed: int = 42) -> dict:
    """Create and save all client shards plus a held-out test set."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    X, y = make_classification(
        n_samples=TOTAL_SAMPLES + TEST_SAMPLES,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=seed,
    )

    # Reserve a held-out test set (balanced)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_SAMPLES, stratify=y, random_state=seed
    )

    test_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(20)])
    test_df["label"] = y_test
    test_path = os.path.join(DATA_DIR, "test_set.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  [test]    {len(test_df):>4} samples → {test_path}")

    # Build per-client shards with non-IID class distributions
    samples_per_client = len(X_pool) // len(CLIENTS)
    shard_paths = {}

    for client_id, client_name, class_0_ratio in CLIENTS:
        X_shard, y_shard = make_noniid_shard(
            X_pool, y_pool, class_0_ratio, samples_per_client, rng
        )
        df = pd.DataFrame(X_shard, columns=[f"feat_{i}" for i in range(20)])
        df["label"] = y_shard

        path = os.path.join(DATA_DIR, f"{client_id}.csv")
        df.to_csv(path, index=False)
        shard_paths[client_id] = path

        class_0_pct = (y_shard == 0).mean() * 100
        print(
            f"  [{client_name:<8}] {len(df):>4} samples | "
            f"class-0: {class_0_pct:.1f}% | class-1: {100-class_0_pct:.1f}% → {path}"
        )

    return shard_paths


if __name__ == "__main__":
    print("Generating non-IID data shards for FedSnow clients...\n")
    paths = generate_shards()
    print(f"\nDone — {len(paths)} client shards + test set written to '{DATA_DIR}/'")
    print("These files stay local and are never uploaded to Snowflake.")
