"""
Download the latest (or a specific) global model from Snowflake
and load it into a FedMLP instance ready for the next training round.
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import snowflake.connector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SNOWFLAKE_CONNECTION_PARAMS
from clients.base_client import FedMLP, BaseClient


def fetch_global_weights(round_id: int | None = None) -> tuple[OrderedDict, dict]:
    """
    Returns (weights_dict, metadata) for the requested round.
    If round_id is None, fetches the latest round.
    """
    if round_id is None:
        where = "ORDER BY ROUND_ID DESC LIMIT 1"
    else:
        where = f"WHERE ROUND_ID = {round_id} ORDER BY ROUND_ID DESC LIMIT 1"

    sql = f"""
        SELECT ROUND_ID, WEIGHTS::STRING AS WEIGHTS_STR,
               GLOBAL_ACCURACY, AVG_LOSS, NUM_CLIENTS
        FROM GLOBAL_MODEL
        {where}
    """

    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")
            cur.execute(sql)
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No global model found{'  for round ' + str(round_id) if round_id else ''}."
        )

    rid, weights_str, global_acc, avg_loss, num_clients = row
    raw = json.loads(weights_str)
    weights = OrderedDict({k: np.array(v) for k, v in raw.items()})

    metadata = {
        "round_id":        rid,
        "global_accuracy": global_acc,
        "avg_loss":        avg_loss,
        "num_clients":     num_clients,
    }
    return weights, metadata


def load_global_model(round_id: int | None = None) -> tuple[FedMLP, dict]:
    """
    Download global weights from Snowflake and return a ready-to-use FedMLP.
    """
    weights, metadata = fetch_global_weights(round_id)

    # Use a throwaway BaseClient just to call set_weights conveniently
    proxy = BaseClient.__new__(BaseClient)
    proxy.model = FedMLP()
    proxy.set_weights(weights)

    print(
        f"  Loaded global model: round={metadata['round_id']} "
        f"acc={metadata['global_accuracy']:.4f} loss={metadata['avg_loss']:.4f}"
    )
    return proxy.model, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None,
                        help="Round ID to fetch (default: latest)")
    args = parser.parse_args()

    model, meta = load_global_model(args.round)
    print("Metadata:", meta)
