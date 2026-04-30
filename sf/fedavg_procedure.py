"""
Register and call the FedAvg stored procedure in Snowflake using Snowpark.

The stored procedure FEDAVG_AGGREGATE(round_id INT):
  1. Reads all CLIENT_WEIGHTS rows for the given round
  2. Computes weighted-average of each weight tensor (weighted by num_samples)
  3. Writes the result to GLOBAL_MODEL
  4. Returns a status string
"""
import json
import os
import sys

from snowflake.snowpark import Session
from snowflake.snowpark.types import IntegerType, StringType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SNOWFLAKE_CONNECTION_PARAMS


# ---------------------------------------------------------------------------
# Stored-procedure body (runs inside Snowflake's Python sandbox)
# ---------------------------------------------------------------------------

def fedavg_aggregate(session: Session, round_id: int) -> str:
    """
    Snowpark Python stored procedure for Federated Averaging.
    Weighted average: global_w = sum(w_i * n_i) / sum(n_i)
    """
    import json
    import numpy as np
    from collections import OrderedDict

    rows = (
        session.table("CLIENT_WEIGHTS")
        .filter(f"ROUND_ID = {round_id}")
        .select("CLIENT_ID", "NUM_SAMPLES", "WEIGHTS", "LOCAL_LOSS", "LOCAL_ACCURACY")
        .collect()
    )

    if not rows:
        return f"ERROR: no client weights found for round {round_id}"

    total_samples = sum(r["NUM_SAMPLES"] for r in rows)
    aggregated: OrderedDict = OrderedDict()

    for row in rows:
        n = row["NUM_SAMPLES"]
        # VARIANT comes back as a Python dict/list from Snowpark
        weights_raw = row["WEIGHTS"]
        if isinstance(weights_raw, str):
            weights_raw = json.loads(weights_raw)

        for layer, tensor_list in weights_raw.items():
            arr = np.array(tensor_list, dtype=np.float64) * (n / total_samples)
            if layer in aggregated:
                aggregated[layer] += arr
            else:
                aggregated[layer] = arr

    serialized = json.dumps({k: v.tolist() for k, v in aggregated.items()})
    avg_loss    = sum(r["LOCAL_LOSS"] * r["NUM_SAMPLES"] for r in rows) / total_samples
    avg_acc     = sum(r["LOCAL_ACCURACY"] * r["NUM_SAMPLES"] for r in rows) / total_samples
    n_clients   = len(rows)

    escaped = serialized.replace("'", "''")
    session.sql(f"""
        INSERT INTO GLOBAL_MODEL (round_id, weights, global_accuracy, avg_loss, num_clients)
        SELECT {round_id}, PARSE_JSON('{escaped}'), {avg_acc}, {avg_loss}, {n_clients}
    """).collect()

    return f"SUCCESS: aggregated {n_clients} clients for round {round_id}"


# ---------------------------------------------------------------------------
# Registration helper — run once to deploy the SP to Snowflake
# ---------------------------------------------------------------------------

def register_procedure() -> None:
    """Register fedavg_aggregate as a Snowpark stored procedure."""
    with Session.builder.configs(SNOWFLAKE_CONNECTION_PARAMS).create() as session:
        session.sproc.register(
            func=fedavg_aggregate,
            name="FEDAVG_AGGREGATE",
            return_type=StringType(),
            input_types=[IntegerType()],
            is_permanent=True,
            stage_location="@MODEL_WEIGHTS_STAGE",
            packages=["snowflake-snowpark-python", "numpy"],
            replace=True,
            execute_as="caller",
        )
        print("Stored procedure FEDAVG_AGGREGATE registered successfully.")


# ---------------------------------------------------------------------------
# Call helper — used from the federation loop
# ---------------------------------------------------------------------------

def call_fedavg(round_id: int) -> str:
    """Call the FEDAVG_AGGREGATE stored procedure for the given round."""
    with Session.builder.configs(SNOWFLAKE_CONNECTION_PARAMS).create() as session:
        result = session.call("FEDAVG_AGGREGATE", round_id)
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true",
                        help="Register (or re-register) the stored procedure")
    parser.add_argument("--call", type=int, metavar="ROUND_ID",
                        help="Call FEDAVG_AGGREGATE for the given round_id")
    args = parser.parse_args()

    if args.register:
        register_procedure()
    if args.call is not None:
        print(call_fedavg(args.call))
