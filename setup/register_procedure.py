import sys, os
sys.path.insert(0, "/opt/anaconda3/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from snowflake.snowpark import Session
from snowflake.snowpark.types import IntegerType, StringType

params = {
    "account":   os.getenv("SNOWFLAKE_ACCOUNT"),
    "user":      os.getenv("SNOWFLAKE_USER"),
    "password":  os.getenv("SNOWFLAKE_PASSWORD"),
    "database":  os.getenv("SNOWFLAKE_DATABASE", "FEDSNOW_DB"),
    "schema":    os.getenv("SNOWFLAKE_SCHEMA", "FEDERATION"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "role":      os.getenv("SNOWFLAKE_ROLE"),
}

def fedavg_aggregate(session: Session, round_id: int) -> str:
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
    aggregated = OrderedDict()

    for row in rows:
        n = row["NUM_SAMPLES"]
        weights_raw = row["WEIGHTS"]
        if isinstance(weights_raw, str):
            weights_raw = json.loads(weights_raw)

        for layer, tensor_list in weights_raw.items():
            arr = np.array(tensor_list, dtype=np.float64) * (n / total_samples)
            if layer in aggregated:
                aggregated[layer] += arr
            else:
                aggregated[layer] = arr

    serialized  = json.dumps({k: v.tolist() for k, v in aggregated.items()})
    avg_loss    = sum(r["LOCAL_LOSS"] * r["NUM_SAMPLES"] for r in rows) / total_samples
    avg_acc     = sum(r["LOCAL_ACCURACY"] * r["NUM_SAMPLES"] for r in rows) / total_samples
    n_clients   = len(rows)

    escaped = serialized.replace("'", "''")
    session.sql(f"""
        INSERT INTO GLOBAL_MODEL (round_id, weights, global_accuracy, avg_loss, num_clients)
        SELECT {round_id}, PARSE_JSON('{escaped}'), {avg_acc}, {avg_loss}, {n_clients}
    """).collect()

    return f"SUCCESS: aggregated {n_clients} clients for round {round_id}"


print("Connecting to Snowflake...")
with Session.builder.configs(params).create() as session:
    print("Registering FEDAVG_AGGREGATE stored procedure...")
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
