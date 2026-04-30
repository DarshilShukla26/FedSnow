"""
Upload a client's serialized model weights to the Snowflake CLIENT_WEIGHTS table.
Uses snowflake-connector-python execute_string with PARSE_JSON for VARIANT storage.
"""
import json
import os
import sys
from datetime import datetime, timezone

import snowflake.connector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SNOWFLAKE_CONNECTION_PARAMS


def upload_weights(
    round_id: int,
    client_id: str,
    client_name: str,
    num_samples: int,
    weights_json: str,
    local_loss: float,
    local_accuracy: float,
) -> None:
    """Insert one client's weights for a given round into CLIENT_WEIGHTS."""
    # Escape the JSON string for embedding inside SQL
    escaped = weights_json.replace("'", "''")

    sql = f"""
        INSERT INTO CLIENT_WEIGHTS
            (round_id, client_id, client_name, num_samples,
             weights, local_loss, local_accuracy)
        SELECT
            {round_id},
            '{client_id}',
            '{client_name}',
            {num_samples},
            PARSE_JSON('{escaped}'),
            {local_loss},
            {local_accuracy}
    """

    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")
            cur.execute(sql)

    print(
        f"  Uploaded weights: round={round_id} client={client_name} "
        f"samples={num_samples} acc={local_accuracy:.4f} loss={local_loss:.4f}"
    )


def upload_round_metric(
    round_id: int,
    client_id: str,
    client_name: str,
    local_accuracy: float,
    local_loss: float,
    global_accuracy: float,
    weight_divergence: float,
    num_samples: int,
) -> None:
    """Append a row to ROUND_METRICS."""
    sql = f"""
        INSERT INTO ROUND_METRICS
            (round_id, client_id, client_name, local_accuracy, local_loss,
             global_accuracy, weight_divergence, num_samples)
        VALUES (
            {round_id}, '{client_id}', '{client_name}',
            {local_accuracy}, {local_loss}, {global_accuracy},
            {weight_divergence}, {num_samples}
        )
    """
    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")
            cur.execute(sql)
