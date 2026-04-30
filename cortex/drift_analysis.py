"""
Cortex-powered drift analysis for each federated client.

Per client per round:
  - Compute L2 weight divergence vs. global model
  - Call Snowflake Cortex COMPLETE() for a 2-sentence drift interpretation
  - Assign cluster_id via k-means(k=2) on [local_accuracy, weight_divergence]
  - Write results to DRIFT_ANALYSIS table
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import snowflake.connector
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SNOWFLAKE_CONNECTION_PARAMS


# ---------------------------------------------------------------------------
# Weight divergence
# ---------------------------------------------------------------------------

def l2_divergence(client_weights: OrderedDict, global_weights: OrderedDict) -> float:
    """Compute total L2 norm between client and global weight tensors."""
    total = 0.0
    for layer in global_weights:
        if layer in client_weights:
            diff   = np.array(client_weights[layer]) - np.array(global_weights[layer])
            total += float(np.linalg.norm(diff))
    return total


# ---------------------------------------------------------------------------
# Cortex COMPLETE call
# ---------------------------------------------------------------------------

def cortex_analyze(
    session_cursor,
    client_name: str,
    round_id: int,
    local_accuracy: float,
    local_loss: float,
    weight_divergence: float,
    num_samples: int,
) -> str:
    """Call Snowflake Cortex COMPLETE and return the analysis string."""
    prompt = (
        f"Given these federated learning metrics for client {client_name} "
        f"in round {round_id}: local_accuracy={local_accuracy:.4f}, "
        f"local_loss={local_loss:.4f}, weight_divergence={weight_divergence:.4f}, "
        f"num_samples={num_samples}. "
        f"In 2 sentences, analyze whether this client's data distribution is causing "
        f"model drift and what it means for federation quality."
    )
    escaped = prompt.replace("'", "''")

    session_cursor.execute(
        f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-7b', '{escaped}')"
    )
    row = session_cursor.fetchone()
    return row[0].strip() if row else "No analysis available."


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def assign_clusters(metrics: list[dict]) -> list[int]:
    """
    k-means(k=2) on [local_accuracy, weight_divergence].
    Returns cluster assignments aligned with the input list.
    """
    if len(metrics) < 2:
        return [0] * len(metrics)

    X = np.array([[m["local_accuracy"], m["weight_divergence"]] for m in metrics])
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    return km.fit_predict(X).tolist()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_drift_analysis(
    round_id: int,
    client_metrics: list[dict],
    global_weights: OrderedDict,
) -> None:
    """
    client_metrics: list of dicts with keys:
        client_id, client_name, local_accuracy, local_loss,
        num_samples, weights (OrderedDict)
    global_weights: aggregated global model weights for this round
    """
    # Compute divergence scores
    for m in client_metrics:
        m["weight_divergence"] = l2_divergence(m["weights"], global_weights)

    cluster_ids = assign_clusters(client_metrics)

    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")

            for i, m in enumerate(client_metrics):
                analysis = cortex_analyze(
                    cur,
                    client_name=m["client_name"],
                    round_id=round_id,
                    local_accuracy=m["local_accuracy"],
                    local_loss=m["local_loss"],
                    weight_divergence=m["weight_divergence"],
                    num_samples=m["num_samples"],
                )
                cluster_id = cluster_ids[i]

                escaped_analysis = analysis.replace("'", "''")
                cur.execute(f"""
                    INSERT INTO DRIFT_ANALYSIS
                        (round_id, client_id, drift_score, cluster_id, analysis_text)
                    VALUES (
                        {round_id},
                        '{m["client_id"]}',
                        {m["weight_divergence"]},
                        {cluster_id},
                        '{escaped_analysis}'
                    )
                """)

                print(
                    f"  Drift [{m['client_name']}] "
                    f"div={m['weight_divergence']:.4f} cluster={cluster_id}"
                )
