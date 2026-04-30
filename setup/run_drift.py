import sys, os
sys.path.insert(0, "/opt/anaconda3/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import json
from collections import OrderedDict
import numpy as np
import snowflake.connector
from sklearn.cluster import KMeans

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    role=os.getenv("SNOWFLAKE_ROLE"),
    database="FEDSNOW_DB",
    schema="FEDERATION",
)
cur = conn.cursor()

# Clear existing drift analysis so we don't duplicate
cur.execute("TRUNCATE TABLE DRIFT_ANALYSIS")
print("Cleared DRIFT_ANALYSIS table.\n")

# Fetch all rounds
cur.execute("SELECT ROUND_ID FROM GLOBAL_MODEL ORDER BY ROUND_ID")
rounds = [r[0] for r in cur.fetchall()]

for round_id in rounds:
    print(f"Round {round_id}/{max(rounds)}")

    # Get global weights for this round
    cur.execute(f"SELECT WEIGHTS::STRING FROM GLOBAL_MODEL WHERE ROUND_ID = {round_id}")
    global_raw = json.loads(cur.fetchone()[0])
    global_weights = OrderedDict({k: np.array(v) for k, v in global_raw.items()})

    # Get all client weights + metrics for this round
    cur.execute(f"""
        SELECT cw.CLIENT_ID, cw.CLIENT_NAME, cw.NUM_SAMPLES, cw.WEIGHTS::STRING,
               rm.LOCAL_ACCURACY, rm.LOCAL_LOSS
        FROM CLIENT_WEIGHTS cw
        JOIN ROUND_METRICS rm
          ON cw.ROUND_ID = rm.ROUND_ID AND cw.CLIENT_ID = rm.CLIENT_ID
        WHERE cw.ROUND_ID = {round_id}
        ORDER BY cw.CLIENT_ID
    """)
    rows = cur.fetchall()

    client_metrics = []
    for client_id, client_name, num_samples, weights_str, local_acc, local_loss in rows:
        raw = json.loads(weights_str)
        client_weights = OrderedDict({k: np.array(v) for k, v in raw.items()})

        # L2 divergence
        div = sum(
            float(np.linalg.norm(np.array(client_weights[l]) - np.array(global_weights[l])))
            for l in global_weights if l in client_weights
        )

        client_metrics.append({
            "client_id": client_id,
            "client_name": client_name,
            "num_samples": num_samples,
            "local_accuracy": local_acc,
            "local_loss": local_loss,
            "weight_divergence": div,
        })

    # K-means clustering on [local_accuracy, weight_divergence]
    X_cluster = np.array([[m["local_accuracy"], m["weight_divergence"]] for m in client_metrics])
    clusters = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_cluster).tolist()

    # Cortex COMPLETE for each client
    for i, m in enumerate(client_metrics):
        prompt = (
            f"Given these federated learning metrics for client {m['client_name']} "
            f"in round {round_id}: local_accuracy={m['local_accuracy']:.4f}, "
            f"local_loss={m['local_loss']:.4f}, weight_divergence={m['weight_divergence']:.4f}, "
            f"num_samples={m['num_samples']}. "
            f"In 2 sentences, analyze whether this client data distribution is causing "
            f"model drift and what it means for federation quality."
        )
        escaped_prompt = prompt.replace("'", "\\'")

        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-7b', '{escaped_prompt}')")
        analysis = (cur.fetchone()[0] or "").strip()

        escaped_analysis = analysis.replace("'", "\\'")
        cur.execute(f"""
            INSERT INTO DRIFT_ANALYSIS (round_id, client_id, drift_score, cluster_id, analysis_text)
            VALUES ({round_id}, '{m['client_id']}', {m['weight_divergence']}, {clusters[i]}, '{escaped_analysis}')
        """)

        print(f"  [{m['client_name']:<8}] div={m['weight_divergence']:.4f} cluster={clusters[i]}")

    print()

cur.close()
conn.close()
print("Cortex drift analysis complete for all rounds.")
