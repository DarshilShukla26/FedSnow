"""
FedSnow — Streamlit in Snowflake dashboard.

Deploy this file inside Snowflake's Streamlit editor.
Connection is managed by get_active_session() — no local .env needed.
"""
import json

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FedSnow — Federated Learning Dashboard",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ FedSnow — Federated Learning Dashboard")
st.caption(
    "Federated Learning simulation with Snowflake as the secure aggregation server."
)

session = get_active_session()

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_global_metrics() -> pd.DataFrame:
    return session.sql(
        "SELECT ROUND_ID, GLOBAL_ACCURACY, AVG_LOSS, NUM_CLIENTS "
        "FROM GLOBAL_MODEL ORDER BY ROUND_ID"
    ).to_pandas()


@st.cache_data(ttl=30)
def load_round_metrics() -> pd.DataFrame:
    return session.sql(
        "SELECT ROUND_ID, CLIENT_NAME, LOCAL_ACCURACY, LOCAL_LOSS, "
        "GLOBAL_ACCURACY, WEIGHT_DIVERGENCE, NUM_SAMPLES "
        "FROM ROUND_METRICS ORDER BY ROUND_ID, CLIENT_NAME"
    ).to_pandas()


@st.cache_data(ttl=30)
def load_drift_analysis(round_id: int) -> pd.DataFrame:
    return session.sql(
        f"SELECT CLIENT_ID, DRIFT_SCORE, CLUSTER_ID, ANALYSIS_TEXT "
        f"FROM DRIFT_ANALYSIS WHERE ROUND_ID = {round_id} "
        f"ORDER BY DRIFT_SCORE DESC"
    ).to_pandas()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

global_df = load_global_metrics()
round_df  = load_round_metrics()

if global_df.empty:
    st.warning("No federation rounds found. Run `federation/run_federation.py` first.")
    st.stop()

max_round = int(global_df["ROUND_ID"].max())
min_round = int(global_df["ROUND_ID"].min())

# ---------------------------------------------------------------------------
# Round selector
# ---------------------------------------------------------------------------

selected_round = st.slider(
    "Select Round", min_value=min_round, max_value=max_round, value=max_round
)

# ---------------------------------------------------------------------------
# Row 1: summary metrics
# ---------------------------------------------------------------------------

latest = global_df[global_df["ROUND_ID"] == selected_round].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Global Accuracy",    f"{latest['GLOBAL_ACCURACY']:.4f}")
col2.metric("Avg Client Loss",    f"{latest['AVG_LOSS']:.4f}")
col3.metric("Clients Aggregated", int(latest["NUM_CLIENTS"]))
col4.metric("Rounds Completed",   max_round)

st.divider()

# ---------------------------------------------------------------------------
# Chart 1: global accuracy over rounds
# ---------------------------------------------------------------------------

st.subheader("Global Accuracy Over Rounds")
st.line_chart(
    global_df.set_index("ROUND_ID")[["GLOBAL_ACCURACY"]],
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Chart 2: per-client local accuracy for selected round
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.subheader(f"Client Local Accuracy — Round {selected_round}")
    round_slice = round_df[round_df["ROUND_ID"] == selected_round]
    if not round_slice.empty:
        st.bar_chart(
            round_slice.set_index("CLIENT_NAME")[["LOCAL_ACCURACY"]],
            use_container_width=True,
        )
    else:
        st.info("No client metrics for this round yet.")

# ---------------------------------------------------------------------------
# Chart 3: weight divergence per client
# ---------------------------------------------------------------------------

with col_b:
    st.subheader(f"Weight Divergence — Round {selected_round}")
    if not round_slice.empty:
        st.bar_chart(
            round_slice.set_index("CLIENT_NAME")[["WEIGHT_DIVERGENCE"]],
            use_container_width=True,
        )
    else:
        st.info("No divergence data for this round yet.")

st.divider()

# ---------------------------------------------------------------------------
# Drift analysis table
# ---------------------------------------------------------------------------

st.subheader(f"Cortex AI Drift Analysis — Round {selected_round}")
drift_df = load_drift_analysis(selected_round)

if not drift_df.empty:
    st.dataframe(
        drift_df.rename(columns={
            "CLIENT_ID":     "Client",
            "DRIFT_SCORE":   "Drift Score",
            "CLUSTER_ID":    "Cluster",
            "ANALYSIS_TEXT": "Analysis",
        }).reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.info("No Cortex drift analysis available for this round.")

st.divider()

# ---------------------------------------------------------------------------
# All rounds table (expandable)
# ---------------------------------------------------------------------------

with st.expander("All Rounds — Raw Metrics"):
    st.dataframe(round_df.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.caption(
    "🔒 Raw data never left the clients — only model weights were aggregated "
    "in Snowflake via the FedAvg stored procedure."
)
