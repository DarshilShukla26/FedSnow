import os
from dotenv import load_dotenv

load_dotenv()

SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE", "FEDSNOW_DB")
SNOWFLAKE_SCHEMA    = os.getenv("SNOWFLAKE_SCHEMA", "FEDERATION")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "")
SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE", "")

SNOWFLAKE_CONNECTION_PARAMS = {
    "account":   SNOWFLAKE_ACCOUNT,
    "user":      SNOWFLAKE_USER,
    "password":  SNOWFLAKE_PASSWORD,
    "database":  SNOWFLAKE_DATABASE,
    "schema":    SNOWFLAKE_SCHEMA,
    "warehouse": SNOWFLAKE_WAREHOUSE,
    "role":      SNOWFLAKE_ROLE,
}

# Federation hyperparameters
NUM_ROUNDS    = 10
NUM_CLIENTS   = 4
LOCAL_EPOCHS  = 5
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# Model architecture
INPUT_DIM  = 20
HIDDEN1    = 64
HIDDEN2    = 32
OUTPUT_DIM = 2
DROPOUT    = 0.3

# Data
TOTAL_SAMPLES = 2000
TEST_SAMPLES  = 200
DATA_DIR      = "data/shards"

# Client definitions: (client_id, client_name, class_0_ratio)
CLIENTS = [
    ("client_hospital", "Hospital", 0.70),
    ("client_bank",     "Bank",     0.40),  # 60% class 1 → 40% class 0
    ("client_device",   "Device",   0.50),
    ("client_retail",   "Retail",   0.20),  # 80% class 1 → 20% class 0
]
