import sys, os
sys.path.insert(0, "/opt/anaconda3/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import snowflake.connector

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    role=os.getenv("SNOWFLAKE_ROLE"),
    database=os.getenv("SNOWFLAKE_DATABASE", "FEDSNOW_DB"),
    schema=os.getenv("SNOWFLAKE_SCHEMA", "FEDERATION"),
)

cur = conn.cursor()

app_path = os.path.join(os.path.dirname(__file__), "..", "streamlit_app", "app.py")

print("1. Creating Streamlit stage...")
cur.execute("CREATE STAGE IF NOT EXISTS FEDSNOW_STREAMLIT_STAGE")

print("2. Uploading app.py to stage...")
cur.execute(f"PUT file://{os.path.abspath(app_path)} @FEDSNOW_STREAMLIT_STAGE/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE")

print("3. Creating Streamlit app in Snowflake...")
cur.execute("""
    CREATE OR REPLACE STREAMLIT FEDSNOW_DASHBOARD
        ROOT_LOCATION = '@FEDSNOW_DB.FEDERATION.FEDSNOW_STREAMLIT_STAGE'
        MAIN_FILE = 'app.py'
        QUERY_WAREHOUSE = 'COMPUTE_WH'
        COMMENT = 'FedSnow Federated Learning Dashboard'
""")

print("4. Fetching dashboard URL...")
cur.execute("SHOW STREAMLITS LIKE 'FEDSNOW_DASHBOARD'")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]

cur.close()
conn.close()

print("\nStreamlit app deployed successfully!")
print(f"\nOpen Snowsight → Streamlit → FEDSNOW_DASHBOARD")
print(f"Or navigate to: https://app.snowflake.com/oofotij/ylb21127/#/streamlit-apps")
