import sys, os
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))

import snowflake.connector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    role=os.getenv("SNOWFLAKE_ROLE"),
)

base = os.path.dirname(__file__)
scripts = [
    "01_create_schema.sql",
    "02_create_tables.sql",
    "03_create_stage.sql",
]

cur = conn.cursor()
for name in scripts:
    sql = open(os.path.join(base, name)).read()
    print(f"\n--- Running {name} ---")
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        try:
            cur.execute(statement)
            print(f"  OK: {statement[:70].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"  WARN: {e}")

cur.close()
conn.close()
print("\nSetup complete.")
