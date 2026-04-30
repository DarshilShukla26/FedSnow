"""
Snowflake Task-based round orchestrator.
Creates a TASK that calls FEDAVG_AGGREGATE on a schedule,
or triggers it manually for a specific round.
"""
import os
import sys

import snowflake.connector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SNOWFLAKE_CONNECTION_PARAMS


def create_round_task(cron_schedule: str = "USING CRON 0 * * * * UTC") -> None:
    """
    Create a Snowflake Task that checks for un-aggregated rounds every hour
    and calls FEDAVG_AGGREGATE for any pending round_id.

    The task uses a simple approach: find the max round_id in CLIENT_WEIGHTS
    that is not yet in GLOBAL_MODEL and aggregate it.
    """
    sql = """
    CREATE OR REPLACE TASK FEDSNOW_ROUND_TASK
        WAREHOUSE = {warehouse}
        {schedule}
    AS
    DECLARE
        pending_round INT;
    BEGIN
        SELECT MAX(cw.ROUND_ID) INTO :pending_round
        FROM CLIENT_WEIGHTS cw
        WHERE NOT EXISTS (
            SELECT 1 FROM GLOBAL_MODEL gm WHERE gm.ROUND_ID = cw.ROUND_ID
        );

        IF (pending_round IS NOT NULL) THEN
            CALL FEDAVG_AGGREGATE(:pending_round);
        END IF;
    END;
    """.format(
        warehouse=SNOWFLAKE_CONNECTION_PARAMS["warehouse"],
        schedule=cron_schedule,
    )

    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")
            cur.execute(sql)
            cur.execute("ALTER TASK FEDSNOW_ROUND_TASK RESUME")
    print("FEDSNOW_ROUND_TASK created and resumed.")


def trigger_round_now(round_id: int) -> None:
    """Manually execute the task for a specific round (bypasses schedule)."""
    with snowflake.connector.connect(**SNOWFLAKE_CONNECTION_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(f"USE DATABASE {SNOWFLAKE_CONNECTION_PARAMS['database']}")
            cur.execute(f"USE SCHEMA {SNOWFLAKE_CONNECTION_PARAMS['schema']}")
            cur.execute(f"EXECUTE TASK FEDSNOW_ROUND_TASK")
    print(f"Task executed for round {round_id}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-task", action="store_true",
                        help="Create the Snowflake scheduled task")
    parser.add_argument("--trigger", type=int, metavar="ROUND_ID",
                        help="Manually trigger the task for a round")
    args = parser.parse_args()

    if args.create_task:
        create_round_task()
    if args.trigger is not None:
        trigger_round_now(args.trigger)
