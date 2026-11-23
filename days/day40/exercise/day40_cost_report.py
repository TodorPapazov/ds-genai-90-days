"""Day 40: a small cost report for the Snowflake week, from YOUR account's usage views.
Uses the Day 36 [ds90] entry in ~/.snowflake/connections.toml (or set SNOWFLAKE_CONNECTION_NAME).
Never put a password or token in this file. Read-only: it runs SELECTs on SNOWFLAKE.ACCOUNT_USAGE, then suspends DS90_XS.

  pip install snowflake-connector-python pandas matplotlib
  python day40_cost_report.py --dry-run      # no account: print the SQL it would run
  python day40_cost_report.py                # writes cost_by_day.csv, coco_requests.csv, cost_by_day.png
"""
import argparse, os, sys

SINCE = "2026-11-16"
QUERIES = {
    "cost_by_day": f"""
        SELECT USAGE_DATE, SERVICE_TYPE, ROUND(SUM(CREDITS_USED), 4) AS CREDITS_USED, ROUND(SUM(CREDITS_BILLED), 4) AS CREDITS_BILLED
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
        WHERE USAGE_DATE >= '{SINCE}' GROUP BY 1, 2 ORDER BY 1, 2""",
    "ai_functions": f"""
        SELECT DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, FUNCTION_NAME, MODEL_NAME, SUM(TOKENS) AS TOKENS,
               ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
        WHERE USAGE_TIME >= '{SINCE}' GROUP BY 1, 2, 3 ORDER BY 1, 5 DESC""",
    "coco_requests": f"""
        SELECT 'Snowsight' AS SURFACE, DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, COUNT(*) AS REQUESTS,
               SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
        WHERE USAGE_TIME >= '{SINCE}' GROUP BY 1, 2 ORDER BY 2""",
    "warehouse": f"""
        SELECT WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE AS DAY, ROUND(SUM(CREDITS_USED), 4) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE START_TIME >= '{SINCE}' GROUP BY 1, 2 ORDER BY 2, 1""",
}

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

if args.dry_run:
    for name, q in QUERIES.items():
        print(f"-- {name}\n{q.strip()};\n")
    print("DRY RUN: nothing was sent to Snowflake.")
    sys.exit(0)

import pandas as pd
import snowflake.connector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

conn = snowflake.connector.connect(connection_name=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "ds90"))
cur = conn.cursor()
cur.execute("USE ROLE ACCOUNTADMIN")
cur.execute("USE WAREHOUSE DS90_XS")
out = {}
for name, q in QUERIES.items():
    try:
        out[name] = cur.execute(q).fetch_pandas_all()
    except Exception as e:                      # a view can be missing or empty on a trial: report and continue
        print(f"{name}: skipped ({type(e).__name__}: {str(e)[:120]})")
        out[name] = pd.DataFrame()
    out[name].to_csv(f"{name}.csv", index=False)
    print(f"{name}: {len(out[name])} rows -> {name}.csv")
try:
    cur.execute("ALTER WAREHOUSE DS90_XS SUSPEND")
except Exception:
    pass                                        # already suspended
conn.close()

c = out["cost_by_day"]
if len(c):
    p = c.pivot_table(index="USAGE_DATE", columns="SERVICE_TYPE", values="CREDITS_USED", aggfunc="sum").fillna(0)
    ax = p.plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_ylabel("credits used"); ax.set_title("My Snowflake week: credits per day by service type (ACCOUNT_USAGE)")
    plt.tight_layout(); plt.savefig("cost_by_day.png", dpi=150)
    print("total credits used since", SINCE, "=", round(float(c.CREDITS_USED.sum()), 4), "-> cost_by_day.png")
else:
    print("No METERING_DAILY_HISTORY rows yet (the view can lag). Try again later.")
