# Day 37 (optional Python path): call Cortex LLM functions from Snowpark in your conda env.
# The Cortex work still runs inside Snowflake. Python only sends SQL and reads results back.
# Connection: ~/.snowflake/connections.toml [ds90] (see the Day 36 connections.toml.example) or SNOWFLAKE_* env vars.
# Never put a password in this file or in git. The trial login stays in your vault.
#
#   python day37_cortex.py --dry-run     # no account: size the run (rows, chars, token estimate)
#   python day37_cortex.py               # trial: 5 notes (default, cheap)
#   python day37_cortex.py --rows 40     # trial: all 40 notes, saved once to NOTE_AI_PY
import os, sys, argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--model", default="llama3.1-8b")  # runs natively in AWS US East 1 (no cross-region needed)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

notes = pd.read_csv("discharge_notes.csv")
labels = pd.read_csv("note_labels.csv")
sub = notes.head(args.rows)
chars = int(sub.NOTE_TEXT.str.len().sum() + sub.PATIENT_FEEDBACK.str.len().sum())
print(f"Plan: {len(sub)} notes, {chars:,} characters of your text, about {chars // 4:,} tokens before "
      "each function's built-in prompt. Rates: Snowflake Service Consumption Table.")
if args.dry_run:
    sys.exit(0)

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, lit
from snowflake.cortex import complete, summarize, sentiment, classify_text, extract_answer

def connect():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()          # inside a Snowsight notebook
    except Exception:
        pass
    keys = ["account", "user", "password", "role", "warehouse", "database", "schema"]
    p = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in keys if os.environ.get(f"SNOWFLAKE_{k.upper()}")}
    if p.get("account") and p.get("user"):
        return Session.builder.configs(p).create()
    try:
        return Session.builder.config("connection_name", os.environ.get("SNOWFLAKE_CONNECTION_NAME", "ds90")).create()
    except Exception as e:
        raise SystemExit("No Snowflake connection found. Add a [ds90] entry to ~/.snowflake/connections.toml "
                         "or set SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER / SNOWFLAKE_PASSWORD in your shell. "
                         "To size the run offline: python day37_cortex.py --dry-run") from e

session = connect()
for s in ["USE WAREHOUSE DS90_XS", "CREATE SCHEMA IF NOT EXISTS DS90.DAY37", "USE SCHEMA DS90.DAY37"]:
    session.sql(s).collect()

# Load the simulated notes and the answer key (same tables as 02_load_notes.sql).
session.write_pandas(notes, "DISCHARGE_NOTES", auto_create_table=True, overwrite=True)
session.write_pandas(labels, "NOTE_LABELS", auto_create_table=True, overwrite=True)

cats = ["low readmission risk", "medium readmission risk", "high readmission risk"]
df = (session.table("DISCHARGE_NOTES").filter(col("NOTE_ID") <= args.rows)
      .with_column("SUMMARY", summarize(col("NOTE_TEXT")))
      .with_column("FEEDBACK_SCORE", sentiment(col("PATIENT_FEEDBACK")))
      .with_column("RISK_OBJ", classify_text(col("NOTE_TEXT"), cats))
      .with_column("FOLLOWUP_RAW", extract_answer(col("NOTE_TEXT"), lit("In how many days should the patient follow up?")))
      .with_column("REMINDER", complete(args.model, lit("In one plain sentence for the patient, say who to see and when. Note: ")
                                        + col("NOTE_TEXT")))
      .select("NOTE_ID", "SUMMARY", "FEEDBACK_SCORE", "RISK_OBJ", "FOLLOWUP_RAW", "REMINDER"))

print("SQL that Snowpark will send (Cortex runs in Snowflake, not on your laptop):")
print(df.queries["queries"][-1][:600], "...")
df.write.save_as_table("NOTE_AI_PY", mode="overwrite")   # pay for tokens once, then reuse the table
out = session.table("NOTE_AI_PY").to_pandas().merge(labels, on="NOTE_ID")

import json
def first(x, key):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        v = v[0] if isinstance(v, list) else v
        return v.get(key) if isinstance(v, dict) else v
    except Exception:
        return x
out["AI_RISK_BAND"] = out.RISK_OBJ.map(lambda x: str(first(x, "label")).split(" ")[0])
out["AI_FOLLOWUP"] = out.FOLLOWUP_RAW.map(lambda x: first(x, "answer"))
out["AI_TONE"] = pd.cut(out.FEEDBACK_SCORE, [-1.01, -0.5, 0.5, 1.01], labels=["negative", "neutral", "positive"])
print(out[["NOTE_ID", "TRUE_RISK_BAND", "AI_RISK_BAND", "TRUE_FOLLOWUP_DAYS", "AI_FOLLOWUP",
           "TRUE_FEEDBACK_TONE", "AI_TONE"]].to_string(index=False))
print("risk accuracy", round((out.AI_RISK_BAND == out.TRUE_RISK_BAND).mean(), 3),
      "| tone accuracy", round((out.AI_TONE.astype(str) == out.TRUE_FEEDBACK_TONE).mean(), 3),
      "| keyword baseline on all 40: risk 0.375, tone 0.800")
print("Sample reminder:", out.REMINDER.iloc[0])

session.sql("ALTER WAREHOUSE DS90_XS SUSPEND").collect()
print("Suspended DS90_XS. Check tokens later with 06_cost_check.sql.")
session.close()
