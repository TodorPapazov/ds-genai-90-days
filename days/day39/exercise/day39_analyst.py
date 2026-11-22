"""Day 39: call the Cortex Analyst REST API from Python, run the SQL it returns (after a guard), and score it
against the gold SQL. Uses your Day 36 [ds90] entry in ~/.snowflake/connections.toml (or SNOWFLAKE_* env vars).
Never put a password or token in this file.

  pip install snowflake-connector-python requests pandas pyyaml
  python day39_analyst.py --dry-run                       # no account: check files, print one request body
  python day39_analyst.py --ask "What is the readmission rate by payer?"
  python day39_analyst.py                                 # all 15 questions = 15 messages, writes analyst_eval_results.csv
  python day39_analyst.py --source stage                  # same, but with the YAML file on the stage
"""
import argparse, json, os, sys, time
import pandas as pd
from analyst_eval import compare_results, guard_sql

VIEW = "DS90.DAY39.READMIT_ANALYST"
STAGE_FILE = "@DS90.DAY39.SEMANTIC_MODELS/readmit_semantic_model.yaml"
API = "/api/v2/cortex/analyst/message"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--ask")
ap.add_argument("--source", choices=["view", "stage"], default="view")
ap.add_argument("--max-messages", type=int, default=20, help="hard cap on Cortex Analyst messages this run")
ap.add_argument("--save-table", action="store_true", help="also write GENERATED_SQL into DS90.DAY39.ANALYST_RUNS")
args = ap.parse_args()


def body_for(question, history=None):
    msgs = (history or []) + [{"role": "user", "content": [{"type": "text", "text": question}]}]
    b = {"messages": msgs, "stream": False}
    if args.source == "view":
        b["semantic_view"] = VIEW
    else:
        b["semantic_model_file"] = STAGE_FILE
    return b


Q = pd.read_csv("eval_questions.csv", keep_default_na=False)
if args.dry_run:
    import yaml
    m = yaml.safe_load(open("readmit_semantic_model.yaml"))
    print(f"YAML ok: {m['name']}, {len(m['tables'])} tables, {len(m['verified_queries'])} verified queries")
    print(f"{len(Q)} questions ({(Q.GOLD_SQL != '').sum()} with gold SQL). Messages for a full run: {len(Q)}")
    print(f"POST https://<account>.snowflakecomputing.com{API}")
    print(json.dumps(body_for(args.ask or Q.QUESTION[1]), indent=2))
    print("Dry run only: no connection, no request, no credits.")
    sys.exit(0)

import requests
import snowflake.connector

conn = snowflake.connector.connect(connection_name=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "ds90"))
cur = conn.cursor()
cur.execute("USE WAREHOUSE DS90_XS")
cur.execute("USE SCHEMA DS90.DAY39")
cur.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 60")      # guardrail: no long-running generated SQL
sent = 0


def ask(question):
    """One message to Cortex Analyst. Returns (text, sql, suggestions, verified_query_name, raw)."""
    global sent
    if sent >= args.max_messages:
        raise SystemExit(f"Stopped: message cap {args.max_messages} reached (each message is billed).")
    r = requests.post(f"https://{conn.host}{API}", json=body_for(question), timeout=90,
                      headers={"Authorization": f'Snowflake Token="{conn.rest.token}"',   # session token, not a password
                               "Content-Type": "application/json", "Accept": "application/json"})
    sent += 1
    if r.status_code != 200:                     # only HTTP 200 responses are billed
        return f"HTTP {r.status_code}: {r.text[:200]}", None, [], None, {}
    raw = r.json()
    text, sql, sugg, vq = "", None, [], None
    for c in raw["message"]["content"]:
        if c["type"] == "text":
            text += c["text"]
        elif c["type"] == "sql":
            sql = c["statement"]
            vq = ((c.get("confidence") or {}).get("verified_query_used") or {}).get("name")
        elif c["type"] in ("suggestion", "suggestions"):
            sugg = c.get("suggestions", [])
    return text, sql, sugg, vq, raw


def run(sql):
    ok, why = guard_sql(sql)
    if not ok:
        return None, "guard: " + why
    try:
        cur.execute(sql.strip().rstrip(";"))
        rows = cur.fetchmany(1000)                       # guardrail: never pull more than 1,000 rows
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description]), "ok"
    except Exception as e:                       # noqa: BLE001  (show the error, keep going)
        return None, f"error: {str(e)[:120]}"


if args.ask:
    text, sql, sugg, vq, raw = ask(args.ask)
    print("Analyst:", text)
    print("Model(s):", raw.get("response_metadata", {}).get("model_names"), " category:",
          raw.get("response_metadata", {}).get("question_category"), " verified query:", vq)
    if sql:
        print("SQL:\n", sql)
        df, status = run(sql)
        print(status)
        if df is not None:
            print(df.head(20).to_string(index=False))
    for s in sugg:
        print("Suggestion:", s)
    print(f"Messages sent: {sent}")
    sys.exit(0)

rows = []
for q in Q.itertuples():
    t0 = time.time()
    text, sql, sugg, vq, raw = ask(q.QUESTION)
    secs = round(time.time() - t0, 1)
    if not q.GOLD_SQL:                                   # Q14, Q15: correct when there is no SQL
        verdict = "correct refusal" if sql is None else "answered when it should not"
        note = (sugg[0] if sugg else text)[:120]
    elif sql is None:
        verdict, note = "no SQL returned", (sugg[0] if sugg else text)[:120]
    else:
        gen, status = run(sql)
        gold, _ = run(q.GOLD_SQL)
        if gen is None:
            verdict, note = "SQL failed", status
        else:
            match, note = compare_results(gold, gen, q.ORDER_MATTERS == "True")
            verdict = "match" if match else "wrong"
    rows.append({"QID": q.QID, "TYPE": q.TYPE, "IN_VQR": q.IN_VQR, "VERDICT": verdict, "NOTE": note,
                 "VERIFIED_QUERY_USED": vq, "SECONDS": secs, "GENERATED_SQL": sql})
    print(f"{q.QID} {verdict:<28} {secs:>5}s  vq={vq}  {note[:70]}")

res = pd.DataFrame(rows)
res.to_csv("analyst_eval_results.csv", index=False)
good = res.VERDICT.isin(["match", "correct refusal"])
print(f"\nCorrect: {good.sum()} / {len(res)}  (in verified queries: {good[res.IN_VQR == 'yes'].mean():.2f}, "
      f"not in verified queries: {good[res.IN_VQR == 'no'].mean():.2f})")
print(f"Cortex Analyst messages sent: {sent}. Check credits later in CORTEX_ANALYST_USAGE_HISTORY (06_guardrails_cost.sql).")
if args.save_table:
    for r in rows:
        cur.execute("UPDATE ANALYST_RUNS SET GENERATED_SQL = %s, VERIFIED_QUERY_USED = %s, NOTE = %s WHERE QID = %s",
                    (r["GENERATED_SQL"], r["VERIFIED_QUERY_USED"], r["VERDICT"] + ": " + str(r["NOTE"])[:200], r["QID"]))
    print("Saved GENERATED_SQL into ANALYST_RUNS. Now run 05_evaluate.sql.")
cur.execute("ALTER WAREHOUSE DS90_XS SUSPEND")
conn.close()
