# Day 38 (Python path): query the Cortex Search service with the snowflake.core Python API, score the
# labeled questions (hit rate@k, MRR), and ask one grounded RAG question with COMPLETE.
# Search and COMPLETE run inside Snowflake. Python only sends requests and reads results back.
# Install once in your conda env:  pip install snowflake snowflake-snowpark-python   (snowflake = snowflake.core)
# Connection: ~/.snowflake/connections.toml [ds90] (Day 36) or SNOWFLAKE_* env vars. Never put a password here.
#
#   python day38_search.py --dry-run                 # no account: check files, print the plan
#   python day38_search.py                           # trial: evaluate the 47 labeled questions (search only)
#   python day38_search.py --ask "How long can an opened insulin pen stay out of the fridge?"
#   python day38_search.py --rag-test                # trial: 3 answerable + 3 unanswerable questions through COMPLETE
import os, sys, csv, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--ask", default=None)
ap.add_argument("--rag-test", action="store_true")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--model", default="llama3.1-8b")   # native in AWS US East 1
args = ap.parse_args()

DB, SCHEMA, SERVICE = "DS90", "DAY38", "CLINICAL_DOC_SEARCH"
chunks = list(csv.DictReader(open("doc_chunks.csv")))
qs_all = list(csv.DictReader(open("eval_questions.csv")))
qs = [q for q in qs_all if q["Q_TYPE"] != "unanswerable"]
PROMPT = ("You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. "
          "If the context does not contain the answer, reply exactly: Not found in the documents.\n\n"
          "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

print(f"Plan: {len(chunks)} chunks in the index, {len(qs)} labeled questions "
      f"({sum(1 for q in qs if q['FILTER_JSON'])} with a filter), {len(qs_all) - len(qs)} unanswerable for the RAG test.")
print(f"Embedding tokens to build the index: about {sum(len(c['CHUNK_TEXT']) for c in chunks) // 4:,} (chars / 4).")
print(f"One grounded prompt with k={args.k}: about {(len(PROMPT) + args.k * 160 + 90) // 4} input tokens for COMPLETE.")
if args.dry_run:
    sys.exit(0)

from snowflake.snowpark import Session
from snowflake.core import Root

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
                         "To check the files offline: python day38_search.py --dry-run") from e

session = connect()
session.sql("USE WAREHOUSE DS90_XS").collect()
svc = Root(session).databases[DB].schemas[SCHEMA].cortex_search_services[SERVICE]

def search(question, flt=None, limit=10, columns=("CHUNK_ID",)):
    kw = dict(query=question, columns=list(columns), limit=limit)
    if flt:
        kw["filter"] = flt                     # e.g. {"@eq": {"DEPARTMENT": "Orthopedics"}}
    return svc.search(**kw).results            # list of dicts, best match first

def rag(question, flt=None):
    hits = search(question, flt, limit=args.k, columns=("CHUNK_ID", "CHUNK_TEXT"))
    ctx = "\n".join(f"[{h['CHUNK_ID']}] {h['CHUNK_TEXT']}" for h in hits)
    ans = session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)", params=[args.model, PROMPT.format(ctx=ctx, q=question)]).collect()[0][0]
    return [h["CHUNK_ID"] for h in hits], ans

if args.ask:
    ids, ans = rag(args.ask)
    print("retrieved:", ids, "\nanswer:", ans)
elif args.rag_test:
    pick = [q for q in qs if q["QID"] in ("5", "29", "47")] + [q for q in qs_all if q["Q_TYPE"] == "unanswerable"]
    for q in pick:
        ids, ans = rag(q["QUESTION"], json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None)
        print(f"\nQ{q['QID']} [{q['Q_TYPE']}] {q['QUESTION']}\n  gold {q['GOLD_CHUNK_ID'] or '-'} | retrieved {ids}\n  {ans.strip()}")
else:
    ranks = []
    for q in qs:
        got = [r["CHUNK_ID"] for r in search(q["QUESTION"], json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None)]
        ranks.append((q, got.index(q["GOLD_CHUNK_ID"]) + 1 if q["GOLD_CHUNK_ID"] in got else None))
    def score(sub):
        n = len(sub)
        return {f"hit@{k}": round(sum(1 for _, r in sub if r and r <= k) / n, 3) for k in (1, 3, 5)} | \
               {"mrr": round(sum(1 / r for _, r in sub if r) / n, 3)}
    print("Cortex Search, all  ", score(ranks))
    for t in ("keyword", "paraphrase"):
        print(f"Cortex Search, {t:10s}", score([x for x in ranks if x[0]["Q_TYPE"] == t]))
    print("Local BM25 baseline (lesson box): hit@1 0.723, hit@3 0.894, MRR 0.801 | paraphrase hit@3 0.792")
    for q, r in ranks:
        if not r or r > 3:
            print(f"  miss Q{q['QID']} [{q['Q_TYPE']}] rank {r}: {q['QUESTION']}")

session.close()
print("Done. Run 08_cost_cleanup.sql to suspend or drop the service when you finish.")
