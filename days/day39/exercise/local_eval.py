"""Day 39 offline checks (no Snowflake, no Cortex): runs on a laptop with pandas, pyyaml and duckdb.
1) validate the semantic model YAML against the CSV columns   2) gold SQL vs pandas reference answers
3) the verified queries (logical names) vs the gold answers  4) the result comparator on hand-written mistakes
5) the SQL guardrail   6) warehouse-second arithmetic for AUTO_SUSPEND.   pip install duckdb pyyaml pandas"""
import json, re
import duckdb
import pandas as pd
import yaml
from analyst_eval import compare_results, guard_sql

E, P, G = (pd.read_csv(f, dtype={"DRG_CODE": str}) for f in ["encounters.csv", "patients.csv", "drg_groups.csv"])
for c in ["ADMIT_DATE", "DISCHARGE_DATE"]:
    E[c] = pd.to_datetime(E[c])
P["FIRST_ADMIT_DATE"] = pd.to_datetime(P.FIRST_ADMIT_DATE)
con = duckdb.connect()
for name, df in [("ENCOUNTERS", E), ("PATIENTS", P), ("DRG_GROUPS", G)]:
    con.register(name + "_DF", df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_DF")
card = {"n_enc": len(E), "n_pat": len(P), "n_drg": len(G), "rate": round(E.READMIT_30D.mean(), 4)}
print(f"Tables: ENCOUNTERS {len(E):,}  PATIENTS {len(P):,}  DRG_GROUPS {len(G)}  overall readmission rate {E.READMIT_30D.mean():.3f}")

# ---------- 1. validate the YAML ----------
M = yaml.safe_load(open("readmit_semantic_model.yaml"))
phys = {"ENCOUNTERS": set(E.columns), "PATIENTS": set(P.columns), "DRG_GROUPS": set(G.columns)}
KW = {"CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "IN", "NULL", "COUNT", "SUM", "AVG", "DISTINCT", "MIN", "MAX"}
problems, syn_seen, counts = [], {}, {}
for t in M["tables"]:
    cols = phys[t["base_table"]["table"]]
    names = set()
    counts[t["name"]] = {k: len(t.get(k, [])) for k in ["dimensions", "time_dimensions", "facts", "metrics", "filters"]}
    for kind in ["dimensions", "time_dimensions", "facts", "filters"]:
        for item in t.get(kind, []):
            if item["name"] in names:
                problems.append(f"duplicate name {t['name']}.{item['name']}")
            names.add(item["name"])
            expr = re.sub(r"'[^']*'", "", item["expr"])
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
                if tok.upper() not in KW and tok.upper() not in cols:
                    problems.append(f"{t['name']}.{item['name']}: column {tok} not in {t['base_table']['table']}")
            if kind != "filters" and "data_type" not in item:
                problems.append(f"{t['name']}.{item['name']}: missing data_type")
    for m in t.get("metrics", []):
        for tab, col in re.findall(r"([a-z_]+)\.([a-z_0-9]+)", m["expr"]):
            if tab != t["name"] or col not in names:
                problems.append(f"metric {m['name']}: {tab}.{col} is not a dimension or fact of {t['name']}")
    for item in t.get("dimensions", []) + t.get("time_dimensions", []) + t.get("facts", []) + t.get("metrics", []):
        for s in item.get("synonyms", []):
            if s.lower() in syn_seen:
                problems.append(f"synonym '{s}' used twice ({syn_seen[s.lower()]} and {item['name']})")
            syn_seen[s.lower()] = item["name"]
    pk = t.get("primary_key", {}).get("columns", [])
    if any(c not in names for c in pk):
        problems.append(f"{t['name']}: primary key column not defined as a dimension")
tabs = {t["name"]: t for t in M["tables"]}
for r in M["relationships"]:
    for rc in r["relationship_columns"]:
        lt, rt = tabs[r["left_table"]], tabs[r["right_table"]]
        if rc["right_column"] not in rt.get("primary_key", {}).get("columns", []):
            problems.append(f"{r['name']}: right column is not the primary key of {rt['name']} (many-to-one needs it)")
        if rc["left_column"] not in {d["name"] for d in lt["dimensions"]}:
            problems.append(f"{r['name']}: left column {rc['left_column']} missing")
n_items = sum(sum(v.values()) for v in counts.values())
print(f"\n1) YAML: {len(M['tables'])} tables, {n_items} dimensions/time dims/facts/metrics/filters, {len(M['relationships'])} relationships, "
      f"{len(M['verified_queries'])} verified queries, {len(syn_seen)} synonyms. Problems: {len(problems)}")
for p in problems:
    print("   -", p)
card.update(yaml_counts=counts, yaml_problems=len(problems), n_syn=len(syn_seen), n_vqr=len(M["verified_queries"]),
            yaml_lines=sum(1 for _ in open("readmit_semantic_model.yaml")))

# ---------- 2. gold SQL vs pandas ----------
Q = pd.read_csv("eval_questions.csv", keep_default_na=False)
EG = E.merge(G, on="DRG_CODE")
r3 = lambda s: s.round(3)
y25 = E[(E.DISCHARGE_DATE >= "2025-01-01") & (E.DISCHARGE_DATE < "2026-01-01")]
ref = {
    "Q01": pd.DataFrame({"r": [round(E.READMIT_30D.mean(), 3)]}),
    "Q02": E.groupby("PAYER").READMIT_30D.mean().round(3).reset_index(),
    "Q03": EG.groupby("SERVICE_LINE").READMIT_30D.mean().round(3).sort_values(ascending=False).head(1).reset_index(),
    "Q04": pd.DataFrame({"n": [len(y25)]}),
    "Q05": y25.groupby(y25.DISCHARGE_DATE.dt.to_period("M").dt.start_time).READMIT_30D.mean().round(3).reset_index(),
    "Q06": EG.groupby(["DRG_CODE", "DRG_DESCRIPTION"]).size().sort_values(ascending=False).head(5).reset_index(),
    "Q07": E.groupby("DISCHARGE_DISPOSITION").LOS_DAYS.mean().round(2).reset_index(),
    "Q08": pd.DataFrame({"r": [round(EG[(EG.PAYER == "Medicare") & EG.DRG_DESCRIPTION.str.contains("heart failure", case=False)].READMIT_30D.mean(), 3)]}),
    "Q09": pd.DataFrame({"n": [int((E.groupby("PATIENT_ID").size() >= 3).sum())]}),
    "Q10": E.merge(P, on="PATIENT_ID").groupby("HOME_COUNTY").READMIT_30D.mean().round(3).reset_index(),
    "Q11": pd.DataFrame({"r": [round(E[E.DISCHARGE_DISPOSITION != "AMA"].READMIT_30D.mean(), 3)]}),
    "Q12": E.assign(a=pd.cut(E.AGE_AT_ADMIT, [-1, 64, 79, 200], labels=["Under 65", "65-79", "80+"]).astype(str))
              .groupby("a").READMIT_30D.mean().round(3).reset_index(),
    "Q13": E[E.DISCHARGE_DISPOSITION.isin(["SNF", "Home health"])].groupby("DISCHARGE_DISPOSITION").READMIT_30D.mean().round(3).reset_index(),
}
gold = {}
print("\n2) Gold SQL (DuckDB) vs pandas reference answers")
ok2 = 0
for r in Q.itertuples():
    if not r.GOLD_SQL:
        print(f"   {r.QID}  no gold SQL: the right behavior is no SQL ({r.TYPE})")
        continue
    gold[r.QID] = con.execute(r.GOLD_SQL).df()
    m, why = compare_results(ref[r.QID], gold[r.QID], bool(r.ORDER_MATTERS == "True"))
    ok2 += m
    head = ", ".join(f"{c}={gold[r.QID][c].iloc[0]}" for c in gold[r.QID].columns[:3])
    print(f"   {r.QID}  {'MATCH' if m else 'DIFF '}  rows={len(gold[r.QID]):>2}  first row: {head}")
card["gold_vs_pandas"] = f"{ok2}/{len(gold)}"

# ---------- 3. verified queries on logical views ----------
print("\n3) Verified queries (logical names, __table) on DuckDB views built from the YAML")
for t in M["tables"]:
    sel = [f"{i['expr']} AS {i['name']}" for k in ["dimensions", "time_dimensions", "facts"] for i in t.get(k, [])]
    con.execute(f"CREATE OR REPLACE VIEW __{t['name']} AS SELECT {', '.join(sel)} FROM {t['base_table']['table']}")
vq_map = {"overall_readmission_rate": "Q01", "readmission_rate_by_payer": "Q02", "monthly_rate_2025": "Q05",
          "top_drgs_by_volume": "Q06", "rate_excluding_ama": "Q11"}
ok3 = 0
for v in M["verified_queries"]:
    qid = vq_map[v["name"]]
    res = con.execute(v["sql"]).df()
    om = Q.set_index("QID").loc[qid, "ORDER_MATTERS"] == "True"
    m, why = compare_results(gold[qid], res, om)
    ok3 += m
    print(f"   {v['name']:<26} vs gold {qid}: {why}   guard: {guard_sql(v['sql'])[1]}")
card["vqr_vs_gold"] = f"{ok3}/{len(M['verified_queries'])}"

# ---------- 4. comparator on hand-written mistakes (NOT Cortex output) ----------
print("\n4) Comparator on hand-written candidate SQL (typical text-to-SQL mistakes, written by the lesson author, NOT Cortex output)")
cands = [
    ("Q02", "correct, other names and order", "SELECT ROUND(AVG(READMIT_30D),3) AS RATE, PAYER AS INSURANCE, COUNT(*) AS N FROM ENCOUNTERS GROUP BY PAYER ORDER BY RATE"),
    ("Q01", "correct, no ROUND", "SELECT AVG(READMIT_30D) FROM ENCOUNTERS"),
    ("Q04", "correct, YEAR() instead of a range", "SELECT COUNT(*) FROM ENCOUNTERS WHERE YEAR(DISCHARGE_DATE) = 2025"),
    ("Q04", "uses admit date, not discharge date", "SELECT COUNT(*) FROM ENCOUNTERS WHERE YEAR(ADMIT_DATE) = 2025"),
    ("Q05", "monthly rate by admit month", "SELECT DATE_TRUNC('month', ADMIT_DATE) AS M, ROUND(AVG(READMIT_30D),3) FROM ENCOUNTERS WHERE YEAR(ADMIT_DATE) = 2025 GROUP BY 1 ORDER BY 1"),
    ("Q09", "counts encounters of those patients", "SELECT COUNT(*) FROM ENCOUNTERS WHERE PATIENT_ID IN (SELECT PATIENT_ID FROM ENCOUNTERS GROUP BY 1 HAVING COUNT(*) >= 3)"),
    ("Q11", "forgets the AMA filter", "SELECT ROUND(AVG(READMIT_30D),3) FROM ENCOUNTERS"),
    ("Q12", "age band off by one (<= 65)", "SELECT CASE WHEN AGE_AT_ADMIT <= 65 THEN 'Under 65' WHEN AGE_AT_ADMIT < 80 THEN '65-79' ELSE '80+' END AS A, ROUND(AVG(READMIT_30D),3) FROM ENCOUNTERS GROUP BY 1"),
    ("Q08", "exact text match on the DRG name", "SELECT ROUND(AVG(E.READMIT_30D),3) FROM ENCOUNTERS E JOIN DRG_GROUPS G ON E.DRG_CODE=G.DRG_CODE WHERE E.PAYER='Medicare' AND G.DRG_DESCRIPTION='Heart failure'"),
    ("Q03", "correct, CTE plus extra column", "WITH s AS (SELECT G.SERVICE_LINE, COUNT(*) N, ROUND(AVG(E.READMIT_30D),3) R FROM ENCOUNTERS E JOIN DRG_GROUPS G USING (DRG_CODE) GROUP BY 1) SELECT * FROM s ORDER BY R DESC LIMIT 1"),
]
demo = []
for qid, label, sql in cands:
    ok, why_g = guard_sql(sql)
    res = con.execute(sql).df()
    om = Q.set_index("QID").loc[qid, "ORDER_MATTERS"] == "True"
    m, why = compare_results(gold[qid], res, om)
    demo.append({"qid": qid, "label": label, "match": bool(m), "why": why})
    print(f"   {qid} {label:<38} -> {'MATCH' if m else 'WRONG'}: {why}")
card["demo"] = demo

# ---------- 5. guardrail ----------
print("\n5) guard_sql on risky SQL")
risky = ["SELECT * FROM ENCOUNTERS LIMIT 10", "DELETE FROM ENCOUNTERS", "SELECT 1; DROP TABLE PATIENTS",
         "SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY", "WITH x AS (SELECT * FROM PATIENTS) SELECT COUNT(*) FROM x",
         "WITH __encounters AS (SELECT payer, readmit_30d FROM ds90.day39.encounters) SELECT payer, AVG(readmit_30d) FROM __encounters GROUP BY payer\n -- Generated by Cortex Analyst",
         "-- note\nSELECT PAYER FROM ENCOUNTERS"]
card["guard"] = []
for s in risky:
    ok, why = guard_sql(s)
    card["guard"].append([s.replace("\n", " "), ok, why])
    print(f"   {'ALLOW' if ok else 'BLOCK'}  {why:<34} {s.replace(chr(10), ' ')[:60]}")

# ---------- 6. warehouse seconds ----------
print("\n6) Warehouse seconds for 15 questions asked 2 minutes apart, each query about 2 s (X-Small = 1 credit/hour)")
wh = {}
for susp in [60, 300, 600]:
    t, busy_until, billed = 0, None, 0
    starts = [120 * i for i in range(15)]
    sessions = []
    for s in starts:
        if busy_until is None or s > busy_until:
            sessions.append([s, s + 2])
        else:
            sessions[-1][1] = s + 2
        busy_until = sessions[-1][1] + susp
    billed = sum(max(60, (e + susp) - s) for s, e in sessions)
    wh[susp] = {"resumes": len(sessions), "seconds": billed, "credits": round(billed / 3600, 3)}
    print(f"   AUTO_SUSPEND={susp:>3}s: {len(sessions):>2} resume(s), {billed:>5} billed seconds = {billed/3600:.3f} credits")
card["wh"] = wh
card["vqr_names"] = list(vq_map)
json.dump(card, open("baseline_card.json", "w"), indent=1, default=str)
print("\nCortex Analyst itself is billed per successful message (HTTP 200), not per token: 15 questions = 15 messages,")
print("plus one per follow-up turn. Rate: Snowflake Service Consumption Table. Wrote baseline_card.json")
