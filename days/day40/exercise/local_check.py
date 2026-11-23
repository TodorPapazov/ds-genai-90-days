"""Day 40 offline checks (no Snowflake, no LLM). Runs on the CSVs from gen_discharges.py.
1) A rule-based (regex) reader for the two note fields, graded against note_labels.csv: the bar the LLM should beat.
2) The worklist logic from 05_worklist.sql, run in DuckDB, with three versions of the note fields:
   none (score only), regex, and the answer key (the best the notes could do).
3) Cost arithmetic: rough token counts and warehouse seconds for today's session under three AUTO_SUSPEND values.
Writes baseline_card.json and prints the numbers quoted in the lesson."""
import json, re
import numpy as np
import pandas as pd
import duckdb

D = pd.read_csv("discharges.csv")
L = pd.read_csv("note_labels.csv")
O = pd.read_csv("outcomes.csv")
B = pd.read_csv("baseline_rates.csv")
G = pd.read_csv("guidance_map.csv")

def rule_lives_alone(t):
    return bool(re.search(r"lives alone|by (him|her)self", t, re.I))

def rule_booked(t):
    if re.search(r"not yet scheduled|to call .* to schedule|no follow-up appointment", t, re.I):
        return False
    return bool(re.search(r"booked|confirmed|scheduled|is set for", t, re.I))

D["LA_RULE"] = D.NOTE_TEXT.map(rule_lives_alone)
D["FB_RULE"] = D.NOTE_TEXT.map(rule_booked)
X = D.merge(L, on="ENCOUNTER_ID").merge(O, on="ENCOUNTER_ID")
short = X.sort_values(["RISK_SCORE", "ENCOUNTER_ID"], ascending=[False, True]).head(60).copy()
short["SCORE_RANK"] = range(1, 61)

def acc(df):
    return {"lives_alone": round(float((df.LA_RULE == (df.LIVES_ALONE == 1)).mean()), 3),
            "followup": round(float((df.FB_RULE == (df.FOLLOWUP_BOOKED == 1)).mean()), 3)}

acc_all, acc_short = acc(X), acc(short)
miss = X[(X.LA_RULE != (X.LIVES_ALONE == 1)) | (X.FB_RULE != (X.FOLLOWUP_BOOKED == 1))]
print(f"Rule reader, all 300 notes: lives_alone acc {acc_all['lives_alone']}, followup acc {acc_all['followup']}")
print(f"Rule reader, 60 shortlisted: lives_alone acc {acc_short['lives_alone']}, followup acc {acc_short['followup']}")
print(f"Rule misses: {len(miss)} notes. Examples:")
for t in miss.NOTE_TEXT.head(3):
    print("   ", re.sub(r"^.*?months\. ", "", t)[:150])

# ---- worklist SQL (same logic as 05_worklist.sql) in DuckDB ----
con = duckdb.connect()
con.register("SHORTLIST_DF", short)
con.register("BASELINE_RATES", B)
con.register("GUIDANCE_MAP", G)
def worklist(la, fb):
    q = f"""
    WITH f AS (SELECT ENCOUNTER_ID, {la} AS LA, {fb} AS FB FROM SHORTLIST_DF)
    SELECT s.ENCOUNTER_ID, s.SERVICE_LINE, s.SCORE_RANK, ROUND(s.RISK_SCORE, 3) AS RISK_SCORE, f.LA, f.FB,
           ROUND(s.RISK_SCORE + 0.10 * CASE WHEN f.LA THEN 1 ELSE 0 END + 0.12 * CASE WHEN f.FB THEN 0 ELSE 1 END, 3) AS PRIORITY,
           b.DAY39_READMIT_RATE, g.TITLE, s.READMIT_30D
    FROM SHORTLIST_DF s JOIN f ON f.ENCOUNTER_ID = s.ENCOUNTER_ID
    LEFT JOIN BASELINE_RATES b ON b.SERVICE_LINE = s.SERVICE_LINE
    LEFT JOIN GUIDANCE_MAP g ON g.SERVICE_LINE = s.SERVICE_LINE
    QUALIFY ROW_NUMBER() OVER (ORDER BY PRIORITY DESC, s.ENCOUNTER_ID) <= 25"""
    return con.execute(q).df()

w_rule = worklist("LA_RULE", "FB_RULE")
w_key = worklist("LIVES_ALONE = 1", "FOLLOWUP_BOOKED = 1")
top_score = short.head(25)
res = {
    "base rate (all 300)": (int(O.READMIT_30D.sum()), round(float(O.READMIT_30D.mean()), 3), 300),
    "score only, top 25": (int(top_score.READMIT_30D.sum()), round(float(top_score.READMIT_30D.mean()), 3), 25),
    "score + rule fields, top 25": (int(w_rule.READMIT_30D.sum()), round(float(w_rule.READMIT_30D.mean()), 3), 25),
    "score + answer-key fields, top 25": (int(w_key.READMIT_30D.sum()), round(float(w_key.READMIT_30D.mean()), 3), 25),
}
print("\nReadmissions caught (simulated outcome):")
for k, (c, p, n) in res.items():
    print(f"  {k:36s} {c:3d} of {n:3d}  precision {p:.3f}")
print(f"Rule worklist: {int((w_rule.SCORE_RANK > 25).sum())} patients pulled up from score ranks 26-60, "
      f"{int((~w_rule.FB).sum())} with no follow-up booked, {int(w_rule.LA.sum())} living alone")
total_readmits_short = int(short.READMIT_30D.sum())
print(f"Readmissions inside the 60-patient shortlist: {total_readmits_short}")

# AUC of the score vs the outcome, and with the answer-key fields added (how much signal the notes carry)
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    r = pd.Series(s).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
auc_score = auc(X.READMIT_30D, X.RISK_SCORE)
auc_key = auc(X.READMIT_30D, X.RISK_SCORE + .10 * X.LIVES_ALONE + .12 * (1 - X.FOLLOWUP_BOOKED))
print(f"AUC on all 300: score {auc_score:.3f}, score + answer-key note fields {auc_key:.3f}")

# ---- cost arithmetic (estimates, no rates) ----
PROMPT = ("Read this simulated hospital discharge note. lives_alone: true only if the patient has no one living with them. "
          "followup_booked: true only if a follow-up appointment has a confirmed date. Note: ")
tok = lambda s: len(s) / 4.0                          # rough rule of thumb: about 4 characters per token in English
in_short = sum(tok(PROMPT + t) for t in short.NOTE_TEXT) + 60 * 45   # + schema overhead guess
in_all = sum(tok(PROMPT + t) for t in X.NOTE_TEXT) + 300 * 45
out_tok = 15
print(f"\nToken estimate (chars/4, plus ~45 tokens of schema per call): 60 notes ~{in_short:,.0f} in + {60*out_tok} out; "
      f"all 300 ~{in_all:,.0f} in + {300*out_tok} out")

# Warehouse: a simulated 80-minute session of 30 statements with human think time between them.
rng = np.random.default_rng(400)
gaps = np.concatenate([rng.uniform(20, 90, 18), rng.uniform(120, 420, 12)])
rng.shuffle(gaps)
durs = rng.uniform(1, 12, 30)
durs[[5, 9, 14]] = [35, 25, 20]                       # 02 load, 03 extraction CTAS, 05 worklist + brief
starts = np.concatenate([[0], np.cumsum(durs[:-1] + gaps[:-1])])
def billed(auto_suspend):
    total, resumes, awake_until, run_start = 0.0, 0, -1e9, None
    for s, d in zip(starts, durs):
        if s > awake_until:                           # warehouse was suspended: resume, 60 s minimum
            if run_start is not None:
                total += max(60, awake_until - run_start)
            run_start, resumes = s, resumes + 1
        awake_until = max(awake_until, s + d + auto_suspend)
    total += max(60, awake_until - run_start)
    return round(total), resumes
wh = {}
for a in [60, 300, 600]:
    sec, r = billed(a)
    wh[str(a)] = {"seconds": sec, "credits": round(sec / 3600, 3), "resumes": r}
sess_min = (starts[-1] + durs[-1]) / 60
print(f"Warehouse, simulated {sess_min:.0f}-minute session, 30 statements, X-Small (1 credit/hour):")
for a, v in wh.items():
    print(f"  AUTO_SUSPEND {a:>3}: {v['seconds']:,} billed seconds = {v['credits']} credits, {v['resumes']} resumes")

card = {"n": 300, "lives_alone_rate": round(float(L.LIVES_ALONE.mean()), 3), "booked_rate": round(float(L.FOLLOWUP_BOOKED.mean()), 3),
        "readmit_rate": round(float(O.READMIT_30D.mean()), 3), "mean_score": round(float(D.RISK_SCORE.mean()), 3),
        "rule_acc_all": acc_all, "rule_acc_short": acc_short, "rule_misses": int(len(miss)),
        "lists": {k: {"caught": c, "precision": p, "n": n} for k, (c, p, n) in res.items()},
        "pulled_up": int((w_rule.SCORE_RANK > 25).sum()), "no_fu_rule": int((~w_rule.FB).sum()),
        "la_rule": int(w_rule.LA.sum()), "short_readmits": total_readmits_short,
        "auc_score": round(float(auc_score), 3), "auc_key": round(float(auc_key), 3),
        "tok_short": round(in_short), "tok_all": round(in_all), "out_tok": out_tok,
        "wh": wh, "session_min": round(sess_min), "starts": starts.round(1).tolist(), "durs": durs.round(1).tolist(),
        "worklist_rule": w_rule[["ENCOUNTER_ID", "SERVICE_LINE", "SCORE_RANK", "RISK_SCORE", "LA", "FB", "PRIORITY", "READMIT_30D"]]
                         .astype({"LA": int, "FB": int}).to_dict("records")}
json.dump(card, open("baseline_card.json", "w"), indent=1)
print("\nwrote baseline_card.json")
