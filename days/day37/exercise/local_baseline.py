# Day 37: a NO-LLM baseline you can run anywhere (no Snowflake, no tokens, no cost).
# Keyword rules for readmission risk, a regex for follow-up timing, a word list for feedback tone.
# Then a token estimate (about 4 characters per token) so you can size the Cortex run before you pay for it.
# Compare these numbers with 05_evaluate.sql after you run Cortex on the trial.
import csv, json, re
from collections import Counter

notes = list(csv.DictReader(open("discharge_notes.csv")))
labels = {r["NOTE_ID"]: r for r in csv.DictReader(open("note_labels.csv"))}

def risk_keywords(t):
    t = t.lower()
    hits = [k for k in ["lives alone", "prior admissions", "short of breath", "shortness of breath",
                        "heart failure", "copd", "sepsis", "falls"] if k in t]
    band = "high" if len(hits) >= 3 else ("medium" if len(hits) >= 1 else "low")
    return band, hits

def followup_regex(t):
    m = re.search(r"follow up with ([a-z ]+?) in (\d+) (day|days|week|weeks|month|months)\b", t.lower())
    if not m:
        return None, None
    n = int(m.group(2)) * (7 if m.group(3).startswith("week") else 30 if m.group(3).startswith("month") else 1)
    return n, m.group(1)

POS = ["kind", "great", "clearly", "easy", "safe", "friendly", "clean", "thank", "listened", "fine"]
NEG = ["confused", "worried", "waited", "rushed", "terrible", "unanswered", "ignored", "do not know", "hard"]
def tone_words(t):
    t = t.lower()
    p, n = sum(w in t for w in POS), sum(w in t for w in NEG)
    return "positive" if p > n else ("negative" if n > p else "neutral")

rows = []
for r in notes:
    L = labels[r["NOTE_ID"]]
    band, hits = risk_keywords(r["NOTE_TEXT"])
    fu, spec = followup_regex(r["NOTE_TEXT"])
    true_fu = int(L["TRUE_FOLLOWUP_DAYS"]) if L["TRUE_FOLLOWUP_DAYS"] else None
    rows.append(dict(id=int(r["NOTE_ID"]), true_band=L["TRUE_RISK_BAND"], kw_band=band, hits=hits,
                     true_fu=true_fu, rx_fu=fu, true_tone=L["TRUE_FEEDBACK_TONE"],
                     kw_tone=tone_words(r["PATIENT_FEEDBACK"]),
                     note_chars=len(r["NOTE_TEXT"]), fb_chars=len(r["PATIENT_FEEDBACK"])))

acc = lambda a, b: round(sum(x[a] == x[b] for x in rows) / len(rows), 3)
risk_acc, fu_acc, tone_acc = acc("true_band", "kw_band"), acc("true_fu", "rx_fu"), acc("true_tone", "kw_tone")
bands = ["low", "medium", "high"]
conf = [[sum(1 for x in rows if x["true_band"] == t and x["kw_band"] == p) for p in bands] for t in bands]
tones = ["positive", "neutral", "negative"]
tconf = [[sum(1 for x in rows if x["true_tone"] == t and x["kw_tone"] == p) for p in tones] for t in tones]
trap = [x["id"] for x in rows if "shortness of breath" in x["hits"] or "falls" in x["hits"] or
        ("prior admissions" in x["hits"] and x["true_band"] == "low")]

# Token sizing: your text only (about 4 chars per token). Billed tokens are HIGHER because SUMMARIZE,
# CLASSIFY_TEXT, SENTIMENT and EXTRACT_ANSWER add their own prompt, and COMPLETE also bills output tokens.
note_tok = sum(x["note_chars"] for x in rows) / 4
fb_tok = sum(x["fb_chars"] for x in rows) / 4
q_tok = len("In how many days should the patient follow up?") / 4
plan = {  # one full pass of 03_cortex_functions.sql part D, your text only
    "SUMMARIZE (notes in + summary out ~40 tok/row)": note_tok + 40 * len(rows),
    "CLASSIFY_TEXT (notes + 3 labels w/ descriptions ~90 tok/row)": note_tok + 90 * len(rows),
    "EXTRACT_ANSWER x2 (notes + question, twice)": 2 * (note_tok + q_tok * len(rows)),
    "SENTIMENT (feedback only)": fb_tok,
}
total = sum(plan.values())

print(f"notes: {len(rows)} | avg note chars {sum(x['note_chars'] for x in rows)/len(rows):.0f} "
      f"| avg feedback chars {sum(x['fb_chars'] for x in rows)/len(rows):.0f}")
print(f"keyword risk accuracy   : {risk_acc:.3f}")
print("  confusion (rows = true low/medium/high, cols = keyword low/medium/high):", conf)
print(f"  notes hit by a negation or 'No prior admissions' trap: {len(trap)} -> {trap[:12]}")
print(f"regex follow-up accuracy: {fu_acc:.3f}  (templated text; real notes are messier)")
print(f"word-list tone accuracy : {tone_acc:.3f}")
print("  tone confusion (rows = true pos/neu/neg):", tconf)
print("estimated input tokens for one full Cortex pass (your text only, chars/4):")
for k, v in plan.items():
    print(f"  {k:58s} {v:8.0f}")
print(f"  {'TOTAL':58s} {total:8.0f}  (= {total/1e6:.4f} million tokens before built-in prompts)")
print("credits = (billed tokens / 1,000,000) x credits-per-million for that function/model")
print("rates: Snowflake Service Consumption Table (not copied here; check the current PDF)")

card = dict(n=len(rows), risk_acc=risk_acc, fu_acc=fu_acc, tone_acc=tone_acc, conf=conf, tconf=tconf,
            trap_ids=trap, band_counts=dict(Counter(x["true_band"] for x in rows)),
            tone_counts=dict(Counter(x["true_tone"] for x in rows)),
            no_followup=sum(1 for x in rows if x["true_fu"] is None),
            avg_note_chars=round(sum(x["note_chars"] for x in rows) / len(rows)),
            avg_fb_chars=round(sum(x["fb_chars"] for x in rows) / len(rows)),
            note_chars=[x["note_chars"] for x in rows],
            plan={k: round(v) for k, v in plan.items()}, plan_total=round(total),
            example_note=notes[0]["NOTE_TEXT"], example_feedback=notes[0]["PATIENT_FEEDBACK"])
json.dump(card, open("baseline_card.json", "w"), indent=1)
print("wrote baseline_card.json")
