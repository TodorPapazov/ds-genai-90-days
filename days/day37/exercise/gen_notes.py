# Day 37: build ~40 SIMULATED hospital discharge notes plus answer-key labels.
# Not real patients. Names, IDs, and dates are made up. Seed 37 makes it repeatable.
# Writes: discharge_notes.csv, note_labels.csv, 02_load_notes.sql (paste into a Snowsight worksheet).
import csv, random, datetime as dt

random.seed(37)
N = 40
DX = {
    "Heart failure":        ("cardiology", "furosemide and metoprolol"),
    "COPD exacerbation":    ("pulmonology", "an albuterol inhaler and a prednisone taper"),
    "Pneumonia":            ("primary care", "amoxicillin-clavulanate"),
    "Type 2 diabetes, hyperglycemia": ("endocrinology", "insulin glargine"),
    "Hip replacement":      ("orthopedics", "apixaban and acetaminophen"),
    "Sepsis, urinary source": ("infectious disease", "cephalexin"),
    "Atrial fibrillation":  ("cardiology", "apixaban and diltiazem"),
    "Cellulitis":           ("primary care", "cephalexin"),
}
WORD = {3: "3 days", 5: "5 days", 7: "7 days", 10: "10 days", 14: "2 weeks", 30: "1 month"}
POS = ["The nurses were kind and explained every medication clearly.",
       "Great care. The discharge instructions were easy to follow.",
       "Dr. Lee listened to us and the team made me feel safe.",
       "Everyone was friendly and the room was clean. Thank you."]
NEU = ["The stay was okay. Food was average.",
       "Discharge took a few hours but the paperwork was fine.",
       "Nothing special to report. Parking was hard to find.",
       "The care was fine. I am not sure about the new pills yet."]
NEG = ["Nobody explained my new medicines and I am confused and worried.",
       "I waited six hours for discharge and the staff seemed rushed.",
       "Terrible night, the call light went unanswered for an hour.",
       "I felt ignored and I still do not know who to call if I get worse."]

def risk(age, prior, lives_alone, los, meds, dx):
    s = (age >= 75) + 2 * (prior >= 2) + (prior == 1) + lives_alone + (los >= 7) + (meds >= 10)
    s += dx in ("Heart failure", "COPD exacerbation", "Sepsis, urinary source")
    return "high" if s >= 4 else ("medium" if s >= 2 else "low"), s

rows, labels = [], []
d0 = dt.date(2026, 10, 1)
for i in range(1, N + 1):
    dx = random.choice(list(DX))
    spec, meds_txt = DX[dx]
    age = random.randint(38, 91)
    los = max(1, int(random.gammavariate(2.0, 2.2)))
    prior = random.choices([0, 1, 2, 3], [.45, .25, .2, .1])[0]
    alone = int(random.random() < .35)
    meds = random.randint(3, 15)
    band, score = risk(age, prior, alone, los, meds, dx)
    sex = random.choice(["male", "female"])
    has_fu = random.random() < .85
    fu_days = random.choice(list(WORD)) if has_fu else None
    if band == "high" and has_fu:
        fu_days = random.choice([3, 5, 7])
    when = d0 + dt.timedelta(days=random.randint(0, 45))
    s = [f"{age}-year-old {sex} admitted for {dx[0].lower() + dx[1:] if not dx.startswith("COPD") else dx}, length of stay {los} day{"s" if los != 1 else ""}."]
    s.append(f"{'One prior admission' if prior == 1 else (str(prior) + ' prior admissions' if prior else 'No prior admissions')} in the last 12 months.")
    s.append("Lives alone." if alone else random.choice(["Lives with spouse.", "Lives with daughter.", "Lives with family."]))
    s.append(f"Discharged home on {meds_txt}; {meds} active medications at discharge.")
    if dx == "Heart failure":
        s.append(random.choice(["Weight down 3 kg after diuresis.", "Denies chest pain. Mild ankle swelling remains."]))
    if dx == "COPD exacerbation":
        s.append(random.choice(["Still short of breath on stairs.", "Denies shortness of breath at rest."]))
    if dx.startswith("Sepsis"):
        s.append("Afebrile for 48 hours before discharge.")
    if band == "low" and random.random() < .4:
        s.append("No history of falls. Denies shortness of breath.")   # negation trap for keyword rules
    if has_fu:
        s.append(f"Follow up with {spec} in {WORD[fu_days]}.")
    else:
        s.append("No follow-up appointment was scheduled before discharge.")
    s.append(random.choice(["Return to the ED for fever, chest pain, or trouble breathing.",
                            "Call the clinic with any new symptoms."]))
    tone = random.choices(["positive", "neutral", "negative"], [.45, .25, .30])[0]
    fb = random.choice({"positive": POS, "neutral": NEU, "negative": NEG}[tone])
    rows.append(dict(NOTE_ID=i, PATIENT_ID=f"SIM{1000 + i}", DISCHARGE_DATE=when.isoformat(), AGE=age,
                     DIAGNOSIS=dx, LOS_DAYS=los, NOTE_TEXT=" ".join(s), PATIENT_FEEDBACK=fb))
    labels.append(dict(NOTE_ID=i, TRUE_RISK_BAND=band, RISK_POINTS=score,
                       TRUE_FOLLOWUP_DAYS=fu_days if has_fu else "",
                       TRUE_FOLLOWUP_WITH=spec if has_fu else "", TRUE_FEEDBACK_TONE=tone))

def wcsv(path, data):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)

wcsv("discharge_notes.csv", rows)
wcsv("note_labels.csv", labels)

q = lambda x: "NULL" if x in ("", None) else ("'" + str(x).replace("'", "''") + "'" if isinstance(x, str) else str(x))
sql = ["-- Day 37: load 40 SIMULATED discharge notes and the answer key (generated by gen_notes.py, seed 37).",
       "-- Not real patient data. Run after 01_setup.sql.",
       "USE WAREHOUSE DS90_XS;", "USE SCHEMA DS90.DAY37;", "",
       "CREATE OR REPLACE TABLE DISCHARGE_NOTES (", "  NOTE_ID INT, PATIENT_ID STRING, DISCHARGE_DATE DATE, AGE INT,",
       "  DIAGNOSIS STRING, LOS_DAYS INT, NOTE_TEXT STRING, PATIENT_FEEDBACK STRING);", "",
       "INSERT INTO DISCHARGE_NOTES VALUES"]
sql += [",\n".join("  (" + ", ".join(q(r[k]) for k in rows[0]) + ")" for r in rows) + ";", ""]
sql += ["CREATE OR REPLACE TABLE NOTE_LABELS (", "  NOTE_ID INT, TRUE_RISK_BAND STRING, RISK_POINTS INT,",
        "  TRUE_FOLLOWUP_DAYS INT, TRUE_FOLLOWUP_WITH STRING, TRUE_FEEDBACK_TONE STRING);", "",
        "INSERT INTO NOTE_LABELS VALUES"]
sql += [",\n".join("  (" + ", ".join(q(r[k]) for k in labels[0]) + ")" for r in labels) + ";", ""]
sql += ["SELECT COUNT(*) AS N_NOTES, AVG(LENGTH(NOTE_TEXT)) AS AVG_CHARS FROM DISCHARGE_NOTES;",
        "SELECT TRUE_RISK_BAND, COUNT(*) FROM NOTE_LABELS GROUP BY 1 ORDER BY 1;"]
open("02_load_notes.sql", "w").write("\n".join(sql) + "\n")

from collections import Counter
print(f"wrote {len(rows)} notes; risk bands {dict(Counter(l['TRUE_RISK_BAND'] for l in labels))}; "
      f"tones {dict(Counter(l['TRUE_FEEDBACK_TONE'] for l in labels))}; "
      f"no follow-up {sum(1 for l in labels if l['TRUE_FOLLOWUP_DAYS'] == '')}")
print("example note 1:", rows[0]["NOTE_TEXT"])
print("example feedback 1:", rows[0]["PATIENT_FEEDBACK"])
