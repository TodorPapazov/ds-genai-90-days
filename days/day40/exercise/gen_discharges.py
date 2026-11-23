"""Day 40: one week of SIMULATED discharges for the mini project (seed 40).
Writes discharges.csv (what Snowflake sees), note_labels.csv (answer key for the note fields),
outcomes.csv (simulated 30-day readmission, the thing we pretend to learn about later),
baseline_rates.csv and guidance_map.csv (fallbacks if your Day 38 / Day 39 objects are gone),
and 02_load.sql. No Snowflake needed. Every row is invented. No real patients, no real hospital."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(40)
N = 300
LINES = ["Cardiology", "Pulmonary", "Nephrology", "Infectious disease", "Gastroenterology",
         "General medicine", "Endocrinology", "Orthopedics", "Behavioral health"]
LINE_P = [.17, .15, .11, .11, .11, .11, .09, .09, .06]
LINE_EFF = {"Cardiology": .45, "Pulmonary": .25, "Nephrology": .30, "Infectious disease": .20, "Gastroenterology": -.15,
            "General medicine": -.30, "Endocrinology": .05, "Orthopedics": -.45, "Behavioral health": .10}
DX = {"Cardiology": ["heart failure", "chest pain", "atrial fibrillation"],
      "Pulmonary": ["COPD exacerbation", "pneumonia"], "Nephrology": ["acute kidney injury", "urinary tract infection"],
      "Infectious disease": ["sepsis", "cellulitis with sepsis"], "Gastroenterology": ["GI bleed", "gastroenteritis"],
      "General medicine": ["cellulitis", "dehydration"], "Endocrinology": ["diabetic ketoacidosis", "hypoglycemia"],
      "Orthopedics": ["hip replacement", "knee replacement"], "Behavioral health": ["alcohol withdrawal", "depression"]}
CLINIC = {"Cardiology": "the heart failure clinic", "Pulmonary": "pulmonology", "Nephrology": "nephrology",
          "Infectious disease": "infectious disease clinic", "Gastroenterology": "GI clinic", "General medicine": "primary care",
          "Endocrinology": "the diabetes clinic", "Orthopedics": "orthopedic surgery", "Behavioral health": "outpatient behavioral health"}

line = rng.choice(LINES, N, p=LINE_P)
age = rng.normal(66, 15, N).clip(19, 98).round().astype(int)
sex = rng.choice(["F", "M"], N, p=[.52, .48])
payer = np.where(age >= 65, rng.choice(["Medicare", "Commercial", "Medicaid", "Self-pay"], N, p=[.82, .10, .06, .02]),
                 rng.choice(["Medicare", "Commercial", "Medicaid", "Self-pay"], N, p=[.06, .56, .28, .10]))
dispo = rng.choice(["Home", "Home health", "SNF", "AMA"], N, p=[.60, .17, .18, .05])
los = np.round(rng.gamma(2.2, 2.0, N) + 0.5, 1)
prior = rng.poisson(0.7 + 0.6 * (age > 75), N).clip(0, 6)
meds = rng.poisson(8 + 3 * (age > 70), N).clip(1, 22)
lives_alone = rng.random(N) < np.where(age > 75, .38, .22)
booked = rng.random(N) < np.where(dispo == "AMA", .25, .62)
fu_days = rng.choice([3, 5, 7, 7, 10, 14], N)
start = pd.Timestamp("2026-11-16")
ddate = start + pd.to_timedelta(rng.integers(0, 7, N), unit="D")

# The "model" score sees ONLY structured columns (a stand-in for the Day 36 registry model).
lin_struct = (-2.25 + .018 * (age - 65) + .42 * prior + .07 * (los - 4) + np.vectorize(LINE_EFF.get)(line)
              + np.select([dispo == "AMA", dispo == "SNF", dispo == "Home health"], [.65, .30, .10], 0)
              + np.select([payer == "Medicaid", payer == "Self-pay"], [.25, .20], 0))
score = 1 / (1 + np.exp(-(lin_struct + rng.normal(0, .25, N))))
# The simulated outcome ALSO depends on two facts that live only in the note text.
lin_true = lin_struct + .95 * lives_alone + 1.0 * (~booked)
p_true = 1 / (1 + np.exp(-lin_true))
readmit = (rng.random(N) < p_true).astype(int)

ALONE = ["Lives alone.", "Lives alone in a second-floor apartment.", "Patient lives by herself.",
         "Widowed, no one else at home.", "Lives on his own; son lives out of state."]
WITH = ["Lives with spouse.", "Lives with daughter, who will help with medications.", "Lives with family.",
        "Wife at home and able to drive him to visits.", "Lives with a roommate who checks on her daily."]
BOOK = ["Follow-up appointment booked with {c} on {d}.", "{C} visit confirmed for {d}.",
        "Seen by case management; appointment with {c} scheduled for {d}.", "Follow-up with {c} is set for {d}.", "Has an appointment with {c} on {d}."]
NOBOOK = ["Follow up with {c} in {k} days; appointment not yet scheduled.", "Patient to call {c} to schedule a visit within {k} days.",
          "No follow-up appointment was made before discharge.", "Scheduling with {c} was attempted, but no openings were found this week.",
          "Advised to see {c} in {k} days.", "The {c} office will call to set up a visit; nothing scheduled yet."]
DISPO_TXT = {"Home": "Discharged home", "Home health": "Discharged home with home health", "SNF": "Discharged to a skilled nursing facility",
             "AMA": "Left against medical advice"}
notes = []
for i in range(N):
    he = "her" if sex[i] == "F" else "him"
    a = (ALONE[rng.integers(len(ALONE))] if lives_alone[i] else WITH[rng.integers(len(WITH))])
    if sex[i] == "M":
        a = a.replace("herself", "himself").replace("checks on her", "checks on him")
    else:
        a = a.replace("drive him", "drive her").replace("his own", "her own").replace("Wife", "Husband")
    when = (ddate[i] + pd.Timedelta(days=int(fu_days[i]))).strftime("%b %d").replace(" 0", " ")
    c = CLINIC[line[i]]
    f = (BOOK if booked[i] else NOBOOK)[rng.integers(len(BOOK) if booked[i] else len(NOBOOK))].format(c=c, C=c[0].upper() + c[1:], d=when, k=fu_days[i])
    dx = DX[line[i]][rng.integers(len(DX[line[i]]))]
    notes.append(f"{age[i]}-year-old {'female' if sex[i] == 'F' else 'male'} admitted for {dx}, length of stay {los[i]:g} days. "
                 f"{prior[i]} admission{'s' if prior[i] != 1 else ''} in the last 12 months. {a} "
                 f"{DISPO_TXT[dispo[i]]} on {meds[i]} active medications. {f} SIMULATED NOTE.")

ids = [f"W{40000 + i + 1}" for i in range(N)]
dis = pd.DataFrame({"ENCOUNTER_ID": ids, "PATIENT_ID": [f"S{50000 + i + 1}" for i in range(N)],
                    "DISCHARGE_DATE": ddate.date, "AGE": age, "SEX": sex, "PAYER": payer, "SERVICE_LINE": line,
                    "DISCHARGE_DISPOSITION": dispo, "LOS_DAYS": los, "PRIOR_ADMITS_12M": prior, "MED_COUNT": meds,
                    "RISK_SCORE": score.round(4), "NOTE_TEXT": notes})
lab = pd.DataFrame({"ENCOUNTER_ID": ids, "LIVES_ALONE": lives_alone.astype(int), "FOLLOWUP_BOOKED": booked.astype(int)})
out = pd.DataFrame({"ENCOUNTER_ID": ids, "READMIT_30D": readmit})

# Fallbacks: Day 39 readmission rate by service line (2024-2025 simulated encounters) and a Day 38 guidance doc per line.
e = pd.read_csv("encounters.csv", dtype={"DRG_CODE": str}).merge(pd.read_csv("drg_groups.csv", dtype={"DRG_CODE": str}))
base = e.groupby("SERVICE_LINE").READMIT_30D.agg(["mean", "size"]).reset_index()
base.columns = ["SERVICE_LINE", "DAY39_READMIT_RATE", "DAY39_ENCOUNTERS"]
base["DAY39_READMIT_RATE"] = base.DAY39_READMIT_RATE.round(3)
guide = pd.DataFrame([
    ("Cardiology", "DOC001", "Heart Failure Discharge Instructions"),
    ("Pulmonary", "DOC006", "COPD Exacerbation Discharge Instructions"),
    ("Nephrology", "DOC027", "Kidney-Friendly Eating (CKD Diet Education)"),
    ("Infectious disease", "DOC009", "Community-Acquired Pneumonia Care Protocol"),
    ("Gastroenterology", "DOC023", "Post-Discharge Follow-up Call Protocol"),
    ("General medicine", "DOC023", "Post-Discharge Follow-up Call Protocol"),
    ("Endocrinology", "DOC016", "Diabetes Discharge Instructions"),
    ("Orthopedics", "DOC013", "Total Hip Replacement Discharge Instructions"),
    ("Behavioral health", "DOC023", "Post-Discharge Follow-up Call Protocol")],
    columns=["SERVICE_LINE", "DOC_ID", "TITLE"])

dis.to_csv("discharges.csv", index=False)
lab.to_csv("note_labels.csv", index=False)
out.to_csv("outcomes.csv", index=False)
base.to_csv("baseline_rates.csv", index=False)
guide.to_csv("guidance_map.csv", index=False)

def q(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return "'" + str(v).replace("'", "''") + "'"

def inserts(table, df, chunk=100):
    rows = [f"({', '.join(q(v) for v in r)})" for r in df.itertuples(index=False)]
    return "\n".join(f"INSERT INTO {table} VALUES\n" + ",\n".join(rows[i:i + chunk]) + ";" for i in range(0, len(rows), chunk))

sql = f"""-- Day 40: load the SIMULATED week of discharges (generated by gen_discharges.py, seed 40). Not real patients.
-- 300 discharges Nov 16-22, 2026. RISK_SCORE is a simulated model probability that only saw structured columns
-- (a stand-in for the Day 36 registry model). Two facts live ONLY in NOTE_TEXT: lives alone, follow-up booked.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

CREATE OR REPLACE TABLE DISCHARGES (ENCOUNTER_ID STRING, PATIENT_ID STRING, DISCHARGE_DATE DATE, AGE INT, SEX STRING,
  PAYER STRING, SERVICE_LINE STRING, DISCHARGE_DISPOSITION STRING, LOS_DAYS FLOAT, PRIOR_ADMITS_12M INT, MED_COUNT INT,
  RISK_SCORE FLOAT, NOTE_TEXT STRING) COMMENT = 'SIMULATED discharges for Day 40. Not real patients.';
{inserts("DISCHARGES", dis)}

-- Answer key for the note fields (to grade the LLM extraction) and the simulated outcome (to grade the worklist).
CREATE OR REPLACE TABLE NOTE_LABELS (ENCOUNTER_ID STRING, LIVES_ALONE INT, FOLLOWUP_BOOKED INT);
{inserts("NOTE_LABELS", lab)}
CREATE OR REPLACE TABLE OUTCOMES (ENCOUNTER_ID STRING, READMIT_30D INT) COMMENT = 'SIMULATED outcome, for grading only';
{inserts("OUTCOMES", out)}

-- Fallbacks if DS90.DAY39 (semantic view) or DS90.DAY38 (documents) are gone. Same numbers, plain tables.
CREATE OR REPLACE TABLE BASELINE_RATES (SERVICE_LINE STRING, DAY39_READMIT_RATE FLOAT, DAY39_ENCOUNTERS INT);
{inserts("BASELINE_RATES", base)}
CREATE OR REPLACE TABLE GUIDANCE_MAP (SERVICE_LINE STRING, DOC_ID STRING, TITLE STRING);
{inserts("GUIDANCE_MAP", guide)}

-- Checks: expect 300 / 300 / 300 / 9 / 9 rows.
SELECT 'DISCHARGES' T, COUNT(*) N FROM DISCHARGES UNION ALL SELECT 'NOTE_LABELS', COUNT(*) FROM NOTE_LABELS
UNION ALL SELECT 'OUTCOMES', COUNT(*) FROM OUTCOMES UNION ALL SELECT 'BASELINE_RATES', COUNT(*) FROM BASELINE_RATES
UNION ALL SELECT 'GUIDANCE_MAP', COUNT(*) FROM GUIDANCE_MAP;
SELECT ROUND(AVG(RISK_SCORE), 3) AS MEAN_SCORE, (SELECT ROUND(AVG(READMIT_30D), 3) FROM OUTCOMES) AS READMIT_RATE FROM DISCHARGES;
"""
open("02_load.sql", "w").write(sql)
print(f"discharges {len(dis)} | lives alone {lab.LIVES_ALONE.mean():.3f} | booked {lab.FOLLOWUP_BOOKED.mean():.3f} | "
      f"mean score {dis.RISK_SCORE.mean():.3f} | readmit rate {out.READMIT_30D.mean():.3f}")
