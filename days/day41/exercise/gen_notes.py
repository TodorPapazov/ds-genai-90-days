# Day 41: build 30 SIMULATED hospital discharge notes plus an answer key.
# Not real patients. Names, IDs and dates are made up. Seed 41 makes it repeatable.
# Writes: discharge_notes.csv (the input) and note_labels.csv (the answer key for grading).
import csv, random, datetime as dt

random.seed(41)
N = 30
DX = {
    "Heart failure": ("cardiology", "furosemide and metoprolol"),
    "COPD exacerbation": ("pulmonology", "an albuterol inhaler and a prednisone taper"),
    "Pneumonia": ("primary care", "amoxicillin-clavulanate"),
    "Type 2 diabetes, hyperglycemia": ("endocrinology", "insulin glargine"),
    "Hip replacement": ("orthopedics", "apixaban and acetaminophen"),
    "Sepsis, urinary source": ("infectious disease", "cephalexin"),
    "Atrial fibrillation": ("cardiology", "apixaban and diltiazem"),
    "Cellulitis": ("primary care", "cephalexin"),
}
WORDS = {7: ["in 7 days", "in one week", "next week"], 14: ["in 14 days", "in two weeks", "in 2 weeks"],
         3: ["in 3 days", "within 3 days"], 5: ["in 5 days"], 10: ["in 10 days"], 30: ["in 1 month", "in 30 days"]}
LIVES = {True: ["Lives alone.", "Lives by herself.", "Lives by himself.", "No caregiver at home; lives alone."],
         False: ["Lives with family.", "Lives with spouse.", "Daughter lives with the patient.", "Lives with a partner."]}
NO_FU = ["Follow-up appointment not yet scheduled; clinic to call patient.",
         "No follow-up booked at discharge.",
         "Patient declined to schedule follow-up today."]

rows, labels = [], []
d0 = dt.date(2026, 11, 9)
for i in range(1, N + 1):
    dx = random.choice(list(DX))
    spec, meds = DX[dx]
    age = random.randint(38, 91)
    sex = random.choice(["female", "male"])
    los = max(1, int(random.gammavariate(2.0, 2.2)))
    alone = random.random() < .33
    lv = random.choice([s for s in LIVES[alone] if not (("herself" in s and sex == "male") or ("himself" in s and sex == "female"))])
    booked = random.random() < .72
    style = random.random()
    if booked:
        days = random.choice([3, 5, 7, 7, 10, 14, 14, 30])
        phr = random.choice(WORDS[days])
        if style < .7:
            fu = f"Follow up with {spec} {phr}."
        else:
            fu = f"{spec.capitalize()} appointment {phr}."
    else:
        days, spec_out = None, None
        fu = random.choice(NO_FU)
    date = d0 + dt.timedelta(days=random.randint(0, 6))
    los_txt = f"length of stay {los} day{'s' if los != 1 else ''}" if random.random() < .8 else f"discharged on hospital day {los}"
    note = (f"{age}-year-old {sex} admitted for {dx if dx.startswith("COPD") else dx.lower()}, {los_txt}. {lv} "
            f"Discharged home on {meds}. Vitals stable at discharge. {fu} "
            f"Return to the ED for chest pain, trouble breathing or fever.")
    rows.append([i, f"SIM41{i:03d}", date.isoformat(), age, note])
    labels.append([i, dx, los, int(alone), int(booked), "" if days is None else days, "" if not booked else spec])

with open("discharge_notes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["NOTE_ID", "PATIENT_ID", "DISCHARGE_DATE", "AGE", "NOTE_TEXT"]); w.writerows(rows)
with open("note_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["NOTE_ID", "TRUE_DIAGNOSIS", "TRUE_LOS_DAYS", "TRUE_LIVES_ALONE", "TRUE_FOLLOWUP_BOOKED", "TRUE_FOLLOWUP_DAYS", "TRUE_FOLLOWUP_WITH"])
    w.writerows(labels)
print(f"wrote {N} simulated notes to discharge_notes.csv and the answer key to note_labels.csv")
