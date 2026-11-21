# Day 38: build a SIMULATED clinical document library for a first look at RAG with Cortex Search.
# Every title, number and instruction below is invented for practice (seed 38). It is not medical advice,
# not any real hospital's policy, and contains no patient data (no PHI).
# Outputs: clinical_docs.csv (whole documents), doc_chunks.csv (one row per chunk, the search column),
#          eval_questions.csv (labeled question set with the gold chunk), 02_load_docs.sql (Snowsight loader).
import csv, json, random, re

random.seed(38)
D = []   # (title, department, doc_type, effective_date, [paragraphs])
def doc(title, dept, dtype, date, *paras):
    D.append((title, dept, dtype, date, list(paras)))

doc("Heart Failure Discharge Instructions", "Cardiology", "discharge_instructions", "2026-03-01",
    "Weigh yourself every morning after you use the bathroom and before breakfast. Write the number down. Call the heart failure clinic if your weight goes up more than 2 pounds in one day or more than 5 pounds in one week.",
    "Limit sodium to 2,000 mg per day. Read food labels, skip canned soups and deli meats, and do not add salt at the table. Limit all fluids to 2 liters (about 8 cups) per day unless your cardiologist gives you a different number.",
    "Take your water pill (furosemide) in the morning so you are not up all night. Bring every pill bottle to your first clinic visit.",
    "Your first heart failure clinic visit should be within 7 days of discharge. Go to the emergency department if you are short of breath at rest or cannot lie flat to sleep.")
doc("Heart Failure Clinic Medication Titration Protocol", "Cardiology", "care_protocol", "2026-02-12",
    "Clinic pharmacists increase guideline-directed heart failure medicines one step at a time, no more often than every 2 weeks, when blood pressure and heart rate allow.",
    "Check potassium and creatinine 1 to 2 weeks after starting or increasing spironolactone. Hold spironolactone and call the cardiologist if potassium is above 5.5 mmol/L.",
    "Do not increase a beta blocker during a visit when the patient has signs of fluid overload such as new leg swelling or weight gain over 5 pounds.")
doc("After Your Heart Stent (PCI) Discharge Instructions", "Cardiology", "discharge_instructions", "2026-02-10",
    "Do not stop aspirin or clopidogrel unless your cardiologist tells you to. Stopping these antiplatelet medicines early can cause the stent to clot. Plan on taking both for 12 months.",
    "If the catheter went in at your wrist, do not lift more than 5 pounds with that arm for 48 hours. A small bruise is normal. Hold firm pressure and call 911 if the site bleeds and does not stop.",
    "If you have chest pain, take one nitroglycerin under the tongue. If the pain is not better after 5 minutes, call 911.",
    "You will be referred to cardiac rehabilitation. Your first rehab session should be within 2 weeks of discharge.")
doc("Warfarin Patient Education", "Pharmacy", "patient_education", "2025-11-15",
    "Warfarin is a blood thinner. For atrial fibrillation, the usual INR target range is 2.0 to 3.0. Your clinic will tell you your own range.",
    "You do not need to avoid leafy green vegetables. Keep the amount of vitamin K foods about the same from week to week so your INR stays steady.",
    "Watch for bleeding: black or tarry stools, pink or red urine, nosebleeds that last more than 10 minutes, or unusual bruising. Call the anticoagulation clinic the same day.",
    "After any dose change, your INR is checked again in 3 to 5 days.")
doc("Anticoagulation Bridging Protocol for Surgery", "Pharmacy", "care_protocol", "2026-01-20",
    "For high-risk patients on warfarin who need surgery, stop warfarin 5 days before the procedure.",
    "Start enoxaparin 1 mg/kg every 12 hours 3 days before surgery. Give the last enoxaparin dose 24 hours before the procedure.",
    "Restart warfarin the evening after surgery if the surgeon agrees. For low bleeding-risk procedures, resume enoxaparin 24 hours after surgery until the INR is in range.")
doc("COPD Exacerbation Discharge Instructions", "Pulmonology", "discharge_instructions", "2026-01-05",
    "Take prednisone 40 mg once a day with breakfast for 5 days, then stop. Do not taper unless your doctor says so.",
    "Use your inhalers exactly as the respiratory therapist showed you. Bring them to every visit so the team can check your technique with teach-back.",
    "Use your COPD action plan. Green zone: usual breathing. Yellow zone: more cough, mucus or shortness of breath, start your rescue plan and call the clinic. Red zone: trouble talking or blue lips, call 911.",
    "You are referred to pulmonary rehabilitation. Start within 4 weeks of discharge.")
doc("Home Oxygen Safety", "Pulmonology", "patient_education", "2025-09-30",
    "Never smoke or let anyone smoke in a room where oxygen is used. Keep oxygen at least 6 feet away from open flames, gas stoves and candles.",
    "Store oxygen tanks upright and secured in a stand or cart so they cannot fall over. Keep them out of hot cars and away from heaters.",
    "Oxygen tubing can be up to 50 feet long. Tape it along walls to lower the chance of tripping.")
doc("Adult Asthma Action Plan Education", "Pulmonology", "patient_education", "2026-02-25",
    "If you need your rescue inhaler more than 2 days a week, your asthma is not well controlled. Tell your doctor at your next visit.",
    "Rinse your mouth with water and spit after each dose of your steroid inhaler to prevent thrush.",
    "Go to the emergency department if your rescue inhaler does not help within 20 minutes or you have trouble walking or talking.")
doc("Community-Acquired Pneumonia Care Protocol", "Infectious Disease", "care_protocol", "2026-02-01",
    "Switch from IV to oral antibiotics when the patient has been clinically stable for 24 hours and can take pills by mouth.",
    "Treat for at least 5 days in total. Stop only when the patient has had no fever for 48 hours and no more than one sign of instability.",
    "A follow-up chest x-ray at 6 to 8 weeks is recommended only for patients over 50 or current smokers, to rule out an underlying mass.")
doc("Sepsis Early Management Bundle", "Emergency", "care_protocol", "2026-04-01",
    "Measure a lactate level within 1 hour of suspected sepsis. Repeat it within 6 hours if the first lactate is above 2 mmol/L.",
    "Draw blood cultures before giving the first dose of broad-spectrum antibiotics, but do not delay antibiotics more than 45 minutes waiting for cultures.",
    "Give 30 mL/kg of IV crystalloid fluid for low blood pressure or a lactate of 4 mmol/L or higher.")
doc("Clostridioides difficile (C. diff) Isolation Policy", "Infection Prevention", "policy", "2025-12-01",
    "Place patients with suspected or confirmed C. diff on contact plus precautions: gown and gloves for every room entry and a dedicated stethoscope.",
    "Clean hands with soap and water after care. Alcohol hand gel does not kill C. diff spores.",
    "Isolation can stop 48 hours after diarrhea resolves. Clean the room with a bleach-based disinfectant at discharge.")
doc("Hand Hygiene Policy", "Infection Prevention", "policy", "2026-01-01",
    "Clean hands at the five moments: before touching a patient, before a clean or aseptic task, after body fluid exposure, after touching a patient, and after touching patient surroundings.",
    "Artificial nails and nail extensions are not allowed for staff who give direct patient care.",
    "Each unit is audited monthly. The compliance target is 95 percent, and results are posted on the unit quality board.")
doc("Total Hip Replacement Discharge Instructions", "Orthopedics", "discharge_instructions", "2026-03-15",
    "Follow your posterior hip precautions for 6 weeks: do not bend your hip past 90 degrees, do not cross your legs, and do not turn your foot inward.",
    "To prevent blood clots, take aspirin 81 mg twice a day for 35 days after surgery.",
    "Your staples will be removed at your clinic visit 10 to 14 days after surgery. Do not lift more than 10 pounds until your surgeon clears you.",
    "Do not drive for at least 4 weeks and never while taking opioid pain medicine.")
doc("Total Knee Replacement Discharge Instructions", "Orthopedics", "discharge_instructions", "2026-03-15",
    "Your goal is to bend your new knee to 90 degrees within 2 weeks. Do the home exercises 3 times a day.",
    "Ice your knee for 20 minutes at a time, several times a day, with a cloth between the ice and your skin.",
    "Outpatient physical therapy is 3 times a week for 6 to 8 weeks.",
    "Call your surgeon right away for new calf pain, calf swelling or a fever over 101 F.")
doc("Cast Care at Home", "Orthopedics", "patient_education", "2025-10-10",
    "Keep your cast dry. Cover it with a plastic bag to shower. Never push objects like coat hangers inside the cast to scratch.",
    "Keep the injured arm or leg raised above the level of your heart for the first 48 to 72 hours to reduce swelling.",
    "Call the clinic for numbness, tingling, toes or fingers that turn blue or cold, or pain that gets worse when you move them.")
doc("Diabetes Discharge Instructions", "Endocrinology", "discharge_instructions", "2026-02-20",
    "For low blood sugar under 70 mg/dL, use the rule of 15: take 15 grams of fast sugar such as 4 ounces of juice, then recheck in 15 minutes.",
    "Check your blood sugar before each meal and at bedtime. Write the numbers in your log and bring it to clinic.",
    "On sick days, keep taking your long-acting (basal) insulin even if you are not eating. Check your sugar every 4 hours.",
    "See the endocrinology clinic within 2 weeks of discharge.")
doc("Insulin Pen Storage and Use", "Pharmacy", "patient_education", "2025-08-01",
    "Store unopened insulin pens in the refrigerator. Do not freeze them, and throw away any pen that has been frozen.",
    "An insulin pen you are using can stay at room temperature for up to 28 days. Write the date you opened it on the label.",
    "Use a new pen needle for every injection and put used needles in a sharps container.")
doc("Inpatient Hypoglycemia Treatment Protocol", "Nursing", "care_protocol", "2026-01-15",
    "Treat any blood glucose below 70 mg/dL. If the patient can swallow, give 15 grams of fast-acting carbohydrate.",
    "If the patient cannot swallow safely, give dextrose 50 percent 25 mL IV, or glucagon 1 mg IM if there is no IV access.",
    "Recheck glucose every 15 minutes until it is above 100 mg/dL. Notify the provider for any value below 40 mg/dL.")
doc("Fall Prevention Policy (2025 version, superseded)", "Nursing", "policy", "2025-01-01",
    "Score every patient with the Morse Fall Scale on admission and every shift. A score of 45 or higher means high fall risk.",
    "High fall risk patients get a bed alarm, a yellow wristband and hourly rounding.")
doc("Fall Prevention Policy (2026 version)", "Nursing", "policy", "2026-01-01",
    "Score every patient with the Morse Fall Scale on admission, every shift and after any fall. A score of 50 or higher means high fall risk.",
    "High fall risk patients get a bed alarm, yellow non-slip socks, a yellow wristband and rounding every hour while awake and every 2 hours overnight.",
    "After any fall, hold a post-fall huddle on the unit within 1 hour and file a safety event report before the end of the shift.")
doc("Pressure Injury Prevention Protocol", "Nursing", "care_protocol", "2026-02-01",
    "A Braden Scale score of 18 or lower means the patient is at risk for pressure injury.",
    "Reposition at-risk patients at least every 2 hours. Float the heels off the bed with a pillow under the calves.",
    "Inspect the skin over the sacrum, heels and elbows at least once each shift and document any redness that does not fade with pressure.")
doc("Discharge Medication Reconciliation Policy", "Pharmacy", "policy", "2026-01-10",
    "A pharmacist reviews the discharge medication list for every patient taking 10 or more medicines, and for every patient on an anticoagulant or insulin.",
    "The meds-to-beds program delivers first fills to the bedside before discharge so patients leave with medicines in hand.",
    "Any change from the home medication list must have a reason documented in the discharge summary.")
doc("Post-Discharge Follow-up Call Protocol", "Case Management", "care_protocol", "2026-03-01",
    "A nurse calls every patient within 48 hours of discharge. Make up to 3 call attempts on different days before closing the outreach.",
    "During the call, review warning signs, confirm the follow-up appointment and ask the patient to read back each medicine name.",
    "Patients with a LACE readmission score of 10 or higher must have a primary care visit scheduled within 7 days of discharge.")
doc("Transportation Assistance Policy", "Case Management", "policy", "2025-11-01",
    "Patients without a ride to a follow-up appointment can get a free ride voucher from case management.",
    "Request the ride at least 24 hours before the appointment. Patients with Medicaid are first referred to the non-emergency medical transportation benefit.")
doc("Stroke and TIA Discharge Instructions", "Neurology", "discharge_instructions", "2026-01-25",
    "Learn BE FAST: Balance loss, Eyes (vision change), Face drooping, Arm weakness, Speech trouble, Time to call 911.",
    "Your blood pressure goal is below 130/80. Check it at home every day and bring the log to your visits.",
    "Take your statin every day, even if your cholesterol numbers look good.",
    "See the neurology clinic within 2 weeks. Do not drive until your neurologist clears you.")
doc("Seizure Safety at Home", "Neurology", "patient_education", "2025-12-15",
    "Take showers instead of baths, and do not swim alone.",
    "If a seizure lasts more than 5 minutes, or a second seizure starts before the person wakes up, call 911.",
    "During a seizure, turn the person on their side, put something soft under their head, and never put anything in their mouth.")
doc("Kidney-Friendly Eating (CKD Diet Education)", "Nephrology", "patient_education", "2025-10-20",
    "Avoid pain relievers such as ibuprofen and naproxen. These NSAIDs can harm your kidneys. Acetaminophen is usually safer; ask your doctor.",
    "If your potassium is high, limit bananas, oranges, potatoes and tomatoes. Your dietitian will give you a daily target.",
    "Take your phosphate binder with meals, not on an empty stomach.")
doc("Contrast Kidney Injury Prevention Protocol", "Radiology", "care_protocol", "2026-02-05",
    "Check eGFR before giving IV contrast to inpatients. For eGFR below 30, give IV normal saline before and after the scan unless the patient has heart failure.",
    "Hold metformin at the time of contrast and for 48 hours after the scan in patients with eGFR below 30, then recheck kidney function before restarting.")
doc("Opioid Safety After Surgery", "Pharmacy", "patient_education", "2026-01-30",
    "Do not drink alcohol or take sleeping pills or benzodiazepines while you take opioid pain medicine. The mix can slow or stop your breathing.",
    "Opioids cause constipation. Take the stool softener you were prescribed and drink plenty of water.",
    "Get rid of leftover opioid pills at a drug take-back site or pharmacy drop box. Ask about a naloxone kit to keep at home.")
doc("Surgical Wound Care Discharge Instructions", "General Surgery", "discharge_instructions", "2025-12-05",
    "You may shower 48 hours after surgery. Do not soak in a bath, hot tub or pool for 2 weeks.",
    "The thin tape strips over your incision will fall off on their own in 7 to 10 days. Do not pull them off.",
    "Call your surgeon for redness that spreads, yellow or green drainage, or a fever over 101 F.")
doc("Postpartum Discharge Instructions", "Obstetrics", "discharge_instructions", "2026-03-20",
    "Call your provider or go to the emergency department if you soak through a pad in an hour or pass clots bigger than an egg.",
    "If you had high blood pressure during pregnancy, get a blood pressure check within 72 hours of going home.",
    "Severe headache, vision changes or pain under the ribs after delivery can be signs of preeclampsia. Get care right away.")
doc("Visitor Policy", "Administration", "policy", "2026-02-15",
    "Up to 2 visitors may be at the bedside at one time. Quiet hours are 9 p.m. to 6 a.m.",
    "Children under 12 must be with an adult who is not the patient.")
doc("Interpreter Services Policy", "Administration", "policy", "2025-09-01",
    "Use a qualified medical interpreter for all clinical conversations with patients who prefer a language other than English. Do not use family members to interpret clinical information.",
    "Phone and video interpreters are available 24 hours a day, 7 days a week through the interpreter line.")

# ---------------- chunking ----------------
MAX_CHARS = 450          # about 110 tokens; well under the 512-token window of snowflake-arctic-embed-m-v1.5
def chunk_doc(title, paras):
    out = []
    for p in paras:
        if len(p) <= MAX_CHARS:
            out.append(p)
        else:   # split long paragraphs on sentence ends
            cur = ""
            for s in re.split(r"(?<=\.) ", p):
                if len(cur) + len(s) + 1 > MAX_CHARS and cur:
                    out.append(cur.strip()); cur = ""
                cur += s + " "
            out.append(cur.strip())
    # prepend the title so a chunk still says what document it came from
    return [f"{title}. {c}" for c in out]

docs, chunks = [], []
for i, (title, dept, dtype, date, paras) in enumerate(D, 1):
    did = f"DOC{i:03d}"
    docs.append(dict(DOC_ID=did, TITLE=title, DEPARTMENT=dept, DOC_TYPE=dtype, EFFECTIVE_DATE=date,
                     DOC_TEXT="\n\n".join(paras)))
    for j, c in enumerate(chunk_doc(title, paras), 1):
        chunks.append(dict(CHUNK_ID=f"{did}-C{j}", DOC_ID=did, CHUNK_NO=j, TITLE=title, DEPARTMENT=dept,
                           DOC_TYPE=dtype, EFFECTIVE_DATE=date, CHUNK_TEXT=c))

# ---------------- labeled questions ----------------
# (question, gold chunk, type, filter or None). Types: "keyword" shares words with the chunk,
# "paraphrase" says the same thing in different words (where embeddings should help).
# filter uses Cortex Search filter syntax on ATTRIBUTES. No apostrophes in questions (keeps SQL simple).
Q = [
 ("How much weight gain in one day should make a heart failure patient call the clinic?", "DOC001-C1", "keyword", None),
 ("What is the daily sodium limit for heart failure patients?", "DOC001-C2", "keyword", None),
 ("My dad keeps getting up at night to pee after starting his water pill. When should he take it?", "DOC001-C3", "paraphrase", None),
 ("When should potassium be checked after starting spironolactone?", "DOC002-C2", "keyword", None),
 ("Can I quit my blood thinners after the stent if I feel fine?", "DOC003-C1", "paraphrase", None),
 ("How heavy can things be that I pick up with the arm they went in through for the heart procedure?", "DOC003-C2", "paraphrase", None),
 ("Chest pain and one nitroglycerin did not help after 5 minutes, what now?", "DOC003-C3", "keyword", None),
 ("What INR range is the usual target for atrial fibrillation on warfarin?", "DOC004-C1", "keyword", None),
 ("Do I have to stop eating spinach and salad greens on my anticoagulant?", "DOC004-C2", "paraphrase", None),
 ("How many days before an operation is warfarin stopped for bridging?", "DOC005-C1", "keyword", None),
 ("When is the final dose of the injectable bridge medicine given before the operation?", "DOC005-C2", "paraphrase", None),
 ("How long do I take the steroid pills after my lung flare-up?", "DOC006-C1", "paraphrase", None),
 ("What does the yellow zone mean on the COPD action plan?", "DOC006-C3", "keyword", None),
 ("How far should oxygen be kept from open flames and gas stoves?", "DOC007-C1", "keyword", None),
 ("How should I store my oxygen cylinders so they do not tip over?", "DOC007-C2", "paraphrase", None),
 ("How often can I use my rescue inhaler before my asthma counts as not well controlled?", "DOC008-C1", "keyword", None),
 ("When can pneumonia patients switch from IV to oral antibiotics?", "DOC009-C1", "keyword", None),
 ("Which pneumonia patients need a repeat chest x-ray at 6 to 8 weeks?", "DOC009-C3", "keyword", None),
 ("How fast must lactate be measured when sepsis is suspected?", "DOC010-C1", "keyword", None),
 ("How much IV fluid per kilogram for septic patients with low blood pressure?", "DOC010-C3", "keyword", None),
 ("Why is hand sanitizer not enough after caring for a patient with C. diff?", "DOC011-C2", "paraphrase", None),
 ("When can contact plus isolation for C. diff be stopped?", "DOC011-C3", "keyword", None),
 ("Are fake fingernails allowed for bedside nurses?", "DOC012-C2", "paraphrase", None),
 ("After a new hip, which leg positions do I need to avoid and for how long?", "DOC013-C1", "paraphrase", None),
 ("How long do patients take aspirin to prevent clots after hip replacement?", "DOC013-C2", "keyword", None),
 ("What is the lifting limit after surgery?", "DOC013-C3", "keyword", {"@eq": {"DEPARTMENT": "Orthopedics"}}),
 ("How far should my knee bend in the first couple of weeks after the replacement?", "DOC014-C1", "paraphrase", None),
 ("How long should the injured limb be raised above the heart after a cast is put on?", "DOC015-C2", "keyword", None),
 ("My blood sugar dropped to 62, how many grams of sugar should I eat and when do I check again?", "DOC016-C1", "paraphrase", None),
 ("I have the flu and cannot eat. Should I still take my long-acting insulin?", "DOC016-C3", "paraphrase", None),
 ("How long can an opened insulin pen stay out of the fridge?", "DOC017-C2", "paraphrase", None),
 ("What should the nurse give for low glucose if the patient cannot swallow and has no IV?", "DOC018-C2", "keyword", None),
 ("What Morse Fall Scale score means high fall risk?", "DOC020-C1", "keyword", {"@gte": {"EFFECTIVE_DATE": "2026-01-01"}}),
 ("How soon must the team meet on the unit after a patient falls?", "DOC020-C3", "paraphrase", None),
 ("How often should we turn bedbound patients to prevent bed sores?", "DOC021-C2", "paraphrase", None),
 ("Which patients get a pharmacist review of their discharge medication list?", "DOC022-C1", "keyword", None),
 ("Within how many hours after going home does a nurse phone the patient?", "DOC023-C1", "paraphrase", None),
 ("Which patients need a primary care visit within 7 days based on the LACE score?", "DOC023-C3", "keyword", None),
 ("How much notice is needed to book a free ride to a clinic appointment?", "DOC024-C2", "paraphrase", None),
 ("What blood pressure goal is given after a stroke?", "DOC025-C2", "keyword", None),
 ("When should you call an ambulance for a seizure?", "DOC026-C2", "paraphrase", None),
 ("Which over-the-counter pain pills hurt the kidneys?", "DOC027-C1", "paraphrase", {"@eq": {"DOC_TYPE": "patient_education"}}),
 ("How long is metformin held after IV contrast for low eGFR?", "DOC028-C2", "keyword", None),
 ("What do I do with leftover pain pills after my surgery?", "DOC029-C3", "paraphrase", None),
 ("When can I take a bath again after my operation?", "DOC030-C1", "paraphrase", {"@eq": {"DEPARTMENT": "General Surgery"}}),
 ("How much bleeding after giving birth is too much?", "DOC031-C1", "paraphrase", None),
 ("Can a relative translate for the doctor if they speak both languages?", "DOC033-C1", "paraphrase", None),
]
# questions the library cannot answer: for the RAG refusal test only (no gold chunk, not in hit rate)
UNANSWERABLE = ["What is the hospital cafeteria menu on Fridays?",
                "What dose of amoxicillin should a 3 year old child take for an ear infection?",
                "How much does a total knee replacement cost without insurance?"]

ids = {c["CHUNK_ID"] for c in chunks}
qs = []
for k, (q, gold, typ, flt) in enumerate(Q, 1):
    assert gold in ids, gold
    assert "'" not in q
    g = next(c for c in chunks if c["CHUNK_ID"] == gold)
    qs.append(dict(QID=k, QUESTION=q, GOLD_CHUNK_ID=gold, GOLD_DOC_ID=g["DOC_ID"], Q_TYPE=typ,
                   FILTER_JSON=json.dumps(flt) if flt else ""))
for u in UNANSWERABLE:
    qs.append(dict(QID=len(qs) + 1, QUESTION=u, GOLD_CHUNK_ID="", GOLD_DOC_ID="", Q_TYPE="unanswerable", FILTER_JSON=""))

def wcsv(path, data):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
wcsv("clinical_docs.csv", docs); wcsv("doc_chunks.csv", chunks); wcsv("eval_questions.csv", qs)

def lit(x):
    if x in ("", None): return "NULL"
    if isinstance(x, int): return str(x)
    return "'" + str(x).replace("'", "''") + "'"
sql = ["-- Day 38: load the SIMULATED clinical document library (generated by gen_docs.py, seed 38).",
       "-- Invented practice text: not medical advice, not a real hospital policy, no patient data. Run after 01_setup.sql.",
       "USE WAREHOUSE DS90_XS;", "USE SCHEMA DS90.DAY38;", "",
       "-- Whole documents (kept for the chunking comparison in 03_chunk.sql)",
       "CREATE OR REPLACE TABLE CLINICAL_DOCS (DOC_ID STRING, TITLE STRING, DEPARTMENT STRING, DOC_TYPE STRING,",
       "  EFFECTIVE_DATE DATE, DOC_TEXT STRING);", "INSERT INTO CLINICAL_DOCS VALUES"]
sql += [",\n".join("  (" + ", ".join(lit(r[k]) for k in docs[0]) + ")" for r in docs).replace("\n\n", "\\n\\n") + ";", ""]
sql += ["-- One row per chunk. CHUNK_TEXT is the search column. CHUNK_ID is TEXT so it can be the PRIMARY KEY.",
        "CREATE OR REPLACE TABLE DOC_CHUNKS (CHUNK_ID STRING, DOC_ID STRING, CHUNK_NO INT, TITLE STRING,",
        "  DEPARTMENT STRING, DOC_TYPE STRING, EFFECTIVE_DATE DATE, CHUNK_TEXT STRING);", "INSERT INTO DOC_CHUNKS VALUES"]
sql += [",\n".join("  (" + ", ".join(lit(r[k]) for k in chunks[0]) + ")" for r in chunks) + ";", ""]
sql += ["-- Labeled questions: the gold chunk that answers each one (simulated ground truth).",
        "CREATE OR REPLACE TABLE EVAL_QUESTIONS (QID INT, QUESTION STRING, GOLD_CHUNK_ID STRING, GOLD_DOC_ID STRING,",
        "  Q_TYPE STRING, FILTER_JSON STRING);", "INSERT INTO EVAL_QUESTIONS VALUES"]
sql += [",\n".join("  (" + ", ".join(lit(r[k]) for k in qs[0]) + ")" for r in qs) + ";", ""]
sql += ["SELECT COUNT(*) AS N_DOCS FROM CLINICAL_DOCS;",
        "SELECT COUNT(*) AS N_CHUNKS, ROUND(AVG(LENGTH(CHUNK_TEXT))) AS AVG_CHARS, MAX(LENGTH(CHUNK_TEXT)) AS MAX_CHARS FROM DOC_CHUNKS;",
        "SELECT DEPARTMENT, COUNT(*) AS N_CHUNKS FROM DOC_CHUNKS GROUP BY 1 ORDER BY 2 DESC;",
        "SELECT Q_TYPE, COUNT(*) AS N FROM EVAL_QUESTIONS GROUP BY 1 ORDER BY 1;"]
open("02_load_docs.sql", "w").write("\n".join(sql) + "\n")

from collections import Counter
print(f"wrote {len(docs)} simulated documents -> {len(chunks)} chunks "
      f"(avg {sum(len(c['CHUNK_TEXT']) for c in chunks) // len(chunks)} chars, max {max(len(c['CHUNK_TEXT']) for c in chunks)})")
print("doc types:", dict(Counter(d["DOC_TYPE"] for d in docs)))
print("departments:", len(set(d["DEPARTMENT"] for d in docs)))
print("questions:", dict(Counter(q["Q_TYPE"] for q in qs)), "| with a filter:", sum(1 for q in qs if q["FILTER_JSON"]))
