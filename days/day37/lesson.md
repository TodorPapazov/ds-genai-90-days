# Day 37: Snowflake Cortex LLM functions in SQL: summarize, classify, extract and score simulated discharge notes

## Concept
Day 37 calls large language models from plain SQL on your Snowflake trial. Cortex functions run inside Snowflake, next to the table: no API keys and no Python required. The five classic functions in the SNOWFLAKE.CORTEX schema: COMPLETE(model, prompt) (any prompt, any supported model), SUMMARIZE(text), CLASSIFY_TEXT(text, labels), SENTIMENT(text) (a score from -1 to 1), and EXTRACT_ANSWER(text, question) (an answer plus a confidence score). Newer names (checked in the Snowflake docs on Nov 20, 2026): AI_COMPLETE (with JSON-schema structured output), AI_CLASSIFY, AI_SENTIMENT (with aspects), AI_EXTRACT (several fields per call), AI_SUMMARIZE, AI_SUMMARIZE_AGG and AI_COUNT_TOKENS. The docs pages for COMPLETE, CLASSIFY_TEXT and EXTRACT_ANSWER are kept for backward compatibility and say those legacy functions will be deprecated by the end of 2026, so you learn both and write new work with AI_*. Healthcare framing: 40 simulated discharge notes (seed 37, not real patients) with an answer key. Today you summarize each note, sort it into a low / medium / high readmission-risk band, pull out the follow-up timing and specialty, score the patient feedback, and build a call list. An LLM risk label is a reading aid, not a validated clinical risk score. Model and region: llama3.1-8b runs natively in AWS US East 1 (your trial region). Claude and OpenAI models need cross-region inference (CORTEX_ENABLED_CROSS_REGION). Check SHOW CORTEX BASE MODELS. Cost, two meters: (1) the X-Small warehouse DS90_XS while it runs, AUTO_SUSPEND = 60 (the docs recommend no larger than MEDIUM for Cortex); (2) Cortex tokens, billed as credits per million tokens per function and model. Rates are in the Snowflake Service Consumption Table; this lesson does not copy prices. Billed tokens exceed your text, because the task functions add a prompt and classify labels count on every row. Habit: 1 row, then 5, then all 40 once into a table. What ran on the lesson box (no ~/.snowflake, no SNOWFLAKE_* vars): the data generator, a no-LLM baseline (keyword risk accuracy 0.375, regex follow-up 1.000, word-list tone 0.800; 16 of 40 notes hit a negation trap like 'No prior admissions'), a token estimate (about 19,471 tokens of your text for one full pass), and day37_cortex.py --dry-run. Not run by the bot: any SQL against Snowflake, and every Cortex call. Your trial results are the real ones. Next: Day 38, Cortex Search (a first look at RAG).

## Key numbers
- **Data:** 40 simulated discharge notes (seed 37), avg 320 chars. Answer key: 13 low / 20 medium / 7 high risk, 8 with no follow-up
- **Local baseline (real run):** Keyword risk 0.375, regex follow-up 1.000, word-list tone 0.800
- **Token estimate:** About 19,471 tokens of your text for one pass (chars / 4), before built-in prompts
- **Model:** llama3.1-8b: native in AWS US East 1. Others via cross-region inference
- **Warehouse:** DS90_XS, X-Small, AUTO_SUSPEND=60, AUTO_RESUME=TRUE. Suspend when you finish
- **Not run here:** All Snowflake SQL and Cortex calls (no connection on the lesson box). Run them in Snowsight

## Docs checked (Nov 20, 2026)
- Cortex AI Functions: https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql
- Models and regional availability: https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql-regional-availability
- Model availability by region: https://docs.snowflake.com/en/user-guide/snowflake-cortex/model-availability
- Cost considerations: https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql-cost
- Cross-region inference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference
- Service Consumption Table (credits per million tokens): https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf

## Hands-on (60-90 min)
1. Offline first: python gen_notes.py and python local_baseline.py. Note the baseline (risk 0.375, follow-up 1.000, tone 0.800) and the token estimate.
2. Open Snowsight at app.snowflake.com/us-east-1/rhc24484/ and a new SQL worksheet. The login is in your vault, not in these files.
3. Run snippet 1 (01_setup.sql): X-Small DS90_XS with AUTO_SUSPEND = 60, schema DS90.DAY37, the cross-region value, SHOW CORTEX BASE MODELS, and a one-line smoke test.
4. Run 02_load_notes.sql (from the exercise folder). Confirm 40 notes and the answer key split 13 / 20 / 7.
5. Snippet 2, part A: one row through all five functions. Write down each return type.
6. Parts B and C on 5 rows: CLASSIFY_TEXT with label descriptions, and COMPLETE with temperature 0, max_tokens, and the usage token counts.
7. Part D: CREATE TABLE NOTE_AI over all 40 notes once. From here on, query the table, not the model.
8. Snippet 3 (04_ai_functions.sql): AI_COUNT_TOKENS first (no token charge), then AI_COMPLETE with a JSON schema, AI_CLASSIFY, AI_SENTIMENT with aspects, AI_EXTRACT, AI_SUMMARIZE_AGG.
9. Snippet 4 (05_evaluate.sql): accuracy vs the answer key, the risk confusion table, the hallucination check on the 8 notes with no follow-up, and the call list.
10. Optional: snippet 5, python day37_cortex.py from VS Code with your Day 36 [ds90] connection (5 notes by default).
11. Snippet 6 (06_cost_check.sql): Cortex usage views and warehouse credits (they can lag), then ALTER WAREHOUSE DS90_XS SUSPEND.
12. Fill in 'My results' in day37/README.md, keep the simulated-data note, then commit and push (snippet 8).

## Code
Exercise files: `exercise/01_setup.sql`, `exercise/02_load_notes.sql`, `exercise/03_cortex_functions.sql`, `exercise/04_ai_functions.sql`, `exercise/05_evaluate.sql`, `exercise/06_cost_check.sql`, `exercise/gen_notes.py`, `exercise/local_baseline.py`, `exercise/day37_cortex.py`, `exercise/discharge_notes.csv`, `exercise/note_labels.csv`, `exercise/README.md`.

### 1. Setup: X-Small warehouse, schema, cross-region check, smoke test (SQL)
```sql
-- Day 37: reuse the Day 36 X-Small warehouse and make a schema for Cortex practice.
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Two meters run today:
--   1) the warehouse (X-Small, billed while running, AUTO_SUSPEND = 60 seconds)
--   2) Cortex AI tokens (billed per million tokens processed, see the Service Consumption Table)

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
-- Snowflake docs: use a warehouse no larger than MEDIUM for Cortex AI functions.
-- A bigger warehouse does not make the LLM faster, it only costs more.
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY37
  COMMENT = 'Day 37: Cortex LLM functions on SIMULATED discharge notes. Not real patient data.';

USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- Can my role call Cortex? (USE AI FUNCTIONS is granted to PUBLIC by default, plus SNOWFLAKE.CORTEX_USER)
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- Cross-region inference setting. Read the "value" column.
-- llama3.1-8b runs natively in AWS US East 1, so today's lesson does not need cross-region.
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;

-- Which base models can my role use, and are any of them legacy?
SHOW CORTEX BASE MODELS LIKE 'llama3%';

-- One tiny smoke test (a few dozen tokens). If this errors, fix access before you run 40 rows.
SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', 'Reply with the single word: ready') AS SMOKE_TEST;
```

### 2. Classic SNOWFLAKE.CORTEX functions: 1 row, 5 rows, then all 40 once (SQL)
```sql
-- Day 37: SNOWFLAKE.CORTEX functions on SIMULATED discharge notes.
-- Cost habit: 1 row first, then 5 rows, then all 40 rows ONCE into a table (CTAS).
-- Re-reading NOTE_AI later costs warehouse time only, not new Cortex tokens.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- A. One row, every function, so you see each output type.
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.SUMMARIZE(NOTE_TEXT)                         AS SUMMARY,         -- string
       SNOWFLAKE.CORTEX.SENTIMENT(PATIENT_FEEDBACK)                  AS FEEDBACK_SCORE,  -- float -1 to 1
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(NOTE_TEXT,
           ['low readmission risk', 'medium readmission risk', 'high readmission risk']) AS RISK_OBJ,  -- {"label": ...}
       SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
           'In how many days should the patient follow up?')        AS FOLLOWUP_RAW,    -- answer + score
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
           'In one plain sentence for the patient, say who to see and when. Note: ' || NOTE_TEXT) AS PATIENT_REMINDER
FROM DISCHARGE_NOTES
WHERE NOTE_ID = 1;

-- B. CLASSIFY_TEXT with label descriptions and a task description (more input tokens per row, often better labels).
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
         NOTE_TEXT,
         [ {'label': 'low readmission risk',    'description': 'younger or stable, no prior admissions, lives with others, short stay'},
           {'label': 'medium readmission risk', 'description': 'one or two risk factors such as age 75 or older, one prior admission, or many medications'},
           {'label': 'high readmission risk',   'description': 'several risk factors: two or more prior admissions, lives alone, long stay, heart failure, COPD, or sepsis'} ],
         {'task_description': 'Estimate 30-day readmission risk from a hospital discharge note.'}
       ):label::STRING AS RISK_BAND_AI
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- C. COMPLETE with options: temperature 0, a max_tokens cap, and token usage you can read.
--    With options, COMPLETE returns JSON text: choices[0].messages plus usage.prompt_tokens / completion_tokens.
SELECT NOTE_ID,
       PARSE_JSON(R):choices[0]:messages::STRING     AS TOP_RISK_FACTORS,
       PARSE_JSON(R):usage:prompt_tokens::INT       AS PROMPT_TOKENS,
       PARSE_JSON(R):usage:completion_tokens::INT   AS COMPLETION_TOKENS
FROM (
  SELECT NOTE_ID,
         SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
           [ {'role': 'system', 'content': 'You are a discharge nurse. Answer in 25 words or fewer. Simulated data.'},
             {'role': 'user',   'content': 'List the readmission risk factors in this note: ' || NOTE_TEXT} ],
           {'temperature': 0, 'max_tokens': 60}) AS R
  FROM DISCHARGE_NOTES
  WHERE NOTE_ID <= 5
)
ORDER BY NOTE_ID;

-- D. All 40 rows, once, saved to a table. This is the main Cortex spend of the day.
CREATE OR REPLACE TABLE NOTE_AI AS
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.SUMMARIZE(NOTE_TEXT)                                   AS SUMMARY,
       SNOWFLAKE.CORTEX.SENTIMENT(PATIENT_FEEDBACK)                            AS FEEDBACK_SCORE,
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
         NOTE_TEXT,
         [ {'label': 'low readmission risk',    'description': 'younger or stable, no prior admissions, lives with others, short stay'},
           {'label': 'medium readmission risk', 'description': 'one or two risk factors such as age 75 or older, one prior admission, or many medications'},
           {'label': 'high readmission risk',   'description': 'several risk factors: two or more prior admissions, lives alone, long stay, heart failure, COPD, or sepsis'} ],
         {'task_description': 'Estimate 30-day readmission risk from a hospital discharge note.'}
       ):label::STRING                                                         AS RISK_LABEL_AI,
       TO_VARIANT(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
         'In how many days should the patient follow up?'))                    AS FOLLOWUP_RAW,
       TO_VARIANT(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
         'Which specialty or clinic should the patient follow up with?'))      AS FOLLOWUP_WITH_RAW
FROM DISCHARGE_NOTES;

-- EXTRACT_ANSWER gives an answer plus a confidence score. This parse works whether you get an array or a string.
SELECT NOTE_ID,
       IFF(IS_ARRAY(FOLLOWUP_RAW), FOLLOWUP_RAW[0]:answer::STRING, FOLLOWUP_RAW::STRING) AS FOLLOWUP_ANSWER,
       IFF(IS_ARRAY(FOLLOWUP_RAW), FOLLOWUP_RAW[0]:score::FLOAT, NULL)                   AS FOLLOWUP_SCORE,
       IFF(IS_ARRAY(FOLLOWUP_WITH_RAW), FOLLOWUP_WITH_RAW[0]:answer::STRING, FOLLOWUP_WITH_RAW::STRING) AS FOLLOWUP_WITH,
       RISK_LABEL_AI,
       ROUND(FEEDBACK_SCORE, 2) AS FEEDBACK_SCORE,
       SUMMARY
FROM NOTE_AI
ORDER BY NOTE_ID;
```

### 3. The newer AI_* names, structured output and token counting (SQL)
```sql
-- Day 37: the same jobs with the newer AI_* function names.
-- Snowflake docs mark COMPLETE, CLASSIFY_TEXT and EXTRACT_ANSWER (SNOWFLAKE.CORTEX) as legacy pages kept for
-- backward compatibility, to be deprecated by the end of 2026. Use AI_* for new work.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- 0. Free check first: AI_COUNT_TOKENS has no token charge (only a little warehouse time).
SELECT NOTE_ID,
       LENGTH(NOTE_TEXT)                                        AS CHARS,
       AI_COUNT_TOKENS('ai_complete', 'llama3.1-8b', NOTE_TEXT)  AS COMPLETE_INPUT_TOKENS,
       AI_COUNT_TOKENS('ai_sentiment', PATIENT_FEEDBACK)         AS SENTIMENT_INPUT_TOKENS
FROM DISCHARGE_NOTES
ORDER BY NOTE_ID
LIMIT 5;

-- 1. AI_COMPLETE with a JSON schema (structured output) instead of free text.
SELECT NOTE_ID,
       AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated discharge note and fill the fields. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 120},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'followup_days': {'type': 'integer'},
             'followup_with': {'type': 'string'},
             'lives_alone':   {'type': 'boolean'}},
           'required': ['followup_days', 'followup_with', 'lives_alone']}}
       ) AS FIELDS
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 2. AI_CLASSIFY returns {"labels": [...]}; take the first label.
SELECT NOTE_ID,
       AI_CLASSIFY(NOTE_TEXT,
         ['low readmission risk', 'medium readmission risk', 'high readmission risk']):labels[0]::STRING AS RISK_LABEL
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 3. AI_SENTIMENT returns labels (positive, negative, neutral, mixed, unknown), overall and per aspect.
SELECT NOTE_ID,
       PATIENT_FEEDBACK,
       AI_SENTIMENT(PATIENT_FEEDBACK, ['staff', 'discharge process', 'medications']) AS SENTIMENT_OBJ
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 4. AI_EXTRACT pulls several named fields in one call.
SELECT NOTE_ID,
       AI_EXTRACT(text => NOTE_TEXT,
                  responseFormat => {'followup_when': 'In how many days or weeks should the patient follow up?',
                                     'followup_with': 'Which specialty or clinic should the patient follow up with?'}
       ):response AS EXTRACTED
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 5. AI_SUMMARIZE_AGG: one summary across many rows (here: feedback by diagnosis).
SELECT DIAGNOSIS,
       COUNT(*)                            AS N,
       AI_SUMMARIZE_AGG(PATIENT_FEEDBACK)  AS FEEDBACK_THEMES
FROM DISCHARGE_NOTES
GROUP BY DIAGNOSIS
ORDER BY N DESC;
```

### 4. Grade against the answer key and build a call list (SQL)
```sql
-- Day 37: grade the LLM outputs against the answer key in NOTE_LABELS (simulated ground truth).
-- No new Cortex calls here: this only reads NOTE_AI, so it costs warehouse time only.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

CREATE OR REPLACE VIEW NOTE_EVAL AS
SELECT a.NOTE_ID,
       l.TRUE_RISK_BAND,
       SPLIT_PART(a.RISK_LABEL_AI, ' ', 1)                                          AS AI_RISK_BAND,
       l.TRUE_FOLLOWUP_DAYS,
       IFF(IS_ARRAY(a.FOLLOWUP_RAW), a.FOLLOWUP_RAW[0]:answer::STRING, a.FOLLOWUP_RAW::STRING) AS AI_FOLLOWUP_TEXT,
       -- turn "7 days" / "2 weeks" / "1 month" into days
       TRY_TO_NUMBER(REGEXP_SUBSTR(AI_FOLLOWUP_TEXT, '[0-9]+'))
         * CASE WHEN AI_FOLLOWUP_TEXT ILIKE '%week%' THEN 7
                WHEN AI_FOLLOWUP_TEXT ILIKE '%month%' THEN 30 ELSE 1 END             AS AI_FOLLOWUP_DAYS,
       l.TRUE_FEEDBACK_TONE,
       CASE WHEN a.FEEDBACK_SCORE >= 0.5 THEN 'positive'
            WHEN a.FEEDBACK_SCORE <= -0.5 THEN 'negative' ELSE 'neutral' END        AS AI_FEEDBACK_TONE,
       a.FEEDBACK_SCORE
FROM NOTE_AI a
JOIN NOTE_LABELS l USING (NOTE_ID);

-- Accuracy per task (compare with the local keyword baseline printed by local_baseline.py)
SELECT ROUND(AVG(IFF(AI_RISK_BAND = TRUE_RISK_BAND, 1, 0)), 3)                       AS RISK_ACCURACY,
       ROUND(AVG(IFF(EQUAL_NULL(AI_FOLLOWUP_DAYS, TRUE_FOLLOWUP_DAYS), 1, 0)), 3)     AS FOLLOWUP_ACCURACY,
       ROUND(AVG(IFF(AI_FEEDBACK_TONE = TRUE_FEEDBACK_TONE, 1, 0)), 3)               AS TONE_ACCURACY,
       COUNT(*)                                                                       AS N
FROM NOTE_EVAL;

-- Confusion table for risk bands: where does the LLM disagree with the answer key?
SELECT TRUE_RISK_BAND, AI_RISK_BAND, COUNT(*) AS N
FROM NOTE_EVAL
GROUP BY 1, 2
ORDER BY 1, 2;

-- Notes with NO follow-up scheduled: did EXTRACT_ANSWER invent one? (hallucination check)
SELECT NOTE_ID, AI_FOLLOWUP_TEXT
FROM NOTE_EVAL
WHERE TRUE_FOLLOWUP_DAYS IS NULL;

-- Operational list: high-risk patients with unhappy feedback get a call within 48 hours.
SELECT e.NOTE_ID, n.PATIENT_ID, e.AI_RISK_BAND, e.FEEDBACK_SCORE, e.AI_FOLLOWUP_TEXT
FROM NOTE_EVAL e JOIN DISCHARGE_NOTES n USING (NOTE_ID)
WHERE e.AI_RISK_BAND = 'high' OR e.FEEDBACK_SCORE <= -0.5
ORDER BY e.AI_RISK_BAND = 'high' DESC, e.FEEDBACK_SCORE;
```

### 5. Optional: the same calls from Snowpark in VS Code (Python)
```python
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
```

### 6. Cost check, then suspend the warehouse (SQL)
```sql
-- Day 37: what did today cost? Two meters: Cortex tokens and warehouse time.
-- ACCOUNT_USAGE views are not real time. If they are empty, run this again later in the day.
-- Credits per million tokens for each function and model: Snowflake Service Consumption Table
--   https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
USE WAREHOUSE DS90_XS;

-- Cortex tokens and credits per query, grouped by model and function
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY
ORDER BY 1 DESC
LIMIT 50;

-- Cortex usage per function and model (hourly)
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
ORDER BY 1 DESC
LIMIT 50;

-- Daily AI services credits for the account
SELECT USAGE_DATE, SERVICE_TYPE, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES'
ORDER BY USAGE_DATE DESC;

-- Warehouse side (the meter that runs while DS90_XS is awake)
SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS'
ORDER BY START_TIME DESC
LIMIT 14;

-- Done for today: stop the warehouse now instead of waiting 60 seconds.
ALTER WAREHOUSE DS90_XS SUSPEND;
```

### 7. Simulated notes generator and the no-LLM baseline (Python, runs offline)
```python
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

# ---- local_baseline.py ----
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
```

### 8. Push to GitHub
```bash
cd ~/ds-genai-90-days
mkdir -p day37        # copy the exercise files (SQL, .py, .csv, README.md) into day37/
cd day37 && python gen_notes.py && python local_baseline.py && cd ..
git status            # check: no connections.toml, no .env, no passwords
git add day37
git commit -m "Day 37: Cortex LLM functions in SQL on simulated discharge notes"
git push
```

## Local run output (lesson box)
```
== python3 gen_notes.py
wrote 40 notes; risk bands {'high': 7, 'low': 13, 'medium': 20}; tones {'negative': 13, 'positive': 15, 'neutral': 12}; no follow-up 8
example note 1: 77-year-old female admitted for COPD exacerbation, length of stay 6 days. One prior admission in the last 12 months. Lives alone. Discharged home on an albuterol inhaler and a prednisone taper; 8 active medications at discharge. Denies shortness of breath at rest. Follow up with pulmonology in 7 days. Call the clinic with any new symptoms.
example feedback 1: I felt ignored and I still do not know who to call if I get worse.

== python3 local_baseline.py
notes: 40 | avg note chars 320 | avg feedback chars 56
keyword risk accuracy   : 0.375
  confusion (rows = true low/medium/high, cols = keyword low/medium/high): [[0, 10, 3], [5, 11, 4], [0, 3, 4]]
  notes hit by a negation or 'No prior admissions' trap: 16 -> [1, 2, 6, 8, 17, 18, 20, 21, 22, 24, 27, 28]
regex follow-up accuracy: 1.000  (templated text; real notes are messier)
word-list tone accuracy : 0.800
  tone confusion (rows = true pos/neu/neg): [[15, 0, 0], [4, 4, 4], [0, 0, 13]]
estimated input tokens for one full Cortex pass (your text only, chars/4):
  SUMMARIZE (notes in + summary out ~40 tok/row)                 4798
  CLASSIFY_TEXT (notes + 3 labels w/ descriptions ~90 tok/row)     6798
  EXTRACT_ANSWER x2 (notes + question, twice)                    7316
  SENTIMENT (feedback only)                                       559
  TOTAL                                                         19471  (= 0.0195 million tokens before built-in prompts)
credits = (billed tokens / 1,000,000) x credits-per-million for that function/model
rates: Snowflake Service Consumption Table (not copied here; check the current PDF)
wrote baseline_card.json

== python day37_cortex.py --rows 40 --dry-run
Plan: 40 notes, 15,028 characters of your text, about 3,757 tokens before each function's built-in prompt. Rates: Snowflake Service Consumption Table.

== python day37_cortex.py   (no connection on the lesson box)
Plan: 5 notes, 1,801 characters of your text, about 450 tokens before each function's built-in prompt. Rates: Snowflake Service Consumption Table.
No Snowflake connection found. Add a [ds90] entry to ~/.snowflake/connections.toml or set SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER / SNOWFLAKE_PASSWORD in your shell. To size the run offline: python day37_cortex.py --dry-run
exit code 1

Box: ~/.snowflake exists: no; SNOWFLAKE_* env vars: 0
```

## What ran, and what did not (note)
The lesson computer has no Snowflake connection, and none was created. The bot did not run any SQL against Snowflake and made no Cortex calls. It ran gen_notes.py, local_baseline.py, and day37_cortex.py --dry-run (plus a no-credentials check that exits cleanly). Cortex accuracy, token counts, and credits come from your trial. Write them in your README.

## Tomorrow (Day 38)
Cortex Search: index the notes and retrieve the right ones before the model answers (a first look at RAG).

## Printables
- Cheat sheet: `day37_v1_cheat_sheet.png`
- Infographic: `day37_v2_infographic.png`
- Code worksheet: `day37_v3_worksheet.png`
- Practice quiz: `day37_v4_quiz.png`
- PDF: `day37_lesson.pdf`

## Podcast script
(Audio: `day37_podcast.mp3`)

Welcome to day thirty seven of the Data Analyst to Data Scientist plan. Yesterday we trained a model inside your Snowflake trial. Today we stay on the trial, but we change tools. We are going to call large language models from plain SQL, with a family of functions Snowflake calls Cortex. No Python is required and no API keys. You write a select statement, and the model runs next to the table.

Here is the setup. I built forty simulated hospital discharge notes. They are made up, with a fixed random seed, and they are not real patients. Each note gives the age, the diagnosis, the length of stay, prior admissions, whether the patient lives alone, and usually a follow up instruction, like follow up with cardiology in seven days. Each note also has a short patient feedback comment. And there is an answer key table, so you can grade the model instead of trusting it.

Now the five functions. Summarize turns a long note into one or two sentences. Sentiment scores the patient feedback from minus one to plus one, where below minus one half is negative and above one half is positive. Classify text sorts each note into labels you choose. Today those labels are low, medium, and high readmission risk. Extract answer takes a note and a question, like in how many days should the patient follow up, and gives back an answer with a confidence score. And complete is the general one. You pick a model, such as llama three point one eight b, which runs natively in the AWS US East region where your trial lives, and you give it any prompt.

One important note from the current Snowflake documentation. The newer names start with A I and an underscore, like AI complete, AI classify, AI sentiment, and AI extract. The older pages for complete, classify text, and extract answer say those versions will be deprecated by the end of twenty twenty six. So you learn the classic names, and then you run the same jobs with the newer ones.

Now money, because a trial is finite. Two meters run today. The first is the warehouse. We use the extra small warehouse from yesterday, with auto suspend set to sixty seconds. The docs say to use nothing bigger than medium for these functions, because a bigger warehouse does not make the model faster. The second meter is tokens. Cortex bills credits per million tokens, and the rate depends on the function and the model. I am not quoting prices. The real numbers are in the Snowflake Service Consumption Table. Also, your billed tokens are higher than your text. The task functions wrap your text in their own prompt, and in one docs example a thirteen word classify call counted one hundred eighty seven tokens. So the habit is simple. Test one row, then five rows, then run all forty rows once and save them into a table. After that, query the table, not the model.

Before any of that, there is a local baseline that costs nothing. A keyword rule for risk got only thirty seven and a half percent right. Why? A phrase like no prior admissions contains the words prior admissions, and denies shortness of breath contains shortness of breath. Sixteen of the forty notes fell into one of those traps. A simple word list for feedback tone got eighty percent, and a regular expression for follow up days got every one right, but only because the text is so regular. Real notes are messier. That is where a language model can help, and your job is to measure whether it does.

I want to be clear about what I ran. The lesson computer has no Snowflake connection, so I did not run any of the SQL or the Snowflake parts of the Python script. I did run the data generator, the baseline, and the offline sizing step. Your trial numbers are the real ones. Write them in your readme, along with how many of the eight notes without a follow up got an invented one anyway. And remember, a language model label is a reading aid, not a validated clinical risk score.

Tomorrow, on day thirty eight, we take the next step with Cortex Search. Instead of asking a model about one note at a time, we index all the notes and let the model find the right ones before it answers. That is your first look at retrieval augmented generation. See you tomorrow.
