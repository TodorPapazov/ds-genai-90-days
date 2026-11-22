# Day 39: Snowflake Cortex Analyst: text-to-SQL with a semantic model over simulated readmission tables, evaluated against gold SQL

## Concept
Day 39 turns plain-English questions into SQL. Cortex Analyst is a managed text-to-SQL service: you send a question plus a semantic model, and it returns its interpretation, one SQL statement (or suggested questions when the question is unclear) and metadata (models used, question category, and which verified query it leaned on). Your warehouse runs the SQL, with your role. The semantic model is the business layer over the tables, written in YAML: logical tables on base tables, dimensions (payer, discharge disposition, age group), time_dimensions (discharge_date vs admit_date), facts (readmit_30d, los_days), metrics (readmission_rate = AVG(encounters.readmit_30d)), named filters (exclude_ama), relationships (encounters to patients and to drg_groups), module_custom_instructions, and verified_queries (question + checked SQL written with logical names such as __encounters). Older tutorials say measures; the current semantic view spec uses facts + metrics. Two ways to publish it: upload the YAML to a stage and send semantic_model_file, or create a semantic view with SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML and send semantic_view (Snowflake's recommended path; the docs also now point new builds toward Cortex Agents). Call it with POST /api/v2/cortex/analyst/message from Python, or from Streamlit in Snowflake with _snowflake.send_snow_api_request. Docs checked Nov 22, 2026. Healthcare framing: three simulated tables built on the Day 30/36 generator: ENCOUNTERS (2,325 stays, 2024-2025), PATIENTS (1,000), DRG_GROUPS (100), overall readmission rate 0.184. No real patients. Measure before you trust: 15 labeled questions (13 with hand-written gold SQL, 2 that should get NO SQL: a prediction request and a vague one), result-set comparison that ignores column names, column order and rounding, plus HASH_AGG(*) fingerprints in SQL. What ran on the lesson box (no ~/.snowflake, no SNOWFLAKE_* vars): the table generator, YAML checks (0 problems), gold SQL on DuckDB vs pandas (13/13), the 5 verified queries vs gold (5/5), the comparator on 10 hand-written candidate SQLs (6 mistakes caught, 4 correct rewrites accepted), the SQL guard, and day39_analyst.py --dry-run. Not run by the bot: any Snowflake SQL, the semantic view, the stage upload, and every Cortex Analyst message. Cost, two meters: Cortex Analyst per successful message (HTTP 200, not per token; re-asks count), and warehouse DS90_XS (X-Small, AUTO_SUSPEND = 60) for the SQL. Rates: Snowflake Service Consumption Table (not copied here). Next: Day 40, a guided Cortex Code session and a mini project that ties the Snowflake week together, with a cost and credit check.

## Key numbers
- **Data:** 2,325 simulated encounters (2024-2025), 1,000 patients, 100 DRG rows; readmission rate 0.184
- **Semantic model:** 3 logical tables, 25 dimensions/time dimensions/facts/metrics/filters, 2 relationships, 39 synonyms, 5 verified queries, 233 YAML lines
- **Eval set:** 15 questions: 13 with gold SQL (5 covered by verified queries), 1 out of scope, 1 ambiguous
- **Local checks (real run):** YAML problems 0; gold SQL vs pandas 13/13; verified queries vs gold 5/5
- **Comparator demo:** 10 hand-written candidate SQLs (not Cortex output): 6 mistakes caught, 4 correct rewrites accepted
- **Warehouse math:** 15 questions 2 min apart: AUTO_SUSPEND 60 = 930 s (0.258 credits) vs 600 = 2,282 s (0.634 credits)
- **Analyst billing:** Per successful message (HTTP 200), not per token. 15 questions = 15 messages; every follow-up or re-ask is another
- **Not run here:** All Snowflake SQL, the semantic view, the stage upload, the REST calls and the Streamlit app (no connection on the lesson box)

## Why it matters for healthcare data science jobs
Healthcare data science and analytics postings now ask for strong SQL, semantic layers or governed metric definitions, Gen AI tools on top of the warehouse, and a way to evaluate model output. Self-serve analytics for clinical and operations teams (readmissions by payer, service line, county) is a common use case. After you run this on your trial, you can say, honestly, that you wrote a semantic model over practice readmission tables, published it as a Snowflake semantic view, called Cortex Analyst from Python, and measured its SQL against hand-written gold answers with guardrails and a cost check. It is a learning project on simulated data, not production experience.

## Docs checked (Nov 22, 2026)
- Cortex Analyst overview (access, regions, cost, multi-turn): https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst
- Cortex Analyst REST API (/api/v2/cortex/analyst/message): https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst/rest-api
- YAML specification for semantic views: https://docs.snowflake.com/en/user-guide/views-semantic/semantic-view-yaml-spec
- SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML: https://docs.snowflake.com/en/sql-reference/stored-procedures/system_create_semantic_view_from_yaml
- Querying semantic views (SEMANTIC_VIEW clause): https://docs.snowflake.com/en/user-guide/views-semantic/querying
- Verified Query Repository: https://docs.snowflake.com/en/user-guide/views-semantic/verified-query-repository
- Cortex Analyst evaluations: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst-evaluations
- CORTEX_ANALYST_USAGE_HISTORY view: https://docs.snowflake.com/en/sql-reference/account-usage/cortex_analyst_usage_history
- Service Consumption Table: https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
- Note: the Cortex Analyst pages now say Snowflake recommends moving to Cortex Agents for new builds; Analyst and semantic views remain supported.

## Hands-on (60-90 min)
1. Offline first: python gen_tables.py, make_eval_set.py, make_semantic_sql.py, local_eval.py. Expect 0 YAML problems, 13/13 gold matches, 5/5 verified queries.
2. Open Snowsight at app.snowflake.com/us-east-1/rhc24484/ and a new SQL worksheet. The login is in your vault, not in these files.
3. Run snippet 1 (01_setup.sql): DS90_XS X-Small with AUTO_SUSPEND = 60, schema DS90.DAY39, stage SEMANTIC_MODELS, access checks.
4. Run 02_load_tables.sql (exercise folder). Expect 100 / 1,000 / 2,325 rows and a readmission rate of 0.185.
5. Read snippet 2 (readmit_semantic_model.yaml): find each block (tables, dimensions, time dimensions, facts, metrics, filters, relationships, verified queries).
6. Snippet 3 (03_semantic_view.sql): upload the YAML to the stage (no gzip), verify it with SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(..., TRUE), create READMIT_ANALYST, query it with SEMANTIC_VIEW(...).
7. Run 04_gold_answers.sql: 13 gold answers and the ANALYST_RUNS table.
8. Snippet 4 (day39_analyst.py): --ask one question, then --save-table for all 15 (15 messages). Note which used a verified query.
9. Optional: snippet 5 (streamlit_app.py) as a Streamlit in Snowflake app. Ask 2 or 3 questions, then close it.
10. Snippet 6 (05_evaluate.sql): HASH_AGG scorecard and mismatches side by side. Decide for each: wrong, or only rounding / an extra column?
11. Fix one miss in the YAML (synonym, description or verified query), re-run make_semantic_sql.py and 03 part B, ask that question again.
12. Snippet 7 (06_guardrails_cost.sql): read-only role, statement timeout, optional resource monitor, usage views, then suspend DS90_XS.
13. Fill in 'My results' in day39/README.md, keep the simulated-data note, then commit and push (snippet 9).

## Code
Exercise files: `exercise/01_setup.sql`, `exercise/02_load_tables.sql`, `exercise/03_semantic_view.sql`, `exercise/04_gold_answers.sql`, `exercise/05_evaluate.sql`, `exercise/06_guardrails_cost.sql`, `exercise/readmit_semantic_model.yaml`, `exercise/gen_tables.py`, `exercise/sim_readmit.py`, `exercise/make_eval_set.py`, `exercise/make_semantic_sql.py`, `exercise/analyst_eval.py`, `exercise/local_eval.py`, `exercise/day39_analyst.py`, `exercise/streamlit_app.py`, `exercise/encounters.csv`, `exercise/patients.csv`, `exercise/drg_groups.csv`, `exercise/eval_questions.csv`, `exercise/local_eval_output.txt`, `exercise/README.md`.

### 1. Setup: X-Small warehouse, schema, stage for the YAML, access checks (SQL)
```sql
-- Day 39: Cortex Analyst setup. Reuse the X-Small warehouse, make schema DS90.DAY39 and a stage for the YAML.
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Two meters today:
--   1) the warehouse runs the SQL that Cortex Analyst writes (X-Small = 1 credit/hour, per second, 60 s minimum per resume)
--   2) Cortex Analyst is billed per successful message (HTTP 200), not per token. Rates: Service Consumption Table.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY39
  COMMENT = 'Day 39: Cortex Analyst over SIMULATED readmission tables. Not real patient data.';
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- Internal stage for the semantic model YAML (directory table on, so Snowsight can list files).
CREATE STAGE IF NOT EXISTS SEMANTIC_MODELS
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Semantic model YAML files for Cortex Analyst (Day 39).';

-- Who am I, and where? Cortex Analyst is native in AWS us-east-1, so no cross-region setting is needed.
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- Access: your role needs SNOWFLAKE.CORTEX_USER (granted to PUBLIC by default) or SNOWFLAKE.CORTEX_ANALYST_USER,
-- plus SELECT on the tables and READ on the stage. Is Cortex Analyst switched on for the account?
SHOW PARAMETERS LIKE 'ENABLE_CORTEX_ANALYST' IN ACCOUNT;
SHOW GRANTS TO ROLE PUBLIC;            -- look for DATABASE_ROLE SNOWFLAKE.CORTEX_USER

-- Leftovers from earlier days that still cost money? (Day 38 search service should be gone.)
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;
SHOW WAREHOUSES LIKE 'DS90%';
```

### 2. The semantic model YAML (tables, dimensions, time dimensions, facts, metrics, filters, relationships, verified queries)
```yaml
# Day 39 semantic model for Cortex Analyst over SIMULATED 30-day readmission tables (DS90.DAY39).
# One YAML, two uses: (1) upload to stage @DS90.DAY39.SEMANTIC_MODELS and send "semantic_model_file",
# or (2) create a semantic view with SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML (recommended by Snowflake).
# Format: YAML specification for semantic views (docs checked Nov 22, 2026). No credentials in this file.
name: READMIT_ANALYST
description: >-
  Simulated inpatient encounters from 2024 and 2025 with a 30-day readmission flag, the patient's home county,
  and the DRG service line. Practice data for learning text-to-SQL. Not real patients.

tables:
  - name: encounters
    synonyms: ["stays", "admissions", "discharges", "hospitalizations", "visits"]
    description: One row per inpatient stay. The grain is one encounter.
    base_table:
      database: DS90
      schema: DAY39
      table: ENCOUNTERS
    primary_key:
      columns: [encounter_id]
    dimensions:
      - name: encounter_id
        description: Unique id of the stay.
        expr: ENCOUNTER_ID
        data_type: VARCHAR
      - name: patient_id
        description: Id of the patient. One patient can have many encounters.
        expr: PATIENT_ID
        data_type: NUMBER
      - name: payer
        synonyms: ["insurance", "insurer", "coverage", "payor"]
        description: Primary payer for the stay.
        expr: PAYER
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Medicare", "Commercial", "Medicaid", "Self-pay"]
      - name: admit_type
        synonyms: ["admission type"]
        description: How the patient was admitted.
        expr: ADMIT_TYPE
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Emergency", "Urgent", "Elective"]
      - name: discharge_disposition
        synonyms: ["discharged to", "discharge destination", "disposition"]
        description: Where the patient went at discharge. SNF means skilled nursing facility. AMA means left against medical advice.
        expr: DISCHARGE_DISPOSITION
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Home", "Home health", "SNF", "AMA"]
      - name: drg_code
        synonyms: ["DRG"]
        description: Three-character DRG code of the stay. Join key to drg_groups.
        expr: DRG_CODE
        data_type: VARCHAR
        sample_values: ["291", "871", "470"]
      - name: age_group
        synonyms: ["age band", "age bracket"]
        description: Age at admission in three bands.
        expr: CASE WHEN AGE_AT_ADMIT < 65 THEN 'Under 65' WHEN AGE_AT_ADMIT < 80 THEN '65-79' ELSE '80+' END
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Under 65", "65-79", "80+"]
    time_dimensions:
      - name: discharge_date
        synonyms: ["discharge day", "date of discharge"]
        description: Date the stay ended. Use this date for readmission rates and for 'discharged in' questions.
        expr: DISCHARGE_DATE
        data_type: DATE
      - name: admit_date
        synonyms: ["admission date"]
        description: Date the stay started. Use only when the user asks about admissions by date.
        expr: ADMIT_DATE
        data_type: DATE
    facts:
      - name: los_days
        synonyms: ["length of stay", "LOS"]
        description: Length of stay in days.
        expr: LOS_DAYS
        data_type: NUMBER
      - name: age_at_admit
        synonyms: ["age"]
        description: Patient age in years at admission.
        expr: AGE_AT_ADMIT
        data_type: NUMBER
      - name: readmit_30d
        synonyms: ["readmitted", "readmission flag", "30-day readmission"]
        description: 1 if the patient was readmitted within 30 days after this discharge, else 0.
        expr: READMIT_30D
        data_type: NUMBER
    metrics:
      - name: encounter_count
        synonyms: ["number of stays", "volume", "discharges count"]
        description: Number of encounters.
        expr: COUNT(encounters.encounter_id)
      - name: readmission_count
        synonyms: ["readmissions"]
        description: Number of encounters followed by a readmission within 30 days.
        expr: SUM(encounters.readmit_30d)
      - name: readmission_rate
        synonyms: ["readmit rate", "30-day readmission rate"]
        description: Share of encounters followed by a 30-day readmission, a number from 0 to 1.
        expr: AVG(encounters.readmit_30d)
      - name: avg_los
        synonyms: ["average length of stay", "ALOS"]
        description: Average length of stay in days.
        expr: AVG(encounters.los_days)
      - name: patient_count
        synonyms: ["unique patients", "distinct patients"]
        description: Number of distinct patients.
        expr: COUNT(DISTINCT encounters.patient_id)
    filters:
      - name: exclude_ama
        synonyms: ["without AMA", "excluding against medical advice"]
        description: Leave out stays where the patient left against medical advice.
        expr: DISCHARGE_DISPOSITION <> 'AMA'

  - name: patients
    synonyms: ["members", "people"]
    description: One row per patient. Simulated sex and home county in central Indiana.
    base_table:
      database: DS90
      schema: DAY39
      table: PATIENTS
    primary_key:
      columns: [patient_id]
    dimensions:
      - name: patient_id
        description: Id of the patient.
        expr: PATIENT_ID
        data_type: NUMBER
      - name: sex
        description: Recorded sex, F or M.
        expr: SEX
        data_type: VARCHAR
        is_enum: true
        sample_values: ["F", "M"]
      - name: home_county
        synonyms: ["county", "county of residence", "where the patient lives"]
        description: Home county of the patient (simulated, central Indiana).
        expr: HOME_COUNTY
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Marion", "Hamilton", "Hendricks", "Johnson", "Boone", "Hancock", "Madison", "Morgan", "Shelby"]
    time_dimensions:
      - name: first_admit_date
        description: Date of the patient's first stay in this data.
        expr: FIRST_ADMIT_DATE
        data_type: DATE

  - name: drg_groups
    synonyms: ["DRG lookup", "diagnosis groups"]
    description: One row per DRG code with a short description and the service line.
    base_table:
      database: DS90
      schema: DAY39
      table: DRG_GROUPS
    primary_key:
      columns: [drg_code]
    dimensions:
      - name: drg_code
        description: Three-character DRG code.
        expr: DRG_CODE
        data_type: VARCHAR
      - name: drg_description
        synonyms: ["diagnosis", "condition", "DRG name"]
        description: Short DRG description, for example Heart failure and shock with MCC.
        expr: DRG_DESCRIPTION
        data_type: VARCHAR
        sample_values: ["Heart failure and shock with MCC", "Septicemia or severe sepsis with MCC", "COPD with MCC"]
      - name: service_line
        synonyms: ["department", "specialty", "clinical service"]
        description: Clinical service line that owns the DRG.
        expr: SERVICE_LINE
        data_type: VARCHAR
        is_enum: true
        sample_values: ["Cardiology", "Pulmonary", "Gastroenterology", "Nephrology", "Infectious disease",
                        "Endocrinology", "Orthopedics", "Behavioral health", "General medicine"]

relationships:
  - name: encounters_to_patients
    left_table: encounters
    right_table: patients
    relationship_columns:
      - left_column: patient_id
        right_column: patient_id
  - name: encounters_to_drg_groups
    left_table: encounters
    right_table: drg_groups
    relationship_columns:
      - left_column: drg_code
        right_column: drg_code

module_custom_instructions:
  sql_generation: >-
    Readmission rate means AVG(readmit_30d) over encounters, rounded to 3 decimals. Use discharge_date for
    time filters on readmission questions. Count patients with COUNT(DISTINCT patient_id), never COUNT(*).
    Return aggregates, not lists of individual patients, and add LIMIT 100 to any row-level result.
  question_categorization: >-
    The data is simulated and historical (2024 and 2025). If the user asks to predict future readmissions,
    asks for clinical advice, or asks about data that is not in these tables, say it cannot be answered with this data.

verified_queries:
  # Checked by the lesson author on a local DuckDB copy of the simulated data (local_eval.py), NOT yet on Snowflake.
  # After you run each one on your trial and the answer matches, add  verified_by: <your name>  and  verified_at: <unix time>.
  - name: overall_readmission_rate
    question: What is the overall 30-day readmission rate?
    use_as_onboarding_question: true
    sql: SELECT ROUND(AVG(readmit_30d), 3) AS readmission_rate FROM __encounters
  - name: readmission_rate_by_payer
    question: What is the readmission rate by payer?
    use_as_onboarding_question: true
    sql: >-
      SELECT payer, COUNT(encounter_id) AS encounters, ROUND(AVG(readmit_30d), 3) AS readmission_rate
      FROM __encounters GROUP BY payer ORDER BY readmission_rate DESC
  - name: monthly_rate_2025
    question: Show the monthly readmission rate for discharges in 2025.
    sql: >-
      SELECT DATE_TRUNC('month', discharge_date) AS discharge_month, COUNT(encounter_id) AS encounters,
      ROUND(AVG(readmit_30d), 3) AS readmission_rate FROM __encounters
      WHERE discharge_date >= '2025-01-01' AND discharge_date < '2026-01-01'
      GROUP BY discharge_month ORDER BY discharge_month
  - name: top_drgs_by_volume
    question: What are the top 5 DRGs by number of encounters?
    use_as_onboarding_question: true
    sql: >-
      SELECT e.drg_code, d.drg_description, COUNT(e.encounter_id) AS encounters
      FROM __encounters AS e JOIN __drg_groups AS d ON e.drg_code = d.drg_code
      GROUP BY e.drg_code, d.drg_description ORDER BY encounters DESC LIMIT 5
  - name: rate_excluding_ama
    question: What is the readmission rate excluding AMA discharges?
    sql: >-
      SELECT ROUND(AVG(readmit_30d), 3) AS readmission_rate FROM __encounters
      WHERE discharge_disposition <> 'AMA'
```

### 3. Publish: stage file and semantic view, then query the view yourself (SQL)
```sql
-- Day 39: two ways to give Cortex Analyst the semantic model (generated by make_semantic_sql.py).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- ========== A. Stage file (the classic "semantic_model_file" path) ==========
-- Snowsight: Ingestion > Add Data > Load files into a stage > pick DS90.DAY39.SEMANTIC_MODELS > upload
-- readmit_semantic_model.yaml (menu names move around; the docs tutorial uses this path).   Or from a terminal with the Snowflake CLI and your own connection:
--   snow stage copy readmit_semantic_model.yaml @DS90.DAY39.SEMANTIC_MODELS --connection ds90
-- Or SnowSQL:  PUT file://readmit_semantic_model.yaml @DS90.DAY39.SEMANTIC_MODELS AUTO_COMPRESS = FALSE;
-- AUTO_COMPRESS = FALSE matters: Cortex Analyst needs the plain .yaml, not .yaml.gz.
LS @SEMANTIC_MODELS;
ALTER STAGE SEMANTIC_MODELS REFRESH;          -- update the directory table after an upload
SELECT RELATIVE_PATH, SIZE, LAST_MODIFIED FROM DIRECTORY(@SEMANTIC_MODELS);
-- API body then uses:  "semantic_model_file": "@DS90.DAY39.SEMANTIC_MODELS/readmit_semantic_model.yaml"

-- ========== B. Semantic view (what Snowflake now recommends) ==========

-- Step 1: verify only (TRUE as the 3rd argument). The full file pastes the whole YAML here.
CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('DS90.DAY39', $$
<contents of readmit_semantic_model.yaml>
$$, TRUE);

-- Step 2: create it (same call without TRUE)
SHOW SEMANTIC VIEWS IN SCHEMA DS90.DAY39;
DESCRIBE SEMANTIC VIEW DS90.DAY39.READMIT_ANALYST;
SHOW SEMANTIC METRICS IN DS90.DAY39.READMIT_ANALYST;
SHOW SEMANTIC DIMENSIONS IN DS90.DAY39.READMIT_ANALYST;

-- Round trip: read the YAML back out of the view (compare with your file)
SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW('DS90.DAY39.READMIT_ANALYST');

-- ========== C. Query the semantic view yourself (no LLM, no per-message charge) ==========
-- The metric definitions do the math, so this is a quick check that readmission_rate is right (expect 0.185 overall).
SELECT * FROM SEMANTIC_VIEW(
    DS90.DAY39.READMIT_ANALYST
    DIMENSIONS encounters.payer
    METRICS encounters.readmission_rate, encounters.encounter_count
  )
  ORDER BY readmission_rate DESC;

SELECT * FROM SEMANTIC_VIEW(
    DS90.DAY39.READMIT_ANALYST
    DIMENSIONS drg_groups.service_line
    METRICS encounters.readmission_rate
  )
  ORDER BY readmission_rate DESC;

-- ========== D. Ask questions (each question = 1 Cortex Analyst message) ==========
-- Fastest: python day39_analyst.py (fills ANALYST_RUNS for you). In the browser: streamlit_app.py as a
-- Streamlit in Snowflake app. If your Snowsight shows a chat or playground panel when you open the semantic view in the
-- catalog, you can ask there too and paste each SQL into ANALYST_RUNS.GENERATED_SQL (see 05_evaluate.sql).
```

### 4. Call the Cortex Analyst REST API from Python, guard and score the SQL (Python)
```python
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
```

### 5. Optional: Streamlit in Snowflake chat (Python)
```python
# Day 39: a tiny Streamlit in Snowflake chat over the READMIT_ANALYST semantic view (SIMULATED data).
# Snowsight: Projects > Streamlit > + Streamlit App, warehouse DS90_XS, location DS90.DAY39, paste this file, Run.
# The app runs as YOUR role inside Snowflake, so there is no password or token anywhere.
# Cost: every question = 1 Cortex Analyst message, plus warehouse time while the app is open. Close the tab when done.
import json
import _snowflake                                  # available inside Streamlit in Snowflake only
import streamlit as st
from snowflake.snowpark.context import get_active_session

VIEW = "DS90.DAY39.READMIT_ANALYST"
MAX_MESSAGES = 15                                  # guardrail: per-session message cap
BLOCKED = ("INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "CALL")
session = get_active_session()
st.title("Readmissions Q&A (simulated data)")
st.caption("Cortex Analyst + semantic view. Practice data only, not real patients.")
st.session_state.setdefault("n", 0)

question = st.chat_input("Ask about readmissions, payers, DRGs, counties ...")
if question:
    if st.session_state.n >= MAX_MESSAGES:
        st.warning("Message cap reached for this session.")
        st.stop()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}], "semantic_view": VIEW}
    resp = _snowflake.send_snow_api_request("POST", "/api/v2/cortex/analyst/message", {}, {}, body, None, 50000)
    st.session_state.n += 1
    content = json.loads(resp["content"])
    if resp["status"] >= 400:
        st.error(f"Analyst error {resp['status']}: {content.get('message')}")
        st.stop()
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        for c in content["message"]["content"]:
            if c["type"] == "text":
                st.write(c["text"])
            elif c["type"] in ("suggestion", "suggestions"):
                for s in c.get("suggestions", []):
                    st.write("- " + s)
            elif c["type"] == "sql":
                sql = c["statement"]
                st.code(sql, language="sql")
                if not sql.lstrip().upper().startswith(("SELECT", "WITH")) or any(f" {k} " in f" {sql.upper()} " for k in BLOCKED):
                    st.error("Blocked: only read-only SELECT statements run here.")
                else:
                    df = session.sql(f"SELECT * FROM (\n{sql.strip().rstrip(';')}\n) LIMIT 1000").to_pandas()
                    st.dataframe(df)
                    if 1 < len(df) <= 40 and df.shape[1] >= 2:
                        st.bar_chart(df.set_index(df.columns[0]).select_dtypes("number"))
    st.caption(f"Messages this session: {st.session_state.n} / {MAX_MESSAGES}")
```

### 6. Evaluate: HASH_AGG fingerprints, scorecard, mismatches (SQL)
```sql
-- Day 39: judge Cortex Analyst's SQL against your hand-written gold SQL, inside Snowflake.
-- Fill ANALYST_RUNS.GENERATED_SQL first: python day39_analyst.py does it, or paste SQL from the Streamlit app.
-- Running two SQL statements per question uses the warehouse only (no Cortex Analyst message is sent here).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- A. Paste by hand (example). Leave GENERATED_SQL NULL when Analyst returned suggestions or refused.
-- UPDATE ANALYST_RUNS SET GENERATED_SQL = $$<paste the SQL here, no trailing semicolon>$$ WHERE QID = 'Q03';

-- B. One pair, the long way: rows in gold that are not in generated, and the other way round. Both must be empty.
--    MINUS compares by column POSITION, so put the columns in the same order first.
SELECT PAYER, ROUND(AVG(READMIT_30D), 3) AS R FROM ENCOUNTERS GROUP BY PAYER
MINUS
SELECT PAYER, ROUND(AVG(READMIT_30D), 3) FROM ENCOUNTERS WHERE PAYER IS NOT NULL GROUP BY PAYER;

-- C. All pairs, the strict way: HASH_AGG(*) is an order-independent fingerprint of a whole result set.
--    Same hash = identical rows. Different hash can still be a fair answer (extra column, other rounding, other
--    column order), so read every mismatch, or use the tolerant compare in day39_analyst.py.
CREATE OR REPLACE TABLE ANSWER_HASHES (QID VARCHAR, SIDE VARCHAR, H NUMBER(38,0), N_ROWS NUMBER, ERR VARCHAR);

EXECUTE IMMEDIATE $$
DECLARE
  c1 CURSOR FOR SELECT QID, GOLD_SQL, GENERATED_SQL FROM ANALYST_RUNS
                WHERE GOLD_SQL IS NOT NULL AND GENERATED_SQL IS NOT NULL;
  stmt VARCHAR;
BEGIN
  FOR r IN c1 DO
    LET qid VARCHAR := r.QID;
    LET g VARCHAR := RTRIM(TRIM(r.GOLD_SQL), ';');
    LET a VARCHAR := RTRIM(TRIM(r.GENERATED_SQL), ';');
    stmt := 'INSERT INTO ANSWER_HASHES SELECT ''' || qid || ''', ''gold'', HASH_AGG(*), COUNT(*), NULL FROM (' || g || ')';
    EXECUTE IMMEDIATE :stmt;
    BEGIN
      stmt := 'INSERT INTO ANSWER_HASHES SELECT ''' || qid || ''', ''generated'', HASH_AGG(*), COUNT(*), NULL FROM (' || a || ')';
      EXECUTE IMMEDIATE :stmt;
    EXCEPTION
      WHEN OTHER THEN
        LET err VARCHAR := SQLERRM;
        INSERT INTO ANSWER_HASHES VALUES (:qid, 'generated', NULL, NULL, :err);
    END;
  END FOR;
  RETURN 'hashed';
END;
$$;

UPDATE ANALYST_RUNS R SET HASH_MATCH = (G.H = A.H)
FROM ANSWER_HASHES G JOIN ANSWER_HASHES A ON G.QID = A.QID AND G.SIDE = 'gold' AND A.SIDE = 'generated'
WHERE R.QID = G.QID;

-- D. Scorecard. Q14 and Q15 are correct when there is NO SQL (a refusal or suggestions).
SELECT QID, QTYPE, IN_VQR,
       CASE WHEN GOLD_SQL IS NULL THEN IFF(GENERATED_SQL IS NULL, 'correct refusal', 'answered when it should not')
            WHEN GENERATED_SQL IS NULL THEN 'no SQL returned'
            WHEN HASH_MATCH THEN 'exact match'
            ELSE 'check by eye' END AS VERDICT,
       VERIFIED_QUERY_USED, NOTE
FROM ANALYST_RUNS ORDER BY QID;

SELECT IN_VQR,
       COUNT(*) AS QUESTIONS,
       COUNT_IF(HASH_MATCH) AS EXACT_MATCHES,
       ROUND(COUNT_IF(HASH_MATCH) / NULLIF(COUNT(*), 0), 2) AS EXACT_MATCH_RATE
FROM ANALYST_RUNS WHERE GOLD_SQL IS NOT NULL GROUP BY IN_VQR;

-- E. Read the mismatches side by side.
SELECT R.QID, R.QUESTION, G.N_ROWS AS GOLD_ROWS, A.N_ROWS AS GEN_ROWS, A.ERR, R.GOLD_SQL, R.GENERATED_SQL
FROM ANALYST_RUNS R
JOIN ANSWER_HASHES G ON G.QID = R.QID AND G.SIDE = 'gold'
JOIN ANSWER_HASHES A ON A.QID = R.QID AND A.SIDE = 'generated'
WHERE NOT COALESCE(R.HASH_MATCH, FALSE);
```

### 7. Guardrails and cost check, then suspend (SQL)
```sql
-- Day 39: guardrails and the cost check. Run at the end of the session.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- ========== A. Guardrails ==========
-- 1) A read-only role for the app: it can read the three tables and the semantic view, and call Cortex Analyst only.
--    Generated SQL runs with the CALLER's role, so this role cannot change or drop anything. (Trial: run as ACCOUNTADMIN.)
USE ROLE ACCOUNTADMIN;
CREATE ROLE IF NOT EXISTS DS90_ANALYST_RO COMMENT = 'Day 39: read-only Cortex Analyst practice role';
GRANT USAGE ON WAREHOUSE DS90_XS TO ROLE DS90_ANALYST_RO;
GRANT USAGE ON DATABASE DS90 TO ROLE DS90_ANALYST_RO;
GRANT USAGE ON SCHEMA DS90.DAY39 TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.ENCOUNTERS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.PATIENTS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.DRG_GROUPS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON SEMANTIC VIEW DS90.DAY39.READMIT_ANALYST TO ROLE DS90_ANALYST_RO;
GRANT READ ON STAGE DS90.DAY39.SEMANTIC_MODELS TO ROLE DS90_ANALYST_RO;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_ANALYST_USER TO ROLE DS90_ANALYST_RO;   -- Analyst only, not every AI function
GRANT ROLE DS90_ANALYST_RO TO USER TODORPAPAZOV;

-- 2) A runaway query cannot run for long: stop any statement on DS90_XS after 120 seconds.
ALTER WAREHOUSE DS90_XS SET STATEMENT_TIMEOUT_IN_SECONDS = 120;

-- 3) Optional budget fence for the practice warehouse (monthly, notify at 75%, suspend at 100%).
CREATE RESOURCE MONITOR IF NOT EXISTS DS90_MONTHLY WITH CREDIT_QUOTA = 10 FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 75 PERCENT DO NOTIFY ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE DS90_XS SET RESOURCE_MONITOR = DS90_MONTHLY;
-- A warehouse monitor does not cap Cortex Analyst messages (they are AI services, not warehouse credits).
-- The app-side guard (one SELECT, allowed tables, LIMIT, message cap) lives in day39_analyst.py and streamlit_app.py.

-- Test the role: this should work ...
USE ROLE DS90_ANALYST_RO;
USE WAREHOUSE DS90_XS;
SELECT COUNT(*) FROM DS90.DAY39.ENCOUNTERS;
-- ... and this should fail with "insufficient privileges". Uncomment to see it.
-- DELETE FROM DS90.DAY39.ENCOUNTERS;
USE ROLE ACCOUNTADMIN;

-- ========== B. Cost check ==========
-- Cortex Analyst: messages and credits per hour (ACCOUNT_USAGE can lag up to a few hours).
SELECT START_TIME, END_TIME, USERNAME, REQUEST_COUNT, CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY
WHERE START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- All AI services together (Cortex Analyst shows up here with SERVICE_TYPE = 'AI_SERVICES').
SELECT SERVICE_TYPE, NAME, START_TIME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES' AND START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- The warehouse that ran the generated SQL.
SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS' AND START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- What actually ran today (available right away, no lag).
SELECT QUERY_TYPE, LEFT(QUERY_TEXT, 80) AS QUERY_TEXT, EXECUTION_STATUS,
       ROUND(TOTAL_ELAPSED_TIME / 1000, 1) AS SEC, START_TIME
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_WAREHOUSE('DS90_XS'))
ORDER BY START_TIME DESC LIMIT 30;

-- ========== C. Cleanup ==========
-- Keep the tables and the semantic view for Day 40 (a semantic view is metadata: no running cost).
-- The stage file costs only storage (a few KB). Stop the warehouse now instead of waiting 60 seconds.
ALTER WAREHOUSE DS90_XS SUSPEND;
```

### 8. Offline: comparator and guard helpers (Python, runs without Snowflake)
```python
"""Day 39 helpers shared by local_eval.py (runs offline) and day39_analyst.py (runs on your trial).
compare_results: is the generated answer the same as the gold answer?  guard_sql: is the SQL safe to run?"""
import re
import pandas as pd

BLOCKED = r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|COPY|PUT|GET|CALL|EXECUTE|USE)\b"


def guard_sql(sql, allowed_tables=("ENCOUNTERS", "PATIENTS", "DRG_GROUPS", "__ENCOUNTERS", "__PATIENTS", "__DRG_GROUPS")):
    """Return (ok, reason). One read-only statement, no DDL/DML, only the three practice tables."""
    if not sql or not sql.strip():
        return False, "no SQL returned"
    s = re.sub(r"--[^\n]*", " ", sql)               # drop line comments
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S).strip().rstrip(";")
    if ";" in s:
        return False, "more than one statement"
    if not re.match(r"^\s*(WITH|SELECT)\b", s, re.I):
        return False, "not a SELECT"
    if re.search(BLOCKED, s, re.I):
        return False, "blocked keyword: " + re.search(BLOCKED, s, re.I).group(1).upper()
    ctes = {m.upper() for m in re.findall(r"(?:WITH|,)\s*([A-Za-z_][\w]*)\s+AS\s*\(", s, re.I)}
    refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.\"]*)", s, re.I)
    for t in refs:
        base = t.replace('"', "").split(".")[-1].upper()
        if base not in allowed_tables and base not in ctes:
            return False, f"table not allowed: {t}"
    return True, "ok"


def _norm_col(col):
    """Turn one column into a comparable list: numbers rounded, dates as ISO text, strings stripped."""
    out = []
    for v in col:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append("NULL")
        elif hasattr(v, "isoformat"):
            out.append(str(v.isoformat())[:10])
        else:
            try:
                out.append(f"{round(float(v), 3):.3f}")
            except (TypeError, ValueError):
                out.append(str(v).strip())
    return out


def compare_results(gold, gen, order_matters=False):
    """Result-set equivalence that ignores column NAMES and column ORDER, and row order unless order_matters.
    Extra columns in the generated answer are allowed if every gold column is matched. Returns (match, reason)."""
    if gen is None:
        return False, "no result"
    if len(gold) != len(gen):
        return False, f"row count {len(gen)} vs gold {len(gold)}"
    g = {c: _norm_col(gold[c]) for c in gold.columns}
    h = {c: _norm_col(gen[c]) for c in gen.columns}
    if not order_matters:                              # sort rows by the gold row tuples, then match columns
        key_g = sorted(range(len(gold)), key=lambda i: tuple(g[c][i] for c in gold.columns))
        g = {c: [v[i] for i in key_g] for c, v in g.items()}
    used, mapping = set(), {}
    for c, vals in g.items():
        for d, wals in h.items():
            if d in used:
                continue
            same = sorted(vals) == sorted(wals) if not order_matters else vals == wals
            if same:
                mapping[c] = d; used.add(d); break
        else:
            if len(vals) == 1 and len(h) >= 1:
                return False, f"{c}: got {list(h.values())[-1][0]}, gold {vals[0]}"
            return False, f"no generated column matches gold column {c}"
    if not order_matters:                              # rows must line up as whole tuples, not column by column
        rows_g = sorted(zip(*[g[c] for c in gold.columns]))
        rows_h = sorted(zip(*[h[mapping[c]] for c in gold.columns]))
        if rows_g != rows_h:
            return False, "same values per column but rows do not line up"
    extra = len(gen.columns) - len(mapping)
    return True, "match" + (f" (+{extra} extra column{'s' if extra > 1 else ''})" if extra else "")
```

### 9. Push to GitHub
```bash
cd ~/ds-genai-90-days
mkdir -p day39        # copy the exercise files (SQL, .py, .yaml, .csv, README.md) into day39/
cd day39 && python gen_tables.py && python make_eval_set.py && python local_eval.py && cd ..
git status            # check: no connections.toml, no .env, no tokens
git add day39
git commit -m "Day 39: Cortex Analyst text-to-SQL with a semantic model over simulated readmission tables"
git push
```

## Local run output (lesson box)
```
$ python gen_tables.py
encounters 2325, patients 1000, drg 100, readmit rate 0.185, admit 2024-01-01 to 2025-12-31, discharge max 2026-01-05

$ python make_eval_set.py
15 questions, 13 with gold SQL, 5 covered by verified queries

$ python make_semantic_sql.py
wrote 03_semantic_view.sql (227 YAML lines embedded twice)

$ python local_eval.py
Tables: ENCOUNTERS 2,325  PATIENTS 1,000  DRG_GROUPS 100  overall readmission rate 0.185

1) YAML: 3 tables, 25 dimensions/time dims/facts/metrics/filters, 2 relationships, 5 verified queries, 39 synonyms. Problems: 0

2) Gold SQL (DuckDB) vs pandas reference answers
   Q01  MATCH  rows= 1  first row: READMISSION_RATE=0.185
   Q02  MATCH  rows= 4  first row: PAYER=Medicaid, READMISSION_RATE=0.211
   Q03  MATCH  rows= 1  first row: SERVICE_LINE=Cardiology, READMISSION_RATE=0.248
   Q04  MATCH  rows= 1  first row: ENCOUNTERS=1117
   Q05  MATCH  rows=12  first row: MONTH=2025-01-01 00:00:00, READMISSION_RATE=0.221
   Q06  MATCH  rows= 5  first row: DRG_CODE=291, DRG_DESCRIPTION=Heart failure and shock with MCC, ENCOUNTERS=505
   Q07  MATCH  rows= 4  first row: DISCHARGE_DISPOSITION=Home, AVG_LOS=3.96
   Q08  MATCH  rows= 1  first row: READMISSION_RATE=0.251
   Q09  MATCH  rows= 1  first row: PATIENTS=382
   Q10  MATCH  rows= 9  first row: HOME_COUNTY=Hendricks, READMISSION_RATE=0.176
   Q11  MATCH  rows= 1  first row: READMISSION_RATE=0.178
   Q12  MATCH  rows= 3  first row: AGE_GROUP=80+, READMISSION_RATE=0.18
   Q13  MATCH  rows= 2  first row: DISCHARGE_DISPOSITION=SNF, READMISSION_RATE=0.233
   Q14  no gold SQL: the right behavior is no SQL (out of scope)
   Q15  no gold SQL: the right behavior is no SQL (ambiguous)

3) Verified queries (logical names, __table) on DuckDB views built from the YAML
   overall_readmission_rate   vs gold Q01: match   guard: ok
   readmission_rate_by_payer  vs gold Q02: match (+1 extra column)   guard: ok
   monthly_rate_2025          vs gold Q05: match (+1 extra column)   guard: ok
   top_drgs_by_volume         vs gold Q06: match   guard: ok
   rate_excluding_ama         vs gold Q11: match   guard: ok

4) Comparator on hand-written candidate SQL (typical text-to-SQL mistakes, written by the lesson author, NOT Cortex output)
   Q02 correct, other names and order         -> MATCH: match (+1 extra column)
   Q01 correct, no ROUND                      -> MATCH: match
   Q04 correct, YEAR() instead of a range     -> MATCH: match
   Q04 uses admit date, not discharge date    -> WRONG: ENCOUNTERS: got 1109.000, gold 1117.000
   Q05 monthly rate by admit month            -> WRONG: no generated column matches gold column READMISSION_RATE
   Q09 counts encounters of those patients    -> WRONG: PATIENTS: got 1336.000, gold 382.000
   Q11 forgets the AMA filter                 -> WRONG: READMISSION_RATE: got 0.185, gold 0.178
   Q12 age band off by one (<= 65)            -> WRONG: no generated column matches gold column READMISSION_RATE
   Q08 exact text match on the DRG name       -> WRONG: READMISSION_RATE: got NULL, gold 0.251
   Q03 correct, CTE plus extra column         -> MATCH: match (+1 extra column)

5) guard_sql on risky SQL
   ALLOW  ok                                 SELECT * FROM ENCOUNTERS LIMIT 10
   BLOCK  not a SELECT                       DELETE FROM ENCOUNTERS
   BLOCK  more than one statement            SELECT 1; DROP TABLE PATIENTS
   BLOCK  table not allowed: SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
   ALLOW  ok                                 WITH x AS (SELECT * FROM PATIENTS) SELECT COUNT(*) FROM x
   ALLOW  ok                                 WITH __encounters AS (SELECT payer, readmit_30d FROM ds90.da
   ALLOW  ok                                 -- note SELECT PAYER FROM ENCOUNTERS

6) Warehouse seconds for 15 questions asked 2 minutes apart, each query about 2 s (X-Small = 1 credit/hour)
   AUTO_SUSPEND= 60s: 15 resume(s),   930 billed seconds = 0.258 credits
   AUTO_SUSPEND=300s:  1 resume(s),  1982 billed seconds = 0.551 credits
   AUTO_SUSPEND=600s:  1 resume(s),  2282 billed seconds = 0.634 credits

Cortex Analyst itself is billed per successful message (HTTP 200), not per token: 15 questions = 15 messages,
plus one per follow-up turn. Rate: Snowflake Service Consumption Table. Wrote baseline_card.json

$ python day39_analyst.py --dry-run
YAML ok: READMIT_ANALYST, 3 tables, 5 verified queries
15 questions (13 with gold SQL). Messages for a full run: 15
POST https://<account>.snowflakecomputing.com/api/v2/cortex/analyst/message
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is the readmission rate by payer?"
        }
      ]
    }
  ],
  "stream": false,
  "semantic_view": "DS90.DAY39.READMIT_ANALYST"
}
Dry run only: no connection, no request, no credits.
```

## What ran, and what did not (note)
The lesson computer has no Snowflake connection, and none was created. The bot did not run any SQL against Snowflake, did not upload the YAML, did not create the semantic view, and sent no Cortex Analyst messages. It ran gen_tables.py, make_eval_set.py, make_semantic_sql.py, local_eval.py (DuckDB and pandas on the CSVs) and day39_analyst.py --dry-run. The 10 candidate SQLs in the comparator demo were written by hand to imitate common text-to-SQL mistakes; they are not Cortex output. Cortex Analyst accuracy, the verified-query hits and credits come from your trial. Write them in your README.

## Tomorrow (Day 40)
A guided Cortex Code session and a mini project that ties the Snowflake week together (Snowpark, Cortex LLM functions, Cortex Search, Cortex Analyst), ending with a cost and credit check.

## Printables
- Cheat sheet: `day39_v1_cheat_sheet.png`
- Infographic: `day39_v2_infographic.png`
- Code worksheet: `day39_v3_worksheet.png`
- Practice quiz: `day39_v4_quiz.png`
- PDF: `day39_lesson.pdf`

## Podcast script
(Audio: `day39_podcast.mp3`)

Welcome to day thirty nine of the ninety day Data Analyst to Data Scientist plan. Yesterday you searched documents with Cortex Search. Today we go back to tables. Someone types a question in plain English, and a service writes the SQL. In Snowflake, that service is called Cortex Analyst.

Here is the idea in one breath. You send Cortex Analyst a question and a semantic model. It sends back its reading of the question, one SQL statement, and some metadata. Then your own warehouse runs that SQL, using your own role. If the question is unclear, it sends back suggested questions instead of SQL.

So what is a semantic model? Think of it as the business dictionary that sits on top of your tables. A raw table has a column called readmit thirty d. A manager says readmission rate. The semantic model connects the two. You write it in a YAML file. It lists your logical tables. It lists dimensions, the things you group and filter by, like payer or discharge disposition. It lists time dimensions, the dates. It lists facts, the row level numbers, and metrics, the formulas, like readmission rate equals the average of the readmission flag. It can hold named filters, like leaving out stays where the patient left against medical advice. It holds relationships, the join keys. And it holds verified queries, which are example questions paired with SQL that you checked by hand.

The practice data builds on this week's simulated readmission stays: three tables, encounters, patients, and a DRG lookup with service lines. Two thousand three hundred twenty five invented stays. No real patients.

You can give Cortex Analyst the model in two ways. One, upload the YAML file to a stage and point the request at it. Two, turn the same YAML into a semantic view with one stored procedure call. Snowflake now recommends the semantic view. The nice part is that you can also query a semantic view yourself with plain SQL, with no language model involved and no message charge.

Then you call the REST API from Python. You post a small JSON body with your question and the name of the semantic view. You read back the text, the SQL, and a field that tells you whether one of your verified queries was used. A short Streamlit in Snowflake app lets you try it as a chat in the browser.

Now the habit that matters most in a job. Do not trust generated SQL because it looks right. Measure it. For fifteen questions I wrote a gold SQL answer by hand. Thirteen have a real answer. Two are traps. One asks to predict who will be readmitted next week, and one is too vague. For those two, the right behavior is no SQL at all. Your script runs both the gold SQL and the generated SQL, and compares the result sets while ignoring column names and small rounding differences.

Here is what ran on the lesson computer, without Snowflake. The YAML passed the local checks. The gold answers matched a pandas version on all thirteen questions. All five verified queries returned the gold answers. And the comparator caught six typical mistakes that I wrote by hand on purpose. For example, using the admit date instead of the discharge date gives one thousand one hundred nine stays instead of one thousand one hundred seventeen. Those were my own test cases, not Cortex output. Your real accuracy number comes from your trial.

Guardrails. Run only one read only select statement on the three practice tables. Cap the rows, set a statement timeout, and use a read only role with the Cortex Analyst user database role.

Cost. Two meters. Cortex Analyst is billed per successful message, not per token, so every re ask counts. The warehouse runs the SQL. Keep it extra small with auto suspend at sixty seconds. In the lesson math, fifteen questions two minutes apart cost about a quarter of a credit of warehouse time with a sixty second suspend, versus more than half a credit with a ten minute one.

Why this matters for jobs. Healthcare data science postings ask for SQL, semantic layers, and Gen AI tools, and for a way to evaluate them. Once you run it on your trial, you can honestly say you built a semantic model and tested text to SQL against hand written answers on practice data.

Tomorrow is day forty. We finish the Snowflake week with a guided Cortex Code session and a mini project that ties the week together, and we end with a cost and credit check. See you tomorrow.
