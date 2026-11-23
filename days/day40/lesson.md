# Day 40: Snowflake week mini project: a guided Cortex Code (CoCo) session, a follow-up worklist from score + notes, and a cost and credit check

## Concept
Day 40 closes the Snowflake week. Cortex Code is Snowflake's AI coding agent; the docs now call it Snowflake CoCo (views and parameters still say CORTEX_CODE). It plans and runs multi-step tasks with your role's privileges: write, explain, fix and review SQL, explore the catalog (type @ to attach a table), and answer cost and admin questions. Surfaces: a panel in Snowsight (strongest inside Workspaces, with a diff view, quick actions on highlighted SQL, and a Fix button for failed statements), the CLI (cortex), CoCo Desktop, and Excel / Google Sheets previews. Access needs SNOWFLAKE.CORTEX_USER or CORTEX_AGENT_USER, cross-region inference, and a daily credit limit that is not 0; self-service trials need a credit card on file before AI features work. It is billed per token. The CLI docs say it is not available on standard trial accounts, so today stays in Snowsight. The lesson author could not see your Snowsight: clicks are quoted from the docs (checked Nov 23, 2026); if your screen differs, trust the screen. Mini project: 300 simulated discharges (Nov 16-22, 2026). A model score (stand-in for the Day 36 registry model) shortlists 60; AI_COMPLETE with a JSON schema reads only those 60 notes for two facts the score cannot see (lives alone, follow-up booked; Day 37); each patient gets a guidance document (Day 38) and a service-line baseline from the semantic view (Day 39); the output is a 25-patient follow-up call worklist plus one brief built from aggregates only. Offline (real run): readmissions caught in 25 = 16 by score alone, 18 with regex note fields, 21 with the answer key. CoCo session: 8 prompts (explore read-only, guardrail test, explain, generate and prove with MINUS, Fix, review a wrong-but-running query, edit through the diff, a cost question). Cost and credit check: METERING_DAILY_HISTORY, WAREHOUSE_METERING_HISTORY, CORTEX_AISQL_USAGE_HISTORY (the older CORTEX_FUNCTIONS_USAGE_HISTORY is no longer updated), Analyst, Search and CORTEX_CODE_*_USAGE_HISTORY views, resource monitors (warehouse credits only), a per-user CoCo daily limit, optional ORGANIZATION_USAGE, X-Small with AUTO_SUSPEND = 60. Rates: Snowflake Service Consumption Table (not copied here). What ran on the lesson box (no ~/.snowflake, no SNOWFLAKE_* vars): the data generator, local_check.py (pandas + DuckDB) and day40_cost_report.py --dry-run. Not run: any Snowflake SQL, AI_COMPLETE, CoCo. Next: Days 41-54 (planned) move from Cortex to LLM APIs from Python, prompt engineering, embeddings and vector search; Day 41 starts with your first LLM API calls from Python, building on Cortex COMPLETE.

## Key numbers
- **Data:** 300 simulated discharges, Nov 16-22, 2026; readmission rate 0.263; 26% live alone; 63% follow-up booked
- **Score (stand-in for Day 36 model):** AUC 0.718 on all 300; with the two note facts added (answer key) 0.798
- **Regex note reader (real run):** lives alone 0.913 / follow-up 0.797 on 300; 0.833 / 0.683 on the 60 shortlisted
- **Readmissions caught in 25:** score only 16 (0.64); + regex fields 18 (0.72); + answer key 21 (0.84); base rate 0.26
- **Token estimate (chars / 4):** 60 shortlisted notes ~9,762 input tokens vs all 300 ~48,322
- **Warehouse math:** simulated 72-min session, 30 statements: AUTO_SUSPEND 60 = 1,731 s (0.481 cr), 300 = 4,307 s (1.196 cr), 600 = 4,944 s (1.373 cr)
- **CoCo session:** 8 prompts in Snowsight; every generated query proved against reference SQL; personal daily credit cap set first
- **Not run here:** All Snowflake SQL, AI_COMPLETE, CoCo, and the real cost report (no connection on the lesson box)

## Why it matters for healthcare data science jobs
Healthcare analytics and data science postings ask for strong SQL, cloud warehouse experience (often Snowflake), Python, turning notes and other text into usable fields, care management or population health use cases such as readmission worklists, and cost awareness on shared platforms. More postings now mention AI coding assistants and responsible use of Gen AI with PHI. After you run this on your trial, you can say, honestly, that you built a small end-to-end practice project on simulated discharges in Snowflake (score shortlist, LLM extraction with a JSON schema, a semantic-view baseline, a graded worklist), used Snowflake's AI coding agent with every generated query checked against reference SQL, and audited the week's credits with the usage views. It is a learning project on simulated data, not production experience.

## Docs checked (Nov 23, 2026)
- Overview of Snowflake CoCo (Cortex Code), surfaces and cost: https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code
- CoCo in Snowsight (panel, Workspaces, Restrict this chat, access, models): https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code-snowsight
- CoCo CLI (install, connections.toml, not on standard trial accounts): https://docs.snowflake.com/en/user-guide/cortex-code/cortex-code-cli
- CoCo CLI reference (--plan, --bypass warning): https://docs.snowflake.com/en/user-guide/cortex-code/cli-reference
- Daily credit usage limits for CoCo: https://docs.snowflake.com/en/user-guide/cortex-code/credit-usage-limit
- Cost controls for CoCo (budgets, per-user quotas): https://docs.snowflake.com/en/user-guide/cortex-code/cost-controls
- CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY view: https://docs.snowflake.com/en/sql-reference/account-usage/cortex_code_snowsight_usage_history
- CORTEX_AISQL_USAGE_HISTORY view: https://docs.snowflake.com/en/sql-reference/account-usage/cortex_aisql_usage_history
- CORTEX_FUNCTIONS_USAGE_HISTORY (no longer updated): https://docs.snowflake.com/en/sql-reference/account-usage/cortex_functions_usage_history
- Organization Usage (access needs the organization account or ORGADMIN): https://docs.snowflake.com/en/sql-reference/organization-usage
- Cross-region inference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cross-region-inference
- Service Consumption Table: https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
- Uncertain: exact Snowsight menu names can change; the lesson quotes only what the docs describe. If the CoCo icon is missing, check the role, cross-region setting, daily limit, and the trial credit card note.

## Hands-on (60-90 min)
1. Offline first: python gen_discharges.py, python local_check.py. Expect precision 0.64 (score only), 0.72 (regex fields), 0.84 (answer key).
2. Open Snowsight at app.snowflake.com/us-east-1/rhc24484/. The login is in your vault, not in these files. Open a workspace (docs: Projects, then Workspaces) and add the Day 40 SQL files.
3. Run snippet 1 (01_setup.sql): DS90_XS X-Small, AUTO_SUSPEND = 60, schema DS90.DAY40, inventory of Days 36-39, CoCo access checks, a personal CoCo daily credit cap, COCO_LOG.
4. Run 02_load.sql (exercise folder). Expect 300 / 300 / 300 / 9 / 9 rows, mean score 0.201, readmission rate 0.263.
5. Open the CoCo panel (docs: the CoCo icon in the lower-right corner). Do prompts P1 and P2 from snippet 2 in a read-only chat (docs: +, then Restrict this chat).
6. Snippet 3 (03_extract.sql): count tokens (free), make SHORTLIST (60), try 3 notes, then NOTE_FIELDS once. Expect 0 unparsed rows. Then CoCo P3 (Explain).
7. Run 04_context.sql: Day 39 semantic view rates by service line (or BASELINE_RATES) and Day 38 guidance docs by plain join (no search service needed).
8. Snippet 4 (05_worklist.sql): FOLLOWUP_WORKLIST (25 rows) and one AI_COMPLETE weekly brief from aggregates only.
9. CoCo P4 to P7: generate the worklist query and prove it with MINUS both ways; Fix button on query 1 of cortex_code_broken.sql; ask for a review of query 2 (wrong denominator); add a comment block through the diff view.
10. Snippet 5 (06_evaluate.sql): LLM accuracy on 60 notes vs the answer key; readmissions caught by score only vs score plus LLM fields.
11. CoCo P8 (cost question), then snippet 6 (07_cost_check.sql): credits by day and service, warehouse, AI functions, Analyst, Search, CoCo, query history, resource monitor, cleanup, SUSPEND.
12. Optional: snippet 7 (python day40_cost_report.py) for CSVs and a chart of your week. Keep those files out of git.
13. Fill in 'My results' in day40/README.md, keep the simulated-data note, then commit and push (snippet 9).

## Code
Exercise files: `exercise/01_setup.sql`, `exercise/02_load.sql`, `exercise/03_extract.sql`, `exercise/04_context.sql`, `exercise/05_worklist.sql`, `exercise/06_evaluate.sql`, `exercise/07_cost_check.sql`, `exercise/cortex_code_session.md`, `exercise/cortex_code_broken.sql`, `exercise/gen_discharges.py`, `exercise/local_check.py`, `exercise/day40_cost_report.py`, `exercise/discharges.csv`, `exercise/note_labels.csv`, `exercise/outcomes.csv`, `exercise/baseline_rates.csv`, `exercise/guidance_map.csv`, `exercise/encounters.csv`, `exercise/drg_groups.csv`, `exercise/local_check_output.txt`, `exercise/README.md`.

### 1. Setup, week inventory, CoCo access checks and a personal CoCo cap (SQL)
```sql
-- Day 40: setup, inventory of the Snowflake week, and Cortex Code (CoCo) access checks.
-- Run in a Snowsight worksheet or a Workspaces SQL file: https://app.snowflake.com/us-east-1/rhc24484/
-- Meters today: warehouse time (X-Small = 1 credit/hour, per second, 60 s minimum per resume),
-- AI Functions tokens (03, 05), and CoCo tokens (every prompt you send to Cortex Code). Rates: Service Consumption Table.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY40
  COMMENT = 'Day 40: Snowflake week mini project on SIMULATED discharges. Not real patient data.';
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_USER() AS ME;

-- ========== A. Inventory: what is left from Days 36-39? ==========
-- Tables and views per schema (metadata only, cheap).
SELECT TABLE_SCHEMA, TABLE_TYPE, COUNT(*) AS N, SUM(ROW_COUNT) AS ROWS_TOTAL
FROM DS90.INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA LIKE 'DAY3%' OR TABLE_SCHEMA = 'DAY40'
GROUP BY 1, 2 ORDER BY 1, 2;
SHOW MODELS IN DATABASE DS90;                      -- Day 36: READMIT30_SNOWML (V1, alias TRIAL_V1) if you logged it
SHOW SEMANTIC VIEWS IN DATABASE DS90;              -- Day 39: READMIT_ANALYST
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;            -- Day 38: should be empty if you dropped it (it bills while resumed)
SHOW WAREHOUSES LIKE 'DS90%';                      -- one X-Small, AUTO_SUSPEND 60, state SUSPENDED when idle

-- ========== B. Cortex Code (CoCo) access checks ==========
-- Docs (checked Nov 23, 2026): the role needs SNOWFLAKE.CORTEX_USER or SNOWFLAKE.CORTEX_AGENT_USER
-- (CORTEX_USER goes to PUBLIC by default), and until behavior change BCR-2445 reaches the account, also SNOWFLAKE.COPILOT_USER.
-- On self-service trials, AI features stay off until an account admin adds a credit card (that does not end the trial).
SHOW GRANTS TO ROLE PUBLIC;                        -- look for DATABASE_ROLE SNOWFLAKE.CORTEX_USER
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;   -- CoCo needs cross-region inference enabled
-- Daily estimated credit limit per user for CoCo in Snowsight: -1 = no limit (default), 0 = blocked.
SHOW PARAMETERS LIKE 'CORTEX_CODE%' IN ACCOUNT;
SHOW PARAMETERS LIKE 'CORTEX_CODE%' IN USER TODORPAPAZOV;

-- ========== C. A personal fence for today (needs ACCOUNTADMIN) ==========
-- A small rolling 24-hour cap on YOUR CoCo in Snowsight usage. 2 is an arbitrary practice number:
-- check the rate table, then pick your own. UNSET later to go back to the account default.
USE ROLE ACCOUNTADMIN;
ALTER USER TODORPAPAZOV SET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER = 2;
SHOW PARAMETERS LIKE 'CORTEX_CODE_SNOWSIGHT%' IN USER TODORPAPAZOV;
-- To remove it:  ALTER USER TODORPAPAZOV UNSET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER;

-- A log of what you asked CoCo today, filled in by hand during the guided session (cortex_code_session.md).
CREATE TABLE IF NOT EXISTS DS90.DAY40.COCO_LOG (
  PROMPT_ID STRING, ASKED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(), SURFACE STRING, PROMPT STRING,
  WHAT_IT_DID STRING, RAN_SQL BOOLEAN, MATCHED_REFERENCE STRING, MY_EDITS STRING);
```

### 2. The guided CoCo session: 8 prompts and what a good answer looks like (Markdown)
```markdown
# Day 40: guided Cortex Code (CoCo) session, about 25 minutes

Snowflake's AI coding agent was launched as **Cortex Code**. The docs now call it **Snowflake CoCo**; the usage views and parameters still say `CORTEX_CODE`. It comes in several forms: CoCo in Snowsight (a panel in the web UI, strongest inside Workspaces), the CoCo CLI (`cortex` in a terminal), CoCo Desktop, and previews for Excel and Google Sheets. Docs checked Nov 23, 2026.

Honest limits of this guide: the lesson author could not open your Snowsight. The clicks below are the ones the Snowflake docs describe. If your screen looks different, trust the screen and the docs, and write down what you saw.

## Before you start (5 minutes)

- [ ] `01_setup.sql` parts B and C: your role has `SNOWFLAKE.CORTEX_USER` (through PUBLIC by default), cross-region inference is not `DISABLED`, and you set a small `CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER` on yourself.
- [ ] Trial note from the docs: on self-service trial accounts, AI features stay off until an account admin adds a credit card. That does not end the trial. If Days 37 to 39 worked, this is already done.
- [ ] Open the CoCo panel: the docs say to select the CoCo icon in the lower-right corner of Snowsight; the panel opens on the right.
- [ ] Open a workspace (the docs path is Projects, then Workspaces) and add the Day 40 SQL files, so CoCo can see the file you are looking at.
- [ ] Look at the model selector (docs: lower right of the message box). Auto options pick a model for you; Auto Efficient adds a small routing fee. Write down which one you used.

Every prompt is billed by tokens (input plus output, at the chosen model's rate). Short, specific prompts with fully qualified names (`DS90.DAY40.SHORTLIST`) are cheaper and more accurate. Type `@` to attach a table or schema as context.

## The prompts

Log each one in `DS90.DAY40.COCO_LOG` (insert template at the bottom) or in the README table.

**P1. Explore, in a read-only chat.** Docs: in the CoCo panel select +, then Restrict this chat, and choose read-only. Then ask:
> List the schemas DAY36 to DAY40 in database DS90 and the tables, models, semantic views and Cortex Search services in them. Which of these can use credits while I am not using them?

Check against `01_setup.sql` part A. Good answer: tables cost only storage; a semantic view and a registered model are metadata plus small storage; a warehouse bills only while running; a resumed Cortex Search service bills serving credits even with no queries.

**P2. Guardrail test, same restricted chat.**
> Create a table DS90.DAY40.RSS_TEST (x INT).

In a read-only chat this should be refused or fail. Harmless if it works (drop it), but then the restriction was not active: note it.

**P3. Explain.** Open `03_extract.sql`, highlight part D, and use the Explain quick action (docs: highlighting SQL offers Quick Edit, Format, Add to Chat, Explain). Then ask:
> How many AI_COMPLETE calls does this statement make, and how does its token use compare with running it on all 300 rows of DS90.DAY40.DISCHARGES?

Good answer: 60 calls, about one fifth of the tokens of 300 rows (lesson-box estimate: about 9,800 vs 48,000 input tokens). If it quotes a credit price, check it against the Service Consumption Table before you believe it.

**P4. Generate, then prove it.** In a new, normal chat:
> Using @DS90.DAY40.SHORTLIST and @DS90.DAY40.NOTE_FIELDS_PARSED, write a query that scores each patient as RISK_SCORE + 0.10 if LIVES_ALONE_AI + 0.12 if FOLLOWUP_BOOKED_AI is false, and returns the top 25 with ENCOUNTER_ID, SERVICE_LINE, and the score.

Run its query, then check it against the reference (expect zero rows both ways):
```sql
SELECT ENCOUNTER_ID FROM (<CoCo query>) MINUS SELECT ENCOUNTER_ID FROM DS90.DAY40.FOLLOWUP_WORKLIST;
SELECT ENCOUNTER_ID FROM DS90.DAY40.FOLLOWUP_WORKLIST MINUS SELECT ENCOUNTER_ID FROM (<CoCo query>);
```
Watch for: ties ordered differently, NULL booleans treated as false or true, `LIMIT 25` without a tie-breaker.

**P5. Fix an error.** Open `cortex_code_broken.sql` and run query 1. It fails (wrong column name, missing GROUP BY column). Use the Fix button in the results grid (docs). Read the diff before you accept it.

**P6. Review a query that runs but is wrong.** Highlight query 2, Add to Chat, and ask:
> Is this result correct for the share of the worklist with no follow-up booked?

The bug: it divides by all 300 discharges instead of the 25 on the list. Did CoCo catch it without a hint? This is the lesson of the day: Fix catches errors; only review catches wrong logic.

**P7. Edit with a diff.**
> Add a short comment block at the top of 05_worklist.sql saying the data is simulated and the weights are teaching choices, not clinical guidance.

Review the diff view, then accept or reject. Do not let it change the SQL itself.

**P8. Cost question.** The docs list a bundled `cost-intelligence` skill (type `/cost-intelligence`, or just ask in plain language):
> Which service types used the most credits in this account since November 16, 2026? Show the SQL you used.

Compare with `07_cost_check.sql` part A. Same view (`METERING_DAILY_HISTORY`)? Same total?

## After the session

- [ ] `07_cost_check.sql` part F: your CoCo requests, tokens and credits (`CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY`, latency up to about 1 hour).
- [ ] Optional: remove your personal cap with `ALTER USER TODORPAPAZOV UNSET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER;` or keep it.

## CoCo CLI: read, do not install today

The CLI (`cortex`) runs in a local terminal, can read and write local files, run git and SQL, and asks before each action in plan mode (`--plan`). It uses `~/.snowflake/connections.toml`, the same file as the Snowflake CLI, and the docs list the Snowflake CLI as a prerequisite. Windows native (PowerShell installer) and WSL are both supported.

The CLI docs say it is **not available on standard Snowflake trial accounts** (accounts made at trial.snowflake.com). It needs a paid account or a separate CoCo CLI trial, and that separate trial turns into a paid subscription after 30 days unless you cancel it. So skip it on this trial. If you use it later on a paid account: start with `cortex -c <connection> --plan`, keep secrets out of the repo, and never use `--bypass` (it approves every tool call).

## COCO_LOG insert template

```sql
INSERT INTO DS90.DAY40.COCO_LOG (PROMPT_ID, SURFACE, PROMPT, WHAT_IT_DID, RAN_SQL, MATCHED_REFERENCE, MY_EDITS)
VALUES ('P4', 'Snowsight', 'top 25 worklist query', 'wrote a QUALIFY query, ordered by score', TRUE, 'yes, 0 rows MINUS', 'none');
```
```

### 3. Shortlist 60 by score, AI_COMPLETE reads only those notes (SQL)
```sql
-- Day 40 step 3 (the Day 37 skill): read the notes with an LLM, but only where it changes a decision.
-- Plan: the model score (structured data) shortlists the top 60; AI_COMPLETE reads only those 60 notes
-- for two facts the score cannot see: lives alone, follow-up booked. Saved ONCE into a table (CTAS).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

-- A. Free size check first: AI_COUNT_TOKENS has no token charge (only a little warehouse time).
SELECT COUNT(*) AS NOTES,
       SUM(AI_COUNT_TOKENS('ai_complete', 'llama3.1-8b', NOTE_TEXT)) AS NOTE_TOKENS_ALL_300
FROM DISCHARGES;

-- B. The shortlist: top 60 by RISK_SCORE (ties broken by id so the list is stable).
CREATE OR REPLACE TABLE SHORTLIST AS
SELECT *, ROW_NUMBER() OVER (ORDER BY RISK_SCORE DESC, ENCOUNTER_ID) AS SCORE_RANK
FROM DISCHARGES
QUALIFY SCORE_RANK <= 60;

-- C. Try 3 rows first and read them. Structured output: a JSON schema instead of free text.
SELECT ENCOUNTER_ID, NOTE_TEXT,
       AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated hospital discharge note. lives_alone: true only if the patient has no one living '
                || 'with them. followup_booked: true only if a follow-up appointment has a confirmed date. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 60},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'lives_alone': {'type': 'boolean'}, 'followup_booked': {'type': 'boolean'}},
           'required': ['lives_alone', 'followup_booked']}}
       ) AS FIELDS
FROM SHORTLIST
WHERE SCORE_RANK <= 3
ORDER BY SCORE_RANK;

-- D. All 60, once. This is the main AI Functions spend of the day. Re-reading NOTE_FIELDS later is warehouse time only.
--    TO_VARIANT + the IFF below work whether the answer comes back as an object or as JSON text.
CREATE OR REPLACE TABLE NOTE_FIELDS AS
SELECT ENCOUNTER_ID, SCORE_RANK,
       TO_VARIANT(AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated hospital discharge note. lives_alone: true only if the patient has no one living '
                || 'with them. followup_booked: true only if a follow-up appointment has a confirmed date. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 60},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'lives_alone': {'type': 'boolean'}, 'followup_booked': {'type': 'boolean'}},
           'required': ['lives_alone', 'followup_booked']}}
       )) AS FIELDS
FROM SHORTLIST;

CREATE OR REPLACE VIEW NOTE_FIELDS_PARSED AS
SELECT ENCOUNTER_ID, SCORE_RANK,
       IFF(IS_OBJECT(FIELDS), FIELDS, TRY_PARSE_JSON(FIELDS::STRING)):lives_alone::BOOLEAN     AS LIVES_ALONE_AI,
       IFF(IS_OBJECT(FIELDS), FIELDS, TRY_PARSE_JSON(FIELDS::STRING)):followup_booked::BOOLEAN AS FOLLOWUP_BOOKED_AI
FROM NOTE_FIELDS;
SELECT * FROM NOTE_FIELDS_PARSED ORDER BY SCORE_RANK LIMIT 10;
SELECT COUNT(*) AS ROWS_, COUNT_IF(LIVES_ALONE_AI IS NULL OR FOLLOWUP_BOOKED_AI IS NULL) AS UNPARSED FROM NOTE_FIELDS_PARSED;  -- expect 60, 0
```

### 4. The worklist and a brief from aggregates only (SQL)
```sql
-- Day 40 step 5: the deliverable. A follow-up call worklist for care management (SIMULATED patients).
-- PRIORITY = model score + 0.10 if lives alone + 0.12 if no follow-up is booked. Simple and explainable on purpose.
-- The weights are teaching choices, not clinical guidance.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

CREATE OR REPLACE TABLE FOLLOWUP_WORKLIST AS
SELECT s.ENCOUNTER_ID, s.PATIENT_ID, s.DISCHARGE_DATE, s.SERVICE_LINE, s.PAYER, s.DISCHARGE_DISPOSITION,
       s.SCORE_RANK, ROUND(s.RISK_SCORE, 3) AS RISK_SCORE,
       f.LIVES_ALONE_AI, f.FOLLOWUP_BOOKED_AI,
       ROUND(s.RISK_SCORE + 0.10 * IFF(f.LIVES_ALONE_AI, 1, 0) + 0.12 * IFF(f.FOLLOWUP_BOOKED_AI, 0, 1), 3) AS PRIORITY,
       b.DAY39_READMIT_RATE AS SERVICE_LINE_BASELINE,
       g.TITLE AS GUIDANCE_DOC
FROM SHORTLIST s
JOIN NOTE_FIELDS_PARSED f ON f.ENCOUNTER_ID = s.ENCOUNTER_ID
LEFT JOIN BASELINE_RATES b ON b.SERVICE_LINE = s.SERVICE_LINE
LEFT JOIN GUIDANCE_MAP g ON g.SERVICE_LINE = s.SERVICE_LINE
QUALIFY ROW_NUMBER() OVER (ORDER BY PRIORITY DESC, s.ENCOUNTER_ID) <= 25;

SELECT * FROM FOLLOWUP_WORKLIST ORDER BY PRIORITY DESC, ENCOUNTER_ID;

-- How did the note fields move people? Rank by score vs rank by priority.
SELECT COUNT_IF(SCORE_RANK > 25) AS PULLED_UP_FROM_RANK_26_TO_60,
       COUNT_IF(NOT FOLLOWUP_BOOKED_AI) AS NO_FOLLOWUP_BOOKED,
       COUNT_IF(LIVES_ALONE_AI) AS LIVES_ALONE
FROM FOLLOWUP_WORKLIST;

-- One LLM call for a short weekly brief, built from AGGREGATES only (no names, no ids, no note text).
WITH agg AS (
  SELECT LISTAGG(SERVICE_LINE || ': ' || N || ' patients, ' || NO_FU || ' with no follow-up booked', '; ')
           WITHIN GROUP (ORDER BY N DESC) AS LINES
  FROM (SELECT SERVICE_LINE, COUNT(*) AS N, COUNT_IF(NOT FOLLOWUP_BOOKED_AI) AS NO_FU
        FROM FOLLOWUP_WORKLIST GROUP BY SERVICE_LINE)
)
SELECT LINES,
       AI_COMPLETE('llama3.1-8b',
         'Write a 3-sentence weekly note for a hospital care management team. Simulated data. '
         || '25 recently discharged patients are on the follow-up call list. By service line: ' || LINES
         || '. Say where to start and why. Do not invent numbers.') AS WEEKLY_BRIEF
FROM agg;
```

### 5. Grade it: extraction accuracy and readmissions caught (SQL)
```sql
-- Day 40 step 6: grade it. Two questions: did the LLM read the notes right, and did the notes help the list?
-- NOTE_LABELS and OUTCOMES are the simulator's answer key. In real work you would not have OUTCOMES for weeks.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

-- A. Extraction accuracy on the 60 shortlisted notes (lesson-box rule baseline: see local_check_output.txt).
SELECT COUNT(*) AS NOTES,
       ROUND(AVG(IFF(f.LIVES_ALONE_AI = (l.LIVES_ALONE = 1), 1, 0)), 3)         AS LIVES_ALONE_ACC,
       ROUND(AVG(IFF(f.FOLLOWUP_BOOKED_AI = (l.FOLLOWUP_BOOKED = 1), 1, 0)), 3) AS FOLLOWUP_ACC
FROM NOTE_FIELDS_PARSED f JOIN NOTE_LABELS l USING (ENCOUNTER_ID);

-- The misses, with the note text, so you can see WHY (negation? "on his own"? "attempted, no openings"?).
SELECT f.ENCOUNTER_ID, l.LIVES_ALONE, f.LIVES_ALONE_AI, l.FOLLOWUP_BOOKED, f.FOLLOWUP_BOOKED_AI, d.NOTE_TEXT
FROM NOTE_FIELDS_PARSED f JOIN NOTE_LABELS l USING (ENCOUNTER_ID) JOIN DISCHARGES d USING (ENCOUNTER_ID)
WHERE f.LIVES_ALONE_AI <> (l.LIVES_ALONE = 1) OR f.FOLLOWUP_BOOKED_AI <> (l.FOLLOWUP_BOOKED = 1);

-- B. Did the notes help? Readmissions caught in a 25-patient list: score only vs score + note fields.
WITH by_score AS (SELECT ENCOUNTER_ID FROM SHORTLIST WHERE SCORE_RANK <= 25),
     by_priority AS (SELECT ENCOUNTER_ID FROM FOLLOWUP_WORKLIST)
SELECT 'score only (top 25)' AS LIST, SUM(o.READMIT_30D) AS READMITS_CAUGHT, ROUND(AVG(o.READMIT_30D), 3) AS PRECISION_AT_25
FROM by_score JOIN OUTCOMES o USING (ENCOUNTER_ID)
UNION ALL
SELECT 'score + note fields (top 25)', SUM(o.READMIT_30D), ROUND(AVG(o.READMIT_30D), 3)
FROM by_priority JOIN OUTCOMES o USING (ENCOUNTER_ID)
UNION ALL
SELECT 'everyone (base rate)', SUM(READMIT_30D), ROUND(AVG(READMIT_30D), 3) FROM OUTCOMES;
-- Lesson-box numbers to compare with (local_check_output.txt): see the lesson. Yours can differ a little
-- because the LLM may read a few notes differently from the answer key.
```

### 6. Cost and credit check for the Snowflake week, then clean up (SQL)
```sql
-- Day 40 step 7: the cost and credit check for the whole Snowflake week (Days 36-40), then clean up.
-- ACCOUNT_USAGE views lag (warehouse metering up to about 3 hours, CoCo views up to about 1 hour, some longer).
-- If today's rows are missing, run this again tomorrow. Rates: Snowflake Service Consumption Table
--   https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf  (not copied here; they change)
USE ROLE ACCOUNTADMIN;          -- ACCOUNT_USAGE is readable by ACCOUNTADMIN by default
USE WAREHOUSE DS90_XS;
SET WEEK_START = '2026-11-16'::TIMESTAMP_LTZ;

-- A. The whole account, per day and service type: the one table to read first.
--    WAREHOUSE_METERING = warehouses; AI_SERVICES = Cortex (functions, Analyst, Search, CoCo); CLOUD_SERVICES is
--    adjusted (you pay only the part above 10% of daily compute: see CREDITS_ADJUSTMENT_CLOUD_SERVICES).
SELECT USAGE_DATE, SERVICE_TYPE,
       ROUND(SUM(CREDITS_USED), 4)        AS CREDITS_USED,
       ROUND(SUM(CREDITS_BILLED), 4)      AS CREDITS_BILLED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE USAGE_DATE >= $WEEK_START::DATE
GROUP BY 1, 2 ORDER BY 1, 2;

-- B. Warehouse credits per hour for DS90_XS (and anything else that ran).
SELECT WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE AS DAY,
       COUNT(*) AS HOURS_WITH_USAGE, ROUND(SUM(CREDITS_USED), 4) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE START_TIME >= $WEEK_START
GROUP BY 1, 2 ORDER BY 2, 1;

-- C. AI Functions tokens and credits by function and model (Days 37, 38, 40: AI_COMPLETE, COMPLETE, ...).
--    Note: CORTEX_FUNCTIONS_USAGE_HISTORY is no longer updated; the docs point to CORTEX_AISQL_USAGE_HISTORY.
SELECT DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, FUNCTION_NAME, MODEL_NAME,
       COUNT(DISTINCT QUERY_ID) AS QUERIES, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START
GROUP BY 1, 2, 3 ORDER BY 1, CREDITS DESC;
-- The older view, for rows written before it stopped updating (may be empty or stale):
SELECT FUNCTION_NAME, MODEL_NAME, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
WHERE START_TIME >= $WEEK_START GROUP BY 1, 2 ORDER BY CREDITS DESC;

-- D. Cortex Analyst (Day 39): billed per message, not per token.
SELECT DATE_TRUNC('day', START_TIME)::DATE AS DAY, SUM(REQUEST_COUNT) AS MESSAGES, ROUND(SUM(CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY
WHERE START_TIME >= $WEEK_START GROUP BY 1 ORDER BY 1;

-- E. Cortex Search (Day 38): serving and embedding credits per day.
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY
WHERE USAGE_DATE >= $WEEK_START::DATE ORDER BY USAGE_DATE;

-- F. Cortex Code (today): one row per request, credits and tokens. Separate views per surface.
SELECT 'Snowsight' AS SURFACE, USER_NAME, COUNT(*) AS REQUESTS, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START GROUP BY 1, 2
UNION ALL
SELECT 'CLI', USER_NAME, COUNT(*), SUM(TOKENS), ROUND(SUM(TOKEN_CREDITS), 5)
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_CLI_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START GROUP BY 1, 2;
-- Which models did CoCo use, per request (TOKENS_GRANULAR is keyed by model name):
SELECT REQUEST_ID, USAGE_TIME, TOKEN_CREDITS, TOKENS_GRANULAR
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
WHERE USAGE_TIME >= DATEADD('day', -1, CURRENT_TIMESTAMP()) ORDER BY USAGE_TIME DESC LIMIT 20;

-- G. Today's queries on DS90_XS (no lag): what woke the warehouse up, and for how long.
SELECT QUERY_TYPE, LEFT(QUERY_TEXT, 70) AS QUERY_TEXT, EXECUTION_STATUS,
       ROUND(TOTAL_ELAPSED_TIME / 1000, 1) AS SEC, START_TIME
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_WAREHOUSE(WAREHOUSE_NAME => 'DS90_XS', RESULT_LIMIT => 200))
WHERE START_TIME >= DATEADD('hour', -12, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- H. Fences: resource monitors (warehouse credits only) and CoCo per-user limits (AI credits).
SHOW RESOURCE MONITORS;                                    -- DS90_MONTHLY from Day 39, if you created it
CREATE RESOURCE MONITOR IF NOT EXISTS DS90_MONTHLY WITH CREDIT_QUOTA = 10 FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY TRIGGERS ON 75 PERCENT DO NOTIFY ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE DS90_XS SET RESOURCE_MONITOR = DS90_MONTHLY;
SHOW WAREHOUSES LIKE 'DS90_XS';                            -- check auto_suspend = 60 and resource_monitor = DS90_MONTHLY
-- A resource monitor does NOT stop Cortex, Analyst, Search or CoCo spend. For CoCo, use the per-user limit from 01_setup.sql.

-- I. Optional: ORGANIZATION_USAGE (money, not just credits). Needs the organization account or an
--    ORGADMIN-enabled account; on a trial these views may be missing or empty, and that is fine.
-- SELECT USAGE_DATE, USAGE_TYPE, ROUND(SUM(USAGE_IN_CURRENCY), 2) AS USD
-- FROM SNOWFLAKE.ORGANIZATION_USAGE.USAGE_IN_CURRENCY_DAILY
-- WHERE USAGE_DATE >= $WEEK_START::DATE GROUP BY 1, 2 ORDER BY 1, 2;
-- SELECT * FROM SNOWFLAKE.ORGANIZATION_USAGE.REMAINING_BALANCE_DAILY ORDER BY DATE DESC LIMIT 7;
-- Snowsight also has cost and billing pages for admins (exact menu names not verified for this lesson; see the docs).

-- J. Cleanup. Keep DS90.DAY40 tables (storage is tiny). Stop the meters.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;                    -- expect none; drop any you no longer need
ALTER WAREHOUSE DS90_XS SUSPEND;                           -- "cannot be suspended" means it already is: fine
```

### 7. Optional: read-only cost report from your usage views (Python)
```python
"""Day 40: a small cost report for the Snowflake week, from YOUR account's usage views.
Uses the Day 36 [ds90] entry in ~/.snowflake/connections.toml (or set SNOWFLAKE_CONNECTION_NAME).
Never put a password or token in this file. Read-only: it runs SELECTs on SNOWFLAKE.ACCOUNT_USAGE, then suspends DS90_XS.

  pip install snowflake-connector-python pandas matplotlib
  python day40_cost_report.py --dry-run      # no account: print the SQL it would run
  python day40_cost_report.py                # writes cost_by_day.csv, coco_requests.csv, cost_by_day.png
"""
import argparse, os, sys

SINCE = "2026-11-16"
QUERIES = {
    "cost_by_day": f"""
        SELECT USAGE_DATE, SERVICE_TYPE, ROUND(SUM(CREDITS_USED), 4) AS CREDITS_USED, ROUND(SUM(CREDITS_BILLED), 4) AS CREDITS_BILLED
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
        WHERE USAGE_DATE >= '{SINCE}' GROUP BY 1, 2 ORDER BY 1, 2""",
    "ai_functions": f"""
        SELECT DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, FUNCTION_NAME, MODEL_NAME, SUM(TOKENS) AS TOKENS,
               ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
        WHERE USAGE_TIME >= '{SINCE}' GROUP BY 1, 2, 3 ORDER BY 1, 5 DESC""",
    "coco_requests": f"""
        SELECT 'Snowsight' AS SURFACE, DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, COUNT(*) AS REQUESTS,
               SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
        WHERE USAGE_TIME >= '{SINCE}' GROUP BY 1, 2 ORDER BY 2""",
    "warehouse": f"""
        SELECT WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE AS DAY, ROUND(SUM(CREDITS_USED), 4) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE START_TIME >= '{SINCE}' GROUP BY 1, 2 ORDER BY 2, 1""",
}

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

if args.dry_run:
    for name, q in QUERIES.items():
        print(f"-- {name}\n{q.strip()};\n")
    print("DRY RUN: nothing was sent to Snowflake.")
    sys.exit(0)

import pandas as pd
import snowflake.connector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

conn = snowflake.connector.connect(connection_name=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "ds90"))
cur = conn.cursor()
cur.execute("USE ROLE ACCOUNTADMIN")
cur.execute("USE WAREHOUSE DS90_XS")
out = {}
for name, q in QUERIES.items():
    try:
        out[name] = cur.execute(q).fetch_pandas_all()
    except Exception as e:                      # a view can be missing or empty on a trial: report and continue
        print(f"{name}: skipped ({type(e).__name__}: {str(e)[:120]})")
        out[name] = pd.DataFrame()
    out[name].to_csv(f"{name}.csv", index=False)
    print(f"{name}: {len(out[name])} rows -> {name}.csv")
try:
    cur.execute("ALTER WAREHOUSE DS90_XS SUSPEND")
except Exception:
    pass                                        # already suspended
conn.close()

c = out["cost_by_day"]
if len(c):
    p = c.pivot_table(index="USAGE_DATE", columns="SERVICE_TYPE", values="CREDITS_USED", aggfunc="sum").fillna(0)
    ax = p.plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_ylabel("credits used"); ax.set_title("My Snowflake week: credits per day by service type (ACCOUNT_USAGE)")
    plt.tight_layout(); plt.savefig("cost_by_day.png", dpi=150)
    print("total credits used since", SINCE, "=", round(float(c.CREDITS_USED.sum()), 4), "-> cost_by_day.png")
else:
    print("No METERING_DAILY_HISTORY rows yet (the view can lag). Try again later.")
```

### 8. Offline: regex baseline, worklist logic in DuckDB, cost arithmetic (Python, runs without Snowflake)
```python
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
```

### 9. Push to GitHub
```bash
cd ~/ds-genai-90-days
mkdir -p day40        # copy the exercise files (SQL, .py, .md, .csv, README.md) into day40/
cd day40 && python gen_discharges.py && python local_check.py && cd ..
git status            # check: no connections.toml, no .env, no tokens, no cost CSVs with account details
git add day40
git commit -m "Day 40: Snowflake week mini project, guided Cortex Code session, cost and credit check"
git push
```

## Local run output (lesson box)
```
discharges 300 | lives alone 0.260 | booked 0.627 | mean score 0.201 | readmit rate 0.263
Rule reader, all 300 notes: lives_alone acc 0.913, followup acc 0.797
Rule reader, 60 shortlisted: lives_alone acc 0.833, followup acc 0.683
Rule misses: 82 notes. Examples:
    Lives with a roommate who checks on her daily. Discharged home on 8 active medications. The infectious disease clinic office will call to set up a vis
    Lives with daughter, who will help with medications. Discharged to a skilled nursing facility on 5 active medications. Has an appointment with nephrol
    Lives with family. Discharged home on 8 active medications. Has an appointment with the heart failure clinic on Nov 24. SIMULATED NOTE.

Readmissions caught (simulated outcome):
  base rate (all 300)                   79 of 300  precision 0.263
  score only, top 25                    16 of  25  precision 0.640
  score + rule fields, top 25           18 of  25  precision 0.720
  score + answer-key fields, top 25     21 of  25  precision 0.840
Rule worklist: 5 patients pulled up from score ranks 26-60, 18 with no follow-up booked, 8 living alone
Readmissions inside the 60-patient shortlist: 33
AUC on all 300: score 0.718, score + answer-key note fields 0.798

Token estimate (chars/4, plus ~45 tokens of schema per call): 60 notes ~9,762 in + 900 out; all 300 ~48,322 in + 4500 out
Warehouse, simulated 72-minute session, 30 statements, X-Small (1 credit/hour):
  AUTO_SUSPEND  60: 1,731 billed seconds = 0.481 credits, 18 resumes
  AUTO_SUSPEND 300: 4,307 billed seconds = 1.196 credits, 8 resumes
  AUTO_SUSPEND 600: 4,944 billed seconds = 1.373 credits, 1 resumes

wrote baseline_card.json
-- cost_by_day
SELECT USAGE_DATE, SERVICE_TYPE, ROUND(SUM(CREDITS_USED), 4) AS CREDITS_USED, ROUND(SUM(CREDITS_BILLED), 4) AS CREDITS_BILLED
        FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
        WHERE USAGE_DATE >= '2026-11-16' GROUP BY 1, 2 ORDER BY 1, 2;

-- ai_functions
SELECT DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, FUNCTION_NAME, MODEL_NAME, SUM(TOKENS) AS TOKENS,
               ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
        WHERE USAGE_TIME >= '2026-11-16' GROUP BY 1, 2, 3 ORDER BY 1, 5 DESC;

-- coco_requests
SELECT 'Snowsight' AS SURFACE, DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, COUNT(*) AS REQUESTS,
               SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
        WHERE USAGE_TIME >= '2026-11-16' GROUP BY 1, 2 ORDER BY 2;

-- warehouse
SELECT WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE AS DAY, ROUND(SUM(CREDITS_USED), 4) AS CREDITS
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE START_TIME >= '2026-11-16' GROUP BY 1, 2 ORDER BY 2, 1;

DRY RUN: nothing was sent to Snowflake.
```

## What ran, and what did not (note)
The lesson computer has no Snowflake connection, and none was created. The bot did not run any SQL against Snowflake, made no AI_COMPLETE calls, did not use Cortex Code (CoCo), and did not read your usage views. It ran gen_discharges.py, local_check.py (pandas and DuckDB on the CSVs) and day40_cost_report.py --dry-run. The regex reader and the answer key stand in for the LLM offline; they are not Cortex output. Warehouse seconds come from a simulated session timeline, and token counts are a characters-divided-by-4 estimate. Your trial numbers (LLM accuracy, CoCo behavior, credits) are the real ones. Write them in your README.

## Coming up: Days 41-54
Planned outline for the next two weeks (the plan's 'Gen AI basics' block; details may shift):
- Day 41: first steps with LLM APIs from Python, building on Cortex COMPLETE (messages, temperature, max tokens, token counts, logging cost per call).
- Days 42-44: prompt engineering (system prompts, few-shot examples, delimiters), structured output with JSON schemas and validation, and small prompt test sets.
- Days 45-46: cost, latency and safety habits: batching and caching, PHI redaction, prompt injection basics.
- Days 47-49: embeddings and cosine similarity (Cortex embedding functions and a small local model on CPU), chunking choices.
- Days 50-53: vector search (Snowflake VECTOR type and similarity functions, a local vector index), hybrid search, reranking, retrieval metrics.
- Day 54: mini project, semantic search over simulated clinical documents with an evaluation and a cost log. RAG and agents come after that.

## Tomorrow (Day 41)
First steps with LLM APIs from Python, building on Cortex COMPLETE: send chat messages yourself, set temperature and max tokens, count tokens, and log the cost of every call.

## Printables
- Cheat sheet: `day40_v1_cheat_sheet.png`
- Infographic: `day40_v2_infographic.png`
- Code worksheet: `day40_v3_worksheet.png`
- Practice quiz: `day40_v4_quiz.png`
- PDF: `day40_lesson.pdf`

## Podcast script
(Audio: `day40_podcast.mp3`)

Welcome to day forty of the ninety day Data Analyst to Data Scientist plan. This is the last day of our Snowflake week, so today we pull it all together. We will meet Snowflake's AI coding agent, build one small project that uses every piece from this week, and then look at the bill.

Let's start with the agent. Snowflake launched it as Cortex Code, and the documentation now calls it Snowflake CoCo. It is an assistant that can plan a task and carry it out inside your account, using your own role and permissions. It can write SQL, explain a query, fix one that failed, review your code, and answer questions about your data and your costs. You can use it as a panel inside Snowsight, which works best inside Workspaces, and there is also a command line tool, a desktop app, and previews for Excel and Google Sheets. One honest note. The command line version is not available on standard trial accounts, so on your trial we stay in Snowsight. And I could not open your screen, so I only describe the clicks the documentation describes. If your screen looks different, trust the screen.

Now the mini project. Picture a care management team on a Monday morning. Three hundred patients went home last week. Who should get a follow-up call first? Our simulated data has a risk score for every patient, standing in for the model from day thirty six. But the score only saw structured columns. Two important facts live only in the discharge note. Does the patient live alone? And is a follow-up appointment actually booked?

So the plan has three moves. First, the score makes a shortlist of the top sixty. Second, an AI function reads only those sixty notes and pulls out the two facts, the same skill you used on day thirty seven. Third, we add a little to the score for living alone, and a little more for no follow-up booked, and keep the top twenty five. Each patient also gets a guidance document from day thirty eight and a baseline rate for their service line from the day thirty nine semantic view. Then one more AI call writes a short weekly note for the team, using only totals, never names or note text.

Here is what the offline check found. Picking twenty five patients by score alone caught sixteen simulated readmissions. Adding the two note facts from a simple pattern matcher caught eighteen. With perfect note facts, twenty one. So the notes really matter in this simulation, and the open question for your trial is how close the language model gets.

Now, where does the agent fit? You will give it eight small tasks. Explore your database in a read-only chat. Explain the extraction step. Write the worklist query for you. Fix a query that fails. Review a query that runs fine but gives the wrong answer. That last one is the big lesson. A fix button reacts to errors. It cannot see a wrong denominator. Only checking against a reference answer catches that. So every query the agent writes gets proved, for example with a MINUS query that should return zero rows both ways.

Last, the cost check. There are several meters. The warehouse bills by the second while it is awake, with a sixty second minimum each time it wakes up. AI functions bill by tokens. Cortex Analyst bills per message. A search service bills while it is running. And the coding agent bills by tokens too. You will read the daily metering view first, then the views for each service, and you will set a small daily limit on your own use of the agent. Remember, a resource monitor only caps warehouse credits. It does not stop AI spending. In our simulated session, an auto suspend of sixty seconds billed about half a credit, and ten minutes billed almost three times as much.

Over the next two weeks we move from Cortex to the wider world of language models: APIs, prompt engineering, embeddings, and vector search. Tomorrow, on day forty one, you take your first steps with a language model API from Python. You already know the idea from Cortex COMPLETE. Tomorrow you will send the messages yourself, control temperature and length, count tokens, and log what each call costs. See you then.
