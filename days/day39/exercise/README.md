# Day 39: Snowflake Cortex Analyst, text-to-SQL with a semantic model

Write a semantic model (YAML) over three simulated readmission tables, give it to Cortex Analyst as a stage file and as a semantic view, ask 15 plain-English questions through the REST API, run the SQL it returns behind a guard, and score every answer against hand-written gold SQL. Then check the cost and lock things down with a read-only role.

All data is SIMULATED (the Day 30/36 generator, seed 30, plus seed 39 for county and sex): 2,325 encounters in 2024-2025, 1,000 patients, 100 DRG rows. No real patients, no real hospital. Never put credentials in this folder or in git.

What ran where: the lesson author did NOT run any SQL against Snowflake and sent no Cortex Analyst messages (the lesson computer has no Snowflake connection). Only the pure-Python parts ran: `gen_tables.py`, `make_eval_set.py`, `make_semantic_sql.py`, `local_eval.py` (DuckDB on the CSVs) and `day39_analyst.py --dry-run`. Your trial results are the real ones. Write them below.

## Offline first (no account, no cost, about 10 minutes)

- [ ] `pip install pyyaml duckdb pandas`
- [ ] `python gen_tables.py`: rebuilds `encounters.csv`, `patients.csv`, `drg_groups.csv` and `02_load_tables.sql`
- [ ] `python make_eval_set.py`: 15 questions with gold SQL in `eval_questions.csv` and `04_gold_answers.sql`
- [ ] `python make_semantic_sql.py`: pastes the YAML into `03_semantic_view.sql`
- [ ] `python local_eval.py`: YAML checks, gold SQL vs pandas, verified queries vs gold, the comparator and the guard

Lesson-box results (real run): YAML 0 problems (3 tables, 25 fields, 39 synonyms, 2 relationships, 5 verified queries). Gold SQL matched pandas on 13 of 13 questions. All 5 verified queries matched their gold answers on DuckDB views built from the YAML. The comparator caught 6 of 6 hand-written mistakes (admit date instead of discharge date, encounters counted as patients, missing AMA filter, an off-by-one age band, exact text match on a DRG name, admit month instead of discharge month) and accepted 4 of 4 correct rewrites. Overall readmission rate 0.185.

## Main path (Snowsight worksheet, about 60 to 90 minutes)

Open https://app.snowflake.com/us-east-1/rhc24484/ and create a SQL worksheet. Run the files in order:

- [ ] `01_setup.sql`: `DS90_XS` (X-Small, AUTO_SUSPEND = 60), schema `DS90.DAY39`, stage `SEMANTIC_MODELS`, access checks, leftovers
- [ ] `02_load_tables.sql`: `ENCOUNTERS` (2,325), `PATIENTS` (1,000), `DRG_GROUPS` (100). Check the counts and the 0.185 rate
- [ ] `03_semantic_view.sql` A: upload `readmit_semantic_model.yaml` to the stage (Snowsight upload, `snow stage copy`, or PUT with AUTO_COMPRESS = FALSE), then `LS @SEMANTIC_MODELS`
- [ ] `03_semantic_view.sql` B: verify the YAML (`SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML(..., TRUE)`), then create the semantic view `READMIT_ANALYST`; SHOW / DESCRIBE it
- [ ] `03_semantic_view.sql` C: query the semantic view yourself with `SEMANTIC_VIEW(... DIMENSIONS ... METRICS ...)`. No LLM, no message charge
- [ ] `04_gold_answers.sql`: run the 13 gold queries, read each answer, create `ANALYST_RUNS`
- [ ] Ask Cortex Analyst: `python day39_analyst.py --ask "What is the readmission rate by payer?"`, then the full run `python day39_analyst.py --save-table` (15 messages)
- [ ] Optional: `python day39_analyst.py --source stage --ask "..."` to use the stage file instead of the view
- [ ] Optional: `streamlit_app.py` as a Streamlit in Snowflake app (ask 2 or 3 questions, then close it)
- [ ] `05_evaluate.sql`: HASH_AGG fingerprints of gold vs generated results, the scorecard, and the mismatches side by side
- [ ] Fix one miss: add a synonym, a description, or a verified query to the YAML, re-run `make_semantic_sql.py` and part B of 03, ask that question again (1 more message)
- [ ] `06_guardrails_cost.sql`: read-only role `DS90_ANALYST_RO`, statement timeout, optional resource monitor, usage views, suspend

## Python setup (VS Code + conda env ds90)

```
pip install snowflake-connector-python requests pandas pyyaml
python day39_analyst.py --dry-run          # no account: check files, print a request body
python day39_analyst.py --ask "Which service line has the highest readmission rate?"
python day39_analyst.py --save-table       # 15 questions = 15 messages, writes analyst_eval_results.csv
```

Uses the `[ds90]` entry in `~/.snowflake/connections.toml` from Day 36 (or `SNOWFLAKE_*` env vars). The script sends the session token in the `Authorization: Snowflake Token="..."` header, the way the Snowflake sample apps do. It caps messages (`--max-messages 20`), refuses anything that is not one SELECT on the three tables, pulls at most 1,000 rows, sets a 60 second statement timeout, and suspends the warehouse at the end.

## Cost notes

- Cortex Analyst: billed per successful message (HTTP 200), not per token. 15 questions = 15 messages; every follow-up turn and every re-ask is another message. Rate: Snowflake Service Consumption Table, https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf (not copied here).
- Warehouse: runs the generated SQL and your gold SQL. X-Small = 1 credit per hour, billed per second with a 60 second minimum per resume. Lesson-box arithmetic for 15 questions two minutes apart: AUTO_SUSPEND = 60 bills about 930 seconds (0.258 credits); AUTO_SUSPEND = 600 keeps it awake about 2,282 seconds (0.634 credits).
- Querying the semantic view directly with `SEMANTIC_VIEW(...)` costs warehouse time only, no Analyst messages.
- A semantic view is metadata (no running cost). The stage file is a few KB of storage. Keep both for Day 40.
- Usage: `SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY` (REQUEST_COUNT, CREDITS per hour; can lag a few hours).

## My results (fill in)

- Questions correct ___ / 15 (match or correct refusal). With a verified query ___ / 5, without ___ / 10
- Did Q14 (predict next week) and Q15 (ambiguous) get NO SQL? ____
- Exact HASH_AGG matches ___ / 13; fair answers that only differed in rounding or extra columns ___
- Verified query used (from the `confidence` field) on which questions? ____
- One miss I fixed in the YAML, and what changed: ____
- Messages sent ___ ; Analyst credits ___ ; warehouse credits ___ (06_guardrails_cost.sql)
- One takeaway: ____

## Files

- `gen_tables.py`, `sim_readmit.py`, `encounters.csv`, `patients.csv`, `drg_groups.csv`: simulated tables
- `readmit_semantic_model.yaml`: the semantic model (tables, dimensions, time dimensions, facts, metrics, filters, relationships, custom instructions, verified queries)
- `make_eval_set.py`, `eval_questions.csv`: 15 labeled questions with gold SQL
- `01_setup.sql` to `06_guardrails_cost.sql`: the Snowsight path (`02`, `03`, `04` are generated)
- `analyst_eval.py`: `compare_results` (tolerant result-set match) and `guard_sql` (read-only check)
- `local_eval.py`, `local_eval_output.txt`: offline checks and their real output
- `day39_analyst.py`: REST API client and scorer; `streamlit_app.py`: Streamlit in Snowflake chat

## Commit (your own repo)

```
cd ~/ds-genai-90-days
mkdir -p day39        # copy the exercise files (SQL, .py, .yaml, .csv, README.md) into day39/
cd day39 && python gen_tables.py && python make_eval_set.py && python local_eval.py && cd ..
git status            # check: no connections.toml, no .env, no tokens, no analyst_eval_results.csv with real account names
git add day39
git commit -m "Day 39: Cortex Analyst text-to-SQL with a semantic model over simulated readmission tables"
git push
```
