# Day 40: Snowflake week mini project with a guided Cortex Code (CoCo) session and a cost check

One small, honest pipeline that uses each Snowflake day once: a **model score** shortlists patients (Day 36 idea), an **LLM reads only the shortlisted notes** for two facts the score cannot see (Day 37), each patient gets a **guidance document** (Day 38 documents) and a **service-line baseline** from the semantic view (Day 39), and the result is a 25-patient follow-up call worklist. You build parts of it with **Cortex Code (CoCo) in Snowsight**, check its work against reference SQL, and finish with a **cost and credit check** of the whole week.

All data is SIMULATED (seed 40): 300 discharges Nov 16-22, 2026, with templated notes. `RISK_SCORE` is a simulated model probability that only saw structured columns (a stand-in for the Day 36 registry model, so you do not need the Day 36 features). No real patients, no real hospital. Never put credentials in this folder or in git.

What ran where: the lesson author did NOT run any SQL against Snowflake, did not use CoCo, and made no AI_COMPLETE calls (the lesson computer has no Snowflake connection). Only the pure-Python parts ran: `gen_discharges.py`, `local_check.py` (pandas and DuckDB on the CSVs) and `day40_cost_report.py --dry-run`. Your trial results are the real ones. Write them below.

## Offline first (no account, no cost, about 10 minutes)

- [ ] `pip install pandas numpy duckdb`
- [ ] `python gen_discharges.py`: rebuilds the CSVs and `02_load.sql`
- [ ] `python local_check.py`: rule-based note reader vs the answer key, the worklist logic in DuckDB, token and warehouse arithmetic

Lesson-box results (real run): 300 discharges, readmission rate 0.263 (simulated), 26% live alone, 63% have a follow-up booked. A regex reader got lives-alone right on 91.3% of notes and follow-up booked on 79.7% (68.3% on the 60 shortlisted). Readmissions caught in a 25-patient list: score only 16 (0.64), score plus regex note fields 18 (0.72), score plus the answer-key fields 21 (0.84). AUC on all 300: score 0.718, score plus answer-key note fields 0.798. The notes carry real signal in this simulation; the question for your trial is how much of it the LLM recovers.

## Main path (Snowsight, about 60 to 90 minutes)

Open https://app.snowflake.com/us-east-1/rhc24484/ (login in your vault, not in these files). Work in a workspace so CoCo can see your files.

- [ ] `01_setup.sql`: `DS90_XS` (X-Small, AUTO_SUSPEND = 60), schema `DS90.DAY40`, inventory of Days 36-39, CoCo access checks, a personal CoCo daily credit cap, `COCO_LOG`
- [ ] `02_load.sql`: 300 / 300 / 300 / 9 / 9 rows; mean score 0.201, readmission rate 0.263
- [ ] CoCo prompts P1 and P2 (`cortex_code_session.md`): explore in a read-only chat, test the restriction
- [ ] `03_extract.sql`: count tokens (free), shortlist the top 60, try 3 notes, then `NOTE_FIELDS` for all 60 once. Expect 0 unparsed rows
- [ ] CoCo prompt P3: explain part D and the token math
- [ ] `04_context.sql`: Day 39 semantic view rates by service line (or the fallback table), Day 38 guidance docs (plain join, no service needed)
- [ ] `05_worklist.sql`: `FOLLOWUP_WORKLIST` (25 rows) and one AI_COMPLETE weekly brief built from aggregates only
- [ ] CoCo prompts P4 to P7: generate the worklist query and prove it with MINUS, Fix button, review a wrong-but-running query, edit with a diff
- [ ] `06_evaluate.sql`: LLM extraction accuracy vs the answer key; readmissions caught, score only vs score plus note fields
- [ ] CoCo prompt P8, then `07_cost_check.sql`: credits by day and service, AI functions, Analyst, Search, CoCo, query history, resource monitor, cleanup and SUSPEND
- [ ] Optional: `python day40_cost_report.py` for CSVs and a chart of your week

## Cost notes

- Warehouse: X-Small = 1 credit per hour, billed per second with a 60 second minimum per resume. Lesson-box arithmetic for a simulated 72-minute session of 30 statements: AUTO_SUSPEND = 60 bills about 1,731 seconds (0.481 credits); 300 bills 4,307 (1.196); 600 bills 4,944 (1.373).
- AI Functions (AI_COMPLETE): billed per token, input plus output. Estimate (characters divided by 4): about 9,800 input tokens for the 60 shortlisted notes vs 48,000 for all 300. Use `AI_COUNT_TOKENS` on your trial for the real count.
- CoCo: billed per token at the chosen model's rate (Auto Efficient adds a routing fee). Separate views per surface: `CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY`, `CORTEX_CODE_CLI_USAGE_HISTORY`, `CORTEX_CODE_DESKTOP_USAGE_HISTORY`.
- A resource monitor caps warehouse credits only. It does not stop AI spend. For CoCo use the per-user daily limit parameters; for everything else, keep calls small and saved once into tables.
- `CORTEX_FUNCTIONS_USAGE_HISTORY` is no longer updated; use `CORTEX_AISQL_USAGE_HISTORY` (docs, Nov 2026).
- Rates: Snowflake Service Consumption Table, https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf (not copied here).

## My results (fill in)

- LLM extraction accuracy on 60 notes: lives alone ___ , follow-up booked ___ (regex baseline 0.833 / 0.683)
- Readmissions caught in 25: score only ___ , score plus LLM fields ___ (lesson box: 16 / regex 18 / answer key 21)
- CoCo prompts: P4 MINUS rows ___ ; did P6 catch the wrong denominator? ___ ; did P2 get blocked? ___
- CoCo requests ___ , CoCo credits ___ ; AI Functions credits ___ ; warehouse credits this week ___
- Biggest cost line of the week: ________ ; one thing I would change: ________
- One takeaway: ____

| Prompt | What CoCo did | Ran SQL? | Matched reference? | My edits |
|---|---|---|---|---|
| P1 | | | | |
| P4 | | | | |
| P6 | | | | |
| P8 | | | | |

## Files

- `gen_discharges.py`, `discharges.csv`, `note_labels.csv`, `outcomes.csv`, `baseline_rates.csv`, `guidance_map.csv`: simulated data and answer keys
- `encounters.csv`, `drg_groups.csv`: copied from Day 39, used only to compute the baseline rates
- `01_setup.sql` to `07_cost_check.sql`: the Snowsight path (`02` is generated)
- `cortex_code_session.md`: the 8 CoCo prompts and what a good answer looks like; `cortex_code_broken.sql`: two bugs on purpose
- `local_check.py`, `local_check_output.txt`: offline checks and their real output
- `day40_cost_report.py`: read-only cost report from your ACCOUNT_USAGE views (`--dry-run` works without an account)

## Commit (your own repo)

```
cd ~/ds-genai-90-days
mkdir -p day40        # copy the exercise files (SQL, .py, .md, .csv, README.md) into day40/
cd day40 && python gen_discharges.py && python local_check.py && cd ..
git status            # check: no connections.toml, no .env, no tokens, no cost CSVs with account details
git add day40
git commit -m "Day 40: Snowflake week mini project, guided Cortex Code session, cost and credit check"
git push
```
