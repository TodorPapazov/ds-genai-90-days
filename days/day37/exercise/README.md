# Day 37: Snowflake Cortex LLM functions in SQL (simulated discharge notes)

Run LLM functions in plain SQL, right next to the table: summarize discharge notes, sort them into readmission-risk bands, pull out the follow-up appointment, and score patient feedback. Then grade the answers against an answer key and check what it cost. All 40 notes are SIMULATED (seed 37). They are not real patients. Never put credentials in this folder or in git.

The lesson author did not run these SQL scripts or `day37_cortex.py` against Snowflake (the lesson computer has no Snowflake connection). Only the pure-Python parts ran: `gen_notes.py`, `local_baseline.py`, and `day37_cortex.py --dry-run`. Your trial results are the real ones. Write them below.

## Main path (Snowsight worksheet, about 60 to 90 minutes)

Open https://app.snowflake.com/us-east-1/rhc24484/ and create a SQL worksheet. Run the files in order:

1. `01_setup.sql`: X-Small warehouse `DS90_XS` (AUTO_SUSPEND = 60), schema `DS90.DAY37`, the cross-region setting, `SHOW CORTEX BASE MODELS`, and a one-line smoke test.
2. `02_load_notes.sql`: creates `DISCHARGE_NOTES` (40 notes) and `NOTE_LABELS` (answer key). Expect 40 rows.
3. `03_cortex_functions.sql`: `SNOWFLAKE.CORTEX.SUMMARIZE`, `SENTIMENT`, `CLASSIFY_TEXT`, `EXTRACT_ANSWER`, `COMPLETE`. One row first, then 5, then all 40 rows ONCE into `NOTE_AI`.
4. `04_ai_functions.sql`: the newer names `AI_COMPLETE` (with a JSON schema), `AI_CLASSIFY`, `AI_SENTIMENT` (with aspects), `AI_EXTRACT`, `AI_SUMMARIZE_AGG`, and the free `AI_COUNT_TOKENS`.
5. `05_evaluate.sql`: accuracy against `NOTE_LABELS`, a confusion table, a hallucination check on the 8 notes with no follow-up, and a call list.
6. `06_cost_check.sql`: Cortex token and credit views, warehouse credits, then `ALTER WAREHOUSE DS90_XS SUSPEND`.

## Offline first (no account, no cost)

```
python gen_notes.py          # rebuilds the CSVs and 02_load_notes.sql (seed 37)
python local_baseline.py     # keyword / regex / word-list baseline + token estimate
```

Lesson-box results for the baseline: keyword risk accuracy 0.375, regex follow-up accuracy 1.000 (the text is templated), word-list tone accuracy 0.800. About 19,500 tokens of your own text for one full pass of part D, before each function's built-in prompt.

## Optional Python path (VS Code + conda env)

```
python day37_cortex.py --dry-run     # size the run, no account
python day37_cortex.py               # 5 notes on the trial
python day37_cortex.py --rows 40     # all 40 notes, saved to NOTE_AI_PY
```

Uses the `[ds90]` entry in `~/.snowflake/connections.toml` from Day 36, or `SNOWFLAKE_*` env vars in your shell.

## Cost notes

- Two meters: the warehouse (X-Small, billed while it runs) and Cortex tokens (credits per million tokens, set per function and model).
- Rates are in the Snowflake Service Consumption Table: https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf. This lesson does not copy prices. Check the current PDF.
- Billed tokens are higher than your text: the task functions add their own prompt, CLASSIFY_TEXT labels and descriptions count for every row, and COMPLETE bills output tokens too.
- Snowflake docs recommend a warehouse no larger than MEDIUM for Cortex. X-Small is enough.
- `llama3.1-8b` runs natively in AWS US East 1. Bigger models (Claude, OpenAI) need cross-region inference.

## My results (fill in)

- Cortex risk accuracy: ____ (keyword baseline 0.375)
- Follow-up accuracy: ____ (regex 1.000). Did EXTRACT_ANSWER invent a follow-up for any of the 8 notes without one? ____
- Feedback tone accuracy: ____ (word list 0.800)
- Cortex credits today (from 06_cost_check.sql): ____
- One takeaway: ____

## Files

- `gen_notes.py`, `discharge_notes.csv`, `note_labels.csv`: simulated data and answer key
- `01_setup.sql` to `06_cost_check.sql`: the Snowsight path
- `local_baseline.py`: the no-LLM baseline
- `day37_cortex.py`: optional Snowpark path

## Commit

```
cd ~/ds-genai-90-days
git add day37
git commit -m "Day 37: Cortex LLM functions in SQL on simulated discharge notes"
git push
```
