# Day 38: Snowflake Cortex Search over simulated clinical documents (a first look at RAG)

Build a small retrieval pipeline inside Snowflake: chunk a library of clinical documents, index it with one `CREATE CORTEX SEARCH SERVICE` statement, search it with filters, feed the top chunks to `COMPLETE` for an answer with citations, and measure retrieval with hit rate@k and MRR on a labeled question set. Then check the cost and suspend or drop the service.

All 33 documents (102 chunks) and the 50 questions are SIMULATED (seed 38). They are invented practice text: not medical advice, not a real hospital's policies, and no patient data. Never put credentials in this folder or in git.

What ran where: the lesson author did NOT run any SQL against Snowflake, did not create a search service, and made no Cortex calls (the lesson computer has no Snowflake connection). Only the pure-Python parts ran: `gen_docs.py`, `local_baseline.py`, and `day38_search.py --dry-run`. Your trial results are the real ones. Write them below.

## Offline first (no account, no cost)

- [ ] `python gen_docs.py`: rebuilds `clinical_docs.csv`, `doc_chunks.csv`, `eval_questions.csv` and `02_load_docs.sql`
- [ ] `python local_baseline.py`: keyword-only retrieval (TF-IDF, BM25, char n-gram TF-IDF), hit rate@k, MRR, filter test, chunks vs whole docs, a sample grounded prompt, and a size/cost estimate

Lesson-box results (real run, 47 answerable questions):

| Method (local, no embeddings) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| TF-IDF words | 0.702 | 0.894 | 0.894 | 0.794 |
| BM25 | 0.723 | 0.894 | 0.894 | 0.801 |
| TF-IDF char 3-5 | 0.851 | 0.915 | 0.936 | 0.885 |

BM25 hit@3 is 1.000 on the 23 keyword questions but 0.792 on the 24 paraphrase questions. On the 4 filtered questions, the attribute filter lifts BM25 hit@3 from 0.750 to 1.000. About 4,050 embedding tokens (chars / 4) to build the index.

## Main path (Snowsight worksheet, about 60 to 90 minutes)

Open https://app.snowflake.com/us-east-1/rhc24484/ and create a SQL worksheet. Run the files in order:

- [ ] `01_setup.sql`: X-Small warehouse `DS90_XS` (AUTO_SUSPEND = 60), schema `DS90.DAY38`, role and grants check, leftover services
- [ ] `02_load_docs.sql`: `CLINICAL_DOCS` (33), `DOC_CHUNKS` (102), `EVAL_QUESTIONS` (50). Check the counts
- [ ] `03_chunk.sql`: `SPLIT_TEXT_RECURSIVE_CHARACTER` in SQL, compare with the Python chunks, see overlap, `COUNT_TOKENS`
- [ ] `04_create_service.sql`: `CREATE CORTEX SEARCH SERVICE CLINICAL_DOC_SEARCH ON CHUNK_TEXT ... ATTRIBUTES ... WAREHOUSE ... TARGET_LAG ... EMBEDDING_MODEL`, then `SHOW` / `DESCRIBE` / `CORTEX_SEARCH_DATA_SCAN`
- [ ] `05_query.sql`: `SNOWFLAKE.CORTEX.SEARCH_PREVIEW` (raw JSON, FLATTEN to rows), filters (`@eq`, `@and` + `@gte` on a date), and `CORTEX_SEARCH_RETRIEVE` with one query per row
- [ ] `06_rag_complete.sql`: top 3 chunks into `SNOWFLAKE.CORTEX.COMPLETE` with citations; save 8 answers ONCE into `RAG_ANSWERS`; check citations and refusals without calling the model again
- [ ] `07_evaluate.sql`: hit@3 in one statement, then full ranks (Snowflake Scripting loop) for hit@1/3/5 and MRR by question type, next to the local baselines; read every miss
- [ ] `08_cost_cleanup.sql`: Cortex Search usage views (they lag), `ALTER CORTEX SEARCH SERVICE ... SUSPEND`, `DROP CORTEX SEARCH SERVICE`, suspend `DS90_XS`

## Optional Python path (VS Code + conda env)

```
pip install snowflake snowflake-snowpark-python   # snowflake = snowflake.core (Root, cortex_search_services)
python day38_search.py --dry-run                  # no account: check files, print the plan
python day38_search.py                            # trial: hit@k and MRR on the 47 labeled questions
python day38_search.py --ask "How long can an opened insulin pen stay out of the fridge?"
python day38_search.py --rag-test                 # 3 answerable + 3 unanswerable through COMPLETE
```

Uses the `[ds90]` entry in `~/.snowflake/connections.toml` from Day 36, or `SNOWFLAKE_*` env vars in your shell. Run it before `08_cost_cleanup.sql` drops the service.

## Cost notes

- Warehouse: builds and refreshes the index and runs your SQL. X-Small uses 1 credit per hour while running, billed per second with a 60 second minimum each time it resumes. Docs: no larger than MEDIUM for Cortex Search.
- Embedding tokens: each chunk is embedded once at build time, and again only when its row changes. About 4,050 tokens here.
- Serving: credits per GB-month of indexed data (your columns plus 768 x 4 bytes of vector per row), billed while the service is RESUMED, even with zero queries. About 0.34 MB here.
- COMPLETE tokens for each RAG answer (about 200 input tokens per question with k = 3).
- Rates: Snowflake Service Consumption Table, https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf (not copied here).
- At the end: suspend the service, then drop it once your results are recorded. Rebuilding is cheap.

## My results (fill in)

- Cortex Search hit@1 ____ hit@3 ____ MRR ____ (BM25 local: 0.723 / 0.894 / 0.801)
- Paraphrase questions hit@3 ____ (BM25 local 0.792)
- Filtered questions: did the date filter keep the 2025 fall policy out? ____
- RAG: answers citing the gold chunk ___ / 5, refusals on unanswerable ___ / 3, uncited or invented claims ___
- Credits today (08_cost_cleanup.sql): embed ____ serving ____ COMPLETE ____ warehouse ____
- One takeaway: ____

## Files

- `gen_docs.py`, `clinical_docs.csv`, `doc_chunks.csv`, `eval_questions.csv`: simulated library and labeled questions
- `01_setup.sql` to `08_cost_cleanup.sql`: the Snowsight path
- `local_baseline.py`, `baseline_output.txt`: the keyword-only baseline and its real output
- `day38_search.py`: optional Python path (snowflake.core)

## Commit (your own repo)

```
cd ~/ds-genai-90-days
mkdir -p day38        # copy the exercise files (SQL, .py, .csv, README.md) into day38/
cd day38 && python gen_docs.py && python local_baseline.py && cd ..
git status            # check: no connections.toml, no .env, no passwords
git add day38
git commit -m "Day 38: Cortex Search over simulated clinical documents, first look at RAG"
git push
```
