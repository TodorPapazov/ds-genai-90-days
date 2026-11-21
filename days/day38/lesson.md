# Day 38: Snowflake Cortex Search over simulated clinical documents: chunking, filters, grounded answers and a first look at RAG

## Concept
Day 38 is your first RAG (retrieval augmented generation) pipeline. Retrieve: search your own documents for the few chunks that answer a question. Augment: paste those chunks, with their ids, into the prompt. Generate: the LLM answers only from that context, cites the ids, and says 'Not found in the documents' when the answer is not there. Chunking: the default embedding model (snowflake-arctic-embed-m-v1.5, 768 dimensions) reads at most 512 tokens per row and truncates the rest, so long documents are split first (docs: no more than 512 tokens, about 385 words). SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER does it in SQL. Cortex Search is a managed hybrid (vector plus keyword) index: one CREATE CORTEX SEARCH SERVICE ... ON text column, ATTRIBUTES (the columns you can filter on), WAREHOUSE, TARGET_LAG and an optional EMBEDDING_MODEL. Query it with SNOWFLAKE.CORTEX.SEARCH_PREVIEW in SQL (literal JSON: query, columns, filter, limit), with CORTEX_SEARCH_RETRIEVE for one query per table row, or with the Python API (snowflake.core Root(...).cortex_search_services[...].search(query, columns, filter, limit)). Filters use @eq, @contains, @gte, @lte, @primarykey and @and / @or / @not. Syntax checked in the Snowflake docs on Nov 21, 2026. Healthcare framing: 33 simulated documents (discharge instructions, care protocols, patient education, policies; 16 departments) split into 102 chunks, plus 47 labeled questions with the one chunk that answers each, and 3 questions the library cannot answer. Invented text: not medical advice, no patient data. Measure before you trust: hit rate@k (gold chunk in the top k) and MRR (mean of 1/rank). What ran on the lesson box (no ~/.snowflake, no SNOWFLAKE_* vars): the data generator, a keyword-only baseline (BM25 hit@1 0.723, hit@3 0.894, MRR 0.801; hit@3 1.000 on keyword questions but 0.792 on paraphrases), and day38_search.py --dry-run. Not run by the bot: any Snowflake SQL, the search service, and every Cortex call. Cost, four meters: warehouse DS90_XS (X-Small, AUTO_SUSPEND = 60) to build and refresh the index; embedding tokens once (about 4,050); serving credits per GB-month while the service is resumed, even with zero queries (about 0.34 MB indexed); and COMPLETE tokens per answer. Rates: Snowflake Service Consumption Table (not copied here). Suspend the service, then drop it when your numbers are written down. Next: Day 39, Cortex Analyst (text-to-SQL with a semantic model over simulated readmission tables).

## Key numbers
- **Data:** 33 simulated clinical documents, 102 chunks (avg 158 chars), 16 departments, 4 doc types. 47 labeled questions (23 keyword, 24 paraphrase, 4 with a filter) + 3 unanswerable
- **Local BM25 (real run):** hit@1 0.723, hit@3 0.894, hit@5 0.894, MRR 0.801
- **Other local baselines:** TF-IDF words hit@3 0.894, MRR 0.794. Char 3-5 TF-IDF hit@3 0.915, MRR 0.885
- **Paraphrase gap:** BM25 hit@3: keyword questions 1.000, paraphrase questions 0.792
- **Filters:** 4 filtered questions, BM25 hit@3 0.75 without the filter, 1.00 with it
- **Size:** About 4,050 embedding tokens once; about 0.34 MB indexed (102 rows x 768-dim vectors + columns)
- **Warehouse:** DS90_XS, X-Small, AUTO_SUSPEND=60. Service TARGET_LAG '1 day', AUTO_SUSPEND 1800 (serving)
- **Not run here:** All Snowflake SQL, the search service, SEARCH_PREVIEW and COMPLETE calls (no connection on the lesson box)

## Why it matters for healthcare data science jobs
Healthcare DS and Gen AI postings increasingly list RAG, embeddings, vector or semantic search, and LLM evaluation. After today you can say, honestly, that you built a small retrieval pipeline in Snowflake on practice data (chunking, a Cortex Search service with attribute filters, grounded answers with citations) and measured retrieval with hit rate@k and MRR against a keyword baseline. It is a learning project on simulated documents, not production experience.

## Docs checked (Nov 21, 2026)
- Cortex Search overview (chunking, embedding models): https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview
- CREATE CORTEX SEARCH SERVICE: https://docs.snowflake.com/en/sql-reference/sql/create-cortex-search
- Query a Cortex Search Service (Python API, filters): https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service
- SEARCH_PREVIEW: https://docs.snowflake.com/en/sql-reference/functions/search_preview-snowflake-cortex
- CORTEX_SEARCH_RETRIEVE: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-retrieve
- ALTER CORTEX SEARCH SERVICE (SUSPEND / RESUME): https://docs.snowflake.com/en/sql-reference/sql/alter-cortex-search
- SPLIT_TEXT_RECURSIVE_CHARACTER: https://docs.snowflake.com/en/sql-reference/functions/split_text_recursive_character-snowflake-cortex
- Cortex Search costs: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-costs
- Service Consumption Table: https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf

## Hands-on (60-90 min)
1. Offline first: python gen_docs.py and python local_baseline.py. Note BM25 hit@3 0.894, MRR 0.801, and paraphrase hit@3 0.792.
2. Open Snowsight at app.snowflake.com/us-east-1/rhc24484/ and a new SQL worksheet. The login is in your vault, not in these files.
3. Run snippet 1 (01_setup.sql): X-Small DS90_XS with AUTO_SUSPEND = 60, schema DS90.DAY38, grants, and SHOW CORTEX SEARCH SERVICES for leftovers.
4. Run 02_load_docs.sql (from the exercise folder). Expect 33 documents, 102 chunks and 50 questions.
5. Snippet 2 (03_chunk.sql): split the documents in SQL with SPLIT_TEXT_RECURSIVE_CHARACTER, compare with the Python chunks, look at overlap, count tokens.
6. Snippet 3 (04_create_service.sql): create CLINICAL_DOC_SEARCH, then SHOW / DESCRIBE it and count indexed rows with CORTEX_SEARCH_DATA_SCAN.
7. Snippet 4 (05_query.sql): SEARCH_PREVIEW as raw JSON and as rows, then filters: @eq Orthopedics and @and policy + @gte 2026-01-01.
8. Snippet 5 (06_rag_complete.sql): one grounded answer with citations, then 8 answers saved once into RAG_ANSWERS. Count citations of the gold chunk and refusals.
9. Snippet 6 (07_evaluate.sql): hit@1/3/5 and MRR by question type next to the local baselines. Read every miss.
10. Optional: snippet 7, python day38_search.py (and --rag-test) from VS Code with your Day 36 [ds90] connection.
11. Snippet 8 (08_cost_cleanup.sql): usage views (they lag), then SUSPEND and DROP the service and suspend DS90_XS.
12. Fill in 'My results' in day38/README.md, keep the simulated-data note, then commit and push (snippet 10).

## Code
Exercise files: `exercise/01_setup.sql`, `exercise/02_load_docs.sql`, `exercise/03_chunk.sql`, `exercise/04_create_service.sql`, `exercise/05_query.sql`, `exercise/06_rag_complete.sql`, `exercise/07_evaluate.sql`, `exercise/08_cost_cleanup.sql`, `exercise/gen_docs.py`, `exercise/local_baseline.py`, `exercise/day38_search.py`, `exercise/clinical_docs.csv`, `exercise/doc_chunks.csv`, `exercise/eval_questions.csv`, `exercise/baseline_output.txt`, `exercise/README.md`.

### 1. Setup: X-Small warehouse, schema, grants, leftover services (SQL)
```sql
-- Day 38: Cortex Search over SIMULATED clinical documents (a first look at RAG).
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Three meters run today:
--   1) the warehouse DS90_XS (X-Small, billed per second while running, AUTO_SUSPEND = 60 seconds).
--      It builds and refreshes the search index and runs your SQL.
--   2) embedding tokens: every chunk is embedded once when the index is built (and again only if the row changes).
--   3) serving: credits per GB per month of indexed data, charged while the service is RESUMED, even with no queries.
--   Plus COMPLETE tokens for the RAG answers in 06_rag_complete.sql.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
-- Docs: no larger than MEDIUM for Cortex Search. Indexing time is mostly the embedding step,
-- which does not speed up with a bigger warehouse.
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY38
  COMMENT = 'Day 38: Cortex Search on SIMULATED clinical documents. Not real patient data or real policies.';

USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- Who am I, where am I? Cortex Search runs natively in AWS US East 1 (your trial region).
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- The creator role needs the SNOWFLAKE.CORTEX_USER (or CORTEX_EMBED_USER) database role,
-- CREATE CORTEX SEARCH SERVICE on the schema, SELECT on the source table and USAGE on the warehouse.
-- On a trial, ACCOUNTADMIN or SYSADMIN normally has all of these. Check what your role inherits:
SHOW GRANTS TO ROLE IDENTIFIER(CURRENT_ROLE());

-- Any search services left over from earlier practice? Each one that is RESUMED costs serving credits.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;
```

### 2. Chunking in SQL with SPLIT_TEXT_RECURSIVE_CHARACTER (SQL)
```sql
-- Day 38: chunking. Cortex Search embeds each row of the search column. The default embedding model
-- (snowflake-arctic-embed-m-v1.5) reads at most 512 tokens per row and TRUNCATES anything longer, so
-- long documents must be split first. Docs advice: chunks of no more than 512 tokens (about 385 words).
-- DOC_CHUNKS (loaded in 02) was already split by paragraph in Python. Here you do the same split in SQL.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. Split one document: 450 characters per chunk, no overlap. Returns an ARRAY of strings.
SELECT DOC_ID, TITLE,
       SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(DOC_TEXT, 'none', 450, 0) AS CHUNKS
FROM CLINICAL_DOCS
WHERE DOC_ID = 'DOC001';

-- B. Split every document into rows with LATERAL FLATTEN. Prepend the title so each chunk still says
--    which document it came from (a cheap trick that often helps retrieval).
CREATE OR REPLACE TABLE DOC_CHUNKS_SQL AS
SELECT d.DOC_ID || '-S' || (c.INDEX + 1)         AS CHUNK_ID,
       d.DOC_ID, c.INDEX + 1 AS CHUNK_NO, d.TITLE, d.DEPARTMENT, d.DOC_TYPE, d.EFFECTIVE_DATE,
       d.TITLE || '. ' || c.VALUE::STRING         AS CHUNK_TEXT
FROM CLINICAL_DOCS d,
     LATERAL FLATTEN(INPUT => SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(d.DOC_TEXT, 'none', 450, 0)) c;

-- C. Compare the two chunkings. The default separators split on paragraph breaks first ("\n\n"),
--    so the counts should be close to the Python version.
SELECT 'python (DOC_CHUNKS)' AS SOURCE, COUNT(*) AS N_CHUNKS, ROUND(AVG(LENGTH(CHUNK_TEXT))) AS AVG_CHARS,
       MAX(LENGTH(CHUNK_TEXT)) AS MAX_CHARS FROM DOC_CHUNKS
UNION ALL
SELECT 'sql (DOC_CHUNKS_SQL)', COUNT(*), ROUND(AVG(LENGTH(CHUNK_TEXT))), MAX(LENGTH(CHUNK_TEXT)) FROM DOC_CHUNKS_SQL;

-- D. Try a tiny chunk size with overlap to SEE what overlap does (each chunk repeats the end of the last one).
SELECT SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(DOC_TEXT, 'none', 120, 30) AS SMALL_CHUNKS
FROM CLINICAL_DOCS WHERE DOC_ID = 'DOC013';

-- E. Count embedding tokens for 5 chunks (COUNT_TOKENS has no token charge; it uses warehouse time only).
--    Every chunk should be far below the 512-token window.
SELECT CHUNK_ID, LENGTH(CHUNK_TEXT) AS CHARS,
       SNOWFLAKE.CORTEX.COUNT_TOKENS('snowflake-arctic-embed-m', CHUNK_TEXT) AS EMBED_TOKENS
FROM DOC_CHUNKS ORDER BY CHUNK_ID LIMIT 5;
-- The local baseline estimates about 4,050 tokens (chars / 4) to embed all 102 chunks once.
```

### 3. Create the Cortex Search service (SQL)
```sql
-- Day 38: create the Cortex Search service over the chunks.
-- One statement builds a hybrid (vector + keyword) index: Snowflake embeds CHUNK_TEXT for you.
-- Syntax checked against the CREATE CORTEX SEARCH SERVICE docs on Nov 21, 2026.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

CREATE OR REPLACE CORTEX SEARCH SERVICE CLINICAL_DOC_SEARCH
  ON CHUNK_TEXT                                    -- the text column to search (must be text)
  PRIMARY KEY (CHUNK_ID)                           -- TEXT key: cheaper, faster incremental refresh
  ATTRIBUTES DEPARTMENT, DOC_TYPE, EFFECTIVE_DATE  -- columns you can FILTER on at query time
  WAREHOUSE = DS90_XS                              -- runs the source query and builds/refreshes the index
  TARGET_LAG = '1 day'                             -- how stale the index may get. Practice data never changes,
                                                   -- so a long lag means fewer refresh checks.
  EMBEDDING_MODEL = 'snowflake-arctic-embed-m-v1.5'  -- the default; 768 dims, 512-token window, English.
                                                     -- Cannot be changed later without CREATE OR REPLACE.
  AUTO_SUSPEND = 1800                              -- suspend SERVING after 30 min with no queries (minimum 1800).
                                                   -- If your account rejects this line, delete it and suspend
                                                   -- by hand in 08_cost_cleanup.sql.
  COMMENT = 'Day 38 practice on SIMULATED clinical documents'
AS (
  SELECT CHUNK_ID, DOC_ID, TITLE, DEPARTMENT, DOC_TYPE, EFFECTIVE_DATE, CHUNK_TEXT
  FROM DOC_CHUNKS
);
-- INITIALIZE defaults to ON_CREATE, so the statement waits until the index is built (102 short rows: quick).

-- Is it ready? Look at indexing_state and serving_state, and the source row count.
SHOW CORTEX SEARCH SERVICES IN SCHEMA DS90.DAY38;
DESCRIBE CORTEX SEARCH SERVICE CLINICAL_DOC_SEARCH;

-- What actually got indexed (the materialized source query). Expect 102 rows.
SELECT COUNT(*) AS INDEXED_ROWS
FROM TABLE(CORTEX_SEARCH_DATA_SCAN(SERVICE_NAME => 'CLINICAL_DOC_SEARCH'));
```

### 4. Query with SEARCH_PREVIEW, filters and CORTEX_SEARCH_RETRIEVE (SQL)
```sql
-- Day 38: query the service. SEARCH_PREVIEW is the SQL way to TEST a service. It takes a JSON string with
--   query (required), columns, filter, limit (default 10). It only accepts constant (literal) arguments.
-- For apps, use the Python or REST API; for one query per table row in SQL, use CORTEX_SEARCH_RETRIEVE (part E).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. The raw response: a JSON object with "results" (best match first) and a request_id.
SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'DS90.DAY38.CLINICAL_DOC_SEARCH',
  '{"query": "how long can an opened insulin pen stay out of the fridge",
    "columns": ["CHUNK_ID", "TITLE", "CHUNK_TEXT"],
    "limit": 3}'
) AS RESPONSE;

-- B. Same search as a table: FLATTEN the results array. INDEX + 1 is the rank.
SELECT r.INDEX + 1                    AS RNK,
       r.VALUE:CHUNK_ID::STRING       AS CHUNK_ID,
       r.VALUE:DEPARTMENT::STRING     AS DEPARTMENT,
       r.VALUE:CHUNK_TEXT::STRING     AS CHUNK_TEXT
FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'DS90.DAY38.CLINICAL_DOC_SEARCH',
  '{"query": "Can I quit my blood thinners after the stent if I feel fine?",
    "columns": ["CHUNK_ID", "DEPARTMENT", "CHUNK_TEXT"], "limit": 5}'
))['results'])) r;
-- Gold answer is DOC003-C1. The question never says "aspirin", "clopidogrel" or "antiplatelet":
-- this is where the vector half of hybrid search should beat the keyword-only local baseline.

-- C. Filters work on ATTRIBUTES only. "What is the lifting limit after surgery?" is ambiguous:
--    the stent sheet (Cardiology) and the hip sheet (Orthopedics) both have a lifting limit.
SELECT r.INDEX + 1 AS RNK, r.VALUE:CHUNK_ID::STRING AS CHUNK_ID, r.VALUE:DEPARTMENT::STRING AS DEPARTMENT,
       r.VALUE:CHUNK_TEXT::STRING AS CHUNK_TEXT
FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'DS90.DAY38.CLINICAL_DOC_SEARCH',
  '{"query": "What is the lifting limit after surgery?",
    "columns": ["CHUNK_ID", "DEPARTMENT", "CHUNK_TEXT"],
    "filter": {"@eq": {"DEPARTMENT": "Orthopedics"}},
    "limit": 3}'
))['results'])) r;

-- D. Combine operators: current policies only. The 2025 fall policy says Morse 45; the 2026 one says 50.
--    @gte / @lte work on numbers and dates ("YYYY-MM-DD"); @and / @or / @not combine them.
SELECT r.INDEX + 1 AS RNK, r.VALUE:CHUNK_ID::STRING AS CHUNK_ID, r.VALUE:TITLE::STRING AS TITLE,
       r.VALUE:EFFECTIVE_DATE::STRING AS EFFECTIVE_DATE, r.VALUE:CHUNK_TEXT::STRING AS CHUNK_TEXT
FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
  'DS90.DAY38.CLINICAL_DOC_SEARCH',
  '{"query": "What Morse Fall Scale score means high fall risk?",
    "columns": ["CHUNK_ID", "TITLE", "EFFECTIVE_DATE", "CHUNK_TEXT"],
    "filter": {"@and": [ {"@eq": {"DOC_TYPE": "policy"}},
                         {"@gte": {"EFFECTIVE_DATE": "2026-01-01"}} ]},
    "limit": 3}'
))['results'])) r;
-- Run D again without the filter. Does the superseded 2025 chunk (DOC019-C1) show up near the top?

-- E. CORTEX_SEARCH_RETRIEVE: a table function that takes a COLUMN as the query, one search per row.
--    Output = the source columns plus METADATA$RESULT_DETAIL (JSON with component scores).
SELECT q.QID, q.QUESTION, cs.CHUNK_ID, q.GOLD_CHUNK_ID,
       PARSE_JSON(cs.METADATA$RESULT_DETAIL):scores AS SCORES
FROM EVAL_QUESTIONS q,
     LATERAL CORTEX_SEARCH_RETRIEVE(
       service_name => 'DS90.DAY38.CLINICAL_DOC_SEARCH',
       query => q.QUESTION,
       limit => 2) cs
WHERE q.QID <= 5
ORDER BY q.QID;
```

### 5. RAG: top chunks into COMPLETE with citations, then check grounding (SQL)
```sql
-- Day 38: RAG = Retrieve, then Augment the prompt, then Generate.
-- 1) Cortex Search finds the top 3 chunks. 2) We paste them into the prompt with their CHUNK_IDs.
-- 3) COMPLETE answers ONLY from that context and cites the ids. Costs: COMPLETE tokens (prompt + answer)
-- plus warehouse time. The search itself is covered by the service's serving cost.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. One question, end to end. SEARCH_PREVIEW needs a literal, so the question text appears twice.
SET Q = 'My blood sugar dropped to 62, how many grams of sugar should I eat and when do I check again?';
WITH hits AS (
  SELECT r.INDEX + 1 AS RNK, r.VALUE:CHUNK_ID::STRING AS CHUNK_ID, r.VALUE:CHUNK_TEXT::STRING AS CHUNK_TEXT
  FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'DS90.DAY38.CLINICAL_DOC_SEARCH',
    '{"query": "My blood sugar dropped to 62, how many grams of sugar should I eat and when do I check again?",
      "columns": ["CHUNK_ID", "CHUNK_TEXT"], "limit": 3}'
  ))['results'])) r
),
ctx AS (
  SELECT LISTAGG('[' || CHUNK_ID || '] ' || CHUNK_TEXT, '\n') WITHIN GROUP (ORDER BY RNK) AS CONTEXT FROM hits
)
SELECT CONTEXT,
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
         'You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. '
         || 'If the context does not contain the answer, reply exactly: Not found in the documents.\n\n'
         || 'Context:\n' || CONTEXT || '\n\nQuestion: ' || $Q || '\nAnswer:') AS ANSWER
FROM ctx;
-- Expected: 15 grams, recheck in 15 minutes, citing [DOC016-C1]. Newer name: AI_COMPLETE('llama3.1-8b', prompt).

-- B. A small batch, saved ONCE into a table (pay for tokens once, then query the table).
--    5 answerable questions + the 3 unanswerable ones, to test that the model refuses instead of guessing.
CREATE OR REPLACE TABLE RAG_ANSWERS AS
WITH qset AS (
  SELECT * FROM EVAL_QUESTIONS WHERE QID IN (5, 21, 29, 33, 47) OR Q_TYPE = 'unanswerable'
),
hits AS (
  SELECT q.QID, cs.CHUNK_ID, cs.CHUNK_TEXT
  FROM qset q,
       LATERAL CORTEX_SEARCH_RETRIEVE(
         service_name => 'DS90.DAY38.CLINICAL_DOC_SEARCH',
         query => q.QUESTION,
         filter => COALESCE(TRY_PARSE_JSON(q.FILTER_JSON), PARSE_JSON('{}')),
         limit => 3) cs
),
ctx AS (
  SELECT QID, LISTAGG('[' || CHUNK_ID || '] ' || CHUNK_TEXT, '\n') WITHIN GROUP (ORDER BY CHUNK_ID) AS CONTEXT,
         ARRAY_AGG(CHUNK_ID) AS RETRIEVED_IDS
  FROM hits GROUP BY QID
)
SELECT q.QID, q.Q_TYPE, q.QUESTION, q.GOLD_CHUNK_ID, c.RETRIEVED_IDS, c.CONTEXT,
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
         'You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. '
         || 'If the context does not contain the answer, reply exactly: Not found in the documents.\n\n'
         || 'Context:\n' || c.CONTEXT || '\n\nQuestion: ' || q.QUESTION || '\nAnswer:') AS ANSWER
FROM qset q JOIN ctx c USING (QID);

-- C. Check the answers without calling the model again.
--    CITED_IDS: ids the model wrote. GOLD_RETRIEVED: did search find the right chunk?
--    CITED_GOLD: did the answer cite it? REFUSED: did it say Not found?
SELECT QID, Q_TYPE, QUESTION, ANSWER,
       REGEXP_SUBSTR_ALL(ANSWER, 'DOC[0-9]{3}-C[0-9]+')                         AS CITED_IDS,
       ARRAY_CONTAINS(GOLD_CHUNK_ID::VARIANT, RETRIEVED_IDS)                    AS GOLD_RETRIEVED,
       ARRAY_CONTAINS(GOLD_CHUNK_ID::VARIANT, REGEXP_SUBSTR_ALL(ANSWER, 'DOC[0-9]{3}-C[0-9]+')) AS CITED_GOLD,
       ANSWER ILIKE '%not found in the documents%'                              AS REFUSED
FROM RAG_ANSWERS ORDER BY QID;
-- Grounding rule of thumb: an answer that cites an id NOT in RETRIEVED_IDS, or answers an unanswerable
-- question, is a hallucination. Count them and write the count in your README.
```

### 6. Retrieval eval: hit rate@k and MRR on the labeled questions (SQL)
```sql
-- Day 38: a small retrieval evaluation on the 47 labeled questions (3 unanswerable ones are left out).
-- hit rate@k = share of questions whose gold chunk is in the top k results.
-- MRR (mean reciprocal rank) = average of 1/rank of the gold chunk (0 if not in the top 10).
-- Local keyword baselines (real run, lesson box): BM25 hit@1 0.723, hit@3 0.894, MRR 0.801;
-- TF-IDF words hit@3 0.894, MRR 0.794; char n-gram TF-IDF hit@3 0.915, MRR 0.885.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. Quick hit@3 with one statement (order inside the top 3 does not matter for hit@3).
SELECT q.Q_TYPE, COUNT(DISTINCT q.QID) AS N,
       COUNT(DISTINCT IFF(cs.CHUNK_ID = q.GOLD_CHUNK_ID, q.QID, NULL)) / COUNT(DISTINCT q.QID) AS HIT_AT_3
FROM EVAL_QUESTIONS q,
     LATERAL CORTEX_SEARCH_RETRIEVE(
       service_name => 'DS90.DAY38.CLINICAL_DOC_SEARCH',
       query => q.QUESTION,
       filter => COALESCE(TRY_PARSE_JSON(q.FILTER_JSON), PARSE_JSON('{}')),
       limit => 3) cs
WHERE q.Q_TYPE <> 'unanswerable'
GROUP BY ROLLUP (q.Q_TYPE) ORDER BY 1;

-- B. Full ranks for MRR. SEARCH_PREVIEW keeps the result order (FLATTEN INDEX = rank - 1) but only takes
--    literals, so a Snowflake Scripting loop writes one literal query per question.
CREATE OR REPLACE VIEW EVAL_PARAMS AS
SELECT QID,
       REPLACE(TO_JSON(OBJECT_CONSTRUCT(
         'query', QUESTION,
         'columns', ARRAY_CONSTRUCT('CHUNK_ID'),
         'filter', TRY_PARSE_JSON(FILTER_JSON),     -- OBJECT_CONSTRUCT drops NULLs, so no filter = no key
         'limit', 10)), '''', '''''') AS PARAMS_SQL
FROM EVAL_QUESTIONS WHERE Q_TYPE <> 'unanswerable';

EXECUTE IMMEDIATE $$
DECLARE
  c1 CURSOR FOR SELECT QID, PARAMS_SQL FROM EVAL_PARAMS ORDER BY QID;
  stmt STRING;
BEGIN
  CREATE OR REPLACE TABLE SEARCH_RANKS (QID INT, RNK INT, CHUNK_ID STRING);
  FOR r IN c1 DO
    stmt := 'INSERT INTO SEARCH_RANKS SELECT ' || r.QID || ', f.INDEX + 1, f.VALUE:CHUNK_ID::STRING ' ||
            'FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(''DS90.DAY38.CLINICAL_DOC_SEARCH'', ''' ||
            r.PARAMS_SQL || '''))[''results''])) f';
    EXECUTE IMMEDIATE :stmt;
  END FOR;
  RETURN 'done: one search per labeled question';
END;
$$;

-- C. Score it, overall and by question type.
CREATE OR REPLACE VIEW EVAL_GOLD_RANK AS
SELECT q.QID, q.Q_TYPE, q.QUESTION, q.GOLD_CHUNK_ID, MIN(s.RNK) AS GOLD_RANK
FROM EVAL_QUESTIONS q
LEFT JOIN SEARCH_RANKS s ON s.QID = q.QID AND s.CHUNK_ID = q.GOLD_CHUNK_ID
WHERE q.Q_TYPE <> 'unanswerable'
GROUP BY 1, 2, 3, 4;

SELECT COALESCE(Q_TYPE, 'ALL')                       AS Q_TYPE,
       COUNT(*)                                      AS N,
       ROUND(AVG(IFF(GOLD_RANK <= 1, 1, 0)), 3)      AS HIT_AT_1,
       ROUND(AVG(IFF(GOLD_RANK <= 3, 1, 0)), 3)      AS HIT_AT_3,
       ROUND(AVG(IFF(GOLD_RANK <= 5, 1, 0)), 3)      AS HIT_AT_5,
       ROUND(AVG(COALESCE(1 / GOLD_RANK, 0)), 3)     AS MRR
FROM EVAL_GOLD_RANK
GROUP BY ROLLUP (Q_TYPE) ORDER BY Q_TYPE;

-- D. Side by side with the local keyword baselines (numbers from local_baseline.py on the lesson box).
SELECT * FROM VALUES
  ('BM25 (local)',               0.723, 0.894, 0.801),
  ('TF-IDF words (local)',       0.702, 0.894, 0.794),
  ('TF-IDF char 3-5 (local)',    0.851, 0.915, 0.885)
  AS t(METHOD, HIT_AT_1, HIT_AT_3, MRR);

-- E. Misses: read every question where the gold chunk is not in the top 3. What came back instead?
SELECT g.QID, g.Q_TYPE, g.QUESTION, g.GOLD_CHUNK_ID, g.GOLD_RANK, s.CHUNK_ID AS TOP_1
FROM EVAL_GOLD_RANK g LEFT JOIN SEARCH_RANKS s ON s.QID = g.QID AND s.RNK = 1
WHERE g.GOLD_RANK IS NULL OR g.GOLD_RANK > 3
ORDER BY g.QID;
```

### 7. Optional: the Python API with snowflake.core (Python)
```python
# Day 38 (Python path): query the Cortex Search service with the snowflake.core Python API, score the
# labeled questions (hit rate@k, MRR), and ask one grounded RAG question with COMPLETE.
# Search and COMPLETE run inside Snowflake. Python only sends requests and reads results back.
# Install once in your conda env:  pip install snowflake snowflake-snowpark-python   (snowflake = snowflake.core)
# Connection: ~/.snowflake/connections.toml [ds90] (Day 36) or SNOWFLAKE_* env vars. Never put a password here.
#
#   python day38_search.py --dry-run                 # no account: check files, print the plan
#   python day38_search.py                           # trial: evaluate the 47 labeled questions (search only)
#   python day38_search.py --ask "How long can an opened insulin pen stay out of the fridge?"
#   python day38_search.py --rag-test                # trial: 3 answerable + 3 unanswerable questions through COMPLETE
import os, sys, csv, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--ask", default=None)
ap.add_argument("--rag-test", action="store_true")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--model", default="llama3.1-8b")   # native in AWS US East 1
args = ap.parse_args()

DB, SCHEMA, SERVICE = "DS90", "DAY38", "CLINICAL_DOC_SEARCH"
chunks = list(csv.DictReader(open("doc_chunks.csv")))
qs_all = list(csv.DictReader(open("eval_questions.csv")))
qs = [q for q in qs_all if q["Q_TYPE"] != "unanswerable"]
PROMPT = ("You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. "
          "If the context does not contain the answer, reply exactly: Not found in the documents.\n\n"
          "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

print(f"Plan: {len(chunks)} chunks in the index, {len(qs)} labeled questions "
      f"({sum(1 for q in qs if q['FILTER_JSON'])} with a filter), {len(qs_all) - len(qs)} unanswerable for the RAG test.")
print(f"Embedding tokens to build the index: about {sum(len(c['CHUNK_TEXT']) for c in chunks) // 4:,} (chars / 4).")
print(f"One grounded prompt with k={args.k}: about {(len(PROMPT) + args.k * 160 + 90) // 4} input tokens for COMPLETE.")
if args.dry_run:
    sys.exit(0)

from snowflake.snowpark import Session
from snowflake.core import Root

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
                         "To check the files offline: python day38_search.py --dry-run") from e

session = connect()
session.sql("USE WAREHOUSE DS90_XS").collect()
svc = Root(session).databases[DB].schemas[SCHEMA].cortex_search_services[SERVICE]

def search(question, flt=None, limit=10, columns=("CHUNK_ID",)):
    kw = dict(query=question, columns=list(columns), limit=limit)
    if flt:
        kw["filter"] = flt                     # e.g. {"@eq": {"DEPARTMENT": "Orthopedics"}}
    return svc.search(**kw).results            # list of dicts, best match first

def rag(question, flt=None):
    hits = search(question, flt, limit=args.k, columns=("CHUNK_ID", "CHUNK_TEXT"))
    ctx = "\n".join(f"[{h['CHUNK_ID']}] {h['CHUNK_TEXT']}" for h in hits)
    ans = session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)", params=[args.model, PROMPT.format(ctx=ctx, q=question)]).collect()[0][0]
    return [h["CHUNK_ID"] for h in hits], ans

if args.ask:
    ids, ans = rag(args.ask)
    print("retrieved:", ids, "\nanswer:", ans)
elif args.rag_test:
    pick = [q for q in qs if q["QID"] in ("5", "29", "47")] + [q for q in qs_all if q["Q_TYPE"] == "unanswerable"]
    for q in pick:
        ids, ans = rag(q["QUESTION"], json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None)
        print(f"\nQ{q['QID']} [{q['Q_TYPE']}] {q['QUESTION']}\n  gold {q['GOLD_CHUNK_ID'] or '-'} | retrieved {ids}\n  {ans.strip()}")
else:
    ranks = []
    for q in qs:
        got = [r["CHUNK_ID"] for r in search(q["QUESTION"], json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None)]
        ranks.append((q, got.index(q["GOLD_CHUNK_ID"]) + 1 if q["GOLD_CHUNK_ID"] in got else None))
    def score(sub):
        n = len(sub)
        return {f"hit@{k}": round(sum(1 for _, r in sub if r and r <= k) / n, 3) for k in (1, 3, 5)} | \
               {"mrr": round(sum(1 / r for _, r in sub if r) / n, 3)}
    print("Cortex Search, all  ", score(ranks))
    for t in ("keyword", "paraphrase"):
        print(f"Cortex Search, {t:10s}", score([x for x in ranks if x[0]["Q_TYPE"] == t]))
    print("Local BM25 baseline (lesson box): hit@1 0.723, hit@3 0.894, MRR 0.801 | paraphrase hit@3 0.792")
    for q, r in ranks:
        if not r or r > 3:
            print(f"  miss Q{q['QID']} [{q['Q_TYPE']}] rank {r}: {q['QUESTION']}")

session.close()
print("Done. Run 08_cost_cleanup.sql to suspend or drop the service when you finish.")
```

### 8. Cost check, then suspend and drop the service (SQL)
```sql
-- Day 38: what did today cost, then stop the meters.
-- ACCOUNT_USAGE views are not real time (often hours behind). If they are empty, check again tomorrow.
-- Rates (credits per GB-month for serving, per million tokens for embeddings and COMPLETE):
--   Snowflake Service Consumption Table, https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. The service right now: state, row count, size. Look for serving_state and indexing_state.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;

-- B. Daily embedding-token and serving credits per service
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY ORDER BY 1 DESC LIMIT 20;

-- C. Hourly serving credits per service
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY ORDER BY 1 DESC LIMIT 48;

-- D. COMPLETE tokens from 06_rag_complete.sql
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY ORDER BY 1 DESC LIMIT 20;

-- E. All AI services credits per day, and the warehouse meter
SELECT USAGE_DATE, SERVICE_TYPE, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES' ORDER BY USAGE_DATE DESC LIMIT 14;

SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS' ORDER BY START_TIME DESC LIMIT 14;

-- F. Stop the meters. A RESUMED service bills serving credits even with zero queries.
ALTER CORTEX SEARCH SERVICE IF EXISTS DS90.DAY38.CLINICAL_DOC_SEARCH SUSPEND;   -- suspends indexing AND serving
-- (Resume later with: ALTER CORTEX SEARCH SERVICE DS90.DAY38.CLINICAL_DOC_SEARCH RESUME;)

-- G. Recommended once your README numbers are written down: drop it. Rebuilding costs only about
--    4,000 embedding tokens plus a minute of X-Small time. Day 39 (Cortex Analyst) does not use it.
DROP CORTEX SEARCH SERVICE IF EXISTS DS90.DAY38.CLINICAL_DOC_SEARCH;

ALTER WAREHOUSE DS90_XS SUSPEND;   -- if it says "cannot be suspended", it already is: fine
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;   -- expect nothing left running from Day 38
```

### 9. Local keyword baseline: TF-IDF, BM25, hit@k, MRR (Python, runs offline)
```python
# Day 38: a LOCAL retrieval baseline you can run anywhere (no Snowflake, no tokens, no cost).
# Keyword-only search (TF-IDF and BM25) over the same simulated chunks, scored on the labeled questions
# with hit rate@k and MRR. Then: attribute filters, chunks vs whole documents, a sample grounded prompt,
# and a size/cost estimate for the Cortex Search service. Compare with 07_evaluate.sql on your trial.
import csv, json, math, re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

chunks = list(csv.DictReader(open("doc_chunks.csv")))
docs = list(csv.DictReader(open("clinical_docs.csv")))
allq = list(csv.DictReader(open("eval_questions.csv")))
qs = [q for q in allq if q["Q_TYPE"] != "unanswerable"]
K = [1, 3, 5, 10]

def tok(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in ENGLISH_STOP_WORDS]

class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.d = [Counter(tok(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.d]); self.avg = self.len.mean()
        df = Counter(w for c in self.d for w in c)
        n = len(texts); self.idf = {w: math.log(1 + (n - f + .5) / (f + .5)) for w, f in df.items()}
        self.k1, self.b = k1, b
    def scores(self, q):
        s = np.zeros(len(self.d))
        for w in set(tok(q)):
            if w not in self.idf: continue
            tf = np.array([c.get(w, 0) for c in self.d])
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s

class Tfidf:
    def __init__(self, texts, **kw):
        self.v = TfidfVectorizer(sublinear_tf=True, **kw); self.m = self.v.fit_transform(texts)
    def scores(self, q):
        return (self.m @ self.v.transform([q]).T).toarray().ravel()   # rows are L2-normalised: cosine

def passes(row, flt):
    """Apply a Cortex Search style filter (@eq, @gte, @lte, @and, @or, @not) to one row."""
    if not flt: return True
    (op, arg), = flt.items()
    if op == "@and": return all(passes(row, f) for f in arg)
    if op == "@or": return any(passes(row, f) for f in arg)
    if op == "@not": return not passes(row, arg)
    (colname, val), = arg.items()
    return {"@eq": row[colname] == val, "@gte": row[colname] >= val, "@lte": row[colname] <= val}[op]

def rank(model, rows, q, flt=None, key="CHUNK_ID", n=10):
    s = model.scores(q)
    order = [i for i in np.argsort(-s, kind="stable") if passes(rows[i], flt)]
    return [rows[i][key] for i in order[:n]]

def evaluate(model, rows, use_filter, key="CHUNK_ID", gold="GOLD_CHUNK_ID", subset=None):
    rr, hits = [], {k: [] for k in K}
    for q in (subset or qs):
        flt = json.loads(q["FILTER_JSON"]) if (use_filter and q["FILTER_JSON"]) else None
        got = rank(model, rows, q["QUESTION"], flt, key)
        r = got.index(q[gold]) + 1 if q[gold] in got else None
        rr.append(1 / r if r else 0.0)
        for k in K: hits[k].append(1 if r and r <= k else 0)
    return {**{f"hit@{k}": round(float(np.mean(hits[k])), 3) for k in K}, "mrr": round(float(np.mean(rr)), 3)}

texts = [c["CHUNK_TEXT"] for c in chunks]
models = {"TF-IDF (words)": Tfidf(texts, stop_words="english", ngram_range=(1, 2)),
          "BM25": BM25(texts),
          "TF-IDF (char 3-5)": Tfidf(texts, analyzer="char_wb", ngram_range=(3, 5))}
print(f"chunks: {len(chunks)} from {len(docs)} simulated documents | labeled questions: {len(qs)} "
      f"({sum(q['Q_TYPE']=='keyword' for q in qs)} keyword, {sum(q['Q_TYPE']=='paraphrase' for q in qs)} paraphrase, "
      f"{sum(bool(q['FILTER_JSON']) for q in qs)} with a filter) + {len(allq)-len(qs)} unanswerable for the RAG test")

card = {"n_chunks": len(chunks), "n_docs": len(docs), "n_q": len(qs), "methods": {}, "by_type": {}, "filter": {}}
print("\n== chunk retrieval, filters applied where the question has one")
for name, m in models.items():
    r = evaluate(m, chunks, True); card["methods"][name] = r
    print(f"  {name:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))

print("\n== by question type (BM25 and TF-IDF words, hit@3 / MRR)")
for name in ["TF-IDF (words)", "BM25"]:
    card["by_type"][name] = {}
    for t in ["keyword", "paraphrase"]:
        r = evaluate(models[name], chunks, True, subset=[q for q in qs if q["Q_TYPE"] == t])
        card["by_type"][name][t] = r
        print(f"  {name:15s} {t:10s} hit@3 {r['hit@3']:.3f}  MRR {r['mrr']:.3f}")

fq = [q for q in qs if q["FILTER_JSON"]]
print(f"\n== the {len(fq)} filtered questions, BM25 with and without the attribute filter")
for use in [False, True]:
    r = evaluate(models["BM25"], chunks, use, subset=fq); card["filter"]["with" if use else "without"] = r
    print(f"  filter {'ON ' if use else 'OFF'}  hit@1 {r['hit@1']:.3f}  hit@3 {r['hit@3']:.3f}  MRR {r['mrr']:.3f}")
for q in fq:
    a = rank(models["BM25"], chunks, q["QUESTION"], None)[:2]
    b = rank(models["BM25"], chunks, q["QUESTION"], json.loads(q["FILTER_JSON"]))[:2]
    print(f"  Q{q['QID']} gold {q['GOLD_CHUNK_ID']} | no filter top2 {a} | {q['FILTER_JSON']} top2 {b}")

print("\n== chunks vs whole documents (BM25, find the right DOCUMENT, k = 3)")
dtexts = [f"{d['TITLE']}. " + d["DOC_TEXT"] for d in docs]
dm = BM25(dtexts)
whole = evaluate(dm, docs, True, key="DOC_ID", gold="GOLD_DOC_ID")
def chunk_doc_hit(k):
    h = []
    for q in qs:
        flt = json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None
        got = rank(models["BM25"], chunks, q["QUESTION"], flt, "DOC_ID")[:k]
        h.append(q["GOLD_DOC_ID"] in got)
    return round(float(np.mean(h)), 3)
avg_chunk = np.mean([len(t) for t in texts]); avg_doc = np.mean([len(t) for t in dtexts])
card["chunk_vs_doc"] = {"doc_hit3": whole["hit@3"], "chunk_doc_hit3": chunk_doc_hit(3),
                        "ctx_chars_chunks_k3": int(3 * avg_chunk), "ctx_chars_docs_k3": int(3 * avg_doc)}
print(f"  whole docs : right doc in top 3 {whole['hit@3']:.3f}, context sent to the LLM at k=3 about {int(3*avg_doc)} chars")
print(f"  chunks     : right doc in top 3 {card['chunk_vs_doc']['chunk_doc_hit3']:.3f}, context at k=3 about {int(3*avg_chunk)} chars "
      f"(and the exact passage, not the whole document)")

best = "BM25"
print("\n== misses: gold chunk not in BM25 top 3")
miss = []
for q in qs:
    flt = json.loads(q["FILTER_JSON"]) if q["FILTER_JSON"] else None
    got = rank(models[best], chunks, q["QUESTION"], flt)[:3]
    if q["GOLD_CHUNK_ID"] not in got:
        miss.append(q["QID"]); print(f"  Q{q['QID']:>2} [{q['Q_TYPE']}] {q['QUESTION'][:70]} -> got {got[0]}")
card["misses_k3"] = miss

# a grounded prompt, exactly what 06_rag_complete.sql builds (not sent to any model here)
qx = next(q for q in qs if q["QID"] == "29")
top = rank(models[best], chunks, qx["QUESTION"], None)[:3]
byid = {c["CHUNK_ID"]: c for c in chunks}
ctx = "\n".join(f"[{i}] {byid[i]['CHUNK_TEXT']}" for i in top)   # in rank order
prompt = ("You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. "
          "If the context does not contain the answer, reply exactly: Not found in the documents.\n\n"
          f"Context:\n{ctx}\n\nQuestion: {qx['QUESTION']}\nAnswer:")
print(f"\n== sample grounded prompt for Q{qx['QID']} (BM25 top 3; gold {qx['GOLD_CHUNK_ID']} "
      f"{'found' if qx['GOLD_CHUNK_ID'] in top else 'NOT found'})\n{prompt}")
card["prompt_example"] = prompt; card["prompt_tokens_est"] = len(prompt) // 4

# size and cost estimate for the Cortex Search service (formulas from the Snowflake cost docs)
emb_tokens = sum(len(t) for t in texts) // 4
row_bytes = np.mean([sum(len(v) for v in c.values()) for c in chunks])
gb = len(chunks) * (768 * 4 + row_bytes) / 1e9
card.update(emb_tokens=int(emb_tokens), index_gb=gb, row_bytes=int(row_bytes),
            avg_chunk_chars=int(avg_chunk), chunk_chars=[len(t) for t in texts],
            dept_counts=dict(Counter(c["DEPARTMENT"] for c in chunks)),
            type_counts=dict(Counter(c["DOC_TYPE"] for c in chunks)))
print("\n== size and cost estimate (rates NOT copied: see the Snowflake Service Consumption Table)")
print(f"  embedding tokens to build the index: about {emb_tokens:,} (chars / 4) = {emb_tokens/1e6:.4f} million")
print(f"  indexed data: {len(chunks)} rows x (768 dims x 4 bytes + about {int(row_bytes)} bytes of columns) = {gb*1e3:.3f} MB = {gb:.6f} GB")
print(f"  serving credits per month = {gb:.6f} GB x (credits per GB-month). With the docs example rate of 6.3: {gb*6.3:.4f} credits/month")
print("  warehouse: X-Small uses 1 credit per hour while running, billed per second, 60 second minimum per resume")
print(f"  grounded prompt above: about {len(prompt)//4} input tokens per question for COMPLETE")
json.dump(card, open("baseline_card.json", "w"), indent=1)
print("wrote baseline_card.json")
```

### 10. Push to GitHub
```bash
cd ~/ds-genai-90-days
mkdir -p day38        # copy the exercise files (SQL, .py, .csv, README.md) into day38/
cd day38 && python gen_docs.py && python local_baseline.py && cd ..
git status            # check: no connections.toml, no .env, no passwords
git add day38
git commit -m "Day 38: Cortex Search over simulated clinical documents, first look at RAG"
git push
```

## Local run output (lesson box)
```
== python3 gen_docs.py
wrote 33 simulated documents -> 102 chunks (avg 158 chars, max 264)
doc types: {'discharge_instructions': 9, 'care_protocol': 8, 'patient_education': 8, 'policy': 8}
departments: 16
questions: {'keyword': 23, 'paraphrase': 24, 'unanswerable': 3} | with a filter: 4

== python3 local_baseline.py
chunks: 102 from 33 simulated documents | labeled questions: 47 (23 keyword, 24 paraphrase, 4 with a filter) + 3 unanswerable for the RAG test

== chunk retrieval, filters applied where the question has one
  TF-IDF (words)     hit@1 0.702  hit@3 0.894  hit@5 0.894  hit@10 0.915  mrr 0.794
  BM25               hit@1 0.723  hit@3 0.894  hit@5 0.894  hit@10 0.915  mrr 0.801
  TF-IDF (char 3-5)  hit@1 0.851  hit@3 0.915  hit@5 0.936  hit@10 0.936  mrr 0.885

== by question type (BM25 and TF-IDF words, hit@3 / MRR)
  TF-IDF (words)  keyword    hit@3 1.000  MRR 0.913
  TF-IDF (words)  paraphrase hit@3 0.792  MRR 0.681
  BM25            keyword    hit@3 1.000  MRR 0.906
  BM25            paraphrase hit@3 0.792  MRR 0.700

== the 4 filtered questions, BM25 with and without the attribute filter
  filter OFF  hit@1 0.750  hit@3 0.750  MRR 0.750
  filter ON   hit@1 0.750  hit@3 1.000  MRR 0.875
  Q26 gold DOC013-C3 | no filter top2 ['DOC001-C2', 'DOC027-C2'] | {"@eq": {"DEPARTMENT": "Orthopedics"}} top2 ['DOC013-C2', 'DOC013-C3']
  Q33 gold DOC020-C1 | no filter top2 ['DOC020-C1', 'DOC019-C1'] | {"@gte": {"EFFECTIVE_DATE": "2026-01-01"}} top2 ['DOC020-C1', 'DOC021-C1']
  Q42 gold DOC027-C1 | no filter top2 ['DOC027-C1', 'DOC029-C1'] | {"@eq": {"DOC_TYPE": "patient_education"}} top2 ['DOC027-C1', 'DOC029-C1']
  Q45 gold DOC030-C1 | no filter top2 ['DOC030-C1', 'DOC001-C1'] | {"@eq": {"DEPARTMENT": "General Surgery"}} top2 ['DOC030-C1', 'DOC030-C2']

== chunks vs whole documents (BM25, find the right DOCUMENT, k = 3)
  whole docs : right doc in top 3 0.894, context sent to the LLM at k=3 about 1247 chars
  chunks     : right doc in top 3 0.894, context at k=3 about 476 chars (and the exact passage, not the whole document)

== misses: gold chunk not in BM25 top 3
  Q 9 [paraphrase] Do I have to stop eating spinach and salad greens on my anticoagulant? -> got DOC022-C1
  Q11 [paraphrase] When is the final dose of the injectable bridge medicine given before  -> got DOC013-C4
  Q12 [paraphrase] How long do I take the steroid pills after my lung flare-up? -> got DOC008-C2
  Q46 [paraphrase] How much bleeding after giving birth is too much? -> got DOC010-C2
  Q47 [paraphrase] Can a relative translate for the doctor if they speak both languages? -> got DOC006-C1

== sample grounded prompt for Q29 (BM25 top 3; gold DOC016-C1 found)
You answer questions using ONLY the context. Cite the chunk ids you used in square brackets. If the context does not contain the answer, reply exactly: Not found in the documents.

Context:
[DOC016-C1] Diabetes Discharge Instructions. For low blood sugar under 70 mg/dL, use the rule of 15: take 15 grams of fast sugar such as 4 ounces of juice, then recheck in 15 minutes.
[DOC016-C2] Diabetes Discharge Instructions. Check your blood sugar before each meal and at bedtime. Write the numbers in your log and bring it to clinic.
[DOC016-C3] Diabetes Discharge Instructions. On sick days, keep taking your long-acting (basal) insulin even if you are not eating. Check your sugar every 4 hours.

Question: My blood sugar dropped to 62, how many grams of sugar should I eat and when do I check again?
Answer:

== size and cost estimate (rates NOT copied: see the Snowflake Service Consumption Table)
  embedding tokens to build the index: about 4,050 (chars / 4) = 0.0040 million
  indexed data: 102 rows x (768 dims x 4 bytes + about 247 bytes of columns) = 0.339 MB = 0.000339 GB
  serving credits per month = 0.000339 GB x (credits per GB-month). With the docs example rate of 6.3: 0.0021 credits/month
  warehouse: X-Small uses 1 credit per hour while running, billed per second, 60 second minimum per resume
  grounded prompt above: about 201 input tokens per question for COMPLETE
wrote baseline_card.json

== python3 day38_search.py --dry-run
Plan: 102 chunks in the index, 47 labeled questions (4 with a filter), 3 unanswerable for the RAG test.
Embedding tokens to build the index: about 4,050 (chars / 4).
One grounded prompt with k=3: about 197 input tokens for COMPLETE.

== python3 day38_search.py   (no Snowflake packages or connection on the lesson box)
    from snowflake.snowpark import Session
ModuleNotFoundError: No module named 'snowflake'
exit code 1

Box: ~/.snowflake exists: no; SNOWFLAKE_* env vars: 0
```

## What ran, and what did not (note)
The lesson computer has no Snowflake connection, and none was created. The bot did not run any SQL against Snowflake, did not create a search service, and made no SEARCH_PREVIEW or COMPLETE calls. It ran gen_docs.py, local_baseline.py and day38_search.py --dry-run (the full script stops without the Snowflake packages and a connection). Cortex Search hit rates, RAG answers and credits come from your trial. Write them in your README.

## Tomorrow (Day 39)
Cortex Analyst: ask questions in plain English and get SQL back, using a semantic model you write over simulated readmission tables.

## Printables
- Cheat sheet: `day38_v1_cheat_sheet.png`
- Infographic: `day38_v2_infographic.png`
- Code worksheet: `day38_v3_worksheet.png`
- Practice quiz: `day38_v4_quiz.png`
- PDF: `day38_lesson.pdf`

## Podcast script
(Audio: `day38_podcast.mp3`)

Welcome to day thirty eight of the Data Analyst to Data Scientist plan. Yesterday you called language models from plain SQL on one discharge note at a time. Today we take the next step, and it is a big idea in Gen AI work. It is called retrieval augmented generation, or RAG for short, and the Snowflake tool for it is Cortex Search.

Here is the problem RAG solves. A language model has never read your hospital's care protocols or discharge sheets. If you ask it how long an opened insulin pen can stay out of the fridge, it may give you a confident answer that does not match your own policy. RAG fixes that in three moves. First, retrieve. Search your own documents for the few passages that answer the question. Second, augment. Paste those passages into the prompt, each with an id. Third, generate. Tell the model to answer only from that context, to cite the ids, and to say not found when the answer is not there.

For practice I wrote thirty three simulated clinical documents. They are invented, not medical advice, and there is no patient data. Each has attributes like department, document type, and effective date.

Step one is chunking. The default embedding model reads at most five hundred twelve tokens per row and cuts off the rest. So you split long documents into smaller pieces, called chunks. I split by paragraph, which gave one hundred two chunks of about one hundred sixty characters each. I also put the document title at the front of every chunk, so a chunk still says where it came from.

Step two is one statement. Create cortex search service. You point it at the text column with the word on. You list attributes, which are the columns you can filter on later. You give it a warehouse, an extra small one that suspends after sixty seconds. You set a target lag, which is how stale the index is allowed to get. And you can name the embedding model. Snowflake embeds every chunk for you and builds a hybrid index, which means it matches on meaning and on keywords.

Then you query it. In a worksheet, search preview takes a small JSON object with your question, the columns you want back, a limit, and an optional filter. Filters matter in healthcare. There are two fall prevention policies in the library, an old one and a current one, with different score cut offs. A date filter keeps the old one out. In Python, the snowflake core package gives you a search method with the same pieces.

Step three is the grounded answer. You take the top three chunks, glue them into a prompt with their ids, and send it to complete. Then you check the work without paying for the model again. Did it cite the right chunk? Did it refuse the three questions the library cannot answer, like the cafeteria menu?

Now, measure before you trust. I labeled forty seven questions with the one chunk that answers each. Two metrics. Hit rate at k is the share of questions where the right chunk shows up in the top k. Mean reciprocal rank rewards finding it near the top. On the lesson computer I ran a free keyword-only baseline, with no embeddings. Here are the real numbers. BM25 put the right chunk first about seventy two percent of the time and in the top three about eighty nine percent of the time, with a mean reciprocal rank of zero point eight. But when the question used different words than the document, like blood thinners instead of aspirin and clopidogrel, top three dropped to about seventy nine percent. That gap is what the meaning-based half of Cortex Search is for. Your trial run will show whether it closes it. I did not run any of the Snowflake parts, so your numbers are the real ones.

Cost. The warehouse builds the index. The embedding step costs tokens once, only about four thousand here. Serving is billed per gigabyte per month while the service is running, even with zero queries, and the answers cost complete tokens. So when you finish, suspend the service, and once your numbers are written down, drop it.

Why this matters for jobs. Healthcare data science postings now ask for RAG, embeddings, vector search, and LLM evaluation. After today you can say you built a small retrieval pipeline on practice data and measured it with hit rate and reciprocal rank.

Tomorrow, on day thirty nine, we switch from documents to tables. Cortex Analyst turns a plain English question into SQL, using a semantic model you write over simulated readmission tables. See you tomorrow.
