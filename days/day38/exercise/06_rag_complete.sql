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
