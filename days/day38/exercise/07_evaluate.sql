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
