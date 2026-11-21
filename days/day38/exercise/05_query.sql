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
