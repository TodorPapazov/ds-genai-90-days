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
