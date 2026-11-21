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
