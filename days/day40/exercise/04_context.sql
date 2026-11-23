-- Day 40 step 4: context for the worklist from Day 39 (semantic view) and Day 38 (documents).
-- Neither part calls an LLM. If those objects are gone, use the fallback tables loaded in 02 (same numbers).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

-- A. Day 39: readmission rate by service line from the semantic view (warehouse time only, no Analyst message).
SELECT * FROM SEMANTIC_VIEW(
    DS90.DAY39.READMIT_ANALYST
    DIMENSIONS drg_groups.service_line
    METRICS encounters.readmission_rate, encounters.encounter_count
  )
  ORDER BY readmission_rate DESC;
-- Fallback (same numbers, computed from the Day 39 CSVs on the lesson box):
SELECT * FROM BASELINE_RATES ORDER BY DAY39_READMIT_RATE DESC;

-- Optional: ask Cortex Analyst the same thing in plain English (1 billed message), with the Day 39 script:
--   python day39_analyst.py --ask "What is the readmission rate by service line?"

-- B. Day 38: one patient-facing guidance document per service line.
-- Default path: a plain join to the Day 38 documents table, so no search service has to be running.
SELECT g.SERVICE_LINE, g.DOC_ID, g.TITLE,
       (SELECT COUNT(*) FROM DS90.DAY38.DOC_CHUNKS c WHERE c.DOC_ID = g.DOC_ID) AS CHUNKS_IN_DAY38
FROM GUIDANCE_MAP g ORDER BY 1;

-- Optional, only if CLINICAL_DOC_SEARCH still exists (it bills serving credits while resumed; Day 38 suggested dropping it):
-- SELECT r.INDEX + 1 AS RNK, r.VALUE:DOC_ID::STRING AS DOC_ID, r.VALUE:TITLE::STRING AS TITLE
-- FROM TABLE(FLATTEN(PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
--   'DS90.DAY38.CLINICAL_DOC_SEARCH',
--   '{"query": "what should a heart failure patient do at home after discharge", "columns": ["DOC_ID", "TITLE"], "limit": 3}'
-- ))['results'])) r;
