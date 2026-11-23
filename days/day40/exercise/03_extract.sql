-- Day 40 step 3 (the Day 37 skill): read the notes with an LLM, but only where it changes a decision.
-- Plan: the model score (structured data) shortlists the top 60; AI_COMPLETE reads only those 60 notes
-- for two facts the score cannot see: lives alone, follow-up booked. Saved ONCE into a table (CTAS).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

-- A. Free size check first: AI_COUNT_TOKENS has no token charge (only a little warehouse time).
SELECT COUNT(*) AS NOTES,
       SUM(AI_COUNT_TOKENS('ai_complete', 'llama3.1-8b', NOTE_TEXT)) AS NOTE_TOKENS_ALL_300
FROM DISCHARGES;

-- B. The shortlist: top 60 by RISK_SCORE (ties broken by id so the list is stable).
CREATE OR REPLACE TABLE SHORTLIST AS
SELECT *, ROW_NUMBER() OVER (ORDER BY RISK_SCORE DESC, ENCOUNTER_ID) AS SCORE_RANK
FROM DISCHARGES
QUALIFY SCORE_RANK <= 60;

-- C. Try 3 rows first and read them. Structured output: a JSON schema instead of free text.
SELECT ENCOUNTER_ID, NOTE_TEXT,
       AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated hospital discharge note. lives_alone: true only if the patient has no one living '
                || 'with them. followup_booked: true only if a follow-up appointment has a confirmed date. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 60},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'lives_alone': {'type': 'boolean'}, 'followup_booked': {'type': 'boolean'}},
           'required': ['lives_alone', 'followup_booked']}}
       ) AS FIELDS
FROM SHORTLIST
WHERE SCORE_RANK <= 3
ORDER BY SCORE_RANK;

-- D. All 60, once. This is the main AI Functions spend of the day. Re-reading NOTE_FIELDS later is warehouse time only.
--    TO_VARIANT + the IFF below work whether the answer comes back as an object or as JSON text.
CREATE OR REPLACE TABLE NOTE_FIELDS AS
SELECT ENCOUNTER_ID, SCORE_RANK,
       TO_VARIANT(AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated hospital discharge note. lives_alone: true only if the patient has no one living '
                || 'with them. followup_booked: true only if a follow-up appointment has a confirmed date. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 60},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'lives_alone': {'type': 'boolean'}, 'followup_booked': {'type': 'boolean'}},
           'required': ['lives_alone', 'followup_booked']}}
       )) AS FIELDS
FROM SHORTLIST;

CREATE OR REPLACE VIEW NOTE_FIELDS_PARSED AS
SELECT ENCOUNTER_ID, SCORE_RANK,
       IFF(IS_OBJECT(FIELDS), FIELDS, TRY_PARSE_JSON(FIELDS::STRING)):lives_alone::BOOLEAN     AS LIVES_ALONE_AI,
       IFF(IS_OBJECT(FIELDS), FIELDS, TRY_PARSE_JSON(FIELDS::STRING)):followup_booked::BOOLEAN AS FOLLOWUP_BOOKED_AI
FROM NOTE_FIELDS;
SELECT * FROM NOTE_FIELDS_PARSED ORDER BY SCORE_RANK LIMIT 10;
SELECT COUNT(*) AS ROWS_, COUNT_IF(LIVES_ALONE_AI IS NULL OR FOLLOWUP_BOOKED_AI IS NULL) AS UNPARSED FROM NOTE_FIELDS_PARSED;  -- expect 60, 0
