-- Day 37: the same jobs with the newer AI_* function names.
-- Snowflake docs mark COMPLETE, CLASSIFY_TEXT and EXTRACT_ANSWER (SNOWFLAKE.CORTEX) as legacy pages kept for
-- backward compatibility, to be deprecated by the end of 2026. Use AI_* for new work.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- 0. Free check first: AI_COUNT_TOKENS has no token charge (only a little warehouse time).
SELECT NOTE_ID,
       LENGTH(NOTE_TEXT)                                        AS CHARS,
       AI_COUNT_TOKENS('ai_complete', 'llama3.1-8b', NOTE_TEXT)  AS COMPLETE_INPUT_TOKENS,
       AI_COUNT_TOKENS('ai_sentiment', PATIENT_FEEDBACK)         AS SENTIMENT_INPUT_TOKENS
FROM DISCHARGE_NOTES
ORDER BY NOTE_ID
LIMIT 5;

-- 1. AI_COMPLETE with a JSON schema (structured output) instead of free text.
SELECT NOTE_ID,
       AI_COMPLETE(
         model => 'llama3.1-8b',
         prompt => 'Read this simulated discharge note and fill the fields. Note: ' || NOTE_TEXT,
         model_parameters => {'temperature': 0, 'max_tokens': 120},
         response_format => {'type': 'json', 'schema': {'type': 'object', 'properties': {
             'followup_days': {'type': 'integer'},
             'followup_with': {'type': 'string'},
             'lives_alone':   {'type': 'boolean'}},
           'required': ['followup_days', 'followup_with', 'lives_alone']}}
       ) AS FIELDS
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 2. AI_CLASSIFY returns {"labels": [...]}; take the first label.
SELECT NOTE_ID,
       AI_CLASSIFY(NOTE_TEXT,
         ['low readmission risk', 'medium readmission risk', 'high readmission risk']):labels[0]::STRING AS RISK_LABEL
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 3. AI_SENTIMENT returns labels (positive, negative, neutral, mixed, unknown), overall and per aspect.
SELECT NOTE_ID,
       PATIENT_FEEDBACK,
       AI_SENTIMENT(PATIENT_FEEDBACK, ['staff', 'discharge process', 'medications']) AS SENTIMENT_OBJ
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 4. AI_EXTRACT pulls several named fields in one call.
SELECT NOTE_ID,
       AI_EXTRACT(text => NOTE_TEXT,
                  responseFormat => {'followup_when': 'In how many days or weeks should the patient follow up?',
                                     'followup_with': 'Which specialty or clinic should the patient follow up with?'}
       ):response AS EXTRACTED
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- 5. AI_SUMMARIZE_AGG: one summary across many rows (here: feedback by diagnosis).
SELECT DIAGNOSIS,
       COUNT(*)                            AS N,
       AI_SUMMARIZE_AGG(PATIENT_FEEDBACK)  AS FEEDBACK_THEMES
FROM DISCHARGE_NOTES
GROUP BY DIAGNOSIS
ORDER BY N DESC;
