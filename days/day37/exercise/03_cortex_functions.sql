-- Day 37: SNOWFLAKE.CORTEX functions on SIMULATED discharge notes.
-- Cost habit: 1 row first, then 5 rows, then all 40 rows ONCE into a table (CTAS).
-- Re-reading NOTE_AI later costs warehouse time only, not new Cortex tokens.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- A. One row, every function, so you see each output type.
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.SUMMARIZE(NOTE_TEXT)                         AS SUMMARY,         -- string
       SNOWFLAKE.CORTEX.SENTIMENT(PATIENT_FEEDBACK)                  AS FEEDBACK_SCORE,  -- float -1 to 1
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(NOTE_TEXT,
           ['low readmission risk', 'medium readmission risk', 'high readmission risk']) AS RISK_OBJ,  -- {"label": ...}
       SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
           'In how many days should the patient follow up?')        AS FOLLOWUP_RAW,    -- answer + score
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
           'In one plain sentence for the patient, say who to see and when. Note: ' || NOTE_TEXT) AS PATIENT_REMINDER
FROM DISCHARGE_NOTES
WHERE NOTE_ID = 1;

-- B. CLASSIFY_TEXT with label descriptions and a task description (more input tokens per row, often better labels).
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
         NOTE_TEXT,
         [ {'label': 'low readmission risk',    'description': 'younger or stable, no prior admissions, lives with others, short stay'},
           {'label': 'medium readmission risk', 'description': 'one or two risk factors such as age 75 or older, one prior admission, or many medications'},
           {'label': 'high readmission risk',   'description': 'several risk factors: two or more prior admissions, lives alone, long stay, heart failure, COPD, or sepsis'} ],
         {'task_description': 'Estimate 30-day readmission risk from a hospital discharge note.'}
       ):label::STRING AS RISK_BAND_AI
FROM DISCHARGE_NOTES
WHERE NOTE_ID <= 5
ORDER BY NOTE_ID;

-- C. COMPLETE with options: temperature 0, a max_tokens cap, and token usage you can read.
--    With options, COMPLETE returns JSON text: choices[0].messages plus usage.prompt_tokens / completion_tokens.
SELECT NOTE_ID,
       PARSE_JSON(R):choices[0]:messages::STRING     AS TOP_RISK_FACTORS,
       PARSE_JSON(R):usage:prompt_tokens::INT       AS PROMPT_TOKENS,
       PARSE_JSON(R):usage:completion_tokens::INT   AS COMPLETION_TOKENS
FROM (
  SELECT NOTE_ID,
         SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
           [ {'role': 'system', 'content': 'You are a discharge nurse. Answer in 25 words or fewer. Simulated data.'},
             {'role': 'user',   'content': 'List the readmission risk factors in this note: ' || NOTE_TEXT} ],
           {'temperature': 0, 'max_tokens': 60}) AS R
  FROM DISCHARGE_NOTES
  WHERE NOTE_ID <= 5
)
ORDER BY NOTE_ID;

-- D. All 40 rows, once, saved to a table. This is the main Cortex spend of the day.
CREATE OR REPLACE TABLE NOTE_AI AS
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.SUMMARIZE(NOTE_TEXT)                                   AS SUMMARY,
       SNOWFLAKE.CORTEX.SENTIMENT(PATIENT_FEEDBACK)                            AS FEEDBACK_SCORE,
       SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
         NOTE_TEXT,
         [ {'label': 'low readmission risk',    'description': 'younger or stable, no prior admissions, lives with others, short stay'},
           {'label': 'medium readmission risk', 'description': 'one or two risk factors such as age 75 or older, one prior admission, or many medications'},
           {'label': 'high readmission risk',   'description': 'several risk factors: two or more prior admissions, lives alone, long stay, heart failure, COPD, or sepsis'} ],
         {'task_description': 'Estimate 30-day readmission risk from a hospital discharge note.'}
       ):label::STRING                                                         AS RISK_LABEL_AI,
       TO_VARIANT(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
         'In how many days should the patient follow up?'))                    AS FOLLOWUP_RAW,
       TO_VARIANT(SNOWFLAKE.CORTEX.EXTRACT_ANSWER(NOTE_TEXT,
         'Which specialty or clinic should the patient follow up with?'))      AS FOLLOWUP_WITH_RAW
FROM DISCHARGE_NOTES;

-- EXTRACT_ANSWER gives an answer plus a confidence score. This parse works whether you get an array or a string.
SELECT NOTE_ID,
       IFF(IS_ARRAY(FOLLOWUP_RAW), FOLLOWUP_RAW[0]:answer::STRING, FOLLOWUP_RAW::STRING) AS FOLLOWUP_ANSWER,
       IFF(IS_ARRAY(FOLLOWUP_RAW), FOLLOWUP_RAW[0]:score::FLOAT, NULL)                   AS FOLLOWUP_SCORE,
       IFF(IS_ARRAY(FOLLOWUP_WITH_RAW), FOLLOWUP_WITH_RAW[0]:answer::STRING, FOLLOWUP_WITH_RAW::STRING) AS FOLLOWUP_WITH,
       RISK_LABEL_AI,
       ROUND(FEEDBACK_SCORE, 2) AS FEEDBACK_SCORE,
       SUMMARY
FROM NOTE_AI
ORDER BY NOTE_ID;
