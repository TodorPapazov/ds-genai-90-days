-- Day 37: grade the LLM outputs against the answer key in NOTE_LABELS (simulated ground truth).
-- No new Cortex calls here: this only reads NOTE_AI, so it costs warehouse time only.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

CREATE OR REPLACE VIEW NOTE_EVAL AS
SELECT a.NOTE_ID,
       l.TRUE_RISK_BAND,
       SPLIT_PART(a.RISK_LABEL_AI, ' ', 1)                                          AS AI_RISK_BAND,
       l.TRUE_FOLLOWUP_DAYS,
       IFF(IS_ARRAY(a.FOLLOWUP_RAW), a.FOLLOWUP_RAW[0]:answer::STRING, a.FOLLOWUP_RAW::STRING) AS AI_FOLLOWUP_TEXT,
       -- turn "7 days" / "2 weeks" / "1 month" into days
       TRY_TO_NUMBER(REGEXP_SUBSTR(AI_FOLLOWUP_TEXT, '[0-9]+'))
         * CASE WHEN AI_FOLLOWUP_TEXT ILIKE '%week%' THEN 7
                WHEN AI_FOLLOWUP_TEXT ILIKE '%month%' THEN 30 ELSE 1 END             AS AI_FOLLOWUP_DAYS,
       l.TRUE_FEEDBACK_TONE,
       CASE WHEN a.FEEDBACK_SCORE >= 0.5 THEN 'positive'
            WHEN a.FEEDBACK_SCORE <= -0.5 THEN 'negative' ELSE 'neutral' END        AS AI_FEEDBACK_TONE,
       a.FEEDBACK_SCORE
FROM NOTE_AI a
JOIN NOTE_LABELS l USING (NOTE_ID);

-- Accuracy per task (compare with the local keyword baseline printed by local_baseline.py)
SELECT ROUND(AVG(IFF(AI_RISK_BAND = TRUE_RISK_BAND, 1, 0)), 3)                       AS RISK_ACCURACY,
       ROUND(AVG(IFF(EQUAL_NULL(AI_FOLLOWUP_DAYS, TRUE_FOLLOWUP_DAYS), 1, 0)), 3)     AS FOLLOWUP_ACCURACY,
       ROUND(AVG(IFF(AI_FEEDBACK_TONE = TRUE_FEEDBACK_TONE, 1, 0)), 3)               AS TONE_ACCURACY,
       COUNT(*)                                                                       AS N
FROM NOTE_EVAL;

-- Confusion table for risk bands: where does the LLM disagree with the answer key?
SELECT TRUE_RISK_BAND, AI_RISK_BAND, COUNT(*) AS N
FROM NOTE_EVAL
GROUP BY 1, 2
ORDER BY 1, 2;

-- Notes with NO follow-up scheduled: did EXTRACT_ANSWER invent one? (hallucination check)
SELECT NOTE_ID, AI_FOLLOWUP_TEXT
FROM NOTE_EVAL
WHERE TRUE_FOLLOWUP_DAYS IS NULL;

-- Operational list: high-risk patients with unhappy feedback get a call within 48 hours.
SELECT e.NOTE_ID, n.PATIENT_ID, e.AI_RISK_BAND, e.FEEDBACK_SCORE, e.AI_FOLLOWUP_TEXT
FROM NOTE_EVAL e JOIN DISCHARGE_NOTES n USING (NOTE_ID)
WHERE e.AI_RISK_BAND = 'high' OR e.FEEDBACK_SCORE <= -0.5
ORDER BY e.AI_RISK_BAND = 'high' DESC, e.FEEDBACK_SCORE;
