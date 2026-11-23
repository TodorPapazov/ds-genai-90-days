-- Day 40 step 5: the deliverable. A follow-up call worklist for care management (SIMULATED patients).
-- PRIORITY = model score + 0.10 if lives alone + 0.12 if no follow-up is booked. Simple and explainable on purpose.
-- The weights are teaching choices, not clinical guidance.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

CREATE OR REPLACE TABLE FOLLOWUP_WORKLIST AS
SELECT s.ENCOUNTER_ID, s.PATIENT_ID, s.DISCHARGE_DATE, s.SERVICE_LINE, s.PAYER, s.DISCHARGE_DISPOSITION,
       s.SCORE_RANK, ROUND(s.RISK_SCORE, 3) AS RISK_SCORE,
       f.LIVES_ALONE_AI, f.FOLLOWUP_BOOKED_AI,
       ROUND(s.RISK_SCORE + 0.10 * IFF(f.LIVES_ALONE_AI, 1, 0) + 0.12 * IFF(f.FOLLOWUP_BOOKED_AI, 0, 1), 3) AS PRIORITY,
       b.DAY39_READMIT_RATE AS SERVICE_LINE_BASELINE,
       g.TITLE AS GUIDANCE_DOC
FROM SHORTLIST s
JOIN NOTE_FIELDS_PARSED f ON f.ENCOUNTER_ID = s.ENCOUNTER_ID
LEFT JOIN BASELINE_RATES b ON b.SERVICE_LINE = s.SERVICE_LINE
LEFT JOIN GUIDANCE_MAP g ON g.SERVICE_LINE = s.SERVICE_LINE
QUALIFY ROW_NUMBER() OVER (ORDER BY PRIORITY DESC, s.ENCOUNTER_ID) <= 25;

SELECT * FROM FOLLOWUP_WORKLIST ORDER BY PRIORITY DESC, ENCOUNTER_ID;

-- How did the note fields move people? Rank by score vs rank by priority.
SELECT COUNT_IF(SCORE_RANK > 25) AS PULLED_UP_FROM_RANK_26_TO_60,
       COUNT_IF(NOT FOLLOWUP_BOOKED_AI) AS NO_FOLLOWUP_BOOKED,
       COUNT_IF(LIVES_ALONE_AI) AS LIVES_ALONE
FROM FOLLOWUP_WORKLIST;

-- One LLM call for a short weekly brief, built from AGGREGATES only (no names, no ids, no note text).
WITH agg AS (
  SELECT LISTAGG(SERVICE_LINE || ': ' || N || ' patients, ' || NO_FU || ' with no follow-up booked', '; ')
           WITHIN GROUP (ORDER BY N DESC) AS LINES
  FROM (SELECT SERVICE_LINE, COUNT(*) AS N, COUNT_IF(NOT FOLLOWUP_BOOKED_AI) AS NO_FU
        FROM FOLLOWUP_WORKLIST GROUP BY SERVICE_LINE)
)
SELECT LINES,
       AI_COMPLETE('llama3.1-8b',
         'Write a 3-sentence weekly note for a hospital care management team. Simulated data. '
         || '25 recently discharged patients are on the follow-up call list. By service line: ' || LINES
         || '. Say where to start and why. Do not invent numbers.') AS WEEKLY_BRIEF
FROM agg;
