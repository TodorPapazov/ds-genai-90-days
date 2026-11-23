-- Day 40 step 6: grade it. Two questions: did the LLM read the notes right, and did the notes help the list?
-- NOTE_LABELS and OUTCOMES are the simulator's answer key. In real work you would not have OUTCOMES for weeks.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;

-- A. Extraction accuracy on the 60 shortlisted notes (lesson-box rule baseline: see local_check_output.txt).
SELECT COUNT(*) AS NOTES,
       ROUND(AVG(IFF(f.LIVES_ALONE_AI = (l.LIVES_ALONE = 1), 1, 0)), 3)         AS LIVES_ALONE_ACC,
       ROUND(AVG(IFF(f.FOLLOWUP_BOOKED_AI = (l.FOLLOWUP_BOOKED = 1), 1, 0)), 3) AS FOLLOWUP_ACC
FROM NOTE_FIELDS_PARSED f JOIN NOTE_LABELS l USING (ENCOUNTER_ID);

-- The misses, with the note text, so you can see WHY (negation? "on his own"? "attempted, no openings"?).
SELECT f.ENCOUNTER_ID, l.LIVES_ALONE, f.LIVES_ALONE_AI, l.FOLLOWUP_BOOKED, f.FOLLOWUP_BOOKED_AI, d.NOTE_TEXT
FROM NOTE_FIELDS_PARSED f JOIN NOTE_LABELS l USING (ENCOUNTER_ID) JOIN DISCHARGES d USING (ENCOUNTER_ID)
WHERE f.LIVES_ALONE_AI <> (l.LIVES_ALONE = 1) OR f.FOLLOWUP_BOOKED_AI <> (l.FOLLOWUP_BOOKED = 1);

-- B. Did the notes help? Readmissions caught in a 25-patient list: score only vs score + note fields.
WITH by_score AS (SELECT ENCOUNTER_ID FROM SHORTLIST WHERE SCORE_RANK <= 25),
     by_priority AS (SELECT ENCOUNTER_ID FROM FOLLOWUP_WORKLIST)
SELECT 'score only (top 25)' AS LIST, SUM(o.READMIT_30D) AS READMITS_CAUGHT, ROUND(AVG(o.READMIT_30D), 3) AS PRECISION_AT_25
FROM by_score JOIN OUTCOMES o USING (ENCOUNTER_ID)
UNION ALL
SELECT 'score + note fields (top 25)', SUM(o.READMIT_30D), ROUND(AVG(o.READMIT_30D), 3)
FROM by_priority JOIN OUTCOMES o USING (ENCOUNTER_ID)
UNION ALL
SELECT 'everyone (base rate)', SUM(READMIT_30D), ROUND(AVG(READMIT_30D), 3) FROM OUTCOMES;
-- Lesson-box numbers to compare with (local_check_output.txt): see the lesson. Yours can differ a little
-- because the LLM may read a few notes differently from the answer key.
