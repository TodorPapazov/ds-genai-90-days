-- Day 40, Cortex Code practice file. Two bugs on purpose. Do not fix them by hand first.
-- Bug 1 makes the query FAIL: run it, then use the Fix button CoCo offers in the results grid (per the docs).
-- Bug 2 does NOT fail: the query runs and returns a wrong number. Ask CoCo to review it and see if it notices.
USE SCHEMA DS90.DAY40;

-- 1) Readmission rate by service line for this week's discharges. Fails: wrong column name and a missing GROUP BY column.
SELECT d.SERVICE_LINE, d.PAYER, ROUND(AVG(o.READMIT30), 3) AS RATE, COUNT(*) AS N
FROM DISCHARGES d JOIN OUTCOMES o ON o.ENCOUNTER_ID = d.ENCOUNTER_ID
GROUP BY d.SERVICE_LINE
ORDER BY RATE DESC;

-- 2) Share of the 25-patient worklist that had no follow-up booked. Runs fine, but the answer is wrong:
--    it divides by all 300 discharges instead of the 25 people on the list.
SELECT ROUND(COUNT_IF(NOT w.FOLLOWUP_BOOKED_AI) / (SELECT COUNT(*) FROM DISCHARGES), 3) AS SHARE_NO_FOLLOWUP
FROM FOLLOWUP_WORKLIST w;
