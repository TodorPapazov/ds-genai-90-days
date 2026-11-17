-- PSI of the risk score as one SQL query (Snowflake syntax; also runs in SQLite for the local check).
-- In a Snowflake worksheet: SET start_date = '2026-08-17'; then write $start_date where this file says :start_date.
-- PSI_EDGES holds the 10 REFERENCE decile bins (lo, hi]; REF_SCORES = validation stays; LIVE_SCORES = new stays.
WITH ref AS (
    SELECT e.bin, COUNT(r.risk) * 1.0 / (SELECT COUNT(*) FROM REF_SCORES) AS p
    FROM PSI_EDGES e LEFT JOIN REF_SCORES r ON r.risk > e.lo AND r.risk <= e.hi
    GROUP BY e.bin),
cur AS (
    SELECT e.bin, COUNT(s.risk) * 1.0 / (SELECT COUNT(*) FROM LIVE_SCORES WHERE admit_date >= :start_date) AS q
    FROM PSI_EDGES e LEFT JOIN LIVE_SCORES s ON s.risk > e.lo AND s.risk <= e.hi AND s.admit_date >= :start_date
    GROUP BY e.bin),
clipped AS (
    SELECT ref.bin,
           CASE WHEN p < 0.0001 THEN 0.0001 ELSE p END AS p,
           CASE WHEN q < 0.0001 THEN 0.0001 ELSE q END AS q
    FROM ref JOIN cur ON ref.bin = cur.bin)
SELECT ROUND(SUM((q - p) * LN(q / p)), 4) AS psi_score FROM clipped;
