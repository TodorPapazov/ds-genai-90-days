-- Day 39: judge Cortex Analyst's SQL against your hand-written gold SQL, inside Snowflake.
-- Fill ANALYST_RUNS.GENERATED_SQL first: python day39_analyst.py does it, or paste SQL from the Streamlit app.
-- Running two SQL statements per question uses the warehouse only (no Cortex Analyst message is sent here).
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- A. Paste by hand (example). Leave GENERATED_SQL NULL when Analyst returned suggestions or refused.
-- UPDATE ANALYST_RUNS SET GENERATED_SQL = $$<paste the SQL here, no trailing semicolon>$$ WHERE QID = 'Q03';

-- B. One pair, the long way: rows in gold that are not in generated, and the other way round. Both must be empty.
--    MINUS compares by column POSITION, so put the columns in the same order first.
SELECT PAYER, ROUND(AVG(READMIT_30D), 3) AS R FROM ENCOUNTERS GROUP BY PAYER
MINUS
SELECT PAYER, ROUND(AVG(READMIT_30D), 3) FROM ENCOUNTERS WHERE PAYER IS NOT NULL GROUP BY PAYER;

-- C. All pairs, the strict way: HASH_AGG(*) is an order-independent fingerprint of a whole result set.
--    Same hash = identical rows. Different hash can still be a fair answer (extra column, other rounding, other
--    column order), so read every mismatch, or use the tolerant compare in day39_analyst.py.
CREATE OR REPLACE TABLE ANSWER_HASHES (QID VARCHAR, SIDE VARCHAR, H NUMBER(38,0), N_ROWS NUMBER, ERR VARCHAR);

EXECUTE IMMEDIATE $$
DECLARE
  c1 CURSOR FOR SELECT QID, GOLD_SQL, GENERATED_SQL FROM ANALYST_RUNS
                WHERE GOLD_SQL IS NOT NULL AND GENERATED_SQL IS NOT NULL;
  stmt VARCHAR;
BEGIN
  FOR r IN c1 DO
    LET qid VARCHAR := r.QID;
    LET g VARCHAR := RTRIM(TRIM(r.GOLD_SQL), ';');
    LET a VARCHAR := RTRIM(TRIM(r.GENERATED_SQL), ';');
    stmt := 'INSERT INTO ANSWER_HASHES SELECT ''' || qid || ''', ''gold'', HASH_AGG(*), COUNT(*), NULL FROM (' || g || ')';
    EXECUTE IMMEDIATE :stmt;
    BEGIN
      stmt := 'INSERT INTO ANSWER_HASHES SELECT ''' || qid || ''', ''generated'', HASH_AGG(*), COUNT(*), NULL FROM (' || a || ')';
      EXECUTE IMMEDIATE :stmt;
    EXCEPTION
      WHEN OTHER THEN
        LET err VARCHAR := SQLERRM;
        INSERT INTO ANSWER_HASHES VALUES (:qid, 'generated', NULL, NULL, :err);
    END;
  END FOR;
  RETURN 'hashed';
END;
$$;

UPDATE ANALYST_RUNS R SET HASH_MATCH = (G.H = A.H)
FROM ANSWER_HASHES G JOIN ANSWER_HASHES A ON G.QID = A.QID AND G.SIDE = 'gold' AND A.SIDE = 'generated'
WHERE R.QID = G.QID;

-- D. Scorecard. Q14 and Q15 are correct when there is NO SQL (a refusal or suggestions).
SELECT QID, QTYPE, IN_VQR,
       CASE WHEN GOLD_SQL IS NULL THEN IFF(GENERATED_SQL IS NULL, 'correct refusal', 'answered when it should not')
            WHEN GENERATED_SQL IS NULL THEN 'no SQL returned'
            WHEN HASH_MATCH THEN 'exact match'
            ELSE 'check by eye' END AS VERDICT,
       VERIFIED_QUERY_USED, NOTE
FROM ANALYST_RUNS ORDER BY QID;

SELECT IN_VQR,
       COUNT(*) AS QUESTIONS,
       COUNT_IF(HASH_MATCH) AS EXACT_MATCHES,
       ROUND(COUNT_IF(HASH_MATCH) / NULLIF(COUNT(*), 0), 2) AS EXACT_MATCH_RATE
FROM ANALYST_RUNS WHERE GOLD_SQL IS NOT NULL GROUP BY IN_VQR;

-- E. Read the mismatches side by side.
SELECT R.QID, R.QUESTION, G.N_ROWS AS GOLD_ROWS, A.N_ROWS AS GEN_ROWS, A.ERR, R.GOLD_SQL, R.GENERATED_SQL
FROM ANALYST_RUNS R
JOIN ANSWER_HASHES G ON G.QID = R.QID AND G.SIDE = 'gold'
JOIN ANSWER_HASHES A ON A.QID = R.QID AND A.SIDE = 'generated'
WHERE NOT COALESCE(R.HASH_MATCH, FALSE);
