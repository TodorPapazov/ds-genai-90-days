-- Day 39: guardrails and the cost check. Run at the end of the session.
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- ========== A. Guardrails ==========
-- 1) A read-only role for the app: it can read the three tables and the semantic view, and call Cortex Analyst only.
--    Generated SQL runs with the CALLER's role, so this role cannot change or drop anything. (Trial: run as ACCOUNTADMIN.)
USE ROLE ACCOUNTADMIN;
CREATE ROLE IF NOT EXISTS DS90_ANALYST_RO COMMENT = 'Day 39: read-only Cortex Analyst practice role';
GRANT USAGE ON WAREHOUSE DS90_XS TO ROLE DS90_ANALYST_RO;
GRANT USAGE ON DATABASE DS90 TO ROLE DS90_ANALYST_RO;
GRANT USAGE ON SCHEMA DS90.DAY39 TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.ENCOUNTERS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.PATIENTS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON TABLE DS90.DAY39.DRG_GROUPS TO ROLE DS90_ANALYST_RO;
GRANT SELECT ON SEMANTIC VIEW DS90.DAY39.READMIT_ANALYST TO ROLE DS90_ANALYST_RO;
GRANT READ ON STAGE DS90.DAY39.SEMANTIC_MODELS TO ROLE DS90_ANALYST_RO;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_ANALYST_USER TO ROLE DS90_ANALYST_RO;   -- Analyst only, not every AI function
GRANT ROLE DS90_ANALYST_RO TO USER TODORPAPAZOV;

-- 2) A runaway query cannot run for long: stop any statement on DS90_XS after 120 seconds.
ALTER WAREHOUSE DS90_XS SET STATEMENT_TIMEOUT_IN_SECONDS = 120;

-- 3) Optional budget fence for the practice warehouse (monthly, notify at 75%, suspend at 100%).
CREATE RESOURCE MONITOR IF NOT EXISTS DS90_MONTHLY WITH CREDIT_QUOTA = 10 FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 75 PERCENT DO NOTIFY ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE DS90_XS SET RESOURCE_MONITOR = DS90_MONTHLY;
-- A warehouse monitor does not cap Cortex Analyst messages (they are AI services, not warehouse credits).
-- The app-side guard (one SELECT, allowed tables, LIMIT, message cap) lives in day39_analyst.py and streamlit_app.py.

-- Test the role: this should work ...
USE ROLE DS90_ANALYST_RO;
USE WAREHOUSE DS90_XS;
SELECT COUNT(*) FROM DS90.DAY39.ENCOUNTERS;
-- ... and this should fail with "insufficient privileges". Uncomment to see it.
-- DELETE FROM DS90.DAY39.ENCOUNTERS;
USE ROLE ACCOUNTADMIN;

-- ========== B. Cost check ==========
-- Cortex Analyst: messages and credits per hour (ACCOUNT_USAGE can lag up to a few hours).
SELECT START_TIME, END_TIME, USERNAME, REQUEST_COUNT, CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY
WHERE START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- All AI services together (Cortex Analyst shows up here with SERVICE_TYPE = 'AI_SERVICES').
SELECT SERVICE_TYPE, NAME, START_TIME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES' AND START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- The warehouse that ran the generated SQL.
SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS' AND START_TIME >= DATEADD('day', -2, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- What actually ran today (available right away, no lag).
SELECT QUERY_TYPE, LEFT(QUERY_TEXT, 80) AS QUERY_TEXT, EXECUTION_STATUS,
       ROUND(TOTAL_ELAPSED_TIME / 1000, 1) AS SEC, START_TIME
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_WAREHOUSE('DS90_XS'))
ORDER BY START_TIME DESC LIMIT 30;

-- ========== C. Cleanup ==========
-- Keep the tables and the semantic view for Day 40 (a semantic view is metadata: no running cost).
-- The stage file costs only storage (a few KB). Stop the warehouse now instead of waiting 60 seconds.
ALTER WAREHOUSE DS90_XS SUSPEND;
