-- Day 38: what did today cost, then stop the meters.
-- ACCOUNT_USAGE views are not real time (often hours behind). If they are empty, check again tomorrow.
-- Rates (credits per GB-month for serving, per million tokens for embeddings and COMPLETE):
--   Snowflake Service Consumption Table, https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- A. The service right now: state, row count, size. Look for serving_state and indexing_state.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;

-- B. Daily embedding-token and serving credits per service
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY ORDER BY 1 DESC LIMIT 20;

-- C. Hourly serving credits per service
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_SERVING_USAGE_HISTORY ORDER BY 1 DESC LIMIT 48;

-- D. COMPLETE tokens from 06_rag_complete.sql
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY ORDER BY 1 DESC LIMIT 20;

-- E. All AI services credits per day, and the warehouse meter
SELECT USAGE_DATE, SERVICE_TYPE, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES' ORDER BY USAGE_DATE DESC LIMIT 14;

SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS' ORDER BY START_TIME DESC LIMIT 14;

-- F. Stop the meters. A RESUMED service bills serving credits even with zero queries.
ALTER CORTEX SEARCH SERVICE IF EXISTS DS90.DAY38.CLINICAL_DOC_SEARCH SUSPEND;   -- suspends indexing AND serving
-- (Resume later with: ALTER CORTEX SEARCH SERVICE DS90.DAY38.CLINICAL_DOC_SEARCH RESUME;)

-- G. Recommended once your README numbers are written down: drop it. Rebuilding costs only about
--    4,000 embedding tokens plus a minute of X-Small time. Day 39 (Cortex Analyst) does not use it.
DROP CORTEX SEARCH SERVICE IF EXISTS DS90.DAY38.CLINICAL_DOC_SEARCH;

ALTER WAREHOUSE DS90_XS SUSPEND;   -- if it says "cannot be suspended", it already is: fine
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;   -- expect nothing left running from Day 38
