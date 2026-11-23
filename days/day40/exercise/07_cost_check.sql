-- Day 40 step 7: the cost and credit check for the whole Snowflake week (Days 36-40), then clean up.
-- ACCOUNT_USAGE views lag (warehouse metering up to about 3 hours, CoCo views up to about 1 hour, some longer).
-- If today's rows are missing, run this again tomorrow. Rates: Snowflake Service Consumption Table
--   https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf  (not copied here; they change)
USE ROLE ACCOUNTADMIN;          -- ACCOUNT_USAGE is readable by ACCOUNTADMIN by default
USE WAREHOUSE DS90_XS;
SET WEEK_START = '2026-11-16'::TIMESTAMP_LTZ;

-- A. The whole account, per day and service type: the one table to read first.
--    WAREHOUSE_METERING = warehouses; AI_SERVICES = Cortex (functions, Analyst, Search, CoCo); CLOUD_SERVICES is
--    adjusted (you pay only the part above 10% of daily compute: see CREDITS_ADJUSTMENT_CLOUD_SERVICES).
SELECT USAGE_DATE, SERVICE_TYPE,
       ROUND(SUM(CREDITS_USED), 4)        AS CREDITS_USED,
       ROUND(SUM(CREDITS_BILLED), 4)      AS CREDITS_BILLED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE USAGE_DATE >= $WEEK_START::DATE
GROUP BY 1, 2 ORDER BY 1, 2;

-- B. Warehouse credits per hour for DS90_XS (and anything else that ran).
SELECT WAREHOUSE_NAME, DATE_TRUNC('day', START_TIME)::DATE AS DAY,
       COUNT(*) AS HOURS_WITH_USAGE, ROUND(SUM(CREDITS_USED), 4) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE START_TIME >= $WEEK_START
GROUP BY 1, 2 ORDER BY 2, 1;

-- C. AI Functions tokens and credits by function and model (Days 37, 38, 40: AI_COMPLETE, COMPLETE, ...).
--    Note: CORTEX_FUNCTIONS_USAGE_HISTORY is no longer updated; the docs point to CORTEX_AISQL_USAGE_HISTORY.
SELECT DATE_TRUNC('day', USAGE_TIME)::DATE AS DAY, FUNCTION_NAME, MODEL_NAME,
       COUNT(DISTINCT QUERY_ID) AS QUERIES, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AISQL_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START
GROUP BY 1, 2, 3 ORDER BY 1, CREDITS DESC;
-- The older view, for rows written before it stopped updating (may be empty or stale):
SELECT FUNCTION_NAME, MODEL_NAME, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
WHERE START_TIME >= $WEEK_START GROUP BY 1, 2 ORDER BY CREDITS DESC;

-- D. Cortex Analyst (Day 39): billed per message, not per token.
SELECT DATE_TRUNC('day', START_TIME)::DATE AS DAY, SUM(REQUEST_COUNT) AS MESSAGES, ROUND(SUM(CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_ANALYST_USAGE_HISTORY
WHERE START_TIME >= $WEEK_START GROUP BY 1 ORDER BY 1;

-- E. Cortex Search (Day 38): serving and embedding credits per day.
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_SEARCH_DAILY_USAGE_HISTORY
WHERE USAGE_DATE >= $WEEK_START::DATE ORDER BY USAGE_DATE;

-- F. Cortex Code (today): one row per request, credits and tokens. Separate views per surface.
SELECT 'Snowsight' AS SURFACE, USER_NAME, COUNT(*) AS REQUESTS, SUM(TOKENS) AS TOKENS, ROUND(SUM(TOKEN_CREDITS), 5) AS CREDITS
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START GROUP BY 1, 2
UNION ALL
SELECT 'CLI', USER_NAME, COUNT(*), SUM(TOKENS), ROUND(SUM(TOKEN_CREDITS), 5)
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_CLI_USAGE_HISTORY
WHERE USAGE_TIME >= $WEEK_START GROUP BY 1, 2;
-- Which models did CoCo use, per request (TOKENS_GRANULAR is keyed by model name):
SELECT REQUEST_ID, USAGE_TIME, TOKEN_CREDITS, TOKENS_GRANULAR
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY
WHERE USAGE_TIME >= DATEADD('day', -1, CURRENT_TIMESTAMP()) ORDER BY USAGE_TIME DESC LIMIT 20;

-- G. Today's queries on DS90_XS (no lag): what woke the warehouse up, and for how long.
SELECT QUERY_TYPE, LEFT(QUERY_TEXT, 70) AS QUERY_TEXT, EXECUTION_STATUS,
       ROUND(TOTAL_ELAPSED_TIME / 1000, 1) AS SEC, START_TIME
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_WAREHOUSE(WAREHOUSE_NAME => 'DS90_XS', RESULT_LIMIT => 200))
WHERE START_TIME >= DATEADD('hour', -12, CURRENT_TIMESTAMP())
ORDER BY START_TIME DESC;

-- H. Fences: resource monitors (warehouse credits only) and CoCo per-user limits (AI credits).
SHOW RESOURCE MONITORS;                                    -- DS90_MONTHLY from Day 39, if you created it
CREATE RESOURCE MONITOR IF NOT EXISTS DS90_MONTHLY WITH CREDIT_QUOTA = 10 FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY TRIGGERS ON 75 PERCENT DO NOTIFY ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE DS90_XS SET RESOURCE_MONITOR = DS90_MONTHLY;
SHOW WAREHOUSES LIKE 'DS90_XS';                            -- check auto_suspend = 60 and resource_monitor = DS90_MONTHLY
-- A resource monitor does NOT stop Cortex, Analyst, Search or CoCo spend. For CoCo, use the per-user limit from 01_setup.sql.

-- I. Optional: ORGANIZATION_USAGE (money, not just credits). Needs the organization account or an
--    ORGADMIN-enabled account; on a trial these views may be missing or empty, and that is fine.
-- SELECT USAGE_DATE, USAGE_TYPE, ROUND(SUM(USAGE_IN_CURRENCY), 2) AS USD
-- FROM SNOWFLAKE.ORGANIZATION_USAGE.USAGE_IN_CURRENCY_DAILY
-- WHERE USAGE_DATE >= $WEEK_START::DATE GROUP BY 1, 2 ORDER BY 1, 2;
-- SELECT * FROM SNOWFLAKE.ORGANIZATION_USAGE.REMAINING_BALANCE_DAILY ORDER BY DATE DESC LIMIT 7;
-- Snowsight also has cost and billing pages for admins (exact menu names not verified for this lesson; see the docs).

-- J. Cleanup. Keep DS90.DAY40 tables (storage is tiny). Stop the meters.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;                    -- expect none; drop any you no longer need
ALTER WAREHOUSE DS90_XS SUSPEND;                           -- "cannot be suspended" means it already is: fine
