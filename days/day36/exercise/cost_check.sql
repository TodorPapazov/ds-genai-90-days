-- Day 36: look at what you just ran, then suspend the warehouse so it stops using credits.
-- Query history is available right away. ACCOUNT_USAGE warehouse credits can lag by up to a few hours.

USE WAREHOUSE DS90_XS;
USE DATABASE DS90;
USE SCHEMA DAY36;

SELECT QUERY_ID,
       QUERY_TYPE,
       LEFT(QUERY_TEXT, 80) AS QUERY_TEXT,
       EXECUTION_STATUS,
       ROUND(TOTAL_ELAPSED_TIME / 1000, 1) AS SEC,
       WAREHOUSE_NAME,
       START_TIME
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_WAREHOUSE('DS90_XS'))
ORDER BY START_TIME DESC
LIMIT 20;

-- Optional, may be empty for a little while after a new trial:
SELECT START_TIME,
       WAREHOUSE_NAME,
       CREDITS_USED,
       CREDITS_USED_COMPUTE
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS'
ORDER BY START_TIME DESC
LIMIT 14;

-- Stop the warehouse now so you are not waiting on the 60-second auto-suspend.
ALTER WAREHOUSE DS90_XS SUSPEND;
