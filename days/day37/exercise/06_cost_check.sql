-- Day 37: what did today cost? Two meters: Cortex tokens and warehouse time.
-- ACCOUNT_USAGE views are not real time. If they are empty, run this again later in the day.
-- Credits per million tokens for each function and model: Snowflake Service Consumption Table
--   https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
USE WAREHOUSE DS90_XS;

-- Cortex tokens and credits per query, grouped by model and function
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_QUERY_USAGE_HISTORY
ORDER BY 1 DESC
LIMIT 50;

-- Cortex usage per function and model (hourly)
SELECT *
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_FUNCTIONS_USAGE_HISTORY
ORDER BY 1 DESC
LIMIT 50;

-- Daily AI services credits for the account
SELECT USAGE_DATE, SERVICE_TYPE, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE SERVICE_TYPE = 'AI_SERVICES'
ORDER BY USAGE_DATE DESC;

-- Warehouse side (the meter that runs while DS90_XS is awake)
SELECT START_TIME, WAREHOUSE_NAME, CREDITS_USED
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE WAREHOUSE_NAME = 'DS90_XS'
ORDER BY START_TIME DESC
LIMIT 14;

-- Done for today: stop the warehouse now instead of waiting 60 seconds.
ALTER WAREHOUSE DS90_XS SUSPEND;
