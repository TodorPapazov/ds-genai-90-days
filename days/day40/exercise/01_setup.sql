-- Day 40: setup, inventory of the Snowflake week, and Cortex Code (CoCo) access checks.
-- Run in a Snowsight worksheet or a Workspaces SQL file: https://app.snowflake.com/us-east-1/rhc24484/
-- Meters today: warehouse time (X-Small = 1 credit/hour, per second, 60 s minimum per resume),
-- AI Functions tokens (03, 05), and CoCo tokens (every prompt you send to Cortex Code). Rates: Service Consumption Table.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY40
  COMMENT = 'Day 40: Snowflake week mini project on SIMULATED discharges. Not real patient data.';
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY40;
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_USER() AS ME;

-- ========== A. Inventory: what is left from Days 36-39? ==========
-- Tables and views per schema (metadata only, cheap).
SELECT TABLE_SCHEMA, TABLE_TYPE, COUNT(*) AS N, SUM(ROW_COUNT) AS ROWS_TOTAL
FROM DS90.INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA LIKE 'DAY3%' OR TABLE_SCHEMA = 'DAY40'
GROUP BY 1, 2 ORDER BY 1, 2;
SHOW MODELS IN DATABASE DS90;                      -- Day 36: READMIT30_SNOWML (V1, alias TRIAL_V1) if you logged it
SHOW SEMANTIC VIEWS IN DATABASE DS90;              -- Day 39: READMIT_ANALYST
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;            -- Day 38: should be empty if you dropped it (it bills while resumed)
SHOW WAREHOUSES LIKE 'DS90%';                      -- one X-Small, AUTO_SUSPEND 60, state SUSPENDED when idle

-- ========== B. Cortex Code (CoCo) access checks ==========
-- Docs (checked Nov 23, 2026): the role needs SNOWFLAKE.CORTEX_USER or SNOWFLAKE.CORTEX_AGENT_USER
-- (CORTEX_USER goes to PUBLIC by default), and until behavior change BCR-2445 reaches the account, also SNOWFLAKE.COPILOT_USER.
-- On self-service trials, AI features stay off until an account admin adds a credit card (that does not end the trial).
SHOW GRANTS TO ROLE PUBLIC;                        -- look for DATABASE_ROLE SNOWFLAKE.CORTEX_USER
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;   -- CoCo needs cross-region inference enabled
-- Daily estimated credit limit per user for CoCo in Snowsight: -1 = no limit (default), 0 = blocked.
SHOW PARAMETERS LIKE 'CORTEX_CODE%' IN ACCOUNT;
SHOW PARAMETERS LIKE 'CORTEX_CODE%' IN USER TODORPAPAZOV;

-- ========== C. A personal fence for today (needs ACCOUNTADMIN) ==========
-- A small rolling 24-hour cap on YOUR CoCo in Snowsight usage. 2 is an arbitrary practice number:
-- check the rate table, then pick your own. UNSET later to go back to the account default.
USE ROLE ACCOUNTADMIN;
ALTER USER TODORPAPAZOV SET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER = 2;
SHOW PARAMETERS LIKE 'CORTEX_CODE_SNOWSIGHT%' IN USER TODORPAPAZOV;
-- To remove it:  ALTER USER TODORPAPAZOV UNSET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER;

-- A log of what you asked CoCo today, filled in by hand during the guided session (cortex_code_session.md).
CREATE TABLE IF NOT EXISTS DS90.DAY40.COCO_LOG (
  PROMPT_ID STRING, ASKED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(), SURFACE STRING, PROMPT STRING,
  WHAT_IT_DID STRING, RAN_SQL BOOLEAN, MATCHED_REFERENCE STRING, MY_EDITS STRING);
