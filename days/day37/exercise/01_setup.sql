-- Day 37: reuse the Day 36 X-Small warehouse and make a schema for Cortex practice.
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Two meters run today:
--   1) the warehouse (X-Small, billed while running, AUTO_SUSPEND = 60 seconds)
--   2) Cortex AI tokens (billed per million tokens processed, see the Service Consumption Table)

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
-- Snowflake docs: use a warehouse no larger than MEDIUM for Cortex AI functions.
-- A bigger warehouse does not make the LLM faster, it only costs more.
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY37
  COMMENT = 'Day 37: Cortex LLM functions on SIMULATED discharge notes. Not real patient data.';

USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY37;

-- Can my role call Cortex? (USE AI FUNCTIONS is granted to PUBLIC by default, plus SNOWFLAKE.CORTEX_USER)
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- Cross-region inference setting. Read the "value" column.
-- llama3.1-8b runs natively in AWS US East 1, so today's lesson does not need cross-region.
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;

-- Which base models can my role use, and are any of them legacy?
SHOW CORTEX BASE MODELS LIKE 'llama3%';

-- One tiny smoke test (a few dozen tokens). If this errors, fix access before you run 40 rows.
SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', 'Reply with the single word: ready') AS SMOKE_TEST;
