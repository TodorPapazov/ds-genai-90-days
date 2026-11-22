-- Day 39: Cortex Analyst setup. Reuse the X-Small warehouse, make schema DS90.DAY39 and a stage for the YAML.
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Two meters today:
--   1) the warehouse runs the SQL that Cortex Analyst writes (X-Small = 1 credit/hour, per second, 60 s minimum per resume)
--   2) Cortex Analyst is billed per successful message (HTTP 200), not per token. Rates: Service Consumption Table.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY39
  COMMENT = 'Day 39: Cortex Analyst over SIMULATED readmission tables. Not real patient data.';
USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY39;

-- Internal stage for the semantic model YAML (directory table on, so Snowsight can list files).
CREATE STAGE IF NOT EXISTS SEMANTIC_MODELS
  DIRECTORY = (ENABLE = TRUE)
  COMMENT = 'Semantic model YAML files for Cortex Analyst (Day 39).';

-- Who am I, and where? Cortex Analyst is native in AWS us-east-1, so no cross-region setting is needed.
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- Access: your role needs SNOWFLAKE.CORTEX_USER (granted to PUBLIC by default) or SNOWFLAKE.CORTEX_ANALYST_USER,
-- plus SELECT on the tables and READ on the stage. Is Cortex Analyst switched on for the account?
SHOW PARAMETERS LIKE 'ENABLE_CORTEX_ANALYST' IN ACCOUNT;
SHOW GRANTS TO ROLE PUBLIC;            -- look for DATABASE_ROLE SNOWFLAKE.CORTEX_USER

-- Leftovers from earlier days that still cost money? (Day 38 search service should be gone.)
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;
SHOW WAREHOUSES LIKE 'DS90%';
