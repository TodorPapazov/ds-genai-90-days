-- Day 36: one X-Small warehouse, 60-second auto-suspend, a practice database and schema.
-- Run this from the VS Code Snowflake extension (SQL worksheet) or a Snowsight worksheet.
-- An X-Small warehouse uses 1 credit per hour while it is running. Auto-suspend stops the clock
-- after 60 seconds idle. Auto-resume starts it again on the next query.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';

CREATE DATABASE IF NOT EXISTS DS90
  COMMENT = 'Simulated 30-day readmission practice data. Not real patient data.';

CREATE SCHEMA IF NOT EXISTS DS90.DAY36
  COMMENT = 'Day 36: Snowpark features, query plans, Snowpark ML, Model Registry.';

USE WAREHOUSE DS90_XS;
USE DATABASE DS90;
USE SCHEMA DAY36;

SELECT CURRENT_ACCOUNT() AS ACCOUNT,
       CURRENT_USER()    AS USER_NAME,
       CURRENT_ROLE()    AS ROLE,
       CURRENT_WAREHOUSE() AS WAREHOUSE,
       CURRENT_DATABASE()  AS DATABASE,
       CURRENT_SCHEMA()    AS SCHEMA;
