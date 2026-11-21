-- Day 38: Cortex Search over SIMULATED clinical documents (a first look at RAG).
-- Run in a Snowsight worksheet: https://app.snowflake.com/us-east-1/rhc24484/
-- Three meters run today:
--   1) the warehouse DS90_XS (X-Small, billed per second while running, AUTO_SUSPEND = 60 seconds).
--      It builds and refreshes the search index and runs your SQL.
--   2) embedding tokens: every chunk is embedded once when the index is built (and again only if the row changes).
--   3) serving: credits per GB per month of indexed data, charged while the service is RESUMED, even with no queries.
--   Plus COMPLETE tokens for the RAG answers in 06_rag_complete.sql.

CREATE WAREHOUSE IF NOT EXISTS DS90_XS
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Days 36-40 practice. X-Small, auto-suspend 60 seconds.';
-- Docs: no larger than MEDIUM for Cortex Search. Indexing time is mostly the embedding step,
-- which does not speed up with a bigger warehouse.
ALTER WAREHOUSE DS90_XS SET WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS DS90;
CREATE SCHEMA IF NOT EXISTS DS90.DAY38
  COMMENT = 'Day 38: Cortex Search on SIMULATED clinical documents. Not real patient data or real policies.';

USE WAREHOUSE DS90_XS;
USE SCHEMA DS90.DAY38;

-- Who am I, where am I? Cortex Search runs natively in AWS US East 1 (your trial region).
SELECT CURRENT_ROLE() AS ROLE, CURRENT_REGION() AS REGION, CURRENT_WAREHOUSE() AS WAREHOUSE;

-- The creator role needs the SNOWFLAKE.CORTEX_USER (or CORTEX_EMBED_USER) database role,
-- CREATE CORTEX SEARCH SERVICE on the schema, SELECT on the source table and USAGE on the warehouse.
-- On a trial, ACCOUNTADMIN or SYSADMIN normally has all of these. Check what your role inherits:
SHOW GRANTS TO ROLE IDENTIFIER(CURRENT_ROLE());

-- Any search services left over from earlier practice? Each one that is RESUMED costs serving credits.
SHOW CORTEX SEARCH SERVICES IN ACCOUNT;
