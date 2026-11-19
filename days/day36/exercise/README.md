# Day 36: Snowpark and Snowpark ML on your Snowflake trial

Re-run the real-account parts of Days 32 and 33: query plans, training in the warehouse, and the Snowflake Model Registry. All data is simulated 30-day readmission stays. Never put credentials in this folder or in git.

## Main path (VS Code)

1. Install the Snowflake extension in VS Code (publisher: Snowflake) and sign in. The trial login is in your vault.
2. Copy `connections.toml.example` to `~/.snowflake/connections.toml` and fill in the placeholders (or set `SNOWFLAKE_*` env vars in your shell). Do not commit the filled file.
3. `conda activate ds90`. Install `snowflake-snowpark-python` and `snowflake-ml-python[xgboost]` if they are missing.
4. Copy these exercise files into the `day36/` folder of your practice repo.
5. Run `setup_warehouse.sql` in a VS Code SQL worksheet (X-Small, auto-suspend 60 seconds, database `DS90`, schema `DAY36`).
6. From `day36/`, run:

```
python day36_snowflake.py
```

7. Read the SQL from `feat.queries` and the plan from `feat.explain()`. Write one takeaway in this README.
8. Record the warehouse test AUC and compare it with 0.734 (local pandas fit, also Day 33) and 0.743 (Day 31 sklearn champion).
9. Confirm `READMIT30_SNOWML` version V1 and alias `TRIAL_V1`, and that SQL `PREDICT_PROBA` returns rows.
10. Run `cost_check.sql` and `ALTER WAREHOUSE DS90_XS SUSPEND`.

## Short path (Snowsight notebook)

Open a Python notebook in Snowsight, attach warehouse `DS90_XS`, database `DS90`, schema `DAY36`. Replace the `connect()` call with:

```
from snowflake.snowpark.context import get_active_session
session = get_active_session()
LOCAL = False
```

Then run snippets 3 to 5 from `day36_snowflake.py`. A notebook keeps the warehouse awake for the whole time it is attached, so it uses credits even while you are reading a cell. Detach when you finish, then suspend the warehouse.

## Local testing fallback (no account)

```
python day36_snowflake.py --local
```

This rebuilds the features and fits the logistic pipeline on pandas. It cannot show a real query plan, train in a warehouse, or use the Model Registry. On the lesson box this scored test AUC 0.734 on 2,655 stays (train 1,997 / test 658).

## Files

- `day36_snowflake.py`: snippets 1, 3, 4, 5 and 6 in one script
- `sim_readmit.py`: `simulate(1000, 30)` rebuilds the Day 30 table
- `setup_warehouse.sql`: X-Small warehouse, database, schema
- `cost_check.sql`: query history, optional credit history, then SUSPEND
- `connections.toml.example`: placeholders only

Keep `connections.toml` and `.env` out of git. All data is simulated for teaching.
