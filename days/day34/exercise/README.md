# Day 34: model drift and calibration (simulated readmission data)

Files for the `day34/` folder of the practice repo:
- `sim_readmit.py`: the Day 30 generator as a function with drift knobs (`simulate(1000, 30)` rebuilds Day 30 exactly)
- `drift_calibration_readmit.py`: snippets 1 to 6 in one script (or paste them into `drift_calibration_readmit.ipynb`)
- `psi_snowflake.sql`: PSI of the risk score as one SQL query (Snowflake; also runs in SQLite)
- also copy `day30/readmit_transformers.py` into `day34/` (the Day 31 champion needs it to load)

Run: `conda activate ds90`, then `cd day34 && python drift_calibration_readmit.py` (about 10 seconds).
Expected: PSI after Aug 17 age 0.190, creatinine 0.173, score 0.055;
O/E 1.08 before, 0.70 after; SQL PSI = Python PSI = 0.0545.

Keep `day34/mlflow.db` and `day34/mlruns/` out of git. All data is simulated for teaching; no credentials are needed.
