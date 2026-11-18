# Day 35: week 5 recap project (simulated readmission data)

One workflow, five steps: pipeline (Days 29-30) > MLflow tracking (Day 31) > registered model with signature and @champion alias (Days 31, 33) > Snowpark scoring in local testing mode (Days 32-33) > drift and calibration monitoring (Day 34).

Files for the `day35/` folder of the practice repo:
- `recap_readmit.py`: snippets 1 to 5 in one script (or paste them into `recap_readmit.ipynb`)
- `sim_readmit.py`: the generator (`simulate(1000, 30)` rebuilds the Day 30 table exactly; drift knobs for the live batches)
- `readmit_transformers.py`: the Day 30 custom transformers (MLflow ships this file with the model)

Run: `conda activate ds90`, then `cd day35 && python recap_readmit.py` (about 10 seconds; needs scikit-learn, mlflow, snowflake-snowpark-python).
Expected: CV AUC freq 0.729 (champion), test AUC 0.745; PSI drifted age 0.242, creatinine 0.156, score 0.071;
O/E stable 1.00, drifted 0.71; Jun-Sep after Platt 0.91. Numbers land in `recap_card.json`.

Keep `day35/mlflow.db` and `day35/mlruns/` out of git. All data is simulated for teaching; no credentials are needed.
