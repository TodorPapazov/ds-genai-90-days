# Day 36: Snowpark and Snowpark ML on your real Snowflake trial: query plans, warehouse training and the Model Registry

## Concept
Day 36 starts the Snowflake hands-on week on the real trial account. Days 32 and 33 used Snowpark local testing (an in-memory emulator, no account). Today we re-run the parts that need a live warehouse: the real SQL from df.queries, query plans from explain(), Snowpark ML training on a Snowpark DataFrame, and the Snowflake Model Registry. Main path: VS Code with the Snowflake extension, Python in your ds90 conda env, talking to the trial. Short path: the same steps in a Snowsight notebook with get_active_session(). A notebook keeps the warehouse awake for the whole attached session, so it uses credits even while you read a cell. VS Code scripts let AUTO_SUSPEND (60 seconds) fire between runs, which is why that is the main path. Connection: ~/.snowflake/connections.toml or SNOWFLAKE_* env vars. Never in code, the repo, or these lesson files. The login stays in your vault. Warehouse: X-Small (1 credit per hour while running), auto-suspend 60 seconds, then ALTER WAREHOUSE DS90_XS SUSPEND when you finish. What ran on the lesson box (no ~/.snowflake, no SNOWFLAKE_* vars): simulate(1000, 30) rebuilt 2,655 stays (train 1,997 / test 658); Snowpark local testing built READMIT_ML (23 cols); connect() with no credentials exited cleanly; Snowpark ML logistic fit on pandas scored test AUC 0.734, matching Day 33; the signature was inferred (14 inputs). Local df.queries was SELECT MOCK_TEST_FAKE_QUERY(). session.sql, explain(), fit on a Snowpark DataFrame, and Registry(session) all raised NotImplementedError. What you run on the trial: write_pandas into DS90.DAY36, real queries and explain(), pipe.fit(train_sf) in the warehouse, log_model / mv.run / MODEL!PREDICT_PROBA, query history, and an optional Snowsight notebook. Compare your warehouse AUC with 0.734 (local) and 0.743 (Day 31 sklearn champion) before you pick a winner. All data is simulated for teaching. Next: Day 37, Cortex LLM functions in SQL on simulated discharge notes.

## Key numbers
- **Data:** Day 30 table: 2,655 simulated stays, 1,000 patients (seed 30), train 1,997 / test 658 by patient
- **Lesson box:** No ~/.snowflake and no SNOWFLAKE_* env vars. Snowpark 1.57.0, snowflake-ml-python 2.9.0, local testing only
- **Local check:** READMIT_ML 2,655 x 23 cols; connect() with no credentials exits cleanly; local df.queries is SELECT MOCK_TEST_FAKE_QUERY()
- **Local pandas AUC:** Snowpark ML logistic 0.734 (matches Day 33). Day 31 sklearn champion 0.743. Day 33 XGBoost 0.734
- **Not run here:** write_pandas to the trial, df.explain() plans, fit on a Snowpark DataFrame, Model Registry, SQL scoring, Snowsight notebook, query history
- **Warehouse to use:** DS90_XS, X-Small, AUTO_SUSPEND=60, AUTO_RESUME=TRUE. 1 credit per hour while running. Suspend when you finish
- **Main path:** VS Code + Snowflake extension + conda env ds90. Short path: Snowsight notebook (warehouse stays up while attached)

## Hands-on (60-90 min)
1. Install the Snowflake extension in VS Code (publisher: Snowflake) and sign in. The trial login is in your vault, not in these files.
2. Copy connections.toml.example to ~/.snowflake/connections.toml and fill in the placeholders, or set SNOWFLAKE_* env vars in your shell. Never commit the filled file.
3. Run conda activate ds90. Install snowflake-snowpark-python and snowflake-ml-python[xgboost] if they are missing. Create day36/ and copy the exercise files into it.
4. Run snippet 2 (setup_warehouse.sql) in a VS Code SQL worksheet: X-Small warehouse, auto-suspend 60 seconds, database DS90, schema DAY36.
5. From day36/, run python day36_snowflake.py (snippet 1 connects, snippet 3 loads the simulated table and prints queries and explain()). Confirm CURRENT_WAREHOUSE is DS90_XS and READMIT_ML has 2,655 rows.
6. Snippet 4 trains the Snowpark ML logistic pipeline with pipe.fit(train_sf) in the warehouse. Record the test AUC and compare it with 0.734 (local) and 0.743 (Day 31 champion).
7. Snippet 5 logs READMIT30_SNOWML version V1 in the Snowflake Model Registry, sets alias TRIAL_V1, scores with mv.run, and calls MODEL!PREDICT_PROBA from SQL.
8. Optional, short: snippet 6, the same steps in a Snowsight notebook with get_active_session(). Detach when you finish. A notebook uses credits the whole time it is attached.
9. Run snippet 7 (cost_check.sql): read query history, then ALTER WAREHOUSE DS90_XS SUSPEND.
10. Write day36/README.md: the warehouse AUC, one sentence about the query plan, and a note that the data is simulated. Then commit and push (snippet 8).

## Code
Exercise files: `exercise/day36_snowflake.py` (snippets 1, 3, 4, 5 and 6 in one script), `exercise/sim_readmit.py`, `exercise/setup_warehouse.sql`, `exercise/cost_check.sql`, `exercise/connections.toml.example`, `exercise/README.md`.

### 1. Connect from VS Code (conda env + connections.toml or env vars)
```python
# 1. Connect from VS Code: Snowpark runs in your conda env and talks to the trial.
# Install the "Snowflake" extension (publisher: Snowflake) in VS Code, then sign in.
# Preferred: ~/.snowflake/connections.toml (see connections.toml.example). Fallback: SNOWFLAKE_* env vars.
# Never put a password in this file, in a notebook, or in git. The login stays in your vault.
# In a Snowsight notebook, skip this file and use snippet 6 (get_active_session).
import os, sys
from snowflake.snowpark import Session

def connect(local=False):
    if local or "--local" in sys.argv:
        print("LOCAL TESTING mode: in-memory emulator, no account, no query plans, no Model Registry.")
        return Session.builder.config("local_testing", True).create(), True
    try:
        from snowflake.snowpark.context import get_active_session
        active = get_active_session()
        if active is not None:
            print("Using the active Snowsight / notebook session.")
            return active, False
    except Exception:
        pass
    keys = ["account", "user", "password", "role", "warehouse", "database", "schema"]
    params = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in keys if os.environ.get(f"SNOWFLAKE_{k.upper()}")}
    if params.get("account") and params.get("user"):
        session = Session.builder.configs(params).create()
        print("Connected with SNOWFLAKE_* environment variables.")
        return session, False
    name = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "ds90")
    try:
        session = Session.builder.config("connection_name", name).create()
        print(f"Connected with connections.toml connection_name={name}.")
        return session, False
    except Exception as e:
        raise SystemExit(
            "No Snowflake connection found. Add a [ds90] entry to ~/.snowflake/connections.toml "
            "(see connections.toml.example), or set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and "
            "SNOWFLAKE_PASSWORD in your shell (never in this file). Then rerun. "
            "To practice offline: python day36_snowflake.py --local"
        ) from e

session, LOCAL = connect()
if not LOCAL:
    who = session.sql(
        "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_USER() AS USER_NAME, "
        "CURRENT_ROLE() AS ROLE, CURRENT_WAREHOUSE() AS WAREHOUSE, "
        "CURRENT_DATABASE() AS DATABASE, CURRENT_SCHEMA() AS SCHEMA"
    ).collect()[0]
    print(dict(who.as_dict()))
    session.sql("USE WAREHOUSE DS90_XS").collect()
    session.sql("USE DATABASE DS90").collect()
    session.sql("USE SCHEMA DAY36").collect()
    print("Using warehouse DS90_XS, database DS90, schema DAY36.")
else:
    print("session.sql and USE WAREHOUSE are not available in local testing.")
```

### 2. Create the X-Small warehouse, database and schema (SQL)
```sql
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
```

### 3. Load simulated stays, build features, print the SQL and the query plan
```python
# 3. Load the simulated Day 30 table, build the Day 33 model-ready features, and (on a real
# account) print the SQL Snowpark generated plus the query plan. All data is simulated.
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from snowflake.snowpark import Window
from snowflake.snowpark import functions as F
from sim_readmit import simulate

raw = simulate(1000, 30)
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
                      .split(raw, groups=raw.patient_id))
split = pd.DataFrame({"PATIENT_ID": raw.patient_id, "SPLIT": "train"})
split.loc[te_idx, "SPLIT"] = "test"
up = raw.copy(); up.columns = up.columns.str.upper()
if LOCAL:
    session.create_dataframe(up).write.save_as_table("READMIT_RAW", mode="overwrite")
    session.create_dataframe(split.drop_duplicates()).write.save_as_table("PATIENT_SPLIT", mode="overwrite")
    print("Loaded READMIT_RAW and PATIENT_SPLIT with create_dataframe (local testing).")
else:
    session.write_pandas(up, "READMIT_RAW", auto_create_table=True, overwrite=True, use_logical_type=True)
    session.write_pandas(split.drop_duplicates(), "PATIENT_SPLIT", auto_create_table=True, overwrite=True)
    print("Loaded READMIT_RAW and PATIENT_SPLIT with write_pandas (bulk upload through a stage).")

w = Window.partition_by("PATIENT_ID").order_by("ADMIT_DATE")
feat = (session.table("READMIT_RAW").join(session.table("PATIENT_SPLIT"), on="PATIENT_ID")
    .with_column("PRIOR_VISITS", F.row_number().over(w) - 1)
    .with_column("DAYS_SINCE_LAST", F.datediff("day", F.lag("ADMIT_DATE").over(w), F.col("ADMIT_DATE")))
    .with_column("LOG_LOS", F.ln(F.col("LOS_DAYS") + 1))
    .with_column("LOG_CREAT", F.iff(F.col("CREATININE").is_null(), F.lit(None),
                                    F.ln(F.coalesce(F.col("CREATININE"), F.lit(0)) + 1)))
    .with_column("ICD_CHAPTER", F.substr("ICD10", 1, 1))
    .with_column("CREAT_MISSING", F.iff(F.col("CREATININE").is_null(), 1, 0))
    .with_column("HGB_MISSING", F.iff(F.col("HEMOGLOBIN").is_null(), 1, 0))
    .with_column("SODIUM_MISSING", F.iff(F.col("SODIUM").is_null(), 1, 0)))
n_train = int(feat.filter(F.col("SPLIT") == "train").count())
drg = (feat.filter(F.col("SPLIT") == "train").group_by("DRG")
       .agg(F.count("*").alias("DRG_N"))
       .with_column("DRG_SHARE", F.col("DRG_N").cast("double") / F.lit(n_train)))
feat = feat.join(drg.select("DRG", "DRG_SHARE"), on="DRG", how="left") \
           .with_column("DRG_SHARE", F.coalesce(F.col("DRG_SHARE"), F.lit(0.0)))

print("Plan is still lazy until an action runs. queries / explain need a real account.")
try:
    sqls = feat.queries.get("queries", [])
    print("SQL Snowpark generated (first query, proof of pushdown):")
    print(sqls[0] if sqls else "no queries list")
except Exception as e:
    print("df.queries not available:", type(e).__name__, str(e)[:180])
try:
    feat.explain()
except Exception as e:
    print("explain() not available here:", type(e).__name__, str(e)[:180])

feat.write.save_as_table("READMIT_ML", mode="overwrite")
ml = session.table("READMIT_ML")
n_test = int(ml.filter(F.col("SPLIT") == "test").count())
print(f"READMIT_ML {ml.count()} rows x {len(ml.columns)} cols | train {n_train} | test {n_test}")
```

### 4. Train a Snowpark ML logistic pipeline in the warehouse
```python
# 4. Snowpark ML: train a logistic pipeline ON a Snowpark DataFrame so fitting happens in the
# warehouse. On --local this falls back to pandas (the Day 33 path) because local testing cannot
# fit Snowpark ML on a Snowpark DataFrame. XGBoost is optional. Skip GroupKFold on the trial
# (it keeps the warehouse running); Day 33 already measured CV on your laptop.
from snowflake.ml.modeling.impute import SimpleImputer as SmlImputer
from snowflake.ml.modeling.preprocessing import StandardScaler as SmlScaler, OneHotEncoder as SmlOneHot
from snowflake.ml.modeling.pipeline import Pipeline as SmlPipeline
from snowflake.ml.modeling.linear_model import LogisticRegression as SmlLogReg
from sklearn.metrics import roc_auc_score

LABS = ["CREATININE", "HEMOGLOBIN", "SODIUM", "LOG_CREAT"]
NUM = ["AGE", "LOG_LOS", "SODIUM", "LOG_CREAT", "HEMOGLOBIN", "PRIOR_VISITS", "DRG_SHARE"]
CAT = ["PAYER", "ADMIT_TYPE", "DISCHARGE", "ICD_CHAPTER"]
FLAGS = ["CREAT_MISSING", "HGB_MISSING", "SODIUM_MISSING"]
NUM_S = [c + "_S" for c in NUM]
keep = ["PATIENT_ID", "ADMIT_DATE", "READMIT30"] + sorted(set(LABS + NUM + CAT + FLAGS))

train_sf = ml.filter(F.col("SPLIT") == "train").select(keep)
test_sf = ml.filter(F.col("SPLIT") == "test").select(keep)
train_pd = train_sf.to_pandas().sort_values(["PATIENT_ID", "ADMIT_DATE"]).reset_index(drop=True)
test_pd = test_sf.to_pandas().sort_values(["PATIENT_ID", "ADMIT_DATE"]).reset_index(drop=True)
for c in NUM + FLAGS:
    train_pd[c] = train_pd[c].astype(float)
    test_pd[c] = test_pd[c].astype(float)

# One-hot names are learned from a small train sample (schema only). The full fit uses them below.
oh_probe = SmlOneHot(input_cols=CAT, output_cols=CAT, handle_unknown="ignore", drop_input_cols=True)
oh_probe.fit(train_pd.head(200))
OH_COLS = [c for c in oh_probe.transform(train_pd.head(5)).columns if c not in train_pd.columns]
print("one-hot columns from 4 categoricals:", len(OH_COLS), "e.g.", OH_COLS[:3], "(names may be quoted)")

X_COLS = NUM_S + FLAGS + OH_COLS
def sml_pipe():
    lr = SmlLogReg(C=0.5, max_iter=3000, input_cols=X_COLS, label_cols=["READMIT30"], output_cols=["PRED"])
    return SmlPipeline(steps=[
        ("impute", SmlImputer(strategy="median", input_cols=LABS, output_cols=LABS)),
        ("scale", SmlScaler(input_cols=NUM, output_cols=NUM_S)),
        ("onehot", SmlOneHot(input_cols=CAT, output_cols=CAT, handle_unknown="ignore", drop_input_cols=True)),
        ("clf", lr)])

pipe = sml_pipe()
if LOCAL:
    print("LOCAL: fitting on a pandas DataFrame (training runs on this computer).")
    pipe.fit(train_pd)
    out = pipe.predict_proba(test_pd)
    prob_col = [c for c in out.columns if "PREDICT_PROBA" in c.upper() and c.upper().endswith("1")][0]
    risk = out[prob_col].to_numpy(float)
    auc = roc_auc_score(test_pd.READMIT30, risk)
    print(f"Snowpark ML logistic, local pandas fit: test AUC {auc:.3f} (Day 33 was 0.734)")
else:
    print("REAL ACCOUNT: fitting on a Snowpark DataFrame (training runs in the warehouse).")
    pipe.fit(train_sf)
    pred = pipe.predict_proba(test_sf)
    pred_pd = pred.to_pandas()
    prob_col = [c for c in pred_pd.columns if "PREDICT_PROBA" in c.upper() and c.upper().endswith("1")][0]
    risk = pred_pd[prob_col].to_numpy(float)
    y = pred_pd["READMIT30"].to_numpy() if "READMIT30" in pred_pd.columns else test_pd.READMIT30.to_numpy()
    try:
        from snowflake.ml.modeling.metrics import roc_auc_score as sf_auc
        auc = float(sf_auc(df=pred, y_true_col_names="READMIT30", y_score_col_names=prob_col))
        print(f"Snowpark ML logistic, trained in the warehouse: test AUC {auc:.3f} (snowflake.ml metric)")
    except Exception as e:
        auc = roc_auc_score(y, risk)
        print(f"Snowpark ML logistic, trained in the warehouse: test AUC {auc:.3f} (sklearn metric; sf metric: {e})")
    print("Day 33 local pandas fit was 0.734; Day 31 sklearn champion was 0.743. Compare before you pick a winner.")
```

### 5. Log the model in the Snowflake Model Registry and score from SQL
```python
# 5. Snowflake Model Registry: log the warehouse-trained pipeline, set an alias, batch-score
# with mv.run, and call the model from SQL. Local testing cannot do this; the snippet then
# only infers the signature you would store.
from snowflake.ml.model import model_signature
import pandas as pd

sig = model_signature.infer_signature(
    input_data=test_pd[NUM + FLAGS + CAT].head(50),
    output_data=pd.DataFrame({"PREDICT_PROBA_1": risk[:50]}))
print("signature:", len(sig.inputs), "inputs, e.g.",
      [f"{s.name}:{s.as_snowpark_type().__class__.__name__}" for s in sig.inputs[:4]],
      "| output", sig.outputs[0].name)

if LOCAL:
    print("LOCAL: Registry(session), log_model, mv.run and MODEL!PREDICT_PROBA need a real account. Skipping.")
else:
    from snowflake.ml.registry import Registry
    db, sch = session.get_current_database().strip('"'), session.get_current_schema().strip('"')
    reg = Registry(session=session, database_name=db, schema_name=sch)
    mv = reg.log_model(
        pipe,
        model_name="READMIT30_SNOWML",
        version_name="V1",
        sample_input_data=train_sf.select(NUM + FLAGS + CAT).limit(50),
        metrics={"test_auc": float(auc), "source": "day36_trial"},
        comment="Snowpark ML logistic, simulated readmissions, Day 36 trial",
    )
    print(reg.show_models())
    m = reg.get_model("READMIT30_SNOWML")
    print(m.show_versions())
    m.default = "V1"
    scores = mv.run(test_sf, function_name="predict_proba")
    scores.write.save_as_table("READMIT_SCORES_ML", mode="overwrite")
    print("Wrote READMIT_SCORES_ML. Sample:")
    session.table("READMIT_SCORES_ML").select("PATIENT_ID", "ADMIT_DATE", prob_col).limit(5).show()
    # Analysts can score the same model from SQL. Column order must match the signature.
    session.sql("""
        SELECT PATIENT_ID, ADMIT_DATE,
               READMIT30_SNOWML!PREDICT_PROBA(AGE, LOG_LOS, SODIUM, LOG_CREAT, HEMOGLOBIN,
                   PRIOR_VISITS, DRG_SHARE, CREAT_MISSING, HGB_MISSING, SODIUM_MISSING,
                   PAYER, ADMIT_TYPE, DISCHARGE, ICD_CHAPTER) AS P
        FROM READMIT_ML
        WHERE SPLIT = 'test'
        LIMIT 5
    """).show()
    session.sql("ALTER MODEL READMIT30_SNOWML VERSION V1 SET ALIAS = TRIAL_V1").collect()
    print("Alias TRIAL_V1 is set. Day 31's sklearn champion is still the local AUC leader at 0.743.")
```

### 6. Optional: the same steps in a Snowsight notebook
```python
# 6. Optional: the same steps in a Snowsight notebook (short path, uses credits while attached).
# Open Snowsight, create a Python notebook, attach warehouse DS90_XS, database DS90, schema DAY36.
# Paste snippets 3 to 5, but replace the connect() call with the two lines below.
# A notebook keeps the warehouse awake for the whole session, so it uses credits the entire time
# the notebook is attached, not only while a cell is running. Auto-suspend starts only after you
# detach or close it and the 60-second idle timer fires. That is why VS Code plus your conda
# env is the main path: the warehouse can suspend between script runs.
#
# from snowflake.snowpark.context import get_active_session
# session = get_active_session()
# LOCAL = False
#
# write_pandas works in a notebook the same way. After you finish, run cost_check.sql and
# ALTER WAREHOUSE DS90_XS SUSPEND so you are not waiting on auto-suspend.
print("Snowsight notebook path: use get_active_session(), attach DS90_XS, suspend when you are done.")
```

### 7. Review query history, then suspend the warehouse (SQL)
```sql
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
```

### 8. Push to GitHub
```bash
cd ~/ds-genai-90-days
printf 'day36/connections.toml\nday36/.env\n' >> .gitignore
git add .gitignore day36
git commit -m "Day 36: Snowpark and Snowpark ML on the real Snowflake trial, query plans and Model Registry"
git push
```

## What ran, and what did not (note)
The lesson computer has no Snowflake connection, and none was created. Local testing rebuilt the Day 30 table, built features, fit the logistic pipeline on pandas (test AUC 0.734), and confirmed that connect() exits cleanly without credentials. Real query plans, warehouse training, the Model Registry, SQL scoring, the Snowsight notebook path, and query history were not run. Those results come from your trial. Write them in your README.

## Tomorrow (Day 37)
Stay on the trial and use Cortex LLM functions in SQL on simulated discharge notes.

## Printables
- Cheat sheet: `day36_v1_cheat_sheet.png`
- Infographic: `day36_v2_infographic.png`
- Code worksheet: `day36_v3_worksheet.png`
- Practice quiz: `day36_v4_quiz.png`
- PDF: `day36_lesson.pdf`

## Podcast script
(Audio: `day36_podcast.mp3`)

Welcome to day thirty six of the Data Analyst to Data Scientist plan. This is the first day of a hands-on Snowflake week, and we finally use your real trial account. On days thirty two and thirty three we learned Snowpark and Snowpark ML in local testing mode, an in-memory emulator that needs no account. That was the right way to learn the API. But an emulator cannot show you a real query plan, it cannot train a model inside a warehouse, and it cannot put a model in the Snowflake Model Registry. Those parts need a live account, and that is today's job.

The main path is VS Code, with the Snowflake extension, and Python running in your conda environment on your own computer. You browse tables and run SQL in the extension, and you run the Python file in the terminal, talking to Snowflake over the network. There is also a short path in a Snowsight notebook, where the same code can run if you swap in get active session. That is handy, but a notebook keeps the warehouse awake for the whole time it is attached, so it uses credits even while you are reading a cell. In VS Code, the warehouse can auto-suspend between script runs. That is why VS Code is the main path.

A word about cost, because a trial is finite. We will use an extra small warehouse with auto-suspend set to sixty seconds and auto-resume turned on. An extra small warehouse uses one credit per hour while it is running, and it should stop shortly after you stop querying. When you finish, run the cost check SQL and suspend the warehouse yourself, so you are not waiting on the timer.

How do you connect without leaking a password? Put the login in a connections file under your home directory, or in environment variables in your shell. The lesson ships an example file with placeholders only. Never put the password in the Python script, in a notebook you might commit, or in the git repo. Your vault already has the trial login.

Once you are connected, the work should look familiar. We rebuild the simulated thirty day readmission table, two thousand six hundred fifty five stays from one thousand patients, with the same patient split as the rest of the week. We load those rows with write pandas, and we build the same model-ready features in Snowpark: prior visits, logs, missing flags, and a diagnosis share learned from train rows only. Because this is a real account, you can print the SQL Snowpark generated, and you can call explain to see the query plan. That is the proof of pushdown: the heavy row work is translated to SQL and run where the data lives.

Then we train. Snowpark ML uses a scikit-learn style pipeline, but you name the columns. When you fit on a pandas frame, training happens on your laptop, which is what we did on day thirty three, with a test AUC of point seven three four, a little under the day thirty one champion at point seven four three. When you fit on a Snowpark DataFrame, training happens in the warehouse, and the rows never have to leave Snowflake. After that, you log the model in the Snowflake Model Registry, give it a version and an alias, score a batch with run, and even call the model from a SQL worksheet.

I want to be honest about what I could run for you. The lesson computer has no Snowflake connection, and I did not set one up. I did run the script in local testing. The table rebuilt correctly, the features landed, the logistic pipeline fit on pandas and again scored point seven three four, and the connection helper exited cleanly when no credentials were present. I could not read a real query plan, train inside the warehouse, or use the Model Registry. Those results will come from your run. Write them in your README, and compare them with the local numbers before you pick a winner.

Your exercise is in the day thirty six folder of your repo. Tomorrow, day thirty seven, we stay on the trial and use Cortex large language model functions in SQL, on simulated discharge notes. See you tomorrow.
