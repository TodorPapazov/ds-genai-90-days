# Day 36: re-run the real-account parts of Days 32 and 33 against your Snowflake trial.
# Main path: VS Code + the Snowflake extension, then:  conda activate ds90 && python day36_snowflake.py
# Offline practice (no account, no query plans, no registry):  python day36_snowflake.py --local
# Snowsight notebook: see snippet 6, use get_active_session(), and suspend the warehouse when you finish.
# Never put credentials in this file. All data is simulated for teaching.
# ---------- snippet 1 ----------
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

# ---------- snippet 3 ----------
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

# ---------- snippet 4 ----------
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

# ---------- snippet 5 ----------
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

# ---------- snippet 6 ----------
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

# ---------- wrap up ----------
if not LOCAL:
    try:
        session.sql("ALTER WAREHOUSE DS90_XS SUSPEND").collect()
        print("Warehouse DS90_XS suspended. Also run cost_check.sql to review query history.")
    except Exception as e:
        print("Could not suspend the warehouse automatically. Run cost_check.sql yourself:", type(e).__name__)
try:
    session.close()
except Exception:
    pass
print("Day 36 script finished. Write your warehouse numbers and the query-plan takeaway in README.md.")
