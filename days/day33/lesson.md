# Day 33: Snowpark ML: sklearn-style preprocessing, training and the Snowflake Model Registry on 30-day readmission

## Concept
Day 32 built readmission features in Snowpark. The next skill healthcare data teams ask for is training and versioning the model where the data lives, so PHI never lands on a laptop. Snowpark ML (pip install snowflake-ml-python) gives you scikit-learn and XGBoost estimators with Snowflake manners: you name columns with input_cols, label_cols and output_cols instead of passing a numeric array, and every call returns a DataFrame with new columns. Fit on a Snowpark DataFrame and training runs inside Snowflake; fit on a pandas DataFrame and the same object runs locally. The Snowflake Model Registry (Registry(session).log_model) stores versioned models with a signature and metrics, and you score with mv.run(df, function_name="predict_proba") or straight from SQL. Honest limits: with no account on the box (snowflake-ml-python 2.9.0, Snowpark local testing), Snowpark ML on a Snowpark DataFrame stopped with SQL Text Expression is not supported / Session.sql is not supported, and so did Registry(session). So we built the features in Snowpark local testing (READMIT_ML, 23 columns: log LOS and creatinine, ICD chapter, missing flags, prior visits, train-only DRG share), pulled train (1,997) and test (658) stays to pandas with the same patient split as Days 30 to 32, and fed them to the real Snowpark ML classes. The lesson includes the real-account version with connection values from environment variables. Every learned number matched plain scikit-learn exactly (imputer medians, scaler means, scaled values, one-hot values, coefficients: max diff 0; final risks within 2e-16). A Snowpark ML Pipeline (impute, scale, one-hot, logistic regression C=0.5, 30 inputs) scored test AUC 0.734; Snowpark ML XGBoost also 0.734; the Day 31 champion 0.743 on the same stays. Model choice uses GroupKFold on train: logistic 0.718 (sd 0.033), XGBoost 0.684, champion 0.724, so the Day 31 champion stays. The two models correlate 0.920 but share only 44 of the 66 highest-risk stays, which is why a registry with versions and a clear default matters. Gotchas from the run: xgboost is an optional extra since snowflake-ml-python 2.0; without input_cols the classifier grabbed PATIENT_ID and ADMIT_DATE and crashed on the date; one-hot names come back wrapped in double quotes ('"PAYER_Commercial"'); Pipeline.to_sklearn() uses column positions and broke on the test rows, so we wrote a plain sklearn twin that gives identical risks. Locally we also inferred the registry signature (14 typed inputs), registered the twin in MLflow as readmit30_snowml@challenger, and scored all 658 test stays with a vectorized UDF in local testing (max diff vs local 1e-16). All data is simulated for teaching.

## Key numbers
- **Data:** Day 30 table: 2,655 simulated stays, 1,000 patients, train 1,997 / test 658 by patient (same split as Days 30 to 32)
- **Engine:** snowflake-ml-python 2.9.0 + Snowpark 1.57.0 local testing; Snowpark ML fit on pandas (local); no Snowflake account
- **What needs an account:** Snowpark ML fit on a Snowpark DataFrame, Registry(session), mv.run, SQL MODEL!PREDICT_PROBA (all hit local testing "not supported")
- **Parity with sklearn:** medians, means, scaled values, one-hot, coefficients: max diff 0; risks 2.2e-16; UDF scores 1.1e-16
- **Test AUC (658 stays):** Snowpark ML logistic 0.734, Snowpark ML XGBoost 0.734, Day 31 champion 0.743
- **GroupKFold CV AUC:** logistic 0.718 (sd 0.033), XGBoost 0.684 (sd 0.030), champion 0.724 (Day 31): champion stays
- **Agreement:** risk correlation 0.920; shared top 10%: 44 of 66 stays; top 10% cut 0.378
- **Worklist:** mean risk Medicaid 0.231, Commercial 0.183, Medicare 0.182 (actual 0.134), Self-pay 0.155
- **Gotchas:** optional xgboost; set input_cols; quoted one-hot names; Pipeline.to_sklearn() broke; UDF next to other columns after filter gave 1156 rows / 498 NULLs in local testing

## Hands-on (60-90 min)
1. In your repo, create day33/, run conda activate ds90, pip install "snowflake-ml-python[xgboost]" mlflow, copy day30/readmit_transformers.py into day33/ (the Day 31 champion needs it) and open day33/snowpark_ml_readmit.ipynb in VS Code. The code reads ../day30/readmit_raw.csv and the Day 31 registry in ../day31/mlflow.db.
2. Run snippet 1: local testing Session, READMIT_ML features in Snowpark with the same patient split, then the honest check: Snowpark ML on the Snowpark DataFrame and Registry(session). Write down each error.
3. Run snippet 2: Snowpark ML SimpleImputer, StandardScaler and OneHotEncoder with input_cols/output_cols, fit on train only. Compare every learned number with plain sklearn and notice the double-quoted one-hot names.
4. Run snippet 3: the Snowpark ML Pipeline with LogisticRegression (C=0.5). Compare test AUC with the Day 31 champion, try pipe.to_sklearn() on the test rows, then build the plain sklearn twin and confirm identical risks.
5. Still in snippet 3: train the Snowpark ML XGBClassifier and run the GroupKFold race on train only. Decide with CV, not with test.
6. Run snippet 4: infer the signature the Snowflake registry would store, then log and register the twin in MLflow as readmit30_snowml with alias @challenger, reload it and check the risks.
7. Run snippet 5: register a vectorized UDF that scores inside the session, write READMIT_SCORES_ML, check the scores match local predictions, build the payer worklist and count the shared top 10% stays with the champion.
8. Optional, on your own account: put connection values in environment variables (never in code) and run snippet 6: fit on a Snowpark DataFrame, Registry.log_model, mv.run and the SQL MODEL!PREDICT_PROBA call.
9. Write day33/README.md: what ran locally and what needs an account, the parity checks, the AUCs, the champion decision, the gotchas, and that the data is simulated. Add day33/mlflow.db, day33/mlruns/ and .env to .gitignore.
10. Commit and push (snippet 7): git add .gitignore day33, git commit -m "Day 33: ...", git push.

## Code
### 1. Snowpark features (local testing) and what Snowpark ML can do without an account
```python
# Day 33 setup: the same Day 30 table and patient split, the model-ready columns built in Snowpark
# (local testing, as on Day 32), and an honest check of what Snowpark ML can do without an account.
import os, time, warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from snowflake.snowpark import Session, Window
from snowflake.snowpark import functions as F
import snowflake.ml.version

session = Session.builder.config("local_testing", True).create()     # in-memory emulator, no account
raw = pd.read_csv("../day30/readmit_raw.csv", parse_dates=["admit_date"], dtype={"drg": str, "icd10": str})
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
                      .split(raw, groups=raw.patient_id))               # identical split to Days 30-32
split = pd.DataFrame({"PATIENT_ID": raw.patient_id, "SPLIT": "train"})
split.loc[te_idx, "SPLIT"] = "test"
up = raw.copy(); up.columns = up.columns.str.upper()
session.create_dataframe(up).write.save_as_table("READMIT_RAW", mode="overwrite")
session.create_dataframe(split.drop_duplicates()).write.save_as_table("PATIENT_SPLIT", mode="overwrite")

# Snowpark ML has no FunctionTransformer or custom transformers, so the row-level features
# (log, ICD chapter, missing flags, per-patient history, train-only DRG share) are built here in Snowpark.
w = Window.partition_by("PATIENT_ID").order_by("ADMIT_DATE")
feat = (session.table("READMIT_RAW").join(session.table("PATIENT_SPLIT"), on="PATIENT_ID")
    .with_column("PRIOR_VISITS", F.row_number().over(w) - 1)
    .with_column("DAYS_SINCE_LAST", F.datediff("day", F.lag("ADMIT_DATE").over(w), F.col("ADMIT_DATE")))
    .with_column("LOG_LOS", F.ln(F.col("LOS_DAYS") + 1))
    .with_column("LOG_CREAT", F.iff(F.col("CREATININE").is_null(), F.lit(None),     # local testing ln() fails on
                                    F.ln(F.coalesce(F.col("CREATININE"), F.lit(0)) + 1)))   # NULL, so keep NULL by hand
    .with_column("ICD_CHAPTER", F.substr("ICD10", 1, 1))
    .with_column("CREAT_MISSING", F.iff(F.col("CREATININE").is_null(), 1, 0))
    .with_column("HGB_MISSING", F.iff(F.col("HEMOGLOBIN").is_null(), 1, 0))
    .with_column("SODIUM_MISSING", F.iff(F.col("SODIUM").is_null(), 1, 0)))
n_train = int(feat.filter(F.col("SPLIT") == "train").count())
drg = (feat.filter(F.col("SPLIT") == "train").group_by("DRG")
       .agg(F.count("*").alias("DRG_N"))
       .with_column("DRG_SHARE", F.col("DRG_N").cast("double") / F.lit(n_train)))   # train rows only
feat = feat.join(drg.select("DRG", "DRG_SHARE"), on="DRG", how="left") \
           .with_column("DRG_SHARE", F.coalesce(F.col("DRG_SHARE"), F.lit(0.0)))
feat.write.save_as_table("READMIT_ML", mode="overwrite")
ml = session.table("READMIT_ML")
print("snowflake-ml-python", snowflake.ml.version.VERSION, "| READMIT_ML", ml.count(), "rows x", len(ml.columns),
      "cols | train", n_train, "| test", ml.filter(F.col("SPLIT") == "test").count())

# Honest check: Snowpark ML on a Snowpark DataFrame, and the Model Registry, in local testing mode.
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.registry import Registry
tries = {"StandardScaler.fit(Snowpark df)": lambda: StandardScaler(input_cols=["AGE"], output_cols=["AGE_S"]).fit(ml),
         "LogisticRegression.fit(Snowpark df)": lambda: LogisticRegression(input_cols=["AGE"], label_cols=["READMIT30"]).fit(ml),
         "Registry(session)": lambda: Registry(session=session)}
limits = {}
for name, fn in tries.items():
    try:
        fn(); limits[name] = "ok"
    except NotImplementedError as e:
        limits[name] = str(e)
    print(f"{name:36s} -> {limits[name]}")
```

### 2. Snowpark ML preprocessing: input_cols / output_cols, parity with sklearn
```python
# Snowpark ML preprocessing, sklearn-style: input_cols / output_cols instead of column positions.
# Without an account we pass pandas DataFrames: Snowpark ML then fits locally with the same API.
from snowflake.ml.modeling.impute import SimpleImputer as SmlImputer
from snowflake.ml.modeling.preprocessing import StandardScaler as SmlScaler, OneHotEncoder as SmlOneHot
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

LABS = ["CREATININE", "HEMOGLOBIN", "SODIUM", "LOG_CREAT"]
NUM = ["AGE", "LOG_LOS", "SODIUM", "LOG_CREAT", "HEMOGLOBIN", "PRIOR_VISITS", "DRG_SHARE"]
CAT = ["PAYER", "ADMIT_TYPE", "DISCHARGE", "ICD_CHAPTER"]
FLAGS = ["CREAT_MISSING", "HGB_MISSING", "SODIUM_MISSING"]
keep = ["PATIENT_ID", "ADMIT_DATE", "READMIT30"] + sorted(set(LABS + NUM + CAT + FLAGS))
train_pd = ml.filter(F.col("SPLIT") == "train").select(keep).to_pandas().sort_values(["PATIENT_ID", "ADMIT_DATE"]).reset_index(drop=True)
test_pd = ml.filter(F.col("SPLIT") == "test").select(keep).to_pandas().sort_values(["PATIENT_ID", "ADMIT_DATE"]).reset_index(drop=True)
for c in NUM + FLAGS: train_pd[c] = train_pd[c].astype(float); test_pd[c] = test_pd[c].astype(float)
print("to_pandas: train", train_pd.shape, "| test", test_pd.shape, "| missing creatinine in train", int(train_pd.CREATININE.isna().sum()))

imp = SmlImputer(strategy="median", input_cols=LABS, output_cols=LABS).fit(train_pd)        # learn on TRAIN only
sc = SmlScaler(input_cols=NUM, output_cols=[c + "_S" for c in NUM]).fit(imp.transform(train_pd))
oh = SmlOneHot(input_cols=CAT, output_cols=CAT, handle_unknown="ignore", drop_input_cols=True).fit(train_pd)
tr_s = sc.transform(imp.transform(test_pd))                                                 # apply to TEST
tr_oh = oh.transform(test_pd)
print("imputer medians (train):", {c: round(float(v), 3) for c, v in zip(LABS, imp.to_sklearn().statistics_)})
OH_COLS = [c for c in tr_oh.columns if c not in test_pd.columns]
print("one-hot columns from 4 categoricals:", len(OH_COLS), "e.g.", OH_COLS[:3], "(note the double quotes)")

# Same steps in plain sklearn: do the learned numbers match?
sk_imp = SimpleImputer(strategy="median").fit(train_pd[LABS])
tr2 = train_pd.copy(); tr2[LABS] = sk_imp.transform(train_pd[LABS])
sk_sc = StandardScaler().fit(tr2[NUM])
te2 = test_pd.copy(); te2[LABS] = sk_imp.transform(test_pd[LABS])
parity = {"imputer medians": float(np.abs(imp.to_sklearn().statistics_ - sk_imp.statistics_).max()),
          "scaler means": float(np.abs(sc.to_sklearn().mean_ - sk_sc.mean_).max()),
          "scaled test values": float(np.abs(tr_s[[c + "_S" for c in NUM]].to_numpy() - sk_sc.transform(te2[NUM])).max()),
          "one-hot test values": float(np.abs(tr_oh[OH_COLS].to_numpy(float)
                                       - OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train_pd[CAT]).transform(test_pd[CAT])).max())}
for k, v in parity.items(): print(f"Snowpark ML vs sklearn, {k:20s} max diff {v:.1e}")
print("to_sklearn() gives:", type(sc.to_sklearn()).__name__, "| scaled AGE in test: mean",
      round(float(tr_s.AGE_S.mean()), 3), "sd", round(float(tr_s.AGE_S.std()), 3), "(train is 0 and 1)")
```

### 3. Snowpark ML Pipeline, logistic vs XGBoost vs the Day 31 champion, GroupKFold
```python
# A full Snowpark ML Pipeline (impute -> scale -> one-hot -> logistic regression), trained on the train
# stays and scored on the test stays, next to the Day 31 MLflow champion and a Snowpark ML XGBoost model.
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from snowflake.ml.modeling.pipeline import Pipeline as SmlPipeline
from snowflake.ml.modeling.linear_model import LogisticRegression as SmlLogReg
from snowflake.ml.modeling.xgboost import XGBClassifier as SmlXGB

NUM_S = [c + "_S" for c in NUM]
def sml_pipe(clf):
    return SmlPipeline(steps=[
        ("impute", SmlImputer(strategy="median", input_cols=LABS, output_cols=LABS)),
        ("scale", SmlScaler(input_cols=NUM, output_cols=NUM_S)),
        ("onehot", SmlOneHot(input_cols=CAT, output_cols=CAT, handle_unknown="ignore", drop_input_cols=True)),
        ("clf", clf)])
X_COLS = NUM_S + FLAGS + OH_COLS        # name the model inputs, or the classifier also grabs IDs and dates
lr = SmlLogReg(C=0.5, max_iter=3000, input_cols=X_COLS, label_cols=["READMIT30"], output_cols=["PRED"])
xgb = SmlXGB(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=33,
             input_cols=X_COLS, label_cols=["READMIT30"], output_cols=["PRED"])

t0 = time.perf_counter()
pipe_lr = sml_pipe(lr); pipe_lr.fit(train_pd)               # pandas in: fits locally, same code as on Snowflake
t_fit = time.perf_counter() - t0
out = pipe_lr.predict_proba(test_pd)
prob_col = [c for c in out.columns if "PREDICT_PROBA" in c.upper() and c.upper().endswith("1")][0]
risk_sml = out[prob_col].to_numpy(float)
y_test = test_pd.READMIT30.to_numpy()
auc_sml = roc_auc_score(y_test, risk_sml)
print(f"Snowpark ML pipeline (logistic, C=0.5): fit {t_fit:.2f} s | test AUC {auc_sml:.3f} | output column {prob_col}")

# Gotcha: to_sklearn() on the WHOLE pipeline returns column-POSITION steps that break after the first step.
try:
    pipe_lr.to_sklearn().predict_proba(test_pd); print("pipeline to_sklearn(): works")
except ValueError as e:
    print("pipeline to_sklearn() fails on the test rows:", str(e)[:45])
# Fix: write the same recipe once in plain sklearn (named columns) and prove it gives the same risks.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
sk_pipe = Pipeline([("prep", ColumnTransformer([
              ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
              ("flags", "passthrough", FLAGS),
              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])),
          ("clf", LogisticRegression(C=0.5, max_iter=3000))]).fit(train_pd, train_pd.READMIT30)
risk_sk = sk_pipe.predict_proba(test_pd)[:, 1]
print(f"plain sklearn twin: test AUC {roc_auc_score(test_pd.READMIT30, risk_sk):.3f} | max risk diff vs Snowpark ML "
      f"{np.abs(risk_sk - risk_sml).max():.1e} | coefficients max diff "
      f"{np.abs(sk_pipe[-1].coef_ - pipe_lr.to_sklearn()[-1].coef_).max():.1e}")
mlflow.set_tracking_uri("sqlite:///../day31/mlflow.db")         # Day 31 registry, read only
champ = mlflow.sklearn.load_model("models:/readmit30_pipeline@champion")
raw_test = raw.iloc[te_idx].sort_values(["patient_id", "admit_date"]).reset_index(drop=True)
risk_champ = champ.predict_proba(raw_test.drop(columns="readmit30"))[:, 1]
auc_champ = roc_auc_score(raw_test.readmit30, risk_champ)
same_rows = bool((raw_test.patient_id.to_numpy() == test_pd.PATIENT_ID.to_numpy()).all())
print(f"Day 31 champion on the same {len(raw_test)} test stays: AUC {auc_champ:.3f} | same row order: {same_rows}"
      f" | correlation of risks {np.corrcoef(risk_sml, risk_champ)[0, 1]:.3f}")

pipe_xgb = sml_pipe(xgb); pipe_xgb.fit(train_pd)
o2 = pipe_xgb.predict_proba(test_pd)
risk_xgb = o2[[c for c in o2.columns if "PREDICT_PROBA" in c.upper() and c.upper().endswith("1")][0]].to_numpy(float)
auc_xgb = roc_auc_score(y_test, risk_xgb)
print(f"Snowpark ML XGBClassifier (200 trees, depth 3): test AUC {auc_xgb:.3f}")

# Honest model choice uses CV on TRAIN only (test is for the final check). GroupKFold keeps a patient in one fold.
cv = {"logistic": [], "xgboost": []}
for k, (a, b) in enumerate(GroupKFold(5).split(train_pd, groups=train_pd.PATIENT_ID)):
    fa, fb = train_pd.iloc[a].reset_index(drop=True), train_pd.iloc[b].reset_index(drop=True)
    for name, clf in [("logistic", lr), ("xgboost", xgb)]:
        p = sml_pipe(clf); p.fit(fa); o = p.predict_proba(fb)
        cv[name].append(roc_auc_score(fb.READMIT30, o[[c for c in o.columns if "PREDICT_PROBA" in c.upper() and c.upper().endswith("1")][0]]))
for name, v in cv.items(): print(f"GroupKFold AUC {name:9s} {np.mean(v):.3f}  sd {np.std(v):.3f}  folds {np.round(v, 3).tolist()}")
```

### 4. Registry: the signature, and an MLflow @challenger stand-in
```python
# The Snowflake Model Registry needs a real account (see the real-account snippet). Locally we do the two
# parts that do run: infer the signature the registry would store, and register to_sklearn() in MLflow.
from snowflake.ml.model import model_signature
sig = model_signature.infer_signature(input_data=test_pd[NUM + FLAGS + CAT].head(50),
                                      output_data=pd.DataFrame({"PREDICT_PROBA_1": risk_sml[:50]}))
print("signature the registry would store:", len(sig.inputs), "inputs, e.g.",
      [f"{s.name}:{s.as_snowpark_type().__class__.__name__}" for s in sig.inputs[:3]], "| output", sig.outputs[0].name)

from mlflow.models import infer_signature
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("day33_snowpark_ml")
with mlflow.start_run(run_name="snowml_logistic") as run:
    mlflow.set_tags({"library": "snowflake-ml-python " + snowflake.ml.version.VERSION, "trained_on": "pandas (local)",
                     "data": "simulated readmissions, Day 30 split"})
    mlflow.log_params({"C": 0.5, "features": len(X_COLS)})
    mlflow.log_metrics({"cv_auc": float(np.mean(cv["logistic"])), "cv_auc_sd": float(np.std(cv["logistic"])),
                        "test_auc": float(auc_sml)})
    info = mlflow.sklearn.log_model(sk_pipe, name="model", registered_model_name="readmit30_snowml",
                                    input_example=test_pd.head(3), signature=infer_signature(test_pd.head(50), risk_sml[:50]),
                                    skops_trusted_types=["numpy.dtype"])   # OneHotEncoder's dtype; allow-list it
client = MlflowClient()
ver = info.registered_model_version
client.set_registered_model_alias("readmit30_snowml", "challenger", ver)
back = mlflow.sklearn.load_model("models:/readmit30_snowml@challenger")
print(f"MLflow: readmit30_snowml version {ver} @challenger | reloaded risks identical:",
      np.allclose(back.predict_proba(test_pd)[:, 1], risk_sk))
decision = "keep Day 31 champion" if np.mean(cv["logistic"]) < 0.724 else "promote"
print(f"decision: CV AUC Snowpark ML logistic {np.mean(cv['logistic']):.3f} vs Day 31 champion 0.724 -> {decision}")
```

### 5. Scoring next to the data with a vectorized UDF, payer worklist
```python
# Scoring where the data lives: a vectorized UDF runs the model next to the table (the registry does this
# for you on a real account with mv.run). Local testing can run UDFs, so we check the scores match.
from snowflake.snowpark.types import FloatType, PandasSeriesType, StringType
from snowflake.snowpark.functions import pandas_udf
IN = NUM + FLAGS + CAT
types = [StringType() if c in CAT else FloatType() for c in IN]
model = sk_pipe                                             # captured by the UDF, like a model file in a stage

@pandas_udf(name="READMIT_RISK", replace=True, return_type=PandasSeriesType(FloatType()),
            input_types=[PandasSeriesType(t) for t in types], session=session)
def readmit_risk(*cols):                    # Snowflake passes one pandas Series per column, per batch;
    batch = isinstance(cols[0], pd.Series)  # local testing passes one row of plain values at a time
    df = pd.concat(cols, axis=1) if batch else pd.DataFrame([cols])
    df.columns = IN
    for c in IN:
        if c not in CAT: df[c] = df[c].astype(float)
    p = model.predict_proba(df)[:, 1]
    return pd.Series(p) if batch else float(p[0])

# Local testing gotcha: a UDF next to other columns right after filter() misaligned rows. Count it, then use a table.
odd = ml.filter(F.col("SPLIT") == "test").select("PATIENT_ID", readmit_risk(*[F.col(c) for c in IN]).alias("RISK")).to_pandas()
print(f"UDF after filter (local testing): {len(odd)} rows, {int(odd.RISK.isna().sum())} NULL risks (expected 658, 0)")
ml.filter(F.col("SPLIT") == "test").write.save_as_table("READMIT_TEST", mode="overwrite")
t0 = time.perf_counter()
scored = (session.table("READMIT_TEST")
            .select("PATIENT_ID", "ADMIT_DATE", "PAYER", "READMIT30", readmit_risk(*[F.col(c) for c in IN]).alias("RISK")))
scored.write.save_as_table("READMIT_SCORES_ML", mode="overwrite")
t_udf = time.perf_counter() - t0
sc_pd = session.table("READMIT_SCORES_ML").to_pandas().sort_values(["PATIENT_ID", "ADMIT_DATE"]).reset_index(drop=True)
print(f"UDF scored {len(sc_pd)} test stays in {t_udf:.2f} s | max diff vs local predict_proba "
      f"{np.abs(sc_pd.RISK.to_numpy() - risk_sk).max():.1e}")
cut = float(np.quantile(risk_sml, 0.9))
work = (session.table("READMIT_SCORES_ML").with_column("TOP10", F.iff(F.col("RISK") >= cut, 1, 0))
        .group_by("PAYER").agg(F.count("*").alias("N"), F.avg("RISK").alias("MEAN_RISK"),
                               F.sum("TOP10").alias("TOP10_N"), F.avg("READMIT30").alias("ACTUAL"))
        .sort(F.col("MEAN_RISK").desc()))
print(f"top-10% cut {cut:.3f} | worklist by payer (Snowpark ML model, test stays):")
for r in work.collect():
    print(f"  {r.PAYER:10s} N {int(r.N):3d}  mean risk {float(r.MEAN_RISK):.3f}  top10 {int(r.TOP10_N):2d}  actual {float(r.ACTUAL):.3f}")
top_sml = set(np.argsort(-risk_sml)[:66]); top_ch = set(np.argsort(-risk_champ)[:66])
print("top 66 (10%) stays shared by Snowpark ML model and Day 31 champion:", len(top_sml & top_ch))
```

### 6. Your real Snowflake account: train in Snowflake + Model Registry (env vars, not run here)
```python
# Day 33 on YOUR Snowflake account (not run on the lesson box: it has no Snowflake credentials).
# pip install "snowflake-ml-python[xgboost]"   (xgboost is optional since snowflake-ml-python 2.0)
# Set in your shell or a git-ignored .env, never in code: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
# (or SSO), SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
import os
from snowflake.snowpark import Session, functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.metrics import roc_auc_score
keys = ["account", "user", "password", "role", "warehouse", "database", "schema"]
params = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in keys if os.environ.get(f"SNOWFLAKE_{k.upper()}")}
session = Session.builder.configs(params).create()
# Build READMIT_ML with the snippet 1 Snowpark code (session.write_pandas(up, "READMIT_RAW", ...) to load).
ml = session.table("READMIT_ML")
train_sf = ml.filter(F.col("SPLIT") == "train")          # stays in Snowflake: no to_pandas()
test_sf = ml.filter(F.col("SPLIT") == "test")

pipe = sml_pipe(lr)                                       # the same snippet 3 Pipeline object
pipe.fit(train_sf)                                        # Snowpark DataFrame in: trains inside Snowflake
pred = pipe.predict_proba(test_sf)                        # lazy Snowpark DataFrame with PREDICT_PROBA_1
auc = roc_auc_score(df=pred, y_true_col_names="READMIT30", y_score_col_names="PREDICT_PROBA_1")
print("test AUC computed in the warehouse:", round(auc, 3))

reg = Registry(session=session, database_name=params["database"], schema_name=params["schema"])
mv = reg.log_model(pipe, model_name="READMIT30_SNOWML", version_name="V1",
                   sample_input_data=train_sf.select(NUM + FLAGS + CAT).limit(50),   # sets the signature
                   metrics={"test_auc": float(auc), "cv_auc": 0.718},
                   comment="Snowpark ML logistic, simulated readmissions, Day 33")
print(reg.show_models())                                  # every model in this schema
m = reg.get_model("READMIT30_SNOWML"); print(m.show_versions())
m.default = "V1"                                          # what callers get with no version named
scores = mv.run(test_sf, function_name="predict_proba")   # batch scoring inside Snowflake
scores.select("PATIENT_ID", "ADMIT_DATE", "PREDICT_PROBA_1").write.save_as_table("READMIT_SCORES_ML", mode="overwrite")
# The same model from plain SQL, for analysts and dashboards:
session.sql("""SELECT PATIENT_ID, READMIT30_SNOWML!PREDICT_PROBA(AGE, LOG_LOS, SODIUM, LOG_CREAT, HEMOGLOBIN,
    PRIOR_VISITS, DRG_SHARE, CREAT_MISSING, HGB_MISSING, SODIUM_MISSING, PAYER, ADMIT_TYPE, DISCHARGE,
    ICD_CHAPTER) AS P FROM READMIT_ML WHERE SPLIT = 'test' LIMIT 5""").show()
session.sql("ALTER MODEL READMIT30_SNOWML VERSION V1 SET ALIAS = CHALLENGER").collect()
session.close()
```

### 7. Push to GitHub
```bash
cd ~/ds-genai-90-days
printf 'day33/mlflow.db\nday33/mlruns/\n.env\n' >> .gitignore   # tracking store and credentials stay local
git add .gitignore day33
git commit -m "Day 33: Snowpark ML preprocessing, training and registry on the readmission data"
git push
```

## When to use Snowpark ML (note)
Use Snowpark ML on a Snowpark DataFrame when the training data is large or must stay governed in Snowflake (PHI): the fit runs in the warehouse and the Model Registry keeps versions, signatures and metrics next to the data, callable from Python or SQL. Use pandas plus scikit-learn when the model-ready frame is small. Snowpark ML is scikit-learn underneath, so model choice rules do not change: build row features in Snowpark or SQL, fit learned steps on train only, choose with grouped CV, open the test set once.

## Tomorrow (Day 34)
After deployment: is the readmission model still healthy next month? Data drift and calibration checks on a fresh batch of simulated stays.

## Printables
- Cheat sheet: `day33_v1_cheat_sheet.png`
- Infographic: `day33_v2_infographic.png`
- Code worksheet: `day33_v3_worksheet.png`
- Practice quiz: `day33_v4_quiz.png`
- PDF: `day33_lesson.pdf`

## Podcast script
(Audio: `day33_podcast.mp3`)

Hey Todor, welcome to day thirty three. Yesterday we moved the feature work for thirty day readmission into Snowpark. Today we take the next step: training and registering the model inside Snowflake with Snowpark ML. The package is called snowflake ML python.

So what is Snowpark ML? Think of it as scikit-learn with Snowflake manners. You get the same scalers, imputers, one hot encoders, logistic regression and XGBoost. The big difference is that you name your columns. Instead of passing a table of numbers, you say which input columns to use, which label column to predict, and which output columns to create. When you fit on a Snowpark DataFrame, the training runs inside Snowflake, so patient data never leaves the warehouse. When you fit on a pandas DataFrame, the same code runs on your laptop.

Now, an honest note. We have no Snowflake account on the lesson computer. In local testing mode, fitting Snowpark ML on a Snowpark DataFrame stopped with a not supported error, and so did the model registry. So here is what we did. We built the features in Snowpark local testing, like yesterday, and then fed pandas frames to the very same Snowpark ML objects. The lesson also includes the real account version, with the connection read from environment variables.

Was the math really the same as scikit-learn? We checked every learned number. The imputer medians, the scaler means, the scaled test values and the one hot columns all matched plain scikit-learn exactly. The final risks differed by about two in ten quadrillion, which is just rounding noise.

Next, the model race, on the same two thousand six hundred fifty five simulated stays and the same patient split. The Snowpark ML logistic regression scored a test AUC of point seven three four. The Snowpark ML XGBoost model also scored point seven three four on test. The Day 31 champion scored point seven four three. But remember, we pick models with cross validation on the training data, not on the test set. With GroupKFold, the logistic model averaged point seven one eight, XGBoost only point six eight four, and the champion point seven two four. So the champion stays. A new tool is not a reason to change models.

Here is a practical detail. The two models agreed closely, with a correlation of point nine two, but they shared only forty four of the sixty six highest risk stays. For a care manager, that means the worklist depends on which model you deploy. That is exactly why a registry, with versions and a clear default, matters.

We also hit some real gotchas. XGBoost is now an optional install. If you forget to name the input columns, the classifier tries to use patient ID and admit date, and it crashed on the date. The one hot column names came back wrapped in double quotes. And turning the whole pipeline back into scikit-learn broke on the test rows, so we wrote a plain scikit-learn twin and proved it gave the same risks.

For the registry, on a real account you call log model with a model name and a version, Snowflake stores the signature and metrics, and you score with run, or straight from SQL. Locally, we inferred the same signature, registered the twin in MLflow as the challenger, and scored all six hundred fifty eight test stays with a vectorized user defined function, matching local predictions.

All the data is simulated for teaching. Your exercise is in the day thirty three folder of your repo. Tomorrow, on day thirty four, we ask what happens after deployment: is the readmission model still healthy next month? We will check data drift and calibration on a fresh batch of simulated stays. See you then.
