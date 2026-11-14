# Day 31: Experiment tracking with MLflow: log, compare, register and reload the 30-day readmission pipeline

## Concept
Healthcare data science postings ask for reproducible work and MLOps basics: experiment tracking, model versioning and knowing exactly which data and settings produced a number. MLflow organizes this as an experiment (one question) with many runs (one try each). In a run you log params (knobs such as C), metrics (scores such as cv_auc), tags (context such as a data fingerprint and the CV scheme) and artifacts (files: charts, feature lists, the model). Everything lives in a local SQLite file (sqlite:///mlflow.db), with artifacts in ./mlruns, so no server is needed. On Day 30's 2,655 simulated stays (seed 30, readmit rate 17.8%, same patient split), mlflow.sklearn.autolog() logged 147 params, 4 artifacts and a training AUC of 0.749, which is optimistic because it scores rows the model just saw. Manual logging gave each of 4 variants one run with 4 params, 5 tags (including the data file md5), fold_auc per step, CV metrics, a chart and a JSON feature list. GroupKFold CV AUC: DRG rare one-hot 0.723, frequency encoder 0.724, both 0.724, frequency with C=0.05 0.723 (fold sd 0.031 to 0.038). mlflow.search_runs returned the runs as a pandas DataFrame; the Day 30 rule (simplest within 1 sd of the best, limit 0.693) picked the frequency encoder with C=0.5 (39 features). Test AUC opened once: 0.743, matching Day 30. The champion was logged with a signature (12 typed input columns), code_paths for the custom transformers and registered as readmit30_pipeline v1 with alias @champion. Gotcha: MLflow 3.18 saves scikit-learn models with skops, which refused our custom classes until we listed them in skops_trusted_types. Loading by alias with pyfunc and with sklearn gave identical predictions; the signature blocked a missing drg column and age sent as text, but let a missing creatinine through (it had NaN in the sample, so it was marked optional) and sklearn caught it. Rerunning the champion's CV reproduced the logged AUC to 6 decimals (0.723927). All data is simulated for teaching.

## Key numbers
- **Data:** Day 30 table: 2,655 simulated stays, 1,000 patients (seed 30), readmit rate 17.8%, train 1,997 / test 658 by patient, md5 312070c916ba
- **Tracking:** MLflow 3.18.0, sqlite:///mlflow.db, artifacts in ./mlruns, experiment day31_readmit30, 6 runs (1 autolog, 4 CV variants, 1 champion)
- **Autolog vs manual:** autolog: 147 params, 4 artifacts, training AUC 0.749 (no CV); manual: 4 params + 5 tags + fold_auc steps + CV metrics per run
- **CV ROC AUC:** rare C=0.5 0.723 (53 cols), freq C=0.5 0.724 (39), freq+rare C=0.5 0.724 (54), freq C=0.05 0.723 (39); fold sd 0.031 to 0.038
- **Rule + test:** best 0.724 - 1 sd = 0.693; fewest features, then higher AUC: freq_C0.5; test AUC 0.743
- **Registry:** readmit30_pipeline v1, alias @champion; pyfunc and sklearn reloads identical to the original predictions
- **Signature:** drop drg or send age as text: blocked by MLflow; drop creatinine (optional, NaN in sample): passed MLflow, failed in sklearn
- **Reproducible:** rerun of the champion CV: 0.723927 logged vs 0.723927 rerun

## Hands-on (60-90 min)
1. In your repo, create day31/, run conda activate ds90, pip install mlflow, copy ../day30/readmit_transformers.py into day31/ and open a notebook day31/mlflow_readmit.ipynb in VS Code.
2. Run snippet 1: load Day 30's readmit_raw.csv, rebuild the same patient split, compute the md5 fingerprint, and define build(drg, C).
3. Run snippet 2: point MLflow at sqlite:///mlflow.db, create the experiment, and try autolog. Note how many params it logged and that its AUC is a training score.
4. Write your selection rule in a markdown cell before racing: best CV AUC, then the simplest variant within 1 sd.
5. Run snippet 3: race 4 variants by hand, one run each, with tags, params, fold_auc per step, CV metrics, a chart and a feature list.
6. In a terminal, run mlflow ui --backend-store-uri sqlite:///mlflow.db, open http://127.0.0.1:5000 and compare the 4 runs (table, fold_auc chart, artifacts).
7. Run snippet 4: search_runs into pandas, apply the rule, refit the champion, open the test set once, and log + register the model with a signature and code_paths. If you hit UntrustedTypesFoundException, read it, then allow-list your own classes.
8. Run snippet 5: set the @champion alias, load with pyfunc and sklearn, compare predictions, run the 3 contract tests, and rerun the champion CV to confirm the logged AUC.
9. Write day31/README.md: the run table from search_runs, autolog vs manual, the signature gotcha, the reproducibility check and limits (simulated data). Add mlflow.db and mlruns/ to .gitignore.
10. Commit and push (snippet 6): git add .gitignore day31, git commit -m "Day 31: ...", git push.
11. Stretch: build the training table in Snowflake with Snowpark (snippet 7), pull it to pandas and log the same race to the same experiment with a tag source=snowflake.

## Code
### 1. Same data, same split, same build() as Day 30 (now with C as a knob)
```python
# Day 31 setup: the SAME raw table, split and pipeline builder as Day 30, so every MLflow run is comparable.
# In the shell first:  cp ../day30/readmit_transformers.py .   (MLflow ships this file with the model)
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from readmit_transformers import FrequencyEncoder, DateFeatures, icd_chapter

RAW = "../day30/readmit_raw.csv"                       # simulated, seed 30 (Day 30 snippet 1)
df = pd.read_csv(RAW, parse_dates=["admit_date"], dtype={"drg": str, "icd10": str})
data_md5 = hashlib.md5(open(RAW, "rb").read()).hexdigest()[:12]    # fingerprint of the exact data file
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
tr_idx, te_idx = next(gss.split(df, groups=df.patient_id))           # patients never cross the split
X = df.drop(columns="readmit30")
train, test = X.iloc[tr_idx], X.iloc[te_idx]
ytr, yte, gtr = df.readmit30.iloc[tr_idx], df.readmit30.iloc[te_idx], train.patient_id
gkf = GroupKFold(5)
log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build(drg="freq", C=0.5):                          # Day 30 build(), now with C as a tracked knob
    parts = [
        ("dates", make_pipeline(DateFeatures(), StandardScaler()), ["patient_id", "admit_date"]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "sodium"]),
        ("log", make_pipeline(SimpleImputer(strategy="median"), log1p, StandardScaler()), ["los_days", "creatinine"]),
        ("hgb_bin", make_pipeline(SimpleImputer(strategy="median"),
                                  KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile",
                                                   quantile_method="averaged_inverted_cdf")), ["hemoglobin"]),
        ("miss", MissingIndicator(), ["creatinine", "hemoglobin", "sodium"]),
        ("icd", make_pipeline(FunctionTransformer(icd_chapter, feature_names_out="one-to-one"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), ["icd10"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["payer", "admit_type", "discharge"]),
    ]
    if "freq" in drg:
        parts.append(("drg_freq", make_pipeline(FrequencyEncoder(), StandardScaler()), ["drg"]))
    if "rare" in drg:
        parts.append(("drg_oh", OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist",
                                              sparse_output=False), ["drg"]))
    pre = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("prep", pre), ("clf", LogisticRegression(C=C, max_iter=3000))]).set_output(transform="pandas")

print(df.shape, "readmit rate", round(df.readmit30.mean(), 3), "| train", len(train), "test", len(test), "| data md5", data_md5)
```

### 2. Local tracking store (SQLite) and the autolog demo
```python
# Tracking store: one local SQLite file (mlflow.db). Artifacts (models, charts) go to a local folder.
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")          # browse later: mlflow ui --backend-store-uri sqlite:///mlflow.db
exp = mlflow.set_experiment("day31_readmit30")
print("experiment", exp.name, "id", exp.experiment_id, "| artifacts:", exp.artifact_location)

# Option A: autolog. One line, logs every pipeline parameter plus TRAINING metrics on the fit call.
mlflow.sklearn.autolog(log_models=False, silent=True)
with mlflow.start_run(run_name="autolog_demo") as r:
    mlflow.set_tag("stage", "autolog_demo")
    build("freq").fit(train, ytr)
mlflow.sklearn.autolog(disable=True)                    # switch off before the manual race

auto = mlflow.get_run(r.info.run_id).data
print("autolog logged", len(auto.params), "params, e.g.", {k: auto.params[k] for k in ["clf__C", "clf__max_iter"]})
print("autolog metrics:", {k: round(v, 3) for k, v in sorted(auto.metrics.items())})
print("missing: no GroupKFold CV score, no data fingerprint, no reason for the run")
```

### 3. Manual logging: 4 pipeline variants, tags, params, fold metrics, artifacts
```python
# Option B: manual logging. Four pipeline variants, each one run with params, tags, CV metrics and artifacts.
from sklearn.model_selection import cross_validate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = {"rare_C0.5": ("rare", 0.5), "freq_C0.5": ("freq", 0.5),
            "freq+rare_C0.5": ("freq+rare", 0.5), "freq_C0.05": ("freq", 0.05)}
for name, (drg, C) in VARIANTS.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": "cv_race", "data_md5": data_md5, "cv": "GroupKFold(5) by patient_id",
                         "data": "simulated readmission, seed 30", "author": "Todor"})
        mlflow.log_params({"drg_encoding": drg, "C": C, "model": "LogisticRegression", "n_splits": 5})
        cv = cross_validate(build(drg, C), train, ytr, groups=gtr, cv=gkf,
                            scoring=["roc_auc", "neg_log_loss"], return_estimator=True)
        feats = list(cv["estimator"][0][:-1].get_feature_names_out())
        for k, a in enumerate(cv["test_roc_auc"], start=1):
            mlflow.log_metric("fold_auc", a, step=k)                    # one value per fold = a small curve
        mlflow.log_metrics({"cv_auc": cv["test_roc_auc"].mean(), "cv_auc_sd": cv["test_roc_auc"].std(),
                            "cv_logloss": -cv["test_neg_log_loss"].mean(), "n_features": len(feats)})
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.bar(range(1, 6), cv["test_roc_auc"]); ax.set_ylim(0.6, 0.85); ax.set_title(f"{name}: fold AUC")
        fig.savefig("fold_auc.png", dpi=100); plt.close(fig)
        mlflow.log_artifact("fold_auc.png", artifact_path="charts")      # any file: charts, reports
        mlflow.log_dict({"features": feats}, "features.json")             # small JSON without a temp file
        print(f"{name:15s} cv_auc {cv['test_roc_auc'].mean():.3f}  sd {cv['test_roc_auc'].std():.3f}  features {len(feats)}")
```

### 4. search_runs into pandas, the 1 sd rule, log + register the champion with a signature
```python
# Read the runs back into pandas, apply the Day 30 rule, then log + register ONE champion with a signature.
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score

runs = mlflow.search_runs(experiment_names=["day31_readmit30"], filter_string="tags.stage = 'cv_race'",
                          order_by=["metrics.cv_auc DESC"])
cols = ["tags.mlflow.runName", "params.drg_encoding", "params.C", "metrics.cv_auc", "metrics.cv_auc_sd",
        "metrics.n_features", "run_id"]
board = runs[cols].rename(columns=lambda c: c.split(".")[-1])
print(board.drop(columns="run_id").round(3).to_string(index=False))

top = board.iloc[0]; limit = top.cv_auc - top.cv_auc_sd
ok = board[board.cv_auc >= limit].sort_values(["n_features", "cv_auc"], ascending=[True, False])
champ = ok.iloc[0]                                      # simplest within 1 sd, ties -> higher CV AUC
print(f"best {top.runName} {top.cv_auc:.3f} | limit {limit:.3f} | champion {champ.runName}")

TRUSTED = ["numpy.dtype", "readmit_transformers.DateFeatures", "readmit_transformers.FrequencyEncoder",
           "readmit_transformers.icd_chapter"]           # first try failed with UntrustedTypesFoundException
with mlflow.start_run(run_name=f"champion_{champ.runName}") as run:
    final = build(champ.drg_encoding, float(champ.C)).fit(train, ytr)
    p_te = final.predict_proba(test)[:, 1]
    test_auc = roc_auc_score(yte, p_te)                 # opened ONCE, logged once
    sig = infer_signature(train.head(100), final.predict_proba(train.head(100)))
    mlflow.set_tags({"stage": "champion", "cv_run_id": champ.run_id, "data_md5": data_md5})
    mlflow.log_params({"drg_encoding": champ.drg_encoding, "C": float(champ.C)})
    mlflow.log_metrics({"cv_auc": champ.cv_auc, "test_auc": test_auc})
    info = mlflow.sklearn.log_model(final, name="model", signature=sig, input_example=train.head(3),
                                    pyfunc_predict_fn="predict_proba",
                                    code_paths=["readmit_transformers.py"],   # the custom classes travel along
                                    skops_trusted_types=TRUSTED,              # MLflow 3 saves with skops: allow-list them
                                    registered_model_name="readmit30_pipeline")
print(f"test AUC (once) {test_auc:.3f} | model {info.model_uri} | registered version {info.registered_model_version}")
print("signature inputs:", len(sig.inputs.input_names()), "columns, e.g.", sig.inputs.to_dict()[:2])
```

### 5. Load by alias, contract tests, reproducibility check
```python
# Load the registered model by alias, prove identical predictions, test the signature, and check reproducibility.
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("readmit30_pipeline", "champion", info.registered_model_version)

pyf = mlflow.pyfunc.load_model("models:/readmit30_pipeline@champion")     # generic loader (what serving uses)
p_pyf = np.asarray(pyf.predict(test))[:, 1]
skm = mlflow.sklearn.load_model("models:/readmit30_pipeline@champion")    # back to a real sklearn Pipeline
print("pyfunc == original:", np.allclose(p_pyf, p_te), "| sklearn == original:", np.allclose(skm.predict_proba(test)[:, 1], p_te))
print("risk for 3 raw rows:", p_pyf[:3].round(3))

def contract(frame):                                    # the signature guards the input contract
    try:
        pyf.predict(frame); return "accepted"
    except Exception as e:
        return type(e).__name__ + (" (MLflow schema)" if "enforce schema" in str(e) else " (sklearn)")
print("drop drg (required):        ", contract(test.drop(columns="drg")))
print("age sent as text:           ", contract(test.assign(age=test.age.astype(str))))
print("drop creatinine (optional): ", contract(test.drop(columns="creatinine")))   # NaN in sample -> optional

# Reproducibility: rerun the champion variant's CV with the same data + seed, compare to the logged value.
again = cross_validate(build(champ.drg_encoding, float(champ.C)), train, ytr, groups=gtr, cv=gkf, scoring="roc_auc")
print(f"logged cv_auc {champ.cv_auc:.6f} | rerun {again['test_score'].mean():.6f} | same: {np.isclose(champ.cv_auc, again['test_score'].mean())}")

allr = mlflow.search_runs(experiment_names=["day31_readmit30"])
print("runs in experiment:", len(allr), allr["tags.stage"].value_counts().to_dict())
auto_art = [a.path for a in client.list_artifacts(r.info.run_id)]
print("autolog params vs manual params per run:", len(auto.params), "vs", 4, "| autolog artifacts:", auto_art)
```

### 6. Push to GitHub
```bash
cd ~/ds-genai-90-days
printf 'day31/mlflow.db\nday31/mlruns/\n' >> .gitignore   # tracking store stays local
git add .gitignore day31
git commit -m "Day 31: MLflow experiment tracking on the readmission pipeline"
git push
```

### 7. Stretch: Snowpark table, same MLflow experiment
```python
# Stretch (not run here, needs your Snowflake account): Snowpark builds the table, MLflow tracks the run.
# pip install snowflake-snowpark-python ; connection values come from your own config, never hard-coded.
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
session = Session.builder.config("connection_name", "ds90").create()     # reads ~/.snowflake/connections.toml
raw = session.table("READMIT_RAW")                                         # Day 30 table loaded into Snowflake
train_sf = raw.filter(col("SPLIT") == "train").to_pandas()                 # pushdown filter, then pandas
train_sf.columns = train_sf.columns.str.lower()

with mlflow.start_run(run_name="freq_C0.5_snowflake"):
    mlflow.set_tags({"stage": "cv_race", "source": "snowflake", "table": "READMIT_RAW"})
    mlflow.log_params({"drg_encoding": "freq", "C": 0.5})
    cv = cross_validate(build("freq", 0.5), train_sf.drop(columns=["readmit30", "split"]), train_sf.readmit30,
                        groups=train_sf.patient_id, cv=GroupKFold(5), scoring="roc_auc")
    mlflow.log_metric("cv_auc", cv["test_score"].mean())                   # should match the local run
```

## Tomorrow (Day 32)
Move the feature work into Snowflake with Snowpark for Python, and log those runs to the same MLflow experiment so local and Snowflake results sit side by side.

## Printables
- Cheat sheet: `day31_v1_cheat_sheet.png`
- Infographic: `day31_v2_infographic.png`
- Code worksheet: `day31_v3_worksheet.png`
- Practice quiz: `day31_v4_quiz.png`
- PDF: `day31_lesson.pdf`

## Podcast script
(Audio: `day31_podcast.mp3`)

Hey Todor, welcome to day thirty one. Yesterday we built one full scikit-learn pipeline for thirty day readmission, from the raw table to a probability. Today we stop keeping score in our heads and start keeping it in MLflow.

Why does this matter for a job? Healthcare data science postings keep asking for reproducible work and some MLOps: experiment tracking, model versioning, being able to say exactly which data and which settings produced a number. MLflow is one of the most common tools for that, and it runs fine on a laptop.

Here is the vocabulary. An experiment is one question, like which readmission pipeline should we use. A run is one try. Inside a run you log params, which are the knobs you set, metrics, which are the scores, tags, which are the context, and artifacts, which are files like charts or the model itself. We stored everything in one small SQLite file, so nothing needs a server.

First we tried autolog. One line, and MLflow recorded one hundred forty seven parameters from the pipeline, plus a confusion matrix, a ROC curve and a training AUC of about point seven four nine. That sounds great, but it is a training score, measured on rows the model had just seen. It is not our honest number.

So we logged by hand. Four pipeline variants, one run each: DRG codes pooled with one hot encoding, the custom frequency encoder from yesterday, both together, and the frequency encoder with ten times stronger regularization. Each run got four params, five tags including a fingerprint of the exact data file, the five fold AUCs as a little curve, the mean and spread, a chart, and the list of feature names.

The results were very close. Group k fold AUC by patient was about point seven two three or point seven two four for all four, with fold to fold swings of about three to four points. Then we pulled every run back into a pandas table with search runs, and applied the same rule as yesterday: take the best, subtract one standard deviation, and pick the simplest variant that still qualifies. The winner was the frequency encoder with the default regularization, thirty nine features. Opened once, the test set gave an AUC of point seven four three. Same numbers as day thirty, which is exactly the point. The tracking reproduced them.

Next we logged that champion as a model with a signature. A signature is the input contract: twelve named columns with types. We also shipped the file with our custom transformers along with the model, because a saved pipeline only remembers where a class lives, not its code. And here is a real gotcha we hit. The newest MLflow saves scikit-learn models in a safer format that refuses unknown classes, so the first save failed. The fix was to list our own classes as trusted.

Then we registered the model, gave version one the alias champion, and loaded it back by that alias. Predictions matched the original pipeline exactly. When we dropped the DRG column or sent age as text, MLflow blocked the request. But when we dropped creatinine, MLflow let it through, because creatinine had missing values in the sample, so the signature marked it optional. Scikit-learn caught it instead. Good to know before anyone calls this in production.

Last check: we reran the champion's cross validation from scratch and got the same AUC as the logged value, to six decimal places. That is reproducibility you can show an interviewer.

All the data is simulated for teaching. Your exercise is in the day thirty one folder of your repo. Remember to keep the database file and the runs folder out of git. Tomorrow, on day thirty two, we move the feature work into Snowflake with Snowpark, and log those runs to the same MLflow experiment. See you then.
