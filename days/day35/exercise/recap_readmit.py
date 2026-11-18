# Day 35: week 5 recap project on 30-day readmission (simulated data).
# pipeline > MLflow tracking > registry @champion > Snowpark scoring (local testing) > drift + calibration.
# Put this file in day35/ next to sim_readmit.py and readmit_transformers.py. No Snowflake account needed.
# ---------- snippet 1 ----------
# Day 35 recap project, part 1: one folder, one story. Rebuild the Day 30 table with the generator,
# split by patient exactly as Days 30 to 34, and reuse the Day 30 pipeline builder (custom transformers inside).
import os, hashlib, warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sim_readmit import simulate                          # Day 34 generator: simulate(1000, 30) == Day 30 table
from readmit_transformers import FrequencyEncoder, DateFeatures, icd_chapter   # Day 30 custom classes

raw = simulate(1000, 30)                                  # simulated 30-day readmission stays (teaching data)
if os.path.exists("../day30/readmit_raw.csv"):            # optional check against your Day 30 file
    old = pd.read_csv("../day30/readmit_raw.csv", parse_dates=["admit_date"], dtype={"drg": str, "icd10": str})
    print("rebuilds Day 30 exactly:", raw.equals(old))
data_md5 = hashlib.md5(pd.util.hash_pandas_object(raw, index=False).values).hexdigest()[:12]
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
                      .split(raw, groups=raw.patient_id))  # patients never cross the split
X, y = raw.drop(columns="readmit30"), raw.readmit30
train, test, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build(drg="freq", C=0.5):                    # a slimmed Day 30 builder (hemoglobin kept numeric)
    """Raw columns in, probability out. Every learned step is fit on the training fold only."""
    parts = [
        ("dates", make_pipeline(DateFeatures(), StandardScaler()), ["patient_id", "admit_date"]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "sodium", "hemoglobin"]),
        ("log", make_pipeline(SimpleImputer(strategy="median"), log1p, StandardScaler()), ["los_days", "creatinine"]),
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
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(C=C, max_iter=3000))])

print(f"stays {len(raw)} | patients {raw.patient_id.nunique()} | readmit rate {y.mean():.3f} | "
      f"train {len(train)} / test {len(test)} stays | data fingerprint {data_md5}")

# ---------- snippet 2 ----------
# Part 2 (Days 29 to 31): race 3 pipeline variants with GroupKFold and log each one to MLflow.
import mlflow
from sklearn.model_selection import cross_validate
mlflow.set_tracking_uri("sqlite:///mlflow.db")            # local store in day35/ (keep it out of git)
mlflow.set_experiment("day35_recap")
VARIANTS = [("freq", 0.5), ("rare", 0.5), ("freq+rare", 0.5)]
rows = []
for drg, C in VARIANTS:
    cv = cross_validate(build(drg, C), train, ytr, groups=train.patient_id, cv=GroupKFold(5),
                        scoring={"auc": "roc_auc", "brier": "neg_brier_score"})
    auc, sd, brier = cv["test_auc"].mean(), cv["test_auc"].std(), -cv["test_brier"].mean()
    n_feat = build(drg, C)[:-1].fit(train, ytr).transform(train.head(5)).shape[1]
    with mlflow.start_run(run_name=f"cv_{drg}_C{C}") as r:
        mlflow.set_tags({"stage": "cv_race", "data_md5": data_md5, "day": "35"})
        mlflow.log_params({"drg_encoding": drg, "C": C, "cv": "GroupKFold(5)"})
        mlflow.log_metrics({"cv_auc": auc, "cv_auc_sd": sd, "cv_brier": brier, "n_features": n_feat})
    rows.append({"drg": drg, "C": C, "cv_auc": round(auc, 3), "sd": round(sd, 3), "brier": round(brier, 4),
                 "n_features": n_feat, "run_id": r.info.run_id})
board = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
print(board.drop(columns="run_id").to_string(index=False))
limit = board.cv_auc.iloc[0] - board.sd.iloc[0]          # the rule, written before testing:
champ_row = board[board.cv_auc >= limit].sort_values(["n_features", "cv_auc"], ascending=[True, False]).iloc[0]
print(f"rule: simplest within 1 sd of the best ({limit:.3f}) -> champion = {champ_row.drg}, C {champ_row.C}")

# ---------- snippet 3 ----------
# Part 3 (Day 31): fit the champion once, open the test set once, log + register it with a signature,
# point the @champion alias at it, then reload BY ALIAS (what every scoring job should do).
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import roc_auc_score, brier_score_loss
TRUSTED = ["numpy.dtype", "readmit_transformers.DateFeatures", "readmit_transformers.FrequencyEncoder",
           "readmit_transformers.icd_chapter"]
with mlflow.start_run(run_name=f"champion_{champ_row.drg}") as run:
    final = build(champ_row.drg, float(champ_row.C)).fit(train, ytr)
    p_te = final.predict_proba(test)[:, 1]
    test_auc, test_brier = roc_auc_score(yte, p_te), brier_score_loss(yte, p_te)
    sig = infer_signature(train.head(100), final.predict_proba(train.head(100)))
    mlflow.set_tags({"stage": "champion", "cv_run_id": champ_row.run_id, "data_md5": data_md5})
    mlflow.log_params({"drg_encoding": champ_row.drg, "C": float(champ_row.C)})
    mlflow.log_metrics({"cv_auc": champ_row.cv_auc, "test_auc": test_auc, "test_brier": test_brier})
    info = mlflow.sklearn.log_model(final, name="model", signature=sig, input_example=train.head(3),
                                    pyfunc_predict_fn="predict_proba", code_paths=["readmit_transformers.py"],
                                    skops_trusted_types=TRUSTED, registered_model_name="readmit30_recap")
client = MlflowClient()
client.set_registered_model_alias("readmit30_recap", "champion", info.registered_model_version)
champ = mlflow.sklearn.load_model("models:/readmit30_recap@champion")
same = np.allclose(champ.predict_proba(test)[:, 1], p_te)
print(f"test (opened once): AUC {test_auc:.3f}, Brier {test_brier:.4f} | registered readmit30_recap "
      f"v{info.registered_model_version} @champion | reload by alias gives same scores: {same}")
print("signature:", len(sig.inputs.input_names()), "input columns, e.g.", sig.inputs.input_names()[:4])

# ---------- snippet 4 ----------
# Part 4 (Days 32 and 33): score NEW stays next to the data. Snowpark local testing mode is an in-memory
# emulator (no account, no credentials): load two months of live stays as a table, push the filter and
# select down, pull only what the model needs, score with @champion, write the scores back, aggregate.
from snowflake.snowpark import Session, Window
from snowflake.snowpark import functions as F
session = Session.builder.config("local_testing", True).create()

# Two live batches (simulated). Each holds 21 months of stays so every 2026 stay is scored WITH its history
# (the date features need prior visits); we report on the window Jan 1 to Sep 30, 2026 only.
# "stable" = same world as training. "drifted" = older patients, a creatinine assay reading 25% high,
# and a care program that cuts readmits after Home health / SNF discharge.
W0, W1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")
hist_kw = dict(start="2025-01-01", days=(W1 - pd.Timestamp("2025-01-01")).days, clip=True)
stable_h = simulate(1500, 351, pid0=100000, **hist_kw)
drifted_h = simulate(1500, 352, pid0=200000, age_mean=71, creat_report=1.25, care_effect=-1.5, **hist_kw)
up = drifted_h.copy(); up.columns = up.columns.str.upper()   # Snowflake stores names in UPPER case
session.create_dataframe(up).write.save_as_table("LIVE_STAYS", mode="overwrite")

RAW_COLS = ["PATIENT_ID", "ADMIT_DATE", "AGE", "PAYER", "ADMIT_TYPE", "DISCHARGE", "DRG", "ICD10",
            "LOS_DAYS", "CREATININE", "HEMOGLOBIN", "SODIUM"]
stays = session.table("LIVE_STAYS")                        # lazy: nothing has run yet
w = Window.partition_by("PATIENT_ID").order_by("ADMIT_DATE")
feat = stays.with_column("PRIOR_VISITS", F.row_number().over(w) - 1)   # a window feature, computed in Snowflake
live_pd = feat.select(RAW_COLS + ["READMIT30", "PRIOR_VISITS"]).to_pandas()   # the action: the query runs here
live_pd.columns = live_pd.columns.str.lower()
live_pd["admit_date"] = pd.to_datetime(live_pd.admit_date)
live_pd["risk"] = champ.predict_proba(live_pd[[c.lower() for c in RAW_COLS]])[:, 1]   # scored WITH history

scores = live_pd[["patient_id", "admit_date", "risk"]].copy(); scores.columns = scores.columns.str.upper()
session.create_dataframe(scores).write.save_as_table("READMIT_SCORES", mode="overwrite")
by_payer = (session.table("READMIT_SCORES").join(stays, on=["PATIENT_ID", "ADMIT_DATE"])
            .filter(F.col("ADMIT_DATE") >= F.lit(W0.to_pydatetime()))  # report on the 2026 window only
            .group_by("PAYER").agg(F.count("*").alias("N"), F.avg("RISK").alias("MEAN_RISK"),
                                   F.avg("READMIT30").alias("ACTUAL"))
            .to_pandas().sort_values("MEAN_RISK", ascending=False))   # sort in pandas: the local emulator
                                                                      # mislabeled rows when sorting after agg
print(f"LIVE_STAYS {stays.count()} rows (2025 history + 2026 window) scored | max prior visits {live_pd.prior_visits.max()}")
print("drifted batch, 2026 window, by payer:")
print(by_payer.round(3).to_string(index=False))
def window(h):
    h = h.assign(risk=champ.predict_proba(h.drop(columns="readmit30"))[:, 1])
    return h[(h.admit_date >= W0) & (h.admit_date < W1)].reset_index(drop=True)
stable, drifted = window(stable_h), window(drifted_h)
chk = live_pd[live_pd.admit_date >= W0].sort_values(["patient_id", "admit_date"]).risk.to_numpy()
print(f"window stays: stable {len(stable)}, drifted {len(drifted)} | Snowpark-path scores match pandas-path scores:",
      np.allclose(chk, drifted.sort_values(["patient_id", "admit_date"]).risk.to_numpy()))

# ---------- snippet 5 ----------
# Part 5 (Day 34): is the registered model still healthy? Drift needs no labels (PSI on inputs AND score);
# calibration needs outcomes (O/E, Brier, reliability). Then a Platt fix on recent labels, logged to MLflow.
import json
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

def psi(ref_col, cur_col, bins=10, eps=1e-4):
    """Population Stability Index: reference decile bins (one bin per category, plus a missing bin)."""
    r, c = pd.Series(ref_col), pd.Series(cur_col)
    if pd.api.types.is_numeric_dtype(r):
        edges = np.unique(np.nanquantile(r, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
        r = pd.cut(r, edges).astype(str).where(r.notna(), "missing")
        c = pd.cut(c, edges).astype(str).where(c.notna(), "missing")
    p = r.value_counts(normalize=True)
    q = c.value_counts(normalize=True).reindex(p.index.union(c.unique()), fill_value=0)
    p = p.reindex(q.index, fill_value=0).clip(lower=eps); q = q.clip(lower=eps)
    return float(((q - p) * np.log(q / p)).sum())

def cal(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"n": len(y), "auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 4),
            "mean_pred": round(p.mean(), 3), "observed": round(y.mean(), 3), "o_e": round(y.mean() / p.mean(), 2)}

ref = test.assign(readmit30=yte, risk=p_te)                # reference = the test stays the model was judged on
FEATS = ["age", "creatinine", "los_days", "hemoglobin", "payer", "discharge", "risk"]
drift = pd.DataFrame({name: {f: round(psi(ref[f], d[f]), 3) for f in FEATS}
                      for name, d in [("stable", stable), ("drifted", drifted)]})
drift["band"] = drift.drifted.map(lambda v: "stable" if v < .10 else ("watch" if v < .25 else "ACT"))
print(drift.to_string())
perf = pd.DataFrame({n: cal(d.readmit30, d.risk) for n, d in
                     [("reference", ref), ("stable", stable), ("drifted", drifted)]}).T
print(perf.to_string())
curves = {n: calibration_curve(d.readmit30, d.risk, n_bins=8, strategy="quantile")
          for n, d in [("reference", ref), ("stable", stable), ("drifted", drifted)]}

# Recalibrate the FROZEN champion on the first 5 months of drifted labels, judge on the last 4 months.
cut_date = pd.Timestamp("2026-06-01")
early, late = drifted[drifted.admit_date < cut_date], drifted[drifted.admit_date >= cut_date]
Xcols = [c for c in X.columns]
platt = CalibratedClassifierCV(FrozenEstimator(champ), method="sigmoid").fit(early[Xcols], early.readmit30)
late = late.assign(risk_platt=platt.predict_proba(late[Xcols])[:, 1])
recal = pd.DataFrame({"raw champion": cal(late.readmit30, late.risk),
                      "after Platt": cal(late.readmit30, late.risk_platt)}).T
print(f"Platt fit on {len(early)} stays (Jan to May), judged on {len(late)} (Jun to Sep):")
print(recal.to_string())

mlflow.set_experiment("day35_monitoring")                   # every check sits next to the model version
with mlflow.start_run(run_name="monitor_drifted_batch"):
    mlflow.set_tags({"model": "readmit30_recap", "model_version": info.registered_model_version, "batch": "drifted"})
    mlflow.log_metrics({f"psi_{f}": v for f, v in drift.drifted.items()})
    mlflow.log_metrics({"o_e": perf.loc["drifted", "o_e"], "brier": perf.loc["drifted", "brier"],
                        "auc": perf.loc["drifted", "auc"], "o_e_after_platt": recal.loc["after Platt", "o_e"]})
alerts = [f for f, v in drift.drifted.items() if v >= .25] + (["o_e"] if not .8 <= perf.loc["drifted", "o_e"] <= 1.25 else [])
card = {"data_md5": data_md5, "n_stays": len(raw), "train_n": len(train), "test_n": len(test),
        "board": board.drop(columns="run_id").to_dict("records"), "champion": champ_row.drg,
        "test_auc": round(test_auc, 3), "test_brier": round(test_brier, 4), "version": int(info.registered_model_version),
        "n_sig": len(sig.inputs.input_names()), "same_by_alias": bool(same), "live_n": {"stable": len(stable), "drifted": len(drifted), "hist": len(drifted_h)},
        "psi": drift.drop(columns="band").to_dict(), "perf": perf.to_dict("index"), "recal": recal.to_dict("index"),
        "early_n": len(early), "late_n": len(late), "alerts": alerts, "by_payer": by_payer.round(3).to_dict("records"),
        "age": [round(ref.age.mean(), 1), round(drifted.age.mean(), 1)]}
json.dump(card, open("recap_card.json", "w"), indent=2, default=float)
print("alerts:", alerts, "| run card saved to recap_card.json")
