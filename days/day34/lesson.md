# Day 34: Model drift and calibration: PSI, KS, reliability diagrams, Brier and recalibration on 30-day readmission

## Concept
Days 30 to 33 built, tracked and registered a readmission model. Today it goes live, and the question changes from "is it accurate?" to "is it still healthy?". Data drift means the inputs change, P(x): older patients, a new lab assay. Concept drift means the link between inputs and outcome changes, P(y | x): a transitional care program that cuts readmissions. We rebuilt the Day 30 generator as a function (it reproduces Day 30 exactly), loaded the Day 31 @champion, and scored 12 weeks of new simulated stays (Jul 6 to Sep 27, 2026, about 210 a week) with each patient's history, because the date features need prior visits. On Aug 17 three things changed: mean age 64.6 to 70.3, a creatinine assay that reads 25% high, and a care program for Home health and SNF discharges. PSI (population stability index) bins the reference into deciles and sums (q - p) ln(q / p); under 0.10 is stable, 0.10 to 0.25 watch, over 0.25 act. After Aug 17: age 0.190, creatinine 0.173, discharge 0.001, and the score itself only 0.055, so watching only the score can hide input drift. The KS test agreed, but at about 1,300 stays its p-values flag tiny shifts, so judge the size of the shift. One stable week of noise alone gives PSI about 0.061. Calibration asks whether "30%" means about 30 of 100 come back. Tools: the reliability diagram (calibration_curve), the Brier score (mean squared error of the probability), O/E (observed over expected), calibration slope and ECE. O/E was 1.08 before Aug 17 and 0.70 after, while AUC went 0.743 to 0.706: the model predicted about 1.4x the readmissions that happened. Response in order: fix the data (mapping creatinine back moved O/E 0.70 to 0.80), recalibrate the frozen model on recent labels (Platt on weeks 7 to 9, judged on weeks 10 to 12: O/E 0.74 to 0.86, slope 0.79 to 0.96, Brier 0.1191 to 0.1147; isotonic overfit on 687 rows), then retrain, because Home health/SNF stays were still at O/E 0.40. Why it matters clinically: at a 25% outreach threshold the drifted model flagged 182 stays vs 116 after Platt, and expected readmits went 123 to 106 against 91 actual. At training time, class_weight="balanced" pushed mean risk to 0.447 (observed 0.170); CalibratedClassifierCV (sigmoid, GroupKFold) brought Brier 0.209 back to 0.123. Finally, a weekly monitor with alert rules written in advance, logged to MLflow (readmit30_monitoring), and PSI as plain SQL that matched Python (0.0545), ready for the Snowflake week. All data is simulated for teaching.

## Key numbers
- **Data:** reference: Day 30 test stays (658); live: 1,202 stays before Aug 17 + 1,321 after, 12 weeks, about 210 a week (simulated)
- **Model:** Day 31 MLflow champion (readmit30_pipeline@champion), scored with patient history
- **PSI after Aug 17:** age 0.190, creatinine 0.173, payer 0.053, discharge 0.001, risk score 0.055
- **Noise floor:** PSI of one stable week (about 210 stays): 0.061; week 4 raised a false calibration alarm
- **Calibration (O/E, AUC):** reference 0.92 / 0.743; before 1.08 / 0.731; after 0.70 / 0.706
- **Assay fix:** mean risk 0.215 -> 0.189; O/E 0.70 -> 0.80
- **Recalibration (weeks 10-12):** raw O/E 0.74, ECE 0.060; Platt 0.86, 0.036; isotonic 0.82, 0.043 (slope 0.16: overfit)
- **Threshold 25%:** flagged raw 182 vs Platt 116; expected readmits 123 vs 106; actual 91
- **class_weight='balanced':** mean risk 0.447 vs observed 0.170; Brier 0.209 -> sigmoid 0.123, isotonic 0.124
- **SQL check:** PSI of the score in SQL (SQLite) 0.0545 = Python; CalibratedClassifierCV(FrozenEstimator) vs manual Platt differs 7e-03 (Platt smoothed targets)

## Hands-on (60-90 min)
1. In your repo, create day34/, run conda activate ds90, copy sim_readmit.py, psi_snowflake.sql (from the exercise folder) and day30/readmit_transformers.py into day34/, and open day34/drift_calibration_readmit.ipynb in VS Code. The code reads ../day30/readmit_raw.csv and the Day 31 registry in ../day31/mlflow.db (read only).
2. Run snippet 1: confirm simulate(1000, 30) rebuilds the Day 30 table exactly, load the @champion, and simulate 12 live weeks. Score each stay with the patient's history, then keep the window.
3. Run snippet 2: write psi() (reference decile bins plus a missing bin, one bin per category), run it with ks_2samp on every input and on the score, and measure the PSI of pure noise for one stable week.
4. Run snippet 3: AUC, Brier, O/E, slope and ECE for reference, before and after; draw the reliability diagram. Then map creatinine back to the old assay, rescore with history and see how much of the gap is left.
5. Run snippet 4a: fit the champion with class_weight="balanced", then repair it with CalibratedClassifierCV (sigmoid and isotonic, GroupKFold folds). Compare Brier and the reliability curves.
6. Run snippet 4b: keep the champion frozen, fit Platt and isotonic on weeks 7 to 9, judge on weeks 10 to 12. Count stays above the 25% threshold and expected vs actual readmits, and check O/E by discharge group.
7. Run snippet 5: build the weekly monitor table with alert rules (PSI 0.25, O/E outside 0.8 to 1.25), add 4-week windows, and log all 12 weeks to MLflow experiment readmit30_monitoring in day34/mlflow.db. Open mlflow ui --backend-store-uri sqlite:///mlflow.db and look at the psi_score chart.
8. Run snippet 6: run psi_snowflake.sql in SQLite and confirm it matches Python. Keep it and the model monitor SQL for the Snowflake week (Days 36 to 40).
9. Write day34/README.md: data drift vs concept drift, your PSI table, O/E before and after, what you would do and in what order (fix data, recalibrate, retrain), and that the data is simulated. Add day34/mlflow.db and day34/mlruns/ to .gitignore.
10. Commit and push (snippet 7): git add .gitignore day34, git commit -m "Day 34: ...", git push.

## Code
Exercise files: `exercise/sim_readmit.py`, `exercise/drift_calibration_readmit.py` (snippets 1 to 6 in one script), `exercise/psi_snowflake.sql`, `exercise/README.md`.

### 1. Go live: rebuild Day 30, load the champion, simulate 12 weeks with history
```python
# Day 34 setup: the Day 31 champion goes "live". Reference = the Day 30 test stays it was validated on.
# Live = 12 weeks of NEW simulated stays (Jul 6 to Sep 27, 2026). On Aug 17 three things change:
# older patients (new Medicare Advantage contract), a new creatinine assay that reads 25% high,
# and a transitional care program that cuts readmissions after Home health / SNF discharge.
import os, warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import GroupShuffleSplit
from sim_readmit import simulate                    # the Day 30 generator as a function, with drift knobs

raw = pd.read_csv("../day30/readmit_raw.csv", parse_dates=["admit_date"], dtype={"drg": str, "icd10": str})
same_as_day30 = simulate(1000, 30).equals(raw)          # the function rebuilds Day 30 exactly
tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
                      .split(raw, groups=raw.patient_id))   # identical split to Days 30-33
train, ref = raw.iloc[tr_idx], raw.iloc[te_idx]

mlflow.set_tracking_uri("sqlite:///../day31/mlflow.db")     # Day 31 registry, read only
champ = mlflow.sklearn.load_model("models:/readmit30_pipeline@champion")
def score(df): return champ.predict_proba(df.drop(columns="readmit30"))[:, 1]
ref = ref.assign(risk=score(ref))

# Simulate each period with 14 months of patient HISTORY, score with that history (the date features need
# prior visits), then keep only the stays admitted inside the period. Never score a window without lookback.
START, CHANGE, END = pd.Timestamp("2026-07-06"), pd.Timestamp("2026-08-17"), pd.Timestamp("2026-09-28")
HIST = {}
def period(seed, lo, hi, pid0, **drift):
    hist = simulate(6000, seed, start="2025-06-01", days=(END - pd.Timestamp("2025-06-01")).days, pid0=pid0, **drift)
    hist["risk"] = score(hist)
    HIST[seed] = hist                               # keep the history: any rescoring needs it again
    return hist[(hist.admit_date >= lo) & (hist.admit_date < hi)]
live = pd.concat([period(341, START, CHANGE, 100000).assign(period="before"),
                  period(342, CHANGE, END, 200000, age_mean=71, creat_report=1.25, care_effect=-1.5).assign(period="after")],
                 ignore_index=True)
live["week"] = (live.admit_date - START).dt.days // 7 + 1
before, after = live[live.period == "before"], live[live.period == "after"]
print(f"rebuilds Day 30 exactly: {same_as_day30} | reference (Day 30 test) {len(ref)} stays, readmit {ref.readmit30.mean():.3f}")
print(f"live stays: before Aug 17 {len(before)} (readmit {before.readmit30.mean():.3f}) | "
      f"after {len(after)} (readmit {after.readmit30.mean():.3f}) | {live.week.nunique()} weeks, "
      f"{len(live) / live.week.nunique():.0f} stays per week")
```

### 2. Data drift: PSI and KS for every input and the score, plus the weekly noise floor
```python
# Data drift: compare each input (and the score) with the reference. No labels needed, so you can run it daily.
from scipy.stats import ks_2samp

def psi(ref_col, cur_col, bins=10, eps=1e-4):
    """Population Stability Index. Numeric: decile edges from the REFERENCE; categorical: one bin per level.
    Missing values get their own bin (a lab that stops being drawn is drift too)."""
    r, c = pd.Series(ref_col), pd.Series(cur_col)
    if pd.api.types.is_numeric_dtype(r):
        edges = np.unique(np.nanquantile(r, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = pd.cut(r, edges).astype(str).where(r.notna(), "missing")
        c = pd.cut(c, edges).astype(str).where(c.notna(), "missing")
    p = r.value_counts(normalize=True)
    q = c.value_counts(normalize=True).reindex(p.index.union(c.unique()), fill_value=0)
    p = p.reindex(q.index, fill_value=0).clip(lower=eps); q = q.clip(lower=eps)
    return float(((q - p) * np.log(q / p)).sum())

def band(v): return "stable" if v < 0.10 else ("watch" if v < 0.25 else "ALERT")
NUM = ["age", "los_days", "creatinine", "hemoglobin", "sodium", "risk"]
CAT = ["payer", "admit_type", "discharge"]
rows = []
for col in NUM + CAT:
    for name, cur in [("before", before), ("after", after)]:
        v = psi(ref[col], cur[col])
        ks = ks_2samp(ref[col].dropna(), cur[col].dropna()) if col in NUM else None
        rows.append({"feature": col, "period": name, "psi": round(v, 3), "band": band(v),
                     "ks_stat": None if ks is None else round(ks.statistic, 3),
                     "ks_p": None if ks is None else float(f"{ks.pvalue:.2g}")})
drift = pd.DataFrame(rows)
print(drift.pivot(index="feature", columns="period", values=["psi", "ks_stat"]).loc[NUM + CAT].to_string())
psi_noise = np.mean([psi(ref.risk, before.sample(len(before) // 6, random_state=s).risk) for s in range(200)])
print(f"PSI of pure sampling noise for ONE stable week (~{len(before) // 6} stays): {psi_noise:.3f}  "
      f"| mean age ref {ref.age.mean():.1f} -> after {after.age.mean():.1f} | median creatinine "
      f"{ref.creatinine.median():.2f} -> {after.creatinine.median():.2f}")
```

### 3. Calibration: AUC, Brier, O/E, slope, ECE, reliability diagram, assay fix
```python
# Performance and calibration once the 30-day outcomes are in. Discrimination (AUC) asks "are risky stays
# ranked above safe ones?" Calibration asks "when the model says 30%, do about 30% come back?"
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

def ece(y, p, bins=10):
    """Expected calibration error: weighted gap between predicted and observed rate in quantile bins."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q")
    return float((g.size() / len(p) * (g.p.mean() - g.y.mean()).abs()).sum())

def cal_stats(y, p):
    """Calibration-in-the-large (O/E), plus slope and intercept from a logistic fit of y on logit(p).
    Perfect: O/E 1, slope 1, intercept 0. Slope < 1 means predictions are too extreme."""
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(lg, y)
    return {"n": len(y), "auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 4),
            "brier_ref": round(brier_score_loss(y, np.full(len(y), y.mean())), 4),   # "always say the base rate"
            "mean_pred": round(p.mean(), 3), "observed": round(y.mean(), 3), "o_e": round(y.mean() / p.mean(), 2),
            "slope": round(float(m.coef_[0, 0]), 2), "intercept": round(float(m.intercept_[0]), 2), "ece": round(ece(y, p), 3)}

perf = pd.DataFrame({name: cal_stats(d.readmit30, d.risk) for name, d in
                     [("reference", ref), ("before", before), ("after", after)]}).T
print(perf.to_string())
curves = {name: calibration_curve(d.readmit30, d.risk, n_bins=10, strategy="quantile")
          for name, d in [("reference", ref), ("before", before), ("after", after)]}

# Which change did what? Undo the assay change (divide by 1.25, what a lab mapping table would do) and rescore.
hist_fix = HIST[342].copy(); hist_fix["creatinine"] = (hist_fix.creatinine / 1.25).round(2)
hist_fix["risk"] = score(hist_fix.drop(columns="risk"))
after_fix = hist_fix[(hist_fix.admit_date >= CHANGE) & (hist_fix.admit_date < END)].copy()
after_fix["week"] = (after_fix.admit_date - START).dt.days // 7 + 1
print(f"after, creatinine mapped back to the old assay: mean risk {after.risk.mean():.3f} -> {after_fix.risk.mean():.3f} "
      f"| O/E {cal_stats(after.readmit30, after.risk)['o_e']} -> {cal_stats(after_fix.readmit30, after_fix.risk)['o_e']} "
      f"| PSI creatinine {psi(ref.creatinine, after_fix.creatinine):.3f}")
```

### 4. Recalibration: CalibratedClassifierCV (sigmoid, isotonic), FrozenEstimator, thresholds
```python
# Calibration fixes. (a) At training time: a common habit, class_weight="balanced", inflates every risk.
# CalibratedClassifierCV (Platt = "sigmoid", or "isotonic") repairs it using out-of-fold predictions.
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

Xtr, ytr = train.drop(columns="readmit30"), train.readmit30
Xref = ref.drop(columns=["readmit30", "risk"])
folds = list(GroupKFold(5).split(Xtr, ytr, groups=train.patient_id))   # patients never cross folds
bal = clone(champ).set_params(clf__class_weight="balanced").fit(Xtr, ytr)
cal_models = {"champion (as is)": champ, "balanced weights": bal,
              "balanced + sigmoid": CalibratedClassifierCV(clone(bal), method="sigmoid", cv=folds).fit(Xtr, ytr),
              "balanced + isotonic": CalibratedClassifierCV(clone(bal), method="isotonic", cv=folds).fit(Xtr, ytr)}
cal_tab = pd.DataFrame({k: cal_stats(ref.readmit30, m.predict_proba(Xref)[:, 1]) for k, m in cal_models.items()}).T
print(cal_tab[["auc", "brier", "mean_pred", "observed", "o_e", "slope", "ece"]].to_string())
p_bal = bal.predict_proba(Xref)[:, 1]
p_sig = cal_models["balanced + sigmoid"].predict_proba(Xref)[:, 1]

# (b) After deployment: keep the champion FROZEN and learn only a new score -> probability map on recent labels.
# Fix the data first (creatinine mapped back), then recalibrate on weeks 7-9 and judge on weeks 10-12.
calib, evalw = after_fix[after_fix.week <= 9], after_fix[after_fix.week >= 10]
def lg(p):                                                         # log-odds as a 1-column matrix
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)
platt = LogisticRegression(C=1e6).fit(lg(calib.risk), calib.readmit30)      # Platt scaling on the log-odds
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(calib.risk, calib.readmit30)
recal = {"champion (raw)": evalw.risk.to_numpy(), "Platt (sigmoid)": platt.predict_proba(lg(evalw.risk))[:, 1],
         "isotonic": iso.predict(evalw.risk)}
recal_tab = pd.DataFrame({k: cal_stats(evalw.readmit30, p) for k, p in recal.items()}).T
print(f"recalibrate on weeks 7-9 ({len(calib)} stays), judge on weeks 10-12 ({len(evalw)} stays):")
print(recal_tab[["auc", "brier", "mean_pred", "observed", "o_e", "slope", "ece"]].to_string())

# The same Platt map through sklearn: CalibratedClassifierCV(FrozenEstimator(model)) on rows the model can score.
# (Live stays need their history to be scored, so on live data we calibrate the logged scores directly.
# The small gap is expected: sklearn uses Platt's smoothed targets, (n1 + 1) / (n1 + 2) and 1 / (n0 + 2), not 1 and 0.)
cc = CalibratedClassifierCV(FrozenEstimator(champ), method="sigmoid").fit(Xref, ref.readmit30)
manual = LogisticRegression(C=1e6).fit(lg(ref.risk), ref.readmit30).predict_proba(lg(ref.risk))[:, 1]
print(f"CalibratedClassifierCV(FrozenEstimator) vs manual Platt on the same rows: max diff "
      f"{np.abs(cc.predict_proba(Xref)[:, 1] - manual).max():.1e}")

# Why it matters: care teams act on thresholds and plan on expected counts.
THR = 0.25
p_new = recal["Platt (sigmoid)"]
use = pd.DataFrame({"version": ["raw", "Platt"], "flagged_ge_25pct": [int((evalw.risk >= THR).sum()), int((p_new >= THR).sum())],
                    "readmit_rate_flagged": [round(evalw.readmit30[evalw.risk >= THR].mean(), 3), round(evalw.readmit30[p_new >= THR].mean(), 3)],
                    "expected_readmits": [round(evalw.risk.sum(), 1), round(p_new.sum(), 1)], "actual_readmits": int(evalw.readmit30.sum())})
print(use.to_string(index=False))
sub = (evalw.assign(p=p_new).groupby(evalw.discharge.isin(["Home health", "SNF"]).map({True: "Home health/SNF", False: "Home/AMA"}))
       .apply(lambda g: pd.Series({"n": len(g), "o_e_after_platt": round(g.readmit30.mean() / g.p.mean(), 2)})))
print(sub.to_string())
```

### 5. A weekly monitor with alert rules, logged to MLflow
```python
# A weekly monitoring job: drift every week (no labels needed), calibration once outcomes mature (30+ days).
# Alert rules are written down BEFORE you look, and every week is logged to MLflow next to the model.
LAB_NUM = ["age", "creatinine", "los_days", "hemoglobin", "sodium"]
def week_report(cur, ref=ref):
    feat_psi = {c: psi(ref[c], cur[c]) for c in LAB_NUM + CAT}
    worst = max(feat_psi, key=feat_psi.get)
    r = {"n": len(cur), "psi_score": psi(ref.risk, cur.risk), "worst_feature": worst, "psi_worst": feat_psi[worst],
         "psi_age": feat_psi["age"], "psi_creatinine": feat_psi["creatinine"],
         "mean_risk": cur.risk.mean(), "observed": cur.readmit30.mean(), "auc": roc_auc_score(cur.readmit30, cur.risk)}
    r["o_e"] = r["observed"] / r["mean_risk"]
    flags = []
    if r["psi_worst"] >= 0.25: flags.append(f"input drift ({worst})")
    if r["psi_score"] >= 0.25: flags.append("score drift")
    if not 0.8 <= r["o_e"] <= 1.25: flags.append("calibration")
    r["status"] = "ALERT: " + ", ".join(flags) if flags else "ok"
    return r

weekly = pd.DataFrame([{"week": w, "week_start": (START + pd.Timedelta(weeks=w - 1)).date(), **week_report(g)}
                       for w, g in live.groupby("week")])
# Weekly samples are small, so also look at a rolling 4-week window (less noise, slower to react).
weekly["psi_creat_4wk"] = [psi(ref.creatinine, live[live.week.between(w - 3, w)].creatinine) for w in weekly.week]
weekly["o_e_4wk"] = [live[live.week.between(w - 3, w)].pipe(lambda d: d.readmit30.mean() / d.risk.mean()) for w in weekly.week]
show = weekly[["week", "week_start", "n", "psi_score", "psi_age", "psi_creatinine", "psi_creat_4wk", "o_e", "o_e_4wk", "auc", "status"]]
print(show.round(3).to_string(index=False))
first_alert = weekly.loc[weekly.status != "ok", "week"].min()
print(f"first ALERT week: {first_alert} | weeks with an alert before Aug 17: {int((weekly[weekly.week <= 6].status != 'ok').sum())} of 6"
      f" | after: {int((weekly[weekly.week >= 7].status != 'ok').sum())} of 6")
weekly.to_csv("monitor_weekly.csv", index=False)

mlflow.set_tracking_uri("sqlite:///mlflow.db")             # Day 34's own tracking store (the Day 31 one stays read only)
mlflow.set_experiment("readmit30_monitoring")
with mlflow.start_run(run_name="champion_v1_weekly_2026-07-06_to_2026-09-27") as run:
    mlflow.log_params({"model": "models:/readmit30_pipeline@champion", "reference": "day30 test stays (658)",
                       "psi_bins": 10, "alert_psi": 0.25, "alert_o_e": "outside 0.8 to 1.25",
                       "platt_a": round(float(platt.coef_[0, 0]), 4), "platt_b": round(float(platt.intercept_[0]), 4)})
    for _, r in weekly.iterrows():
        mlflow.log_metrics({k: float(r[k]) for k in ["n", "psi_score", "psi_age", "psi_creatinine", "o_e", "auc"]}, step=int(r.week))
    mlflow.log_artifact("monitor_weekly.csv")
    mlflow.set_tag("decision", "fix creatinine mapping, Platt recalibrate now, retrain with post-program labels")
n_logged = len(mlflow.MlflowClient().get_metric_history(run.info.run_id, "psi_score"))
print(f"MLflow run {run.info.run_id[:8]} in readmit30_monitoring: {n_logged} weekly psi_score points logged")
```

### 6. Snowflake preview: PSI in SQL (checked in SQLite) and a Model Monitor
```python
# Snowflake preview for Days 36-40: PSI as plain SQL, checked here in SQLite against the Python psi().
# On your trial: load the same tables, run psi_snowflake.sql in a worksheet (bind :start_date or paste a date).
import sqlite3
edges = np.unique(np.nanquantile(ref.risk, np.linspace(0, 1, 11)))
edges[0], edges[-1] = -1e9, 1e9
db = sqlite3.connect(":memory:")
pd.DataFrame({"bin": range(len(edges) - 1), "lo": edges[:-1], "hi": edges[1:]}).to_sql("PSI_EDGES", db, index=False)
ref[["risk"]].to_sql("REF_SCORES", db, index=False)
live.assign(admit_date=live.admit_date.dt.strftime("%Y-%m-%d"))[["admit_date", "risk"]].to_sql("LIVE_SCORES", db, index=False)
sql = open("psi_snowflake.sql").read()
psi_sql = db.execute(sql, {"start_date": "2026-08-17"}).fetchone()[0]
print(f"PSI of the score after Aug 17: SQL {psi_sql:.4f} | Python {psi(ref.risk, after.risk):.4f}")

# Snowflake ML Observability does this for you once the model sits in the Model Registry (not run here):
MONITOR_SQL = """
CREATE MODEL MONITOR readmit30_monitor WITH
    MODEL = readmit30_pipeline  VERSION = 'V1'  FUNCTION = 'predict_proba'
    SOURCE = READMIT_SCORES  BASELINE = READMIT_BASELINE
    WAREHOUSE = COMPUTE_WH  REFRESH_INTERVAL = '1 day'  AGGREGATION_WINDOW = '7 days'
    TIMESTAMP_COLUMN = ADMIT_DATE  ID_COLUMNS = ('STAY_ID')
    PREDICTION_SCORE_COLUMNS = ('RISK')  ACTUAL_CLASS_COLUMNS = ('READMIT30');
SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC('READMIT30_MONITOR', 'POPULATION_STABILITY_INDEX', 'RISK', '7 DAY',
                                               '2026-07-06'::TIMESTAMP, '2026-09-28'::TIMESTAMP));"""
```

### 6b. psi_snowflake.sql (Snowflake SQL; the same file runs in SQLite)
```sql
-- PSI of the risk score as one SQL query (Snowflake syntax; also runs in SQLite for the local check).
-- In a Snowflake worksheet: SET start_date = '2026-08-17'; then write $start_date where this file says :start_date.
-- PSI_EDGES holds the 10 REFERENCE decile bins (lo, hi]; REF_SCORES = validation stays; LIVE_SCORES = new stays.
WITH ref AS (
    SELECT e.bin, COUNT(r.risk) * 1.0 / (SELECT COUNT(*) FROM REF_SCORES) AS p
    FROM PSI_EDGES e LEFT JOIN REF_SCORES r ON r.risk > e.lo AND r.risk <= e.hi
    GROUP BY e.bin),
cur AS (
    SELECT e.bin, COUNT(s.risk) * 1.0 / (SELECT COUNT(*) FROM LIVE_SCORES WHERE admit_date >= :start_date) AS q
    FROM PSI_EDGES e LEFT JOIN LIVE_SCORES s ON s.risk > e.lo AND s.risk <= e.hi AND s.admit_date >= :start_date
    GROUP BY e.bin),
clipped AS (
    SELECT ref.bin,
           CASE WHEN p < 0.0001 THEN 0.0001 ELSE p END AS p,
           CASE WHEN q < 0.0001 THEN 0.0001 ELSE q END AS q
    FROM ref JOIN cur ON ref.bin = cur.bin)
SELECT ROUND(SUM((q - p) * LN(q / p)), 4) AS psi_score FROM clipped;
```

### 7. Push to GitHub
```bash
cd ~/ds-genai-90-days
printf 'day34/mlflow.db\nday34/mlruns/\n' >> .gitignore   # tracking store stays local
git add .gitignore day34
git commit -m "Day 34: drift (PSI, KS) and calibration (reliability, Brier, Platt) on the readmission model"
git push
```

## Monitoring playbook (note)
Drift checks need no labels, so run them daily or weekly on inputs AND on the score. Calibration needs outcomes, so for 30-day readmission it lags by a month. Write the alert rules before you look, decide on 4-week windows, not single weeks, and respond in order: fix the data feed (units, assays, mappings), recalibrate the frozen model on recent labels (Platt for a few hundred rows, isotonic for 1,000+), and retrain when a subgroup stays off. Log every check next to the model version (MLflow here, Snowflake Model Monitor on a real account).

## Tomorrow (Day 35)
Week 5 recap: feature pipelines, MLflow, Snowpark, Snowpark ML and monitoring as one story you can tell in an interview, with a self test. Then Days 36 to 40: a hands-on Snowflake week on the real trial account.

## Printables
- Cheat sheet: `day34_v1_cheat_sheet.png`
- Infographic: `day34_v2_infographic.png`
- Code worksheet: `day34_v3_worksheet.png`
- Practice quiz: `day34_v4_quiz.png`
- PDF: `day34_lesson.pdf`

## Podcast script
(Audio: `day34_podcast.mp3`)

Welcome to day thirty four of the Data Analyst to Data Scientist plan. Today we ask the question that comes after every deployment: is the model still healthy? We take the readmission champion from day thirty one and let it score twelve weeks of brand new simulated hospital stays.

First, two words you will hear in every interview. Data drift means the inputs change. The patients get older, or the lab switches to a new assay that reads higher. Concept drift means the relationship changes. The same patient, with the same numbers, now has a different chance of coming back, for example because the hospital started a transitional care program. In our simulation all three changes start on August seventeenth.

How do you catch data drift? With the population stability index, or PSI. You cut the reference data into ten equal bins, then you compare the share of new stays in each bin with the share you expected. Under point one is stable, point one to point two five is worth watching, and above point two five you act. After August seventeenth, age scored about point one nine and creatinine about point one seven, while discharge type stayed flat. The KS test agrees, but with over a thousand stays its p values get tiny even for small shifts, so judge the size of the shift.

Here is a surprise. The PSI of the risk score itself stayed low, around point zero five. So watching only the score can hide drift in the inputs. And no input check at all can see the care program, because the discharge mix never moved. Only the outcomes reveal concept drift.

That brings us to calibration. A risk score is a promise. When the model says thirty percent, about thirty out of a hundred similar patients should come back. A reliability diagram plots predicted risk against what actually happened, and a well calibrated model sits on the diagonal. The Brier score is the average squared gap between probability and outcome. Before the changes, observed to expected was close to one. After, it dropped to point seven. The model was predicting about one point four times the readmissions that really happened, while AUC moved only a little. Ranking can look fine while the probabilities are wrong.

Why does that matter in a hospital? Care teams act on thresholds, like call every patient above twenty five percent, and managers staff on expected counts. With the drifted model, one hundred eighty two stays crossed that line in the last three weeks. After recalibration, one hundred sixteen did, and the expected readmissions moved much closer to the actual ninety one.

So what do you do, and in what order? First fix the data. Mapping creatinine back to the old assay took observed to expected from point seven to point eight. Second, recalibrate. Platt scaling, the sigmoid method in scikit-learn's CalibratedClassifierCV, learns just two numbers on recent labels, and with a FrozenEstimator you keep the deployed model untouched. On the held out weeks it raised observed to expected to about point eight six and brought the calibration slope near one. Isotonic is more flexible, but with under seven hundred stays it overfit. Third, retrain. Patients discharged to home health or a skilled nursing facility were still badly over predicted, and only a retrain on post program labels fixes that.

One more lesson. We also tried class weight balanced, a popular trick for rare outcomes. It inflated every risk to an average of forty five percent when the real rate was seventeen. Calibration brought the Brier score right back. So if you reweight classes, always calibrate.

Finally, we built a weekly monitor with alert rules written in advance, logged every week to MLflow, and wrote PSI as plain SQL that matched Python exactly. Notice that week four raised a false alarm. Weekly samples are noisy, so make decisions on four week windows.

All the data today is simulated for teaching. Your exercise is in the day thirty four folder of your repo. Tomorrow is day thirty five, our week five recap. We will tie pipelines, MLflow, Snowpark and today's monitoring into one interview story, with a quick self test. Then the Snowflake hands on week begins on your real trial account. See you tomorrow.
