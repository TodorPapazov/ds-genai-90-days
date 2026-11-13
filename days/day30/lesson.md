# Day 30: Custom scikit-learn transformers and one full pipeline from raw table to prediction on 30-day readmission

## Concept
Healthcare data science postings ask for production-ready scikit-learn pipelines, reproducible work and comfort with ICD/DRG codes. Today you write your own pipeline steps. A custom transformer is a class (TransformerMixin, BaseEstimator): __init__ only stores parameters, fit learns state into attributes ending in _ and returns self, transform uses only that learned state, and get_feature_names_out names the output so set_output(transform="pandas") works. Use FunctionTransformer for stateless math (log1p, ICD-10 code to chapter letter) and a class when something must be learned: a FrequencyEncoder (each DRG code becomes its share of training stays, unseen codes become 0) and a DateFeatures step (weekday, month, weekend, winter, prior visits, days since the same patient's last stay, capped at a quantile learned in fit). On 2,655 simulated stays from 1,000 patients (seed 30, readmit rate 17.8%), one pipeline takes the 12 raw columns to a probability. With GroupKFold by patient, DRG rare one-hot scored CV ROC AUC 0.723 (53 columns), the custom frequency encoder 0.724 (39 columns) and both 0.724 (54 columns); the rule (simplest within 1 sd) picks the frequency encoder, test AUC opened once 0.743. Leak check: in all 5 folds the encoder's learned shares match the fit fold exactly (max diff 0) and differ from all of train by up to 0.0099; the date cap differs per fold (433 to 455 days). check_estimator first failed on mixin order, then 45 checks passed. The pipeline is saved with joblib, reloads with identical predictions, and is ready for MLflow tomorrow. All data is simulated for teaching.

## Key numbers
- **Data:** 2,655 simulated stays, 1,000 patients (seed 30), readmit rate 17.8%, split by patient 1,997/658
- **Pipeline:** 12 raw columns in, 39 features out (champion), set_output(transform='pandas')
- **CV ROC AUC:** DRG rare one-hot 0.723 (53 cols), FrequencyEncoder 0.724 (39 cols), both 0.724 (54 cols); fold sd about 0.03 to 0.04
- **Rule + test:** best 0.724 - 1 sd = 0.693; simplest qualifying = FrequencyEncoder; test AUC 0.743
- **Leak check:** learned DRG shares vs fit fold: max diff 0 in all 5 folds; vs all train: up to 0.0099; date cap 433 to 455 days per fold
- **API + save:** check_estimator: 45 passed, 0 failed (after fixing mixin order); joblib reload gives identical predictions
- **Gotcha:** a return stay scored alone: risk 0.072; with the patient's earlier stay: 0.104

## Hands-on (60-90 min)
1. In your repo, create day30/, run conda activate ds90, and open the folder in VS Code with a notebook day30/readmit_pipeline.ipynb.
2. Run snippet 1 to simulate 2,655 raw stays from 1,000 patients (seed 30), save readmit_raw.csv, split by patient and set up GroupKFold(5). No engineered columns yet.
3. Save snippet 2 as day30/readmit_transformers.py: FrequencyEncoder, DateFeatures and the named function icd_chapter. In a markdown cell, explain when FunctionTransformer is enough and when you need a class.
4. Run snippet 3: assemble ONE pipeline with build() (custom classes plus Day 29 recipes inside a ColumnTransformer, then logistic regression) and inspect the pandas output from set_output.
5. Write your selection rule in a markdown cell before racing: best CV AUC, then the simplest recipe within 1 sd.
6. Run snippet 4: race three DRG options with GroupKFold by patient and run the leak check (learned shares vs fit-fold shares in every fold).
7. Run snippet 5: check_estimator, apply the rule, open the test set once, save with joblib, reload, predict raw rows, and score a return stay alone vs with history.
8. Write day30/README.md: a pipeline diagram, the CV table, the leak check, the history gotcha, and limits (simulated data).
9. Commit and push (snippet 6): git add day30, git commit -m "Day 30: ...", git push.
10. Stretch: build the train-only DRG frequency table and the date features in Snowflake SQL, or with a Snowpark DataFrame (snippet 7), and compare a few rows with pandas.

## Code
### 1. Simulate the RAW readmission table and split by patient (seed 30)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Simulated 30-day readmission data (seed 30): 1,000 patients with 1 to 6 stays each.
# Real-world shape: skewed labs with gaps, payer and disposition categories, ~100 DRG codes with a long tail.
rng = np.random.default_rng(30)
n_pat = 1000
visits = np.clip(1 + rng.poisson(1.6, n_pat), 1, 6)
pid = np.repeat(np.arange(n_pat), visits)
n = len(pid)
frailty = np.repeat(rng.normal(0, 1, n_pat), visits)                     # hidden, never a feature
age = np.repeat(rng.normal(64, 15, n_pat).clip(18, 99).round(), visits)

# admit dates: first stay in 2024-01 to 2025-06, later stays after a gap that is shorter for frail patients
first = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 546, n_pat), unit="D")
gap = rng.exponential(110 * np.exp(-0.5 * frailty)) + 3
order = pd.Series(pid).groupby(pid).cumcount().values
offset = pd.Series(np.where(order == 0, 0, gap)).groupby(pid).cumsum().values
admit_date = np.repeat(first.values, visits) + pd.to_timedelta(offset.round(), unit="D").values

# DRG codes: a few big ones (heart failure 291, sepsis 871, joint replacement 470, COPD 190) and a long tail
top = ["291", "871", "470", "190", "392", "683", "690", "193", "603", "378"]
tail = [f"{c:03d}" for c in rng.choice(np.setdiff1d(np.arange(1, 999), np.array(top, int)), 90, replace=False)]
drg_codes = np.array(top + tail)
p = 1 / np.arange(1, len(drg_codes) + 1) ** 1.1; p /= p.sum()
drg = rng.choice(drg_codes, n, p=p)
drg_eff = dict(zip(drg_codes, rng.normal(0, 0.35, len(drg_codes))))   # tail codes: small, noisy effects
drg_eff.update({"291": 0.7, "871": 0.6, "470": -1.0, "190": 0.6, "392": -0.3, "683": 0.4})
chap_of = {"291": "I", "871": "A", "470": "M", "190": "J", "392": "K", "683": "N", "690": "N", "193": "J", "603": "L", "378": "K"}
chapters = np.array(list("IJKNAEMFL"))
chap = np.array([chap_of.get(d, chapters[int(d) % 9]) for d in drg])
icd10 = np.array([f"{c}{rng.integers(10, 99)}.{rng.integers(0, 9)}" for c in chap])   # e.g. I50.9

payer = np.where(age >= 65, rng.choice(["Medicare", "Commercial", "Medicaid"], n, p=[.85, .1, .05]),
                 rng.choice(["Commercial", "Medicaid", "Self-pay"], n, p=[.6, .3, .1]))
admit_type = np.where(drg == "470", "Elective", rng.choice(["Emergency", "Urgent", "Elective"], n, p=[.62, .2, .18]))
discharge = rng.choice(["Home", "Home health", "SNF", "AMA"], n, p=[.6, .18, .17, .05])
los_days = np.round(np.exp(1.2 + 0.25 * frailty + rng.normal(0, 0.5, n)), 1)
creatinine = np.round(np.exp(0.05 + 0.25 * frailty + rng.normal(0, 0.4, n)), 2)          # right-skewed
hemoglobin = np.round(12.6 - 0.8 * frailty + rng.normal(0, 1.4, n), 1)
sodium = np.round(139 + rng.normal(0, 3.5, n))
creatinine[rng.random(n) < np.where(admit_type == "Elective", 0.25, 0.05)] = np.nan       # missing is NOT random
hemoglobin[rng.random(n) < 0.10] = np.nan
sodium[rng.random(n) < 0.06] = np.nan

df = pd.DataFrame({"patient_id": pid, "admit_date": admit_date, "age": age, "payer": payer, "admit_type": admit_type,
                   "discharge": discharge, "drg": drg, "icd10": icd10, "los_days": los_days,
                   "creatinine": creatinine, "hemoglobin": hemoglobin, "sodium": sodium, "_frailty": frailty})
df = df.sort_values(["patient_id", "admit_date"]).reset_index(drop=True)
gap_true = df.groupby("patient_id").admit_date.diff().dt.days
dow, mon, hgb = df.admit_date.dt.dayofweek, df.admit_date.dt.month, df.hemoglobin.fillna(12.5)
logit = (-3.1 + 0.012 * (df.age - 64) + 0.7 * np.log(df.creatinine.fillna(1.0)) + 1.1 * (hgb < 10) + 0.3 * hgb.between(10, 11.5)
         + df.drg.map(drg_eff) + df.discharge.map({"Home": 0, "Home health": .25, "SNF": .45, "AMA": 1.0})
         + 0.35 * (df.payer == "Medicaid") + 0.4 * np.log(df.los_days) + 0.25 * (dow >= 5) + 0.2 * mon.isin([12, 1, 2])
         + np.where(gap_true < 30, 0.9, np.where(gap_true < 90, 0.35, 0)) + 0.3 * df._frailty)
df["readmit30"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = df.drop(columns="_frailty")
df.to_csv("readmit_raw.csv", index=False)

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=30)
tr_idx, te_idx = next(gss.split(df, groups=df.patient_id))          # patients never cross the split
X = df.drop(columns="readmit30")                                    # the RAW table: no engineered columns yet
train, test = X.iloc[tr_idx], X.iloc[te_idx]
ytr, yte, gtr = df.readmit30.iloc[tr_idx], df.readmit30.iloc[te_idx], train.patient_id
gkf = GroupKFold(5)
print(df.shape, "readmit rate", round(df.readmit30.mean(), 3), "DRG levels", df.drg.nunique(),
      "train", len(train), "test", len(test))
print("raw columns:", list(X.columns))
```

### 2. day30/readmit_transformers.py: FrequencyEncoder, DateFeatures, icd_chapter
```python
# day30/readmit_transformers.py  (a real .py file, so joblib and MLflow can import the classes later)
import numpy as np
import pandas as pd
# Mixin order matters: TransformerMixin LEFT of BaseEstimator (check_estimator enforces it).
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category with its share of the FIT rows. Codes never seen in fit get 0."""

    def __init__(self, min_count=1):
        self.min_count = min_count          # store params as-is in __init__, no logic here

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=None, ensure_all_finite=False)   # sets n_features_in_, feature_names_in_
        self.freq_ = []
        for col in X.T:
            counts = pd.Series(col).value_counts()
            counts = counts[counts >= self.min_count]
            self.freq_.append((counts / len(col)).to_dict())               # learned state ends with "_"
        return self

    def transform(self, X):
        check_is_fitted(self, "freq_")
        X = validate_data(self, X, dtype=None, ensure_all_finite=False, reset=False)
        return np.column_stack([pd.Series(col).map(f).fillna(0.0).to_numpy(float)
                                for col, f in zip(X.T, self.freq_)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "freq_")
        names = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        return np.array([f"{n}_freq" for n in names], dtype=object)

    def __sklearn_tags__(self):                 # tell check_estimator what input we accept on purpose
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True        # a missing code just maps to 0
        tags.input_tags.string = True           # DRG codes are strings like "291"
        return tags


class DateFeatures(TransformerMixin, BaseEstimator):
    """Backward-looking date features from (patient_id, admit_date). Learns one thing in fit:
    a cap for days_since_last (a high quantile of the FIT rows), also used to fill first visits."""

    def __init__(self, cap_quantile=0.95):
        self.cap_quantile = cap_quantile

    def _gaps(self, X):
        X = pd.DataFrame(X, columns=["patient_id", "admit_date"]).copy()
        X["admit_date"] = pd.to_datetime(X["admit_date"])
        s = X.sort_values(["patient_id", "admit_date"])
        gap = s.groupby("patient_id")["admit_date"].diff().dt.days      # only looks BACKWARD in time
        prior = s.groupby("patient_id").cumcount()
        return X, gap.reindex(X.index), prior.reindex(X.index)

    def fit(self, X, y=None):
        _, gap, _ = self._gaps(X)
        self.n_features_in_ = 2
        self.cap_ = float(gap.quantile(self.cap_quantile))
        return self

    def transform(self, X):
        check_is_fitted(self, "cap_")
        X, gap, prior = self._gaps(X)
        d = X["admit_date"]
        return pd.DataFrame({
            "admit_dow": d.dt.dayofweek, "weekend_admit": (d.dt.dayofweek >= 5).astype(int),
            "admit_month": d.dt.month, "winter": d.dt.month.isin([12, 1, 2]).astype(int),
            "first_visit": gap.isna().astype(int), "prior_visits": prior,
            "log_days_since_last": np.log1p(gap.fillna(self.cap_).clip(upper=self.cap_)),
        }, index=X.index).to_numpy(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["admit_dow", "weekend_admit", "admit_month", "winter",
                         "first_visit", "prior_visits", "log_days_since_last"], dtype=object)


def icd_chapter(X):
    """'I50.9' -> 'I'. A named function (not a lambda) so the pipeline can be pickled."""
    return pd.DataFrame(X).apply(lambda s: s.astype(str).str[0])
```

### 3. ONE pipeline from raw columns to probability, with pandas output
```python
# ONE pipeline from the raw table to a probability. Every step below is fit on the training fold only.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from readmit_transformers import FrequencyEncoder, DateFeatures, icd_chapter

log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")    # stateless: a function is enough


def build(drg="freq+rare"):
    parts = [
        ("dates", make_pipeline(DateFeatures(), StandardScaler()), ["patient_id", "admit_date"]),   # custom class
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
    if "freq" in drg:   # custom class: 1 column, the code's share of training stays
        parts.append(("drg_freq", make_pipeline(FrequencyEncoder(), StandardScaler()), ["drg"]))
    if "rare" in drg:   # Day 29 champion: common codes one-hot, rare codes pooled
        parts.append(("drg_oh", OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist",
                                              sparse_output=False), ["drg"]))
    pre = ColumnTransformer(parts, remainder="drop")   # patient_id is only used by DateFeatures, never as a number
    return Pipeline([("prep", pre), ("clf", LogisticRegression(C=0.5, max_iter=3000))]).set_output(transform="pandas")

pipe = build("freq+rare")
Xt = pipe[:-1].fit_transform(train, ytr)          # everything except the model
print(type(Xt).__name__, Xt.shape)
print(Xt.filter(like="dates__").head(3).round(2).to_string())
fe = pipe["prep"].named_transformers_["drg_freq"][0]       # the fitted FrequencyEncoder inside the pipeline
print("learned DRG shares (top 4):", {k: round(v, 3) for k, v in list(fe.freq_[0].items())[:4]}, "| codes learned:", len(fe.freq_[0]))
print("DateFeatures cap learned on train:", round(pipe["prep"].named_transformers_["dates"][0].cap_, 1), "days")
```

### 4. GroupKFold race and the leak check (custom steps learn only on the fit fold)
```python
# Validate the WHOLE pipeline with GroupKFold by patient, and prove the custom steps learn only on the fit fold.
from sklearn.model_selection import cross_validate
from sklearn.base import clone

rows, fold_auc = {}, {}
for name in ["rare", "freq", "freq+rare"]:
    cv = cross_validate(build(name), train, ytr, groups=gtr, cv=gkf, scoring=["roc_auc", "neg_log_loss"],
                        return_estimator=True)
    rows[name] = {"cv_auc": cv["test_roc_auc"].mean(), "auc_sd": cv["test_roc_auc"].std(),
                  "cv_logloss": -cv["test_neg_log_loss"].mean(),
                  "n_cols": cv["estimator"][0][:-1].get_feature_names_out().size}
    fold_auc[name] = cv["test_roc_auc"]
    if name == "freq+rare": fold_models = cv["estimator"]
board = pd.DataFrame(rows).T
print(board.round(3))

# Leakage check: inside each fold, the encoder's learned shares must equal the shares of the FIT rows only.
all_share = train.drg.value_counts(normalize=True)
checks = []
for k, ((fit_i, val_i), m) in enumerate(zip(gkf.split(train, ytr, gtr), fold_models)):
    fe_k = m["prep"].named_transformers_["drg_freq"][0]
    df_k = m["prep"].named_transformers_["dates"][0]
    learned = pd.Series(fe_k.freq_[0])
    fold_share = train.drg.iloc[fit_i].value_counts(normalize=True)
    unseen = set(train.drg.iloc[val_i]) - set(learned.index)
    checks.append({"fold": k + 1, "fit_rows": len(fit_i), "codes_learned": len(learned),
                   "max_diff_vs_fit_fold": (learned - fold_share).abs().max(),
                   "max_diff_vs_all_train": (learned - all_share).abs().max(),
                   "val_codes_unseen_to_0": len(unseen), "date_cap_days": round(df_k.cap_, 1)})
print(pd.DataFrame(checks).set_index("fold").to_string(float_format=lambda v: f"{v:.4f}"))
```

### 5. check_estimator, rule, test once, joblib save and reload, the history gotcha
```python
# check_estimator, pick by rule, test ONCE, save with joblib, reload and predict from raw rows.
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import roc_auc_score
import joblib, json, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = check_estimator(FrequencyEncoder(), on_fail=None)      # list of every API check and its status
status = pd.Series([r["status"] for r in res]).value_counts()
print("check_estimator(FrequencyEncoder()):", status.to_dict())

best = board.cv_auc.idxmax(); limit = board.loc[best, "cv_auc"] - board.loc[best, "auc_sd"]
champion = board[board.cv_auc >= limit].sort_values("n_cols").index[0]          # simplest within 1 sd
print("best:", best, "| limit", round(limit, 3), "| champion:", champion, int(board.loc[champion, "n_cols"]), "columns")

final = build(champion).fit(train, ytr)
p_te = final.predict_proba(test)[:, 1]
print(f"test AUC (opened once): {roc_auc_score(yte, p_te):.3f}")

joblib.dump(final, "readmit_pipeline.joblib")
loaded = joblib.load("readmit_pipeline.joblib")                 # needs readmit_transformers.py importable
print("reloaded == original:", np.allclose(loaded.predict_proba(test)[:, 1], p_te))
raw_rows = test.head(3)                                         # raw columns in, probability out
print(raw_rows[["patient_id", "admit_date", "drg", "creatinine"]].to_string())
print("risk:", loaded.predict_proba(raw_rows)[:, 1].round(3))

# Gotcha: days_since_last needs the patient's HISTORY in the batch. Score a return stay alone vs with history.
ret = test[test.patient_id.duplicated(keep=False)].groupby("patient_id").tail(1).iloc[[0]]
hist = test[test.patient_id == ret.patient_id.iloc[0]]
alone = loaded.predict_proba(ret)[0, 1]
with_hist = loaded.predict_proba(hist)[-1, 1]
print(f"patient {ret.patient_id.iloc[0]}, stay {len(hist)} of {len(hist)}: risk alone {alone:.3f} vs with history {with_hist:.3f}")

json.dump({"champion": champion, "features": list(final[:-1].get_feature_names_out()),
           "cv_auc": round(float(board.loc[champion, "cv_auc"]), 3),
           "test_auc": round(float(roc_auc_score(yte, p_te)), 3), "next": "log with MLflow on Day 31"},
          open("pipeline_card.json", "w"), indent=2)
```

### 6. Push to GitHub
```bash
git add day30
git commit -m "Day 30: custom transformers and one full pipeline on 30-day readmission"
git push
```

### 7. Stretch: the custom steps in Snowflake SQL and Snowpark
```sql
-- Snowflake SQL: the same two custom steps. Rule: anything LEARNED (frequencies, the cap) uses TRAIN rows only.
-- Assumes readmit_raw has a split column ('train' / 'test') assigned by patient.
CREATE OR REPLACE TABLE drg_freq AS                      -- FrequencyEncoder.fit
SELECT drg, COUNT(*) / SUM(COUNT(*)) OVER () AS drg_freq
FROM readmit_raw WHERE split = 'train' GROUP BY drg;

WITH dated AS (                                          -- DateFeatures: backward-looking only
  SELECT r.*,
         DAYOFWEEKISO(admit_date) - 1                    AS admit_dow,      -- 0 = Monday, like pandas
         MONTH(admit_date)                               AS admit_month,
         DATEDIFF('day', LAG(admit_date) OVER (PARTITION BY patient_id ORDER BY admit_date),
                  admit_date)                            AS days_since_last,
         ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY admit_date) - 1 AS prior_visits
  FROM readmit_raw r),
cap AS (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY days_since_last) AS cap_days
        FROM dated WHERE split = 'train')                -- DateFeatures.fit learns the cap
SELECT d.*, COALESCE(f.drg_freq, 0)                      AS drg_freq,       -- unseen code -> 0
       IFF(d.days_since_last IS NULL, 1, 0)              AS first_visit,
       LN(1 + LEAST(COALESCE(d.days_since_last, c.cap_days), c.cap_days)) AS log_days_since_last
FROM dated d CROSS JOIN cap c LEFT JOIN drg_freq f USING (drg);

-- Snowpark (Python) version of FrequencyEncoder.fit, pushed down to Snowflake:
-- from snowflake.snowpark.functions import col, count, lit
-- tr = session.table("READMIT_RAW").filter(col("SPLIT") == "train")
-- n = tr.count()
-- freq = tr.group_by("DRG").agg((count(lit(1)) / n).alias("DRG_FREQ"))
-- freq.write.save_as_table("DRG_FREQ", mode="overwrite")
```

## Tomorrow (Day 31)
Track this exact pipeline with MLflow: log parameters, GroupKFold metrics and the fitted pipeline (with `readmit_transformers.py` as code), then compare runs in the MLflow UI.

## Printables
- Cheat sheet: `day30_v1_cheat_sheet.png`
- Infographic: `day30_v2_infographic.png`
- Code worksheet: `day30_v3_worksheet.png`
- Practice quiz: `day30_v4_quiz.png`
- PDF: `day30_lesson.pdf`

## Podcast script
(Audio: `day30_podcast.mp3`)

Hey Todor, welcome to day thirty. Yesterday we built a feature engineering toolkit out of scikit-learn parts. Today we write our own parts, and then we snap everything into one pipeline that goes from the raw table all the way to a prediction.

Why does this matter for a job? Healthcare data science postings keep asking for production-ready scikit-learn pipelines, reproducible work, and comfort with claims and EHR codes. A custom transformer is how you turn a smart idea about DRG codes or visit dates into a step that runs the same way in training, in cross validation, and on next month's data.

Here is the idea. A custom transformer is a small class with three habits. The init method only stores settings, nothing else. The fit method looks at the training rows, learns something, saves it with a name that ends in an underscore, and returns itself. And the transform method uses only what was learned. It never re-learns from the batch you are scoring. Add a method that names the output columns, and your step plays nicely with everything else, including pandas output.

When do you need a class at all? If a step is pure math, like a log transform, or cutting an ICD ten code down to its chapter letter, a function transformer is enough. If the step has to learn something from data, you want a class. We wrote two. A frequency encoder that replaces each DRG code with its share of training stays, so the heart failure DRG becomes about twenty two percent. Codes it never saw become zero, so next month's new codes do not crash anything. And a date features step that builds weekday, month, a weekend flag, a winter flag, prior visits, and days since the same patient's last stay. It learns one thing, a cap on that gap, and it gives first visits the cap plus a first visit flag.

We simulated about two thousand six hundred fifty stays from one thousand patients with a new seed, and about eighteen percent came back within thirty days. The data is simulated for teaching. The pipeline takes twelve raw columns and builds thirty nine features. We validated it with group k fold, so a patient never sits on both sides of a split.

Three versions of DRG went into the same race. Rare codes pooled with one hot encoding scored an AUC of about point seven two three. Our single frequency column scored point seven two four. Both together, also point seven two four. They are basically tied, so our rule picks the simplest one: the frequency encoder, with thirty nine columns instead of fifty three. On the test set, opened once, it scored point seven four three.

Now the part I really want you to remember. How do you prove a custom step did not leak? We kept the fitted pipeline from every fold and pulled out the encoder. In all five folds, its learned shares matched the fit fold exactly, a difference of zero, while they differed from the full training set by up to about one percentage point. The date step learned a different cap in each fold too, from about four hundred thirty three to four hundred fifty five days. That is what learning only on the fit fold looks like.

We also ran scikit-learn's estimator checker on the frequency encoder. The first run failed, because the parent classes were listed in the wrong order. The transformer mixin has to come before the base estimator. After fixing that and declaring that we accept text and missing values, forty five checks passed.

One gotcha from the real world. Days since last visit needs the patient's history. When we scored a return stay all by itself, it looked like a first visit and got a risk of about seven percent. Scored together with that patient's earlier stay, it was about ten percent. So at scoring time, send the history, or compute the gap in SQL.

Finally, we saved the whole pipeline with joblib, reloaded it, and got identical predictions from raw rows. Keep the custom classes in a real Python file, because the saved model only remembers where the class lives.

Your exercise is in the day thirty folder of your repo, with a Snowflake and Snowpark stretch. Tomorrow, on day thirty one, we track this exact pipeline with MLflow: parameters, metrics, and the model itself. See you then.
