# Day 27: Data leakage and pipelines inside cross-validation: honest scores on length of stay

## Concept
Week 4 Day 6 of classic ML. Leakage is any path by which information from validation or test rows reaches fit, so the score looks better than the model is. Four kinds on a fresh simulation of 2,400 encounters from 800 patients (3 stays each, seed 27), 8 clinical features, 300 pure-noise markers, 12% missing albumin. Selection leak: with a shuffled target (nothing to learn), picking the top 20 markers on all rows then running CV gave R² +0.077 (mean +0.17 over 20 shuffles); the same SelectKBest inside a Pipeline gave -0.394. Target leak: total_charges is billed after discharge; with it CV MAE is 0.439 days, with admission-time features only it is 1.478. Group leak: a random forest scored 1.514 with KFold but 1.578 with GroupKFold by patient. Preprocessing leak: imputing before CV changed MAE by under 0.001 days here: small, but the habit matters. Honest scorecard (Pipelines, GroupKFold on train patients only): champion ridge on clinical features. Test once on held-out patients: MAE 1.461 days, RMSE 1.853, R² 0.746. All data is simulated for teaching, so the numbers say nothing about real hospitals. Tomorrow: the week 4 regression scorecard and model card.

## Key numbers
- **Data:** 2,400 simulated encounters, 800 patients x 3 stays (seed 27), 8 clinical + 300 noise markers, split by patient 1,800/600
- **Selection leak (shuffled y):** select-then-CV R2 +0.077 vs pipeline -0.394; 20 shuffles: +0.17 vs -0.42
- **Target leak:** CV MAE 0.439 days with total_charges vs 1.478 admission-time only
- **Group leak (forest):** KFold 1.514 vs GroupKFold 1.578 days
- **Preprocessing leak:** impute before CV 1.4771 vs in pipeline 1.4776 (tiny)
- **Test set, once:** ridge, clinical: MAE 1.461 days, RMSE 1.853, R2 0.746

## Hands-on
1. In your repo, create day27/, run conda activate ds90, and open day27/leakage_los.ipynb in VS Code.
2. Run snippet 1 to simulate 2,400 encounters from 800 patients (seed 27) and save los_encounters.csv. Note the traps: noise markers, missing albumin, total_charges.
3. Run snippet 2: shuffle the target and compare select-then-CV against a Pipeline with SelectKBest inside.
4. In a markdown cell, explain why R² above 0 on a shuffled target means a leak.
5. Run snippet 3: split by patient with GroupShuffleSplit, build GroupKFold(5), and compare CV MAE with and without total_charges.
6. Write a short feature audit: for each column, is it known at admission? Yes, no, or unsure.
7. Run snippet 4: random forest with KFold vs GroupKFold, then impute-before-CV vs impute-in-pipeline. Record both gaps.
8. Run snippet 5: the four-candidate honest scorecard. Pick the champion by CV only and score the held-out patients once.
9. Write day27/README.md: the four leaks with numbers, your feature audit, the champion and why, limits (simulated data).
10. Commit and push (snippet 6).
11. Stretch: in Snowflake, flag features recorded after the prediction time (snippet 7).

## Code
### 1. Simulate encounters with built-in traps (seed 27)
```python
import numpy as np
import pandas as pd

# Fresh simulation (seed 27): 2,400 encounters from 800 patients (3 stays each).
# Patients carry a hidden "frailty" that makes all their stays longer.
# total_charges is billed AFTER discharge, so it secretly contains the answer.
rng = np.random.default_rng(27)
n_pat, per = 800, 3
pid = np.repeat(np.arange(n_pat), per)
n = n_pat * per
frailty = np.repeat(rng.normal(0, 1, n_pat), per)
age = np.repeat(rng.normal(66, 12, n_pat).clip(18, 98).round(), per)
bmi = np.repeat(rng.normal(29, 5, n_pat).clip(16, 55).round(1), per)   # fixed per patient
comorbidities = np.clip(np.round(np.repeat(rng.normal(2.2, 1.3, n_pat), per)
                         + 0.6 * frailty), 0, 10).astype(int)
acuity = np.clip(np.round(3 + 0.9 * rng.normal(0, 1, n) + 0.3 * frailty), 1, 5).astype(int)
procedure = (rng.random(n) < 0.30 + 0.06 * (acuity - 1)).astype(int)
icu_stay = (rng.random(n) < 0.06 + 0.05 * (acuity - 1)).astype(int)
emergency = (rng.random(n) < 0.50).astype(int)
albumin = (4.0 - 0.25 * frailty + rng.normal(0, 0.15, n)).round(2)
albumin[rng.random(n) < 0.12] = np.nan          # missing labs, like real EHR data

mu = (2.0 + 0.02 * (age - 66) + 0.8 * (acuity - 1) + 1.4 * procedure
      + 0.35 * comorbidities + 2.2 * icu_stay + 0.5 * emergency
      + 0.04 * (bmi - 29) + 3.0 * frailty - 0.6 * (np.nan_to_num(albumin, nan=4.0) - 4.0))
los_days = np.clip(mu + rng.normal(0, 1.0, n), 1.0, None).round(1)
total_charges = (2600 * los_days + 9000 * procedure + rng.normal(0, 1500, n)).round(-1)

df = pd.DataFrame({"patient_id": pid, "age": age, "bmi": bmi, "acuity": acuity, "procedure": procedure,
                   "comorbidities": comorbidities, "icu_stay": icu_stay, "emergency": emergency,
                   "albumin": albumin, "total_charges": total_charges, "los_days": los_days})
# 300 "biomarker" columns of pure noise, to tempt feature selection
noise = pd.DataFrame(rng.normal(0, 1, (n, 300)), columns=[f"marker_{i:03d}" for i in range(300)])
df = pd.concat([df, noise.round(3)], axis=1)
df.to_csv("los_encounters.csv", index=False)

clinical = ["age", "bmi", "acuity", "procedure", "comorbidities", "icu_stay", "emergency", "albumin"]
markers = list(noise.columns)
print(df.shape, "patients", df.patient_id.nunique(), "mean LOS", round(df.los_days.mean(), 2))
print("missing albumin", round(df.albumin.isna().mean(), 3))
print("corr(total_charges, los_days)", round(df.total_charges.corr(df.los_days), 3))
```

### 2. Selection leak: shuffled target, select-then-CV vs Pipeline
```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge

# LEAK 1: feature selection before CV. Sanity test: shuffle the target so there is NOTHING to learn.
small = df[df.patient_id < 60].copy()                  # 180 rows, 300 noise markers
Xs = small[markers].values
y_shuf = np.random.default_rng(27).permutation(small.los_days.values)
cv5 = KFold(5, shuffle=True, random_state=27)

sel = SelectKBest(f_regression, k=20).fit(Xs, y_shuf)   # WRONG: sees every row's target
leaky_r2 = cross_val_score(Ridge(1.0), sel.transform(Xs), y_shuf, cv=cv5, scoring="r2").mean()
pipe = make_pipeline(SelectKBest(f_regression, k=20), Ridge(1.0))  # RIGHT: refit per fold
honest_r2 = cross_val_score(pipe, Xs, y_shuf, cv=cv5, scoring="r2").mean()
print("shuffled target, select-then-CV R2:", round(leaky_r2, 3))
print("shuffled target, pipeline R2:      ", round(honest_r2, 3))
```

### 3. Target leak: split by patient, GroupKFold, total_charges
```python
# LEAK 2: target leakage. Ask of every column: would I know this at admission?
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=27)
tr_idx, te_idx = next(gss.split(df, groups=df.patient_id))   # patients never cross the split
train, test = df.iloc[tr_idx], df.iloc[te_idx]
gkf = GroupKFold(5)
lin = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(1.0))
mae = lambda m, cols: -cross_val_score(m, train[cols], train.los_days, groups=train.patient_id,
                                       cv=gkf, scoring="neg_mean_absolute_error").mean()
mae_with_charges = mae(lin(), clinical + ["total_charges"])
mae_admission = mae(lin(), clinical)
print("CV MAE with total_charges:", round(mae_with_charges, 3), "days (too good to be true)")
print("CV MAE admission-time only:", round(mae_admission, 3), "days")
```

### 4. Group leak and preprocessing leak
```python
# LEAK 3: the same patient in train and validation folds. A flexible model memorizes them.
from sklearn.ensemble import RandomForestRegressor
rf = lambda: make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestRegressor(300, n_jobs=-1, random_state=27))
Xc, yc, g = train[clinical], train.los_days, train.patient_id
kf_mae = -cross_val_score(rf(), Xc, yc, cv=KFold(5, shuffle=True, random_state=27),
                          scoring="neg_mean_absolute_error").mean()
gk_mae = -cross_val_score(rf(), Xc, yc, groups=g, cv=gkf, scoring="neg_mean_absolute_error").mean()
print("random forest, KFold (patients leak):", round(kf_mae, 3))
print("random forest, GroupKFold (honest): ", round(gk_mae, 3))

# LEAK 4 (small but real): imputing and scaling on ALL training rows before CV
imp_all = SimpleImputer(strategy="median").fit(train[clinical])
pre_mae = -cross_val_score(make_pipeline(StandardScaler(), Ridge(1.0)), imp_all.transform(train[clinical]),
                           yc, groups=g, cv=gkf, scoring="neg_mean_absolute_error").mean()
print("impute-before-CV MAE:", round(pre_mae, 4), " pipeline MAE:", round(mae_admission, 4))
```

### 5. Honest scorecard, champion by CV, test once
```python
# The honest scorecard: every candidate is a Pipeline, scored with GroupKFold on TRAIN only.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
cols_all = clinical + markers
cands = {
    "ridge, clinical": (lin(), clinical),
    "ridge, clinical + 300 markers": (lin(), cols_all),
    "select 10 + ridge, all": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               SelectKBest(f_regression, k=10), Ridge(1.0)), cols_all),
    "random forest, clinical": (rf(), clinical),
}
score = {k: mae(m, c) for k, (m, c) in cands.items()}
print(pd.Series(score).round(3))
champion = min(score, key=score.get)
m, c = cands[champion]
final = m.fit(train[c], train.los_days)
pred = final.predict(test[c])
print("champion:", champion)
print("test MAE", round(mean_absolute_error(test.los_days, pred), 3),
      "RMSE", round(mean_squared_error(test.los_days, pred) ** 0.5, 3),
      "R2", round(r2_score(test.los_days, pred), 3))
```

### 6. Push to GitHub
```bash
git add day27
git commit -m "Day 27: data leakage and pipelines inside CV on LOS"
git push
```

### Stretch: leakage audit in Snowflake SQL
```sql
-- leakage audit: flag any feature recorded AFTER the moment we predict (admission)
CREATE TABLE IF NOT EXISTS feature_catalog (
  feature_name STRING, source_table STRING, recorded_event STRING);  -- e.g. 'admission', 'discharge', 'billing'

SELECT f.feature_name, f.recorded_event,
       IFF(f.recorded_event IN ('admission', 'pre_admission'), 'OK', 'LEAK RISK') AS audit
FROM feature_catalog f
ORDER BY audit DESC, feature_name;

-- repeat patients: how many encounters per patient? (if many, use GroupKFold)
SELECT COUNT(*) AS encounters, COUNT(DISTINCT patient_id) AS patients,
       ROUND(COUNT(*) / COUNT(DISTINCT patient_id), 2) AS stays_per_patient
FROM encounters;
```

## Printables
- Cheat sheet: `day27_v1_cheat_sheet.png`
- Infographic: `day27_v2_infographic.png`
- Code worksheet: `day27_v3_worksheet.png`
- Practice quiz: `day27_v4_quiz.png`
- PDF: `day27_lesson.pdf`

## Podcast script
(Audio: `day27_podcast.mp3`)

Hey Todor, welcome to day twenty seven. Yesterday we learned to read learning curves and validation curves. Today is about something that quietly ruins more machine learning projects than a bad algorithm ever will. It is called data leakage, and the fix is a habit you will use in every project from now on: put every step inside a pipeline, and cross validate the whole thing.

So what is leakage? It is any time information from the rows you are supposed to be testing on sneaks into the part where the model learns. When that happens, your score looks better than the model really is. Then the model goes live, and the real performance is worse than you promised. In healthcare that is a credibility problem, and sometimes a patient safety problem.

Let me walk through four kinds, using our simulated hospital data of two thousand four hundred encounters from eight hundred patients, each with three stays, predicting length of stay in days.

The first kind is selection leakage. We added three hundred columns of pure random noise, pretend biomarkers. Then we shuffled the target, so there is truly nothing to learn. If you pick the twenty best looking columns using all the rows, and only then run cross validation, you get a positive R squared, around plus point zero eight, and across twenty shuffles it averaged about plus point one seven. That is a model finding signal in pure noise. Put the same selector inside a pipeline, so it is refit on each training fold, and the score drops below zero, which is the honest answer. A shuffled target is a great sanity test. If you score above zero on noise, go hunting for the leak.

The second kind is target leakage. Our data has a column called total charges. It is billed after discharge, so it is basically length of stay times a daily rate. With it, the error drops to about point four four days, which looks amazing. Without it, using only what we know at admission, the error is about one point four eight days. The question to ask of every column is simple: would I actually know this at the moment I make the prediction? Discharge codes, final charges, and anything recorded after the outcome all fail that test.

The third kind is group leakage. Hospitals see the same patient again and again. If one stay sits in the training fold and another stay from the same person sits in the validation fold, a flexible model like a random forest can partly memorize that person. With regular K fold, the forest scored about one point five one days. With group K fold, which keeps each patient's rows together, it scored about one point five eight. The fix is to pass the patient I D as groups, and to split the test set by patient too.

The fourth kind is preprocessing leakage, like filling missing albumin values or scaling with statistics from all rows before cross validation. In today's data that effect was tiny, less than a thousandth of a day. That is worth saying out loud. Some leaks are huge and some are small, but the habit matters, because you will not always know which one you have.

Once we cleaned all that up, every candidate was a pipeline scored with group K fold on the training patients only. The winner was a simple ridge model on the clinical features, with a cross validation error of about one point four eight days. Adding the three hundred noise markers made it worse, and the random forest did not beat it. Then we scored the held out patients once, and got an error of about one point four six days and an R squared of about point seven five. Remember, this data is simulated for teaching, so these numbers say nothing about any real hospital.

In job interviews for healthcare data science roles, leakage questions come up a lot, because they show whether you can be trusted with real patient data. If you can explain these four leaks and show a pipeline with group K fold in your GitHub repo, that is a strong, concrete answer.

Tomorrow is the last day of week four. We will pull the whole week together into a regression scorecard and a short model card, so you have a clean portfolio piece on length of stay. See you then.
