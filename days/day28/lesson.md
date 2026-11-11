# Day 28: The week 4 regression scorecard: seven models in one fair race, a rule written first, test once, and a model card

## Concept
Week 4 wrap-up. Instead of judging each length-of-stay model in its own notebook, we race seven of them under the same rules: the same 8 admission-time features, the same split by patient, the same 5 GroupKFold folds, and every model inside a Pipeline. cross_validate scores each on MAE, RMSE, R² and fit time. On 2,400 simulated encounters from 800 patients (seed 28): baseline (median) 3.065 days, linear 1.432, ridge 1.432, lasso 1.432, poly2 + ridge 1.413, random forest 1.490, XGBoost 1.427. The fold-to-fold sd is about 0.05 days, bigger than most gaps, so a rule written before testing (best CV MAE, then the simplest model within 1 sd) picks plain linear regression. Test set, opened once: MAE 1.534 days, RMSE 1.941, R² 0.723 (baseline 3.083). Test is worse than CV because one split of 200 patients is noisy, so report both and quote CV. An 80% band from out-of-fold residual quantiles (-2.34 / +2.24 days) covered 77.8% of test stays, and error was highest at acuity 4 (1.64 days). The model is saved with joblib and documented in model_card.json. All data is simulated for teaching, so the numbers say nothing about real hospitals. Tomorrow: week 5 opens with feature engineering inside a ColumnTransformer.

## Key numbers
- **Data:** 2,400 simulated encounters, 800 patients x 3 stays (seed 28), 8 admission-time features, split by patient 1,800/600
- **CV MAE (days):** baseline 3.065, linear 1.432, ridge 1.432, lasso 1.432, poly2+ridge 1.413, forest 1.490, XGBoost 1.427
- **Selection rule:** best 1.413 + 1 sd = 1.468; 5 models qualify; simplest = linear
- **Test set, once:** linear: MAE 1.534 days, RMSE 1.941, R2 0.723 (baseline MAE 3.083)
- **80% band:** pred -2.34 / +2.24 days, test coverage 77.8%
- **Weakest subgroup:** acuity 4, test MAE 1.64 days

## Hands-on
1. In your repo, create day28/, run conda activate ds90, and open day28/los_scorecard.ipynb in VS Code.
2. Run snippet 1 to simulate 2,400 encounters from 800 patients (seed 28), save los_week4.csv, split by patient and set up GroupKFold(5).
3. Run snippet 2: seven Pipelines ordered simplest to most complex, each scored with cross_validate on MAE, RMSE, R² and fit time.
4. Before going further, write your selection rule in a markdown cell.
5. Run snippet 3: apply the rule (best CV MAE, then simplest within 1 sd) and plot per-fold MAE for each model.
6. Run snippet 4: build an 80% band from out-of-fold residuals on train, refit the champion, score the test set once, check coverage and error by acuity.
7. Run snippet 5: save the model with joblib and write model_card.json.
8. Write day28/README.md: leaderboard, the rule, test numbers, the model card, the negative-prediction quirk, and limits (simulated data).
9. Commit and push (snippet 6).
10. Stretch: store the CV leaderboard in a Snowflake table so every future run is comparable (snippet 7).

## Code
### 1. Simulate, split by patient, set up GroupKFold (seed 28)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Fresh simulation (seed 28): 2,400 encounters from 800 patients (3 stays each).
# Only admission-time features this time (yesterday's leak audit already removed total_charges).
rng = np.random.default_rng(28)
n_pat, per = 800, 3
pid = np.repeat(np.arange(n_pat), per)
n = n_pat * per
frailty = np.repeat(rng.normal(0, 1, n_pat), per)          # hidden, never a feature
age = np.repeat(rng.normal(66, 12, n_pat).clip(18, 98).round(), per)
bmi = np.repeat(rng.normal(29, 5, n_pat).clip(16, 55).round(1), per)
comorbidities = np.clip(np.round(np.repeat(rng.normal(2.2, 1.3, n_pat), per) + 0.6 * frailty), 0, 10).astype(int)
acuity = np.clip(np.round(3 + 0.9 * rng.normal(0, 1, n) + 0.3 * frailty), 1, 5).astype(int)
procedure = (rng.random(n) < 0.30 + 0.06 * (acuity - 1)).astype(int)
icu_stay = (rng.random(n) < 0.06 + 0.05 * (acuity - 1)).astype(int)
emergency = (rng.random(n) < 0.50).astype(int)
albumin = (4.0 - 0.25 * frailty + rng.normal(0, 0.15, n)).round(2)
albumin[rng.random(n) < 0.12] = np.nan

mu = (2.0 + 0.02 * (age - 66) + 0.8 * (acuity - 1) + 1.4 * procedure + 0.35 * comorbidities
      + 2.2 * icu_stay + 0.5 * emergency + 1.2 * icu_stay * procedure   # one interaction
      + 0.04 * (bmi - 29) + 3.0 * frailty - 0.6 * (np.nan_to_num(albumin, nan=4.0) - 4.0))
los_days = np.clip(mu + rng.normal(0, 1.0, n), 1.0, None).round(1)

df = pd.DataFrame({"patient_id": pid, "age": age, "bmi": bmi, "acuity": acuity, "procedure": procedure,
                   "comorbidities": comorbidities, "icu_stay": icu_stay, "emergency": emergency,
                   "albumin": albumin, "los_days": los_days})
df.to_csv("los_week4.csv", index=False)
features = ["age", "bmi", "acuity", "procedure", "comorbidities", "icu_stay", "emergency", "albumin"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=28)
tr_idx, te_idx = next(gss.split(df, groups=df.patient_id))     # patients never cross the split
train, test = df.iloc[tr_idx], df.iloc[te_idx]
Xtr, ytr, gtr = train[features], train.los_days, train.patient_id
Xte, yte = test[features], test.los_days
gkf = GroupKFold(5)
print(df.shape, "mean LOS", round(df.los_days.mean(), 2), "train", len(train), "test", len(test))
```

### 2. Seven Pipelines, one fair race with cross_validate
```python
# One fair race: same features, same patient split, same 5 GroupKFold folds, every model a Pipeline.
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor

imp = lambda: SimpleImputer(strategy="median")
cands = {   # ordered simplest -> most complex
    "baseline (median)": make_pipeline(imp(), DummyRegressor(strategy="median")),
    "linear": make_pipeline(imp(), StandardScaler(), LinearRegression()),
    "ridge": make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "lasso": make_pipeline(imp(), StandardScaler(), LassoCV(cv=5, random_state=28)),
    "poly2 + ridge": make_pipeline(imp(), StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                   StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "random forest": make_pipeline(imp(), RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=28)),
    "xgboost": make_pipeline(imp(), XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                 subsample=0.8, random_state=28)),
}
scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
rows = []
for name, model in cands.items():
    cv = cross_validate(model, Xtr, ytr, groups=gtr, cv=gkf, scoring=scoring)
    rows.append({"model": name, "cv_mae": -cv["test_mae"].mean(), "mae_sd": cv["test_mae"].std(),
                 "cv_rmse": -cv["test_rmse"].mean(), "cv_r2": cv["test_r2"].mean(),
                 "fit_sec": cv["fit_time"].mean()})
board = pd.DataFrame(rows).set_index("model")
print(board.round(3))
```

### 3. The selection rule: simplest within 1 sd
```python
# Selection rule, written BEFORE looking at the test set:
# take the best CV MAE, then prefer the simplest model whose MAE is within 1 sd of it.
best = board.cv_mae.idxmin()
limit = board.loc[best, "cv_mae"] + board.loc[best, "mae_sd"]
eligible = [m for m in cands if board.loc[m, "cv_mae"] <= limit]   # dict order = simplest first
champion = eligible[0]
print("best CV MAE:", best, round(board.loc[best, "cv_mae"], 3), "+1 sd limit", round(limit, 3))
print("within 1 sd:", eligible)
print("champion (simplest within 1 sd):", champion)
```

### 4. Test once, 80% band, error by acuity
```python
# Open the test set ONCE. Add an 80% prediction band from out-of-fold residuals on TRAIN.
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
oof = cross_val_predict(cands[champion], Xtr, ytr, groups=gtr, cv=gkf)
lo_q, hi_q = np.quantile(ytr - oof, [0.10, 0.90])                # residual quantiles, train only
final = cands[champion].fit(Xtr, ytr)
pred = final.predict(Xte)
base = cands["baseline (median)"].fit(Xtr, ytr).predict(Xte)
t_mae, t_rmse, t_r2 = mean_absolute_error(yte, pred), mean_squared_error(yte, pred) ** 0.5, r2_score(yte, pred)
coverage = np.mean((yte >= pred + lo_q) & (yte <= pred + hi_q))
print(f"test MAE {t_mae:.3f}  RMSE {t_rmse:.3f}  R2 {t_r2:.3f}  baseline MAE {mean_absolute_error(yte, base):.3f}")
print(f"80% band: pred {lo_q:+.2f} to {hi_q:+.2f} days, test coverage {coverage:.1%}")
by_acuity = (test.assign(err=(yte - pred).abs()).groupby("acuity").err.agg(["count", "mean"]).round(2))
print(by_acuity)                                                  # where is the model weakest?
```

### 5. Save the model and write a model card
```python
# Save the model and a model card: what it is, how it was judged, and where it must not be used.
import joblib, json, sklearn
joblib.dump(final, "los_champion.joblib")
card = {
    "model": champion, "target": "length of stay (days), predicted at admission",
    "data": f"{len(df):,} SIMULATED encounters, {df.patient_id.nunique()} patients, seed 28; split by patient",
    "features": features, "selection_rule": "lowest 5-fold GroupKFold MAE; simplest model within 1 sd wins",
    "cv_leaderboard_mae": board.cv_mae.round(3).to_dict(),
    "test": {"mae": round(t_mae, 3), "rmse": round(t_rmse, 3), "r2": round(t_r2, 3),
             "band_80": [round(lo_q, 2), round(hi_q, 2)], "band_coverage": round(float(coverage), 3)},
    "error_by_acuity_mae": by_acuity["mean"].to_dict(),
    "intended_use": "teaching; bed-planning style estimates, never individual clinical decisions",
    "limits": ["simulated data", "no time split", "hidden frailty not captured", "not validated on real patients"],
    "sklearn_version": sklearn.__version__,
}
json.dump(card, open("model_card.json", "w"), indent=2, default=str)
print(json.dumps(card["test"], indent=2))
```

### 6. Push to GitHub
```bash
git add day28
git commit -m "Day 28: week 4 regression scorecard, test once, model card"
git push
```

### Stretch: a model scorecard table in Snowflake SQL
```sql
-- keep every model run comparable: one row per model per run
CREATE TABLE IF NOT EXISTS model_scorecard (
  run_ts TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(), project STRING, model STRING,
  cv_mae FLOAT, mae_sd FLOAT, cv_rmse FLOAT, cv_r2 FLOAT, is_champion BOOLEAN);

-- load board.reset_index() with write_pandas, then compare runs:
SELECT model, cv_mae, mae_sd, cv_r2,
       cv_mae - MIN(cv_mae) OVER () AS gap_to_best,
       IFF(cv_mae <= MIN(cv_mae + mae_sd) OVER (), 'within 1 sd', '') AS eligible
FROM model_scorecard
WHERE project = 'los_week4'
QUALIFY run_ts = MAX(run_ts) OVER ()
ORDER BY cv_mae;
```

## Printables
- Cheat sheet: `day28_v1_cheat_sheet.png`
- Infographic: `day28_v2_infographic.png`
- Code worksheet: `day28_v3_worksheet.png`
- Practice quiz: `day28_v4_quiz.png`
- PDF: `day28_lesson.pdf`

## Podcast script
(Audio: `day28_podcast.mp3`)

Hey Todor, welcome to day twenty eight. That is four weeks done, and today we close out week four, the regression week, with something hiring managers love to see: a fair model scorecard, an honest test, and a model card.

Quick recap of the week. We predicted hospital length of stay in days. We learned MAE, RMSE and R squared, we tamed correlated features with ridge and lasso, we tried interaction terms, we read learning curves, and yesterday we hunted down data leakage. Today we put it all together.

Here is the idea. Instead of judging each model in its own notebook, on its own split, we race all of them under the same rules. Same eight admission-time features. Same split by patient, so nobody's stays end up on both sides. Same five group k fold folds. And every model lives inside a pipeline, so the imputer and the scaler are refit inside each fold. That way, any difference you see comes from the models, not from one of them getting easier folds.

The line-up had seven entries. A baseline that just guesses the median stay for everyone. Plain linear regression. Ridge. Lasso. Polynomial terms with ridge. A random forest. And XGBoost. On our simulated data of two thousand four hundred encounters, the baseline was off by about three point oh six days on average. Every real model cut that error roughly in half. The best cross validated MAE came from polynomial terms with ridge, at one point four one days. Linear, ridge and lasso were all at one point four three. XGBoost one point four three. The forest trailed at one point four nine.

Now the important part. The fold to fold standard deviation was about five hundredths of a day. So the gap between first place and plain linear regression, about two hundredths, is smaller than the noise. That is a lean, not a win. So before we ever opened the test set, we wrote down a rule: find the best score, then pick the simplest model that is within one standard deviation of it. Five models qualified, and the simplest was plain linear regression. Fewer moving parts, coefficients a clinician can read, and nothing to tune.

Then, and only then, we opened the test set, once. Mean absolute error one point five three days, R squared about point seven two. That is a bit worse than cross validation said. Did something break? Probably not. The test set is just two hundred patients, one roll of the dice. The honest move is to report both numbers, quote the cross validated one as your estimate, and never go back and retune.

We also added two things that make a model useful in real life. First, a range, not just a single number. We took the out of fold errors on the training data and used their tenth and ninetieth percentiles to build an eighty percent band around each prediction. On the test set, that band caught about seventy eight percent of the actual stays, close to the target. Second, we checked error by subgroup. The model was sharpest for the lowest acuity patients and weakest for acuity level four. An average can hide a weak spot like that.

Finally, we saved the model with joblib and wrote a model card. Think of it as the label on the bottle: what data it was trained on, how it was judged, the numbers, what it is meant for, and what it must never be used for. Here, that is teaching and bed planning style estimates, never decisions about an individual patient. And of course, all of this data is simulated, so the numbers say nothing about real hospitals.

One more thing to notice on the chart: linear regression happily predicts negative stays for a few low risk patients. That is a known quirk, and a good discussion point for your README.

That wraps week four, and with it, classic machine learning for both classification and regression. Tomorrow we start week five: feature engineering for real. We will turn messy dates, diagnosis codes and categories into features a model can actually use, and build it all inside a column transformer pipeline. See you then.
