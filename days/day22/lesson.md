# Day 22: Predicting length of stay: MAE, RMSE, R squared and residual plots on hospital discharges

## Concept
Week 4 Day 1. We switch from classification (yes/no readmission) to regression: predict a continuous hospital length of stay in days. Fresh simulation of 4,000 discharges (seed 22) where true LOS depends on age, acuity, procedure, comorbidities, ICU stay, emergency admit and home discharge. Features never include los_days. Models: DummyRegressor (mean baseline), LinearRegression + StandardScaler, Ridge, and a shallow DecisionTreeRegressor. cross_validate on train scores MAE (average days off), RMSE (penalizes big misses) and R² (share of variance vs the mean). Linear and Ridge tie at CV MAE 1.256 days and R² 0.772, beating the tree (1.619) and baseline (2.682). Selection rule: lowest CV MAE, prefer linear if within 1 fold sd. Test once: MAE 1.262 days, RMSE 1.587, R² 0.760; about 79% within 2 days. Residual plot should be random scatter with no fan or curve. Never report R² alone without MAE/RMSE in days. Tomorrow: Ridge and Lasso regularization deeper dive.

## Key numbers
- **Data:** 4,000 simulated discharges (seed 22), 7 clinical features, continuous los_days, 75/25 split
- **Train LOS:** mean 7.41 days, sd 3.31
- **5-fold CV MAE:** linear 1.256, Ridge 1.256, tree 1.619, baseline 2.682 days
- **5-fold CV R2:** linear 0.772, Ridge 0.772, tree 0.632, baseline ~0
- **Champion:** Linear regression (tied with Ridge; simplest within 1 fold sd)
- **Test set, once:** MAE 1.262 days, RMSE 1.587, R2 0.760; 79% within 2 days

## Hands-on
1. In your repo, create day22/, run conda activate ds90, and open day22/los_regression.ipynb in VS Code.
2. Run snippet 1 to simulate ~4,000 discharges with a continuous los_days target (seed 22).
3. Confirm the feature list does not include los_days; print mean and sd of the target.
4. Run snippet 2 to define DummyRegressor, Linear+scaler, Ridge and a shallow tree.
5. Run snippet 3: cross_validate every model with MAE, RMSE and R² on the same folds.
6. Plot CV MAE with +/- 1 sd error bars; note who beats the mean baseline.
7. In a markdown cell, write your selection rule before touching the test set.
8. Run snippet 4: pick the champion, score the test set once, and compute residuals.
9. Draw a residual plot (predicted vs residual) and a predicted-vs-actual scatter.
10. Run snippet 5: read coefficients in days and summarize residual health (median |error|, share within 2 days).
11. Write day22/README.md: scorecard, two charts, champion and why, limits.
12. Commit and push (snippet 6).
13. Stretch: Snowflake SQL for AVG/ABS prediction error on scored discharges (snippet 7).

## Code
### 1. Simulate hospital discharges with a continuous LOS target
```python
import numpy as np
import pandas as pd

# Fresh LOS simulation (seed 22). Target is continuous los_days; features never include it.
rng = np.random.default_rng(22)
n = 4000
age = rng.normal(67, 14, n).clip(18, 98).round()
acuity = rng.integers(1, 6, n)                              # 1=low severity ... 5=critical
procedure = (rng.random(n) < 0.38).astype(int)              # major procedure during stay
comorbidities = rng.poisson(2.2, n).clip(0, 10)             # count of chronic conditions
icu_stay = (rng.random(n) < (0.08 + 0.06 * (acuity - 1))).astype(int)
emergency = (rng.random(n) < 0.55).astype(int)              # emergency vs elective admit
home_dc = (rng.random(n) < (0.72 - 0.08 * (acuity - 1))).astype(int)  # home vs SNF/other

# True expected LOS (days): clinical drivers, then positive noise so LOS stays realistic
mu = (2.4
      + 0.035 * (age - 67)
      + 1.15 * (acuity - 1)
      + 2.4 * procedure
      + 0.55 * comorbidities
      + 3.2 * icu_stay
      + 0.9 * emergency
      - 1.1 * home_dc)
noise = rng.normal(0, 1.6, n)
los_days = np.clip(mu + noise, 1.0, None).round(1)

df = pd.DataFrame({
    "age": age, "acuity": acuity, "procedure": procedure, "comorbidities": comorbidities,
    "icu_stay": icu_stay, "emergency": emergency, "home_dc": home_dc, "los_days": los_days,
})
df.to_csv("hospital_los.csv", index=False)

df = pd.read_csv("hospital_los.csv")
from sklearn.model_selection import train_test_split
features = ["age", "acuity", "procedure", "comorbidities", "icu_stay", "emergency", "home_dc"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.los_days, test_size=0.25, random_state=22)
print("train", X_train.shape, "test", X_test.shape,
      "mean LOS", round(y_train.mean(), 2), "sd", round(y_train.std(), 2))
```

### 2. Regression model line-up (baseline, linear, Ridge, tree)
```python
# Regression line-up: mean baseline, linear, Ridge, shallow tree for comparison
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

models = {
    "Baseline (predict mean)": DummyRegressor(strategy="mean"),
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Decision tree (depth 4)": DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=40, random_state=22),
}
```

### 3. One scorecard: MAE, RMSE and R2 on the same folds
```python
# One scorecard on TRAIN only: MAE (days off), RMSE (big misses), R2 (vs mean baseline)
import time
from sklearn.model_selection import KFold, cross_validate

cv = KFold(5, shuffle=True, random_state=22)
scoring = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error",
           "r2": "r2"}
rows, fold_mae = [], {}
for name, model in models.items():
    t0 = time.time()
    r = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    fold_mae[name] = -r["test_mae"]
    rows.append({
        "model": name,
        "cv_mae": (-r["test_mae"]).mean(),
        "mae_sd": (-r["test_mae"]).std(),
        "cv_rmse": (-r["test_rmse"]).mean(),
        "cv_r2": r["test_r2"].mean(),
        "r2_sd": r["test_r2"].std(),
        "seconds": time.time() - t0,
    })
scorecard = pd.DataFrame(rows).set_index("model").round(3)
print(scorecard.sort_values("cv_mae"))
# Lower MAE/RMSE is better; higher R2 is better. Never quote R2 alone without days.
```

### 4. Pick by a rule, test once, compute residuals
```python
# Pick by lowest CV MAE (prefer linear if within 1 fold sd of the best). Then test once.
best = scorecard.cv_mae.idxmin()
lin_ok = (scorecard.loc["Linear regression", "cv_mae"]
          <= scorecard.loc[best, "cv_mae"] + scorecard.loc[best, "mae_sd"])
champion = "Linear regression" if lin_ok else best
print("best CV MAE:", best, "| champion by the rule:", champion)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
final = models[champion].fit(X_train, y_train)
y_pred = final.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
test_r2 = r2_score(y_test, y_pred)
print(f"test MAE {test_mae:.3f} days | RMSE {test_rmse:.3f} | R2 {test_r2:.3f}")

# Residuals for the two teaching charts: predicted vs residual, predicted vs actual
residuals = y_test.values - y_pred
print("residual mean", round(residuals.mean(), 3),
      "residual sd", round(residuals.std(), 3))
# Healthy residuals look like random scatter around 0, no fan and no curve.
```

### 5. Coefficients in days and a residual health check
```python
# Plain-English read of the champion: coefficients in days, plus a quick residual check
from sklearn.pipeline import Pipeline
pipe = final if isinstance(final, Pipeline) else None
if pipe is not None and hasattr(pipe.named_steps.get("linearregression",
        pipe.named_steps.get("ridge", None)), "coef_"):
    step = pipe.named_steps.get("linearregression") or pipe.named_steps.get("ridge")
    coefs = pd.Series(step.coef_, index=features).sort_values(key=abs, ascending=False)
    print("coefficients (days per 1-sd feature change):")
    print(coefs.round(3))
else:
    print("tree champion: leaf means vary by acuity/ICU splits; inspect with plot_tree")

# Residual health check in numbers (charts live in the printables)
abs_err = np.abs(residuals)
print(f"median |error| {np.median(abs_err):.2f} days | "
      f"P90 |error| {np.quantile(abs_err, 0.90):.2f} days | "
      f"share within 2 days {(abs_err <= 2).mean():.0%}")
```

### 6. Push to GitHub
```bash
git add day22
git commit -m "Day 22: predicting length of stay with MAE, RMSE, R squared and residual plots"
git push
```

### Stretch: MAE and RMSE in Snowflake SQL
```sql
-- score a held-out batch in Snowflake and summarize days-off errors
CREATE TABLE IF NOT EXISTS los_predictions (
  discharge_id STRING, actual_los FLOAT, predicted_los FLOAT, model STRING, scored_on DATE);

-- after loading predictions from your notebook:
SELECT model,
       ROUND(AVG(ABS(actual_los - predicted_los)), 3) AS mae_days,
       ROUND(SQRT(AVG(POWER(actual_los - predicted_los, 2))), 3) AS rmse_days,
       COUNT(*) AS n
FROM los_predictions
WHERE scored_on = CURRENT_DATE
GROUP BY model
ORDER BY mae_days;
```

## Printables
- Cheat sheet: `day22_v1_cheat_sheet.png`
- Infographic: `day22_v2_infographic.png`
- Code worksheet: `day22_v3_worksheet.png`
- Practice quiz: `day22_v4_quiz.png`
- PDF: `day22_lesson.pdf`

## Podcast script
(Audio: `day22_podcast.mp3`)

Hey Todor, welcome to day twenty two. That is the start of week four, and we are switching gears. For the last week we predicted a yes or no: will this patient come back within thirty days. Today we predict a number: how many days will this patient stay in the hospital. That is regression, and hospitals care about it because length of stay drives bed planning, staffing, and cost.

We simulate four thousand fresh hospital discharges with a fixed seed of twenty two. The target is length of stay in days. The features are clinical drivers only: age, acuity from one to five, whether there was a major procedure, how many comorbidities, an I C U stay flag, emergency versus elective admission, and whether the patient went home. Notice we do not reuse length of stay as a feature the way we did for readmission. That would be cheating. Length of stay is what we are trying to predict.

The model line up is simple on purpose. A baseline DummyRegressor that always predicts the training mean, about seven point four days. A linear regression wrapped in a StandardScaler pipeline. Ridge, which is linear with a little shrinkage. And a shallow depth four decision tree for comparison. We score every model with five fold cross validation on the training set only, using three metrics that belong together.

M A E, mean absolute error, is the one you can say in plain English: on average, how many days off is the prediction. Ours lands around one point three days for the linear models. R M S E, root mean squared error, is similar but it penalizes big misses more, so a few patients you are way off on drag it up. Here it sits around one point six days. And R squared tells you what share of the variance in length of stay your model explains compared to just guessing the mean. The linear models hit about point seven seven. The baseline is near zero by design, which is a useful sanity check.

Never report R squared alone. A model can look great on R squared and still be two days off on average, and a care manager hears days, not fractions of variance. Always pair R squared with M A E and R M S E in days.

On this data, linear regression and Ridge are basically tied, both about one point two six M A E, and they clearly beat the tree and the mean baseline. That makes sense: we built length of stay as a smooth add up of clinical factors, so a linear model fits the shape. We pick the champion with a rule written before the test set: lowest cross validated M A E, and prefer the simpler linear model if it is within one fold standard deviation of the best. Linear wins.

Then we open the test set once. Test M A E is about one point three days, R M S E about one point six, R squared about point seven six. Roughly four out of five patients are within two days. The residual plot, prediction on the x axis and residual on the y, should look like random scatter around zero with no fan shape and no curve. A fan would mean the model gets worse as stays get longer. A curve would mean we missed a nonlinear pattern. Ours looks healthy, which is what you want before you trust the numbers.

Your homework is to build the notebook, run the scorecard, draw the residual and predicted versus actual charts, and push day twenty two with a short README. Tomorrow we go deeper on regularization, Ridge and Lasso, and why shrinking coefficients can help when features are correlated or when you have more noise than signal. See you then, Todor.
