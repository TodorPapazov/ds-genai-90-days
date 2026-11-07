# Day 23: Ridge and Lasso on multicollinear length-of-stay features: shrink, zero, and read stable coefficients

## Concept
Week 4 Day 2. When clinical features are near-duplicates (acuity ≈ severity_score, comorbidities ≈ chronic_burden), unregularized OLS coefficients fight each other. Fresh simulation of 4,000 discharges (seed 23) with 12 features including those echoes plus frailty, lab noise and a pure distractor. Peak |corr| among echoes 0.998. Always StandardScaler first. Models: Dummy mean, OLS, Ridge, Lasso, ElasticNet. Pick alpha with RidgeCV/LassoCV on TRAIN by MAE (RidgeCV α≈12.66, LassoCV α≈0.0033). 5-fold CV MAE ≈ 1.247 days for OLS, RidgeCV and LassoCV vs baseline 2.681. Selection: lowest CV MAE, prefer regularized within 1 fold sd → champion Lasso (CV alpha). Test once: MAE 1.301 days, RMSE 1.633, R² 0.766; about 79% within 2 days. Ridge shares acuity/severity credit; Lasso paths can zero redundant echoes at larger α. Tomorrow: feature interactions and polynomial terms for LOS.

## Key numbers
- **Data:** 4,000 simulated discharges (seed 23), 12 features with clinical echoes, continuous los_days, 75/25 split
- **Train LOS:** mean 6.37 days, sd 3.28; peak |corr| among echoes 0.998
- **CV alphas:** RidgeCV alpha=12.66 | LassoCV alpha=0.003257
- **5-fold CV MAE:** OLS/RidgeCV/LassoCV 1.247, ElasticNet 1.249, baseline 2.681 days
- **Champion:** Lasso (CV alpha) by stability rule (prefer regularized within 1 fold sd)
- **Test set, once:** MAE 1.301 days, RMSE 1.633, R2 0.766; 79% within 2 days

## Hands-on
1. In your repo, create day23/, run conda activate ds90, and open day23/ridge_lasso_los.ipynb in VS Code.
2. Run snippet 1 to simulate ~4,000 discharges with clinical echo features (seed 23).
3. Confirm peak |corr| among acuity/severity_score and comorbidities/chronic_burden.
4. Run snippet 2 to define Dummy, OLS+scaler, Ridge, Lasso and ElasticNet.
5. Run snippet 3: 5-fold CV scorecard, then RidgeCV/LassoCV to pick alpha by MAE on TRAIN.
6. Record the chosen alphas; rebuild tuned Ridge and Lasso into the lineup.
7. In a markdown cell, write your selection rule before touching the test set.
8. Run snippet 4: compare OLS vs RidgeCV vs LassoCV coefficients; pick champion; score test once.
9. Note which features Lasso zeros (if any) and how Ridge redistributes acuity/severity credit.
10. Run snippet 5: plot Ridge and Lasso coefficient paths vs alpha; check residual health numbers.
11. Write day23/README.md: scorecard, coef comparison, paths, champion and why, limits.
12. Commit and push (snippet 6).
13. Stretch: Snowflake SQL for AVG/ABS prediction error on scored discharges (snippet 7).

## Code
### 1. Simulate LOS with multicollinear clinical echo features
```python
import numpy as np
import pandas as pd

# Fresh LOS simulation (seed 23) with CORRELATED clinical features.
# Near-duplicate columns make unregularized OLS coefficients fight each other.
rng = np.random.default_rng(23)
n = 4000

latent = rng.normal(0, 1, n)
age = (67 + 11 * rng.normal(0, 1, n) + 2.5 * latent).clip(18, 98).round()
acuity = np.clip(np.round(3 + 1.15 * latent + rng.normal(0, 0.25, n)), 1, 5).astype(int)
comorbidities = np.clip(
    np.round(1.2 + 0.05 * (age - 67) + 0.85 * latent + rng.integers(0, 2, n)),
    0, 10).astype(int)
icu_stay = (rng.random(n) < (0.05 + 0.16 * (acuity - 1) / 4 + 0.04 * comorbidities / 5)).astype(int)
procedure = (rng.random(n) < (0.28 + 0.08 * (acuity - 1) / 4)).astype(int)
emergency = (rng.random(n) < (0.48 + 0.10 * (acuity - 1) / 4)).astype(int)
home_dc = (rng.random(n) < (0.76 - 0.09 * (acuity - 1) / 4 - 0.05 * icu_stay)).astype(int)

# Near-duplicates of true drivers (tiny noise -> multicollinearity)
severity_score = (acuity + rng.normal(0, 0.08, n)).round(2)          # almost acuity
chronic_burden = (comorbidities + rng.normal(0, 0.12, n)).round(2)    # almost comorbidities
frailty_index = (0.03 * (age - 50) + 0.45 * comorbidities + 0.40 * acuity
                 + rng.normal(0, 0.20, n)).clip(0, 10).round(2)
lab_noise = (1.2 * acuity + rng.normal(0, 2.0, n)).round(2)           # weak acuity echo
random_noise = rng.normal(0, 1, n).round(3)                           # pure distractor

mu = (2.0
      + 0.038 * (age - 67)
      + 1.25 * (acuity - 1)
      + 2.45 * procedure
      + 0.55 * comorbidities
      + 3.3 * icu_stay
      + 0.80 * emergency
      - 1.10 * home_dc)
noise = rng.normal(0, 1.65, n)
los_days = np.clip(mu + noise, 1.0, None).round(1)

df = pd.DataFrame({
    "age": age, "acuity": acuity, "procedure": procedure, "comorbidities": comorbidities,
    "icu_stay": icu_stay, "emergency": emergency, "home_dc": home_dc,
    "severity_score": severity_score, "chronic_burden": chronic_burden,
    "frailty_index": frailty_index, "lab_noise": lab_noise, "random_noise": random_noise,
    "los_days": los_days,
})
df.to_csv("hospital_los.csv", index=False)

df = pd.read_csv("hospital_los.csv")
from sklearn.model_selection import train_test_split
features = ["age", "acuity", "procedure", "comorbidities", "icu_stay", "emergency",
            "home_dc", "severity_score", "chronic_burden", "frailty_index",
            "lab_noise", "random_noise"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.los_days, test_size=0.25, random_state=23)
sub = df[["acuity", "severity_score", "comorbidities", "chronic_burden",
          "frailty_index", "age"]].corr().abs().to_numpy().copy()
np.fill_diagonal(sub, 0)
print("train", X_train.shape, "test", X_test.shape,
      "mean LOS", round(y_train.mean(), 2), "sd", round(y_train.std(), 2))
print("peak |corr| among clinical echoes", round(float(sub.max()), 3))
print("acuity~severity_score corr",
      round(float(df[["acuity", "severity_score"]].corr().iloc[0, 1]), 3),
      "| comorbidities~chronic_burden",
      round(float(df[["comorbidities", "chronic_burden"]].corr().iloc[0, 1]), 3))
```

### 2. Regularization line-up: OLS, Ridge, Lasso, ElasticNet (all scaled)
```python
# Regularization line-up: mean baseline, OLS, Ridge, Lasso, ElasticNet (all scaled)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

models = {
    "Baseline (predict mean)": DummyRegressor(strategy="mean"),
    "OLS (unregularized)": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=10000)),
    "ElasticNet": make_pipeline(StandardScaler(),
                                ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)),
}
```

### 3. CV scorecard + RidgeCV/LassoCV to pick alpha by MAE
```python
# CV scorecard on TRAIN + RidgeCV / LassoCV to pick alpha by MAE
import time
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error

cv = KFold(5, shuffle=True, random_state=23)
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

# Alpha search on TRAIN only (inner CV). Prefer MAE for clinical days-off.
alphas = np.logspace(-3, 2, 40)
ridge_cv = make_pipeline(StandardScaler(),
    RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error", cv=cv))
lasso_cv = make_pipeline(StandardScaler(),
    LassoCV(alphas=alphas, cv=cv, max_iter=20000, random_state=23))
ridge_cv.fit(X_train, y_train)
lasso_cv.fit(X_train, y_train)
ridge_alpha = float(ridge_cv.named_steps["ridgecv"].alpha_)
lasso_alpha = float(lasso_cv.named_steps["lassocv"].alpha_)
print(f"RidgeCV alpha={ridge_alpha:.4g} | LassoCV alpha={lasso_alpha:.4g}")

# Rebuild tuned models into the lineup for the final pick
models["Ridge (CV alpha)"] = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
models["Lasso (CV alpha)"] = make_pipeline(
    StandardScaler(), Lasso(alpha=lasso_alpha, max_iter=20000))
for name in ["Ridge (CV alpha)", "Lasso (CV alpha)"]:
    t0 = time.time()
    r = cross_validate(models[name], X_train, y_train, cv=cv, scoring=scoring)
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
```

### 4. Coefficient comparison, stability champion, test once
```python
# Coefficient comparison: OLS fights; Ridge shrinks; Lasso can zero. Pick champion, test once.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

fitted = {}
coef_table = {}
for name in ["OLS (unregularized)", "Ridge (CV alpha)", "Lasso (CV alpha)"]:
    pipe = models[name].fit(X_train, y_train)
    fitted[name] = pipe
    step = list(pipe.named_steps.values())[-1]
    coef_table[name] = pd.Series(step.coef_, index=features)

coefs = pd.DataFrame(coef_table).round(3)
print("coefficients (days per 1-sd feature change):")
print(coefs)
lasso_zeroed = list(coefs.index[coefs["Lasso (CV alpha)"].abs() < 1e-6])
print("Lasso zeroed features:", lasso_zeroed if lasso_zeroed else "(none)")
ols_abs = coefs["OLS (unregularized)"].abs()
ridge_abs = coefs["Ridge (CV alpha)"].abs()
print(f"OLS |coef| sum {ols_abs.sum():.3f} | Ridge |coef| sum {ridge_abs.sum():.3f}")
print(f"acuity+severity OLS {coefs.loc['acuity','OLS (unregularized)']+coefs.loc['severity_score','OLS (unregularized)']:.3f}"
      f" | Ridge {coefs.loc['acuity','Ridge (CV alpha)']+coefs.loc['severity_score','Ridge (CV alpha)']:.3f}")

# Prefer regularized when CV MAE is within 1 fold sd of the best (stability over raw OLS)
candidates = scorecard.drop(index=["Baseline (predict mean)"], errors="ignore")
best = candidates.cv_mae.idxmin()
best_mae = candidates.loc[best, "cv_mae"]
best_sd = candidates.loc[best, "mae_sd"]
# Prefer Lasso (CV) then Ridge (CV) then others if within 1 sd of best
prefer_order = ["Lasso (CV alpha)", "Ridge (CV alpha)", "ElasticNet", "Ridge", "Lasso",
                "OLS (unregularized)"]
champion = best
for name in prefer_order:
    if name in candidates.index:
        if candidates.loc[name, "cv_mae"] <= best_mae + best_sd:
            champion = name
            break
print("best CV MAE:", best, "| champion by stability rule:", champion)

final = models[champion].fit(X_train, y_train)
y_pred = final.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
test_r2 = r2_score(y_test, y_pred)
print(f"test MAE {test_mae:.3f} days | RMSE {test_rmse:.3f} | R2 {test_r2:.3f}")
residuals = y_test.values - y_pred
print("residual mean", round(residuals.mean(), 3),
      "residual sd", round(residuals.std(), 3))
```

### 5. Coefficient paths vs alpha and residual health check
```python
# Coefficient paths vs alpha (Ridge and Lasso) + a quick residual health check
from sklearn.linear_model import Ridge, Lasso

path_alphas = np.logspace(-3, 2, 30)
scaler = StandardScaler().fit(X_train)
Xs = scaler.transform(X_train)
ridge_paths, lasso_paths = [], []
for a in path_alphas:
    ridge_paths.append(Ridge(alpha=a).fit(Xs, y_train).coef_)
    lasso_paths.append(Lasso(alpha=a, max_iter=20000).fit(Xs, y_train).coef_)
ridge_paths = np.array(ridge_paths)
lasso_paths = np.array(lasso_paths)
print("path alphas from", float(path_alphas.min()), "to", float(path_alphas.max()))
print("at largest alpha, Ridge nonzero", int((np.abs(ridge_paths[-1]) > 1e-8).sum()),
      "| Lasso nonzero", int((np.abs(lasso_paths[-1]) > 1e-8).sum()))

abs_err = np.abs(residuals)
print(f"median |error| {np.median(abs_err):.2f} days | "
      f"P90 |error| {np.quantile(abs_err, 0.90):.2f} days | "
      f"share within 2 days {(abs_err <= 2).mean():.0%}")
# When features move together, OLS coefficients fight; Ridge shrinks all;
# Lasso can zero redundant echoes so care managers see a shorter feature list.
```

### 6. Push to GitHub
```bash
git add day23
git commit -m "Day 23: Ridge and Lasso on multicollinear LOS features"
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
- Cheat sheet: `day23_v1_cheat_sheet.png`
- Infographic: `day23_v2_infographic.png`
- Code worksheet: `day23_v3_worksheet.png`
- Practice quiz: `day23_v4_quiz.png`
- PDF: `day23_lesson.pdf`

## Podcast script
(Audio: `day23_podcast.mp3`)

Hey Todor, welcome to day twenty three. Yesterday we predicted length of stay with M A E, R M S E, and R squared. Today we dig into a problem that shows up the moment clinical features start echoing each other: multicollinearity.

Imagine acuity and a severity score that are almost the same number, or comorbidities and a chronic burden index that move together. Unregularized O L S will fight over who gets the credit. One coefficient goes big, the other shrinks or even flips sign, and a care manager looking at the feature list will not trust either number. The model still predicts okay, but the story it tells is unstable.

We simulate four thousand fresh discharges with a fixed seed of twenty three. Twelve features this time. The true drivers are still age, acuity, procedure, comorbidities, I C U stay, emergency, and home discharge. But we also add near duplicates: severity score almost equals acuity, chronic burden almost equals comorbidities, plus a frailty index that mixes age and acuity, a weak lab noise echo, and a pure random distractor. Peak absolute correlation among those clinical echoes sits around point nine nine eight. That is the whole point of today.

The model line up is a mean baseline, O L S with a StandardScaler, Ridge, Lasso, and ElasticNet. Always scale first. Ridge and Lasso penalties assume features are on a comparable scale. If you skip the scaler, age in years will drown acuity on a one to five scale, and the penalty will punish the wrong thing.

We score every model with five fold cross validation on the training set, using M A E as the clinical days off metric. Then we let Ridge C V and Lasso C V pick alpha on train only, searching a log spaced grid from point zero zero one to one hundred. On this data Ridge C V lands around twelve point seven, and Lasso C V lands mild, around point zero zero three. Cross validated M A E for O L S, Ridge C V, and Lasso C V all sit near one point two five days, clearly beating the mean baseline at about two point seven.

The selection rule is written before we open the test set: lowest cross validated M A E, and prefer a regularized model if it is within one fold standard deviation of the best. That stability preference is what picks Lasso C V as the champion even when O L S ties on raw M A E. Then we open the test set once. Test M A E is about one point three days, R M S E about one point six, R squared about point seven seven. Roughly four out of five patients land within two days.

Look at the coefficients. O L S puts most of the acuity plus severity credit on acuity and almost none on severity. Ridge shares that credit more evenly. Lasso at the mild C V alpha barely zeros anything here, but the Lasso path chart shows the story: as alpha grows, redundant echoes drop to exactly zero, and at the largest alpha every coefficient is gone. That is the sparse feature list care managers can actually read.

Your homework is to build the notebook, run Ridge C V and Lasso C V, compare the coefficient table, draw the paths, and push day twenty three with a short README. Tomorrow we add feature interactions and polynomial terms for length of stay, so the model can learn things like acuity times I C U stay instead of only straight add ups. See you then, Todor.
