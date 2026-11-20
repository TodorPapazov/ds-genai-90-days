# Day 25: Skewed targets: log-transforming length of stay, TransformedTargetRegressor, and back-transforming predictions (smearing) to days

## Concept
Week 4 Day 4, a catch-up lesson: on November 8 the schedule jumped from Day 24 to Day 26, so this day never went out. We stay on simulated hospital length of stay, but today the target is right-skewed. Fresh simulation of 4,000 discharges (seed 25) with 8 features, where los_days is lognormal (many short stays, a long right tail). Train mean 9.40 days, median 7.50, skew 2.52. Days 22-24 used even noise on the day scale; raw Ridge here still gets the average close, but residuals fan out, and a linear model on days has no floor. The habit is to fit on log(LOS) and translate back to days. TransformedTargetRegressor (func=np.log, inverse_func=np.exp) does that, and predict() already returns days, so cross-validation stays on the original scale. Care managers hear days, not log days. The catch: a plain exp() back-transform estimates the conditional median, not the mean. 5-fold CV MAE: TTR 3.841, Duan smearing 3.970, raw Ridge 4.057, baseline 4.893. CV mean bias: TTR -1.288 days, Duan -0.046, raw Ridge -0.001. Duan smearing multiplies the naive exp() prediction by the mean of exp(train log residuals) (factor 1.153 here; the parametric cousin exp(mu + sigma2/2) is 1.155). Selection, written before test: among models with |CV mean bias| under 0.3 days, lowest CV RMSE, prefer Duan if it is within 1 fold sd (fewer assumptions) → champion Log + Duan smearing. TTR wins the MAE rule (typical patient error). Test once (n=1,000): champion MAE 3.752 days, RMSE 5.365, R² 0.285, mean bias +0.18, 38% within 2 days. TTR test MAE is better (3.580) and 44% are within 2 days, but it undercounts 1,052 bed-days (mean pred 8.04 vs actual 9.09). On 165 long test stays (over 14 days), TTR MAE 9.69 vs Duan 8.43. After back-transform, a residual fan can remain on the day scale; judge the log residuals. All data is simulated for teaching. Day 26, already on the site, continues with learning curves and validation curves.

## Key numbers
- **Data:** 4,000 simulated discharges (seed 25), 8 features, lognormal los_days, 75/25 split
- **Train LOS:** mean 9.40 days, median 7.50, sd 7.08, skew 2.52; 17% of stays exceed 14 days
- **Smear factor:** Duan 1.153; parametric exp(sigma^2/2) 1.155 (sigma 0.537)
- **5-fold CV MAE:** TTR 3.841, Duan 3.970, raw Ridge 4.057, baseline 4.893
- **CV mean bias:** TTR -1.288 days, Duan -0.046, raw Ridge -0.001
- **Champion:** Log + Duan smearing by the planning rule (|bias| < 0.3 d, then lowest CV RMSE; prefer Duan)
- **Test set, once:** Duan MAE 3.752, RMSE 5.365, R2 0.285, mean bias +0.18; TTR MAE 3.580 but 1,052 bed-days short

## Hands-on
1. In your repo, create day25/, run conda activate ds90, and open day25/skewed_los.ipynb in VS Code.
2. Run snippet 1 to simulate ~4,000 right-skewed discharges (seed 25) and save hospital_los.csv.
3. Print train mean, median, skew and percentiles, and confirm the feature list does not include los_days.
4. Plot raw versus log histograms and mark the mean and the median.
5. Run snippet 2 to define Dummy, Ridge on raw days, TransformedTargetRegressor (log/exp), Duan smearing, and the parametric cousin.
6. Run snippet 3: 5-fold CV on TRAIN for MAE, RMSE, R² and mean bias, all in days.
7. In a markdown cell, write your planning rule before touching the test set (|CV mean bias| under 0.3 days, then lowest CV RMSE; prefer Duan within 1 fold sd).
8. Run snippet 4: pick the planning champion and the MAE winner, record the smear factor, and confirm TTR matches a manual exp().
9. Score the test set once: MAE, RMSE, R², mean bias, and total bed-days.
10. Run snippet 5: draw raw-day residuals and log-scale residuals; compute MAE on stays longer than 14 days and the bed-day gap.
11. Write day25/README.md: skew, scorecard, back-transform comparison, champion and why, limits (simulated data).
12. Commit and push (snippet 6).
13. Stretch: Snowflake SQL for mean vs median and MAE by stay band (snippet 7).

## Code
### 1. Simulate a right-skewed, lognormal length of stay
```python
import numpy as np
import pandas as pd

# Fresh LOS simulation (seed 25). Unlike Days 22-24, the target is right-skewed:
# we generate log(los_days) as a linear function of clinical features plus noise,
# then exponentiate, so stays are lognormal (many short, a long right tail).
rng = np.random.default_rng(25)
n = 4000

latent = rng.normal(0, 1, n)
age = (67 + 11 * rng.normal(0, 1, n) + 2.5 * latent).clip(18, 98).round()
acuity = np.clip(np.round(3 + 1.15 * latent + rng.normal(0, 0.25, n)), 1, 5).astype(int)
comorbidities = np.clip(
    np.round(1.2 + 0.05 * (age - 67) + 0.85 * latent + rng.integers(0, 2, n)),
    0, 10).astype(int)
icu_stay = (rng.random(n) < (0.08 + 0.18 * (acuity - 1) / 4 + 0.04 * comorbidities / 5)).astype(int)
procedure = (rng.random(n) < (0.30 + 0.08 * (acuity - 1) / 4)).astype(int)
emergency = (rng.random(n) < (0.48 + 0.10 * (acuity - 1) / 4)).astype(int)
home_dc = (rng.random(n) < (0.76 - 0.09 * (acuity - 1) / 4 - 0.05 * icu_stay)).astype(int)
random_noise = rng.normal(0, 1, n).round(3)

log_mu = (1.45
          + 0.008 * (age - 67)
          + 0.16 * (acuity - 1)
          + 0.28 * procedure
          + 0.07 * comorbidities
          + 0.40 * icu_stay
          + 0.12 * emergency
          - 0.18 * home_dc)
los_days = np.clip(np.exp(log_mu + rng.normal(0, 0.55, n)), 1.0, None).round(1)

df = pd.DataFrame({
    "age": age, "acuity": acuity, "procedure": procedure, "comorbidities": comorbidities,
    "icu_stay": icu_stay, "emergency": emergency, "home_dc": home_dc,
    "random_noise": random_noise, "los_days": los_days,
})
df.to_csv("hospital_los.csv", index=False)

df = pd.read_csv("hospital_los.csv")
from sklearn.model_selection import train_test_split
features = ["age", "acuity", "procedure", "comorbidities", "icu_stay",
            "emergency", "home_dc", "random_noise"]
assert "los_days" not in features
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.los_days, test_size=0.25, random_state=25)

print("train", X_train.shape, "test", X_test.shape)
print("train mean", round(y_train.mean(), 2),
      "median", round(y_train.median(), 2),
      "sd", round(y_train.std(), 2),
      "skew", round(y_train.skew(), 3))
print("all-rows mean", round(df.los_days.mean(), 2),
      "median", round(df.los_days.median(), 2),
      "skew", round(df.los_days.skew(), 3),
      "min", df.los_days.min(), "max", df.los_days.max())
print("percentiles 50/90/95/99:",
      np.round(np.percentile(y_train, [50, 90, 95, 99]), 1))
print("share of train stays > 14 days", round((y_train > 14).mean(), 3),
      "> 21 days", round((y_train > 21).mean(), 3))
print("features do not include los_days:", features)
```

### 2. Raw Ridge vs TransformedTargetRegressor vs Duan smearing
```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inner model: scale, then Ridge, with alpha chosen on TRAIN folds by MAE.
# The same inner pipeline is used on raw days and on log days.
cv = KFold(5, shuffle=True, random_state=25)
alphas = np.logspace(-3, 3, 40)

def make_ridge():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error", cv=cv),
    )

# Naive back-transform: TTR.predict() is exp(inner.predict(X)).
# For a log-linear model that estimates the conditional MEDIAN of LOS, not the mean.
ttr = TransformedTargetRegressor(
    regressor=make_ridge(), func=np.log, inverse_func=np.exp, check_inverse=True)

# Duan smearing (1983): multiply the naive exp() prediction by the mean of
# exp(train log-residuals). That estimates the conditional MEAN of LOS.
# The smear factor is learned on the training rows of each fit, so it is CV-safe.
class SmearLogRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kind="smear"):
        self.kind = kind

    def fit(self, X, y):
        self.reg_ = make_ridge()
        self.reg_.fit(X, np.log(y))
        resid = np.log(y) - self.reg_.predict(X)
        self.smear_ = float(np.mean(np.exp(resid)))          # Duan factor
        self.sigma2_ = float(np.var(resid, ddof=1))          # for the parametric cousin
        return self

    def predict(self, X):
        median_hat = np.exp(self.reg_.predict(X))
        if self.kind == "median":
            return median_hat
        if self.kind == "parametric":
            return median_hat * np.exp(self.sigma2_ / 2.0)   # exp(mu + sigma^2 / 2)
        return median_hat * self.smear_

models = {
    "Baseline (predict mean)": DummyRegressor(strategy="mean"),
    "Ridge on raw days": make_ridge(),
    "TTR log/exp (naive)": ttr,
    "Log + Duan smearing": SmearLogRegressor("smear"),
    "Log + parametric smear": SmearLogRegressor("parametric"),
}
print("models:", list(models))
print("note: stays are at least 1 day, so np.log / np.exp is safe;")
print("      if a stay can be 0, use np.log1p and np.expm1 instead.")
```

### 3. CV scorecard in days, including mean bias
```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

def signed_bias(y_true, y_pred):
    """Mean predicted minus mean actual, in days. Near 0 means the average is calibrated."""
    return float(np.mean(y_pred) - np.mean(y_true))

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "bias": make_scorer(signed_bias),
}

rows = []
for name, model in models.items():
    cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows.append({
        "model": name,
        "cv_mae": -cvres["test_mae"].mean(),
        "mae_sd": cvres["test_mae"].std(),
        "cv_rmse": -cvres["test_rmse"].mean(),
        "rmse_sd": cvres["test_rmse"].std(),
        "cv_r2": cvres["test_r2"].mean(),
        "cv_bias": cvres["test_bias"].mean(),
        "bias_sd": cvres["test_bias"].std(),
    })

scorecard = pd.DataFrame(rows).set_index("model")
print()
print(scorecard[["cv_mae","mae_sd","cv_rmse","rmse_sd","cv_r2","cv_bias","bias_sd"]].to_string(float_format=lambda x: f"{x:7.3f}"))
print()
print("Planning rule (write this BEFORE opening the test set):")
print("  among models with |CV mean bias| under 0.3 days, pick the lowest CV RMSE;")
print("  if Duan smearing is within 1 fold sd of that best RMSE, prefer Duan")
print("  (it does not assume normal log-residuals).")
print("MAE rule (typical patient error): lowest CV MAE, for comparison.")

planning = scorecard[scorecard.cv_bias.abs() < 0.3]
if len(planning) == 0:
    champion = scorecard.cv_rmse.idxmin()
else:
    best = planning.cv_rmse.idxmin()
    duan = "Log + Duan smearing"
    if duan in planning.index and planning.loc[duan, "cv_rmse"] <= (
            planning.loc[best, "cv_rmse"] + planning.loc[duan, "rmse_sd"]):
        champion = duan
    else:
        champion = best
mae_winner = scorecard.cv_mae.idxmin()
print("planning champion:", champion,
      "CV RMSE", round(scorecard.loc[champion, "cv_rmse"], 3),
      "CV bias", round(scorecard.loc[champion, "cv_bias"], 3))
print("MAE winner:", mae_winner,
      "CV MAE", round(scorecard.loc[mae_winner, "cv_mae"], 3),
      "CV bias", round(scorecard.loc[mae_winner, "cv_bias"], 3))
```

### 4. Planning champion, smear factor, test once
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model

# Confirm TTR's naive exp() matches a manual exp() of the log-model predictions.
smear_m = fitted["Log + Duan smearing"]
ttr_m = fitted["TTR log/exp (naive)"]
manual_median = np.exp(smear_m.reg_.predict(X_test))
ttr_pred = ttr_m.predict(X_test)
print("smear factor (Duan)", round(smear_m.smear_, 4))
print("parametric factor exp(sigma^2/2)", round(np.exp(smear_m.sigma2_ / 2), 4),
      "sigma", round(np.sqrt(smear_m.sigma2_), 3))
print("max |TTR pred - manual exp(log pred)|",
      float(np.max(np.abs(ttr_pred - manual_median))))

def metrics(name, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    n_neg = int((pred < 0).sum())
    mean_bias = float(pred.mean() - y_test.mean())
    med_bias = float(np.median(pred) - y_test.median())
    long = y_test > 14
    mae_long = mean_absolute_error(y_test[long], pred[long])
    within2 = float((np.abs(y_test - pred) <= 2).mean())
    bed_days_gap = float(pred.sum() - y_test.sum())
    return dict(model=name, mae=mae, rmse=rmse, r2=r2, n_neg=n_neg,
                mean_pred=float(pred.mean()), mean_actual=float(y_test.mean()),
                mean_bias=mean_bias, med_pred=float(np.median(pred)),
                med_actual=float(y_test.median()), med_bias=med_bias,
                mae_long=mae_long, n_long=int(long.sum()),
                within2=within2, bed_days_gap=bed_days_gap, n=len(y_test))

test_rows = []
preds = {}
for name, model in fitted.items():
    pred = np.asarray(model.predict(X_test), dtype=float)
    preds[name] = pred
    test_rows.append(metrics(name, pred))

test_tab = pd.DataFrame(test_rows).set_index("model")
print(test_tab.round(3))
print()
print("test actual mean", round(y_test.mean(), 2),
      "median", round(y_test.median(), 2),
      "total bed-days", round(y_test.sum(), 1), "n", len(y_test))
print("champion (planning):", champion)
print("test MAE", round(test_tab.loc[champion, "mae"], 3),
      "RMSE", round(test_tab.loc[champion, "rmse"], 3),
      "R2", round(test_tab.loc[champion, "r2"], 3),
      "mean bias", round(test_tab.loc[champion, "mean_bias"], 3),
      "within 2 days", round(test_tab.loc[champion, "within2"], 3))
```

### 5. Residual fan, long-stay MAE, and the bed-day gap
```python
# Residual health and the long-stay tail, all on the original day scale
# (plus log-scale residuals for the log model, where a fan should disappear).

raw_name = "Ridge on raw days"
ttr_name = "TTR log/exp (naive)"
champ_name = champion

y_pred_raw = preds[raw_name]
y_pred_ttr = preds[ttr_name]
y_pred_champ = preds[champ_name]
resid_raw = y_test.to_numpy() - y_pred_raw
resid_ttr = y_test.to_numpy() - y_pred_ttr
resid_champ = y_test.to_numpy() - y_pred_champ
log_resid_test = np.log(y_test.to_numpy()) - smear_m.reg_.predict(X_test)

def fan_corr(resid, pred):
    return float(np.corrcoef(np.abs(resid), pred)[0, 1])

print("corr(|residual|, predicted)  raw", round(fan_corr(resid_raw, y_pred_raw), 3),
      "  TTR (day scale)", round(fan_corr(resid_ttr, y_pred_ttr), 3),
      "  smearing (day scale)", round(fan_corr(resid_champ, y_pred_champ), 3))
print("log-scale residual mean", round(log_resid_test.mean(), 3),
      "sd", round(log_resid_test.std(), 3),
      "(a healthy log model is centered near 0 with no fan on this scale)")
print("raw min prediction", round(y_pred_raw.min(), 2),
      "negative predictions", int((y_pred_raw < 0).sum()),
      "(none today, but a raw-day linear model has no floor)")
print("long-stay test rows (actual > 14 days):")
for name in [raw_name, ttr_name, champ_name]:
    print(" ", name, "MAE", round(test_tab.loc[name, "mae_long"], 3),
          "n", int(test_tab.loc[name, "n_long"]))
print("total bed-days on test:",
      "actual", round(y_test.sum(), 1),
      "| TTR", round(preds[ttr_name].sum(), 1),
      "gap", round(test_tab.loc[ttr_name, "bed_days_gap"], 1),
      "| smearing", round(preds[champ_name].sum(), 1),
      "gap", round(test_tab.loc[champ_name, "bed_days_gap"], 1))
print("within 2 days: TTR", round(test_tab.loc[ttr_name, "within2"], 3),
      "smearing", round(test_tab.loc[champ_name, "within2"], 3),
      "raw", round(test_tab.loc[raw_name, "within2"], 3))
# Trap: R^2 on the log scale looks cheerier and is not what you report.
log_r2 = r2_score(np.log(y_test), smear_m.reg_.predict(X_test))
print("trap: log-scale R2", round(log_r2, 3),
      "vs day-scale R2 for smearing", round(test_tab.loc[champ_name, "r2"], 3),
      "(always report days)")
```

### 6. Push to GitHub
```bash
git add day25
git commit -m "Day 25: log-transform, TransformedTargetRegressor, and smearing on skewed LOS"
git push
```

### Stretch: mean vs median and MAE by stay band in Snowflake SQL
```sql
-- after scoring a held-out batch, compare mean vs median and MAE by stay band
-- (always on the original day scale)
CREATE TABLE IF NOT EXISTS los_predictions (
  discharge_id STRING, actual_los FLOAT, predicted_los FLOAT,
  model STRING, scored_on DATE);

SELECT model,
       ROUND(AVG(actual_los), 2) AS mean_actual,
       ROUND(MEDIAN(actual_los), 2) AS median_actual,
       ROUND(AVG(predicted_los), 2) AS mean_pred,
       ROUND(AVG(predicted_los) - AVG(actual_los), 2) AS mean_bias_days,
       ROUND(AVG(ABS(actual_los - predicted_los)), 3) AS mae_days,
       COUNT(*) AS n
FROM los_predictions
WHERE scored_on = CURRENT_DATE
GROUP BY model
ORDER BY mae_days;

SELECT model,
       CASE WHEN actual_los <= 7 THEN '1-7 days'
            WHEN actual_los <= 14 THEN '8-14 days'
            ELSE '15+ days' END AS stay_band,
       ROUND(AVG(ABS(actual_los - predicted_los)), 3) AS mae_days,
       COUNT(*) AS n
FROM los_predictions
WHERE scored_on = CURRENT_DATE
GROUP BY model, stay_band
ORDER BY model, stay_band;
```

## Printables
- Cheat sheet: `day25_v1_cheat_sheet.png`
- Infographic: `day25_v2_infographic.png`
- Code worksheet: `day25_v3_worksheet.png`
- Practice quiz: `day25_v4_quiz.png`
- PDF: `day25_lesson.pdf`

## Podcast script
(Audio: `day25_podcast.mp3`)

Hey Todor, welcome to day twenty five. This lesson is a catch-up. On November eighth the schedule jumped from day twenty four, which was about feature interactions, straight to day twenty six, which is about learning curves. The missing day is this one, and today we fill that gap. We are still in week four, still predicting hospital length of stay on simulated discharges, and we take on a problem the earlier days left in the background: the target is often right skewed.

Length of stay is a classic example. Most people go home in a few days, and a smaller group stays much longer. That long tail pulls the mean above the median. On days twenty two through twenty four we added even noise on the day scale, so the residual plots looked tidy. Today we simulate four thousand fresh discharges with a fixed seed of twenty five, and we generate length of stay as log normal, a common shape for stays. On the training set the mean is about nine point four days, the median is about seven point five, and the skew is about two point five. A histogram of days has a pile on the left and a tail on the right. A histogram of the log of days is much more symmetric. That picture is the whole reason for today.

If you fit a Ridge model on the raw days, it can get the average about right, because a linear model is aimed at the mean. But the residuals often fan out: short stays are tight, long stays are all over the place. And a raw day model has no floor, so in other data it can even predict a negative stay. The habit is to model the log of length of stay, then translate the predictions back into days. Scikit-learn has a wrapper for this called Transformed Target Regressor. You hand it a pipeline, plus a function, usually log, and an inverse, usually exp. Its predict method already returns days, not logs, so your cross validation scores stay in days. That matters. Care managers hear days, not log days, so M A E and R M S E always get reported on the original scale.

Here is the catch, and it is the number I want you to remember. A plain exponential back transform estimates something like the typical stay, the conditional median, not the average stay. On our held out patients, that naive back transform had the best M A E, about three point six days, and the most patients within two days, about forty five percent. But its average prediction was about eight point zero days, while the real average was about nine point one. That is a one day shortfall. If you were planning beds, you would have undercounted total days by about one thousand and fifty on a thousand discharges. The fix is called smearing, after Duan. You take the training residuals on the log scale, exponentiate them, and average those values. That average, about one point one five on this data, is a multiplier you apply after the exponential. With that correction, the predicted mean moves back near nine point three, and cross validated R M S E improves.

So which model is the champion? It depends on the job, and you write that down before you open the test set. If you care about typical error for one patient, M A E, the naive log exp wrapper can win. If you care about expected total days or staffing, you want the smearing correction, even if M A E is a little worse. Today our planning rule picks log plus Duan smearing: among models whose average is within point three days of the truth on cross validation, take the lowest R M S E, and prefer Duan when it is close, because it does not assume normal residuals. On the test set the champion lands around three point seven five M A E, five point three seven R M S E, and a mean bias of about plus point two days. One more reminder: after you transform back to days, a residual fan can still appear, because long stays really are more variable. Check the residuals on the log scale to see whether the model itself is healthy.

Your homework is to build the notebook, plot raw versus log histograms, compare Ridge on days with the transformed wrapper and with smearing, and score everything in days. Push day twenty five with a short README. Day twenty six is already on the site, and it continues this week with learning curves and validation curves, so you can tell whether a model is under fitting or just memorizing the training set. See you then, Todor.
