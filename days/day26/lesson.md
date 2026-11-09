# Day 26: Learning and validation curves: diagnosing under and overfitting on length of stay

## Concept
Week 4 Day 5 of classic ML. Training error always falls as a model grows, so it cannot tell you when to stop. A validation curve varies one knob (polynomial degree, Ridge alpha) with data fixed and plots train vs 5-fold CV error. A learning curve fixes the model and grows the training size. Small gap with high error = bias (underfit); low train error with a large gap = variance (overfit). Fresh simulation of 3,000 discharges (seed 26), 8 features, 75/25 split. Degree 1 (8 terms): train 1.303, CV 1.310, gap 0.008. Degree 2 (44 terms): CV 1.247, gap 0.029. Degree 3 (164 terms): CV 1.288, gap 0.100. Degree 4 (494 terms): train 1.107, CV 1.402, gap 0.295. At 1,800 rows the degree 4 gap is still 0.295, so it is variance limited. Ridge alpha sweep on degree 4: best alpha about 316, CV MAE 1.262, still behind plain degree 2. Champion by CV only: degree 2. Test once: MAE 1.254 days, RMSE 1.570, R² 0.774. All data is simulated for teaching, so the numbers say nothing about real hospitals. Tomorrow: data leakage and scikit-learn pipelines inside cross-validation.

## Key numbers
- **Data:** 3,000 simulated discharges (seed 26), 8 features, continuous los_days, 75/25 split
- **Validation curve:** CV MAE deg1 1.310, deg2 1.247, deg3 1.288, deg4 1.402
- **Gap (CV - train):** deg1 0.008, deg2 0.029, deg3 0.100, deg4 0.295
- **Terms:** 8, 44, 164, 494 for degrees 1 to 4
- **Alpha sweep (deg 4):** best alpha about 316, CV MAE 1.262 (vs 1.402 at alpha 0.001)
- **Test set, once:** degree 2: MAE 1.254 days, RMSE 1.570, R2 0.774

## Hands-on
1. In your repo, create day26/, run conda activate ds90, and open day26/curves_los.ipynb in VS Code.
2. Run snippet 1 to simulate ~3,000 discharges (seed 26) and save hospital_los.csv.
3. Run snippet 2: split 75/25, build a 5-fold KFold on TRAIN only, and define make(degree, alpha) (poly, scale, Ridge).
4. Run validation_curve over degrees 1 to 4. Print train MAE, CV MAE, gap and term count.
5. In a markdown cell, write your diagnosis rule before touching the test set (for example gap above 0.15 days = variance).
6. Run snippet 3: learning_curve for degrees 1, 2 and 4. Plot train and CV against training rows.
7. Label each model: bias, good fit or variance. Does the gap still close at the largest size?
8. Run snippet 4: validation curve over Ridge alpha for degree 4, then a four-candidate CV scorecard.
9. Pick the champion by CV only. Run snippet 5 to score the test set once.
10. Write day26/README.md: three curve plots, diagnosis, champion and why, limits (simulated data).
11. Commit and push (snippet 6).
12. Stretch: log each curve run to a Snowflake table and query the gap (snippet 7).

## Code
### 1. Simulate length of stay (seed 26)
```python
import numpy as np
import pandas as pd

# Fresh LOS simulation (seed 26). Effects are NOT only additive:
# ICU after a procedure adds extra days beyond ICU + procedure alone,
# high acuity stretches an ICU stay, and age bends (a quadratic).
rng = np.random.default_rng(26)
n = 3000

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

mu = (2.2
      + 0.018 * (age - 67)
      + 0.95 * (acuity - 1)
      + 1.60 * procedure
      + 0.40 * comorbidities
      + 1.80 * icu_stay
      + 0.65 * emergency
      - 0.95 * home_dc
      + 2.80 * icu_stay * procedure
      + 0.45 * (acuity - 1) * icu_stay
      + 0.0016 * (age - 55) ** 2)
noise = rng.normal(0, 1.55, n)
los_days = np.clip(mu + noise, 1.0, None).round(1)

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
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.los_days, test_size=0.25, random_state=26)

# Interaction leftover: (ICU1,Proc1) - (ICU1,Proc0) - (ICU0,Proc1) + (ICU0,Proc0)
cell = df.groupby(["icu_stay", "procedure"])["los_days"].agg(["mean", "count"])
means = df.groupby(["icu_stay", "procedure"])["los_days"].mean()
leftover = (means.loc[(1, 1)] - means.loc[(1, 0)]
            - means.loc[(0, 1)] + means.loc[(0, 0)])
print("train", X_train.shape, "test", X_test.shape,
      "mean LOS", round(y_train.mean(), 2), "sd", round(y_train.std(), 2))
print("ICU x procedure cell means:")
print(cell.round(2))
print("interaction leftover (days)", round(float(leftover), 3))
print("share ICU", round(df.icu_stay.mean(), 3),
      "share procedure", round(df.procedure.mean(), 3),
      "share both", round(((df.icu_stay == 1) & (df.procedure == 1)).mean(), 3))
```

### 2. Split, CV object, make(), validation curve on degree
```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("hospital_los.csv")
features = ["age", "acuity", "procedure", "comorbidities", "icu_stay",
            "emergency", "home_dc", "random_noise"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df.los_days, test_size=0.25, random_state=26)
cv = KFold(5, shuffle=True, random_state=26)

def make(degree, alpha=1e-3):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))

# VALIDATION CURVE: one knob (polynomial degree), train vs CV error
degrees = [1, 2, 3, 4]
tr, va = validation_curve(make(1), X_train, y_train, param_name="polynomialfeatures__degree",
                          param_range=degrees, cv=cv, scoring="neg_mean_absolute_error")
val_tab = pd.DataFrame({"train_mae": -tr.mean(1), "cv_mae": -va.mean(1),
                        "cv_sd": va.std(1)}, index=degrees)
val_tab["gap"] = val_tab.cv_mae - val_tab.train_mae
val_tab["n_terms"] = [make(d).fit(X_train, y_train).named_steps["polynomialfeatures"].n_output_features_ for d in degrees]
print(val_tab.round(3))
```

### 3. Learning curves for degree 1, 2 and 4
```python
from sklearn.model_selection import learning_curve

# LEARNING CURVE: fixed model, growing training size
sizes = np.array([100, 200, 400, 800, 1000, 1800])
lc = {}
for d in [1, 2, 4]:
    n, tr, va = learning_curve(make(d), X_train, y_train, train_sizes=sizes, cv=cv,
                               scoring="neg_mean_absolute_error", shuffle=True, random_state=26)
    lc[d] = pd.DataFrame({"n": n, "train_mae": -tr.mean(1), "cv_mae": -va.mean(1)})
    lc[d]["gap"] = lc[d].cv_mae - lc[d].train_mae
    print(f"degree {d}"); print(lc[d].round(3).to_string(index=False))
```

### 4. Alpha validation curve and CV scorecard
```python
# Diagnose with a rule written BEFORE test: gap > 0.15 days = variance;
# both errors high and close = bias. Then tune alpha for the flexible model.
alphas = np.logspace(-2, 4, 13)
tr, va = validation_curve(make(4), X_train, y_train, param_name="ridge__alpha",
                          param_range=alphas, cv=cv, scoring="neg_mean_absolute_error")
alpha_tab = pd.DataFrame({"train_mae": -tr.mean(1), "cv_mae": -va.mean(1)}, index=alphas)
alpha_tab["gap"] = alpha_tab.cv_mae - alpha_tab.train_mae
best_alpha = alpha_tab.cv_mae.idxmin()
print(alpha_tab.round(3))
print("best alpha for degree 4:", round(best_alpha, 3))

candidates = {"degree 1": make(1), "degree 2": make(2),
              "degree 4, alpha 0.001": make(4, 1e-3),
              f"degree 4, alpha {best_alpha:.3g}": make(4, best_alpha)}
from sklearn.model_selection import cross_val_score
score = {k: -cross_val_score(m, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
         for k, m in candidates.items()}
print(pd.Series(score).round(3))
```

### 5. Champion by CV, test once, does more data help
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
champion = min(score, key=score.get)
final = candidates[champion].fit(X_train, y_train)
pred = final.predict(X_test)
print("champion:", champion)
print("test MAE", round(mean_absolute_error(y_test, pred), 3),
      "RMSE", round(mean_squared_error(y_test, pred) ** 0.5, 3),
      "R2", round(r2_score(y_test, pred), 3))

# Would more data help? Compare the last two learning-curve points for degree 4
d4 = lc[4]
print("degree 4 CV MAE: n=%d -> %.3f, n=%d -> %.3f; gap now %.3f"
      % (d4.n.iloc[-2], d4.cv_mae.iloc[-2], d4.n.iloc[-1], d4.cv_mae.iloc[-1], d4.gap.iloc[-1]))
```

### 6. Push to GitHub
```bash
git add day26
git commit -m "Day 26: learning and validation curves on LOS"
git push
```

### Stretch: log and query curve runs in Snowflake SQL
```sql
-- log curve results per run, then find the degree with the lowest CV error
CREATE TABLE IF NOT EXISTS curve_runs (
  run_date DATE, model_name STRING, poly_degree INT, ridge_alpha FLOAT,
  train_size INT, train_mae FLOAT, cv_mae FLOAT);

SELECT poly_degree, ridge_alpha, train_size,
       ROUND(cv_mae - train_mae, 3) AS gap_days,
       ROUND(cv_mae, 3) AS cv_mae
FROM curve_runs
WHERE run_date = CURRENT_DATE
QUALIFY cv_mae = MIN(cv_mae) OVER (PARTITION BY train_size)
ORDER BY train_size;
```

## Printables
- Cheat sheet: `day26_v1_cheat_sheet.png`
- Infographic: `day26_v2_infographic.png`
- Code worksheet: `day26_v3_worksheet.png`
- Practice quiz: `day26_v4_quiz.png`
- PDF: `day26_lesson.pdf`

## Podcast script
(Audio: `day26_podcast.mp3`)

Hey Todor, welcome to day twenty six. This is week four of classic machine learning, and today we learn to diagnose a model before we try to fix it. The tools are two simple pictures: the validation curve and the learning curve.

Here is the situation. We are predicting hospital length of stay, in days, from eight features like age, acuity, procedure, and I C U stay. Last time we added interactions and squares to capture effects that are more than a plain sum. That raises a fair question. How many extra terms is too many? When we expand the features, the model can always fit the training rows a little better. So training error alone can never answer that question. We need an honest, held-out number, and we need to see how it moves.

The validation curve changes one knob and keeps everything else fixed. Today the knob is the polynomial degree. In scikit-learn, you call validation curve with a pipeline, the parameter name, the values to try, and five fold cross validation on the training set only. For each degree you get a training error and a cross validation error. On our simulated data of three thousand discharges, degree one has eight terms and a mean absolute error of about one point three one days, with train and cross validation almost identical. That is a model that is too simple. Degree two has forty four terms and cross validation error drops to about one point two five. Degree four has four hundred ninety four terms, training error keeps falling, but cross validation error climbs to one point four. The gap between the two is the signal. A small gap with a high error points to bias. A large gap points to variance, which is overfitting.

The learning curve changes a different thing. It keeps the model fixed and grows the amount of training data. If the two curves meet and sit flat, more rows will not help, and you should improve the features instead. If there is still a visible gap that keeps closing as you add rows, then more data, or stronger regularization, can help. For degree four, the gap was huge with a hundred rows and was still about a third of a day at eighteen hundred rows. That model is limited by variance.

Then we bring back Ridge from day twenty three. Keeping degree four, we sweep the alpha penalty and watch the cross validation curve. The best alpha was around three hundred, and cross validation error improved from one point four to about one point two six. That is much better, but it still did not beat plain degree two. So the champion was degree two, picked by cross validation alone. We then scored the test set once and got a mean absolute error of about one point two five days.

A few honest notes. This is simulated data, built to teach the idea, so these numbers say nothing about real hospitals. And a small gap is not the goal. Degree one had the smallest gap and still lost, because its error was higher.

For your hands-on session, which takes sixty to ninety minutes, you will simulate the discharges, draw both curves, write your diagnosis rule before touching the test set, tune alpha, and push the results to your repo.

Tomorrow, on day twenty seven, we look at a quieter danger: data leakage, and how scikit-learn pipelines keep your scaling and feature steps honest inside cross validation. See you then.
