# Day 24: Feature interactions and polynomial terms: when ICU plus a procedure is more than the sum, on length of stay

## Concept
Week 4 Day 3. Length of stay is not always a sum of main effects: icu_stay * procedure can add extra recovery days, and age can bend (age2). Fresh simulation of 4,000 discharges (seed 24) with 8 features and those planted terms plus a distractor. The 2x2 table of mean LOS by ICU and procedure has an interaction leftover of 2.81 days (both-together 12.61 vs no/no 4.82, no/proc 6.51, ICU/no 8.11). PolynomialFeatures(degree=2) expands 8 columns to 36 interaction-only terms or 44 full degree-2 terms. Expand, then StandardScaler, then RidgeCV (alpha on TRAIN by MAE). 5-fold CV MAE: baseline 2.472, Additive Ridge 1.313, Interactions + Ridge 1.236, Poly2 + Ridge 1.229, Poly2 + OLS 1.229. Selection: lowest CV MAE, prefer the simpler expansion within 1 fold sd → champion Interactions + Ridge (alpha≈0.001). Test once: MAE 1.249 days, RMSE 1.555, R² 0.761; 80% within 2 days. Overall additive test MAE 1.335 vs 1.249 looks like a small win; on the 71 ICU-plus-procedure test patients, MAE drops from 1.833 to 1.350. Additive predicted cell mean for both-together 11.18 vs actual 12.41; champion 12.74. Tomorrow: learning curves and validation curves.

## Key numbers
- **Data:** 4,000 simulated discharges (seed 24), 8 features, continuous los_days, 75/25 split
- **Train LOS:** mean 6.12 days, sd 3.16; interaction leftover 2.81 days
- **Expansion:** 8 raw features → 36 interaction-only terms → 44 full degree-2 terms
- **5-fold CV MAE:** Poly2 Ridge 1.229, Interactions 1.236, Additive Ridge 1.313, baseline 2.472
- **Champion:** Interactions + Ridge by simplicity rule (prefer simpler within 1 fold sd)
- **Test set, once:** MAE 1.249 days, RMSE 1.555, R2 0.761; 80% within 2 days; ICU+proc MAE 1.350 vs additive 1.833

## Hands-on
1. In your repo, create day24/, run conda activate ds90, and open day24/interactions_los.ipynb in VS Code.
2. Run snippet 1 to simulate ~4,000 discharges with planted interactions and an age curve (seed 24).
3. Print the ICU × procedure cell means and the interaction leftover.
4. Run snippet 2 to define Dummy, additive Ridge, interactions + Ridge, poly degree 2 + Ridge, and poly degree 2 + OLS.
5. Confirm term counts: 8 raw, 36 interaction-only, 44 full degree-2.
6. Run snippet 3: 5-fold CV scorecard on TRAIN, and record each RidgeCV alpha.
7. In a markdown cell, write your simplicity rule before touching the test set.
8. Run snippet 4: pick the champion, score test once, and read recovered product terms.
9. Note whether procedure icu_stay and acuity icu_stay rank among the larger terms.
10. Run snippet 5: compare test cell means (actual vs additive vs champion) and the ICU-plus-procedure MAE.
11. Write day24/README.md: leftover, scorecard, cell-mean chart, champion and why, limits.
12. Commit and push (snippet 6).
13. Stretch: Snowflake SQL for MAE grouped by ICU and procedure (snippet 7).

## Code
### 1. Simulate LOS with planted interactions and a quadratic age term
```python
import numpy as np
import pandas as pd

# Fresh LOS simulation (seed 24). Effects are NOT only additive:
# ICU after a procedure adds extra days beyond ICU + procedure alone,
# high acuity stretches an ICU stay, and age bends (a quadratic).
rng = np.random.default_rng(24)
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
    df[features], df.los_days, test_size=0.25, random_state=24)

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

### 2. Additive vs interaction-only vs full degree-2 models
```python
# Model line-up: additive vs interaction-only vs full degree-2 polynomials.
# Expand first, then scale: age^2 and icu_stay*procedure live on very different raw scales.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold

cv = KFold(5, shuffle=True, random_state=24)
alphas = np.logspace(-3, 3, 40)

def poly_ridge(interaction_only):
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False,
                           interaction_only=interaction_only),
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error", cv=cv),
    )

models = {
    "Baseline (predict mean)": DummyRegressor(strategy="mean"),
    "Additive Ridge": make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error", cv=cv),
    ),
    "Interactions + Ridge": poly_ridge(True),
    "Poly degree 2 + Ridge": poly_ridge(False),
    "Poly degree 2 + OLS": make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    ),
}

pf_int = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
pf_full = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
n_int = pf_int.fit_transform(X_train).shape[1]
n_full = pf_full.fit_transform(X_train).shape[1]
print("raw features", len(features),
      "| interaction-only terms", n_int,
      "| full degree-2 terms", n_full)
print("models:", list(models))
```

### 3. CV scorecard and RidgeCV alphas on TRAIN
```python
# 5-fold CV scorecard on TRAIN. Alphas are chosen inside RidgeCV, still on train only.
import time
from sklearn.model_selection import cross_validate

scoring = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error",
           "r2": "r2"}
rows, fold_mae, chosen_alpha = [], {}, {}
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

# Fit each RidgeCV pipeline on TRAIN to read the chosen alpha (not a test peek)
for name in ["Additive Ridge", "Interactions + Ridge", "Poly degree 2 + Ridge"]:
    models[name].fit(X_train, y_train)
    step = models[name].named_steps["ridgecv"]
    chosen_alpha[name] = float(step.alpha_)
    print(f"{name} alpha={chosen_alpha[name]:.4g}")
```

### 4. Simplicity champion, test once, read recovered terms
```python
# Write the selection rule BEFORE opening test, then score once and read the terms.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

candidates = scorecard.drop(index=["Baseline (predict mean)"], errors="ignore")
best = candidates.cv_mae.idxmin()
best_mae = candidates.loc[best, "cv_mae"]
best_sd = candidates.loc[best, "mae_sd"]
# Prefer simpler expansions when CV MAE is within 1 fold sd of the best
prefer_order = ["Interactions + Ridge", "Poly degree 2 + Ridge",
                "Additive Ridge", "Poly degree 2 + OLS"]
champion = best
for name in prefer_order:
    if name in candidates.index:
        if candidates.loc[name, "cv_mae"] <= best_mae + best_sd:
            champion = name
            break
print("best CV MAE:", best, "| champion by simplicity rule:", champion)

final = models[champion]
# RidgeCV models were already fit on train in snippet 3; OLS/others may need a fit
final.fit(X_train, y_train)
y_pred = final.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
test_r2 = r2_score(y_test, y_pred)
residuals = y_test.values - y_pred
print(f"test MAE {test_mae:.3f} days | RMSE {test_rmse:.3f} | R2 {test_r2:.3f}")
print("residual mean", round(residuals.mean(), 3),
      "residual sd", round(residuals.std(), 3))

def term_table(pipe, label):
    steps = pipe.named_steps
    if "polynomialfeatures" in steps:
        names = steps["polynomialfeatures"].get_feature_names_out(features)
    else:
        names = np.array(features)
    if "ridgecv" in steps:
        coefs_ = steps["ridgecv"].coef_
    else:
        coefs_ = steps["linearregression"].coef_
    s = pd.Series(coefs_, index=names, name=label)
    return s

champ_terms = term_table(final, champion)
special = [t for t in champ_terms.index
           if ("icu_stay" in t and "procedure" in t)
           or ("acuity" in t and "icu_stay" in t)
           or t in ("age^2", "age")]
print("key recovered terms (scaled coefs):")
print(champ_terms.reindex(special).dropna().round(3))
poly_like = champ_terms[[c for c in champ_terms.index if (" " in c) or ("^" in c)]]
print("top |interaction/square| terms:")
print(poly_like.abs().sort_values(ascending=False).head(8).round(3))
```

### 5. Cell means and the ICU-plus-procedure subgroup MAE
```python
# Group-mean check: can an additive model reproduce the ICU x procedure cells?
add_pipe = models["Additive Ridge"]
add_pipe.fit(X_train, y_train)
test_view = X_test.copy()
test_view["actual"] = y_test.values
test_view["pred_add"] = add_pipe.predict(X_test)
test_view["pred_champ"] = y_pred

def cell_means(frame, col):
    return frame.groupby(["icu_stay", "procedure"])[col].mean().unstack()

actual_cells = cell_means(test_view, "actual")
add_cells = cell_means(test_view, "pred_add")
champ_cells = cell_means(test_view, "pred_champ")
cell_counts = test_view.groupby(["icu_stay", "procedure"]).size().unstack()
print("test actual cell means:")
print(actual_cells.round(2))
print("additive predicted cell means:")
print(add_cells.round(2))
print("champion predicted cell means:")
print(champ_cells.round(2))
print("cell counts:")
print(cell_counts)

both = (X_test["icu_stay"] == 1) & (X_test["procedure"] == 1)
from sklearn.metrics import mean_absolute_error as _mae
add_both = _mae(y_test[both], add_pipe.predict(X_test[both]))
champ_both = _mae(y_test[both], y_pred[both.to_numpy()])
add_overall = _mae(y_test, add_pipe.predict(X_test))
print("ICU+procedure test n", int(both.sum()),
      "| additive MAE", round(add_both, 3),
      "| champion MAE", round(champ_both, 3))
print("additive test MAE overall", round(add_overall, 3),
      "| champion test MAE overall", round(test_mae, 3))

abs_err = np.abs(residuals)
print(f"median |error| {np.median(abs_err):.2f} days | "
      f"P90 |error| {np.quantile(abs_err, 0.90):.2f} days | "
      f"share within 2 days {(abs_err <= 2).mean():.0%}")
# An additive model averages the extra ICU-plus-procedure days across rows.
# Interactions (and a quadratic age term) give those extra days back to the right patients.
```

### 6. Push to GitHub
```bash
git add day24
git commit -m "Day 24: feature interactions and polynomial terms on LOS"
git push
```

### Stretch: MAE by ICU and procedure in Snowflake SQL
```sql
-- after scoring a held-out batch, see whether errors concentrate in one cell
CREATE TABLE IF NOT EXISTS los_predictions (
  discharge_id STRING, actual_los FLOAT, predicted_los FLOAT,
  icu_stay INT, procedure INT, model STRING, scored_on DATE);

SELECT model, icu_stay, procedure,
       ROUND(AVG(ABS(actual_los - predicted_los)), 3) AS mae_days,
       ROUND(AVG(actual_los), 2) AS mean_actual,
       ROUND(AVG(predicted_los), 2) AS mean_pred,
       COUNT(*) AS n
FROM los_predictions
WHERE scored_on = CURRENT_DATE
GROUP BY model, icu_stay, procedure
ORDER BY model, icu_stay, procedure;
```

## Printables
- Cheat sheet: `day24_v1_cheat_sheet.png`
- Infographic: `day24_v2_infographic.png`
- Code worksheet: `day24_v3_worksheet.png`
- Practice quiz: `day24_v4_quiz.png`
- PDF: `day24_lesson.pdf`

## Podcast script
(Audio: `day24_podcast.mp3`)

Hey Todor, welcome to day twenty four. Yesterday we used Ridge and Lasso so that near-duplicate clinical features would stop fighting over the same credit. Today we ask a different question: what if two features together do more than either one does alone?

Think about a patient who goes to the I C U after a procedure. An additive model says, add some days for the procedure, add some days for the I C U, and you are done. Real discharge data often disagrees. The combination carries extra recovery time that is not sitting in either main effect. That extra piece is an interaction. Age can bend too. The jump from seventy to eighty is not the same as the jump from fifty to sixty, and that curve is a polynomial term, a square of age.

We simulate four thousand fresh discharges with a fixed seed of twenty four. Eight features this time: age, acuity, procedure, comorbidities, I C U stay, emergency, home discharge, and a pure random distractor. Hidden in the data are two true interactions, I C U times procedure and acuity times I C U, plus a gentle age curve. When you build the two by two table of mean length of stay by I C U and procedure, the leftover after subtracting the two main effects is about two point eight days. That leftover is the whole point of today.

An additive Ridge model, the same habit we used yesterday, gets the ordinary stays roughly right and still looks decent on overall M A E. It systematically undershoots the I C U plus procedure cell, because it has no place to put those extra days except by smearing them across other rows. Sklearn's Polynomial Features with degree two builds the products for you, and if you leave interaction only off, it also builds the squares. Eight columns become thirty six interaction terms, or forty four if you keep the squares. That explosion is why yesterday's regularization still matters. Expand first, then scale, then let Ridge C V pick alpha on the training set. Age squared and a zero one product live on very different raw scales, so a penalty without a scaler will punish the wrong thing.

Five fold cross validation on train tells a clean story. The mean baseline sits around two point five days of M A E. Additive Ridge improves that to about one point three one. Adding interactions drops it to about one point two four, and the full degree two model is a hair better still, about one point two three. Those last two are within one fold standard deviation of each other, so our selection rule prefers the simpler expansion: interactions plus Ridge. We write that rule before opening the test set. On the held out thousand patients, the champion lands around one point two five days of M A E, R M S E about one point five six, R squared about point seven six, and roughly four out of five patients within two days.

Here is the number I want you to remember. Overall test M A E only moves a little, from about one point three four with the additive model to one point two five with the champion. On the seventy one test patients who had both an I C U stay and a procedure, M A E drops from about one point eight three to about one point three five. Average metrics can hide a clinically important subgroup. The champion also recovers the two product terms we planted, procedure times I C U and acuity times I C U. Some extra products like age times comorbidities show up too, because age and comorbidities already move together, and those products can stand in for a curve that an interactions only model is not allowed to fit directly.

Your homework is to build the notebook, print the leftover from the two by two table, compare the five models, pick a champion with the simplicity rule, and draw the cell mean chart of actual versus additive versus champion. Push day twenty four with a short README. Tomorrow we read learning curves and validation curves, so you can tell whether extra polynomial terms are fitting signal or just memorizing the training set. See you then, Todor.
