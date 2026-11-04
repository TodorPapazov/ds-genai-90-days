# Day 21: The week 3 model scorecard: baseline, logistic, tree, forest and XGBoost in one fair race, then test once and save with a model card

## Concept
Week 3 wrap-up. Instead of judging each model in its own notebook, we race all of them on the same 4,000 simulated discharges, the same 6 features, the same split and the same 5 cross-validation folds. The line-up: a baseline that predicts the 9% rate for everyone, logistic regression, a depth-4 tree, a random forest and XGBoost. cross_validate scores each on ROC AUC (ranking), PR-AUC (the rare positives) and the Brier score (how accurate the probabilities are). Result: logistic 0.656, forest 0.623, XGBoost 0.613, tree 0.581, baseline 0.500. Fancier lost because the simulated risk is a smooth add-up of a few factors, exactly what logistic regression assumes. With fold sd about 0.04, the 0.03 gap is a lean, not a landslide, so a selection rule written before testing (prefer the simpler model within 1 sd) picks logistic. Then the test set is opened once: AUC 0.723, Brier 0.077, good calibration, and flagging the top 20% catches 41% of readmissions. Test beating CV is luck of one split, so quote CV. Finally, save the model with joblib and write a model card (data, metrics, intended use, limits). Tomorrow: week 4 opens with regression.

## Key numbers
- **Data:** days 14-20 simulated discharges (4,000 patients, seed 14), 6 features, 9% readmitted, 75/25 split
- **5-fold CV AUC:** logistic 0.656, random forest 0.623, XGBoost 0.613, tree 0.581, baseline 0.500
- **Noise ruler:** fold-to-fold AUC sd about 0.04
- **Champion:** logistic regression (highest CV AUC and simplest)
- **Test set, once:** AUC 0.723, PR-AUC 0.231, Brier 0.077
- **Top 20% flagged:** catches 41% of readmissions at 18% precision vs 9% base rate

## Hands-on
1. In your repo, create day21/, run conda activate ds90, and open day21/model_scorecard.ipynb in VS Code.
2. Run snippet 1 to rebuild the same simulated data and split as days 14-20.
3. Run snippet 2 to define the 5-model line-up, baseline included.
4. Run snippet 3: cross_validate every model on the same folds with AUC, PR-AUC and Brier, and print the scorecard.
5. Plot the scorecard with +/- 1 sd error bars and box-plot the per-fold AUCs.
6. In a markdown cell, write your selection rule before touching the test set.
7. Run snippet 4: pick the champion, score the test set once, draw the calibration curve, and check the top 20% flag.
8. Run snippet 5: save the model with joblib, write model_card.json, reload and confirm identical predictions.
9. Write day21/README.md: scorecard table, three charts, the champion and why, and its limits.
10. Commit and push (snippet 6).
11. Stretch: log the scorecard to a Snowflake table so every future model run lands in one place (snippet 7).

## Code
### 1. Same simulated data and split as days 14-20
```python
import numpy as np
import pandas as pd

# Same simulated data and split as days 14-20, so all models face the same patients
# Simulate 4,000 hospital discharges with a 30-day readmission flag (fixed seed)
rng = np.random.default_rng(14)
n = 4000
age = rng.normal(66, 13, n).clip(18, 98).round()
prior_admits = rng.poisson(0.9, n)                         # admissions in the past 12 months
los_days = rng.gamma(2.0, 2.4, n).round(1).clip(1)         # length of stay
n_meds = rng.poisson(9, n)                                 # discharge medications
diabetes = (rng.random(n) < 0.32).astype(int)
followup_visit = (rng.random(n) < 0.55).astype(int)        # clinic visit within 7 days
# True log-odds of readmission: prior admits and long stays matter most
logit = (-2.6 + 0.45 * prior_admits + 0.06 * (los_days - 5) + 0.012 * (age - 66)
         + 0.03 * (n_meds - 9) + 0.25 * diabetes - 0.40 * followup_visit)
readmit30 = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"age": age, "prior_admits": prior_admits, "los_days": los_days, "n_meds": n_meds,
                   "diabetes": diabetes, "followup_visit": followup_visit, "readmit30": readmit30})
df.to_csv("readmissions_30d.csv", index=False)

df = pd.read_csv("readmissions_30d.csv")
from sklearn.model_selection import train_test_split
features = ["age", "prior_admits", "los_days", "n_meds", "diabetes", "followup_visit"]
from sklearn.model_selection import train_test_split
features = ["age", "prior_admits", "los_days", "n_meds", "diabetes", "followup_visit"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df.readmit30, test_size=0.25,
                                                    stratify=df.readmit30, random_state=14)
print("train", X_train.shape, "test", X_test.shape, "readmit rate", round(y_train.mean(), 3))
```

### 2. The week 3 model line-up
```python
# The week 3 line-up: every model wrapped the same way, scored on the same folds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
models = {
    "Baseline (always the rate)": DummyClassifier(strategy="prior"),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=21),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=21),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=21),
}
```

### 3. One scorecard: same folds, several metrics
```python
# One scorecard: 5-fold CV on the TRAIN set only, several metrics per model
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
cv = StratifiedKFold(5, shuffle=True, random_state=21)
scoring = {"auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
rows, fold_auc = [], {}
for name, model in models.items():
    t0 = time.time()
    r = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    fold_auc[name] = r["test_auc"]
    rows.append({"model": name, "cv_auc": r["test_auc"].mean(), "auc_sd": r["test_auc"].std(),
                 "pr_auc": r["test_pr_auc"].mean(), "brier": -r["test_brier"].mean(),
                 "seconds": time.time() - t0})
scorecard = pd.DataFrame(rows).set_index("model").round(3)
print(scorecard.sort_values("cv_auc", ascending=False))
# Read it like a reviewer: is the gap between the top models bigger than the fold-to-fold sd?
```

### 4. Pick by a rule, test once, check calibration
```python
# Pick a champion with a rule you wrote down BEFORE looking at the test set:
# highest CV AUC, but prefer the simpler model if it is within one fold sd.
best = scorecard.cv_auc.idxmax()
simple_ok = scorecard.loc["Logistic regression", "cv_auc"] >= scorecard.loc[best, "cv_auc"] - scorecard.loc[best, "auc_sd"]
champion = "Logistic regression" if simple_ok else best
print("top CV AUC:", best, "| champion by the rule:", champion)
# Open the test set ONCE, for the champion only
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
final = models[champion].fit(X_train, y_train)
p_test = final.predict_proba(X_test)[:, 1]
print(f"test AUC {roc_auc_score(y_test, p_test):.3f} | PR-AUC {average_precision_score(y_test, p_test):.3f}"
      f" | Brier {brier_score_loss(y_test, p_test):.3f}")
# Calibration: when the model says 30%, do about 30% come back?
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
print(pd.DataFrame({"predicted": mean_pred.round(3), "observed": frac_pos.round(3)}))
# Capacity view from day 18: flag the top 20% riskiest discharges for a follow-up call
cut = np.quantile(p_test, 0.80)
flag = p_test >= cut
print(f"top 20% flagged: catches {y_test[flag].sum() / y_test.sum():.0%} of readmissions,"
      f" precision {y_test[flag].mean():.0%} vs base rate {y_test.mean():.0%}")
```

### 5. Save the champion and a model card
```python
# Save the champion and a small model card so the work is reusable and reviewable
import joblib, json, sklearn
joblib.dump(final, "readmit_champion.joblib")
card = {"model": champion, "trained_on": "4,000 simulated discharges (seed 14), 75% train",
        "target": "readmit30 (30-day readmission)", "features": features,
        "cv_auc": float(scorecard.loc[champion, "cv_auc"]),
        "test_auc": round(float(roc_auc_score(y_test, p_test)), 3),
        "intended_use": "rank discharges for follow-up calls; not for denying care",
        "limits": "simulated data; no fairness audit yet; recheck calibration on real data",
        "sklearn_version": sklearn.__version__}
json.dump(card, open("model_card.json", "w"), indent=2)
reloaded = joblib.load("readmit_champion.joblib")
assert np.allclose(reloaded.predict_proba(X_test)[:, 1], p_test)   # same predictions after reload
print(json.dumps(card, indent=2))
```

### 6. Push to GitHub
```bash
git add day21
git commit -m "Day 21: week 3 model scorecard, selection rule, one test, calibration, joblib model and model card"
git push
```

### Stretch: log the scorecard in Snowflake
```sql
-- one table for every model run, so week 4+ results stack up in one scorecard
CREATE TABLE IF NOT EXISTS model_scorecard (
  run_date DATE, project STRING, model STRING,
  cv_auc FLOAT, auc_sd FLOAT, pr_auc FLOAT, brier FLOAT, is_champion BOOLEAN);

INSERT INTO model_scorecard VALUES
  (CURRENT_DATE, 'readmit30', 'Logistic regression', 0.656, 0.043, 0.172, 0.080, TRUE),
  (CURRENT_DATE, 'readmit30', 'Random forest',       0.623, 0.045, 0.148, 0.081, FALSE),
  (CURRENT_DATE, 'readmit30', 'XGBoost',             0.613, 0.043, 0.135, 0.082, FALSE);

SELECT model, cv_auc, auc_sd,
       cv_auc - MAX(cv_auc) OVER (PARTITION BY project, run_date) AS gap_to_best
FROM model_scorecard
WHERE project = 'readmit30'
ORDER BY cv_auc DESC;
```

## Printables
- Cheat sheet: `day21_v1_cheat_sheet.png`
- Infographic: `day21_v2_infographic.png`
- Code worksheet: `day21_v3_worksheet.png`
- Practice quiz: `day21_v4_quiz.png`
- PDF: `day21_lesson.pdf`

## Podcast script
(Audio: `day21_podcast.mp3`)

Hey Todor, welcome to day twenty one. That is the end of week three, and three full weeks of showing up. Nice work. Today is a wrap up day, and it is one of the most useful habits you will ever build as a data scientist. We are going to put every model from this week into one fair race and write up the result like a professional would.

Here is the setup. Same four thousand simulated hospital discharges we have used since day fourteen. Same six features: age, prior admissions, length of stay, number of medications, diabetes, and whether the patient had a follow up visit within a week. Same train and test split. The only thing that changes is the model. We line up five of them: a baseline that just predicts the nine percent readmission rate for everyone, logistic regression, a shallow decision tree, a random forest, and XGBoost.

Why so strict about using the same everything? Because if each model gets a different split, you cannot tell whether one model is better or just got easier patients. So every model is scored on the exact same five cross validation folds, using only the training data. And we score three things at once. AUC tells you how well the model ranks patients, so a readmitted patient usually scores higher than one who was fine. P R AUC focuses on the rare positive cases. And the Brier score checks whether the probabilities themselves are accurate, lower is better.

Now the results, and they are a little surprising. Logistic regression comes out on top, with a cross validated AUC of about point six six. The random forest is at point six two, XGBoost at point six one, the single tree at point five eight, and the baseline sits at exactly point five, which is what a coin flip ranking looks like. So the fancy models lost. Why? Because the way we built this data, readmission risk is a smooth add up of a few factors. That is exactly the shape logistic regression assumes. Boosting and forests are great when there are tricky interactions and thresholds hiding in the data, but here they mostly add noise. In real hospital data, with messy features and interactions, the ranking could easily flip. That is why you run the race instead of assuming.

The second big idea is reading the gap honestly. The AUC wobbles by about point zero four from fold to fold. Logistic beats the forest by about point zero three. So that is a lean, not a landslide. A good rule, written down before you peek at the test set, is: take the highest score, but if a simpler model is within one fold standard deviation, take the simpler one. Simpler models are easier to explain to clinicians, faster, and easier to monitor.

Only after you choose do you open the test set, once, for the champion. Our logistic model gets a test AUC of about point seven two. That is higher than cross validation, and here is the trap: it does not mean the model got better. One test split of a thousand patients can just be lucky. The five fold average is your honest estimate, so that is the number you quote in an interview or a README. We also check calibration, which means when the model says twenty percent risk, do about twenty percent actually come back. Ours tracks the diagonal pretty well. And in practical terms, if a care team can call the riskiest twenty percent of discharges, this model catches about forty one percent of readmissions, at roughly double the base rate precision.

Last step, and this is what makes it portfolio ready: save the model with joblib, reload it to prove the predictions match, and write a short model card. A model card says what the model is, what data trained it, its metrics, what it is meant for, like ranking patients for follow up calls, what it is not meant for, like denying care, and its limits. Healthcare hiring managers love seeing that, because it shows you think about how a model gets used, not just how it scores.

So your homework is to build the scorecard notebook, write your selection rule before you test, and push day twenty one with a clean README. Tomorrow we start week four, and we switch from predicting a yes or no to predicting a number, like how many days a patient will stay in the hospital. That means regression models and new metrics. See you then, Todor.
